#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <Eigen/Dense>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "kernel.hpp"
#include "quadrature.hpp"
#include "heat_problem.hpp"
#include "mann_solver.hpp"
#include "time_stepper.hpp"

int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
    int rank = 0, size = 1;
#endif

    // ----------------------------
    // 1. Problem Parameters
    // ----------------------------
    double T = 1.0;               // final time
    double L = 1.0;               // domain length
    int M = 100;                  // number of interior spatial points
    double alpha = 0.5;           // conductivity coefficient
    double rho_cp = 1.0;

    // Time stepping
    double dt_init = 0.01;
    double dt_min = 1e-6;
    double dt_max = 0.1;
    double tol = 5e-5;            // target local error

    // Mann solver parameters
    double lambda = 0.8;
    int window = 5;               // Anderson window size (m)
    double eps_abs = 1e-8;
    double eps_rel = 1e-5;
    int max_iter = 1000;

    // ----------------------------
    // 2. Initialise Problem
    // ----------------------------
    HeatProblem problem(M, L, alpha, rho_cp);
    CubicKernel kernel;
    ExponentialMemory memory_kernel(0.2); // tau0 = 0.2

    // State vector: interior points (size M) + boundaries
    Eigen::VectorXd u_int = Eigen::VectorXd::Zero(M);
    double u_left = 0.0;          // u(0,t) = 0 (for simplicity)
    double u_right = 0.0;         // u(L,t) = 0

    // History storage for convolution quadrature
    ConvolutionQuadrature<ExponentialMemory> quadrature(memory_kernel, dt_init);
    quadrature.reset();

    // ----------------------------
    // 3. Time Loop
    // ----------------------------
    AdaptiveTimeStepper stepper(dt_init, dt_min, dt_max, tol);
    double t = 0.0;
    int step_count = 0;
    std::vector<double> times;
    std::vector<Eigen::VectorXd> solutions;
    times.push_back(t);
    solutions.push_back(u_int);

    auto start_time = std::chrono::high_resolution_clock::now();

    while (t < T - 1e-12) {
        double dt = stepper.get_dt();
        // Ensure we don't overshoot
        if (t + dt > T) dt = T - t;

        bool step_accepted = false;
        int retries = 0;
        while (!step_accepted && retries < 10) {
            // Build the known term F_n from history
            // For our heat equation, the discrete system is:
            // u^{n+1} = u^n + dt * ( N(u^{n+1}) + Integral_memory(N(u^{n+1})) ) + dt * Q
            // We define T_n(y) = u^n + dt * ( N(y) + Integral_memory(N(y)) )
            // Actually, we need to handle the memory integral separately.
            // For simplicity, we approximate memory integral using the explicit history.

            // Compute N(u^n) for the current step (used for memory integral)
            Eigen::VectorXd N_current = problem.compute_N(u_int);

            // Define the operator T_n for the Mann solver:
            auto T_n = [&](const Eigen::VectorXd& y) -> Eigen::VectorXd {
                // Compute N(y)
                Eigen::VectorXd Ny = problem.compute_N(y);

                // Compute memory integral: integral_0^t g(t-tau) N(u(tau)) dtau
                // Using quadrature with current time t+dt
                // For simplicity, we approximate the integral using the explicit history + a trapezoidal correction.
                // A more accurate version would include the implicit term in the quadrature.
                // Here we use the explicit value: I_mem = sum_{j=0}^{n} w_j g(t+dt - t_j) N(u_j)
                // For demonstration, we just use the existing quadrature (which stores N(u_j)).
                // But our quadrature stores Phi(u_j) = N(u_j).
                // We need to push N(y) if we want to include it implicitly.
                // For a fully implicit scheme, the memory integral should involve N(y).
                // We adopt a simple semi-implicit approach: memory uses explicit history only.
                double mem_integral = quadrature.integrate(t + dt);

                // Known term: u^n + dt * Q (no external source here)
                Eigen::VectorXd F = u_int + dt * (Ny + mem_integral * Eigen::VectorXd::Ones(M));

                // For the heat equation, T_n(y) = F (since we already computed Ny)
                // But to match the paper's notation: y = F_n + dt * Phi_n(y)
                // Here Phi_n(y) = N(y) + mem_integral_explicit.
                // So T_n(y) = F_n + dt * N(y) + dt * mem_integral_explicit.
                // Since mem_integral_explicit is known, we treat it as part of F_n.
                // Thus:
                Eigen::VectorXd T_y = u_int + dt * Ny + dt * mem_integral * Eigen::VectorXd::Ones(M);
                return T_y;
            };

            // Initial guess: previous solution
            Eigen::VectorXd y0 = u_int;

            // Solve using Mann + Anderson
            AndersonMannSolver<decltype(T_n)> solver(lambda, window, eps_abs, eps_rel);
            Eigen::VectorXd y_new = solver.solve(y0, T_n, max_iter);

            // Compute predictor for error estimation (explicit Euler)
            Eigen::VectorXd N_old = problem.compute_N(u_int);
            double mem_old = quadrature.integrate(t);
            Eigen::VectorXd predictor = u_int + dt * (N_old + mem_old * Eigen::VectorXd::Ones(M));

            // Estimate error
            double error_est = stepper.estimate_error(y_new, u_int, predictor);

            // Check if step is accepted
            double next_dt;
            if (stepper.accept_step(error_est, next_dt)) {
                // Accept
                u_int = y_new;
                // Update quadrature history: store N(u_new) for future memory integrals
                Eigen::VectorXd N_new = problem.compute_N(u_int);
                quadrature.push(N_new.sum()); // store scalar sum for simplicity; in full code store vector
                // In a real implementation, quadrature stores a vector per time step.
                // For this demo, we store a scalar approximation.
                t += dt;
                step_count++;
                step_accepted = true;
                times.push_back(t);
                solutions.push_back(u_int);
                // Update dt for next iteration
                stepper.set_dt(next_dt);
                // Update quadrature dt
                // In full code, quadrature dt should match stepper dt.
            } else {
                // Reject and retry
                dt = next_dt;
                retries++;
            }
        }
        if (!step_accepted) {
            std::cerr << "Step rejection failed after 10 retries. Exiting." << std::endl;
            break;
        }
        if (rank == 0 && step_count % 50 == 0) {
            std::cout << "t = " << t << ", dt = " << stepper.get_dt()
                      << ", steps = " << step_count << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    if (rank == 0) {
        std::cout << "\nSimulation completed." << std::endl;
        std::cout << "Time steps: " << step_count << std::endl;
        std::cout << "Total CPU time: " << elapsed.count() << " s" << std::endl;

        // Output solution to CSV
        std::ofstream out("solution.csv");
        out << "time, u_mean\n";
        for (std::size_t i = 0; i < times.size(); ++i) {
            double mean = solutions[i].mean();
            out << times[i] << ", " << mean << "\n";
        }
        out.close();
        std::cout << "Solution saved to solution.csv" << std::endl;
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
