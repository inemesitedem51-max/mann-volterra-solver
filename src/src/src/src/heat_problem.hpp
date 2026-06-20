#ifndef HEAT_PROBLEM_HPP
#define HEAT_PROBLEM_HPP

#include <vector>
#include <cmath>
#include <Eigen/Dense>

// 1D heat conduction with memory after spatial discretisation.
// The discrete operator N(u) approximates d/dx (k(u) du/dx).
class HeatProblem {
private:
    int M_;              // number of spatial points (excluding boundaries)
    double L_;           // domain length
    double dx_;          // grid spacing
    double alpha_;       // coefficient in k(u) = 1 - alpha * u^3
    double rho_cp_;      // rho * cp

public:
    HeatProblem(int M, double L, double alpha, double rho_cp = 1.0)
        : M_(M), L_(L), dx_(L / (M+1)), alpha_(alpha), rho_cp_(rho_cp) {}

    // Evaluate the nonlinear operator N(u) at the interior points.
    // Input: u is (M+2) vector including boundary values at indices 0 and M+1.
    // Output: N(u) at interior points (size M).
    Eigen::VectorXd compute_N(const Eigen::VectorXd& u) const {
        Eigen::VectorXd N(M_);
        for (int i = 0; i < M_; ++i) {
            int idx = i + 1; // interior index
            double u_left = u(idx - 1);
            double u_cent = u(idx);
            double u_right = u(idx + 1);

            // Conductivity at midpoints (arithmetic average)
            double k_left  = conductivity(0.5 * (u_left + u_cent));
            double k_right = conductivity(0.5 * (u_cent + u_right));

            // Second-order finite difference: (k_right*(u_right-u_cent) - k_left*(u_cent-u_left)) / dx^2
            double flux_left  = k_left  * (u_cent - u_left);
            double flux_right = k_right * (u_right - u_cent);
            N(i) = (flux_right - flux_left) / (dx_ * dx_);
        }
        return N / rho_cp_;
    }

    // Temperature-dependent conductivity: k(u) = 1 - alpha * u^3
    double conductivity(double u) const {
        double val = 1.0 - alpha_ * u * u * u;
        return (val > 1e-12) ? val : 1e-12; // ensure positivity
    }

    // Number of interior points
    int size() const { return M_; }

    // Domain length
    double length() const { return L_; }
};

#endif // HEAT_PROBLEM_HPP
