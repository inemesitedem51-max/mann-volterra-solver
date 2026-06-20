#ifndef MANN_SOLVER_HPP
#define MANN_SOLVER_HPP

#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/QR>   // for ColPivHouseholderQR

// Mann iteration with Anderson acceleration for solving y = T(y)
template <typename OperatorType>
class AndersonMannSolver {
private:
    double lambda_;    // Mann relaxation parameter (0 < lambda <= 1-k)
    int window_;       // Anderson window size (m)
    double eps_abs_;
    double eps_rel_;

public:
    AndersonMannSolver(double lambda, int window, double eps_abs, double eps_rel)
        : lambda_(lambda), window_(window), eps_abs_(eps_abs), eps_rel_(eps_rel) {}

    // Solve y = T(y) using Mann iteration with optional Anderson acceleration.
    // Input: y0 (initial guess), T (callable operator), max_iter (optional).
    // Output: converged solution.
    template <typename Func>
    Eigen::VectorXd solve(const Eigen::VectorXd& y0, Func&& T, int max_iter = 1000) {
        Eigen::VectorXd y = y0;
        Eigen::VectorXd y_prev = y0;
        std::vector<Eigen::VectorXd> history_y;
        std::vector<Eigen::VectorXd> history_T;

        for (int k = 0; k < max_iter; ++k) {
            // Compute T(y)
            Eigen::VectorXd Ty = T(y);

            // Residual
            Eigen::VectorXd residual = Ty - y;
            double res_norm = residual.norm();

            // Stopping criterion
            double tol = std::max(eps_abs_, eps_rel_ * y.norm());
            if (res_norm < tol) {
                return y;
            }

            // If Anderson window size > 0 and we have enough history, apply AA
            if (window_ > 0 && static_cast<int>(history_y.size()) >= window_) {
                // Build least-squares problem for coefficients alpha
                int m = window_;
                Eigen::MatrixXd F(m, m);
                Eigen::VectorXd ones = Eigen::VectorXd::Ones(m);

                // history_y[0] = y_{k-m}, ..., history_y[m-1] = y_{k-1}
                // We want to solve: min_alpha || sum_{i=1}^m alpha_i * residual_{k-i} ||^2, sum(alpha)=1
                // Equivalent: R * alpha = 1, where R_{ij} = <res_i, res_j>
                for (int i = 0; i < m; ++i) {
                    const auto& res_i = history_T[history_T.size() - m + i] - history_y[history_y.size() - m + i];
                    for (int j = 0; j < m; ++j) {
                        const auto& res_j = history_T[history_T.size() - m + j] - history_y[history_y.size() - m + j];
                        F(i, j) = res_i.dot(res_j);
                    }
                }

                // Solve F * alpha = 1 (constrained least squares, but we just solve normal equations)
                // Use Eigen's QR solver for stability
                Eigen::VectorXd alpha = F.colPivHouseholderQr().solve(ones);
                // Normalize alpha to sum to 1
                double sum_alpha = alpha.sum();
                if (std::abs(sum_alpha) > 1e-12) {
                    alpha /= sum_alpha;
                } else {
                    alpha = ones / m; // fallback uniform weights
                }

                // Compute new iterate: y_new = sum_{i=1}^m alpha_i * T(y_{k-m+i-1})
                Eigen::VectorXd y_new = Eigen::VectorXd::Zero(y.size());
                for (int i = 0; i < m; ++i) {
                    y_new += alpha(i) * history_T[history_T.size() - m + i];
                }
                y = y_new;

                // Update history (push current y and T(y) for next iteration)
                // But we already have T(y) computed? We just computed y_new from past T's.
                // We need to compute T(y_new) for the next residual.
                // So we should compute Ty_new = T(y_new) before storing.
                // Actually, standard AA stores the history of y and T(y).
                // We will store after computing T(y_new) in the next loop iteration.
                // But to maintain proper history, we push the current pair before updating.
                // For simplicity, we just proceed to the next iteration and store there.
            }

            // Standard Mann step (if AA not used or fallback)
            if (window_ == 0 || static_cast<int>(history_y.size()) < window_) {
                y = (1.0 - lambda_) * y + lambda_ * Ty;
            }

            // Store history for Anderson acceleration
            history_y.push_back(y);
            history_T.push_back(T(y)); // store T(y) for the new y

            // Limit history size
            if (static_cast<int>(history_y.size()) > 2 * window_ + 10) {
                history_y.erase(history_y.begin());
                history_T.erase(history_T.begin());
            }

            // Check for divergence
            if (k > 0 && (y - y_prev).norm() > 1e10) {
                break;
            }
            y_prev = y;
        }
        return y; // return best effort
    }
};

#endif // MANN_SOLVER_HPP
