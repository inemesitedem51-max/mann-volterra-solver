#ifndef TIME_STEPPER_HPP
#define TIME_STEPPER_HPP

#include <vector>
#include <cmath>
#include <Eigen/Dense>

// Adaptive time stepper with local error control
class AdaptiveTimeStepper {
private:
    double dt_;          // current time step
    double dt_min_;
    double dt_max_;
    double safety_;      // safety factor for step change (e.g., 0.8)
    double tol_;         // target local error tolerance

public:
    AdaptiveTimeStepper(double dt_init, double dt_min, double dt_max,
                        double tol, double safety = 0.8)
        : dt_(dt_init), dt_min_(dt_min), dt_max_(dt_max),
          safety_(safety), tol_(tol) {}

    // Estimate local error by comparing implicit Euler solution with a predictor (e.g., explicit Euler).
    // u_new: solution computed by implicit Euler.
    // u_old: solution at previous time step.
    // predictor: simple explicit Euler prediction from u_old using dt.
    // Returns estimated error.
    double estimate_error(const Eigen::VectorXd& u_new,
                          const Eigen::VectorXd& u_old,
                          const Eigen::VectorXd& predictor) const {
        // Simple estimate: norm of difference between predictor and implicit solution
        return (u_new - predictor).norm() / (u_new.norm() + 1e-12);
    }

    // Check if the step is accepted. If not, reduce dt and return false.
    // If accepted, optionally increase dt for next step.
    bool accept_step(double error_est, double& next_dt) {
        if (error_est <= 1.2 * tol_) {
            // Accept step
            if (error_est <= 0.5 * tol_) {
                next_dt = std::min(dt_ * 1.2, dt_max_); // increase
            } else {
                next_dt = dt_;
            }
            dt_ = next_dt;
            return true;
        } else {
            // Reject step: reduce dt and retry
            double factor = safety_ * std::sqrt(tol_ / (error_est + 1e-12));
            next_dt = std::max(dt_min_, std::min(dt_max_, factor * dt_));
            dt_ = next_dt;
            return false;
        }
    }

    double get_dt() const { return dt_; }
    void set_dt(double dt) { dt_ = dt; }
    double get_dt_min() const { return dt_min_; }
    double get_dt_max() const { return dt_max_; }
};

#endif // TIME_STEPPER_HPP
