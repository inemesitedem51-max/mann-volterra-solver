#ifndef QUADRATURE_HPP
#define QUADRATURE_HPP

#include <vector>
#include <cmath>

// Simple trapezoidal rule for the history integral
// Assumes uniform time step dt and stores past values of the nonlinearity.
// For convolution kernels K(t,s,u) = g(t-s) * Phi(u(s)),
// we store Phi(u_j) at each past time step.
template <typename MemoryKernel>
class ConvolutionQuadrature {
private:
    MemoryKernel kernel_;
    double dt_;
    std::vector<double> history_; // stores Phi(u_j) for j=0..n
public:
    explicit ConvolutionQuadrature(const MemoryKernel& kernel, double dt)
        : kernel_(kernel), dt_(dt) {}

    // Reset history (called when dt changes or simulation restarts)
    void reset() {
        history_.clear();
    }

    // Add a new value at the current time step
    void push(double phi_val) {
        history_.push_back(phi_val);
    }

    // Compute integral from 0 to t_{n+1} of g(t_{n+1} - tau) * Phi(u(tau)) dtau
    // using the trapezoidal rule on the stored history.
    // current_time = t_{n+1}
    double integrate(double current_time) const {
        if (history_.empty()) return 0.0;
        double integral = 0.0;
        const int n = static_cast<int>(history_.size()) - 1;
        for (int j = 0; j < n; ++j) {
            double tau_j = j * dt_;
            double tau_jp1 = (j+1) * dt_;
            double w_j = kernel_(current_time - tau_j);
            double w_jp1 = kernel_(current_time - tau_jp1);
            integral += 0.5 * dt_ * (w_j * history_[j] + w_jp1 * history_[j+1]);
        }
        return integral;
    }

    // Get the last stored value (for convenience)
    double last() const {
        return history_.empty() ? 0.0 : history_.back();
    }

    // Size of history
    std::size_t size() const {
        return history_.size();
    }
};

#endif // QUADRATURE_HPP
