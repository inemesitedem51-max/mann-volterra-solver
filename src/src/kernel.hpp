cpp
#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <cmath>

// Abstract base class for the Volterra kernel K(t,s,u)
class Kernel {
public:
    virtual ~Kernel() = default;
    // Evaluate K(t, s, u)
    virtual double evaluate(double t, double s, double u) const = 0;
    // Compute the one-sided Lipschitz constant L (for adaptive parameter selection)
    virtual double lipschitz_constant() const = 0;
};

// Example: Cubic kernel K(t,s,u) = -u^3 (dissipative case, L=0)
// This matches the heat conduction model with k(u) = 1 - 0.5 u^3
class CubicKernel : public Kernel {
public:
    double evaluate(double /*t*/, double /*s*/, double u) const override {
        return -u * u * u;
    }
    double lipschitz_constant() const override {
        return 0.0; // dissipative case
    }
};

// Memory kernel g(tau) = exp(-tau/tau0) for convolution
class ExponentialMemory {
private:
    double tau0_;
public:
    explicit ExponentialMemory(double tau0) : tau0_(tau0) {}
    double operator()(double tau) const {
        return std::exp(-tau / tau0_);
    }
};

#endif // KERNEL_HPP
