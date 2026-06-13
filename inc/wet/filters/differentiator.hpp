#pragma once

/**
 * @file differentiator.hpp
 * @brief Levant's robust exact differentiator — clean derivative of a noisy or
 *        quantized signal, model-free, with far less phase lag than an LPF.
 *
 * A first-order sliding-mode (super-twisting) differentiator. Fed a measured
 * signal f(t), it produces a denoised estimate of f *and* its derivative ḟ. It is
 * the natural companion to the super-twisting controller (`controllers/smc.hpp`):
 * same robustness philosophy, and it solves the problem that makes super-twisting
 * chatter in practice — having to differentiate a noisy/quantized measurement to
 * form the sliding variable.
 *
 * Two headline uses:
 *  - **Servo encoder velocity.** An incremental encoder gives *quantized*
 *    position; finite-differencing it at low speed (a count or two per sample) is
 *    pure quantization noise. The differentiator recovers a smooth velocity.
 *  - **Any rate signal feeding an SMC/STA surface** where plain differentiation
 *    would inject noise.
 *
 * Continuous law (Levant, 1998), z0 → f, z1 → ḟ:
 *
 *     ż0 = −λ0·L^½·|z0 − f|^½·sign(z0 − f) + z1
 *     ż1 = −λ1·L·sign(z0 − f)
 *
 * L bounds the signal's |second derivative| (so for position in, it is a bound on
 * the acceleration). The standard gains λ0 = 1.5, λ1 = 1.1 are the same pair as
 * the super-twisting algorithm and are the defaults. Convergence is finite-time
 * and *exact* in the absence of noise; with noise the derivative error scales like
 * the noise magnitude, not its derivative — the whole point versus naive
 * differentiation.
 *
 * @note Explicit-Euler discretization. Keep L·dt bounded (don't over-estimate L on
 *       a slow loop); if L is wildly too large for the step the discrete iteration
 *       can ring. All constexpr, allocation-free, float/double.
 *
 * @see "Robust exact differentiation via sliding mode technique" (Levant, 1998),
 *      Automatica, https://doi.org/10.1016/S0005-1098(98)00209-2
 * @see controllers/smc.hpp for the super-twisting controller it pairs with
 */

#include <type_traits>

#include "wet/math/math.hpp" // wet::sqrt, wet::abs, wet::sgn

namespace wet {

/**
 * @ingroup filters
 * @brief First-order robust exact differentiator (super-twisting differentiator).
 *
 * Feed it samples with @ref update; it returns the derivative estimate and also
 * exposes a denoised copy of the signal via @ref value.
 *
 * @tparam T Scalar type (float or double)
 */
template<typename T = float>
class RobustExactDifferentiator {
public:
    constexpr RobustExactDifferentiator() = default;

    /**
     * @brief Configure for second-derivative bound @p L at sample time @p dt.
     * @param L       Bound on |f̈| (for a position signal, the acceleration bound). Must be > 0.
     * @param dt      Sample time [s]. Must be > 0.
     * @param lambda0 First gain (default 1.5).
     * @param lambda1 Second gain (default 1.1).
     */
    constexpr RobustExactDifferentiator(T L, T dt, T lambda0 = T{1.5}, T lambda1 = T{1.1})
        : L_(L), sqrt_L_(wet::sqrt(L)), dt_(dt), lambda0_(lambda0), lambda1_(lambda1), valid_(L > T{0} && dt > T{0}) {}

    /**
     * @brief Feed one sample; advance the differentiator one step.
     * @param f Latest measurement of the signal.
     * @return Updated derivative estimate ḟ.
     */
    constexpr T update(T f) {
        if (!valid_) {
            return T{0};
        }
        const T e = z0_ - f;
        const T sign_e = static_cast<T>(wet::sgn(e));
        const T z0_dot = (-lambda0_ * sqrt_L_ * wet::sqrt(wet::abs(e)) * sign_e) + z1_;
        const T z1_dot = -lambda1_ * L_ * sign_e;
        z0_ += dt_ * z0_dot;
        z1_ += dt_ * z1_dot;
        return z1_;
    }

    /// Denoised estimate of the signal itself (z0 → f).
    [[nodiscard]] constexpr T value() const { return z0_; }

    /// Latest derivative estimate (z1 → ḟ).
    [[nodiscard]] constexpr T derivative() const { return z1_; }

    [[nodiscard]] constexpr bool valid() const { return valid_; }

    /// Re-seed the internal states (e.g. to the first measurement and a known rate).
    constexpr void reset(T value0 = T{0}, T derivative0 = T{0}) {
        z0_ = value0;
        z1_ = derivative0;
    }

private:
    T    L_{T{1}};
    T    sqrt_L_{T{1}};
    T    dt_{T{1}};
    T    lambda0_{T{1.5}};
    T    lambda1_{T{1.1}};
    T    z0_{T{0}};
    T    z1_{T{0}};
    bool valid_{false};
};

/// Searchable alias: the first-order robust exact differentiator is "the Levant
/// differentiator" in most of the literature.
template<typename T = float>
using LevantDifferentiator = RobustExactDifferentiator<T>;

} // namespace wet
