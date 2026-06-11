#pragma once

/**
 * @file harmonic_suppression.hpp
 * @brief Active harmonic suppression — a bank of proportional-resonant (PR)
 *        resonators that cancel a chosen set of harmonics in a control loop.
 *
 * The companion to the spectral *detector* (filters/spectral.hpp `Goertzel` /
 * `HarmonicAnalyzer`): once you know which harmonics to kill, drop a
 * `HarmonicSuppressor` into the current/voltage loop. Each PR resonator places
 * (near-)infinite loop gain at one harmonic, so the loop drives that harmonic's
 * error to zero. Targets: grid-tied inverter current control (reject 5th/7th/…
 * line harmonics), active filters, motor torque-ripple cancellation.
 *
 * Repetitive control (controllers/repetitive.hpp) rejects *every* harmonic of a
 * period with one delay loop; this PR bank instead targets a *specific, sparse*
 * set (e.g. {1, 5, 7, 11, 13}) with independent per-harmonic gains — cheaper and
 * more selective when only a few harmonics matter.
 *
 * @see controllers/pr.hpp for the single PR resonator and design::pr_harmonics
 * @see filters/spectral.hpp for the detection side
 */

#include <cstddef>

#include "wet/backend.hpp" // wet::array, wet::numbers
#include "wet/controllers/pr.hpp"

namespace wet {

namespace design {

/**
 * @brief Design result for a multi-resonant harmonic suppressor.
 * @tparam N Number of harmonics
 * @tparam T Scalar type
 */
template<size_t N, typename T = double>
struct HarmonicSuppressorResult {
    using value_type = std::remove_const_t<T>;

    wet::array<PRResult<value_type>, N> gains{}; //!< one PR resonator per harmonic
    bool                                success{false};

    template<typename U>
    [[nodiscard]] constexpr HarmonicSuppressorResult<N, std::remove_const_t<U>> as() const {
        HarmonicSuppressorResult<N, std::remove_const_t<U>> out{};
        for (size_t i = 0; i < N; ++i) {
            out.gains[i] = gains[i].template as<std::remove_const_t<U>>();
        }
        out.success = success;
        return out;
    }
};

/**
 * @brief Synthesize a multi-resonant harmonic suppressor.
 *
 * Builds one PR resonator per harmonic (via @ref pr_harmonics) and validates the
 * spec — frequencies positive, sampling time positive, and the highest harmonic
 * strictly below Nyquist (a resonator at or above fs/2 cannot be realized). The
 * proportional gain Kp is carried on the fundamental's resonator only.
 *
 * @tparam N Number of harmonics
 * @param Kp        Proportional gain (shared; placed on the fundamental)
 * @param Ki_fund   Resonant gain for the fundamental (scaled down per harmonic)
 * @param w_fund    Fundamental frequency [rad/s]
 * @param wc        Resonant-term bandwidth [rad/s] (0 = ideal PR)
 * @param Ts        Sample time [s]
 * @param harmonics Harmonic orders to suppress (e.g. {1, 5, 7, 11})
 */
template<size_t N, typename T = double>
[[nodiscard]] constexpr HarmonicSuppressorResult<N, T> synthesize_harmonic_suppressor(
    T                            Kp,
    T                            Ki_fund,
    T                            w_fund,
    T                            wc,
    T                            Ts,
    const wet::array<size_t, N>& harmonics
) {
    HarmonicSuppressorResult<N, T> result{};
    if (Ts <= T{0} || w_fund <= T{0} || wc < T{0}) {
        return result;
    }
    const T nyquist = wet::numbers::pi_v<T> / Ts; // [rad/s]
    for (size_t i = 0; i < N; ++i) {
        if (harmonics[i] == 0) {
            return result; // 0th harmonic (DC) is not a resonant target
        }
        if (w_fund * static_cast<T>(harmonics[i]) >= nyquist) {
            return result; // resonator at/above Nyquist is unrealizable
        }
    }
    result.gains = pr_harmonics(Kp, Ki_fund, w_fund, wc, Ts, harmonics);
    result.success = true;
    return result;
}

} // namespace design

/**
 * @ingroup controllers
 * @brief Multi-resonant harmonic suppressor — a parallel bank of PR resonators.
 *
 * Sums the outputs of N PR resonators tuned to the target harmonics; drop it into
 * a current/voltage loop on the error signal. Satisfies the (r, y) controller
 * protocol so it composes like any other controller.
 *
 * @tparam N Number of harmonics
 * @tparam T Scalar type (float or double)
 */
template<size_t N, typename T = float>
class HarmonicSuppressor {
public:
    using value_type = std::remove_const_t<T>;
    static_assert(N >= 1, "HarmonicSuppressor needs at least one harmonic");

    constexpr HarmonicSuppressor() = default;

    constexpr explicit HarmonicSuppressor(const wet::array<design::PRResult<value_type>, N>& gains) {
        for (size_t i = 0; i < N; ++i) {
            resonators_[i] = PRController<value_type>(gains[i]);
        }
    }

    constexpr explicit HarmonicSuppressor(const design::HarmonicSuppressorResult<N, value_type>& design)
        : valid_(design.success) {
        for (size_t i = 0; i < N; ++i) {
            resonators_[i] = PRController<value_type>(design.gains[i]);
        }
    }

    /// Suppression command from the loop error (sum of the resonator outputs).
    [[nodiscard]] constexpr value_type control(value_type error) {
        value_type u = value_type{0};
        for (size_t i = 0; i < N; ++i) {
            u += resonators_[i].control(error);
        }
        return u;
    }

    /// Reference-tracking overload (computes the error internally).
    [[nodiscard]] constexpr value_type control(value_type r, value_type y) { return control(r - y); }

    constexpr void reset() {
        for (size_t i = 0; i < N; ++i) {
            resonators_[i].reset();
        }
    }

    /// Re-tune the bank to a new fundamental (grid-frequency adaptation); the
    /// harmonic ratios are preserved (resonator i tracks its original order).
    constexpr void set_fundamental(value_type w_fund) {
        // resonator i was tuned to w0_i = order_i · w_fund_old; rescale by the
        // ratio so the per-resonator order is preserved.
        for (size_t i = 0; i < N; ++i) {
            const value_type order = resonators_[i].w0 / w_fund_;
            resonators_[i].set_frequency(order * w_fund);
        }
        w_fund_ = w_fund;
    }

    [[nodiscard]] constexpr const auto& resonator(size_t i) const { return resonators_[i]; }
    [[nodiscard]] constexpr bool        valid() const { return valid_; }

private:
    wet::array<PRController<value_type>, N> resonators_{};
    value_type                              w_fund_{value_type{1}};
    bool                                    valid_{true};
};

} // namespace wet
