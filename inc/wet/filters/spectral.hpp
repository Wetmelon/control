#pragma once

/**
 * @file spectral.hpp
 * @brief Single-frequency spectral primitives for control: a generalized
 *        Goertzel single-bin DFT and a thin harmonic analyzer (THD / RMS).
 *
 * Deliberately *basic* — this is the controls layer, not a DSP library. The use
 * case is measuring a *known* frequency component (the grid fundamental and its
 * harmonics, a motor's electrical/mechanical ripple order) cheaply, in an ISR,
 * without a full FFT. For general spectral analysis use a dedicated DSP library.
 *
 * The Goertzel algorithm computes one DFT bin with a two-state recurrence — no
 * sample buffer, O(1) memory, O(N) per block. It is exact for *coherent*
 * sampling (an integer number of cycles of the target frequency in the block);
 * choose the block length N ≈ integer·fs/f for the best results.
 *
 * All constexpr, allocation-free, float/double.
 */

#include <cstddef>

#include "wet/backend.hpp"              // wet::array, wet::numbers
#include "wet/math/math.hpp"            // wet::sin/cos/sqrt/atan2
#include "wet/matrix/matrix_traits.hpp" // scalar_type_t

namespace wet {

/**
 * @ingroup filters
 * @brief Generalized Goertzel single-bin DFT — amplitude/phase at one frequency.
 *
 * Feed samples one at a time with @ref push; when it returns true a block of N
 * samples is complete and @ref amplitude / @ref phase / @ref power are valid
 * until the next @ref push (which begins the next block). Detecting the same
 * frequency continuously is just a loop of `if (g.push(x)) { use g.amplitude() }`.
 *
 * @tparam T Scalar type (float or double)
 */
template<typename T = float>
class Goertzel {
public:
    constexpr Goertzel() = default;

    /**
     * @brief Configure for @p freq_hz at sample rate @p fs_hz over @p block_size samples.
     * @param freq_hz    Target frequency [Hz] (0 ≤ freq_hz < fs_hz/2 for a meaningful bin)
     * @param fs_hz      Sample rate [Hz]
     * @param block_size Samples per block, N (≥ 1)
     */
    constexpr Goertzel(T freq_hz, T fs_hz, size_t block_size)
        : n_(block_size) {
        const T omega = (T{2} * wet::numbers::pi_v<T> * freq_hz) / fs_hz;
        cos_ = wet::cos(omega);
        sin_ = wet::sin(omega);
        coeff_ = T{2} * cos_;
    }

    /**
     * @brief Feed one sample.
     * @return true when the block just completed (results valid until next push).
     */
    constexpr bool push(T x) {
        if (complete_) {
            // Auto-restart for the next block.
            s1_ = T{0};
            s2_ = T{0};
            count_ = 0;
            complete_ = false;
        }
        const T s = x + (coeff_ * s1_) - s2_;
        s2_ = s1_;
        s1_ = s;
        if (++count_ >= n_) {
            real_ = s1_ - (s2_ * cos_);
            imag_ = s2_ * sin_;
            complete_ = true;
            return true;
        }
        return false;
    }

    /// True once a full block has been accumulated.
    [[nodiscard]] constexpr bool complete() const { return complete_; }

    /// Magnitude of the DFT bin, |X_k|.
    [[nodiscard]] constexpr T magnitude() const { return wet::sqrt((real_ * real_) + (imag_ * imag_)); }

    /// Power, |X_k|².
    [[nodiscard]] constexpr T power() const { return (real_ * real_) + (imag_ * imag_); }

    /// Peak amplitude of a sinusoid at the bin frequency (2·|X_k| / N).
    [[nodiscard]] constexpr T amplitude() const {
        return (T{2} * magnitude()) / static_cast<T>(n_);
    }

    /// Phase of the bin [rad] (Goertzel convention; consistent across harmonics).
    [[nodiscard]] constexpr T phase() const { return wet::atan2(imag_, real_); }

    /// Discard the in-progress block and start over.
    constexpr void reset() {
        s1_ = T{0};
        s2_ = T{0};
        count_ = 0;
        complete_ = false;
        real_ = T{0};
        imag_ = T{0};
    }

    [[nodiscard]] constexpr size_t block_size() const { return n_; }

private:
    T      coeff_{T{0}};
    T      cos_{T{1}};
    T      sin_{T{0}};
    size_t n_{1};

    T      s1_{T{0}};
    T      s2_{T{0}};
    size_t count_{0};

    T    real_{T{0}};
    T    imag_{T{0}};
    bool complete_{false};
};

/**
 * @ingroup filters
 * @brief Harmonic analyzer — a Goertzel bank over a fundamental and K−1 harmonics.
 *
 * Measures the amplitude/phase of the fundamental f0 and its harmonics 2·f0 …
 * K·f0 in one pass, and reports total harmonic distortion and RMS. The headline
 * uses are grid voltage/current THD and motor torque-ripple order analysis. The
 * fundamental @ref rms is leakage-immune on a coherently-sampled block (unlike a
 * boxcar RMS, which sees the harmonics).
 *
 * @tparam K Number of harmonics tracked, including the fundamental (K ≥ 1)
 * @tparam T Scalar type (float or double)
 */
template<size_t K, typename T = float>
class HarmonicAnalyzer {
public:
    static_assert(K >= 1, "HarmonicAnalyzer needs at least the fundamental (K >= 1)");

    constexpr HarmonicAnalyzer() = default;

    /**
     * @brief Configure for fundamental @p f0_hz at sample rate @p fs_hz over @p block_size samples.
     */
    constexpr HarmonicAnalyzer(T f0_hz, T fs_hz, size_t block_size) {
        for (size_t h = 0; h < K; ++h) {
            bins_[h] = Goertzel<T>(f0_hz * static_cast<T>(h + 1), fs_hz, block_size);
        }
    }

    /**
     * @brief Feed one sample to every harmonic bin.
     * @return true when the block just completed (results valid until next push).
     */
    constexpr bool push(T x) {
        bool done = false;
        for (size_t h = 0; h < K; ++h) {
            done = bins_[h].push(x);
        }
        return done;
    }

    /// Peak amplitude of harmonic @p harmonic (1 = fundamental, … K).
    [[nodiscard]] constexpr T amplitude(size_t harmonic) const {
        return bins_[harmonic - 1].amplitude();
    }

    /// Phase of harmonic @p harmonic [rad] (1 = fundamental, … K).
    [[nodiscard]] constexpr T phase(size_t harmonic) const { return bins_[harmonic - 1].phase(); }

    /// RMS of the fundamental (A₁ / √2).
    [[nodiscard]] constexpr T rms() const {
        return bins_[0].amplitude() / wet::numbers::sqrt2_v<T>;
    }

    /// RMS of the harmonics above the fundamental, √(Σ_{h≥2} (Aₕ/√2)²).
    [[nodiscard]] constexpr T total_harmonic_rms() const {
        T sum_sq = T{0};
        for (size_t h = 1; h < K; ++h) {
            const T a = bins_[h].amplitude();
            sum_sq += a * a;
        }
        return wet::sqrt(sum_sq) / wet::numbers::sqrt2_v<T>;
    }

    /// Total harmonic distortion, √(Σ_{h≥2} Aₕ²) / A₁ (0 if the fundamental is ~0).
    [[nodiscard]] constexpr T thd() const {
        const T fundamental = bins_[0].amplitude();
        if (fundamental <= default_tol<T>()) {
            return T{0};
        }
        T sum_sq = T{0};
        for (size_t h = 1; h < K; ++h) {
            const T a = bins_[h].amplitude();
            sum_sq += a * a;
        }
        return wet::sqrt(sum_sq) / fundamental;
    }

    [[nodiscard]] constexpr bool complete() const { return bins_[0].complete(); }

    constexpr void reset() {
        for (size_t h = 0; h < K; ++h) {
            bins_[h].reset();
        }
    }

private:
    wet::array<Goertzel<T>, K> bins_{};
};

} // namespace wet
