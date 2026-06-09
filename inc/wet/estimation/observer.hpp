#pragma once

/**
 * @file observer.hpp
 * @brief Luenberger state observer: deterministic pole-placement state estimation.
 *
 * The deterministic counterpart to the Kalman filter. Given a plant
 *
 *     x[k+1] = A x[k] + B u[k]
 *     y[k]   = C x[k] + D u[k]
 *
 * an observer reconstructs the state from inputs and outputs:
 *
 *     x̂[k+1] = A x̂[k] + B u[k] + L (y[k] − C x̂[k] − D u[k])
 *
 * The estimation error e = x − x̂ evolves as e[k+1] = (A − L C) e[k], so the
 * gain L is chosen to place the eigenvalues of (A − L C) at desired locations.
 * This is the dual of state-feedback pole placement: L = acker(Aᵀ, Cᵀ, p)ᵀ.
 *
 * The gain is computed by the observer form of Ackermann's formula:
 *
 *     L = φ(A) · O⁻¹ · [0 … 0 1]ᵀ,   O = [C; CA; …; CA^{n−1}]
 *
 * where φ is the desired characteristic polynomial. Ackermann placement is
 * single-output only (NY == 1) — a limitation of the algorithm, not of
 * observers: multi-output Luenberger observers exist but need a MIMO placement
 * routine. For multiple outputs today, use the Kalman filter.
 *
 * A reduced-order (Gopinath) observer is also provided: it estimates only the
 * unmeasured states and reads the measured state directly from y — ideal when
 * the measurement is essentially noise-free (e.g. encoder counts).
 *
 * Example: observer for a double integrator (position measured, velocity estimated)
 * @code
 * #include "wet/estimation/observer.hpp"
 * #include "wet/systems/state_space.hpp"
 *
 * using namespace wet;
 *
 * constexpr StateSpace sys{
 *     .A = Matrix<2,2>{{1.0, 0.1}, {0.0, 1.0}},   // 10 Hz discrete double integrator
 *     .B = Matrix<2,1>{{0.005}, {0.1}},
 *     .C = Matrix<1,2>{{1.0, 0.0}},               // measure position only
 *     .Ts = 0.1
 * };
 *
 * // Place observer error poles inside the unit circle (faster than the plant).
 * constexpr auto result = design::synthesize_observer(sys, ColVec<2>{0.3, 0.4});
 * static_assert(result.success);
 *
 * Observer observer(sys, result.as<float>());   // deploy in float
 * // In the loop: observer.step(y, u); auto xhat = observer.state();
 * @endcode
 *
 * @see "An Introduction to Observers" (Luenberger, 1971), IEEE TAC
 * @see kalman.hpp for the stochastic estimator (and MIMO outputs)
 */

#include <cstddef>

#include "wet/backend.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/state_space.hpp"

namespace wet {

namespace design {

/**
 * @struct ObserverResult
 * @brief Luenberger observer design result.
 *
 * Contains the observer gain L and the achieved observer error poles. Use
 * .as<float>() to convert for embedded deployment.
 *
 * @tparam NX Number of states
 * @tparam NY Number of outputs (must be 1 for Ackermann placement)
 * @tparam T  Scalar type
 *
 * @see "An Introduction to Observers" (Luenberger, 1971)
 */
template<size_t NX, size_t NY, typename T = double>
struct ObserverResult {
    Matrix<NX, NY, T>           L{};            ///< Observer gain: x̂ ← x̂ + L (y − ŷ)
    ColVec<NX, wet::complex<T>> e{};            ///< Observer error poles (eigenvalues of A − LC)
    bool                        success{false}; ///< true if the system is observable and L was placed

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return ObserverResult<NX, NY, U>{
            L.template as<U>(),
            e.template as<wet::complex<U>>(),
            success
        };
    }

    /**
     * @brief Check whether the observer error dynamics are stable (discrete-time).
     * @return true if all error poles lie strictly inside the unit circle.
     */
    [[nodiscard]] constexpr bool is_stable() const {
        for (size_t i = 0; i < NX; ++i) {
            if (e[i].abs() >= T{1}) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief Design a Luenberger observer by pole placement (matrix form).
 *
 * Places the eigenvalues of (A − L C) at @p desired_poles using the observer
 * form of Ackermann's formula. Single-output only (NY == 1); for NY ≠ 1 the
 * result is returned with success = false.
 *
 * @note Compare with MATLAB's L = place(A', C', p)' or L = acker(A', C', p)'.
 *
 * @param A             State transition matrix (NX × NX)
 * @param C             Output matrix (NY × NX), NY must be 1
 * @param desired_poles Desired error poles (complex; conjugate pairs allowed)
 * @return ObserverResult with gain L and achieved poles e
 *
 * @see acker() for the dual state-feedback placement
 */
template<size_t NX, size_t NY, typename T = double>
[[nodiscard]] constexpr ObserverResult<NX, NY, T> synthesize_observer(
    const Matrix<NX, NX, T>&           A,
    const Matrix<NY, NX, T>&           C,
    const ColVec<NX, wet::complex<T>>& desired_poles
) {
    ObserverResult<NX, NY, T> result{};
    result.e = desired_poles;

    if constexpr (NY != 1) {
        // Ackermann placement is single-output; use the Kalman filter for MIMO.
        return result;
    } else {
        // Observability matrix O = [C; CA; …; CA^{NX-1}]  (NX × NX for NY == 1).
        Matrix<NX, NX, T> O{};
        Matrix<NY, NX, T> CA = C;
        for (size_t k = 0; k < NX; ++k) {
            for (size_t c = 0; c < NX; ++c) {
                O(k, c) = CA(0, c);
            }
            if (k + 1 < NX) {
                CA = CA * A;
            }
        }

        const auto O_inv_opt = O.inverse();
        if (!O_inv_opt) {
            return result; // not observable: success stays false
        }
        const Matrix<NX, NX, T> O_inv = *O_inv_opt;

        // Desired characteristic polynomial: φ(s) = Π (s − pᵢ).
        // Build with complex arithmetic so conjugate pairs cancel to real coeffs.
        wet::array<wet::complex<T>, NX + 1> cc{};
        cc[0] = wet::complex<T>{T{1}, T{0}};
        for (size_t i = 0; i < NX; ++i) {
            const wet::complex<T> root = desired_poles[i];
            wet::complex<T>       carry = cc[0];
            cc[0] = wet::complex<T>{T{0}, T{0}} - (root * cc[0]);
            for (size_t j = 1; j <= NX; ++j) {
                const wet::complex<T> next = cc[j];
                cc[j] = carry - (root * cc[j]);
                carry = next;
            }
        }
        wet::array<T, NX + 1> coeffs{};
        for (size_t k = 0; k <= NX; ++k) {
            coeffs[k] = cc[k].real();
        }

        // φ(A) = Σ coeffs[k] · Aᵏ
        Matrix<NX, NX, T> phi_A = Matrix<NX, NX, T>::zeros();
        Matrix<NX, NX, T> A_power = Matrix<NX, NX, T>::identity();
        for (size_t k = 0; k <= NX; ++k) {
            phi_A = phi_A + (coeffs[k] * A_power);
            A_power = A_power * A;
        }

        // L = φ(A) · O⁻¹ · [0 … 0 1]ᵀ
        ColVec<NX, T> e_N{};
        e_N(NX - 1, 0) = T{1};
        result.L = Matrix<NX, NY, T>(phi_A * O_inv * e_N);
        result.success = true;
        return result;
    }
}

/**
 * @brief Design a Luenberger observer from real desired poles.
 */
template<size_t NX, size_t NY, typename T = double>
[[nodiscard]] constexpr ObserverResult<NX, NY, T> synthesize_observer(
    const Matrix<NX, NX, T>& A,
    const Matrix<NY, NX, T>& C,
    const ColVec<NX, T>&     desired_poles
) {
    ColVec<NX, wet::complex<T>> poles{};
    for (size_t i = 0; i < NX; ++i) {
        poles(i, 0) = wet::complex<T>{desired_poles[i], T{0}};
    }
    return synthesize_observer<NX, NY, T>(A, C, poles);
}

/**
 * @brief Design a Luenberger observer from a StateSpace system (complex poles).
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr ObserverResult<NX, NY, T> synthesize_observer(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const ColVec<NX, wet::complex<T>>&       desired_poles
) {
    return synthesize_observer<NX, NY, T>(sys.A, sys.C, desired_poles);
}

/**
 * @brief Design a Luenberger observer from a StateSpace system (real poles).
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr ObserverResult<NX, NY, T> synthesize_observer(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const ColVec<NX, T>&                     desired_poles
) {
    return synthesize_observer<NX, NY, T>(sys.A, sys.C, desired_poles);
}

/**
 * @struct ReducedObserverResult
 * @brief Reduced-order (Gopinath) observer design result.
 *
 * Estimates only the NX−1 unmeasured states from a single measured output,
 * reading the measured state directly from y. Carries the runtime-ready
 * matrices; use .as<float>() for embedded deployment.
 *
 * @tparam NX Number of plant states
 * @tparam NU Number of inputs
 * @tparam T  Scalar type
 *
 * @see "On the Synthesis of Minimal-Order Observers" (Gopinath, 1971)
 */
template<size_t NX, size_t NU, typename T = double>
struct ReducedObserverResult {
    static constexpr size_t NM = NX - 1; ///< Number of estimated (unmeasured) states

    Matrix<NM, NM, T>           F{};            ///< Internal dynamics (Abb − L Aab); error poles
    Matrix<NM, 1, T>            L{};            ///< Reduced gain (x̂_b = z + L y)
    Matrix<NM, 1, T>            Gy{};           ///< Internal-state update gain on y
    Matrix<NM, NU, T>           Gu{};           ///< Internal-state update gain on u
    Matrix<NX, NX, T>           Tinv{};         ///< State reconstruction: x = Tinv [y; x̂_b]
    ColVec<NM, wet::complex<T>> e{};            ///< Placed error poles
    bool                        success{false}; ///< true if observable and L was placed

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return ReducedObserverResult<NX, NU, U>{
            F.template as<U>(),
            L.template as<U>(),
            Gy.template as<U>(),
            Gu.template as<U>(),
            Tinv.template as<U>(),
            e.template as<wet::complex<U>>(),
            success
        };
    }

    /**
     * @brief Check whether the reduced error dynamics are stable (discrete-time).
     */
    [[nodiscard]] constexpr bool is_stable() const {
        for (size_t i = 0; i < NM; ++i) {
            if (e[i].abs() >= T{1}) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief Design a reduced-order (Gopinath) observer by pole placement (matrix form).
 *
 * Estimates the NX−1 unmeasured states from a single measured output, reading
 * the measured state directly from y (no filtering of the measurement). Places
 * the NX−1 error poles eig(A_bb − L A_ab) via Ackermann on the reduced
 * subsystem. Single-output only (NY == 1).
 *
 * @param A             State matrix (NX × NX)
 * @param B             Input matrix (NX × NU)
 * @param C             Output matrix (1 × NX); one dominant entry (a state measurement)
 * @param desired_poles NX−1 desired error poles (complex; conjugate pairs allowed)
 * @return ReducedObserverResult with runtime matrices
 *
 * @see "On the Synthesis of Minimal-Order Observers" (Gopinath, 1971)
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] constexpr ReducedObserverResult<NX, NU, T> synthesize_reduced_observer(
    const Matrix<NX, NX, T>&               A,
    const Matrix<NX, NU, T>&               B,
    const Matrix<NY, NX, T>&               C,
    const ColVec<NX - 1, wet::complex<T>>& desired_poles
) {
    static_assert(NX >= 2, "Reduced-order observer requires at least two states.");
    static_assert(NY == 1, "Reduced-order observer is single-output (NY == 1).");
    constexpr size_t NM = NX - 1;

    ReducedObserverResult<NX, NU, T> result{};
    result.e = desired_poles;

    // Pick the measured-state pivot (largest |C| entry) and build the transform
    // Tf = [C; eⱼ for j ≠ m] so the first transformed state equals the output y.
    size_t m = 0;
    T      best = wet::abs(C(0, 0));
    for (size_t j = 1; j < NX; ++j) {
        const T v = wet::abs(C(0, j));
        if (v > best) {
            best = v;
            m = j;
        }
    }

    Matrix<NX, NX, T> Tf = Matrix<NX, NX, T>::zeros();
    for (size_t j = 0; j < NX; ++j) {
        Tf(0, j) = C(0, j);
    }
    size_t r = 1;
    for (size_t j = 0; j < NX; ++j) {
        if (j == m) {
            continue;
        }
        Tf(r, j) = T{1};
        ++r;
    }

    const auto Tf_inv_opt = Tf.inverse();
    if (!Tf_inv_opt) {
        return result; // C not full rank: success stays false
    }
    const Matrix<NX, NX, T> Tf_inv = *Tf_inv_opt;

    // Transform to coordinates where the first state is the measurement.
    const Matrix<NX, NX, T> A_bar = Tf * A * Tf_inv;
    const Matrix<NX, NU, T> B_bar = Tf * B;

    const Matrix<1, 1, T>   A_aa = A_bar.template block<1, 1>(0, 0);
    const Matrix<1, NM, T>  A_ab = A_bar.template block<1, NM>(0, 1);
    const Matrix<NM, 1, T>  A_ba = A_bar.template block<NM, 1>(1, 0);
    const Matrix<NM, NM, T> A_bb = A_bar.template block<NM, NM>(1, 1);
    const Matrix<1, NU, T>  B_a = B_bar.template block<1, NU>(0, 0);
    const Matrix<NM, NU, T> B_b = B_bar.template block<NM, NU>(1, 0);

    // Place eig(A_bb − L A_ab) via the observer Ackermann on the reduced system.
    const auto sub = synthesize_observer<NM, 1, T>(A_bb, A_ab, desired_poles);
    if (!sub.success) {
        return result; // unobservable reduced subsystem
    }
    const Matrix<NM, 1, T> L = sub.L;

    const Matrix<NM, NM, T> F = A_bb - L * A_ab;
    result.F = F;
    result.L = L;
    result.Gy = Matrix<NM, 1, T>(F * L + (A_ba - L * A_aa));
    result.Gu = Matrix<NM, NU, T>(B_b - L * B_a);
    result.Tinv = Tf_inv;
    result.success = true;
    return result;
}

/**
 * @brief Design a reduced-order observer from real desired poles.
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] constexpr ReducedObserverResult<NX, NU, T> synthesize_reduced_observer(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NY, NX, T>& C,
    const ColVec<NX - 1, T>& desired_poles
) {
    ColVec<NX - 1, wet::complex<T>> poles{};
    for (size_t i = 0; i < NX - 1; ++i) {
        poles(i, 0) = wet::complex<T>{desired_poles[i], T{0}};
    }
    return synthesize_reduced_observer<NX, NU, NY, T>(A, B, C, poles);
}

/**
 * @brief Design a reduced-order observer from a StateSpace system (complex poles).
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr ReducedObserverResult<NX, NU, T> synthesize_reduced_observer(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const ColVec<NX - 1, wet::complex<T>>&   desired_poles
) {
    return synthesize_reduced_observer<NX, NU, NY, T>(sys.A, sys.B, sys.C, desired_poles);
}

/**
 * @brief Design a reduced-order observer from a StateSpace system (real poles).
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr ReducedObserverResult<NX, NU, T> synthesize_reduced_observer(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const ColVec<NX - 1, T>&                 desired_poles
) {
    return synthesize_reduced_observer<NX, NU, NY, T>(sys.A, sys.B, sys.C, desired_poles);
}

} // namespace design

/**
 * @ingroup discrete_estimators
 * @brief Luenberger state observer (runtime).
 *
 * Reconstructs the plant state from inputs and outputs using a precomputed
 * gain L. Call step(y, u) once per sample; the estimate is the one-step-ahead
 * prediction x̂[k+1|k].
 *
 * @tparam NX Number of states
 * @tparam NU Number of inputs
 * @tparam NY Number of outputs
 * @tparam T  Scalar type (default: float for embedded deployment)
 */
template<size_t NX, size_t NU, size_t NY, typename T = float>
class Observer {
public:
    constexpr Observer() = default;

    /// Construct from system matrices and a precomputed gain.
    constexpr Observer(
        const Matrix<NX, NX, T>& A,
        const Matrix<NX, NU, T>& B,
        const Matrix<NY, NX, T>& C,
        const Matrix<NY, NU, T>& D,
        const Matrix<NX, NY, T>& L
    )
        : A_(A), B_(B), C_(C), D_(D), L_(L) {}

    /// Construct from a StateSpace system and a precomputed gain.
    template<size_t NW, size_t NV>
    constexpr Observer(const StateSpace<NX, NU, NY, NW, NV, T>& sys, const Matrix<NX, NY, T>& L)
        : A_(sys.A), B_(sys.B), C_(sys.C), D_(sys.D), L_(L) {}

    /// Construct from a StateSpace system and an observer design result.
    template<size_t NW, size_t NV>
    constexpr Observer(const StateSpace<NX, NU, NY, NW, NV, T>& sys, const design::ObserverResult<NX, NY, T>& result)
        : A_(sys.A), B_(sys.B), C_(sys.C), D_(sys.D), L_(result.L) {}

    /**
     * @brief One observer recursion (predictor form):
     *
     *     x̂ ← A x̂ + B u + L (y − C x̂ − D u)
     *
     * The estimate is the one-step-ahead prediction x̂[k+1|k]; the estimation
     * error then evolves as e ← (A − L C) e, matching the placed poles.
     *
     * @param y Measured output at the current step
     * @param u Input applied at the current step
     */
    constexpr void step(const ColVec<NY, T>& y, const ColVec<NU, T>& u = ColVec<NU, T>{}) {
        const ColVec<NY, T> innovation = ColVec<NY, T>(y - (C_ * x_ + D_ * u));
        x_ = ColVec<NX, T>(A_ * x_ + B_ * u + L_ * innovation);
    }

    /// Current state estimate.
    [[nodiscard]] constexpr const ColVec<NX, T>& state() const { return x_; }

    /// Reset the estimate (default zero).
    constexpr void reset(const ColVec<NX, T>& x0 = ColVec<NX, T>{}) { x_ = x0; }

    /**
     * @brief Overwrite the state estimate (constraint enforcement).
     *
     * Use after step() to clamp the estimate to a physically meaningful range,
     * wrap an angle, or zero a non-physical state before the next step()
     * propagates from it. Unlike reset(), this is for per-tick constraint
     * projection, not re-initialization.
     */
    constexpr void set_state(const ColVec<NX, T>& x_new) { x_ = x_new; }
    constexpr void set_state(size_t i, T value) { x_[i] = value; }

private:
    Matrix<NX, NX, T> A_{};
    Matrix<NX, NU, T> B_{};
    Matrix<NY, NX, T> C_{};
    Matrix<NY, NU, T> D_{};
    Matrix<NX, NY, T> L_{};
    ColVec<NX, T>     x_{};
};

/**
 * @ingroup discrete_estimators
 * @brief Reduced-order (Gopinath) state observer (runtime).
 *
 * Estimates the NX−1 unmeasured states from a single measured output and
 * reconstructs the full state, reading the measured state directly from y.
 * Use when the measurement is essentially noise-free (e.g. encoder counts):
 * it does no filtering of the measured channel. For noisy measurements prefer
 * the full-order Observer or the Kalman filter.
 *
 * @tparam NX Number of plant states
 * @tparam NU Number of inputs
 * @tparam T  Scalar type (default: float for embedded deployment)
 */
template<size_t NX, size_t NU, typename T = float>
class ReducedOrderObserver {
    static constexpr size_t NM = NX - 1;

public:
    constexpr ReducedOrderObserver() = default;

    constexpr explicit ReducedOrderObserver(const design::ReducedObserverResult<NX, NU, T>& result)
        : F_(result.F), L_(result.L), Gy_(result.Gy), Gu_(result.Gu), Tinv_(result.Tinv) {}

    /**
     * @brief One observer recursion.
     *
     * Forms the unmeasured-state estimate x̂_b = z + L y, reconstructs the full
     * state x̂ = Tinv [y; x̂_b], then advances the internal state z. The reduced
     * estimation error evolves as e ← (A_bb − L A_ab) e.
     *
     * @param y Measured output (single channel)
     * @param u Input applied at the current step
     */
    constexpr void step(const ColVec<1, T>& y, const ColVec<NU, T>& u = ColVec<NU, T>{}) {
        const ColVec<NM, T> x_b = ColVec<NM, T>(z_ + L_ * y);

        ColVec<NX, T> x_bar{};
        x_bar(0, 0) = y(0, 0);
        x_bar.template block<NM, 1>(1, 0) = x_b;
        x_ = ColVec<NX, T>(Tinv_ * x_bar);

        z_ = ColVec<NM, T>(F_ * z_ + Gy_ * y + Gu_ * u);
    }

    /// Scalar-measurement convenience overload.
    constexpr void step(T y, const ColVec<NU, T>& u = ColVec<NU, T>{}) {
        step(ColVec<1, T>{y}, u);
    }

    /// Current full-state estimate.
    [[nodiscard]] constexpr const ColVec<NX, T>& state() const { return x_; }

    /// Internal recursion state (estimate of the NX−1 unmeasured states in
    /// transformed coordinates), before the +L·y reconstruction.
    [[nodiscard]] constexpr const ColVec<NM, T>& internal_state() const { return z_; }

    /// Reset the internal and reconstructed states to zero.
    constexpr void reset() {
        z_ = ColVec<NM, T>{};
        x_ = ColVec<NX, T>{};
    }

    /**
     * @brief Overwrite the internal recursion state z (constraint enforcement).
     *
     * The reduced observer reconstructs the full estimate x̂ = Tinv·[y; z + L·y]
     * fresh on every step(), so writing the full state directly would be
     * discarded next tick — the persistent state is the internal z. To clamp an
     * unmeasured state, map your constraint into z. The measured channel is read
     * straight from y and is never filtered, so it needs no clamping here.
     */
    constexpr void set_internal_state(const ColVec<NM, T>& z_new) { z_ = z_new; }
    constexpr void set_internal_state(size_t i, T value) { z_[i] = value; }

private:
    Matrix<NM, NM, T> F_{};
    Matrix<NM, 1, T>  L_{};
    Matrix<NM, 1, T>  Gy_{};
    Matrix<NM, NU, T> Gu_{};
    Matrix<NX, NX, T> Tinv_{};
    ColVec<NM, T>     z_{};
    ColVec<NX, T>     x_{};
};

} // namespace wet
