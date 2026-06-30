#pragma once

#include "wet/estimation/kalman.hpp" // KalmanFilter (parameter random-walk)
#include "wet/math/math.hpp"         // wet::abs
#include "wet/matrix/colvec.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/motor/foc.hpp"           // DqCommand, design::voltage_circle_radius
#include "wet/systems/state_space.hpp" // StateSpace
#include "wet/transforms.hpp"          // DirectQuadrature

namespace wet::motor {

/**
 * @brief PMSM electrical nameplate the predictive controller inverts.
 * @tparam T Scalar type
 */
template<typename T = float>
struct PmsmModel {
    DirectQuadrature<T> Ldq{};    //!< [H] dq inductances (L_d, L_q)
    T                   R{};      //!< [ohm] phase resistance
    T                   lambda{}; //!< [Wb] permanent-magnet flux linkage
};

/**
 * @brief Deadbeat (one-step predictive) dq current controller — an alternative to
 *        the PI @ref FOController current loop.
 *
 * Same plug point as @ref FOController::current_controller (Idq_ref → Vdq), but
 * instead of PI feedback it inverts the discrete dq plant to command, in one sample,
 * the voltage that places the current at its reference. Forward-Euler discretization
 * of @f$ L\,\dot i = v - R i - \text{(cross/back-EMF)} @f$ gives the deadbeat law
 * @f[
 *   v_d = R\,i_d - \omega L_q i_q + \tfrac{L_d}{T_s}(i_d^* - i_d), \quad
 *   v_q = R\,i_q + \omega L_d i_d + \omega\lambda + \tfrac{L_q}{T_s}(i_q^* - i_q),
 * @f]
 * with cross-axis decoupling and back-EMF feedforward folded in. The command is held
 * to the SVPWM voltage circle (magnitude scaled, vector angle preserved) exactly as
 * @ref FOController does, so a saturating step simply slews as fast as the bus allows.
 *
 *
 * @warning Do not deploy this bare with a fixed nameplate model. Real @f$ R(T) @f$,
 *          @f$ L(\text{saturation}) @f$, and @f$ \lambda(\text{temp}) @f$ drift, and the
 *          deadbeat loop goes unstable once the inductance error reaches ~2× (the error
 *          multiplier @f$ 1-L_c/L_t @f$ leaves the unit disk). Use
 *          @ref AdaptivePredictiveCurrentController, which couples this law to the online
 *          @ref PmsmParameterEstimator so the model tracks the machine.
 *
 * @see FOController — the PI current loop this can replace.
 * @see Holtz & Quan, "Drift- and parameter-compensated flux estimator..."; Moon/Kim/Youn,
 *      "A discrete-time predictive current control for PMSM", IEEE T-PEL 18(1), 2003.
 *
 * @tparam T Scalar type
 */
template<typename T = float>
class PredictiveCurrentController {
public:
    constexpr PredictiveCurrentController() = default;

    constexpr explicit PredictiveCurrentController(const PmsmModel<T>& model, T max_modulation = T{1})
        : model_(model), max_modulation_(max_modulation) {}

    /**
     * @brief One deadbeat step.
     * @param Idq_ref    [A] dq current reference (from MTPA + FW upstream)
     * @param Idq        [A] measured dq current
     * @param omega_elec [rad/s] electrical speed
     * @param Vdc        [V] DC-bus voltage
     * @param dt         [s] sample period
     * @return @ref DqCommand: Vdq clamped to the voltage circle, plus saturation flags.
     */
    [[nodiscard]] constexpr DqCommand<T>
    control(const DirectQuadrature<T>& Idq_ref, const DirectQuadrature<T>& Idq, T omega_elec, T Vdc, T dt) const {
        const T Ld = model_.Ldq.d;
        const T Lq = model_.Ldq.q;
        const T R = model_.R;
        const T lam = model_.lambda;

        // Deadbeat inverse of the discrete dq plant (decoupling + back-EMF feedforward).
        DirectQuadrature<T> Vdq{
            .d = (R * Idq.d) - (omega_elec * Lq * Idq.q) + ((Ld / dt) * (Idq_ref.d - Idq.d)),
            .q = (R * Idq.q) + (omega_elec * Ld * Idq.d) + (omega_elec * lam) + ((Lq / dt) * (Idq_ref.q - Idq.q)),
        };

        // Circular voltage limit: scale magnitude, keep the vector angle.
        const T      Vmax = design::voltage_circle_radius(Vdc, max_modulation_);
        const T      Vmag = Vdq.abs();
        DqCommand<T> cmd;
        cmd.is_saturated = Vmag > Vmax;
        cmd.v_excess = (Vmax > T{0}) ? Vmag / Vmax : T{0};
        if (cmd.is_saturated) {
            Vdq = Vdq * (Vmax / Vmag);
        }
        cmd.Vdq = Vdq;
        return cmd;
    }

    constexpr void                              set_model(const PmsmModel<T>& m) { model_ = m; }
    [[nodiscard]] constexpr const PmsmModel<T>& model() const { return model_; }

private:
    PmsmModel<T> model_{};
    T            max_modulation_{T{1}};
};

/**
 * @brief Configuration for @ref PmsmParameterEstimator.
 *
 * The parameter vector is @f$ \theta = [R,\ L_d,\ L_q,\ \lambda] @f$, modelled as a
 * random walk. @p q_diag is the per-parameter process-noise variance — directional
 * forgetting: set it large for fast-drifting parameters (@f$ R(T) @f$) and small for
 * slow ones (@f$ L,\ \lambda @f$). Because the parameters span orders of magnitude
 * (@f$ R\!\sim\!10^{-1} @f$, @f$ L\!\sim\!10^{-3} @f$), @p p0_diag should be scaled per
 * parameter too.
 *
 * @tparam T Scalar type
 */
template<typename T = float>
struct PmsmEstimatorConfig {
    PmsmModel<T>    model0{};                         //!< initial parameter guess (seeds θ₀)
    Matrix<4, 4, T> Q{};                              //!< process-noise covariance, e.g. diagonal({qR,qLd,qLq,qλ})
    Matrix<4, 4, T> P0 = Matrix<4, 4, T>::identity(); //!< initial covariance (scale per-parameter)
    T               r{T{1}};                          //!< measurement-noise variance [V²]
};

/**
 * @brief Online PMSM electrical-parameter estimator (linear Kalman filter).
 *
 * Tracks @f$ \theta = [R, L_d, L_q, \lambda] @f$ from the dq voltage equations, which
 * are **linear in the parameters** (the cross/back-EMF products are of measured signals,
 * so they are known regressors, not unknowns):
 * @f[
 *   v_d = R\,i_d + L_d\,\dot i_d - L_q\,(\omega i_q), \qquad
 *   v_q = R\,i_q + L_d\,(\omega i_d) + L_q\,\dot i_q + \lambda\,\omega .
 * @f]
 * Modelling @f$ \theta @f$ as a random walk (@f$ A=I @f$, process noise @p q_diag) turns
 * this into a linear KF: each tick is one @ref KalmanFilter::predict followed by two
 * scalar updates (the d- and q-axis rows as the measurement matrix @f$ C @f$). The
 * per-parameter @f$ Q @f$ is the explicit, directional generalisation of an RLS
 * forgetting factor — it forgets each parameter at its own rate, so the poorly-excited
 * ones don't wind up. @f$ L_d, L_q @f$ are observable only while @f$ \dot i \neq 0 @f$;
 * @f$ R, \lambda @f$ from the quasi-steady terms (the latter needs @f$ \omega \neq 0 @f$).
 *
 * Differences use the previous current sample, so update with the voltage applied over
 * the last interval and the current measured now.
 *
 * @see RLS is the same recursion with a scalar forgetting factor; the KF form is chosen
 *      for the per-parameter Q. Simon, "Optimal State Estimation" (2006), §5.
 *
 * @tparam T Scalar type
 */
template<typename T = float>
class PmsmParameterEstimator {
public:
    constexpr PmsmParameterEstimator() = default;

    constexpr explicit PmsmParameterEstimator(const PmsmEstimatorConfig<T>& cfg) : r_(cfg.r) {
        StateSpace<4, 1, 1, 4, 1, T> sys{};
        sys.A = Matrix<4, 4, T>::identity(); // θ is a random walk
        sys.G = Matrix<4, 4, T>::identity();
        sys.H = Matrix<1, 1, T>{{T{1}}};
        ColVec<4, T> th0{};
        th0[0] = cfg.model0.R;
        th0[1] = cfg.model0.Ldq.d;
        th0[2] = cfg.model0.Ldq.q;
        th0[3] = cfg.model0.lambda;
        kf_ = KalmanFilter<4, 1, 1, 4, 1, T>{sys, cfg.Q, Matrix<1, 1, T>{{cfg.r}}, th0, cfg.P0};
    }

    /**
     * @brief One adaptation step.
     * @param Vdq        [V] voltage applied over the last interval (previous command)
     * @param Idq        [A] current measured now
     * @param omega_elec [rad/s] electrical speed
     * @param dt         [s] sample period
     */
    constexpr void update(const DirectQuadrature<T>& Vdq, const DirectQuadrature<T>& Idq, T omega_elec, T dt) {
        if (initialized_) {
            const T did = (Idq.d - prev_.d) / dt;
            const T diq = (Idq.q - prev_.q) / dt;
            const T id = prev_.d; // currents at the start of the interval (i[k])
            const T iq = prev_.q;

            kf_.predict();
            const Matrix<1, 1, T> D0{{T{0}}};
            const Matrix<1, 1, T> Rm{{r_}};
            // d-axis: vd = R·id + Ld·did − Lq·(ω·iq)
            (void)kf_.update(ColVec<1, T>{Vdq.d}, Matrix<1, 4, T>{{id, did, -omega_elec * iq, T{0}}}, D0, Rm);
            // q-axis: vq = R·iq + Ld·(ω·id) + Lq·diq + λ·ω
            (void)kf_.update(ColVec<1, T>{Vdq.q}, Matrix<1, 4, T>{{iq, omega_elec * id, diq, omega_elec}}, D0, Rm);
        }
        prev_ = Idq;
        initialized_ = true;
    }

    /// Current parameter estimate as a model, floored to physical (positive L, R).
    [[nodiscard]] constexpr PmsmModel<T> model() const {
        const auto& th = kf_.state();
        constexpr T floor = T{1e-9};
        return PmsmModel<T>{
            .Ldq = {.d = th[1] > floor ? th[1] : floor, .q = th[2] > floor ? th[2] : floor},
            .R = th[0] > floor ? th[0] : floor,
            .lambda = th[3],
        };
    }

private:
    KalmanFilter<4, 1, 1, 4, 1, T> kf_{};
    DirectQuadrature<T>            prev_{};
    T                              r_{T{1}};
    bool                           initialized_{false};
};

/**
 * @brief Self-tuning deadbeat current controller: @ref PredictiveCurrentController plus
 *        the online @ref PmsmParameterEstimator.
 *
 * This is the shippable predictive loop. Each tick it adapts the model from the last
 * applied voltage and the current measurement, refreshes the deadbeat law, then commands.
 * The estimator keeps @f$ L_{dq}/R/\lambda @f$ tracking the real machine, which is what
 * makes deadbeat current control robust enough to deploy — see the bare
 * @ref PredictiveCurrentController warning for why the fixed-model version is not.
 *
 * Same plug point as @ref FOController::current_controller (Idq_ref → Vdq).
 *
 * @tparam T Scalar type
 */
template<typename T = float>
class AdaptivePredictiveCurrentController {
public:
    constexpr AdaptivePredictiveCurrentController() = default;

    constexpr explicit AdaptivePredictiveCurrentController(const PmsmEstimatorConfig<T>& ecfg, T max_modulation = T{1})
        : est_(ecfg), ctrl_(ecfg.model0, max_modulation) {}

    /// One step: adapt the model, then deadbeat-command. See PredictiveCurrentController::control.
    [[nodiscard]] constexpr DqCommand<T>
    control(const DirectQuadrature<T>& Idq_ref, const DirectQuadrature<T>& Idq, T omega_elec, T Vdc, T dt) {
        est_.update(prev_vdq_, Idq, omega_elec, dt); // adapt from the last interval
        ctrl_.set_model(est_.model());
        const auto cmd = ctrl_.control(Idq_ref, Idq, omega_elec, Vdc, dt);
        prev_vdq_ = cmd.Vdq;
        return cmd;
    }

    [[nodiscard]] constexpr const PmsmParameterEstimator<T>& estimator() const { return est_; }
    [[nodiscard]] constexpr PmsmModel<T>                     model() const { return ctrl_.model(); }

private:
    PmsmParameterEstimator<T>      est_{};
    PredictiveCurrentController<T> ctrl_{};
    DirectQuadrature<T>            prev_vdq_{};
};

} // namespace wet::motor
