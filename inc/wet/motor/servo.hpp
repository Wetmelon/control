#pragma once

#include <cstdint>
#include <limits>

#include "wet/backend.hpp"
#include "wet/controllers/pid.hpp"
#include "wet/design/pid_design.hpp"
#include "wet/math/math.hpp" // wrap, numbers::pi_v
#include "wet/matrix/colvec.hpp"
#include "wet/motor/foc.hpp"
#include "wet/motor/limits.hpp"
#include "wet/motor/modulation.hpp"
#include "wet/motor/rotor_observer.hpp"
#include "wet/transforms.hpp"

namespace wet::motor {

/// Servo control mode: which outer loops are active.
enum class ControlMode : std::uint8_t {
    Torque,   //!< command is q-axis torque [Nm]; current loop only
    Velocity, //!< command is speed [turns/s]; velocity + current loops
    Position, //!< command is angle [turns]; position + velocity + current loops
};

/**
 * @brief The three bandwidth knobs of the position/velocity/current cascade.
 *
 * One number per loop, in rad/s. Nested loops must be bandwidth-separated for the
 * successive-loop-closure tuning to hold — roughly a factor of ~5 per stage; @ref valid
 * checks it.
 *
 * @tparam T Scalar type.
 */
template<typename T = float>
struct CascadeBandwidths {
    T omega_position{}; //!< [rad/s] outer position-loop bandwidth
    T omega_velocity{}; //!< [rad/s] middle velocity-loop bandwidth
    T omega_current{};  //!< [rad/s] inner current/torque-loop bandwidth

    [[nodiscard]] constexpr bool valid(T ratio = T{5}) const {
        return (omega_position > T{0}) && (omega_velocity > T{0}) && (omega_current > T{0})
            && (omega_velocity >= ratio * omega_position) && (omega_current >= ratio * omega_velocity);
    }
};

/**
 * @brief Configuration for @ref PmacServo.
 * @tparam T Scalar type.
 */
template<typename T = float>
struct PmacServoConfig {
    DirectQuadrature<T> Ldq{}; //!< [H] dq inductances (from calibration)

    T R{};              //!< [ohm] phase resistance (from calibration)
    T lambda{};         //!< [Wb] PM flux linkage
    T pole_pairs{T{1}}; //!< pole pairs
    T J{T{1}};          //!< [kg·m²] reflected inertia
    T b{T{0}};          //!< [Nm·s] viscous friction

    T iq_max{std::numeric_limits<T>::max()};     //!< [A] q-axis current ceiling
    T vel_max{std::numeric_limits<T>::max()};    //!< [turns/s] velocity-command ceiling
    T torque_max{std::numeric_limits<T>::max()}; //!< [Nm] Maximum allowed motor torque
    T zeta{T{1}};                                //!< closed-loop damping ratio

    DcBusLimits<T>         bus_limits{}; //!< DC-bus limits
    CascadeBandwidths<T>   bandwidths{}; //!< loop bandwidths (rad/s)
    RotorObserverConfig<T> observer{};   //!< rotor observer tuning (J/b/Ts are taken from above)

    T Ts{T{1} / T{24000}}; //!< [s] electrical/estimator design rate (KF discretization); pass each loop's period to its update_*
};

/**
 * @brief Thin field-oriented PMAC servo: {Iabc, Vdc, θ} in, duties out.
 *
 * Orchestrates the library primitives without adding control math of its own:
 * a @ref FOController current loop, bandwidth-tuned velocity and position @ref
 * PIDController loops, a @ref RotorObserver for angle/speed feedback,
 * and a @ref DcBusLimiter plus an externally-supplied thermal derate for the current
 * ceiling. The whole cascade is tuned by three bandwidths (@ref CascadeBandwidths),
 * never raw gains: the current loop via @ref FOController::tune, the velocity loop
 * via @ref design::pi_pole_placement_first_order on the inertia plant `1/(Js+b)`, and
 * the position loop as a P gain on the integrator plant (Kp = ω). Torque-current
 * saturation (bus/thermal/ceiling) is enforced through the velocity loop's own
 * output limits, so its integrator anti-winds up; the current loop's voltage-circle
 * saturation is handled inside @ref FOController.
 *
 * Mode selects which loops run: Torque (current only), Velocity (+velocity), Position
 * (+position). Stream a trajectory by calling @ref set_target each tick.
 *
 * The torque command is mapped to a dq current reference by @ref design::mtpa_reference
 * (which collapses to `id_ref = 0` for a surface-PM machine, so the default is the
 * classic behaviour) and then passed through a pluggable @ref FieldWeakeningPolicy.
 * The default policy is @ref NoFieldWeakening; supply a @ref FieldWeakening instance to
 * weaken above base speed. The current loop is the PI @ref FOController.
 *
 * @see PmacServoConfig, ServoFeedback, FieldWeakeningPolicy.
 *
 * @tparam T  Scalar type.
 */
template<typename T = float>
class PmacServo {
    // using T = float;

public:
    constexpr PmacServo() = default;

    constexpr explicit PmacServo(const PmacServoConfig<T>& config)
        : config_(config),
          foc_(FOController<T>(config.Ldq, config.R, config.lambda, T{0})),
          bus_(config.bus_limits),
          Kt_(design::torque_constant_from_flux(config.pole_pairs, config.lambda)) {

        RotorObserverConfig<T> oc = config.observer;
        estimator_ = RotorObserver<T>(oc);

        tuned_ = tune(config.bandwidths, config.zeta);
    }

    /// True if the velocity loop has valid gains (J > 0). False ⇒ Torque mode only until retuned.
    [[nodiscard]] constexpr bool tuned() const { return tuned_; }

    /**
     * @brief Synthesize every loop's gains from three bandwidths (rad/s).
     *
     * The current and position loops tune unconditionally. The velocity loop needs the
     * inertia as its plant gain: @p config.J must be > 0, or its gains would come out zero
     * or negative (an unstable controller). A missing/zero J (no datasheet) is rejected and
     * the velocity gains are left untouched — run in Torque mode, or identify J first.
     *
     * @return true if every loop was tuned; false if J <= 0 (velocity loop left as-is).
     */
    [[nodiscard]] constexpr bool tune(const CascadeBandwidths<T>& bw, T zeta = T{1}) {
        foc_.tune(bw.omega_current, zeta, T{0});      // I-P current loop (no proportional step-kick)
        pos_ctrl = PController<T>{bw.omega_position}; // P on the 1/s position plant: bandwidth = Kp

        if (config_.J <= T{0}) {
            return false; // no inertia -> velocity loop has no plant gain; leave its gains untouched
        }
        // Velocity loop runs in turns/s but J/b are SI (rad-based): the plant 1/(Js+b) becomes
        // 1/((2π·J)s + 2π·b) when its output speed is in turns/s, so scale J and b by 2π.
        vel_ctrl = PIController<T>(
            design::pi_pole_placement_first_order(two_pi * config_.J, two_pi * config_.b, bw.omega_velocity, zeta)
        );
        return true;
    }

    constexpr void set_mode(ControlMode mode) {
        mode_ = mode;

        if (mode_ >= ControlMode::Position) {
            pos_ctrl.enable();
        }

        if (mode_ >= ControlMode::Velocity) {
            vel_ctrl.enable();
        }
    }

    constexpr void set_thermal_scale(T scale) { thermal_scale_ = scale; } //!< [0,1] derate from an external thermal model

    [[nodiscard]] constexpr T speed() const { return estimator_.omega(); }       //!< [turns/s] mechanical speed
    [[nodiscard]] constexpr T position() const { return estimator_.position(); } //!< [turns] continuous (multi-turn)

    /**
     * @brief Update commutation estimator with a new encoder reading (absolute)
     *
     * @param turns Absolute angle in turns [0 .. 1)
     * @param dt    Time since last update
     */
    constexpr void encoder_update_abs(const T turns, T dt) {
        enc_angle_ = wrap(turns, -half, half);
        estimator_.update(enc_angle_, dt);
    }

    /**
     * @brief Update commutation estimator with a new encoder reading (incremental turns)
     *
     * @param turns Number of terms since last encoder update
     * @param dt    Time since last update
     */
    constexpr void encoder_update_inc(const T turns, T dt) {
        enc_angle_ = wrap(enc_angle_ + turns, -half, half);
        estimator_.update(enc_angle_, dt);
    }

    /**
     * @brief Electrical loop — call from the PWM/ADC ISR (current-loop rate).
     *
     * Clarke/Park the measured phases on the tracker's own predicted rotor angle, run the
     * FOController current loop and SVPWM, then advance the kinematic tracker one step
     * (predict last, so the angle is always valid for the upcoming tick). Consumes the
     * q-current reference last produced by @ref update_velocity, plus any torque
     * feedforward; the feedback needed by the slower loops (Idq, Vdc) is latched here.
     *
     * @param dt this loop's period [s]. Should match the estimator's design rate
     *           (@ref PmacServoConfig::Ts), since the KF predict is discretized at it.
     * @return @ref FocResult with the SVPWM duties and saturation status.
     */
    [[nodiscard]] constexpr FocResult<T> current_control_step(const DirectQuadrature<T>& Idq_ref, const ColVec<3, T>& Iabc, T Vdc, T dt) {
        // Mechanical [turns] -> Electrical [rad] conversion (the one turns->rad hop, for Park).
        const T theta_elec = wrap(config_.pole_pairs * estimator_.theta() * two_pi, -pi, pi);
        foc_.omega = config_.pole_pairs * estimator_.omega() * two_pi;

        // Convert from raw 3ph current measurements to DQ frame
        Idq_ = clarke_park_transform(Iabc, theta_elec);
        Vdc_ = Vdc;

        // Run standard dual PI FOC with plant cancellation --> SVM Duty control
        const auto Vmax = foc_.max_modulation * Vdc * wet::numbers::inv_sqrt3_v<T>;
        const auto cmd = foc_.current_controller(Idq_ref, Idq_, dt, Vmax);
        const auto Vab = inverse_park_transform(cmd.Vdq, theta_elec);
        const auto svm = svm_duty_cycles(Vab, Vdc);

        // Extrapolate θ by ω·Ts for the next tick
        estimator_.predict(dt);

        prev_vdq_ = cmd.Vdq; // bus power uses the applied voltage on the next evaluate

        return FocResult<T>{
            .duties = svm.duties,
            .Idq = Idq_,
            .v_saturated = cmd.is_saturated,
            .svm_clipped = svm.is_clipped,
            .v_excess = cmd.v_excess,
        };
    }

    /**
     * @brief Current-command summing junction — torque feedforward, limit, anti-windup.
     *
     * In Velocity/Position mode @p torque_target is the current/torque feedforward summed onto the velocity loop's
     * torque command; the sum is clamped to the bus/thermal torque ceiling and the clipped
     * excess is back-calculated into the velocity loop so its integrator anti-winds. In Torque
     * mode the velocity loop is idle and @p torque_target is the direct torque command, clamped
     * to the same ceiling.
     *
     * @param torque_target [Nm] Torque target / feedforward term
     * @param dt            [sec] Time since last torque control update
     */
    constexpr void torque_control_step(T torque_target, T dt) {
        if (mode_ > ControlMode::Torque) {
            const T torque_lim = iq_lim_ * Kt_; // ceiling latched by velocity_control_step
            const T torque_sat = wet::clamp(torque_cmd_ + torque_target, -torque_lim, torque_lim);

            // Back-calculate the velocity loop for the headroom the feedforward consumed.
            vel_ctrl.back_calculate(torque_cmd_, torque_sat - torque_target, dt);
            torque_cmd_ = torque_sat;
        } else {
            const auto torque_lim = iq_lim_ * Kt_;
            torque_cmd_ = wet::clamp(torque_target, -torque_lim, torque_lim);
        }

        // Compute a DQ current reference from torque target and torque constant
        // TODO:  Add MTPA and motor model to support other motor types (currently assumes surface mount PMAC)
        Idq_ref_ = {.d = T{0}, .q = torque_cmd_ / Kt_};
    }

    /**
     * @brief Velocity controller tracking mode and output limiting
     *
     * @param vel_target [turn/sec] Velocity controller target / feedforward
     * @param dt         [sec] Time since last velocity control update
     */
    constexpr void velocity_control_step(T vel_target, T dt) {
        if (mode_ >= ControlMode::Velocity) {
            // Velocity feedforward sums onto the position loop's held velocity command, limited to the ceiling.
            const T vel_cmd = wet::clamp(pos_vel_cmd_ + vel_target, -config_.vel_max, config_.vel_max);
            torque_cmd_ = vel_ctrl.control(vel_cmd, estimator_.omega(), dt);
        } else {
            // Hold the velocity loop in tracking mode following the applied torque, so a
            // switch back to Velocity/Position resumes from the right integrator with no bump.
            const T omega = estimator_.omega();
            vel_ctrl.track(torque_cmd_, omega, dt);
        }
    }

    /**
     * @brief Position loop — call from the slow preemptive task.
     *
     * P on the integrator plant → a velocity command consumed by @ref update_velocity.
     * No-op outside Position mode.
     *
     * @param pos_target [turn] Desired (unrolled) servo position
     */
    constexpr void position_control_step(T pos_target) {
        if (mode_ == ControlMode::Position) {
            pos_vel_cmd_ = pos_ctrl.control(pos_target, estimator_.position());
        } else {
            pos_vel_cmd_ = T{0};
        }
    }

    /**
     * @brief Calculates Idq limit from bus limits, then propagates up the chain
     */
    constexpr void recalculate_limits() {
        const auto bus_state = bus_.evaluate(prev_vdq_, Idq_, Vdc_);
        iq_lim_ = config_.iq_max * wet::max(wet::min(bus_state.scale, thermal_scale_), T{0});

        torque_lim_ = wet::min(iq_lim_ * Kt_, config_.torque_max);
        vel_ctrl.u_min = -torque_lim_;
        vel_ctrl.u_max = torque_lim_;
    }

    /**
     * @brief Run a servo control update at a single rate
     *
     * @param pos_target        [turn] Position target
     * @param vel_target        [turn/sec] Velocity loop target / feedforward
     * @param torque_target     [Nm] Torque target / feedforward
     * @param Iabc              [A] Individual phase current measurements
     * @param Vdc               [V] DC Bus voltage measurement
     * @param encoder_meas      [turn] Raw encoder measurement in turns [0 .. 1)
     *
     * @return Half-bridge duty cycles
     */
    [[nodiscard]] constexpr FocResult<T> update(T pos_target, T vel_target, T torque_target, const ColVec<3, T>& Iabc, T Vdc, T encoder_meas) {
        encoder_update_abs(encoder_meas, config_.Ts);

        recalculate_limits();
        position_control_step(pos_target);
        velocity_control_step(vel_target, config_.Ts);
        torque_control_step(torque_target, config_.Ts);

        return current_control_step(Idq_ref_, Iabc, Vdc, config_.Ts);
    }

    constexpr void reset() {
        foc_.reset();
        vel_ctrl.reset();
        pos_ctrl.reset();
        estimator_.reset();
        prev_vdq_ = {};
        Idq_ = {};
        pos_vel_cmd_ = T{0};
        enc_angle_ = T{0};
    }

    DirectQuadrature<T> Idq_ref_ = {}; //!< DQ current reference after torque controller / MTPA

private:
    static constexpr T pi = wet::numbers::pi_v<T>;
    static constexpr T two_pi = T{2} * wet::numbers::pi_v<T>;
    static constexpr T half = T{0.5}; // wrap half-range for the turns-based encoder angle

    PmacServoConfig<T> config_{};

    PController<T>   pos_ctrl;   // Position controller
    PIController<T>  vel_ctrl;   // Velocity controller
    FOController<T>  foc_;       // Current Controller
    RotorObserver<T> estimator_; // Estimates rotor theta and omega (commutation/speed)
    DcBusLimiter<T>  bus_;

    ControlMode mode_{ControlMode::Torque};
    bool        tuned_{false}; //!< velocity loop has valid gains (J > 0)

    T thermal_scale_{T{1}};
    T Kt_{T{1}}; //!< Torque Constant

    // Cross-rate state: written by the fast loop, read by the slower ones (and back).
    DirectQuadrature<T> prev_vdq_ = {}; //!< latched Vdq being applied
    DirectQuadrature<T> Idq_ = {};      //!< latched measured dq current (ISR → velocity)

    T Vdc_ = T{0};       //!< latched bus voltage (ISR → velocity)
    T enc_angle_ = T{0}; //!< [turns] wrapped mechanical angle accumulated from encoder feedback
    T iq_lim_ = T{0};    //!< [A] Current ceiling held by the velocity loop

    T pos_vel_cmd_ = T{0}; //!< [turn/sec] Velocity command held by the position loop (position → velocity)

    T torque_cmd_ = T{0};                          //!< [Nm] Torque command (velocity → torque)
    T torque_lim_ = std::numeric_limits<T>::max(); //!< [Nm] Runtime calculated torque limit
};

} // namespace wet::motor
