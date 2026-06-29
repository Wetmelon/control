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
    Velocity, //!< command is speed [rad/s]; velocity + current loops
    Position, //!< command is angle [rad]; position + velocity + current loops
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

    T iq_max{std::numeric_limits<T>::max()}; //!< [A] q-axis current ceiling
    T zeta{T{1}};                            //!< closed-loop damping

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
public:
    constexpr PmacServo() = default;

    constexpr explicit PmacServo(const PmacServoConfig<T>& config)
        : config_(config),
          foc_(FOController<T>(config.Ldq, config.R, config.lambda, T{0})),
          bus_(config.bus_limits),
          Kt_(design::torque_constant_from_flux(config.pole_pairs, config.lambda)) {

        RotorObserverConfig<T> oc = config.observer;
        estimator_ = RotorObserver<T>(oc);

        tune(config.bandwidths, config.zeta);
    }

    /// Synthesize every loop's gains from three bandwidths (rad/s).
    constexpr void tune(const CascadeBandwidths<T>& bw, T zeta = T{1}) {
        foc_.tune(bw.omega_current, zeta, T{0}); // I-P current loop (no proportional step-kick)
        vel_pid_ = PIController<T>(design::pi_pole_placement_first_order(config_.J, config_.b, bw.omega_velocity, zeta));
        pos_pid_ = PController<T>{bw.omega_position}; // P on the 1/s position plant: bandwidth = Kp
    }

    constexpr void set_mode(ControlMode mode) { mode_ = mode; }
    constexpr void set_target(T target) { target_ = target; }             //!< torque[Nm]/speed[rad/s]/angle[rad] per mode
    constexpr void set_thermal_scale(T scale) { thermal_scale_ = scale; } //!< [0,1] derate from an external thermal model

    [[nodiscard]] constexpr T speed() const { return estimator_.omega(); }
    [[nodiscard]] constexpr T position() const { return estimator_.theta_unwrapped(); } //!< [rad] continuous (multi-turn)

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
    [[nodiscard]] constexpr FocResult<T> current_control_step(float torque_ref, const ColVec<3, T>& Iabc, T Vdc, T dt) {

        // Mechanical -> Electrical angle / omega conversion
        const T theta_elec = wrap(config_.pole_pairs * estimator_.theta(), -pi, pi);
        foc_.omega = config_.pole_pairs * estimator_.omega();

        // Convert from raw 3ph current measurements to DQ frame
        Idq_ = clarke_park_transform(Iabc, theta_elec);
        Vdc_ = Vdc;

        // Compute a DQ current reference from torque target and torque constant
        // TODO:  Add MTPA and motor model to support other motor types (currently assumes surface mount PMAC)
        const DirectQuadrature<T> Idq_ref = {.d = T{0}, .q = torque_ref / Kt_};

        const auto Vmax = foc_.max_modulation * Vdc * wet::numbers::inv_sqrt3_v<T>;
        const auto cmd = foc_.current_controller(Idq_ref, Idq_, dt, Vmax);
        const auto Vab = inverse_park_transform(cmd.Vdq, theta_elec);
        const auto svm = svm_duty_cycles(Vab, Vdc);

        prev_vdq_ = cmd.Vdq;    // bus power uses the applied voltage on the next evaluate
        estimator_.predict(dt); // kinematic tracker: extrapolate θ by ω·Ts for the next tick

        return FocResult<T>{
            .duties = svm.duties,
            .Idq = Idq_,
            .v_saturated = cmd.is_saturated,
            .svm_clipped = svm.is_clipped,
            .v_excess = cmd.v_excess,
        };
    }

    /**
     * @brief Encoder correction — call whenever a rotor-angle sample is available.
     *
     * Decoupled from the control loops: encoders may sample slower than the current
     * loop, so the correction runs at its own rate. Wraps/unrolls the @ref
     * EncoderFeedback to the wrapped mechanical angle and fuses it into the estimator.
     */
    constexpr void update_encoder(const T turns, T dt) {
        enc_angle_ = wrap(turns * two_pi, -pi, pi);
        estimator_.update(enc_angle_, dt);
    }

    /**
     * @brief Velocity loop — call from the medium-rate async task.
     *
     * Recompute the bus/thermal current ceiling and run the velocity PI (or, in Torque
     * mode, just clamp the command) to produce the q-current reference for @ref
     * update_current. The bus derate lives here, not in the ISR: it only sets this
     * loop's output clamp (so the integrator anti-winds), and its dynamics are far
     * slower than the current loop. In Position mode this tracks the velocity command
     * from @ref update_position.
     *
     * @param dt this task's period [s].
     */
    constexpr void update_velocity(T dt) {
        const T omega_mech = estimator_.omega();

        // Effective q-current ceiling: configured limit derated by bus and thermal.
        const auto bus_state = bus_.evaluate(prev_vdq_, Idq_, Vdc_);
        iq_lim_ = config_.iq_max * wet::min(bus_state.scale, thermal_scale_);

        const T torque_lim = iq_lim_ * Kt_;
        vel_pid_.u_min = -torque_lim;
        vel_pid_.u_max = torque_lim;

        // The velocity PI is clamped in torque units; its output is the torque command,
        // which update_current maps to (id,iq) via MTPA (not a flat /Kt) so salient
        // machines get the reluctance contribution right.
        switch (mode_) {
            case ControlMode::Position:
                torque_ref_ = vel_pid_.control(omega_cmd_, omega_mech, dt);
                break;
            case ControlMode::Velocity:
                torque_ref_ = vel_pid_.control(target_, omega_mech, dt);
                break;
            case ControlMode::Torque:
                torque_ref_ = wet::clamp(target_, -torque_lim, torque_lim);
                break;
        }
    }

    /**
     * @brief Position loop — call from the slow preemptive task.
     *
     * P on the integrator plant → a velocity command consumed by @ref update_velocity.
     * No-op outside Position mode.
     *
     * @param dt this task's period [s].
     */
    constexpr void update_position(T dt) {
        if (mode_ == ControlMode::Position) {
            omega_cmd_ = pos_pid_.control(target_, estimator_.theta_unwrapped(), dt);
        }
    }

    /// Single-rate convenience (tests / a single-task port): run the full cascade at Ts.
    [[nodiscard]] constexpr FocResult<T> update(const ColVec<3, T>& Iabc, T Vdc, T delta_turns) {
        update_encoder(delta_turns, config_.Ts);
        update_position(config_.Ts);
        update_velocity(config_.Ts);

        return current_control_step(torque_ref_, Iabc, Vdc, config_.Ts);
    }

    constexpr void reset() {
        foc_.reset();
        vel_pid_.reset();
        pos_pid_.reset();
        estimator_.reset();
        prev_vdq_ = {};
        Idq_ = {};
        torque_ref_ = T{0};
        omega_cmd_ = T{0};
        enc_angle_ = T{0};
    }

private:
    static constexpr T pi = wet::numbers::pi_v<T>;
    static constexpr T two_pi = T{2} * wet::numbers::pi_v<T>;

    PmacServoConfig<T> config_{};

    PController<T>  pos_pid_{}; // Position controller
    PIController<T> vel_pid_{}; // Velocity controller
    FOController<T> foc_{};     // Current Controller

    RotorObserver<T> estimator_{}; // Estimates rotor theta and omega (commutation/speed)

    DcBusLimiter<T> bus_{};

    ControlMode mode_{ControlMode::Torque};

    T target_{T{0}};
    T thermal_scale_{T{1}};
    T Kt_{T{1}}; //!< Torque Constant

    // Cross-rate state: written by the fast loop, read by the slower ones (and back).
    DirectQuadrature<T> prev_vdq_{};
    DirectQuadrature<T> Idq_{}; //!< latched measured dq current (ISR → velocity)

    T Vdc_{T{0}};        //!< latched bus voltage (ISR → velocity)
    T enc_angle_{T{0}};  //!< wrapped mechanical angle accumulated from encoder feedback
    T torque_ref_{T{0}}; //!< torque reference (velocity → ISR), mapped to (id,iq) by MTPA
    T iq_lim_{T{0}};     //!< current ceiling held by the velocity loop
    T omega_cmd_{T{0}};  //!< velocity command (position → velocity)
};

} // namespace wet::motor
