#pragma once

#include <cstdint>
#include <limits>

#include "wet/controllers/pid.hpp"
#include "wet/design/pid_design.hpp" // pi_pole_placement_first_order
#include "wet/matrix/colvec.hpp"
#include "wet/motor/foc.hpp"
#include "wet/motor/limits.hpp"
#include "wet/motor/mechanical_estimator.hpp"
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

    DcBusLimits<T>               bus{};        //!< DC-bus limits
    CascadeBandwidths<T>         bandwidths{}; //!< loop bandwidths (rad/s)
    MechanicalEstimatorConfig<T> estimator{};  //!< estimator tuning (J/b/Kt/Ts are taken from above)

    T Ts{T{1} / T{24000}}; //!< [s] electrical/estimator design rate (KF discretization); pass each loop's period to its update_*
};

/**
 * @brief Sensor feedback for one @ref PmacServo::update tick.
 * @tparam T Scalar type.
 */
template<typename T = float>
struct ServoFeedback {
    ColVec<3, T> Iabc{};       //!< [A] measured phase currents
    T            Vdc{};        //!< [V] DC bus voltage
    T            theta_mech{}; //!< [rad] measured mechanical angle (continuous/unwrapped)
};

/**
 * @brief Thin field-oriented PMAC servo: {Iabc, Vdc, θ} in, duties out.
 *
 * Orchestrates the library primitives without adding control math of its own:
 * a @ref FOController current loop, bandwidth-tuned velocity and position @ref
 * PIDController loops, a @ref MechanicalEstimator for speed/position/load feedback,
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
 * @note `id_ref = 0` (surface-PM, max torque-per-amp); field weakening is future work.
 *
 * @see PmacServoConfig, ServoFeedback.
 *
 * @tparam T Scalar type.
 */
template<typename T = float>
class PmacServo {
public:
    PmacServo() = default;

    explicit PmacServo(const PmacServoConfig<T>& config)
        : config_(config),
          foc_(FOController<T>(config.Ldq, config.R, config.lambda, T{0})),
          bus_(config.bus),
          Kt_(design::torque_constant_from_flux(config.pole_pairs, config.lambda)) {

        MechanicalEstimatorConfig<T> ec = config.estimator;
        ec.J = config.J;
        ec.b = config.b;
        ec.Kt = Kt_;
        ec.Ts = config.Ts;
        estimator_ = MechanicalEstimator<T>(ec);

        tune(config.bandwidths, config.zeta);
    }

    /// Synthesize every loop's gains from three bandwidths (rad/s).
    constexpr void tune(const CascadeBandwidths<T>& bw, T zeta = T{1}) {
        foc_.tune(bw.omega_current, zeta, T{0}); // I-P current loop (no proportional step-kick)
        vel_pid_ = PIDController<T, PIDMode::PID>(design::pi_pole_placement_first_order(config_.J, config_.b, bw.omega_velocity, zeta));
        pos_pid_ = PIDController<T, PIDMode::PID>{};
        pos_pid_.Kp = bw.omega_position; // P on the 1/s position plant: bandwidth = Kp
    }

    constexpr void set_mode(ControlMode mode) { mode_ = mode; }
    constexpr void set_target(T target) { target_ = target; }             //!< torque[Nm]/speed[rad/s]/angle[rad] per mode
    constexpr void set_thermal_scale(T scale) { thermal_scale_ = scale; } //!< [0,1] derate from an external thermal model

    [[nodiscard]] constexpr T speed() const { return estimator_.omega(); }
    [[nodiscard]] constexpr T position() const { return estimator_.theta(); }
    [[nodiscard]] constexpr T load_torque() const { return estimator_.load_torque(); }

    /**
     * @brief Electrical loop — call from the PWM/ADC ISR (current-loop rate).
     *
     * Clarke/Park the measured phases, KF predict from the measured iq (for a fresh
     * speed feedforward), then run the FOController current loop and SVPWM. Consumes
     * the q-current reference last produced by @ref update_velocity; the feedback
     * needed by the slower loops (Idq, Vdc, θ) is latched here for them to read.
     *
     * @param dt this loop's period [s]. Should match the estimator's design rate
     *           (@ref PmacServoConfig::Ts), since the KF predict is discretized at it.
     * @return @ref FocResult with the SVPWM duties and saturation status.
     */
    [[nodiscard]] FocResult<T> update_current(const ServoFeedback<T>& fb, T dt) {
        const T theta_elec = config_.pole_pairs * fb.theta_mech;
        Idq_ = clarke_park_transform(fb.Iabc, theta_elec);
        Vdc_ = fb.Vdc;
        theta_mech_ = fb.theta_mech; // latched for update_velocity's encoder correction

        estimator_.predict(Idq_.q);
        foc_.omega = config_.pole_pairs * estimator_.omega(); // electrical speed feedforward

        const T                   Vmax = foc_.max_modulation * fb.Vdc * wet::numbers::inv_sqrt3_v<T>;
        const DirectQuadrature<T> Idq_ref{.d = T{0}, .q = iq_ref_};
        const auto                cmd = foc_.current_controller(Idq_ref, Idq_, dt, Vmax);
        const auto                Vab = inverse_park_transform(cmd.Vdq, theta_elec);
        const auto                svm = svm_duty_cycles(Vab, fb.Vdc);

        prev_vdq_ = cmd.Vdq; // bus power uses the applied voltage on the next evaluate

        FocResult<T> result;
        result.duties = svm.duties;
        result.Idq = Idq_;
        result.v_saturated = cmd.is_saturated;
        result.v_excess = cmd.v_excess;
        result.svm_clipped = svm.is_clipped;
        return result;
    }

    /**
     * @brief Velocity loop — call from the medium-rate async task.
     *
     * Fuse the latched encoder angle, recompute the bus/thermal current ceiling, and
     * run the velocity PI (or, in Torque mode, just clamp the command) to produce the
     * q-current reference for @ref update_current. The bus derate lives here, not in
     * the ISR: it only sets this loop's output clamp (so the integrator anti-winds),
     * and its dynamics are far slower than the current loop. In Position mode this
     * tracks the velocity command from @ref update_position.
     *
     * @param dt this task's period [s].
     */
    void update_velocity(T dt) {
        estimator_.update_encoder(theta_mech_);
        const T omega_mech = estimator_.omega();

        // Effective q-current ceiling: configured limit derated by bus and thermal.
        const auto bus_state = bus_.evaluate(prev_vdq_, Idq_, Vdc_);
        iq_lim_ = config_.iq_max * wet::min(bus_state.scale, thermal_scale_);

        const T torque_lim = iq_lim_ * Kt_;
        vel_pid_.u_min = -torque_lim;
        vel_pid_.u_max = torque_lim;

        switch (mode_) {
            case ControlMode::Position:
                iq_ref_ = vel_pid_.control(omega_cmd_, omega_mech, dt) / Kt_;
                break;
            case ControlMode::Velocity:
                iq_ref_ = vel_pid_.control(target_, omega_mech, dt) / Kt_;
                break;
            case ControlMode::Torque:
                iq_ref_ = wet::clamp(target_ / Kt_, -iq_lim_, iq_lim_);
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
    void update_position(T dt) {
        if (mode_ == ControlMode::Position) {
            omega_cmd_ = pos_pid_.control(target_, estimator_.theta(), dt);
        }
    }

    /// Single-rate convenience (tests / a single-task port): run the full cascade at Ts.
    [[nodiscard]] FocResult<T> update(const ServoFeedback<T>& fb) {
        update_position(config_.Ts);
        update_velocity(config_.Ts);
        return update_current(fb, config_.Ts);
    }

    constexpr void reset() {
        foc_.reset();
        vel_pid_.reset();
        pos_pid_.reset();
        estimator_.reset();
        prev_vdq_ = {};
        Idq_ = {};
        iq_ref_ = T{0};
        omega_cmd_ = T{0};
    }

private:
    PmacServoConfig<T>             config_{};
    FOController<T>                foc_{};
    PIDController<T, PIDMode::PID> vel_pid_{};
    PIDController<T, PIDMode::PID> pos_pid_{};
    MechanicalEstimator<T>         estimator_{};
    DcBusLimiter<T>                bus_{};

    ControlMode mode_{ControlMode::Torque};

    T                   target_{T{0}};
    T                   thermal_scale_{T{1}};
    T                   Kt_{T{1}};
    DirectQuadrature<T> prev_vdq_{};

    // Cross-rate state: written by the fast loop, read by the slower ones (and back).
    DirectQuadrature<T> Idq_{};           //!< latched measured dq current (ISR → velocity)
    T                   Vdc_{T{0}};       //!< latched bus voltage (ISR → velocity)
    T                   theta_mech_{};    //!< latched encoder angle (ISR → velocity)
    T                   iq_ref_{T{0}};    //!< q-current reference (velocity → ISR)
    T                   iq_lim_{T{0}};    //!< current ceiling held by the velocity loop
    T                   omega_cmd_{T{0}}; //!< velocity command (position → velocity)
};

} // namespace wet::motor
