# API Reference — Alphabetical Index

Auto-generated from `@brief` doc comments in `inc/wet/`. Regenerate with `python tools/gen_reference.py`. Grouped-by-domain view: [REFERENCE.md](REFERENCE.md).

| Name | Kind | Domain | Description |
| ---- | ---- | ------ | ----------- |
| [`abs`](inc/wet/math/complex.hpp#L353) | function | Scalar math & complex | Compute magnitude (absolute value) of a complex number |
| [`acker`](inc/wet/matlab.hpp#L408) | function | MATLAB-style aliases (host) | Pole placement for state-feedback control |
| [`ackermann`](inc/wet/design/pole_placement.hpp#L1146) | function | Design & synthesis | Single-input pole placement via Ackermann's formula |
| [`acos`](inc/wet/math/math.hpp#L136) | function | Scalar math & complex | Arccosine ∈ [0, π]. Input is clamped to [−1, 1] in both paths |
| [`AdaptivePredictiveCurrentController`](inc/wet/motor/predictive_current.hpp#L213) | block | Motor control | Self-tuning deadbeat current controller: PredictiveCurrentController plus the online PmsmParameterEstimator |
| [`AdaptiveStepSolver`](inc/wet/simulation/solver.hpp#L209) | block | Simulation (host) | Adaptive-step ODE solver |
| [`adrc`](inc/wet/controllers/adrc.hpp#L39) | function | Runtime controllers | Active Disturbance Rejection Control design |
| [`ADRCController`](inc/wet/controllers/adrc.hpp#L89) | block | Runtime controllers | Active Disturbance Rejection Control (ADRC) |
| [`ADRCResult`](inc/wet/controllers/adrc.hpp#L13) | block | Runtime controllers | Active Disturbance Rejection Control design result |
| [`AffineCal`](inc/wet/toolbox/scaling.hpp#L62) | block | Utilities & toolbox | Affine sensor calibration `y = gain·x + offset` |
| [`AlphaBeta`](inc/wet/transforms.hpp#L130) | block | Motor control | Alpha-beta (stationary-frame) component pair |
| [`AlphaBetaZero`](inc/wet/transforms.hpp#L195) | block | Motor control | Alpha-beta-zero (stationary-frame) component triple |
| [`AnalogInput`](inc/wet/toolbox/io.hpp#L23) | block | Utilities & toolbox | A single analog input: range/fault check on the raw reading, then affine calibration to engineering units |
| [`arg`](inc/wet/math/complex.hpp#L364) | function | Scalar math & complex | Compute argument (phase angle) of a complex number |
| [`arm_spherical_wrist`](inc/wet/kinematics/serial_arm.hpp#L466) | function | Kinematics | Tier-2 builder for a standard 6R elbow arm with a spherical wrist |
| [`asin`](inc/wet/math/math.hpp#L115) | function | Scalar math & complex | Arcsine ∈ [−π/2, π/2]. Input is clamped to [−1, 1] in both paths so behavior matches at compile and run time (std::asin would return NaN for \|x\| > 1) |
| [`atan`](inc/wet/math/math.hpp#L103) | function | Scalar math & complex | Single-argument arctangent ∈ (−π/2, π/2) |
| [`atan2`](inc/wet/math/math.hpp#L90) | function | Scalar math & complex | Two-argument arctangent, atan2(y, x) ∈ [−π, π] |
| [`AxisInput`](inc/wet/toolbox/io.hpp#L66) | block | Utilities & toolbox | Operator-axis conditioning chain (joystick / RC stick → command) |
| [`BackwardEuler`](inc/wet/simulation/integrator.hpp#L119) | block | Simulation (host) | Backward Euler integrator |
| [`bandpass`](inc/wet/filters/filters.hpp#L480) | function | Filters & signal conditioning | Second-order band-pass filter (constant 0 dB peak gain) |
| [`bandwidth`](inc/wet/matlab.hpp#L692) | function | MATLAB-style aliases (host) | -3 dB bandwidth of a SISO system over a frequency grid |
| [`bandwidth_from_settling_time`](inc/wet/design/pid_design.hpp#L457) | function | Design & synthesis | Map settling-time and damping-ratio targets to a bandwidth estimate |
| [`base_speed`](inc/wet/motor/foc.hpp#L215) | function | Motor control | Base (corner) electrical speed where the voltage circle is first hit |
| [`BDF2`](inc/wet/simulation/integrator.hpp#L182) | block | Simulation (host) | Backward Differentiation Formula 2 (BDF2) integrator |
| [`beta`](inc/wet/toolbox/thermistor.hpp#L50) | function | Utilities & toolbox | Fit NTC coefficients from the Beta-parameter model |
| [`Biquad`](inc/wet/filters/filters.hpp#L720) | block | Filters & signal conditioning | Second-order IIR (biquad) section runtime |
| [`BiquadCascade`](inc/wet/filters/filters.hpp#L792) | block | Filters & signal conditioning | Cascade of second-order sections (SOS) for higher-order IIR filters |
| [`BLINK`](inc/wet/toolbox/iec61131.hpp#L530) | block | Utilities & toolbox | BLINK (free-running square-wave / flasher) |
| [`blkdiag`](inc/wet/matlab.hpp#L205) | function | MATLAB-style aliases (host) | Block diagonal matrix construction |
| [`Block`](inc/wet/matrix/block.hpp#L17) | block | Linear algebra | Block view (non-owning) into a parent matrix |
| [`bode`](inc/wet/analysis/analysis.hpp#L288) | function | Frequency-domain analysis (host) | Compute Bode plot data for a SISO state-space system |
| [`bode_discrete`](inc/wet/analysis/analysis.hpp#L383) | function | Frequency-domain analysis (host) | Compute Bode plot data for a discrete-time SISO state-space system |
| [`bodemag`](inc/wet/simulation/plot_plotly.hpp#L392) | function | Simulation (host) | Plot a magnitude-only Bode diagram (log frequency, dB magnitude) |
| [`bodeplot`](inc/wet/simulation/plot_plotly.hpp#L378) | function | Simulation (host) | Plot magnitude and phase Bode subplots |
| [`Bounds`](inc/wet/toolbox/bounds.hpp#L32) | block | Utilities & toolbox | A per-channel closed-interval box constraint |
| [`Button`](inc/wet/toolbox/io.hpp#L117) | block | Utilities & toolbox | Debounced momentary push-button with edge and long-press detection |
| [`c2d`](inc/wet/matlab.hpp#L153) | function | MATLAB-style aliases (host) | Matlab interface function c2d to discretize a continuous-time state-space system |
| [`canonical_phase_margin`](inc/wet/analysis/analysis.hpp#L200) | function | Frequency-domain analysis (host) | Normalize phase margin to (-180, 180] |
| [`care`](inc/wet/design/riccati.hpp#L701) | function | Design & synthesis | Solve the Continuous-time Algebraic Riccati Equation (CARE) |
| [`care_schur`](inc/wet/design/riccati.hpp#L492) | function | Design & synthesis | Solve CARE via the ordered real-Schur method (Laub's method) |
| [`CartesianMap`](inc/wet/kinematics/motion_maps.hpp#L31) | block | Kinematics | Cartesian gantry: independent per-axis affine map `task = scale·act + offset` (the "kinematics" is the identity, exposed for a uniform forward/inverse interface) |
| [`CartesianMove`](inc/wet/trajectory/cartesian_move.hpp#L97) | block | Trajectory & motion planning | Path-preserving task-space move (Pipeline B / LIN) |
| [`CascadeBandwidths`](inc/wet/motor/servo.hpp#L23) | block | Motor control | The three bandwidth knobs of the position/velocity/current cascade |
| [`cauer_thermal_ss`](inc/wet/motor/thermal.hpp#L66) | function | Motor control | Continuous state-space model of a physical Cauer RC thermal ladder |
| [`cbrt`](inc/wet/math/math.hpp#L78) | function | Scalar math & complex | Cube root (preserves sign for negative x) |
| [`ceil`](inc/wet/math/math.hpp#L304) | function | Scalar math & complex | Ceiling — smallest integer ≥ x |
| [`Chirp`](inc/wet/estimation/excitation.hpp#L611) | block | Observers & estimators | Linear or logarithmic chirp runtime generator |
| [`ChirpConfig`](inc/wet/estimation/excitation.hpp#L159) | block | Observers & estimators | Configuration for a sine chirp excitation |
| [`ChirpResult`](inc/wet/estimation/excitation.hpp#L201) | block | Observers & estimators | Chirp design payload |
| [`cholesky`](inc/wet/matrix/decomposition.hpp#L59) | function | Linear algebra | Cholesky decomposition for positive-definite matrices |
| [`clarke_park_transform`](inc/wet/transforms.hpp#L405) | function | Motor control | Fused Clarke-Park transform (abc → dq) |
| [`clarke_park_zero_transform`](inc/wet/transforms.hpp#L494) | function | Motor control | Fused Clarke-Park transform with zero (abc → dq0) |
| [`clarke_transform`](inc/wet/transforms.hpp#L314) | function | Motor control | Clarke transform (abc → αβ) |
| [`clarke_zero_transform`](inc/wet/transforms.hpp#L235) | function | Motor control | Zero-retaining Clarke transform (abc → αβ0) |
| [`ClassicalDisturbanceObserver`](inc/wet/estimation/disturbance_observer.hpp#L357) | block | Observers & estimators | Classical Pn^-1·Q disturbance observer runtime (bolt-on compensator) |
| [`ClassicalDobResult`](inc/wet/estimation/disturbance_observer.hpp#L284) | block | Observers & estimators | Design result for the classical Pn^-1·Q disturbance observer |
| [`classify_range`](inc/wet/toolbox/conditioning.hpp#L242) | function | Utilities & toolbox | Classify x against the four band edges `[fault_lo (valid_lo, valid_hi) fault_hi]` (assumed ordered, non-decreasing) |
| [`closed_loop_poles`](inc/wet/design/stability.hpp#L303) | function | Design & synthesis | Compute closed-loop poles (eigenvalues) with state feedback |
| [`cohen_coon`](inc/wet/design/pid_design.hpp#L184) | function | Design & synthesis | Cohen-Coon tuning from first-order-plus-dead-time model |
| [`ColVec`](inc/wet/matrix/colvec.hpp#L9) | block | Linear algebra | Concrete Column vector specialization of Matrix<N, 1, T> |
| [`ColView`](inc/wet/matrix/views.hpp#L215) | block | Linear algebra | Non-owning column view of a matrix |
| [`comb_notch_window`](inc/wet/filters/filters.hpp#L890) | function | Filters & signal conditioning | Window length for a moving-average comb that notches f_notch and all its harmonics: N = round(fs / f_notch) |
| [`Complementary`](inc/wet/filters/filters.hpp#L1095) | block | Filters & signal conditioning | Scalar (1-D) complementary filter — fuse a fast rate with a slow absolute |
| [`ComplementaryFilter`](inc/wet/estimation/sensor_fusion.hpp#L41) | block | Observers & estimators | Simple complementary filter for orientation estimation |
| [`complex`](inc/wet/math/complex.hpp#L18) | block | Scalar math & complex | Constexpr complex number class for compile-time computations |
| [`complex_scatter`](inc/wet/simulation/plot_plotly.hpp#L280) | function | Simulation (host) | Build a markers-only scatter of complex points (real vs imaginary) |
| [`compute_eigenvalues`](inc/wet/matrix/eigen.hpp#L340) | function | Linear algebra | Compute the eigenvalues (and Schur vectors) of a real square matrix |
| [`ConstantInertiaFeedforward`](inc/wet/toolbox/actuator.hpp#L164) | block | Utilities & toolbox | Per-axis decoupled torque feedforward: `τ = J·a + b·v + τ_c·sign(v) + g` |
| [`continuous_lqr`](inc/wet/controllers/lqr.hpp#L265) | function | Runtime controllers | Continuous-time Linear-Quadratic Regulator design |
| [`controllability_gramian`](inc/wet/design/stability.hpp#L90) | function | Design & synthesis | Continuous/discrete controllability Gramian @f$ W_c @f$ |
| [`controllability_matrix`](inc/wet/design/stability.hpp#L30) | function | Design & synthesis | Compute the controllability matrix [B, AB, A²B, ..., A^(N-1)B] |
| [`Convention`](inc/wet/transforms.hpp#L43) | block | Motor control | Scaling convention for the Clarke/Park family |
| [`copysign`](inc/wet/math/math.hpp#L375) | function | Scalar math & complex | Copy sign — magnitude of mag with the sign of sgn_src |
| [`CoreXY`](inc/wet/kinematics/motion_maps.hpp#L59) | block | Kinematics | CoreXY belt mapping (2 motors A/B → Cartesian X/Y) |
| [`cos`](inc/wet/matrix/functions.hpp#L662) | function | Linear algebra | Matrix cosine via scaling and double-angle reconstruction |
| [`cos`](inc/wet/math/math.hpp#L154) | function | Scalar math & complex | Cosine |
| [`cosh`](inc/wet/matrix/functions.hpp#L732) | function | Linear algebra | Matrix hyperbolic cosine |
| [`Counter`](inc/wet/toolbox/logic.hpp#L262) | block | Utilities & toolbox | Edge-counting up/down counter: increments on each rising edge of up, decrements on each rising edge of down. Returns the running count |
| [`CTD`](inc/wet/toolbox/iec61131.hpp#L348) | block | Utilities & toolbox | CTD Counter (Count Down) |
| [`CTU`](inc/wet/toolbox/iec61131.hpp#L308) | block | Utilities & toolbox | CTU Counter (Count Up) |
| [`CTUD`](inc/wet/toolbox/iec61131.hpp#L388) | block | Utilities & toolbox | CTUD Counter (Count Up Down) |
| [`current_loop_pi`](inc/wet/motor/foc.hpp#L26) | function | Motor control | Current-loop PI gains by closed-loop pole placement on the R–L plant |
| [`damp`](inc/wet/analysis/analysis.hpp#L859) | function | Frequency-domain analysis (host) | Compute natural frequency and damping for each pole |
| [`damping_ratio_from_overshoot_percent`](inc/wet/design/pid_design.hpp#L394) | function | Design & synthesis | Map percent overshoot target to equivalent damping ratio |
| [`dare`](inc/wet/design/riccati.hpp#L580) | function | Design & synthesis | Solve the Discrete Algebraic Riccati Equation (DARE) |
| [`dare_rde`](inc/wet/design/riccati.hpp#L169) | function | Design & synthesis | Solve DARE via Riccati Difference Equation (RDE) iteration |
| [`dare_sda`](inc/wet/design/riccati.hpp#L82) | function | Design & synthesis | Solve DARE via Structure-Preserving Doubling Algorithm (SDA) |
| [`db2mag`](inc/wet/math/math.hpp#L415) | function | Scalar math & complex | Decibels to magnitude, 10^(db/20) |
| [`DcBusLimiter`](inc/wet/motor/limits.hpp#L39) | block | Motor control | Holds the inverter's torque current within DC-bus current/power limits |
| [`DcBusLimits`](inc/wet/motor/limits.hpp#L10) | block | Motor control | DC-bus current and voltage limits for an inverter |
| [`DcBusState`](inc/wet/motor/limits.hpp#L27) | block | Motor control | DC-bus state and the torque-current derate it implies |
| [`dcgain`](inc/wet/analysis/analysis.hpp#L653) | function | Frequency-domain analysis (host) | Compute DC gain of a continuous-time system |
| [`deadband`](inc/wet/toolbox/conditioning.hpp#L23) | function | Utilities & toolbox | Dead zone over `[lower, upper]`, matching Simulink's Dead Zone block |
| [`Debounce`](inc/wet/toolbox/logic.hpp#L201) | block | Utilities & toolbox | Debounce: the output adopts in only after in differs from the current output continuously for stable_time. Rejects contact bounce and brief glitches. (Not an IEC block — the one everyone hand-rolls.) |
| [`deg2rad`](inc/wet/math/math.hpp#L437) | function | Scalar math & complex | Degrees to radians, deg·π/180 |
| [`Delay`](inc/wet/filters/filters.hpp#L831) | block | Filters & signal conditioning | Discrete-time delay buffer |
| [`derate_window`](inc/wet/motor/thermal.hpp#L14) | function | Motor control | A two-breakpoint derating curve: 1 below derate_start, 0 at cutoff |
| [`det`](inc/wet/matrix/functions.hpp#L126) | function | Linear algebra | Matrix determinant |
| [`DFF`](inc/wet/toolbox/iec61131.hpp#L445) | block | Utilities & toolbox | D Flip-Flop (edge-triggered data latch) |
| [`DhChain`](inc/wet/kinematics/serial_arm.hpp#L90) | block | Kinematics | An N-joint DH chain (the arm geometry) |
| [`DhJoint`](inc/wet/kinematics/serial_arm.hpp#L67) | block | Kinematics | One joint's standard (distal) DH parameters and motion limits |
| [`diag`](inc/wet/matlab.hpp#L227) | function | MATLAB-style aliases (host) | Returns a square diagonal matrix from the given array |
| [`Diagonal`](inc/wet/matrix/views.hpp#L29) | block | Linear algebra | Diagonal view of a square matrix |
| [`DirectQuadrature`](inc/wet/transforms.hpp#L62) | block | Motor control | Direct-quadrature (rotor-frame) component pair |
| [`DirectQuadratureZero`](inc/wet/transforms.hpp#L215) | block | Motor control | Direct-quadrature-zero (rotor-frame) component triple |
| [`discrete_lqg`](inc/wet/controllers/lqg.hpp#L67) | function | Runtime controllers | Linear-Quadratic-Gaussian regulator design combining LQR and Kalman filter |
| [`discrete_lqgi`](inc/wet/controllers/lqgi.hpp#L80) | function | Runtime controllers | Linear-Quadratic-Gaussian with integral action for tracking |
| [`discrete_lqi`](inc/wet/controllers/lqi.hpp#L39) | function | Runtime controllers | Linear-Quadratic Integral design for tracking with servo action |
| [`discrete_lqr`](inc/wet/controllers/lqr.hpp#L62) | function | Runtime controllers | Discrete-time Linear-Quadratic Regulator design |
| [`discrete_lqr_from_continuous`](inc/wet/controllers/lqr.hpp#L207) | function | Runtime controllers | Design discrete LQR from continuous-time system via discretization |
| [`DiscretizationMethod`](inc/wet/systems/discretization.hpp#L8) | block | LTI models | Discretization methods for continuous-time state-space systems |
| [`discretize`](inc/wet/systems/discretization.hpp#L193) | function | LTI models | Discretize a continuous-time state-space system |
| [`discretize_forward_euler_impl`](inc/wet/systems/discretization.hpp#L21) | function | LTI models | Discretize using Forward Euler (explicit Euler) |
| [`discretize_lqr_cost`](inc/wet/controllers/lqr.hpp#L125) | function | Runtime controllers | Discretize a continuous LQR cost integral over one sample (Van Loan) |
| [`discretize_tustin_impl`](inc/wet/systems/discretization.hpp#L127) | function | LTI models | Discretize a continuous-time state-space system using Tustin method |
| [`discretize_zoh_impl`](inc/wet/systems/discretization.hpp#L66) | function | LTI models | Discretize using Zero-Order Hold (ZOH) |
| [`DisturbanceObserverConfig`](inc/wet/estimation/disturbance_observer.hpp#L17) | block | Observers & estimators | Configuration for a first-order disturbance observer |
| [`DLATCH`](inc/wet/toolbox/iec61131.hpp#L475) | block | Utilities & toolbox | D Latch (level-sensitive / transparent latch) |
| [`dlqr`](inc/wet/matlab.hpp#L477) | function | MATLAB-style aliases (host) | Discrete-time Linear-Quadratic Regulator design |
| [`dlyap`](inc/wet/design/lyapunov.hpp#L101) | function | Design & synthesis | Solve the discrete-time Lyapunov (Stein) equation @f$ A X A^\top - X + Q = 0 @f$ |
| [`dlyap`](inc/wet/matlab.hpp#L716) | function | MATLAB-style aliases (host) | MATLAB alias for the discrete Lyapunov solve @f$ AXA^\top-X+Q=0 @f$ |
| [`DqCommand`](inc/wet/motor/foc.hpp#L261) | block | Motor control | Result of FOController::current_controller(): the dq voltage command plus its saturation signals |
| [`eig`](inc/wet/matlab.hpp#L327) | function | MATLAB-style aliases (host) | MATLAB short alias for the eigenvalues of a square matrix |
| [`EigenResult`](inc/wet/matrix/eigen.hpp#L13) | block | Linear algebra | Eigenvalue computation result |
| [`ErrorStateJacobian`](inc/wet/estimation/eskf.hpp#L118) | block | Observers & estimators | Error-state prediction Jacobians (nominal state updated externally) |
| [`ErrorStateKalmanFilter`](inc/wet/estimation/eskf.hpp#L153) | block | Observers & estimators | Error-State Kalman Filter for attitude estimation |
| [`ESCConfig`](inc/wet/controllers/esc.hpp#L49) | block | Runtime controllers | Extremum-seeking controller configuration (discrete realization) |
| [`eskf_design`](inc/wet/estimation/eskf.hpp#L47) | function | Observers & estimators | Error-State Kalman Filter design from IMU sensor specifications |
| [`eskf_update_imu`](inc/wet/estimation/sensor_fusion.hpp#L190) | function | Observers & estimators | Full ESKF predict+update+inject cycle for 6-axis IMU fusion |
| [`ESKFOrientationFilter`](inc/wet/estimation/sensor_fusion.hpp#L312) | block | Observers & estimators | ESKF-based orientation estimator (convenience wrapper) |
| [`ESKFResult`](inc/wet/estimation/eskf.hpp#L23) | block | Observers & estimators | Error-State Kalman Filter design result |
| [`estim`](inc/wet/matlab.hpp#L341) | function | MATLAB-style aliases (host) | Form state estimator from system and estimator gain |
| [`eval_frf`](inc/wet/systems/state_space.hpp#L82) | function | LTI models | Evaluate frequency response of state-space system |
| [`Exact`](inc/wet/simulation/integrator.hpp#L35) | block | Simulation (host) | Exact integrator for LTI systems |
| [`exp`](inc/wet/math/math.hpp#L212) | function | Scalar math & complex | Exponential function |
| [`expm`](inc/wet/matrix/functions.hpp#L247) | function | Linear algebra | Matrix exponential using scaling and squaring with Padé approximation |
| [`expo`](inc/wet/toolbox/conditioning.hpp#L101) | function | Utilities & toolbox | Exponential response curve `y = (1−k)·x + k·x³` (RC "expo") |
| [`ExtendedKalmanFilter`](inc/wet/estimation/ekf.hpp#L81) | block | Observers & estimators | Extended Kalman Filter for nonlinear discrete-time systems |
| [`ExtremumSeekingController`](inc/wet/controllers/esc.hpp#L206) | block | Runtime controllers | Extremum-seeking controller runtime (model-free online optimizer) |
| [`eye`](inc/wet/matlab.hpp#L270) | function | MATLAB-style aliases (host) | Create an identity matrix of size n x n |
| [`F_TRIG`](inc/wet/toolbox/iec61131.hpp#L132) | block | Utilities & toolbox | F_TRIG (Falling Edge Trigger) |
| [`feedback`](inc/wet/systems/state_space.hpp#L254) | function | LTI models | Negative feedback connection of two state-space systems |
| [`FetLossModel`](inc/wet/motor/thermal.hpp#L116) | block | Motor control | First-order inverter FET loss model (conduction + switching) |
| [`field_weakening_id`](inc/wet/motor/field_weakening.hpp#L17) | function | Motor control | Feedforward field-weakening d-axis current from the voltage ellipse |
| [`FieldWeakening`](inc/wet/motor/field_weakening.hpp#L91) | block | Motor control | Field-weakening current-reference regulator (voltage-feedback or feedforward) |
| [`FieldWeakeningConfig`](inc/wet/motor/field_weakening.hpp#L77) | block | Motor control | Configuration for FieldWeakening |
| [`five_bar_symmetric`](inc/wet/kinematics/scara.hpp#L231) | function | Kinematics | Build a symmetric five-bar parallel SCARA |
| [`FiveBar`](inc/wet/kinematics/scara.hpp#L76) | block | Kinematics | Planar five-bar parallel manipulator (parallel SCARA) |
| [`FiveBarGeometry`](inc/wet/kinematics/scara.hpp#L41) | block | Kinematics | Symmetric five-bar geometry (two base motors, equal proximal/distal links) |
| [`FixedStepSolver`](inc/wet/simulation/solver.hpp#L84) | block | Simulation (host) | Fixed-step ODE solver |
| [`floor`](inc/wet/math/math.hpp#L292) | function | Scalar math & complex | Floor — largest integer ≤ x |
| [`flux_from_Kv`](inc/wet/motor/foc.hpp#L139) | function | Motor control | PM flux linkage from the datasheet velocity constant @f$ K_v @f$ |
| [`flux_from_torque_constant`](inc/wet/motor/foc.hpp#L75) | function | Motor control | PM flux linkage from a motor's torque constant (amplitude-invariant) |
| [`fmod`](inc/wet/math/math.hpp#L330) | function | Scalar math & complex | Floating-point remainder, x − y·trunc(x/y) (sign of x), matching std::fmod's truncated-quotient convention |
| [`FocResult`](inc/wet/motor/foc.hpp#L245) | block | Motor control | Result of one FOController::step(), carrying the actuator command plus the saturation/measurement signals an outer (velocity/position) loop needs to propagate anti-windup back up a cascade |
| [`forward_substitute`](inc/wet/matrix/solve.hpp#L23) | function | Linear algebra | Forward substitution to solve L x = b |
| [`ForwardEuler`](inc/wet/simulation/integrator.hpp#L79) | block | Simulation (host) | Forward Euler integrator |
| [`foster_thermal_ss`](inc/wet/motor/thermal.hpp#L32) | function | Motor control | Continuous state-space model of a Foster RC thermal network |
| [`francis_qr`](inc/wet/matrix/eigen.hpp#L108) | function | Linear algebra | Francis double-shift QR on an upper Hessenberg matrix |
| [`frobenius_norm`](inc/wet/matrix/functions.hpp#L55) | function | Linear algebra | Frobenius norm: square root of sum of squares of all elements |
| [`full_qr`](inc/wet/matrix/decomposition.hpp#L265) | function | Linear algebra | Full QR factorization via Householder reflections (real or complex T) |
| [`FullQR`](inc/wet/matrix/decomposition.hpp#L250) | block | Linear algebra | Result of a full (complete) QR factorization |
| [`gain_margin_unwrapped`](inc/wet/analysis/analysis.hpp#L252) | function | Frequency-domain analysis (host) | Find gain margin using unwrapped phase trajectory |
| [`Goertzel`](inc/wet/filters/spectral.hpp#L29) | block | Filters & signal conditioning | Generalized Goertzel single-bin DFT — amplitude/phase at one frequency |
| [`gram`](inc/wet/matlab.hpp#L726) | function | MATLAB-style aliases (host) | MATLAB alias for the controllability/observability Gramian of a system |
| [`GreyBoxIdentificationResult`](inc/wet/analysis/identification.hpp#L177) | block | Frequency-domain analysis (host) | Selected model used by downstream model-based controller design |
| [`HarmonicAnalyzer`](inc/wet/filters/spectral.hpp#L127) | block | Filters & signal conditioning | Harmonic analyzer — a Goertzel bank over a fundamental and K−1 harmonics |
| [`HarmonicSuppressor`](inc/wet/controllers/harmonic_suppression.hpp#L104) | block | Runtime controllers | Multi-resonant harmonic suppressor — a parallel bank of PR resonators |
| [`HarmonicSuppressorResult`](inc/wet/controllers/harmonic_suppression.hpp#L34) | block | Runtime controllers | Design result for a multi-resonant harmonic suppressor |
| [`hessenberg_reduce`](inc/wet/matrix/eigen.hpp#L32) | function | Linear algebra | Reduce a square matrix to upper Hessenberg form by Householder reflections |
| [`HighPass`](inc/wet/filters/filters.hpp#L983) | block | Filters & signal conditioning | First-order high-pass (washout) filter runtime |
| [`highpass_2nd`](inc/wet/filters/filters.hpp#L499) | function | Filters & signal conditioning | Second-order high-pass filter (RBJ) |
| [`highshelf`](inc/wet/filters/filters.hpp#L571) | function | Filters & signal conditioning | High-shelf EQ filter: boost or cut everything above fc |
| [`hinfnorm`](inc/wet/matlab.hpp#L753) | function | MATLAB-style aliases (host) | MATLAB alias for the H∞ system norm norm(sys,Inf) / hinfnorm(sys) |
| [`hypot`](inc/wet/math/math.hpp#L56) | function | Scalar math & complex | Euclidean distance hypot(x, y) = √(x² + y²), without overflow |
| [`Hysteresis`](inc/wet/toolbox/conditioning.hpp#L177) | block | Utilities & toolbox | Hysteresis comparator (Schmitt trigger): bool output with separate on/off thresholds to reject chatter |
| [`impedance`](inc/wet/analysis/analysis.hpp#L1528) | function | Frequency-domain analysis (host) | Compute impedance frequency response from a SISO admittance system |
| [`impedance_direct`](inc/wet/analysis/analysis.hpp#L1563) | function | Frequency-domain analysis (host) | Compute impedance frequency response from a SISO impedance transfer function |
| [`ImpedanceResult`](inc/wet/analysis/analysis.hpp#L1425) | block | Frequency-domain analysis (host) | Result of impedance frequency response evaluation |
| [`impulse`](inc/wet/analysis/analysis.hpp#L991) | function | Frequency-domain analysis (host) | Impulse response of a (MIMO) state-space system |
| [`impulseplot`](inc/wet/simulation/plot_plotly.hpp#L327) | function | Simulation (host) | Plot an impulse response, one trace per input/output pair |
| [`infinity_norm`](inc/wet/matrix/functions.hpp#L11) | function | Linear algebra | Infinity norm: maximum absolute row sum |
| [`initial`](inc/wet/analysis/analysis.hpp#L1021) | function | Frequency-domain analysis (host) | Initial-condition (free) response of a (MIMO) state-space system |
| [`InputShaper`](inc/wet/trajectory/input_shaper.hpp#L176) | block | Trajectory & motion planning | Input-shaper runtime — convolves a command stream with the shaper impulses |
| [`InputShaperBank`](inc/wet/trajectory/input_shaper.hpp#L239) | block | Trajectory & motion planning | Multi-axis input-shaper bank — one shaper per axis, shared buffer length |
| [`InputShaperResult`](inc/wet/trajectory/input_shaper.hpp#L60) | block | Trajectory & motion planning | Input-shaper design result: impulse amplitudes and sample delays |
| [`instantaneous_power`](inc/wet/transforms.hpp#L573) | function | Motor control | Instantaneous active and reactive power from dq quantities |
| [`InstantaneousPower`](inc/wet/transforms.hpp#L525) | block | Motor control | Instantaneous active and reactive power |
| [`IntegrationResult`](inc/wet/simulation/integrator.hpp#L11) | block | Simulation (host) | Result of an integration step |
| [`inverse_clarke_transform`](inc/wet/transforms.hpp#L336) | function | Motor control | Inverse Clarke transform (αβ → abc) |
| [`inverse_clarke_zero_transform`](inc/wet/transforms.hpp#L280) | function | Motor control | Inverse zero-retaining Clarke transform (αβ0 → abc) |
| [`inverse_deadband`](inc/wet/toolbox/conditioning.hpp#L53) | function | Utilities & toolbox | Inverse dead zone: add an offset to overcome a physical dead zone (valve overlap, static friction, motor stiction), with independent negative/positive offsets |
| [`inverse_lerp`](inc/wet/toolbox/scaling.hpp#L34) | function | Utilities & toolbox | Inverse of lerp: the fraction t such that `lerp(a, b, t) == x` |
| [`inverse_park_clarke_transform`](inc/wet/transforms.hpp#L433) | function | Motor control | Fused inverse Park-Clarke transform (dq → abc) |
| [`inverse_park_clarke_zero_transform`](inc/wet/transforms.hpp#L512) | function | Motor control | Fused inverse Park-Clarke transform with zero (dq0 → abc) |
| [`inverse_park_transform`](inc/wet/transforms.hpp#L380) | function | Motor control | Inverse Park transform (dq → αβ) |
| [`inverse_park_zero_transform`](inc/wet/transforms.hpp#L480) | function | Motor control | Inverse Park transform with zero passthrough (dq0 → αβ0) |
| [`inverse_symmetrical_components`](inc/wet/transforms.hpp#L646) | function | Motor control | Inverse symmetrical-component transform (012 → abc) |
| [`iq_from_torque`](inc/wet/motor/foc.hpp#L178) | function | Motor control | q-axis current command for a requested torque (non-salient PMSM, Id=0) |
| [`is_closed_loop_stable_discrete`](inc/wet/design/stability.hpp#L220) | function | Design & synthesis | Check closed-loop stability for discrete system with state feedback |
| [`is_controllable`](inc/wet/design/stability.hpp#L159) | function | Design & synthesis | Check if a system is controllable |
| [`is_observable`](inc/wet/design/stability.hpp#L176) | function | Design & synthesis | Check if a system is observable |
| [`is_stabilizable`](inc/wet/design/riccati.hpp#L14) | function | Design & synthesis | Check if (A, B) is a stabilizable pair |
| [`is_stable_continuous`](inc/wet/analysis/analysis.hpp#L826) | function | Frequency-domain analysis (host) | Check continuous-time stability |
| [`is_stable_discrete`](inc/wet/design/stability.hpp#L193) | function | Design & synthesis | Check if a discrete-time system matrix A is stable |
| [`isfinite`](inc/wet/math/math.hpp#L387) | function | Scalar math & complex | Finiteness test — false for NaN and ±∞ |
| [`jacobi_svd_tall`](inc/wet/matrix/svd.hpp#L45) | function | Linear algebra | One-sided Jacobi SVD of a tall/square matrix A (P×Q, P ≥ Q) |
| [`JointLimits`](inc/wet/trajectory/cartesian_move.hpp#L77) | block | Trajectory & motion planning | Per-joint velocity and acceleration limits for a task-space move |
| [`JordanBlock`](inc/wet/design/pole_placement.hpp#L420) | block | Design & synthesis | One Jordan mini-block of a desired closed-loop spectrum |
| [`JordanPlan`](inc/wet/design/pole_placement.hpp#L437) | block | Design & synthesis | Precomputed, K-independent data for the Klein–Moore construction |
| [`JunctionEstimator`](inc/wet/motor/thermal.hpp#L176) | block | Motor control | FET junction-temperature estimator: case temperature plus a thermal model |
| [`kalman`](inc/wet/estimation/kalman.hpp#L44) | function | Observers & estimators | Steady-state Kalman filter design |
| [`KalmanFilter`](inc/wet/estimation/kalman.hpp#L121) | block | Observers & estimators | Runtime Kalman filter for embedded systems |
| [`KalmanResult`](inc/wet/estimation/kalman.hpp#L12) | block | Observers & estimators | Steady-state Kalman filter design result |
| [`lag`](inc/wet/controllers/lead_lag.hpp#L143) | function | Runtime controllers | Design a lag compensator from desired low-frequency gain boost |
| [`lambda_tuning`](inc/wet/design/pid_design.hpp#L285) | function | Design & synthesis | Lambda tuning for FOPDT model |
| [`lead`](inc/wet/controllers/lead_lag.hpp#L97) | function | Runtime controllers | Design a lead compensator from desired phase boost at a target frequency |
| [`lead_lag`](inc/wet/controllers/lead_lag.hpp#L178) | function | Runtime controllers | Design a lead-lag compensator (cascade of lead + lag sections) |
| [`lead_lag_direct`](inc/wet/controllers/lead_lag.hpp#L210) | function | Runtime controllers | Direct lead-lag specification from zero/pole locations |
| [`LeadLagController`](inc/wet/controllers/lead_lag.hpp#L228) | block | Runtime controllers | Discrete Lead-Lag Compensator |
| [`lerp`](inc/wet/toolbox/scaling.hpp#L22) | function | Utilities & toolbox | Linear interpolation between a and b by fraction t |
| [`LIMIT`](inc/wet/toolbox/iec61131.hpp#L581) | function | Utilities & toolbox | LIMIT (IEC 61131-3 selection function): clamp in to [mn, mx] |
| [`linear_screw`](inc/wet/toolbox/actuator.hpp#L147) | function | Utilities & toolbox | Build a ServoAxis for a linear axis driven by a leadscrew/belt |
| [`LinearDelta`](inc/wet/kinematics/motion_maps.hpp#L264) | block | Kinematics | Linear delta robot — per-carriage closed-form inverse, sphere- trilateration forward. Towers at 90°, 210°, 330° |
| [`LinearDeltaGeometry`](inc/wet/kinematics/motion_maps.hpp#L249) | block | Kinematics | Linear delta geometry (three vertical carriages, fixed-length rods) |
| [`linearize`](inc/wet/design/linearization.hpp#L110) | function | Design & synthesis | Linearize nonlinear dynamics and output maps about an operating point |
| [`LinearPath`](inc/wet/trajectory/cartesian_move.hpp#L48) | block | Trajectory & motion planning | A straight-line path `p(s) = start + s·dir`, `s ∈ [0, length]` |
| [`linmod`](inc/wet/matlab.hpp#L185) | function | MATLAB-style aliases (host) | MATLAB-style nonlinear linearization about an operating point |
| [`log`](inc/wet/matrix/functions.hpp#L361) | function | Linear algebra | Matrix logarithm using inverse scaling and squaring |
| [`log`](inc/wet/math/math.hpp#L229) | function | Scalar math & complex | Natural logarithm |
| [`log10`](inc/wet/math/math.hpp#L349) | function | Scalar math & complex | Base-10 logarithm, log10(x) = ln(x) / ln(10) |
| [`loop_response`](inc/wet/analysis/analysis.hpp#L570) | function | Frequency-domain analysis (host) | Compute open-loop L, sensitivity S, complementary sensitivity T, and Nyquist data |
| [`LoopResponseResult`](inc/wet/analysis/analysis.hpp#L443) | block | Frequency-domain analysis (host) | Open-loop and closed-loop frequency response package |
| [`LoopSummary`](inc/wet/analysis/analysis.hpp#L478) | block | Frequency-domain analysis (host) | Compact loop summary metrics for quick stability/robustness checks |
| [`LowerTriangle`](inc/wet/matrix/views.hpp#L121) | block | Linear algebra | Lower triangular view of a square matrix |
| [`LowPass`](inc/wet/filters/filters.hpp#L600) | block | Filters & signal conditioning | Nth-order low-pass filter |
| [`lowpass_1st`](inc/wet/filters/filters.hpp#L93) | function | Filters & signal conditioning | First-order low-pass filter design |
| [`lowpass_2nd`](inc/wet/filters/filters.hpp#L136) | function | Filters & signal conditioning | Second-order low-pass filter design |
| [`lowshelf`](inc/wet/filters/filters.hpp#L542) | function | Filters & signal conditioning | Low-shelf EQ filter: boost or cut everything below fc |
| [`lqg`](inc/wet/matlab.hpp#L536) | function | MATLAB-style aliases (host) | Linear-Quadratic-Gaussian regulator design |
| [`LQG`](inc/wet/controllers/lqg.hpp#L113) | block | Runtime controllers | Linear-Quadratic-Gaussian (LQG) controller |
| [`lqg_from_parts`](inc/wet/controllers/lqg.hpp#L95) | function | Runtime controllers | Combine separate Kalman filter and LQR designs into an LQG controller |
| [`LQGI`](inc/wet/controllers/lqgi.hpp#L107) | block | Runtime controllers | Linear-Quadratic-Gaussian-Integral (LQGI) controller |
| [`LQGIResult`](inc/wet/controllers/lqgi.hpp#L12) | block | Runtime controllers | LQGI design result |
| [`lqgreg`](inc/wet/matlab.hpp#L552) | function | MATLAB-style aliases (host) | Combine separate Kalman filter and LQR designs into an LQG controller |
| [`LQGResult`](inc/wet/controllers/lqg.hpp#L14) | block | Runtime controllers | LQG design result |
| [`lqgtrack`](inc/wet/matlab.hpp#L564) | function | MATLAB-style aliases (host) | Linear-Quadratic-Gaussian design with integral action for tracking |
| [`lqi`](inc/wet/matlab.hpp#L523) | function | MATLAB-style aliases (host) | Linear-Quadratic Integral design for tracking |
| [`LQI`](inc/wet/controllers/lqi.hpp#L99) | block | Runtime controllers | Linear-Quadratic-Integral (LQI) controller |
| [`LQIResult`](inc/wet/controllers/lqi.hpp#L14) | block | Runtime controllers | LQI design result |
| [`lqr`](inc/wet/matlab.hpp#L457) | function | MATLAB-style aliases (host) | Continuous-time LQR design (MATLAB's lqr) |
| [`LQR`](inc/wet/controllers/lqr.hpp#L333) | block | Runtime controllers | Runtime Linear-Quadratic Regulator |
| [`lqr_gain`](inc/wet/design/riccati.hpp#L667) | function | Design & synthesis | Optimal LQR state-feedback gain from a Riccati solution |
| [`lqrd`](inc/wet/matlab.hpp#L492) | function | MATLAB-style aliases (host) | Design discrete LQR from continuous-time system via discretization |
| [`LQRResult`](inc/wet/controllers/lqr.hpp#L18) | block | Runtime controllers | Linear-Quadratic Regulator design result |
| [`lsim`](inc/wet/analysis/analysis.hpp#L1052) | function | Frequency-domain analysis (host) | Forced time response of a (MIMO) state-space system to an input signal |
| [`LsimInfo`](inc/wet/analysis/analysis.hpp#L1267) | block | Frequency-domain analysis (host) | Transient characteristics of an arbitrary response signal |
| [`lsiminfo`](inc/wet/analysis/analysis.hpp#L1284) | function | Frequency-domain analysis (host) | Compute transient characteristics from an output/time signal |
| [`lsimplot`](inc/wet/simulation/plot_plotly.hpp#L343) | function | Simulation (host) | Plot a forced (lsim) simulation, one trace per output |
| [`LsimResult`](inc/wet/analysis/analysis.hpp#L900) | block | Frequency-domain analysis (host) | Result of a single-trajectory simulation: time, output, and state history |
| [`lu_decomposition`](inc/wet/matrix/decomposition.hpp#L111) | function | Linear algebra | LU decomposition with partial pivoting |
| [`Lut1D`](inc/wet/toolbox/lookup.hpp#L64) | block | Utilities & toolbox | 1-D interpolating lookup table over monotonic breakpoints |
| [`Lut2D`](inc/wet/toolbox/lookup.hpp#L122) | block | Utilities & toolbox | 2-D bilinear interpolating lookup table over a regular grid |
| [`lut_segment`](inc/wet/toolbox/lookup.hpp#L31) | function | Utilities & toolbox | Index of the interpolation segment containing x |
| [`lyap`](inc/wet/design/lyapunov.hpp#L74) | function | Design & synthesis | Solve the continuous-time Lyapunov equation @f$ A X + X A^\top + Q = 0 @f$ |
| [`lyap`](inc/wet/matlab.hpp#L706) | function | MATLAB-style aliases (host) | MATLAB alias for the continuous Lyapunov solve @f$ AX+XA^\top+Q=0 @f$ |
| [`MadgwickFilter`](inc/wet/estimation/sensor_fusion.hpp#L79) | block | Observers & estimators | Madgwick gradient-descent AHRS filter |
| [`mag2db`](inc/wet/math/math.hpp#L404) | function | Scalar math & complex | Magnitude to decibels, 20·log10(mag) |
| [`MahonyFilter`](inc/wet/estimation/sensor_fusion.hpp#L137) | block | Observers & estimators | Mahony nonlinear complementary filter with PI correction |
| [`margin`](inc/wet/matlab.hpp#L663) | function | MATLAB-style aliases (host) | Gain and phase margins of a SISO loop over a frequency grid |
| [`MarginResult`](inc/wet/matlab.hpp#L649) | block | MATLAB-style aliases (host) | Gain/phase margins and their crossover frequencies |
| [`Matrix`](inc/wet/matrix/matrix.hpp#L49) | block | Linear algebra | Fixed-size, stack-allocated matrix for linear algebra operations |
| [`MeasJacobian`](inc/wet/estimation/ekf.hpp#L53) | block | Observers & estimators | Measurement prediction result from the user's observation function |
| [`MechanicalEstimator`](inc/wet/motor/mechanical_estimator.hpp#L97) | block | Motor control | Cheap-predict mechanical estimator for position, speed, and load torque |
| [`MechanicalEstimatorConfig`](inc/wet/motor/mechanical_estimator.hpp#L70) | block | Motor control | Configuration for MechanicalEstimator |
| [`MedianFilter`](inc/wet/filters/filters.hpp#L1026) | block | Filters & signal conditioning | Sliding-window median filter — nonlinear spike/outlier rejection |
| [`middlebrook`](inc/wet/analysis/analysis.hpp#L1597) | function | Frequency-domain analysis (host) | Middlebrook stability analysis for cascaded source-load systems |
| [`MiddlebrookResult`](inc/wet/analysis/analysis.hpp#L1443) | block | Frequency-domain analysis (host) | Result of Middlebrook minor loop gain analysis |
| [`minmax`](inc/wet/backend.hpp#L139) | function | Core, configuration & backend vocabulary | Ordered {min, max} pair returned by value |
| [`motor_constant`](inc/wet/motor/foc.hpp#L156) | function | Motor control | Motor constant @f$ K_m @f$ (torque per √copper-loss) — a figure of merit |
| [`MovingAverage`](inc/wet/filters/filters.hpp#L913) | block | Filters & signal conditioning | Moving-average (boxcar) filter — also a DC-preserving harmonic-notch comb |
| [`mstogi`](inc/wet/filters/sogi.hpp#L69) | function | Filters & signal conditioning | Mixed Second/Third-Order Generalized Integrator (MSTOGI) |
| [`MSTOGI`](inc/wet/filters/sogi.hpp#L211) | block | Filters & signal conditioning | Runtime MSTOGI with exact resonator and forward-Euler washout |
| [`mtpa_id_from_iq`](inc/wet/motor/mtpa.hpp#L11) | function | Motor control | MTPA d-axis current on the trajectory for a given q-axis current |
| [`mtpa_reference`](inc/wet/motor/mtpa.hpp#L50) | function | Motor control | MTPA dq current reference for a commanded torque |
| [`MtpaReference`](inc/wet/motor/mtpa.hpp#L100) | block | Motor control | Maximum-torque-per-ampere current-reference generator (PMSM / IPMSM / SynRM) |
| [`MultiPRController`](inc/wet/controllers/pr.hpp#L298) | block | Runtime controllers | Multi-harmonic PR Controller |
| [`MultiSine`](inc/wet/estimation/excitation.hpp#L1065) | block | Observers & estimators | Sum-of-tones multi-sine runtime generator |
| [`MultiSineConfig`](inc/wet/estimation/excitation.hpp#L521) | block | Observers & estimators | Configuration for fixed-component multi-sine excitation |
| [`MultiSineResult`](inc/wet/estimation/excitation.hpp#L566) | block | Observers & estimators | Multi-sine design payload |
| [`MUX`](inc/wet/toolbox/iec61131.hpp#L600) | function | Utilities & toolbox | MUX (IEC 61131-3 multiplexer): select input k of N (0-based) |
| [`nearbyint`](inc/wet/math/math.hpp#L316) | function | Scalar math & complex | Round to nearest integer. Runtime follows the backend (round half to even); the compile-time path rounds ties away from zero — immaterial for range reduction |
| [`negative_sequence_ab`](inc/wet/filters/pll.hpp#L378) | function | Filters & signal conditioning | Instantaneous negative-sequence αβ from a quadrature signal pair |
| [`NoFieldWeakening`](inc/wet/motor/field_weakening.hpp#L187) | block | Motor control | Null field-weakening policy — passes the base reference through unchanged |
| [`norm`](inc/wet/matlab.hpp#L743) | function | MATLAB-style aliases (host) | MATLAB alias for the H2 system norm norm(sys,2) |
| [`norm_h2`](inc/wet/analysis/analysis.hpp#L681) | function | Frequency-domain analysis (host) | H2 norm of a state-space system |
| [`norm_hinf`](inc/wet/analysis/analysis.hpp#L733) | function | Frequency-domain analysis (host) | H∞ norm of a state-space system: @f$ \sup_\omega \bar\sigma\,G(j\omega) @f$ |
| [`notch`](inc/wet/filters/filters.hpp#L457) | function | Filters & signal conditioning | Second-order band-reject (notch) filter |
| [`null`](inc/wet/matlab.hpp#L314) | function | MATLAB-style aliases (host) | MATLAB short alias for an orthonormal null-space basis |
| [`null_space`](inc/wet/matrix/svd.hpp#L331) | function | Linear algebra | Orthonormal basis for the null space {x : A·x = 0} via SVD |
| [`nyquist`](inc/wet/analysis/analysis.hpp#L525) | function | Frequency-domain analysis (host) | Compute Nyquist data for a SISO state-space system |
| [`nyquistplot`](inc/wet/simulation/plot_plotly.hpp#L417) | function | Simulation (host) | Plot a Nyquist locus with the -1 critical point marked |
| [`observability_gramian`](inc/wet/design/stability.hpp#L119) | function | Design & synthesis | Continuous/discrete observability Gramian @f$ W_o @f$ |
| [`observability_matrix`](inc/wet/design/stability.hpp#L60) | function | Design & synthesis | Compute the observability matrix [C; CA; CA²; ...; CA^(N-1)] |
| [`Observer`](inc/wet/estimation/observer.hpp#L412) | block | Observers & estimators | Luenberger state observer (runtime) |
| [`ObserverResult`](inc/wet/estimation/observer.hpp#L69) | block | Observers & estimators | Luenberger observer design result |
| [`OffDelayTimer`](inc/wet/toolbox/logic.hpp#L122) | block | Utilities & toolbox | Off-delay timer: output goes true immediately when in is true and stays true until in has been false continuously for delay |
| [`OnDelayTimer`](inc/wet/toolbox/logic.hpp#L85) | block | Utilities & toolbox | On-delay timer: output goes true once in has been held true continuously for delay; drops immediately when in goes false |
| [`one_norm`](inc/wet/matrix/functions.hpp#L33) | function | Linear algebra | One norm: maximum absolute column sum |
| [`pade_delay_1st`](inc/wet/filters/filters.hpp#L258) | function | Filters & signal conditioning | First-order Pade approximation of time delay |
| [`pade_delay_2nd`](inc/wet/filters/filters.hpp#L296) | function | Filters & signal conditioning | Second-order Pade approximation of time delay |
| [`parallel`](inc/wet/systems/state_space.hpp#L188) | function | LTI models | Parallel connection of two state-space systems |
| [`park_transform`](inc/wet/transforms.hpp#L353) | function | Motor control | Park transform (αβ → dq) |
| [`park_zero_transform`](inc/wet/transforms.hpp#L463) | function | Motor control | Park transform with zero passthrough (αβ0 → dq0) |
| [`peaking`](inc/wet/filters/filters.hpp#L520) | function | Filters & signal conditioning | Peaking (bell) EQ filter: boost or cut a band around f0 |
| [`Periodic`](inc/wet/toolbox/timing.hpp#L102) | block | Utilities & toolbox | Periodic trigger — fires once per elapsed period |
| [`phase_margin_from_damping_ratio`](inc/wet/design/pid_design.hpp#L426) | function | Design & synthesis | Approximate phase margin from damping ratio |
| [`phase_margin_unwrapped`](inc/wet/analysis/analysis.hpp#L216) | function | Frequency-domain analysis (host) | Find phase margin using unwrapped phase trajectory |
| [`PhaseCalibrationCommand`](inc/wet/motor/calibration.hpp#L39) | block | Motor control | One step's output from PhaseParameterCalibrator |
| [`PhaseCalibrationConfig`](inc/wet/motor/calibration.hpp#L13) | block | Motor control | Configuration for online phase resistance/inductance commissioning |
| [`PhaseParameterCalibrator`](inc/wet/motor/calibration.hpp#L49) | block | Motor control | Online phase R/L identification by recursive least squares (PRBS injected) |
| [`pi_pole_placement_first_order`](inc/wet/design/pid_design.hpp#L612) | function | Design & synthesis | PI gains that place the closed-loop poles of a first-order plant |
| [`pid`](inc/wet/matlab.hpp#L102) | function | MATLAB-style aliases (host) | MATLAB-style parallel-form PID controller constructor |
| [`pid`](inc/wet/controllers/pid.hpp#L66) | function | Runtime controllers | 2-DOF PID controller design |
| [`pid_from_bandwidth`](inc/wet/design/pid_design.hpp#L311) | function | Design & synthesis | Design PID from desired bandwidth and phase margin |
| [`pid_from_performance_spec`](inc/wet/design/pid_design.hpp#L490) | function | Design & synthesis | Design PID directly from settling-time and overshoot targets |
| [`pid_pole_placement`](inc/wet/design/pid_design.hpp#L511) | function | Design & synthesis | Direct PID pole placement for a first-order-plus-dead-time model |
| [`PIDController`](inc/wet/controllers/pid.hpp#L146) | block | Runtime controllers | Discrete 2-DOF PID controller specialization |
| [`PIDMode`](inc/wet/controllers/pid.hpp#L106) | block | Runtime controllers | Compile-time selection of the PID control-law structure |
| [`PIDResult`](inc/wet/controllers/pid.hpp#L12) | block | Runtime controllers | 2-DOF PID controller design result |
| [`PIDRuntimeMode`](inc/wet/controllers/pid.hpp#L119) | block | Runtime controllers | Runtime operating mode for PIDController |
| [`pidtune`](inc/wet/matlab.hpp#L579) | function | MATLAB-style aliases (host) | PID controller tuning using frequency domain method |
| [`pinv`](inc/wet/matlab.hpp#L305) | function | MATLAB-style aliases (host) | MATLAB short alias for the Moore–Penrose pseudoinverse |
| [`place`](inc/wet/design/pole_placement.hpp#L44) | function | Design & synthesis | Robust multi-input pole placement (Kautsky–Nichols–Van Dooren, real poles) |
| [`place`](inc/wet/matlab.hpp#L430) | function | MATLAB-style aliases (host) | Robust multi-input pole placement (MATLAB's place) |
| [`place_jordan`](inc/wet/design/pole_placement.hpp#L692) | function | Design & synthesis | Exact pole placement with an arbitrary Jordan structure (Schmid–Ntogramatzidis–Nguyen–Pandey / Klein–Moore parametric form) |
| [`place_jordan_optimal`](inc/wet/design/pole_placement.hpp#L779) | function | Design & synthesis | Robust / minimum-gain arbitrary pole placement (Schmid et al., Methods 1–2) |
| [`plan_for_sign`](inc/wet/trajectory/trapezoidal.hpp#L135) | function | Trajectory & motion planning | Plan the three-segment profile assuming the cruise velocity has sign s |
| [`plot_bode`](inc/wet/simulation/plot_plotly.hpp#L138) | function | Simulation (host) | Plot Bode magnitude and phase as subplots |
| [`plot_line`](inc/wet/simulation/plot_plotly.hpp#L191) | function | Simulation (host) | Simple line plot of time vs value |
| [`plot_simulation`](inc/wet/simulation/plot_plotly.hpp#L71) | function | Simulation (host) | Plot simulation results with subplots for states, outputs, and inputs |
| [`plot_step`](inc/wet/simulation/plot_plotly.hpp#L220) | function | Simulation (host) | Plot step response data |
| [`PmacServo`](inc/wet/motor/servo.hpp#L80) | block | Motor control | Thin field-oriented PMAC servo: {Iabc, Vdc, θ} in, duties out |
| [`PmacServoConfig`](inc/wet/motor/servo.hpp#L44) | block | Motor control | Configuration for PmacServo |
| [`PmsmEstimatorConfig`](inc/wet/motor/predictive_current.hpp#L105) | block | Motor control | Configuration for PmsmParameterEstimator |
| [`PmsmModel`](inc/wet/motor/predictive_current.hpp#L13) | block | Motor control | PMSM electrical nameplate the predictive controller inverts |
| [`PmsmParameterEstimator`](inc/wet/motor/predictive_current.hpp#L125) | block | Motor control | Online PMSM electrical-parameter estimator (linear Kalman filter) |
| [`PolarMap`](inc/wet/kinematics/motion_maps.hpp#L85) | block | Kinematics | Polar / R-θ mapping (radius + angle ↔ Cartesian X/Y) |
| [`pole`](inc/wet/matlab.hpp#L640) | function | MATLAB-style aliases (host) | MATLAB short alias for the open-loop poles of a system |
| [`poles`](inc/wet/analysis/analysis.hpp#L814) | function | Frequency-domain analysis (host) | Compute open-loop poles (eigenvalues of A matrix) |
| [`PoleZeroMap`](inc/wet/analysis/analysis.hpp#L1345) | block | Frequency-domain analysis (host) | Poles and zeros of a system, for pole-zero plotting |
| [`poly_horner`](inc/wet/toolbox/scaling.hpp#L105) | function | Utilities & toolbox | Evaluate a polynomial at x by Horner's method |
| [`poly_roots`](inc/wet/analysis/analysis.hpp#L1358) | function | Frequency-domain analysis (host) | Roots of a polynomial given in ascending powers (MATLAB `roots`, reversed order) |
| [`PolynomialTrajectory`](inc/wet/trajectory/polynomial.hpp#L204) | block | Trajectory & motion planning | Runtime evaluator for a precomputed polynomial trajectory |
| [`PolyTrajectory`](inc/wet/trajectory/polynomial.hpp#L59) | block | Trajectory & motion planning | A synthesized polynomial trajectory: the coefficients of p(t) = Σ cᵢ·tⁱ over t ∈ [0, T], plus the duration |
| [`Pose`](inc/wet/kinematics/pose.hpp#L50) | block | Kinematics | Rigid-body pose: a translation and an orientation (unit quaternion) |
| [`positive_sequence_ab`](inc/wet/filters/pll.hpp#L349) | function | Filters & signal conditioning | Instantaneous positive-sequence αβ from a quadrature signal pair |
| [`pow`](inc/wet/matrix/functions.hpp#L543) | function | Linear algebra | Matrix power for real exponent |
| [`pow`](inc/wet/math/math.hpp#L245) | function | Scalar math & complex | Power function, base^exponent |
| [`power`](inc/wet/matrix/functions.hpp#L500) | function | Linear algebra | Matrix power for integer exponent |
| [`pr`](inc/wet/controllers/pr.hpp#L114) | function | Runtime controllers | Design a Proportional-Resonant controller |
| [`pr_harmonics`](inc/wet/controllers/pr.hpp#L130) | function | Runtime controllers | Design multiple-harmonic PR controller gains |
| [`PRBS`](inc/wet/estimation/excitation.hpp#L731) | block | Observers & estimators | Maximal-length PRBS runtime generator |
| [`PRBSConfig`](inc/wet/estimation/excitation.hpp#L247) | block | Observers & estimators | Configuration for maximal-length pseudo-random binary excitation |
| [`PRBSResult`](inc/wet/estimation/excitation.hpp#L291) | block | Observers & estimators | PRBS design payload |
| [`PRController`](inc/wet/controllers/pr.hpp#L159) | block | Runtime controllers | Discrete Proportional-Resonant Controller |
| [`PredictiveCurrentController`](inc/wet/motor/predictive_current.hpp#L24) | block | Motor control | Deadbeat (one-step predictive) dq current controller — an alternative to the PI FOController current loop |
| [`pseudo_inverse`](inc/wet/matrix/svd.hpp#L282) | function | Linear algebra | Moore–Penrose pseudoinverse A⁺ via SVD |
| [`PulseTimer`](inc/wet/toolbox/logic.hpp#L159) | block | Utilities & toolbox | Pulse timer (non-retriggerable): a rising edge of in emits a fixed |
| [`pzmap`](inc/wet/analysis/analysis.hpp#L1386) | function | Frequency-domain analysis (host) | Pole-zero map of a SISO transfer function (MATLAB `pzmap(tf)`) |
| [`pzplot`](inc/wet/simulation/plot_plotly.hpp#L444) | function | Simulation (host) | Plot a pole-zero map on the complex plane (poles as ×, zeros as ○) |
| [`qr_decompose`](inc/wet/matrix/decomposition.hpp#L198) | function | Linear algebra | Perform QR decomposition on a matrix |
| [`QRDecomposition`](inc/wet/matrix/decomposition.hpp#L173) | block | Linear algebra | QR decomposition via Gram-Schmidt orthogonalization |
| [`QuadratureDecoder`](inc/wet/toolbox/encoder.hpp#L47) | block | Utilities & toolbox | Software A/B quadrature decoder with optional index |
| [`R_TRIG`](inc/wet/toolbox/iec61131.hpp#L103) | block | Utilities & toolbox | R_TRIG (Rising Edge Trigger) |
| [`rad2deg`](inc/wet/math/math.hpp#L426) | function | Scalar math & complex | Radians to degrees, rad·180/π |
| [`Ramp`](inc/wet/estimation/excitation.hpp#L966) | block | Observers & estimators | Rate-limited ramp runtime generator |
| [`RampConfig`](inc/wet/estimation/excitation.hpp#L418) | block | Observers & estimators | Configuration for a slew-rate-limited ramp excitation |
| [`RampResult`](inc/wet/estimation/excitation.hpp#L451) | block | Observers & estimators | Ramp design payload |
| [`RangeMonitor`](inc/wet/toolbox/conditioning.hpp#L268) | block | Utilities & toolbox | Analog-input range/fault monitor (NAMUR NE43 pattern) |
| [`rank`](inc/wet/design/stability.hpp#L147) | function | Design & synthesis | Compute rank of a matrix via Gaussian elimination with partial pivoting |
| [`rank`](inc/wet/matrix/functions.hpp#L199) | function | Linear algebra | Matrix rank via Gaussian elimination with partial pivoting |
| [`rank_from_svd`](inc/wet/matrix/svd.hpp#L260) | function | Linear algebra | Numerical rank from a precomputed SVD result |
| [`ReducedObserverResult`](inc/wet/estimation/observer.hpp#L230) | block | Observers & estimators | Reduced-order (Gopinath) observer design result |
| [`ReducedOrderObserver`](inc/wet/estimation/observer.hpp#L492) | block | Observers & estimators | Reduced-order (Gopinath) state observer (runtime) |
| [`reg`](inc/wet/matlab.hpp#L365) | function | MATLAB-style aliases (host) | Form dynamic regulator from system, state-feedback gain, and estimator gain |
| [`RelayAutotuneConfig`](inc/wet/estimation/relay_autotune.hpp#L102) | block | Observers & estimators | Configuration for the relay-feedback autotuning experiment |
| [`RelayAutotuneOutput`](inc/wet/estimation/relay_autotune.hpp#L206) | block | Observers & estimators | Per-tick output of RelayAutotuner::step |
| [`RelayAutotuner`](inc/wet/estimation/relay_autotune.hpp#L221) | block | Observers & estimators | Runtime relay-feedback autotuner |
| [`RelayAutotuneResult`](inc/wet/estimation/relay_autotune.hpp#L150) | block | Observers & estimators | Relay-autotuner design payload |
| [`reorder_schur`](inc/wet/design/riccati.hpp#L423) | function | Design & synthesis | Reorder a real Schur form so eigenvalues satisfying in_front lead |
| [`RepetitiveConfig`](inc/wet/controllers/repetitive.hpp#L54) | block | Runtime controllers | Repetitive-controller tuning + period (with optional zero-phase FIR Q) |
| [`RepetitiveController`](inc/wet/controllers/repetitive.hpp#L248) | block | Runtime controllers | Plug-in repetitive controller runtime (fixed-size internal model) |
| [`requires`](inc/wet/filters/filters.hpp#L203) | function | Filters & signal conditioning | Butterworth low-pass filter design |
| [`requires`](inc/wet/matrix/matrix.hpp#L864) | function | Linear algebra | Symmetric congruence (quadratic) form  S = M X Mᵀ |
| [`requires`](inc/wet/motor/field_weakening.hpp#L170) | function | Motor control | Concept for a pluggable field-weakening / current-reference policy |
| [`requires`](inc/wet/estimation/ekf.hpp#L42) | function | Observers & estimators | Concept for EKF state functions |
| [`rescale`](inc/wet/toolbox/scaling.hpp#L45) | function | Utilities & toolbox | Affine map of x from the input range to the output range |
| [`ResistiveLossModel`](inc/wet/motor/thermal.hpp#L158) | block | Motor control | Minimal conduction-only loss model for a weak datasheet |
| [`Resonator`](inc/wet/filters/pll.hpp#L399) | block | Filters & signal conditioning | Dual-SOGI three-phase positive-sequence PLL (DSOGI-PLL) |
| [`RobustExactDifferentiator`](inc/wet/filters/differentiator.hpp#L49) | block | Filters & signal conditioning | First-order robust exact differentiator (super-twisting differentiator) |
| [`rotary_gearbox`](inc/wet/toolbox/actuator.hpp#L131) | function | Utilities & toolbox | Build a ServoAxis for a rotary joint behind a gearbox |
| [`RotaryDelta`](inc/wet/kinematics/motion_maps.hpp#L146) | block | Kinematics | Rotary delta robot — closed-form inverse, quadratic-intersection forward |
| [`RotaryDeltaGeometry`](inc/wet/kinematics/motion_maps.hpp#L129) | block | Kinematics | Rotary delta geometry (three base servos, parallelogram arms) |
| [`rotational_load_ss`](inc/wet/motor/mechanical_estimator.hpp#L13) | function | Motor control | Continuous state-space model of a 1-DOF rotational drivetrain with an augmented load-torque state |
| [`RowVec`](inc/wet/matrix/rowvec.hpp#L9) | block | Linear algebra | Row vector specialization of Matrix<1, N, T> |
| [`RowView`](inc/wet/matrix/views.hpp#L148) | block | Linear algebra | Non-owning row view of a matrix |
| [`RS`](inc/wet/toolbox/iec61131.hpp#L75) | block | Utilities & toolbox | RS Latch (Reset-Set Latch) |
| [`scaled_deadband`](inc/wet/toolbox/conditioning.hpp#L82) | function | Utilities & toolbox | Center dead zone that rescales the surviving range back to full span |
| [`scara_arm`](inc/wet/kinematics/scara.hpp#L243) | function | Kinematics | Build a series SCARA (RRPR) as a 4-joint DH chain |
| [`ScurveProfile`](inc/wet/trajectory/scurve.hpp#L60) | block | Trajectory & motion planning | A synthesized jerk-limited (double-S) profile: a sequence of constant-jerk segments, evaluated exactly (cubic in t within a segment) |
| [`ScurveTrajectory`](inc/wet/trajectory/scurve.hpp#L283) | block | Trajectory & motion planning | Runtime evaluator for a precomputed jerk-limited (double-S) profile |
| [`select_nearest`](inc/wet/kinematics/serial_arm.hpp#L411) | function | Kinematics | Pick the solution branch nearest a reference configuration |
| [`SensorlessEstimator`](inc/wet/filters/pll.hpp#L189) | block | Filters & signal conditioning | Sensorless rotor flux/position estimator for a PMSM, with optional sensor fusion |
| [`SequenceComponents`](inc/wet/transforms.hpp#L593) | block | Motor control | Symmetrical (sequence) components of a three-phase phasor set |
| [`SerialArm`](inc/wet/kinematics/serial_arm.hpp#L168) | block | Kinematics | Serial N-DOF revolute manipulator runtime |
| [`SerialArmConfig`](inc/wet/kinematics/serial_arm.hpp#L151) | block | Kinematics | Validated serial-arm configuration (the design payload) |
| [`series`](inc/wet/systems/state_space.hpp#L121) | function | LTI models | Series connection of two state-space systems |
| [`ServoAxis`](inc/wet/toolbox/actuator.hpp#L90) | block | Utilities & toolbox | One servoactuator transmission: SI joint unit ⟷ drive (motor) units |
| [`ServoBank`](inc/wet/toolbox/actuator.hpp#L203) | block | Utilities & toolbox | A bank of ServoAxis transmissions: maps a synchronized multi-axis |
| [`ServoCommand`](inc/wet/toolbox/actuator.hpp#L68) | block | Utilities & toolbox | A drive-native servoactuator setpoint: position, velocity, torque |
| [`ServoFeedback`](inc/wet/motor/servo.hpp#L68) | block | Motor control | Sensor feedback for one PmacServo::update tick |
| [`sgn`](inc/wet/math/math.hpp#L364) | function | Scalar math & complex | Sign function — −1 if val < 0, 1 if val > 0, 0 if val == 0 |
| [`SignalStatus`](inc/wet/toolbox/conditioning.hpp#L217) | block | Utilities & toolbox | Classification of an analog input against its valid/fault bands |
| [`simc`](inc/wet/design/pid_design.hpp#L236) | function | Design & synthesis | SIMC (Skogestad Internal Model Control) tuning for FOPDT models |
| [`simulate`](inc/wet/simulation/simulate.hpp#L46) | function | Simulation (host) | Simulate a nonlinear plant with a controller in closed loop |
| [`simulate_discrete`](inc/wet/simulation/simulate.hpp#L322) | function | Simulation (host) | Simulate a discrete-time system with a controller |
| [`simulate_discrete_nonlinear`](inc/wet/simulation/simulate.hpp#L258) | function | Simulation (host) | Simulate a discrete-time nonlinear plant with a controller |
| [`simulate_lti`](inc/wet/simulation/simulate.hpp#L227) | function | Simulation (host) | Simulate a continuous LTI system with a controller |
| [`simulate_sampled`](inc/wet/simulation/simulate.hpp#L114) | function | Simulation (host) | Simulate a continuous plant under a discrete (sampled) controller — multi-rate |
| [`simulate_state_feedback`](inc/wet/simulation/simulate.hpp#L173) | function | Simulation (host) | Simulate a nonlinear plant with state-feedback controller |
| [`SimulationResult`](inc/wet/simulation/simulate.hpp#L30) | block | Simulation (host) | Result of a closed-loop simulation |
| [`sin`](inc/wet/matrix/functions.hpp#L647) | function | Linear algebra | Matrix sine via scaling and double-angle reconstruction |
| [`sin`](inc/wet/math/math.hpp#L167) | function | Scalar math & complex | Sine |
| [`sincos`](inc/wet/matrix/functions.hpp#L567) | function | Linear algebra | Compute sin(A) and cos(A) together via scaling and double-angle reconstruction |
| [`sincos`](inc/wet/math/math.hpp#L180) | function | Scalar math & complex | Combined sine and cosine, {sin(x), cos(x)} |
| [`SinglePhasePLL`](inc/wet/filters/pll.hpp#L11) | block | Filters & signal conditioning | Single-Phase PLL |
| [`sinh`](inc/wet/matrix/functions.hpp#L717) | function | Linear algebra | Matrix hyperbolic sine |
| [`SlewLimiter`](inc/wet/toolbox/conditioning.hpp#L116) | block | Utilities & toolbox | Slew-rate limiter: bound how fast the output may follow the target |
| [`smc`](inc/wet/controllers/smc.hpp#L35) | function | Runtime controllers | Bundle hand-picked SMC parameters into an SMCResult |
| [`SMCController`](inc/wet/controllers/smc.hpp#L63) | block | Runtime controllers | First-order sliding-mode controller (SMC) for a SISO plant |
| [`SMCResult`](inc/wet/controllers/smc.hpp#L9) | block | Runtime controllers | Tuning parameters for a first-order sliding-mode controller |
| [`sogi`](inc/wet/filters/sogi.hpp#L13) | function | Filters & signal conditioning | Second-Order Generalized Integrator (SOGI) design |
| [`SOGI`](inc/wet/filters/sogi.hpp#L154) | block | Filters & signal conditioning | Runtime SOGI wrapper around design::sogi(w0, alpha, Ts) |
| [`SogiFll`](inc/wet/filters/sogi.hpp#L286) | block | Filters & signal conditioning | SOGI with a Frequency-Locked Loop — self-tuning single-tone tracker |
| [`solve`](inc/wet/matrix/solve.hpp#L63) | function | Linear algebra | Solve lower-triangular system L X = B via forward substitution |
| [`solve_lyapunov_kron`](inc/wet/design/lyapunov.hpp#L13) | function | Design & synthesis | Solve a linear matrix equation L(X) + Q = 0 by Kronecker vectorization |
| [`SolveResult`](inc/wet/simulation/solver.hpp#L28) | block | Simulation (host) | Result of an ODE solve operation |
| [`SplineProfile`](inc/wet/trajectory/spline.hpp#L52) | block | Trajectory & motion planning | A synthesized multi-waypoint spline: per-segment polynomial coefficients (ascending power, in segment-local time) plus the knot times |
| [`SplineTrajectory`](inc/wet/trajectory/spline.hpp#L258) | block | Trajectory & motion planning | Runtime player for a multi-waypoint spline (design::SplineProfile) |
| [`split_real_2x2`](inc/wet/design/riccati.hpp#L299) | function | Design & synthesis | Split a real-eigenvalue 2×2 Schur block into two 1×1 blocks |
| [`sqrt`](inc/wet/matrix/functions.hpp#L444) | function | Linear algebra | Matrix square root via Denman–Beavers iteration |
| [`sqrt`](inc/wet/math/complex.hpp#L321) | function | Scalar math & complex | Compute complex square root (constexpr) |
| [`SR`](inc/wet/toolbox/iec61131.hpp#L42) | block | Utilities & toolbox | SR Latch (Set-dominant Set-Reset Latch) |
| [`ss`](inc/wet/matlab.hpp#L65) | function | MATLAB-style aliases (host) | MATLAB-style state-space model constructor |
| [`stability_margin_continuous`](inc/wet/design/stability.hpp#L244) | function | Design & synthesis | Compute stability margin for continuous system |
| [`stability_margin_discrete`](inc/wet/design/stability.hpp#L273) | function | Design & synthesis | Compute stability margin for discrete system |
| [`StateJacobian`](inc/wet/estimation/ekf.hpp#L26) | block | Observers & estimators | State prediction result from the user's dynamics function |
| [`StateSpace`](inc/wet/systems/state_space.hpp#L27) | block | LTI models | State-space representation for linear time-invariant systems (discrete or continuous) |
| [`steinhart_hart`](inc/wet/toolbox/thermistor.hpp#L81) | function | Utilities & toolbox | Fit the Steinhart-Hart coefficients from three calibration points |
| [`step`](inc/wet/analysis/analysis.hpp#L961) | function | Frequency-domain analysis (host) | Step response of a (MIMO) state-space system |
| [`StepInfo`](inc/wet/analysis/analysis.hpp#L1136) | block | Frequency-domain analysis (host) | Step-response characteristics of a single output signal |
| [`stepinfo`](inc/wet/analysis/analysis.hpp#L1155) | function | Frequency-domain analysis (host) | Compute step-response characteristics from an output/time signal |
| [`stepplot`](inc/wet/simulation/plot_plotly.hpp#L311) | function | Simulation (host) | Plot a step response, one trace per input/output pair |
| [`StepTrain`](inc/wet/estimation/excitation.hpp#L878) | block | Observers & estimators | Alternating +/- step train runtime generator |
| [`StepTrainConfig`](inc/wet/estimation/excitation.hpp#L342) | block | Observers & estimators | Configuration for alternating +/- step excitation |
| [`StepTrainResult`](inc/wet/estimation/excitation.hpp#L375) | block | Observers & estimators | Step-train design payload |
| [`stewart_symmetric`](inc/wet/kinematics/stewart.hpp#L326) | function | Kinematics | Tier-2 builder for the common symmetric hexagonal layout |
| [`StewartConfig`](inc/wet/kinematics/stewart.hpp#L103) | block | Kinematics | Validated Stewart configuration (the design payload) |
| [`StewartGeometry`](inc/wet/kinematics/stewart.hpp#L48) | block | Kinematics | Rig geometry: the six fixed base anchors `bᵢ`, the six moving-platform anchors `pᵢ`, the actuator stroke limits, and the nominal home height |
| [`StewartPlatform`](inc/wet/kinematics/stewart.hpp#L136) | block | Kinematics | Gough–Stewart platform runtime — closed-form inverse, Newton forward |
| [`Stopwatch`](inc/wet/toolbox/timing.hpp#L26) | block | Utilities & toolbox | Free-running elapsed-time accumulator |
| [`stsmc`](inc/wet/controllers/stsmc.hpp#L103) | function | Runtime controllers | Super-twisting controller from gains you specify directly |
| [`STSMCResult`](inc/wet/controllers/stsmc.hpp#L14) | block | Runtime controllers | Super-twisting (second-order sliding-mode) controller design result |
| [`subtract`](inc/wet/systems/state_space.hpp#L323) | function | LTI models | Subtraction/differencing connection of two state-space systems |
| [`SuperTwistingController`](inc/wet/controllers/stsmc.hpp#L132) | block | Runtime controllers | Super-twisting controller (second-order sliding mode) |
| [`svd`](inc/wet/matrix/svd.hpp#L230) | function | Linear algebra | Full singular value decomposition A = U·Σ·Vᴴ (one-sided Jacobi) |
| [`svd`](inc/wet/matlab.hpp#L291) | function | MATLAB-style aliases (host) | MATLAB short alias for the singular value decomposition |
| [`SVDResult`](inc/wet/matrix/svd.hpp#L215) | block | Linear algebra | Result of a full singular value decomposition A = U·Σ·Vᴴ |
| [`svm_duty_cycles`](inc/wet/motor/modulation.hpp#L75) | function | Motor control | Space-vector PWM duty cycles from an αβ voltage command |
| [`SvmDuties`](inc/wet/motor/modulation.hpp#L57) | block | Motor control | Result of svm_duty_cycles(): the half-bridge duties plus an over-modulation flag |
| [`svpwm_zero_sequence`](inc/wet/motor/modulation.hpp#L26) | function | Motor control | Min-max zero-sequence injection for space-vector PWM |
| [`swap_schur_blocks`](inc/wet/design/riccati.hpp#L359) | function | Design & synthesis | Swap two adjacent diagonal blocks of a real Schur form |
| [`Switch`](inc/wet/toolbox/io.hpp#L162) | block | Utilities & toolbox | Debounced maintained switch (toggle/selector contact) with change flag |
| [`symmetrical_components`](inc/wet/transforms.hpp#L610) | function | Motor control | Forward symmetrical-component (Fortescue) transform (abc → 012) |
| [`synthesize_chirp`](inc/wet/estimation/excitation.hpp#L235) | function | Observers & estimators | Build a chirp design payload from a configuration |
| [`synthesize_classical_dob`](inc/wet/estimation/disturbance_observer.hpp#L327) | function | Observers & estimators | Synthesize a classical disturbance observer from a nominal plant and Q-filter |
| [`synthesize_esc`](inc/wet/controllers/esc.hpp#L133) | function | Runtime controllers | Synthesize an extremum-seeking controller |
| [`synthesize_esc_mppt`](inc/wet/controllers/esc.hpp#L184) | function | Runtime controllers | MPPT-flavored ESC: maximize a power measurement by perturbing the operating point (e.g. converter duty or reference voltage) |
| [`synthesize_harmonic_suppressor`](inc/wet/controllers/harmonic_suppression.hpp#L58) | function | Runtime controllers | Synthesize a multi-resonant harmonic suppressor |
| [`synthesize_input_shaper`](inc/wet/trajectory/input_shaper.hpp#L96) | function | Trajectory & motion planning | Synthesize an input shaper for a second-order mode |
| [`synthesize_multi_sine`](inc/wet/estimation/excitation.hpp#L595) | function | Observers & estimators | Build a multi-sine design payload from a configuration |
| [`synthesize_observer`](inc/wet/estimation/observer.hpp#L111) | function | Observers & estimators | Design a Luenberger observer by pole placement (matrix form) |
| [`synthesize_poly_trajectory`](inc/wet/trajectory/polynomial.hpp#L117) | function | Trajectory & motion planning | Synthesize a fixed-duration polynomial trajectory matching boundary conditions on position and its derivatives at both endpoints |
| [`synthesize_prbs`](inc/wet/estimation/excitation.hpp#L326) | function | Observers & estimators | Build a PRBS design payload from a configuration |
| [`synthesize_ramp`](inc/wet/estimation/excitation.hpp#L482) | function | Observers & estimators | Build a ramp design payload from a configuration |
| [`synthesize_reduced_observer`](inc/wet/estimation/observer.hpp#L282) | function | Observers & estimators | Design a reduced-order (Gopinath) observer by pole placement (matrix form) |
| [`synthesize_relay_autotune`](inc/wet/estimation/relay_autotune.hpp#L181) | function | Observers & estimators | Build a validated relay-autotune design payload |
| [`synthesize_repetitive`](inc/wet/controllers/repetitive.hpp#L140) | function | Runtime controllers | Synthesize a repetitive controller with a scalar robustness filter Q |
| [`synthesize_repetitive_binomial`](inc/wet/controllers/repetitive.hpp#L200) | function | Runtime controllers | Synthesize a repetitive controller with a binomial zero-phase FIR Q |
| [`synthesize_scurve`](inc/wet/trajectory/scurve.hpp#L164) | function | Trajectory & motion planning | Synthesize a minimum-time jerk-limited (7-segment double-S) profile from (Xi, Vi) to (Xf, Vf) under asymmetric kinematic limits |
| [`synthesize_serial_arm`](inc/wet/kinematics/serial_arm.hpp#L453) | function | Kinematics | Validate a serial-arm DH chain and flag a spherical wrist |
| [`synthesize_spline`](inc/wet/trajectory/spline.hpp#L128) | function | Trajectory & motion planning | Synthesize a multi-waypoint spline through points at times |
| [`synthesize_step_train`](inc/wet/estimation/excitation.hpp#L406) | function | Observers & estimators | Build a step-train design payload from a configuration |
| [`synthesize_stewart`](inc/wet/kinematics/stewart.hpp#L298) | function | Kinematics | Validate a hand-entered Stewart geometry and confirm the home pose is reachable |
| [`synthesize_stsmc`](inc/wet/controllers/stsmc.hpp#L42) | function | Runtime controllers | Synthesize super-twisting gains from a disturbance-derivative bound |
| [`synthesize_trapezoidal`](inc/wet/trajectory/trapezoidal.hpp#L183) | function | Trajectory & motion planning | Synthesize the minimum-time asymmetric trapezoidal profile from (Xi, Vi) to (Xf, Vf) under the given limits |
| [`Tachometer`](inc/wet/toolbox/encoder.hpp#L126) | block | Utilities & toolbox | Pulse-based speed (tachometer) with frequency/period crossover |
| [`tan`](inc/wet/math/math.hpp#L198) | function | Scalar math & complex | Tangent |
| [`tf`](inc/wet/matlab.hpp#L32) | function | MATLAB-style aliases (host) | MATLAB-style transfer function constructor |
| [`TFF`](inc/wet/toolbox/iec61131.hpp#L501) | block | Utilities & toolbox | T Flip-Flop (toggle on rising edge) |
| [`ThermalLimiter`](inc/wet/motor/thermal.hpp#L269) | block | Motor control | Derates the current command from a temperature (Tj for FETs, winding for the motor) |
| [`ThermalLimits`](inc/wet/motor/thermal.hpp#L233) | block | Motor control | A derating curve plus a hard fault threshold |
| [`ThermalState`](inc/wet/motor/thermal.hpp#L259) | block | Motor control | State from a ThermalLimiter evaluation |
| [`Thermistor`](inc/wet/toolbox/thermistor.hpp#L138) | block | Utilities & toolbox | NTC thermistor linearization (resistance → temperature) |
| [`ThermistorCoeffs`](inc/wet/toolbox/thermistor.hpp#L25) | block | Utilities & toolbox | Fitted NTC coefficients in Steinhart-Hart form |
| [`ThreePhasePLL`](inc/wet/filters/pll.hpp#L113) | block | Filters & signal conditioning | Synchronous-reference-frame (SRF) PLL for balanced three-phase input |
| [`time_response_figure`](inc/wet/simulation/plot_plotly.hpp#L240) | function | Simulation (host) | Build a time-response figure with one line per (output, input) pair |
| [`Timeout`](inc/wet/toolbox/timing.hpp#L56) | block | Utilities & toolbox | One-shot timeout |
| [`TimeResponse`](inc/wet/analysis/analysis.hpp#L884) | block | Frequency-domain analysis (host) | Multi-channel time-domain response sampled on a time grid |
| [`to_coeffs`](inc/wet/filters/filters.hpp#L335) | function | Filters & signal conditioning | Convert StateSpace system to first-order DSP coefficients |
| [`TOF`](inc/wet/toolbox/iec61131.hpp#L210) | block | Utilities & toolbox | TOF Timer (Timer Off Delay) |
| [`TON`](inc/wet/toolbox/iec61131.hpp#L167) | block | Utilities & toolbox | TON Timer (Timer On Delay) |
| [`Tone`](inc/wet/estimation/excitation.hpp#L494) | block | Observers & estimators | One sinusoidal component in a multi-sine excitation |
| [`ToppMove`](inc/wet/trajectory/topp.hpp#L103) | block | Trajectory & motion planning | Time-optimal task-space move (path-preserving, pointwise minimum-time) |
| [`ToppProfile`](inc/wet/trajectory/topp.hpp#L50) | block | Trajectory & motion planning | The scalar time-optimal path-timing produced by TOPP |
| [`torque_constant_from_flux`](inc/wet/motor/foc.hpp#L56) | function | Motor control | Torque constant @f$ K_t @f$ of a PMSM (amplitude-invariant convention) |
| [`torque_constant_from_Kv`](inc/wet/motor/foc.hpp#L99) | function | Motor control | Torque constant from the datasheet velocity constant @f$ K_v @f$ |
| [`TP`](inc/wet/toolbox/iec61131.hpp#L255) | block | Utilities & toolbox | TP Timer (Timer Pulse) |
| [`TrajectoryBank`](inc/wet/trajectory/polynomial.hpp#L253) | block | Trajectory & motion planning | Multi-axis coordination: time-scale each axis's profile to the slowest so a multi-DOF move starts and finishes synchronized ("linear" / coordinated joint moves — the feedforward reference for a manipulator) |
| [`TrajectoryBoundary`](inc/wet/trajectory/trajectory_types.hpp#L61) | block | Trajectory & motion planning | Boundary conditions at one endpoint of a polynomial trajectory: a position and its time derivatives |
| [`TrajectoryLimits`](inc/wet/trajectory/trajectory_types.hpp#L23) | block | Trajectory & motion planning | Asymmetric kinematic limits for a trapezoidal or S-curve motion profile |
| [`TrajectoryState`](inc/wet/trajectory/trajectory_types.hpp#L48) | block | Trajectory & motion planning | A point on a motion profile: commanded position, velocity, acceleration |
| [`Translation3`](inc/wet/kinematics/pose.hpp#L27) | block | Kinematics | A 3-D translation — a thin Vec3 with domain-named conveniences |
| [`TransposeView`](inc/wet/matrix/views.hpp#L280) | block | Linear algebra | Non-owning transpose view of a matrix (zero-copy) |
| [`Trapezoidal`](inc/wet/simulation/integrator.hpp#L277) | block | Simulation (host) | Trapezoidal (Tustin) integrator |
| [`TrapezoidalProfile`](inc/wet/trajectory/trapezoidal.hpp#L41) | block | Trajectory & motion planning | Planned trapezoidal profile: the segment durations, reached values, and boundary state needed to evaluate the trajectory at any time |
| [`TrapezoidalTrajectory`](inc/wet/trajectory/trapezoidal.hpp#L271) | block | Trajectory & motion planning | Runtime evaluator for a precomputed trapezoidal profile |
| [`two_norm`](inc/wet/matrix/functions.hpp#L74) | function | Linear algebra | Spectral norm (2-norm): largest singular value of A |
| [`two_point_cal`](inc/wet/toolbox/scaling.hpp#L89) | function | Utilities & toolbox | Fit an AffineCal through two `(raw, engineering)` points |
| [`tyreus_luyben`](inc/wet/design/pid_design.hpp#L143) | function | Design & synthesis | Tyreus-Luyben tuning from ultimate gain and ultimate period |
| [`UnscentedKalmanFilter`](inc/wet/estimation/ukf.hpp#L82) | block | Observers & estimators | Unscented (sigma-point) Kalman Filter for nonlinear discrete-time systems |
| [`UnscentedParams`](inc/wet/estimation/ukf.hpp#L64) | block | Observers & estimators | Tuning parameters for the scaled unscented transform |
| [`unwrap_phase_deg`](inc/wet/analysis/analysis.hpp#L174) | function | Frequency-domain analysis (host) | Unwrap phase data in degrees to avoid +/-180 discontinuities |
| [`UpperTriangle`](inc/wet/matrix/views.hpp#L94) | block | Linear algebra | Upper triangular view of a square matrix |
| [`voltage_circle_radius`](inc/wet/motor/foc.hpp#L195) | function | Motor control | Radius of the SVPWM voltage circle (max synthesizable @f$ \|V_{dq}\| @f$) |
| [`wrap`](inc/wet/math/math.hpp#L448) | function | Scalar math & complex | Wrap x into the half-open interval [min, max) (period max − min) |
| [`wrapped_delta`](inc/wet/toolbox/encoder.hpp#L23) | function | Utilities & toolbox | Signed difference between two unsigned counter readings, wrap-safe |
| [`ziegler_nichols`](inc/wet/design/pid_design.hpp#L51) | function | Design & synthesis | Ziegler-Nichols tuning from ultimate gain and ultimate period |
| [`ziegler_nichols_step`](inc/wet/design/pid_design.hpp#L94) | function | Design & synthesis | Ziegler-Nichols step response method (reaction curve) |
| [`ZPK`](inc/wet/systems/zpk.hpp#L12) | block | LTI models | Zero-pole-gain (ZPK) representation of a SISO LTI system |
| [`zpk`](inc/wet/matlab.hpp#L87) | function | MATLAB-style aliases (host) | MATLAB-style zero-pole-gain model constructor |
