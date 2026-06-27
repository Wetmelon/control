# API Reference

Auto-generated from `@brief` doc comments in `inc/wet/`. Regenerate with `python tools/gen_reference.py`.


- [API Reference](#api-reference)
  - [Core, configuration \& backend vocabulary](#core-configuration--backend-vocabulary)
  - [Scalar math \& complex](#scalar-math--complex)
  - [Linear algebra](#linear-algebra)
  - [LTI models](#lti-models)
  - [Runtime controllers](#runtime-controllers)
  - [Design \& synthesis](#design--synthesis)
  - [Observers \& estimators](#observers--estimators)
  - [Filters \& signal conditioning](#filters--signal-conditioning)
  - [Trajectory \& motion planning](#trajectory--motion-planning)
  - [Kinematics](#kinematics)
  - [Motor control](#motor-control)
  - [Utilities \& toolbox](#utilities--toolbox)
  - [Frequency-domain analysis (host)](#frequency-domain-analysis-host)
  - [Simulation (host)](#simulation-host)
  - [MATLAB-style aliases (host)](#matlab-style-aliases-host)
  - [Math backends](#math-backends)
  - [Examples](#examples)

## Core, configuration & backend vocabulary

**Functions**

| Name | Description |
| ---- | ----------- |
| [`minmax`](inc/wet/backend.hpp#L139) | Ordered {min, max} pair returned by value |

## Scalar math & complex

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`complex`](inc/wet/math/complex.hpp#L18) | Constexpr complex number class for compile-time computations |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`abs`](inc/wet/math/complex.hpp#L353) | Compute magnitude (absolute value) of a complex number |
| [`acos`](inc/wet/math/math.hpp#L136) | Arccosine ∈ [0, π]. Input is clamped to [−1, 1] in both paths |
| [`arg`](inc/wet/math/complex.hpp#L364) | Compute argument (phase angle) of a complex number |
| [`asin`](inc/wet/math/math.hpp#L115) | Arcsine ∈ [−π/2, π/2]. Input is clamped to [−1, 1] in both paths so behavior matches at compile and run time (std::asin would return NaN for \|x\| > 1) |
| [`atan`](inc/wet/math/math.hpp#L103) | Single-argument arctangent ∈ (−π/2, π/2) |
| [`atan2`](inc/wet/math/math.hpp#L90) | Two-argument arctangent, atan2(y, x) ∈ [−π, π] |
| [`cbrt`](inc/wet/math/math.hpp#L78) | Cube root (preserves sign for negative x) |
| [`ceil`](inc/wet/math/math.hpp#L304) | Ceiling — smallest integer ≥ x |
| [`copysign`](inc/wet/math/math.hpp#L375) | Copy sign — magnitude of mag with the sign of sgn_src |
| [`cos`](inc/wet/math/math.hpp#L154) | Cosine |
| [`db2mag`](inc/wet/math/math.hpp#L415) | Decibels to magnitude, 10^(db/20) |
| [`deg2rad`](inc/wet/math/math.hpp#L437) | Degrees to radians, deg·π/180 |
| [`exp`](inc/wet/math/math.hpp#L212) | Exponential function |
| [`floor`](inc/wet/math/math.hpp#L292) | Floor — largest integer ≤ x |
| [`fmod`](inc/wet/math/math.hpp#L330) | Floating-point remainder, x − y·trunc(x/y) (sign of x), matching std::fmod's truncated-quotient convention |
| [`hypot`](inc/wet/math/math.hpp#L56) | Euclidean distance hypot(x, y) = √(x² + y²), without overflow |
| [`isfinite`](inc/wet/math/math.hpp#L387) | Finiteness test — false for NaN and ±∞ |
| [`log`](inc/wet/math/math.hpp#L229) | Natural logarithm |
| [`log10`](inc/wet/math/math.hpp#L349) | Base-10 logarithm, log10(x) = ln(x) / ln(10) |
| [`mag2db`](inc/wet/math/math.hpp#L404) | Magnitude to decibels, 20·log10(mag) |
| [`nearbyint`](inc/wet/math/math.hpp#L316) | Round to nearest integer. Runtime follows the backend (round half to even); the compile-time path rounds ties away from zero — immaterial for range reduction |
| [`pow`](inc/wet/math/math.hpp#L245) | Power function, base^exponent |
| [`rad2deg`](inc/wet/math/math.hpp#L426) | Radians to degrees, rad·180/π |
| [`sgn`](inc/wet/math/math.hpp#L364) | Sign function — −1 if val < 0, 1 if val > 0, 0 if val == 0 |
| [`sin`](inc/wet/math/math.hpp#L167) | Sine |
| [`sincos`](inc/wet/math/math.hpp#L180) | Combined sine and cosine, {sin(x), cos(x)} |
| [`sqrt`](inc/wet/math/complex.hpp#L321) | Compute complex square root (constexpr) |
| [`tan`](inc/wet/math/math.hpp#L198) | Tangent |
| [`wrap`](inc/wet/math/math.hpp#L448) | Wrap x into the half-open interval [min, max) (period max − min) |

## Linear algebra

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`Block`](inc/wet/matrix/block.hpp#L17) | Block view (non-owning) into a parent matrix |
| [`ColVec`](inc/wet/matrix/colvec.hpp#L9) | Concrete Column vector specialization of Matrix<N, 1, T> |
| [`ColView`](inc/wet/matrix/views.hpp#L215) | Non-owning column view of a matrix |
| [`Diagonal`](inc/wet/matrix/views.hpp#L29) | Diagonal view of a square matrix |
| [`EigenResult`](inc/wet/matrix/eigen.hpp#L13) | Eigenvalue computation result |
| [`FullQR`](inc/wet/matrix/decomposition.hpp#L249) | Result of a full (complete) QR factorization |
| [`LowerTriangle`](inc/wet/matrix/views.hpp#L121) | Lower triangular view of a square matrix |
| [`Matrix`](inc/wet/matrix/matrix.hpp#L49) | Fixed-size, stack-allocated matrix for linear algebra operations |
| [`QRDecomposition`](inc/wet/matrix/decomposition.hpp#L172) | QR decomposition via Gram-Schmidt orthogonalization |
| [`RowVec`](inc/wet/matrix/rowvec.hpp#L9) | Row vector specialization of Matrix<1, N, T> |
| [`RowView`](inc/wet/matrix/views.hpp#L148) | Non-owning row view of a matrix |
| [`SVDResult`](inc/wet/matrix/svd.hpp#L215) | Result of a full singular value decomposition A = U·Σ·Vᴴ |
| [`TransposeView`](inc/wet/matrix/views.hpp#L280) | Non-owning transpose view of a matrix (zero-copy) |
| [`UpperTriangle`](inc/wet/matrix/views.hpp#L94) | Upper triangular view of a square matrix |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`cholesky`](inc/wet/matrix/decomposition.hpp#L59) | Cholesky decomposition for positive-definite matrices |
| [`compute_eigenvalues`](inc/wet/matrix/eigen.hpp#L340) | Compute the eigenvalues (and Schur vectors) of a real square matrix |
| [`cos`](inc/wet/matrix/functions.hpp#L662) | Matrix cosine via scaling and double-angle reconstruction |
| [`cosh`](inc/wet/matrix/functions.hpp#L732) | Matrix hyperbolic cosine |
| [`det`](inc/wet/matrix/functions.hpp#L126) | Matrix determinant |
| [`expm`](inc/wet/matrix/functions.hpp#L247) | Matrix exponential using scaling and squaring with Padé approximation |
| [`forward_substitute`](inc/wet/matrix/solve.hpp#L23) | Forward substitution to solve L x = b |
| [`francis_qr`](inc/wet/matrix/eigen.hpp#L108) | Francis double-shift QR on an upper Hessenberg matrix |
| [`frobenius_norm`](inc/wet/matrix/functions.hpp#L55) | Frobenius norm: square root of sum of squares of all elements |
| [`full_qr`](inc/wet/matrix/decomposition.hpp#L264) | Full QR factorization via Householder reflections (real or complex T) |
| [`hessenberg_reduce`](inc/wet/matrix/eigen.hpp#L32) | Reduce a square matrix to upper Hessenberg form by Householder reflections |
| [`infinity_norm`](inc/wet/matrix/functions.hpp#L11) | Infinity norm: maximum absolute row sum |
| [`jacobi_svd_tall`](inc/wet/matrix/svd.hpp#L45) | One-sided Jacobi SVD of a tall/square matrix A (P×Q, P ≥ Q) |
| [`log`](inc/wet/matrix/functions.hpp#L361) | Matrix logarithm using inverse scaling and squaring |
| [`lu_decomposition`](inc/wet/matrix/decomposition.hpp#L110) | LU decomposition with partial pivoting |
| [`null_space`](inc/wet/matrix/svd.hpp#L331) | Orthonormal basis for the null space {x : A·x = 0} via SVD |
| [`one_norm`](inc/wet/matrix/functions.hpp#L33) | One norm: maximum absolute column sum |
| [`pow`](inc/wet/matrix/functions.hpp#L543) | Matrix power for real exponent |
| [`power`](inc/wet/matrix/functions.hpp#L500) | Matrix power for integer exponent |
| [`pseudo_inverse`](inc/wet/matrix/svd.hpp#L282) | Moore–Penrose pseudoinverse A⁺ via SVD |
| [`qr_decompose`](inc/wet/matrix/decomposition.hpp#L197) | Perform QR decomposition on a matrix |
| [`rank`](inc/wet/matrix/functions.hpp#L199) | Matrix rank via Gaussian elimination with partial pivoting |
| [`rank_from_svd`](inc/wet/matrix/svd.hpp#L260) | Numerical rank from a precomputed SVD result |
| [`requires`](inc/wet/matrix/matrix_traits.hpp#L41) | Concept for any type that provides 2D matrix-like element access |
| [`sin`](inc/wet/matrix/functions.hpp#L647) | Matrix sine via scaling and double-angle reconstruction |
| [`sincos`](inc/wet/matrix/functions.hpp#L567) | Compute sin(A) and cos(A) together via scaling and double-angle reconstruction |
| [`sinh`](inc/wet/matrix/functions.hpp#L717) | Matrix hyperbolic sine |
| [`solve`](inc/wet/matrix/solve.hpp#L63) | Solve lower-triangular system L X = B via forward substitution |
| [`sqrt`](inc/wet/matrix/functions.hpp#L444) | Matrix square root via Denman–Beavers iteration |
| [`svd`](inc/wet/matrix/svd.hpp#L230) | Full singular value decomposition A = U·Σ·Vᴴ (one-sided Jacobi) |
| [`two_norm`](inc/wet/matrix/functions.hpp#L74) | Spectral norm (2-norm): largest singular value of A |

## LTI models

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`DiscretizationMethod`](inc/wet/systems/discretization.hpp#L8) | Discretization methods for continuous-time state-space systems |
| [`StateSpace`](inc/wet/systems/state_space.hpp#L24) | State-space representation for linear time-invariant systems (discrete or continuous) |
| [`ZPK`](inc/wet/systems/zpk.hpp#L12) | Zero-pole-gain (ZPK) representation of a SISO LTI system |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`discretize`](inc/wet/systems/discretization.hpp#L193) | Discretize a continuous-time state-space system |
| [`discretize_forward_euler_impl`](inc/wet/systems/discretization.hpp#L21) | Discretize using Forward Euler (explicit Euler) |
| [`discretize_tustin_impl`](inc/wet/systems/discretization.hpp#L127) | Discretize a continuous-time state-space system using Tustin method |
| [`discretize_zoh_impl`](inc/wet/systems/discretization.hpp#L66) | Discretize using Zero-Order Hold (ZOH) |
| [`eval_frf`](inc/wet/systems/state_space.hpp#L79) | Evaluate frequency response of state-space system |
| [`feedback`](inc/wet/systems/state_space.hpp#L251) | Negative feedback connection of two state-space systems |
| [`parallel`](inc/wet/systems/state_space.hpp#L185) | Parallel connection of two state-space systems |
| [`series`](inc/wet/systems/state_space.hpp#L118) | Series connection of two state-space systems |
| [`subtract`](inc/wet/systems/state_space.hpp#L320) | Subtraction/differencing connection of two state-space systems |

## Runtime controllers

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`ADRCController`](inc/wet/controllers/adrc.hpp#L89) | Active Disturbance Rejection Control (ADRC) |
| [`ADRCResult`](inc/wet/controllers/adrc.hpp#L13) | Active Disturbance Rejection Control design result |
| [`ESCConfig`](inc/wet/controllers/esc.hpp#L49) | Extremum-seeking controller configuration (discrete realization) |
| [`ExtremumSeekingController`](inc/wet/controllers/esc.hpp#L206) | Extremum-seeking controller runtime (model-free online optimizer) |
| [`HarmonicSuppressor`](inc/wet/controllers/harmonic_suppression.hpp#L104) | Multi-resonant harmonic suppressor — a parallel bank of PR resonators |
| [`HarmonicSuppressorResult`](inc/wet/controllers/harmonic_suppression.hpp#L34) | Design result for a multi-resonant harmonic suppressor |
| [`LeadLagController`](inc/wet/controllers/lead_lag.hpp#L228) | Discrete Lead-Lag Compensator |
| [`LQG`](inc/wet/controllers/lqg.hpp#L113) | Linear-Quadratic-Gaussian (LQG) controller |
| [`LQGI`](inc/wet/controllers/lqgi.hpp#L107) | Linear-Quadratic-Gaussian-Integral (LQGI) controller |
| [`LQGIResult`](inc/wet/controllers/lqgi.hpp#L12) | LQGI design result |
| [`LQGResult`](inc/wet/controllers/lqg.hpp#L14) | LQG design result |
| [`LQI`](inc/wet/controllers/lqi.hpp#L99) | Linear-Quadratic-Integral (LQI) controller |
| [`LQIResult`](inc/wet/controllers/lqi.hpp#L14) | LQI design result |
| [`LQR`](inc/wet/controllers/lqr.hpp#L333) | Runtime Linear-Quadratic Regulator |
| [`LQRResult`](inc/wet/controllers/lqr.hpp#L18) | Linear-Quadratic Regulator design result |
| [`MultiPRController`](inc/wet/controllers/pr.hpp#L298) | Multi-harmonic PR Controller |
| [`PIDController`](inc/wet/controllers/pid.hpp#L146) | Discrete 2-DOF PID controller specialization |
| [`PIDMode`](inc/wet/controllers/pid.hpp#L106) | Compile-time selection of the PID control-law structure |
| [`PIDResult`](inc/wet/controllers/pid.hpp#L12) | 2-DOF PID controller design result |
| [`PIDRuntimeMode`](inc/wet/controllers/pid.hpp#L119) | Runtime operating mode for PIDController |
| [`PRController`](inc/wet/controllers/pr.hpp#L159) | Discrete Proportional-Resonant Controller |
| [`RepetitiveConfig`](inc/wet/controllers/repetitive.hpp#L54) | Repetitive-controller tuning + period (with optional zero-phase FIR Q) |
| [`RepetitiveController`](inc/wet/controllers/repetitive.hpp#L248) | Plug-in repetitive controller runtime (fixed-size internal model) |
| [`SMCController`](inc/wet/controllers/smc.hpp#L63) | First-order sliding-mode controller (SMC) for a SISO plant |
| [`SMCResult`](inc/wet/controllers/smc.hpp#L9) | Tuning parameters for a first-order sliding-mode controller |
| [`STSMCResult`](inc/wet/controllers/stsmc.hpp#L14) | Super-twisting (second-order sliding-mode) controller design result |
| [`SuperTwistingController`](inc/wet/controllers/stsmc.hpp#L132) | Super-twisting controller (second-order sliding mode) |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`adrc`](inc/wet/controllers/adrc.hpp#L39) | Active Disturbance Rejection Control design |
| [`continuous_lqr`](inc/wet/controllers/lqr.hpp#L265) | Continuous-time Linear-Quadratic Regulator design |
| [`discrete_lqg`](inc/wet/controllers/lqg.hpp#L67) | Linear-Quadratic-Gaussian regulator design combining LQR and Kalman filter |
| [`discrete_lqgi`](inc/wet/controllers/lqgi.hpp#L80) | Linear-Quadratic-Gaussian with integral action for tracking |
| [`discrete_lqi`](inc/wet/controllers/lqi.hpp#L39) | Linear-Quadratic Integral design for tracking with servo action |
| [`discrete_lqr`](inc/wet/controllers/lqr.hpp#L62) | Discrete-time Linear-Quadratic Regulator design |
| [`discrete_lqr_from_continuous`](inc/wet/controllers/lqr.hpp#L207) | Design discrete LQR from continuous-time system via discretization |
| [`discretize_lqr_cost`](inc/wet/controllers/lqr.hpp#L125) | Discretize a continuous LQR cost integral over one sample (Van Loan) |
| [`lag`](inc/wet/controllers/lead_lag.hpp#L143) | Design a lag compensator from desired low-frequency gain boost |
| [`lead`](inc/wet/controllers/lead_lag.hpp#L97) | Design a lead compensator from desired phase boost at a target frequency |
| [`lead_lag`](inc/wet/controllers/lead_lag.hpp#L178) | Design a lead-lag compensator (cascade of lead + lag sections) |
| [`lead_lag_direct`](inc/wet/controllers/lead_lag.hpp#L210) | Direct lead-lag specification from zero/pole locations |
| [`lqg_from_parts`](inc/wet/controllers/lqg.hpp#L95) | Combine separate Kalman filter and LQR designs into an LQG controller |
| [`pid`](inc/wet/controllers/pid.hpp#L66) | 2-DOF PID controller design |
| [`pr`](inc/wet/controllers/pr.hpp#L114) | Design a Proportional-Resonant controller |
| [`pr_harmonics`](inc/wet/controllers/pr.hpp#L130) | Design multiple-harmonic PR controller gains |
| [`smc`](inc/wet/controllers/smc.hpp#L35) | Bundle hand-picked SMC parameters into an SMCResult |
| [`stsmc`](inc/wet/controllers/stsmc.hpp#L103) | Super-twisting controller from gains you specify directly |
| [`synthesize_esc`](inc/wet/controllers/esc.hpp#L133) | Synthesize an extremum-seeking controller |
| [`synthesize_esc_mppt`](inc/wet/controllers/esc.hpp#L184) | MPPT-flavored ESC: maximize a power measurement by perturbing the operating point (e.g. converter duty or reference voltage) |
| [`synthesize_harmonic_suppressor`](inc/wet/controllers/harmonic_suppression.hpp#L58) | Synthesize a multi-resonant harmonic suppressor |
| [`synthesize_repetitive`](inc/wet/controllers/repetitive.hpp#L140) | Synthesize a repetitive controller with a scalar robustness filter Q |
| [`synthesize_repetitive_binomial`](inc/wet/controllers/repetitive.hpp#L200) | Synthesize a repetitive controller with a binomial zero-phase FIR Q |
| [`synthesize_stsmc`](inc/wet/controllers/stsmc.hpp#L42) | Synthesize super-twisting gains from a disturbance-derivative bound |

## Design & synthesis

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`JordanBlock`](inc/wet/design/pole_placement.hpp#L420) | One Jordan mini-block of a desired closed-loop spectrum |
| [`JordanPlan`](inc/wet/design/pole_placement.hpp#L437) | Precomputed, K-independent data for the Klein–Moore construction |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`ackermann`](inc/wet/design/pole_placement.hpp#L1146) | Single-input pole placement via Ackermann's formula |
| [`bandwidth_from_settling_time`](inc/wet/design/pid_design.hpp#L457) | Map settling-time and damping-ratio targets to a bandwidth estimate |
| [`care`](inc/wet/design/riccati.hpp#L701) | Solve the Continuous-time Algebraic Riccati Equation (CARE) |
| [`care_schur`](inc/wet/design/riccati.hpp#L492) | Solve CARE via the ordered real-Schur method (Laub's method) |
| [`closed_loop_poles`](inc/wet/design/stability.hpp#L244) | Compute closed-loop poles (eigenvalues) with state feedback |
| [`cohen_coon`](inc/wet/design/pid_design.hpp#L184) | Cohen-Coon tuning from first-order-plus-dead-time model |
| [`controllability_matrix`](inc/wet/design/stability.hpp#L28) | Compute the controllability matrix [B, AB, A²B, ..., A^(N-1)B] |
| [`damping_ratio_from_overshoot_percent`](inc/wet/design/pid_design.hpp#L394) | Map percent overshoot target to equivalent damping ratio |
| [`dare`](inc/wet/design/riccati.hpp#L580) | Solve the Discrete Algebraic Riccati Equation (DARE) |
| [`dare_rde`](inc/wet/design/riccati.hpp#L169) | Solve DARE via Riccati Difference Equation (RDE) iteration |
| [`dare_sda`](inc/wet/design/riccati.hpp#L82) | Solve DARE via Structure-Preserving Doubling Algorithm (SDA) |
| [`is_closed_loop_stable_discrete`](inc/wet/design/stability.hpp#L161) | Check closed-loop stability for discrete system with state feedback |
| [`is_controllable`](inc/wet/design/stability.hpp#L100) | Check if a system is controllable |
| [`is_observable`](inc/wet/design/stability.hpp#L117) | Check if a system is observable |
| [`is_stabilizable`](inc/wet/design/riccati.hpp#L14) | Check if (A, B) is a stabilizable pair |
| [`is_stable_discrete`](inc/wet/design/stability.hpp#L134) | Check if a discrete-time system matrix A is stable |
| [`lambda_tuning`](inc/wet/design/pid_design.hpp#L285) | Lambda tuning for FOPDT model |
| [`linearize`](inc/wet/design/linearization.hpp#L110) | Linearize nonlinear dynamics and output maps about an operating point |
| [`lqr_gain`](inc/wet/design/riccati.hpp#L667) | Optimal LQR state-feedback gain from a Riccati solution |
| [`observability_matrix`](inc/wet/design/stability.hpp#L58) | Compute the observability matrix [C; CA; CA²; ...; CA^(N-1)] |
| [`phase_margin_from_damping_ratio`](inc/wet/design/pid_design.hpp#L426) | Approximate phase margin from damping ratio |
| [`pi_pole_placement_first_order`](inc/wet/design/pid_design.hpp#L612) | PI gains that place the closed-loop poles of a first-order plant |
| [`pid_from_bandwidth`](inc/wet/design/pid_design.hpp#L311) | Design PID from desired bandwidth and phase margin |
| [`pid_from_performance_spec`](inc/wet/design/pid_design.hpp#L490) | Design PID directly from settling-time and overshoot targets |
| [`pid_pole_placement`](inc/wet/design/pid_design.hpp#L511) | Direct PID pole placement for a first-order-plus-dead-time model |
| [`place`](inc/wet/design/pole_placement.hpp#L44) | Robust multi-input pole placement (Kautsky–Nichols–Van Dooren, real poles) |
| [`place_jordan`](inc/wet/design/pole_placement.hpp#L692) | Exact pole placement with an arbitrary Jordan structure (Schmid–Ntogramatzidis–Nguyen–Pandey / Klein–Moore parametric form) |
| [`place_jordan_optimal`](inc/wet/design/pole_placement.hpp#L779) | Robust / minimum-gain arbitrary pole placement (Schmid et al., Methods 1–2) |
| [`rank`](inc/wet/design/stability.hpp#L88) | Compute rank of a matrix via Gaussian elimination with partial pivoting |
| [`reorder_schur`](inc/wet/design/riccati.hpp#L423) | Reorder a real Schur form so eigenvalues satisfying in_front lead |
| [`simc`](inc/wet/design/pid_design.hpp#L236) | SIMC (Skogestad Internal Model Control) tuning for FOPDT models |
| [`split_real_2x2`](inc/wet/design/riccati.hpp#L299) | Split a real-eigenvalue 2×2 Schur block into two 1×1 blocks |
| [`stability_margin_continuous`](inc/wet/design/stability.hpp#L185) | Compute stability margin for continuous system |
| [`stability_margin_discrete`](inc/wet/design/stability.hpp#L214) | Compute stability margin for discrete system |
| [`swap_schur_blocks`](inc/wet/design/riccati.hpp#L359) | Swap two adjacent diagonal blocks of a real Schur form |
| [`tyreus_luyben`](inc/wet/design/pid_design.hpp#L143) | Tyreus-Luyben tuning from ultimate gain and ultimate period |
| [`ziegler_nichols`](inc/wet/design/pid_design.hpp#L51) | Ziegler-Nichols tuning from ultimate gain and ultimate period |
| [`ziegler_nichols_step`](inc/wet/design/pid_design.hpp#L94) | Ziegler-Nichols step response method (reaction curve) |

## Observers & estimators

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`Chirp`](inc/wet/estimation/excitation.hpp#L611) | Linear or logarithmic chirp runtime generator |
| [`ChirpConfig`](inc/wet/estimation/excitation.hpp#L159) | Configuration for a sine chirp excitation |
| [`ChirpResult`](inc/wet/estimation/excitation.hpp#L201) | Chirp design payload |
| [`ClassicalDisturbanceObserver`](inc/wet/estimation/disturbance_observer.hpp#L357) | Classical Pn^-1·Q disturbance observer runtime (bolt-on compensator) |
| [`ClassicalDobResult`](inc/wet/estimation/disturbance_observer.hpp#L284) | Design result for the classical Pn^-1·Q disturbance observer |
| [`ComplementaryFilter`](inc/wet/estimation/sensor_fusion.hpp#L41) | Simple complementary filter for orientation estimation |
| [`DisturbanceObserverConfig`](inc/wet/estimation/disturbance_observer.hpp#L17) | Configuration for a first-order disturbance observer |
| [`ErrorStateJacobian`](inc/wet/estimation/eskf.hpp#L118) | Error-state prediction Jacobians (nominal state updated externally) |
| [`ErrorStateKalmanFilter`](inc/wet/estimation/eskf.hpp#L153) | Error-State Kalman Filter for attitude estimation |
| [`ESKFOrientationFilter`](inc/wet/estimation/sensor_fusion.hpp#L312) | ESKF-based orientation estimator (convenience wrapper) |
| [`ESKFResult`](inc/wet/estimation/eskf.hpp#L23) | Error-State Kalman Filter design result |
| [`ExtendedKalmanFilter`](inc/wet/estimation/ekf.hpp#L81) | Extended Kalman Filter for nonlinear discrete-time systems |
| [`KalmanFilter`](inc/wet/estimation/kalman.hpp#L121) | Runtime Kalman filter for embedded systems |
| [`KalmanResult`](inc/wet/estimation/kalman.hpp#L12) | Steady-state Kalman filter design result |
| [`MadgwickFilter`](inc/wet/estimation/sensor_fusion.hpp#L79) | Madgwick gradient-descent AHRS filter |
| [`MahonyFilter`](inc/wet/estimation/sensor_fusion.hpp#L137) | Mahony nonlinear complementary filter with PI correction |
| [`MeasJacobian`](inc/wet/estimation/ekf.hpp#L53) | Measurement prediction result from the user's observation function |
| [`MultiSine`](inc/wet/estimation/excitation.hpp#L1065) | Sum-of-tones multi-sine runtime generator |
| [`MultiSineConfig`](inc/wet/estimation/excitation.hpp#L521) | Configuration for fixed-component multi-sine excitation |
| [`MultiSineResult`](inc/wet/estimation/excitation.hpp#L566) | Multi-sine design payload |
| [`Observer`](inc/wet/estimation/observer.hpp#L412) | Luenberger state observer (runtime) |
| [`ObserverResult`](inc/wet/estimation/observer.hpp#L69) | Luenberger observer design result |
| [`PRBS`](inc/wet/estimation/excitation.hpp#L731) | Maximal-length PRBS runtime generator |
| [`PRBSConfig`](inc/wet/estimation/excitation.hpp#L247) | Configuration for maximal-length pseudo-random binary excitation |
| [`PRBSResult`](inc/wet/estimation/excitation.hpp#L291) | PRBS design payload |
| [`Ramp`](inc/wet/estimation/excitation.hpp#L966) | Rate-limited ramp runtime generator |
| [`RampConfig`](inc/wet/estimation/excitation.hpp#L418) | Configuration for a slew-rate-limited ramp excitation |
| [`RampResult`](inc/wet/estimation/excitation.hpp#L451) | Ramp design payload |
| [`ReducedObserverResult`](inc/wet/estimation/observer.hpp#L230) | Reduced-order (Gopinath) observer design result |
| [`ReducedOrderObserver`](inc/wet/estimation/observer.hpp#L492) | Reduced-order (Gopinath) state observer (runtime) |
| [`RelayAutotuneConfig`](inc/wet/estimation/relay_autotune.hpp#L102) | Configuration for the relay-feedback autotuning experiment |
| [`RelayAutotuneOutput`](inc/wet/estimation/relay_autotune.hpp#L206) | Per-tick output of RelayAutotuner::step |
| [`RelayAutotuner`](inc/wet/estimation/relay_autotune.hpp#L221) | Runtime relay-feedback autotuner |
| [`RelayAutotuneResult`](inc/wet/estimation/relay_autotune.hpp#L150) | Relay-autotuner design payload |
| [`StateJacobian`](inc/wet/estimation/ekf.hpp#L26) | State prediction result from the user's dynamics function |
| [`StepTrain`](inc/wet/estimation/excitation.hpp#L878) | Alternating +/- step train runtime generator |
| [`StepTrainConfig`](inc/wet/estimation/excitation.hpp#L342) | Configuration for alternating +/- step excitation |
| [`StepTrainResult`](inc/wet/estimation/excitation.hpp#L375) | Step-train design payload |
| [`Tone`](inc/wet/estimation/excitation.hpp#L494) | One sinusoidal component in a multi-sine excitation |
| [`UnscentedKalmanFilter`](inc/wet/estimation/ukf.hpp#L82) | Unscented (sigma-point) Kalman Filter for nonlinear discrete-time systems |
| [`UnscentedParams`](inc/wet/estimation/ukf.hpp#L64) | Tuning parameters for the scaled unscented transform |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`eskf_design`](inc/wet/estimation/eskf.hpp#L47) | Error-State Kalman Filter design from IMU sensor specifications |
| [`eskf_update_imu`](inc/wet/estimation/sensor_fusion.hpp#L190) | Full ESKF predict+update+inject cycle for 6-axis IMU fusion |
| [`kalman`](inc/wet/estimation/kalman.hpp#L44) | Steady-state Kalman filter design |
| [`requires`](inc/wet/estimation/ekf.hpp#L42) | Concept for EKF state functions |
| [`synthesize_chirp`](inc/wet/estimation/excitation.hpp#L235) | Build a chirp design payload from a configuration |
| [`synthesize_classical_dob`](inc/wet/estimation/disturbance_observer.hpp#L327) | Synthesize a classical disturbance observer from a nominal plant and Q-filter |
| [`synthesize_multi_sine`](inc/wet/estimation/excitation.hpp#L595) | Build a multi-sine design payload from a configuration |
| [`synthesize_observer`](inc/wet/estimation/observer.hpp#L111) | Design a Luenberger observer by pole placement (matrix form) |
| [`synthesize_prbs`](inc/wet/estimation/excitation.hpp#L326) | Build a PRBS design payload from a configuration |
| [`synthesize_ramp`](inc/wet/estimation/excitation.hpp#L482) | Build a ramp design payload from a configuration |
| [`synthesize_reduced_observer`](inc/wet/estimation/observer.hpp#L282) | Design a reduced-order (Gopinath) observer by pole placement (matrix form) |
| [`synthesize_relay_autotune`](inc/wet/estimation/relay_autotune.hpp#L181) | Build a validated relay-autotune design payload |
| [`synthesize_step_train`](inc/wet/estimation/excitation.hpp#L406) | Build a step-train design payload from a configuration |

## Filters & signal conditioning

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`Biquad`](inc/wet/filters/filters.hpp#L720) | Second-order IIR (biquad) section runtime |
| [`BiquadCascade`](inc/wet/filters/filters.hpp#L792) | Cascade of second-order sections (SOS) for higher-order IIR filters |
| [`Complementary`](inc/wet/filters/filters.hpp#L1095) | Scalar (1-D) complementary filter — fuse a fast rate with a slow absolute |
| [`Delay`](inc/wet/filters/filters.hpp#L831) | Discrete-time delay buffer |
| [`Goertzel`](inc/wet/filters/spectral.hpp#L29) | Generalized Goertzel single-bin DFT — amplitude/phase at one frequency |
| [`HarmonicAnalyzer`](inc/wet/filters/spectral.hpp#L127) | Harmonic analyzer — a Goertzel bank over a fundamental and K−1 harmonics |
| [`HighPass`](inc/wet/filters/filters.hpp#L983) | First-order high-pass (washout) filter runtime |
| [`LowPass`](inc/wet/filters/filters.hpp#L600) | Nth-order low-pass filter |
| [`MedianFilter`](inc/wet/filters/filters.hpp#L1026) | Sliding-window median filter — nonlinear spike/outlier rejection |
| [`MovingAverage`](inc/wet/filters/filters.hpp#L913) | Moving-average (boxcar) filter — also a DC-preserving harmonic-notch comb |
| [`MSTOGI`](inc/wet/filters/sogi.hpp#L211) | Runtime MSTOGI with exact resonator and forward-Euler washout |
| [`Resonator`](inc/wet/filters/pll.hpp#L399) | Dual-SOGI three-phase positive-sequence PLL (DSOGI-PLL) |
| [`RobustExactDifferentiator`](inc/wet/filters/differentiator.hpp#L49) | First-order robust exact differentiator (super-twisting differentiator) |
| [`SensorlessEstimator`](inc/wet/filters/pll.hpp#L189) | Sensorless rotor flux/position estimator for a PMSM, with optional sensor fusion |
| [`SinglePhasePLL`](inc/wet/filters/pll.hpp#L11) | Single-Phase PLL |
| [`SOGI`](inc/wet/filters/sogi.hpp#L154) | Runtime SOGI wrapper around design::sogi(w0, alpha, Ts) |
| [`SogiFll`](inc/wet/filters/sogi.hpp#L286) | SOGI with a Frequency-Locked Loop — self-tuning single-tone tracker |
| [`ThreePhasePLL`](inc/wet/filters/pll.hpp#L113) | Synchronous-reference-frame (SRF) PLL for balanced three-phase input |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`bandpass`](inc/wet/filters/filters.hpp#L480) | Second-order band-pass filter (constant 0 dB peak gain) |
| [`comb_notch_window`](inc/wet/filters/filters.hpp#L890) | Window length for a moving-average comb that notches f_notch and all its harmonics: N = round(fs / f_notch) |
| [`highpass_2nd`](inc/wet/filters/filters.hpp#L499) | Second-order high-pass filter (RBJ) |
| [`highshelf`](inc/wet/filters/filters.hpp#L571) | High-shelf EQ filter: boost or cut everything above fc |
| [`lowpass_1st`](inc/wet/filters/filters.hpp#L93) | First-order low-pass filter design |
| [`lowpass_2nd`](inc/wet/filters/filters.hpp#L136) | Second-order low-pass filter design |
| [`lowshelf`](inc/wet/filters/filters.hpp#L542) | Low-shelf EQ filter: boost or cut everything below fc |
| [`mstogi`](inc/wet/filters/sogi.hpp#L69) | Mixed Second/Third-Order Generalized Integrator (MSTOGI) |
| [`negative_sequence_ab`](inc/wet/filters/pll.hpp#L378) | Instantaneous negative-sequence αβ from a quadrature signal pair |
| [`notch`](inc/wet/filters/filters.hpp#L457) | Second-order band-reject (notch) filter |
| [`pade_delay_1st`](inc/wet/filters/filters.hpp#L258) | First-order Pade approximation of time delay |
| [`pade_delay_2nd`](inc/wet/filters/filters.hpp#L296) | Second-order Pade approximation of time delay |
| [`peaking`](inc/wet/filters/filters.hpp#L520) | Peaking (bell) EQ filter: boost or cut a band around f0 |
| [`positive_sequence_ab`](inc/wet/filters/pll.hpp#L349) | Instantaneous positive-sequence αβ from a quadrature signal pair |
| [`requires`](inc/wet/filters/filters.hpp#L203) | Butterworth low-pass filter design |
| [`sogi`](inc/wet/filters/sogi.hpp#L13) | Second-Order Generalized Integrator (SOGI) design |
| [`to_coeffs`](inc/wet/filters/filters.hpp#L335) | Convert StateSpace system to first-order DSP coefficients |

## Trajectory & motion planning

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`CartesianMove`](inc/wet/trajectory/cartesian_move.hpp#L97) | Path-preserving task-space move (Pipeline B / LIN) |
| [`InputShaper`](inc/wet/trajectory/input_shaper.hpp#L176) | Input-shaper runtime — convolves a command stream with the shaper impulses |
| [`InputShaperBank`](inc/wet/trajectory/input_shaper.hpp#L239) | Multi-axis input-shaper bank — one shaper per axis, shared buffer length |
| [`InputShaperResult`](inc/wet/trajectory/input_shaper.hpp#L60) | Input-shaper design result: impulse amplitudes and sample delays |
| [`JointLimits`](inc/wet/trajectory/cartesian_move.hpp#L77) | Per-joint velocity and acceleration limits for a task-space move |
| [`LinearPath`](inc/wet/trajectory/cartesian_move.hpp#L48) | A straight-line path `p(s) = start + s·dir`, `s ∈ [0, length]` |
| [`PolynomialTrajectory`](inc/wet/trajectory/polynomial.hpp#L204) | Runtime evaluator for a precomputed polynomial trajectory |
| [`PolyTrajectory`](inc/wet/trajectory/polynomial.hpp#L59) | A synthesized polynomial trajectory: the coefficients of p(t) = Σ cᵢ·tⁱ over t ∈ [0, T], plus the duration |
| [`ScurveProfile`](inc/wet/trajectory/scurve.hpp#L60) | A synthesized jerk-limited (double-S) profile: a sequence of constant-jerk segments, evaluated exactly (cubic in t within a segment) |
| [`ScurveTrajectory`](inc/wet/trajectory/scurve.hpp#L283) | Runtime evaluator for a precomputed jerk-limited (double-S) profile |
| [`SplineProfile`](inc/wet/trajectory/spline.hpp#L52) | A synthesized multi-waypoint spline: per-segment polynomial coefficients (ascending power, in segment-local time) plus the knot times |
| [`SplineTrajectory`](inc/wet/trajectory/spline.hpp#L258) | Runtime player for a multi-waypoint spline (design::SplineProfile) |
| [`ToppMove`](inc/wet/trajectory/topp.hpp#L103) | Time-optimal task-space move (path-preserving, pointwise minimum-time) |
| [`ToppProfile`](inc/wet/trajectory/topp.hpp#L50) | The scalar time-optimal path-timing produced by TOPP |
| [`TrajectoryBank`](inc/wet/trajectory/polynomial.hpp#L253) | Multi-axis coordination: time-scale each axis's profile to the slowest so a multi-DOF move starts and finishes synchronized ("linear" / coordinated joint moves — the feedforward reference for a manipulator) |
| [`TrajectoryBoundary`](inc/wet/trajectory/trajectory_types.hpp#L61) | Boundary conditions at one endpoint of a polynomial trajectory: a position and its time derivatives |
| [`TrajectoryLimits`](inc/wet/trajectory/trajectory_types.hpp#L23) | Asymmetric kinematic limits for a trapezoidal or S-curve motion profile |
| [`TrajectoryState`](inc/wet/trajectory/trajectory_types.hpp#L48) | A point on a motion profile: commanded position, velocity, acceleration |
| [`TrapezoidalProfile`](inc/wet/trajectory/trapezoidal.hpp#L41) | Planned trapezoidal profile: the segment durations, reached values, and boundary state needed to evaluate the trajectory at any time |
| [`TrapezoidalTrajectory`](inc/wet/trajectory/trapezoidal.hpp#L271) | Runtime evaluator for a precomputed trapezoidal profile |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`plan_for_sign`](inc/wet/trajectory/trapezoidal.hpp#L135) | Plan the three-segment profile assuming the cruise velocity has sign s |
| [`synthesize_input_shaper`](inc/wet/trajectory/input_shaper.hpp#L96) | Synthesize an input shaper for a second-order mode |
| [`synthesize_poly_trajectory`](inc/wet/trajectory/polynomial.hpp#L117) | Synthesize a fixed-duration polynomial trajectory matching boundary conditions on position and its derivatives at both endpoints |
| [`synthesize_scurve`](inc/wet/trajectory/scurve.hpp#L164) | Synthesize a minimum-time jerk-limited (7-segment double-S) profile from (Xi, Vi) to (Xf, Vf) under asymmetric kinematic limits |
| [`synthesize_spline`](inc/wet/trajectory/spline.hpp#L128) | Synthesize a multi-waypoint spline through points at times |
| [`synthesize_trapezoidal`](inc/wet/trajectory/trapezoidal.hpp#L183) | Synthesize the minimum-time asymmetric trapezoidal profile from (Xi, Vi) to (Xf, Vf) under the given limits |

## Kinematics

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`CartesianMap`](inc/wet/kinematics/motion_maps.hpp#L31) | Cartesian gantry: independent per-axis affine map `task = scale·act + offset` (the "kinematics" is the identity, exposed for a uniform forward/inverse interface) |
| [`CoreXY`](inc/wet/kinematics/motion_maps.hpp#L59) | CoreXY belt mapping (2 motors A/B → Cartesian X/Y) |
| [`DhChain`](inc/wet/kinematics/serial_arm.hpp#L90) | An N-joint DH chain (the arm geometry) |
| [`DhJoint`](inc/wet/kinematics/serial_arm.hpp#L67) | One joint's standard (distal) DH parameters and motion limits |
| [`FiveBar`](inc/wet/kinematics/scara.hpp#L76) | Planar five-bar parallel manipulator (parallel SCARA) |
| [`FiveBarGeometry`](inc/wet/kinematics/scara.hpp#L41) | Symmetric five-bar geometry (two base motors, equal proximal/distal links) |
| [`LinearDelta`](inc/wet/kinematics/motion_maps.hpp#L264) | Linear delta robot — per-carriage closed-form inverse, sphere- trilateration forward. Towers at 90°, 210°, 330° |
| [`LinearDeltaGeometry`](inc/wet/kinematics/motion_maps.hpp#L249) | Linear delta geometry (three vertical carriages, fixed-length rods) |
| [`PolarMap`](inc/wet/kinematics/motion_maps.hpp#L85) | Polar / R-θ mapping (radius + angle ↔ Cartesian X/Y) |
| [`Pose`](inc/wet/kinematics/pose.hpp#L50) | Rigid-body pose: a translation and an orientation (unit quaternion) |
| [`RotaryDelta`](inc/wet/kinematics/motion_maps.hpp#L146) | Rotary delta robot — closed-form inverse, quadratic-intersection forward |
| [`RotaryDeltaGeometry`](inc/wet/kinematics/motion_maps.hpp#L129) | Rotary delta geometry (three base servos, parallelogram arms) |
| [`SerialArm`](inc/wet/kinematics/serial_arm.hpp#L168) | Serial N-DOF revolute manipulator runtime |
| [`SerialArmConfig`](inc/wet/kinematics/serial_arm.hpp#L151) | Validated serial-arm configuration (the design payload) |
| [`StewartConfig`](inc/wet/kinematics/stewart.hpp#L103) | Validated Stewart configuration (the design payload) |
| [`StewartGeometry`](inc/wet/kinematics/stewart.hpp#L48) | Rig geometry: the six fixed base anchors `bᵢ`, the six moving-platform anchors `pᵢ`, the actuator stroke limits, and the nominal home height |
| [`StewartPlatform`](inc/wet/kinematics/stewart.hpp#L136) | Gough–Stewart platform runtime — closed-form inverse, Newton forward |
| [`Translation3`](inc/wet/kinematics/pose.hpp#L27) | A 3-D translation — a thin Vec3 with domain-named conveniences |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`arm_spherical_wrist`](inc/wet/kinematics/serial_arm.hpp#L466) | Tier-2 builder for a standard 6R elbow arm with a spherical wrist |
| [`five_bar_symmetric`](inc/wet/kinematics/scara.hpp#L231) | Build a symmetric five-bar parallel SCARA |
| [`scara_arm`](inc/wet/kinematics/scara.hpp#L243) | Build a series SCARA (RRPR) as a 4-joint DH chain |
| [`select_nearest`](inc/wet/kinematics/serial_arm.hpp#L411) | Pick the solution branch nearest a reference configuration |
| [`stewart_symmetric`](inc/wet/kinematics/stewart.hpp#L326) | Tier-2 builder for the common symmetric hexagonal layout |
| [`synthesize_serial_arm`](inc/wet/kinematics/serial_arm.hpp#L453) | Validate a serial-arm DH chain and flag a spherical wrist |
| [`synthesize_stewart`](inc/wet/kinematics/stewart.hpp#L298) | Validate a hand-entered Stewart geometry and confirm the home pose is reachable |

## Motor control

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`AlphaBeta`](inc/wet/power/transforms.hpp#L130) | Alpha-beta (stationary-frame) component pair |
| [`AlphaBetaZero`](inc/wet/power/transforms.hpp#L195) | Alpha-beta-zero (stationary-frame) component triple |
| [`CascadeBandwidths`](inc/wet/power/servo.hpp#L23) | The three bandwidth knobs of the position/velocity/current cascade |
| [`Convention`](inc/wet/power/transforms.hpp#L43) | Scaling convention for the Clarke/Park family |
| [`DcBusLimiter`](inc/wet/power/limits.hpp#L39) | Holds the inverter's torque current within DC-bus current/power limits |
| [`DcBusLimits`](inc/wet/power/limits.hpp#L10) | DC-bus current and voltage limits for an inverter |
| [`DcBusState`](inc/wet/power/limits.hpp#L27) | DC-bus state and the torque-current derate it implies |
| [`DirectQuadrature`](inc/wet/power/transforms.hpp#L62) | Direct-quadrature (rotor-frame) component pair |
| [`DirectQuadratureZero`](inc/wet/power/transforms.hpp#L215) | Direct-quadrature-zero (rotor-frame) component triple |
| [`DqCommand`](inc/wet/power/foc.hpp#L261) | Result of FOController::current_controller(): the dq voltage command plus its saturation signals |
| [`FetLossModel`](inc/wet/power/thermal.hpp#L116) | First-order inverter FET loss model (conduction + switching) |
| [`FocResult`](inc/wet/power/foc.hpp#L245) | Result of one FOController::step(), carrying the actuator command plus the saturation/measurement signals an outer (velocity/position) loop needs to propagate anti-windup back up a cascade |
| [`InstantaneousPower`](inc/wet/power/transforms.hpp#L525) | Instantaneous active and reactive power |
| [`JunctionEstimator`](inc/wet/power/thermal.hpp#L176) | FET junction-temperature estimator: case temperature plus a thermal model |
| [`MechanicalEstimator`](inc/wet/power/mechanical_estimator.hpp#L81) | Cheap-predict mechanical estimator for position, speed, and load torque |
| [`MechanicalEstimatorConfig`](inc/wet/power/mechanical_estimator.hpp#L56) | Configuration for MechanicalEstimator |
| [`PhaseCalibrationCommand`](inc/wet/power/calibration.hpp#L39) | One step's output from PhaseParameterCalibrator |
| [`PhaseCalibrationConfig`](inc/wet/power/calibration.hpp#L13) | Configuration for online phase resistance/inductance commissioning |
| [`PhaseParameterCalibrator`](inc/wet/power/calibration.hpp#L49) | Online phase R/L identification by recursive least squares (PRBS injected) |
| [`PmacServo`](inc/wet/power/servo.hpp#L79) | Thin field-oriented PMAC servo: {Iabc, Vdc, θ} in, duties out |
| [`PmacServoConfig`](inc/wet/power/servo.hpp#L44) | Configuration for PmacServo |
| [`ResistiveLossModel`](inc/wet/power/thermal.hpp#L158) | Minimal conduction-only loss model for a weak datasheet |
| [`SequenceComponents`](inc/wet/power/transforms.hpp#L593) | Symmetrical (sequence) components of a three-phase phasor set |
| [`ServoFeedback`](inc/wet/power/servo.hpp#L68) | Sensor feedback for one PmacServo::update tick |
| [`SvmDuties`](inc/wet/power/modulation.hpp#L57) | Result of svm_duty_cycles(): the half-bridge duties plus an over-modulation flag |
| [`ThermalLimiter`](inc/wet/power/thermal.hpp#L269) | Derates the current command from a temperature (Tj for FETs, winding for the motor) |
| [`ThermalLimits`](inc/wet/power/thermal.hpp#L233) | A derating curve plus a hard fault threshold |
| [`ThermalState`](inc/wet/power/thermal.hpp#L259) | State from a ThermalLimiter evaluation |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`base_speed`](inc/wet/power/foc.hpp#L215) | Base (corner) electrical speed where the voltage circle is first hit |
| [`cauer_thermal_ss`](inc/wet/power/thermal.hpp#L66) | Continuous state-space model of a physical Cauer RC thermal ladder |
| [`clarke_park_transform`](inc/wet/power/transforms.hpp#L405) | Fused Clarke-Park transform (abc → dq) |
| [`clarke_park_zero_transform`](inc/wet/power/transforms.hpp#L494) | Fused Clarke-Park transform with zero (abc → dq0) |
| [`clarke_transform`](inc/wet/power/transforms.hpp#L314) | Clarke transform (abc → αβ) |
| [`clarke_zero_transform`](inc/wet/power/transforms.hpp#L235) | Zero-retaining Clarke transform (abc → αβ0) |
| [`current_loop_pi`](inc/wet/power/foc.hpp#L26) | Current-loop PI gains by closed-loop pole placement on the R–L plant |
| [`derate_window`](inc/wet/power/thermal.hpp#L14) | A two-breakpoint derating curve: 1 below derate_start, 0 at cutoff |
| [`flux_from_Kv`](inc/wet/power/foc.hpp#L139) | PM flux linkage from the datasheet velocity constant @f$ K_v @f$ |
| [`flux_from_torque_constant`](inc/wet/power/foc.hpp#L75) | PM flux linkage from a motor's torque constant (amplitude-invariant) |
| [`foster_thermal_ss`](inc/wet/power/thermal.hpp#L32) | Continuous state-space model of a Foster RC thermal network |
| [`instantaneous_power`](inc/wet/power/transforms.hpp#L573) | Instantaneous active and reactive power from dq quantities |
| [`inverse_clarke_transform`](inc/wet/power/transforms.hpp#L336) | Inverse Clarke transform (αβ → abc) |
| [`inverse_clarke_zero_transform`](inc/wet/power/transforms.hpp#L280) | Inverse zero-retaining Clarke transform (αβ0 → abc) |
| [`inverse_park_clarke_transform`](inc/wet/power/transforms.hpp#L433) | Fused inverse Park-Clarke transform (dq → abc) |
| [`inverse_park_clarke_zero_transform`](inc/wet/power/transforms.hpp#L512) | Fused inverse Park-Clarke transform with zero (dq0 → abc) |
| [`inverse_park_transform`](inc/wet/power/transforms.hpp#L380) | Inverse Park transform (dq → αβ) |
| [`inverse_park_zero_transform`](inc/wet/power/transforms.hpp#L480) | Inverse Park transform with zero passthrough (dq0 → αβ0) |
| [`inverse_symmetrical_components`](inc/wet/power/transforms.hpp#L646) | Inverse symmetrical-component transform (012 → abc) |
| [`iq_from_torque`](inc/wet/power/foc.hpp#L178) | q-axis current command for a requested torque (non-salient PMSM, Id=0) |
| [`mechanical_ss`](inc/wet/power/mechanical_estimator.hpp#L13) | Continuous state-space model of a 1-DOF mechanical drivetrain |
| [`motor_constant`](inc/wet/power/foc.hpp#L156) | Motor constant @f$ K_m @f$ (torque per √copper-loss) — a figure of merit |
| [`park_transform`](inc/wet/power/transforms.hpp#L353) | Park transform (αβ → dq) |
| [`park_zero_transform`](inc/wet/power/transforms.hpp#L463) | Park transform with zero passthrough (αβ0 → dq0) |
| [`requires`](inc/wet/power/thermal.hpp#L107) | A loss model usable by JunctionEstimator |
| [`svm_duty_cycles`](inc/wet/power/modulation.hpp#L75) | Space-vector PWM duty cycles from an αβ voltage command |
| [`svpwm_zero_sequence`](inc/wet/power/modulation.hpp#L26) | Min-max zero-sequence injection for space-vector PWM |
| [`symmetrical_components`](inc/wet/power/transforms.hpp#L610) | Forward symmetrical-component (Fortescue) transform (abc → 012) |
| [`torque_constant_from_flux`](inc/wet/power/foc.hpp#L56) | Torque constant @f$ K_t @f$ of a PMSM (amplitude-invariant convention) |
| [`torque_constant_from_Kv`](inc/wet/power/foc.hpp#L99) | Torque constant from the datasheet velocity constant @f$ K_v @f$ |
| [`voltage_circle_radius`](inc/wet/power/foc.hpp#L195) | Radius of the SVPWM voltage circle (max synthesizable @f$ \|V_{dq}\| @f$) |

## Utilities & toolbox

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`AffineCal`](inc/wet/toolbox/scaling.hpp#L62) | Affine sensor calibration `y = gain·x + offset` |
| [`AnalogInput`](inc/wet/toolbox/io.hpp#L23) | A single analog input: range/fault check on the raw reading, then affine calibration to engineering units |
| [`AxisInput`](inc/wet/toolbox/io.hpp#L66) | Operator-axis conditioning chain (joystick / RC stick → command) |
| [`BLINK`](inc/wet/toolbox/iec61131.hpp#L530) | BLINK (free-running square-wave / flasher) |
| [`Bounds`](inc/wet/toolbox/bounds.hpp#L32) | A per-channel closed-interval box constraint |
| [`Button`](inc/wet/toolbox/io.hpp#L117) | Debounced momentary push-button with edge and long-press detection |
| [`ConstantInertiaFeedforward`](inc/wet/toolbox/actuator.hpp#L164) | Per-axis decoupled torque feedforward: `τ = J·a + b·v + τ_c·sign(v) + g` |
| [`Counter`](inc/wet/toolbox/logic.hpp#L262) | Edge-counting up/down counter: increments on each rising edge of up, decrements on each rising edge of down. Returns the running count |
| [`CTD`](inc/wet/toolbox/iec61131.hpp#L348) | CTD Counter (Count Down) |
| [`CTU`](inc/wet/toolbox/iec61131.hpp#L308) | CTU Counter (Count Up) |
| [`CTUD`](inc/wet/toolbox/iec61131.hpp#L388) | CTUD Counter (Count Up Down) |
| [`Debounce`](inc/wet/toolbox/logic.hpp#L201) | Debounce: the output adopts in only after in differs from the current output continuously for stable_time. Rejects contact bounce and brief glitches. (Not an IEC block — the one everyone hand-rolls.) |
| [`DFF`](inc/wet/toolbox/iec61131.hpp#L445) | D Flip-Flop (edge-triggered data latch) |
| [`DLATCH`](inc/wet/toolbox/iec61131.hpp#L475) | D Latch (level-sensitive / transparent latch) |
| [`F_TRIG`](inc/wet/toolbox/iec61131.hpp#L132) | F_TRIG (Falling Edge Trigger) |
| [`Hysteresis`](inc/wet/toolbox/conditioning.hpp#L177) | Hysteresis comparator (Schmitt trigger): bool output with separate on/off thresholds to reject chatter |
| [`Lut1D`](inc/wet/toolbox/lookup.hpp#L64) | 1-D interpolating lookup table over monotonic breakpoints |
| [`Lut2D`](inc/wet/toolbox/lookup.hpp#L122) | 2-D bilinear interpolating lookup table over a regular grid |
| [`OffDelayTimer`](inc/wet/toolbox/logic.hpp#L122) | Off-delay timer: output goes true immediately when in is true and stays true until in has been false continuously for delay |
| [`OnDelayTimer`](inc/wet/toolbox/logic.hpp#L85) | On-delay timer: output goes true once in has been held true continuously for delay; drops immediately when in goes false |
| [`Periodic`](inc/wet/toolbox/timing.hpp#L102) | Periodic trigger — fires once per elapsed period |
| [`PulseTimer`](inc/wet/toolbox/logic.hpp#L159) | Pulse timer (non-retriggerable): a rising edge of in emits a fixed |
| [`QuadratureDecoder`](inc/wet/toolbox/encoder.hpp#L47) | Software A/B quadrature decoder with optional index |
| [`R_TRIG`](inc/wet/toolbox/iec61131.hpp#L103) | R_TRIG (Rising Edge Trigger) |
| [`RangeMonitor`](inc/wet/toolbox/conditioning.hpp#L268) | Analog-input range/fault monitor (NAMUR NE43 pattern) |
| [`RS`](inc/wet/toolbox/iec61131.hpp#L75) | RS Latch (Reset-Set Latch) |
| [`ServoAxis`](inc/wet/toolbox/actuator.hpp#L90) | One servoactuator transmission: SI joint unit ⟷ drive (motor) units |
| [`ServoBank`](inc/wet/toolbox/actuator.hpp#L203) | A bank of ServoAxis transmissions: maps a synchronized multi-axis |
| [`ServoCommand`](inc/wet/toolbox/actuator.hpp#L68) | A drive-native servoactuator setpoint: position, velocity, torque |
| [`SignalStatus`](inc/wet/toolbox/conditioning.hpp#L217) | Classification of an analog input against its valid/fault bands |
| [`SlewLimiter`](inc/wet/toolbox/conditioning.hpp#L116) | Slew-rate limiter: bound how fast the output may follow the target |
| [`SR`](inc/wet/toolbox/iec61131.hpp#L42) | SR Latch (Set-dominant Set-Reset Latch) |
| [`Stopwatch`](inc/wet/toolbox/timing.hpp#L26) | Free-running elapsed-time accumulator |
| [`Switch`](inc/wet/toolbox/io.hpp#L162) | Debounced maintained switch (toggle/selector contact) with change flag |
| [`Tachometer`](inc/wet/toolbox/encoder.hpp#L126) | Pulse-based speed (tachometer) with frequency/period crossover |
| [`TFF`](inc/wet/toolbox/iec61131.hpp#L501) | T Flip-Flop (toggle on rising edge) |
| [`Thermistor`](inc/wet/toolbox/thermistor.hpp#L138) | NTC thermistor linearization (resistance → temperature) |
| [`ThermistorCoeffs`](inc/wet/toolbox/thermistor.hpp#L25) | Fitted NTC coefficients in Steinhart-Hart form |
| [`Timeout`](inc/wet/toolbox/timing.hpp#L56) | One-shot timeout |
| [`TOF`](inc/wet/toolbox/iec61131.hpp#L210) | TOF Timer (Timer Off Delay) |
| [`TON`](inc/wet/toolbox/iec61131.hpp#L167) | TON Timer (Timer On Delay) |
| [`TP`](inc/wet/toolbox/iec61131.hpp#L255) | TP Timer (Timer Pulse) |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`beta`](inc/wet/toolbox/thermistor.hpp#L50) | Fit NTC coefficients from the Beta-parameter model |
| [`classify_range`](inc/wet/toolbox/conditioning.hpp#L242) | Classify x against the four band edges `[fault_lo (valid_lo, valid_hi) fault_hi]` (assumed ordered, non-decreasing) |
| [`deadband`](inc/wet/toolbox/conditioning.hpp#L23) | Dead zone over `[lower, upper]`, matching Simulink's Dead Zone block |
| [`expo`](inc/wet/toolbox/conditioning.hpp#L101) | Exponential response curve `y = (1−k)·x + k·x³` (RC "expo") |
| [`inverse_deadband`](inc/wet/toolbox/conditioning.hpp#L53) | Inverse dead zone: add an offset to overcome a physical dead zone (valve overlap, static friction, motor stiction), with independent negative/positive offsets |
| [`inverse_lerp`](inc/wet/toolbox/scaling.hpp#L34) | Inverse of lerp: the fraction t such that `lerp(a, b, t) == x` |
| [`lerp`](inc/wet/toolbox/scaling.hpp#L22) | Linear interpolation between a and b by fraction t |
| [`LIMIT`](inc/wet/toolbox/iec61131.hpp#L581) | LIMIT (IEC 61131-3 selection function): clamp in to [mn, mx] |
| [`linear_screw`](inc/wet/toolbox/actuator.hpp#L147) | Build a ServoAxis for a linear axis driven by a leadscrew/belt |
| [`lut_segment`](inc/wet/toolbox/lookup.hpp#L31) | Index of the interpolation segment containing x |
| [`MUX`](inc/wet/toolbox/iec61131.hpp#L600) | MUX (IEC 61131-3 multiplexer): select input k of N (0-based) |
| [`poly_horner`](inc/wet/toolbox/scaling.hpp#L105) | Evaluate a polynomial at x by Horner's method |
| [`rescale`](inc/wet/toolbox/scaling.hpp#L45) | Affine map of x from the input range to the output range |
| [`rotary_gearbox`](inc/wet/toolbox/actuator.hpp#L131) | Build a ServoAxis for a rotary joint behind a gearbox |
| [`scaled_deadband`](inc/wet/toolbox/conditioning.hpp#L82) | Center dead zone that rescales the surviving range back to full span |
| [`steinhart_hart`](inc/wet/toolbox/thermistor.hpp#L81) | Fit the Steinhart-Hart coefficients from three calibration points |
| [`two_point_cal`](inc/wet/toolbox/scaling.hpp#L89) | Fit an AffineCal through two `(raw, engineering)` points |
| [`wrapped_delta`](inc/wet/toolbox/encoder.hpp#L23) | Signed difference between two unsigned counter readings, wrap-safe |

## Frequency-domain analysis (host)

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`GreyBoxIdentificationResult`](inc/wet/analysis/identification.hpp#L177) | Selected model used by downstream model-based controller design |
| [`ImpedanceResult`](inc/wet/analysis/analysis.hpp#L1289) | Result of impedance frequency response evaluation |
| [`LoopResponseResult`](inc/wet/analysis/analysis.hpp#L440) | Open-loop and closed-loop frequency response package |
| [`LoopSummary`](inc/wet/analysis/analysis.hpp#L475) | Compact loop summary metrics for quick stability/robustness checks |
| [`LsimInfo`](inc/wet/analysis/analysis.hpp#L1131) | Transient characteristics of an arbitrary response signal |
| [`LsimResult`](inc/wet/analysis/analysis.hpp#L764) | Result of a single-trajectory simulation: time, output, and state history |
| [`MiddlebrookResult`](inc/wet/analysis/analysis.hpp#L1307) | Result of Middlebrook minor loop gain analysis |
| [`PoleZeroMap`](inc/wet/analysis/analysis.hpp#L1209) | Poles and zeros of a system, for pole-zero plotting |
| [`StepInfo`](inc/wet/analysis/analysis.hpp#L1000) | Step-response characteristics of a single output signal |
| [`TimeResponse`](inc/wet/analysis/analysis.hpp#L748) | Multi-channel time-domain response sampled on a time grid |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`bode`](inc/wet/analysis/analysis.hpp#L285) | Compute Bode plot data for a SISO state-space system |
| [`bode_discrete`](inc/wet/analysis/analysis.hpp#L380) | Compute Bode plot data for a discrete-time SISO state-space system |
| [`canonical_phase_margin`](inc/wet/analysis/analysis.hpp#L197) | Normalize phase margin to (-180, 180] |
| [`damp`](inc/wet/analysis/analysis.hpp#L723) | Compute natural frequency and damping for each pole |
| [`dcgain`](inc/wet/analysis/analysis.hpp#L650) | Compute DC gain of a continuous-time system |
| [`gain_margin_unwrapped`](inc/wet/analysis/analysis.hpp#L249) | Find gain margin using unwrapped phase trajectory |
| [`impedance`](inc/wet/analysis/analysis.hpp#L1392) | Compute impedance frequency response from a SISO admittance system |
| [`impedance_direct`](inc/wet/analysis/analysis.hpp#L1427) | Compute impedance frequency response from a SISO impedance transfer function |
| [`impulse`](inc/wet/analysis/analysis.hpp#L855) | Impulse response of a (MIMO) state-space system |
| [`initial`](inc/wet/analysis/analysis.hpp#L885) | Initial-condition (free) response of a (MIMO) state-space system |
| [`is_stable_continuous`](inc/wet/analysis/analysis.hpp#L690) | Check continuous-time stability |
| [`loop_response`](inc/wet/analysis/analysis.hpp#L567) | Compute open-loop L, sensitivity S, complementary sensitivity T, and Nyquist data |
| [`lsim`](inc/wet/analysis/analysis.hpp#L916) | Forced time response of a (MIMO) state-space system to an input signal |
| [`lsiminfo`](inc/wet/analysis/analysis.hpp#L1148) | Compute transient characteristics from an output/time signal |
| [`middlebrook`](inc/wet/analysis/analysis.hpp#L1461) | Middlebrook stability analysis for cascaded source-load systems |
| [`nyquist`](inc/wet/analysis/analysis.hpp#L522) | Compute Nyquist data for a SISO state-space system |
| [`phase_margin_unwrapped`](inc/wet/analysis/analysis.hpp#L213) | Find phase margin using unwrapped phase trajectory |
| [`poles`](inc/wet/analysis/analysis.hpp#L678) | Compute open-loop poles (eigenvalues of A matrix) |
| [`poly_roots`](inc/wet/analysis/analysis.hpp#L1222) | Roots of a polynomial given in ascending powers (MATLAB `roots`, reversed order) |
| [`pzmap`](inc/wet/analysis/analysis.hpp#L1250) | Pole-zero map of a SISO transfer function (MATLAB `pzmap(tf)`) |
| [`step`](inc/wet/analysis/analysis.hpp#L825) | Step response of a (MIMO) state-space system |
| [`stepinfo`](inc/wet/analysis/analysis.hpp#L1019) | Compute step-response characteristics from an output/time signal |
| [`unwrap_phase_deg`](inc/wet/analysis/analysis.hpp#L171) | Unwrap phase data in degrees to avoid +/-180 discontinuities |

## Simulation (host)

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`AdaptiveStepSolver`](inc/wet/simulation/solver.hpp#L209) | Adaptive-step ODE solver |
| [`BackwardEuler`](inc/wet/simulation/integrator.hpp#L119) | Backward Euler integrator |
| [`BDF2`](inc/wet/simulation/integrator.hpp#L182) | Backward Differentiation Formula 2 (BDF2) integrator |
| [`Exact`](inc/wet/simulation/integrator.hpp#L35) | Exact integrator for LTI systems |
| [`FixedStepSolver`](inc/wet/simulation/solver.hpp#L84) | Fixed-step ODE solver |
| [`ForwardEuler`](inc/wet/simulation/integrator.hpp#L79) | Forward Euler integrator |
| [`IntegrationResult`](inc/wet/simulation/integrator.hpp#L11) | Result of an integration step |
| [`SimulationResult`](inc/wet/simulation/simulate.hpp#L28) | Result of a closed-loop simulation |
| [`SolveResult`](inc/wet/simulation/solver.hpp#L28) | Result of an ODE solve operation |
| [`Trapezoidal`](inc/wet/simulation/integrator.hpp#L277) | Trapezoidal (Tustin) integrator |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`bodemag`](inc/wet/simulation/plot_plotly.hpp#L392) | Plot a magnitude-only Bode diagram (log frequency, dB magnitude) |
| [`bodeplot`](inc/wet/simulation/plot_plotly.hpp#L378) | Plot magnitude and phase Bode subplots |
| [`complex_scatter`](inc/wet/simulation/plot_plotly.hpp#L280) | Build a markers-only scatter of complex points (real vs imaginary) |
| [`impulseplot`](inc/wet/simulation/plot_plotly.hpp#L327) | Plot an impulse response, one trace per input/output pair |
| [`lsimplot`](inc/wet/simulation/plot_plotly.hpp#L343) | Plot a forced (lsim) simulation, one trace per output |
| [`nyquistplot`](inc/wet/simulation/plot_plotly.hpp#L417) | Plot a Nyquist locus with the -1 critical point marked |
| [`plot_bode`](inc/wet/simulation/plot_plotly.hpp#L138) | Plot Bode magnitude and phase as subplots |
| [`plot_line`](inc/wet/simulation/plot_plotly.hpp#L191) | Simple line plot of time vs value |
| [`plot_simulation`](inc/wet/simulation/plot_plotly.hpp#L71) | Plot simulation results with subplots for states, outputs, and inputs |
| [`plot_step`](inc/wet/simulation/plot_plotly.hpp#L220) | Plot step response data |
| [`pzplot`](inc/wet/simulation/plot_plotly.hpp#L444) | Plot a pole-zero map on the complex plane (poles as ×, zeros as ○) |
| [`simulate`](inc/wet/simulation/simulate.hpp#L44) | Simulate a nonlinear plant with a controller in closed loop |
| [`simulate_discrete`](inc/wet/simulation/simulate.hpp#L261) | Simulate a discrete-time system with a controller |
| [`simulate_discrete_nonlinear`](inc/wet/simulation/simulate.hpp#L197) | Simulate a discrete-time nonlinear plant with a controller |
| [`simulate_lti`](inc/wet/simulation/simulate.hpp#L166) | Simulate a continuous LTI system with a controller |
| [`simulate_state_feedback`](inc/wet/simulation/simulate.hpp#L112) | Simulate a nonlinear plant with state-feedback controller |
| [`stepplot`](inc/wet/simulation/plot_plotly.hpp#L311) | Plot a step response, one trace per input/output pair |
| [`time_response_figure`](inc/wet/simulation/plot_plotly.hpp#L240) | Build a time-response figure with one line per (output, input) pair |

## MATLAB-style aliases (host)

**Blocks (structs, classes, enums)**

| Name | Description |
| ---- | ----------- |
| [`MarginResult`](inc/wet/matlab.hpp#L592) | Gain/phase margins and their crossover frequencies |

**Functions**

| Name | Description |
| ---- | ----------- |
| [`acker`](inc/wet/matlab.hpp#L351) | Pole placement for state-feedback control |
| [`bandwidth`](inc/wet/matlab.hpp#L635) | -3 dB bandwidth of a SISO system over a frequency grid |
| [`blkdiag`](inc/wet/matlab.hpp#L148) | Block diagonal matrix construction |
| [`c2d`](inc/wet/matlab.hpp#L96) | Matlab interface function c2d to discretize a continuous-time state-space system |
| [`diag`](inc/wet/matlab.hpp#L170) | Returns a square diagonal matrix from the given array |
| [`dlqr`](inc/wet/matlab.hpp#L420) | Discrete-time Linear-Quadratic Regulator design |
| [`eig`](inc/wet/matlab.hpp#L270) | MATLAB short alias for the eigenvalues of a square matrix |
| [`estim`](inc/wet/matlab.hpp#L284) | Form state estimator from system and estimator gain |
| [`eye`](inc/wet/matlab.hpp#L213) | Create an identity matrix of size n x n |
| [`linmod`](inc/wet/matlab.hpp#L128) | MATLAB-style nonlinear linearization about an operating point |
| [`lqg`](inc/wet/matlab.hpp#L479) | Linear-Quadratic-Gaussian regulator design |
| [`lqgreg`](inc/wet/matlab.hpp#L495) | Combine separate Kalman filter and LQR designs into an LQG controller |
| [`lqgtrack`](inc/wet/matlab.hpp#L507) | Linear-Quadratic-Gaussian design with integral action for tracking |
| [`lqi`](inc/wet/matlab.hpp#L466) | Linear-Quadratic Integral design for tracking |
| [`lqr`](inc/wet/matlab.hpp#L400) | Continuous-time LQR design (MATLAB's lqr) |
| [`lqrd`](inc/wet/matlab.hpp#L435) | Design discrete LQR from continuous-time system via discretization |
| [`margin`](inc/wet/matlab.hpp#L606) | Gain and phase margins of a SISO loop over a frequency grid |
| [`null`](inc/wet/matlab.hpp#L257) | MATLAB short alias for an orthonormal null-space basis |
| [`pidtune`](inc/wet/matlab.hpp#L522) | PID controller tuning using frequency domain method |
| [`pinv`](inc/wet/matlab.hpp#L248) | MATLAB short alias for the Moore–Penrose pseudoinverse |
| [`place`](inc/wet/matlab.hpp#L373) | Robust multi-input pole placement (MATLAB's place) |
| [`pole`](inc/wet/matlab.hpp#L583) | MATLAB short alias for the open-loop poles of a system |
| [`reg`](inc/wet/matlab.hpp#L308) | Form dynamic regulator from system, state-feedback gain, and estimator gain |
| [`svd`](inc/wet/matlab.hpp#L234) | MATLAB short alias for the singular value decomposition |
| [`tf`](inc/wet/matlab.hpp#L31) | MATLAB-style transfer function constructor |

## Math backends

Internal, compile-time-selected implementations of the `wet::` scalar-math surface (`sin`/`cos`/`sqrt`/`exp`/…), chosen via `wet/config.hpp`. Every backend exposes the same functions, so they're listed once here as files rather than repeated in the tables above. The public dispatcher is [`wet/math/math.hpp`](inc/wet/math/math.hpp).

| File | Role |
| ---- | ---- |
| [`wet/math/math_backend.hpp`](inc/wet/math/math_backend.hpp) | Pluggable runtime math backend selection (freestanding-safe) |
| [`wet/math/std_fallback.hpp`](inc/wet/math/std_fallback.hpp) | Composable std:: (<cmath>) base for the hosted math backends |
| [`wet/math/wet_backend.hpp`](inc/wet/math/wet_backend.hpp) | Fast single-precision float math backend (trig.hpp) over the std:: fallback |
| [`wet/math/series_backend.hpp`](inc/wet/math/series_backend.hpp) | Freestanding runtime math backend: routes scalar math to the constexpr series in constexpr_math.hpp, pulling no hosted headers (no <cmath>) |
| [`wet/math/constexpr_math.hpp`](inc/wet/math/constexpr_math.hpp) | Compile-time scalar math: series / Newton-Raphson implementations |
| [`wet/math/trig.hpp`](inc/wet/math/trig.hpp) | Fast float sine and cosine with full-range wrapping |

## Examples

Runnable programs in `examples/` (25 total). Build with `make` (or `tup --quiet examples`); outputs go to `examples/build/`.

| Example | Description |
| ------- | ----------- |
| [`example_buck_converter.cpp`](examples/example_buck_converter.cpp) | Buck converter |
| [`example_cart_pole.cpp`](examples/example_cart_pole.cpp) | Inverted pendulum on cart (cart-pole) LQR control example |
| [`example_coordinated_joints.cpp`](examples/example_coordinated_joints.cpp) | Coordinated joints |
| [`example_corexy_move.cpp`](examples/example_corexy_move.cpp) | Corexy move |
| [`example_encoder_velocity.cpp`](examples/example_encoder_velocity.cpp) | Encoder velocity |
| [`example_eskf_arduino.cpp`](examples/example_eskf_arduino.cpp) | Eskf arduino |
| [`example_foc.cpp`](examples/example_foc.cpp) | PMSM current-loop (FOC) electrical sim: PI vs I-P on a step + disturbance |
| [`example_input_shaper_resonance.cpp`](examples/example_input_shaper_resonance.cpp) | Input shaper resonance |
| [`example_inverted_pendulum.cpp`](examples/example_inverted_pendulum.cpp) | Continuous-time state-space control of the cart-pendulum, replicating the classic University of Michigan CTMS example: "Inverted Pendulum: State-Space Methods for Controller Design" https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace |
| [`example_kinematic_maps.cpp`](examples/example_kinematic_maps.cpp) | Kinematic maps |
| [`example_lpf.cpp`](examples/example_lpf.cpp) | Lpf |
| [`example_lqr_pendulum.cpp`](examples/example_lqr_pendulum.cpp) | Lqr pendulum |
| [`example_math_backend.cpp`](examples/example_math_backend.cpp) | Pluggable math backend example |
| [`example_pendulum_sim.cpp`](examples/example_pendulum_sim.cpp) | Pendulum sim |
| [`example_pmac_bus_limit.cpp`](examples/PMAC/example_pmac_bus_limit.cpp) | Leaf demo: DC-bus current/power/regen limiting and the UV/OV gate |
| [`example_pmac_calibration.cpp`](examples/PMAC/example_pmac_calibration.cpp) | Leaf demo: recover phase R and L by RLS with PRBS injection |
| [`example_pmac_estimator.cpp`](examples/PMAC/example_pmac_estimator.cpp) | Leaf demo: the [theta, omega, tau_load] Kalman estimator in isolation |
| [`example_pmac_servo.cpp`](examples/PMAC/example_pmac_servo.cpp) | End-to-end PMAC servo: bandwidth tuning, R/L calibration, and the three control modes on an average dq + mechanical plant |
| [`example_pmac_thermal.cpp`](examples/PMAC/example_pmac_thermal.cpp) | Leaf demo: FET junction-temperature estimation and thermal derating |
| [`example_servo_commands.cpp`](examples/example_servo_commands.cpp) | Servo commands |
| [`example_servo_drive.cpp`](examples/example_servo_drive.cpp) | Servo drive |
| [`example_swashplate_stsmc.cpp`](examples/example_swashplate_stsmc.cpp) | Swashplate stsmc |
| [`example_trajectory_gallery.cpp`](examples/example_trajectory_gallery.cpp) | Trajectory gallery |
| [`example_workflow_end_to_end.cpp`](examples/example_workflow_end_to_end.cpp) | Workflow end to end |
| [`servo_sim.cpp`](examples/servo_drive/servo_sim.cpp) | Servo sim |
