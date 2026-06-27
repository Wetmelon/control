# wet Library Roadmap

Status: living · Owner: controls core

Forward-looking plan — **what's next**, not what exists. Companion docs:

- **What's in the box** → [../../README.md](../../README.md) (the inventory of shipped features).
- **Why it's built this way** → [decisions.md](decisions.md) (design decisions + rationale).

Items carry a stable ID (`#N`) used by cross-references in source and the other docs;
the numbers are not sequential or a ranking — order on the page is build priority.
Status: `◐` partially built (remaining work listed) · `☐` planned. Done items have
moved to the README inventory.

Every addition follows the three-tier pattern ([decisions.md](decisions.md) D3):
constexpr `design::` function → result struct (`.as<U>()`, `bool success`) → runtime.

## Constraints on new work

The foundational design rules live in [decisions.md](decisions.md) (D1–D6): no mandatory
third-party dependency, universal plant types (Tier 1/Tier 2), the three-tier pattern,
embedded-vs-host placement, the ETL scope boundary, explicit failure signaling. New
items must hold to them. Item-specific reminders:

- Scalar support for `float`, `double`, `wet::complex<float>`, `wet::complex<double>` where applicable.
- Compile-time **and** runtime validation tests for each new API; one test TU per new header.
- `make embedded-check` must stay green — keep `wet/control.hpp` vs `wet/toolbox.hpp` placement correct.
- Tuning/adaptation APIs must not require runtime coupling to identification internals; model inputs are optional and disable-able.

## Architecture layers & build priority

Build bottom-up; each layer depends only on the ones beneath it. Layers 0–1 (generic
plumbing via ETL; linear algebra, constexpr math, LTI types, filters, embedded
primitives) and most of Layer 2 are **done** (see the README inventory). Remaining work
is concentrated in:

- **Layer 2** — the identification / model-builder infrastructure (#3).
- **Layer 3** — advanced/adaptive/optimization-based controls: online tuning (#7),
  harmonic detection front-end (#9), LPV (#10), H∞ (#13), MPC (#14), MHE (#15),
  serial-arm closed-form IK (#23), power-converter modulation (#24).
- **Layer 4** — host-only tooling: multi-rate simulation harness (#18), the host-side
  identification / FRF parts of #3.

## Roadmap

### ◐ Identification and excitation infrastructure (#3)

The commissioning workflow: excite the plant, log the response, identify a model,
synthesize gains, deploy. The user describes the system as a `StateSpace` /
`TransferFunction` (directly or via Tier-2 builders), and identification produces those
same universal types from time-series data.

**Done:** excitation generators (`estimation/excitation.hpp` — Chirp/PRBS/StepTrain/Ramp/MultiSine + `synthesize_*`).

**Remaining:**

- **Generic `Cascade` + unified controller/observer concept** (on-target, embeddable). `Cascade<Outer, Inner>` composes any two controllers satisfying a small protocol (`T control(T r, T y)`, `void reset()`); outer output → inner reference. The protocol is the planned **unified controller + observer concept** (one box every controller and observer fits, SISO + MIMO/`StateSpace`), not a narrow one-off. Tier-2 alias `CascadePPI<T>`.
- **Tier 2 model builders** (`models.hpp`, embeddable — construct universal types only): `models::single_mass`, `second_order`, `two_mass` → `StateSpace`, `fopdt` → `TransferFunction` (Padé dead-time). Downstream design never knows it came from a builder. **Each builder is named for the specific physical system, not a vague category** — the existing `power/mechanical_ss` (rotational inertia + viscous damping + load-torque disturbance, *no* spring) is mis-named and becomes e.g. `rotational_inertia_load`; its compliant successor (motor + load inertias + shaft spring/damper — the resonant drivetrain) is the rotational counterpart of `two_mass`, accurately a `rotational_coupled_mass_spring_damper`. Rotational/servo builders mirror the translational family with rotational, physically-explicit names.
- **Host-side identification** (`analysis/identification.hpp`, `toolbox.hpp`): `tfest<Nn,Nd>` (output-error/PEM), `ssest<NX>` (N4SID subspace), `validate` (cross-validation), plus Tier-2 wrappers `identify_fopdt_from_step`, `identify_two_mass`, etc. — each returns both a named-parameter struct and the universal type. Reuse the model structs already in `estimation/identification.hpp`; online RLS (`estimation/recursive_least_squares.hpp`) stays embedded. A **recursive polynomial estimator** (ARX/ARMAX/OE on-line, the Sys-ID Toolbox "Recursive Polynomial Model Estimator") is the embeddable companion — a thin layer over the shipped RLS.
- **Batch linear estimator family** (audited against the Sys-ID Toolbox, all sharing the NLS/least-squares engine below): `arx`/`iv4` (linear-regression / instrumental-variable, the cheap first pass), `armax`/`oe`/`bj`/`polyest` (PEM polynomial models), `greyest` (linear grey-box — fit a physical-parameter ODE; the host counterpart of the Tier-2 builders), and `procest`/`delayest` (continuous process model + dead-time, feeding the `fopdt` builder). `era`/`impulseest` give the impulse-response / realization route.
- **Validation & order selection**: `compare`/`resid`/`goodnessOfFit` + the `aic`/`fpe`/`pexcit` criteria, and `selstruc`/`arxstruc` for model-order search — the workflow around the bare `validate` above.
- **Shared nonlinear-least-squares core** (the `lsqnonlin`/`lsqcurvefit` equivalent — the only Optimization-Toolbox primitive this library actually needs): a small Gauss-Newton / Levenberg-Marquardt solver over the existing `solve`/`pinv` factorizations, used by the output-error/PEM fit above and any curve fit. Linear solve (`matrix/solve.hpp`) and linear least-squares (`pinv`/SVD, `RLS`) are already shipped; QP/NLP live with MPC/MHE (#14/#15); general optimization frameworks and global optimizers (GA/PSO/SA/surrogate) stay out of scope — the tuning methods here are deliberately method-specific (relay autotune, IFT #7, ESC #8).
- **FRF / nonparametric estimation** (`analysis/frf.hpp`, `toolbox.hpp`): `frfest(time, u, y, freqs)` → magnitude/phase/coherence, Welch-style averaging; pairs with Chirp/MultiSine. Companions: `spa`/`etfe` (spectral-analysis FRF) and `cra` (correlation impulse response).
- **Out of scope** (the ML / framework parts of the Sys-ID Toolbox): ML-based nonlinear ID — neural state-space (`idNeuralStateSpace`/`nlssest`), Gaussian-process / SVM / tree-ensemble / sigmoid / wavelet network nonlinearities (`nlarx`/`nlhw` mapping objects) — these need a deep-learning/ML stack, a different product. Nonlinear *grey-box* (`nlgreyest` — fit a known physical nonlinear ODE) is the in-scope exception, a nonlinear extension of `greyest` over the NLS core. Also out: the `iddata` container + plotting + `systemIdentification` app, and the identified-model object framework with parameter covariance (`idtf`/`idss`/`idpoly` — wet uses the plain universal types).
- References: Ljung, "System Identification: Theory for the User," 2nd ed., 1999; Van Overschee & De Moor, "Subspace Identification for Linear Systems," 1996; Pintelon & Schoukens, "System Identification: A Frequency Domain Approach," 2nd ed., 2012.
- Acceptance: `tfest`/`ssest` recover known plants on noiseless synthetic data within 5%; Tier-2 wrappers recover physical parameters within 10% from chirp/step data with realistic encoder noise; `make embedded-check` stays green (only excitation/cascade/`models.hpp` reach the embeddable umbrella).

### ◐ Online PID tuning (relay + IFT) (#7)

Online-first PID/PI tuning that runs in the closed loop without time-series export — the
"no toolchain, no laptop" complement to the full identification pipeline (#3). First
method (relay autotuner) is done with the Tyreus-Luyben default + shared safety policy
([decisions.md](decisions.md) D14).

**Done:** Åström–Hägglund relay autotuner (`design/relay_autotune.hpp`) — extracts `(Kᵤ, Tᵤ)` from a relay-induced limit cycle, plugs into the `pid_design.hpp` tuning rules.

**Remaining:**

- **Biased / asymmetric relay + AMIGO.** Run the relay with asymmetric magnitudes so the limit cycle is asymmetric; track average `u`/`y` to recover static gain `Kₛ ≈ ū/ȳ`, returning `(Kᵤ, Tᵤ, Kₛ)`. Pairs with `design::amigo_kappa_tau(Ku, Tu, Ks, Ts)` (constrained to Ms ≤ ~1.4), which out-performs ZN/Tyreus-Luyben across the FOPDT family. ~30-line extension to `RelayAutotuner` plus the AMIGO formula. Refs: Åström & Hägglund, "Revisiting the Ziegler-Nichols step response method for PID control," J. Process Control 14(6), 2004, https://doi.org/10.1016/j.jprocont.2004.01.002; *Advanced PID Control*, ISA, 2006, ch. 6 & 8.
- **IFT (iterative feedback tuning).** Gradient-based model-free tuning; perturbs the closed-loop gains and estimates the tracking-cost gradient from 2–3 experiments per iteration. `design::synthesize_ift(reference, cost_weights, learning_rate)` → `IFTResult`/`IFTRuntime` with bumpless transfer between iterations. Refs: Åström & Hägglund, Automatica 20(5), 1984, https://doi.org/10.1016/0005-1098(84)90014-1; Hjalmarsson et al., "Iterative feedback tuning," IEEE CSM 18(4), 1998, https://doi.org/10.1109/37.710876.
- **Frequency-response PID autotuners** (the Simulink Control Design Closed-Loop / Open-Loop / Gain-Scheduled PID Autotuner blocks): inject a perturbation (sinestream/PRBS from `excitation.hpp`), estimate the loop FRF at a few frequencies (`frfest`, #3), and set gains for a target bandwidth + phase margin — the online, real-time cousin of the host `pidtune`. The gain-scheduled variant tunes at several operating points and fills a `PID Gain Scheduler` table (→ #10/#30).
- **VRFT (Virtual Reference Feedback Tuning).** One-shot data-driven tuning of a linearly-parameterized controller from a single input/output record — no plant model, no iteration (unlike IFT). `design::synthesize_vrft(io_data, reference_model, controller_basis)`. Ref: Campi, Lecchini & Savaresi, "Virtual reference feedback tuning," Automatica 38(8), 2002, https://doi.org/10.1016/S0005-1098(02)00032-8.
- **Safety policy (shared, every tuning runtime):** command-amplitude clamps, slew limits, output-saturation passthrough, bumpless transfer, rollback to pre-tuning gains on degraded measurement / timeout.
- Acceptance (remaining): IFT convergence with bounded excitation; safety-policy tests (clamps, rate limits, bumpless transfer, rollback) for both runtimes.

### ◐ Harmonic detection & suppression — anti-chatter (#9)

Online dominant-harmonic detection and suppression. Targets: lathe-turning chatter,
spindle-tool resonance, gearbox tonal vibration.

**Done:** spectral primitives (`filters/spectral.hpp` — Goertzel + harmonic analyzer/THD), the self-tuning SOGI-FLL tracker (`filters/sogi.hpp`), and the selective PR-resonator suppression bank (`controllers/harmonic_suppression.hpp`).

**Remaining:**

- **Host-side `analysis::detect_dominant_harmonics(signal_window, sample_rate)`** — auto-pick the harmonic set to suppress; wire `HarmonicAnalyzer` output into `synthesize_harmonic_suppressor`.
- **Turning/milling chatter front-end** `design::synthesize_chatter_suppressor_turning(machine_model, spindle_speed, sensor_config)`.
- **Adaptive online retune** as the detected resonance drifts.
- References: Altintas & Budak, "Analytical Prediction of Stability Lobes in Milling," CIRP Annals, 1995, https://doi.org/10.1016/S0007-8506(07)62342-7; Mojiri & Bakhshai, "An Adaptive Notch Filter for Frequency Estimation," IEEE TAC, 2004, https://doi.org/10.1109/TAC.2003.822862.
- Open decision: continuous adaptation vs gated updates ([decisions.md](decisions.md) O2).

### ☐ LPV gain-scheduled LQG/LQI (#10)

Gain scheduling over an operating-point grid. Targets: UAV dynamics vs airspeed/altitude,
vehicle lateral dynamics vs speed, manipulators with configuration-dependent models.

- Workflow: local linear models over the grid → local controller/estimator gains → scheduled runtime bundle with interpolation. Artifacts: schedule map, gain tables, local closed-loop analysis.
- **Scheduled-controller realizations** (audited against the Aerospace Blockset 1D/2D/3D controller blocks — these are the concrete runtime forms the scheduled bundle should offer, all reusing the LTV/"Varying" wrapper from #29 and the matrix-valued `Lut` from #30):
  - 1-/2-/3-parameter scheduled state-space controller `[A(v),B(v),C(v),D(v)]` — interpolate the realization matrices over the schedule grid (the "Interpolate Matrix(x[,y,z])" blocks are exactly a matrix-returning `Lut1D`/`Lut2D`).
  - **Observer-form** realization `[A(v),B(v),C(v),F(v),H(v)]` — schedule an observer+state-feedback controller (pairs with `observer.hpp`).
  - **Self-conditioned form** — Hanus-style anti-windup/bumpless realization of an arbitrary state-space controller, so switching/saturating between scheduled gains doesn't wind up the dynamics (generalizes the PID back-calculation already in `PIDController` to any `StateSpace` controller; useful on its own, not just for scheduling).
  - **Controller blend** `u = (1−L)·K1·y + L·K2·y` — output-blend interpolation between adjacent scheduled controllers; and gain-scheduled lead-lag as the scalar special case of the above.
- References: Rugh & Shamma, "Research on Gain Scheduling," Automatica, 2000, https://doi.org/10.1016/S0005-1098(00)00058-3; Apkarian & Gahinet, IEEE TAC, 1995, https://doi.org/10.1109/9.384219; Hanus, Kinnaert & Henrotte, "Conditioning technique… the conditioned controller," Automatica 23(6), 1987 (self-conditioned form).
- Acceptance: continuity across transitions; stability over the certified envelope.

### ☐ H-infinity output feedback (#13)

Output-feedback synthesis for weighted generalized plants. Targets: flexible structures
with modal uncertainty, flight control with structured uncertainty, active
suspension/vibration isolation. Host-only (`toolbox.hpp` — gamma search allocates/iterates,
[decisions.md](decisions.md) D16).

- Interface: `design::synthesize_hinf(augmented_plant, weighting_filters, gamma_search_bounds)` → `HInfResult` / `HInfArtifacts` + S/T analysis models. Workflow: coupled Riccati solves over gamma; select feasible controller.
- **Synthesis methods** (audited against the Robust Control Toolbox — the Riccati-based, embeddable-design slice; all produce a plain `StateSpace` controller):
  - `mixsyn` / `augw` — weighted **mixed-sensitivity** S/KS/T loop shaping (the common entry point), with `makeweight` for the W₁/W₂/W₃ weighting filters.
  - `h2syn` — H2 optimal on the generalized plant (the LQG-equivalent in generalized-plant form; reuses `care`/`dare`).
  - `loopsyn` / `ncfsyn` (Glover–McFarlane) — loop-shaping design, with `ncfmargin` and left/right coprime factorization (`lncf`/`rncf`) for the normalized-coprime robustness margin.
  - `sdhinfsyn` (sampled-data H∞, pairs with #18) and `hinfgs` (gain-scheduled H∞ → ×#10) as later variants.
- **Out of scope — the robust-analysis/SDP framework** (heavy host tooling, same boundary as the optimization framework in [[project-optimization-scope-boundary]]): the uncertain-systems modeling layer (`uss`/`ureal`/`ultidyn`/`umat`/`ucover`/`lftdata`/`usample`), the LMI/SDP machinery (`lmivar`/`lmiterm`/`feasp`/`mincx`/`gevp`), μ-synthesis and worst-case analysis (`musyn`/`dksyn`/`mussv`/`robstab`/`robgain`/`wcgain`/`wcnorm`), and the fixed-structure tuning framework (`systune`/`slTuner`/`hinfstruct`). H∞ + weighting filters on a known generalized plant is the embeddable-design sweet spot; structured-uncertainty μ-synthesis is a different product. (`fitfrd`/`genphase`, FRF→model fitting, tie #3.)
- References: Doyle et al., "State-Space Solutions to Standard H2 and H-infinity Control Problems," IEEE TAC, 1989, https://doi.org/10.1109/9.29425; Glover & Doyle, Systems & Control Letters, 1988, https://doi.org/10.1016/0167-6911(88)90055-2; McFarlane & Glover, "A loop-shaping design procedure using H∞ synthesis," IEEE TAC 37(6), 1992.
- Acceptance: weighted robust-stability/performance; regression vs known references.

### ☐ Constrained MPC (#14)

Finite-horizon constrained control with a deterministic runtime iteration budget. Targets:
multivariable process control with limits, AV trajectory tracking with actuator limits,
power electronics with current/voltage bounds. Host-only (`toolbox.hpp` — QP solver
allocates/iterates, [decisions.md](decisions.md) D16).

- Interface: `design::synthesize_mpc(plant, horizon, weights, constraints, solver_budget)` → `MPCResult` / `MPCArtifacts` + fixed-iteration warm-start runtime.
- **QP solver** (host-side, allocates/iterates): start with one dense active-set or interior-point solver; ADMM and branch-and-bound MIQP (for explicit-integer / hybrid moves) are later variants, not first-cut.
- **Offset-free / disturbance modeling:** augment the prediction model with input/output disturbance states and a Kalman estimator (reuse `estimation/kalman.hpp`); a `cloffset`-style closed-loop DC-gain check confirms zero steady-state offset. This is the `mpc` default integrator-disturbance behaviour.
- **Terminal weights & constraints** (`setterminal`): a terminal cost (e.g. the LQR-optimal `S` from the shipped `care`/`dare`) and/or terminal set on the final horizon state — the standard route to a guaranteed-stability MPC with a short horizon, which is exactly what a memory-/compute-limited target wants.
- **Adaptive / gain-scheduled MPC:** the runtime accepts an updated prediction model per tick (LTV) or switches among a precomputed bank (gain-scheduled), reusing the LTV/LPV model types and the schedule from **#10**.
- **Nonlinear / multistage MPC** (`nlmpc`): host-heavy, needs an NLP solver — a stretch goal layered on the linear core, not first-cut. **Data-driven (subspace predictive) MPC** ties to the identification infra in **#3**.

Audited against the MPC Toolbox; the embedded-relevant additions above are the gaps worth planning. Two callouts:

| Addition | Why it matters for embedded control |
|---|---|
| **Explicit MPC** — `design::generate_explicit_mpc(mpc)` solves the multiparametric QP offline into a piecewise-affine control law (polyhedral regions + per-region affine gains), evaluated on-target by a region lookup with **no solver in the loop**; `simplify` trims regions to fit memory | The real way constrained MPC ships to a microcontroller — deterministic, allocation-free runtime; pairs with `Lut`/region-search primitives |
| **Fixed-iteration implicit runtime** (already in the interface above) | Where explicit MPC's region count explodes, a warm-started solver with a hard per-tick iteration cap keeps the implicit controller real-time-bounded |

Out of scope (no embedded payoff): MATLAB code-gen workflow (`buildMEX`/`mpcmoveCodeGeneration`/`getCodeGenerationData`/`createParameterBus`/`generateJacobianFunction` — this library *is* the C++ implementation), object plumbing (`get`/`set`/`getname`/`mpcprops`/`mpcverbosity`/`size`/`review`/`compare`/`sensitivity`/`plot`/`*simopt`/`mpcDesigner` GUI, deprecated `mpcqpsolver`), and the reference-application subsystems (adaptive cruise, lane-keeping, path-following/-planning).

- References: García, Prett & Morari, "Model Predictive Control: Theory and Practice — A Survey," Automatica, 1989, https://doi.org/10.1016/0005-1098(89)90002-2; Qin & Badgwell, Control Engineering Practice, 2003, https://doi.org/10.1016/S0967-0661(02)00186-7; Bemporad, Morari, Dua & Pistikopoulos, "The explicit linear quadratic regulator for constrained systems," Automatica 38(1), 2002, https://doi.org/10.1016/S0005-1098(01)00174-1 (explicit MPC).
- Acceptance: deterministic per-tick runtime bound; constraint-satisfaction regression; tracking/regulation tests; explicit-MPC PWA law matches the implicit solver's moves region-by-region and runs solver-free on-target.

### ☐ Moving horizon estimation (MHE) (#15)

Optimization-based constrained estimator; the estimation counterpart to MPC. Host-only
(allocates/solver, [decisions.md](decisions.md) D16).

- Reference: Rao, Rawlings & Mayne, "Constrained State Estimation for Nonlinear Discrete-Time Systems," IEEE TAC, 2003, https://doi.org/10.1109/TAC.2003.812777.
- Acceptance: matches Kalman on linear/unconstrained problems; respects state constraints on nonlinear references.

### ☐ Distributed-IMU pose estimation for articulated robots (#25)

Estimate the configuration (joint angles, end-effector pose) of a serial or
parallel manipulator from an IMU mounted on each link, fusing the inertial
measurements with the known kinematics rather than (or alongside) joint encoders.
Builds on the shipped attitude fusion (`estimation/sensor_fusion.hpp`,
`estimation/eskf.hpp`) and the kinematics chains (`kinematics/serial_arm.hpp`
DH chains, `kinematics/stewart.hpp`).

- **Per-link orientation → joint angle:** each link's IMU gives its orientation in the world frame; the relative rotation between consecutive links' frames yields the joint angle directly for revolute joints. Reuse `Quaternion`/`geometry.hpp` for the relative-rotation math; the DH frame assignments in `serial_arm.hpp` define which axis each joint rotates about.
- **Kinematic fusion (the real win):** a filter that couples all link IMUs through the chain constraint — gyro rates relate to joint velocities via the geometric Jacobian (already in `serial_arm.hpp`), and gravity direction per link disambiguates absolute angle. This bounds the per-link integration drift that standalone attitude filters can't, and gives encoderless or encoder-redundant joint sensing.
- **Parallel mechanisms:** the loop-closure constraints (Stewart platform, five-bar) over-determine the pose from the distributed IMUs — fusion must respect the closed-loop constraint, not treat legs independently.
- **External pose references (aiding):** the same estimator must accept absolute references that bound the inertial drift, fused as measurement updates — optical tracking (reflector/prism balls, total stations, motion-capture markers → end-effector or link pose), laser-plane references (excavator/dozer grade-control: a laser receiver gives blade/boom height against a reference plane), and ranging (sonar/ultrasonic blade-height on graders/dozers). Each is just another measurement model into the kinematic filter — a partial pose, a height/elevation scalar, or a single link's position — not a separate pipeline. The machine-control cases (excavator boom, grader/dozer blade) are the headline industrial driver.
- Open question: how much this leans on the planned unified observer concept (#3) vs. a kinematics-specific estimator; whether encoders are fused in or replaced.
- Reference: El-Gohary & McNames, "Human Joint Angle Estimation with Inertial Sensors and Validation with a Robot Arm," IEEE T-BME 62(7), 2015, https://doi.org/10.1109/TBME.2015.2403368; Seel, Raisch & Schauer, "IMU-Based Joint Angle Measurement for Gait Analysis," Sensors 14(4), 2014. Implementation inspiration for the absolute-reference + ESKF measurement update: https://github.com/madcowswe/ESKF (15-state position-aided ESKF) — a starting point, not the ceiling.
- Acceptance: on a simulated arm with known joint trajectory, per-link IMU streams reconstruct the joint angles (and end-effector pose via forward kinematics) within tolerance; the fused estimator bounds drift that per-link attitude-only filters accumulate, and an external pose/height reference (optical, laser-plane, or sonar) further bounds the absolute error when supplied.

### ☐ GPS + IMU navigation fusion (#26)

Loosely-coupled INS/GNSS: strapdown mechanization (gravity-compensate accel in
the nav frame, double-integrate to velocity/position) aided by GPS position/
velocity fixes to bound the integration drift. Extends the shipped attitude
fusion (`estimation/eskf.hpp`) from orientation-only to full navigation state.

- **Mechanization core** (embeddable): propagate position/velocity from bias-corrected accel + the existing orientation estimate; reuse `Quaternion`/`geometry.hpp`. Pure dead-reckoning drifts unbounded — documented, not hidden.
- **Aided ESKF** (15-state: attitude, velocity, position, gyro+accel bias): generalize the 6/9-state `ErrorStateKalmanFilter` to take position/velocity aiding (GPS fix, or ZUPT zero-velocity updates as the cheap GPS-denied fallback). Reuse the existing error-state inject/reset machinery; nominal state stays external.
- Loosely-coupled first (GPS reports a position/velocity solution); tightly-coupled raw-pseudorange fusion is out of scope.
- Reference: Sola, "Quaternion kinematics for the error-state Kalman filter," 2017, https://arxiv.org/abs/1711.02508; Groves, *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd ed., 2013. Implementation inspiration: https://github.com/madcowswe/ESKF (15-state position-aided ESKF).
- Acceptance: with a noiseless IMU the mechanization round-trips a known trajectory; the aided ESKF bounds position error under a drift-inducing bias when GPS/ZUPT updates are supplied, and degrades to free-running INS drift when they are withheld.

### ◐ Serial N-DOF manipulator kinematics — closed-form IK (#23)

The N-generic numerical foundation is **done** (`kinematics/serial_arm.hpp` — DH chains,
geometric Jacobian, damped-least-squares IK for any N≤6, manipulability/singularity flags;
plus prismatic joints, SCARA, and the parallel five-bar in `kinematics/scara.hpp`).

**Remaining — the spherical-wrist (Pieper) closed-form (`N == 6` only):** the headline
path for classic industrial 6-axis arms (PUMA-class). IK decouples — first three joints
solve the wrist-centre position, last three solve orientation — enumerating up to **8**
branches (shoulder L/R × elbow up/down × wrist flip), filtered to joint limits.
`SerialArm<6,T>::inverse → {array<JointSet,8> solutions, uint8 count}`, with the existing
`select_nearest` for continuous branch tracking. The `spherical_wrist` flag (Pieper's
criterion) already gates its future auto-selection; the DLS path already solves `N==6`
numerically, so this is a speed/branch-completeness refinement, not a capability gap. The
general 6R case (16th-degree Raghavan–Roth) stays out of scope analytically.

- References: D. L. Pieper, "The Kinematics of Manipulators under Computer Control," PhD thesis, Stanford, 1968; Raghavan & Roth, "Inverse Kinematics of the General 6R Manipulator," ASME J. Mech. Des. 115(3), 1993, https://doi.org/10.1115/1.2919218; Craig, "Introduction to Robotics," 3rd ed., 2005.
- Acceptance: closed-form inverse round-trips against forward for each returned branch and enumerates the correct solution count, filtered to joint limits; `select_nearest` tracks a continuous branch without flips.
- Open decisions: DH convention (standard vs modified/Craig — expose both?); how branch identity (shoulder/elbow/wrist tags) is reported.

### ☐ Multi-rate simulation harness (#18)

Host-only (`toolbox.hpp`). The current `simulation/simulate.hpp` advances every block at a
single fixed `Ts`; real deployments are multi-rate (an ISR-level inner loop at 8 kHz–128+
kHz, RTOS outer loops at 1–1000 Hz). A faithful closed-loop sim must reproduce the rate
hierarchy and its inter-rate effects — sample-and-hold across the rate boundary, the
one-sample transport delay an outer task sees on inner-loop state, aliasing of fast
dynamics into slow samplers. This is where most "works in sim, oscillates on hardware"
gaps come from.

- **Model.** Each discrete block runs at an integer divisor of the fastest (base) rate; the harness ticks at the base rate and fires slower blocks on their decimation boundary. The plant is *not* a discrete block — it integrates on a much finer sub-step (e.g. 1 µs), advanced to each control-tick boundary where it is sampled/held.
- **Rate-boundary semantics.** Every crossing is sample-and-hold: ZOH from slower producer to faster consumer and from the fastest discrete block onto the continuous plant; latch/decimate faster→slower. Make the outer→inner reference handoff and inner→outer state feedback explicit so the simulated transport delay matches the RTOS.
- **Relationship to existing code.** Generalizes `simulate.hpp` (single-rate = one block, base rate = block rate); reuses `simulation/integrator.hpp` for the plant sub-step; pairs with the cascade controllers (#3).
- Open questions: schedule spec (per-block divisor vs explicit Hz with divisibility validation); model task jitter or assume ideal periodic firing (ideal first); plotting channels sampled at different rates.
- Reference: Franklin, Powell & Workman, "Digital Control of Dynamic Systems," 3rd ed., 1998, multirate sampling chapter.
- Acceptance: a two-rate example (fast inner current loop + slow outer loop) reproduces the inner-loop transport delay and sample-and-hold seen on hardware; single-rate results match `simulate.hpp` when all blocks share one rate.

### ◐ Power converter modulation & control — DC-DC, PFC, multilevel (#24)

Modulator and control blocks for the standard switching-converter topologies. Gating/
modulation leaf utilities in `utility/modulation.hpp` (pure constexpr duty/carrier math);
control laws in `controllers/` on the existing PI/PR/observer infrastructure; averaged
converter models as Tier-2 `StateSpace`/`TransferFunction` builders.

**Done:** continuous **SVPWM** via min-max zero-sequence injection (`svpwm_zero_sequence`, `svm_duty_cycles`) with the carrier ⇔ space-vector equivalence.

**Remaining:**

- **Cross-cutting modulator primitives:** trailing/leading/center-aligned PWM carrier comparison; complementary gating + dead-time insertion and compensation; N-phase interleaving (360°/N carrier offsets).
- **Three-phase VSI schemes:** SPWM (baseline), THIPWM (closed-form 1/6 injection), the DPWM family (DPWMMAX/MIN, DPWM0–3, GDPWM — clamp 60° windows, ~33% switching-loss cut), overmodulation (Mode I/II → six-step), random/spread-spectrum PWM (low priority).
- **DC-DC converter abstractions** (audited against the Simscape Electrical converter blocks — make these first-class named pieces, not loose control laws). Two layers, each a thin reuse:

  *Averaged plant models* (Tier-2 → `StateSpace`/`TransferFunction`, CCM state-space-averaged, embeddable):

  | Model | Plan |
  |---|---|
  | `buck_model` / `boost_model` / `buck_boost_model` | Small-signal control-to-output `Gvd(s)` + input-to-output for the three canonical topologies (boost/buck-boost carry the RHP zero — surface it, don't hide it) |
  | bidirectional / four-switch buck-boost, chopper (1/2/4-quadrant) | Same averaged core with the quadrant/direction sign convention |

  *Controller abstractions* (cascaded current+voltage loops over `PIDController`):

  | Controller | Plan | Reuse |
  |---|---|---|
  | DC Current Controller | Inner current PI tuned to the L/R pole, with integral anti-windup | `PIDController` (`Kbc` anti-windup) + `current_loop_pi_gains`-style placement |
  | DC Voltage Controller / DC-DC Voltage Controller | Outer voltage PI with feedforward zero cancellation + anti-windup | `PIDController` + the averaged model above for the feedforward term |
  | Duty feedforward | `D = Vo/Vi` (buck), boost/buck-boost ratios — added ahead of the PI | closed-form, constexpr |
  | Peak/average current-mode + slope compensation; MPPT (P&O, Inc-Conductance — overlaps ESC #8); interleaved boost | the current-mode and MPPT laws on top of the above | — |

  The **d-q Voltage Limiter** (clamp the dq command to the SVPWM circle) is the AC/inverter-side analog and lives with FOC in **#31** (`voltage_circle_radius`), not here.
- **PFC / AC-DC:** boost PFC with average-current-mode (inner current shapes inductor current to a rectified-sine reference, outer DC-bus loop); totem-pole bridgeless PFC (HF GaN/SiC leg + line-frequency unfolding leg, zero-cross commutation sequencer).
- **Grid-tie / renewables** (the Simscape "Renewables Control" + droop blocks): grid-following inverter control (SOGI/DSOGI-PLL synchronization + dq current control + DC-bus loop — reuses the shipped PLL/SOGI and FOC current loop), the Solar PV grid-following controller (PFC-style current shaping + MPPT from the DC-DC group), frequency/voltage **droop** (incl. SM Governor with Droop — a speed-droop genset loop), and an impedance-scan stability probe (extends `analysis::impedance`). Grid-forming and full power-system gear stay out of scope.
- **Multilevel:** 3-level NPC (phase-disposition carriers or 3-level SVM + neutral-point balancing); ANPC as the loss-balancing extension (do NPC first).
- References: Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd ed., 2020 (DC-DC, PFC, current-mode); Holmes & Lipo, *Pulse Width Modulation for Power Converters*, IEEE Press, 2003 (inverter PWM, multilevel); Hava et al., IEEE T-PEL 14(1), 1999 (CPWM/DPWM/GDPWM).
- Acceptance (per piece): duty feedforward matches the CCM conversion ratio; center-aligned/interleaved carriers give the expected phase/ripple relationships; MPPT converges to the true MPP under irradiance steps (matches ESC on a smooth P-V curve); boost-PFC current tracks the rectified-sine template at unity displacement factor; NPC neutral-point stays balanced; ANPC switch-state selection equalizes per-device loss.

### ☐ Motor-drive control & sensorless FOC (#31)

Completes the field-oriented drive stack on top of the shipped FOC core (`power/foc.hpp`
`FOController` with cross-axis decoupling + back-EMF feedforward, `current_loop_pi_gains`),
the Clarke/Park family (`power/transforms.hpp`), SVPWM (`power/modulation.hpp`), and the
encoder/PLL/SOGI-FLL front-ends. Audited against the Motor Control Blockset. Same rule as
#29: each entry is a thin runtime/`design::` piece reusing the existing transforms, PI loops,
observers, and `Lut1D`/`Lut2D` — not a parallel framework. On-target embeddable unless noted.

**Sensorless rotor position / speed estimation** (the headline gap — encoderless drives):

| Block | Plan | Reuse |
|---|---|---|
| Sliding Mode Observer | Back-EMF SMO → electrical angle/speed for SMPMSM | `smc.hpp` sign law + `atan2`; pairs with a PLL/SOGI angle tracker |
| Flux Observer | Stator-flux integrator with drift compensation → angle/torque | `transforms.hpp` αβ, `filters.hpp` washout |
| Extended EMF Observer | Extended-EMF model for salient (I)PMSM angle/speed | observer (`observer.hpp`) + `atan2` |
| Pulsating-HF (HFI) Observer | Low-/zero-speed initial position via HF injection | `sogi.hpp` demod + injection on the dq command |

**Current references / operating point:**

| Block | Plan | Reuse |
|---|---|---|
| MTPA Control Reference | Max-torque-per-amp (id,iq) for salient PMSM/SynRM + field-weakening above base speed | `foc.hpp` `voltage_circle_radius`/base-speed already there; add the id<0 reference law |
| LUT-based PMSM/ACIM/SynRM Control Reference | Precomputed (id,iq) maps vs torque/speed | `Lut2D` (#30) |
| ACIM Control / Slip Speed / Torque, Vector Control Reference | Induction rotor-flux orientation + slip, salient torque estimate | `transforms.hpp`, extends `iq_from_torque` |

**Startup, scalar & commutation:**

| Block | Plan | Reuse |
|---|---|---|
| V/F (VbyF) Controller, I-F Controller | Open-loop scalar / forced-current startup | ramp + `inverse_park` |
| Position Generator / Position Compensation | Fixed-frequency angle ramp / delay-offset correction | trivial |
| Six-Step / Sensorless Six-Step Commutation | BLDC trapezoidal commutation (Hall or back-EMF zero-cross) | new small runtime |
| Hall Speed & Position / Hall Validity | Hall decode → sector/speed + sequence validity | sibling to `QuadratureDecoder` |
| Resolver Decoder | Angle-tracking observer on sin/cos resolver | PLL-style tracker, `atan2` |
| SRM Commutation | n-phase switched-reluctance sequencing (niche, low priority) | new |
| Wound-field synchronous machine (SM) FOC + current-reference generator | Same dq FOC structure as PMSM with an extra field-winding axis; the headline use is gensets/grid machines, lower priority than the drive motors above | `FOController` + `transforms.hpp` |

**Motor plant models** (Tier-2 builders → `StateSpace`/nonlinear plant for simulation, embeddable per [decisions.md](decisions.md)): PMSM (SMPMSM/IPMSM), BLDC (trapezoidal), three-phase Induction, SynRM, wound-field SM, plus averaged-value inverters (PMSM/BLDC). Construct universal types like the other Tier-2 builders in #3; downstream design/sim never knows it came from a builder.

**Parameter estimation** (`Rs`/`Ld`/`Lq`/`RrL`/`Id0`/mechanical inertia+damping for PMSM/ACIM): motor-specific wrappers over the identification infra in **#3** (RLS / excitation already shipped) — returns named parameters and feeds the model builders above.

**Drive design calculators** (host-side `design::`, audited against the `mcb.*` functions — `base_speed`/`voltage_circle_radius` and `current_loop_pi_gains` already shipped): PMSM/ACIM characteristic curves (torque–speed, current/voltage constraint circles), rated torque, max speed and milestone speeds, steady-state `Vd`/`Vq` from operating point — all closed-form extensions of the existing voltage-circle/base-speed math. Plus a **per-unit / SI base-value** helper (`getPUSystemParameters`/`getSISystemParameters` — not in `scaling.hpp` today) and `generateMotorLUT` to bake the MTPA/field-weakening maps into the `Lut2D` references above (#30). The plot halves and the `mcb.internal.launch*App` GUIs stay out of scope; the frequency-domain analysis reuses the shipped `bode`/`margin`.

**FOC Autotuner** (*Field Oriented Control Autotuner* — sequentially tune the current and speed PI loops on-target): a motor-drive front-end over the online-tuning work in **#3/#7** (relay autotuner shipped) seeded by `current_loop_pi_gains`; the *Configuration* blocks (BLDC/PMSM/Induction) are just the parameter sets consumed by the motor-model builders above.

**Modulation / protection** already belongs to **#24**: Dead-Time Compensator, the multi-method PWM Reference Generator (beyond SVPWM), single-shunt reconstruction (Phase Current Extractor, PWM Phase Shift for Single-Shunt FOC), and a Protection Relay (DMT overcurrent trip).

- Acceptance (per piece): each sensorless observer reconstructs rotor angle within tolerance on a simulated PMSM and the closed FOC loop runs encoderless above its valid speed; MTPA reference minimizes current magnitude for a commanded torque on a salient model and stays inside the voltage circle under field weakening; LUT references match the closed-form law at the breakpoints; commutation blocks energize the correct sequence vs. Hall/back-EMF; motor builders round-trip their physical parameters; `make embedded-check` stays green for the on-target pieces.
- References: Sul, *Control of Electric Machine Drive Systems*, IEEE/Wiley, 2011; Krishnan, *Permanent Magnet Synchronous and Brushless DC Motor Drives*, CRC, 2010; Holtz, "Sensorless Control of Induction Machines," Proc. IEEE 90(8), 2002, https://doi.org/10.1109/JPROC.2002.800726; Jung, Kim & Sul, HF-injection sensorless, IEEE T-IA, where applicable.

### ☐ Control & signal-processing abstractions (Simscape Electrical control library) (#32)

Simscape Electrical is acausal physical modeling (out of scope), but its **Control** and
**Sensors/Measurements** sublibraries are causal runtime/design blocks — and most already
map to shipped modules (transforms, `PIDController`, `lead_lag`, `smc`, filters, `pll`/`sogi`,
`spectral`, `encoder`, `thermistor`, `iec61131`) or to the motor/converter items in **#31**
(machine controllers, FOC, DTC, field-weakening, commutation) and **#24** (converter
controllers, multilevel PWM, hysteresis current control). The genuinely-new, *generic*
abstractions worth planning — each a thin runtime/`design::` piece, embeddable:

| Block | Plan | Reuse |
|---|---|---|
| Smith Predictor | Dead-time compensator: model-based predictor that hides transport delay from a PI/PID loop | `PIDController` + a delay/Padé model (#14/#29 delay group) |
| Model Reference Adaptive Controller (MRAC) | MIT-rule / Lyapunov adaptive law driving a controller toward a reference model | sits beside `adrc.hpp`; reuses `PIDController` structure |
| RST polynomial controller | Two-degree-of-freedom pole-placement in `R`/`S`/`T` polynomial form | `TransferFunction` + `place`/Diophantine solve |
| Iterative Learning Control (ILC) | Trial-to-trial feedforward learning for repetitive tasks — update the command from the previous trial's error | the trial-domain cousin of the shipped `repetitive.hpp`; stores one trial buffer |
| Ultra Local Model / model-free (iPID) | Fliess model-free control: estimate the local `ẏ ≈ F + αu` term online and cancel it under a PID — "intelligent PID" | `differentiator.hpp` + `PIDController` |
| Moving statistics — RMS / std / var / min / max | Running-window stats beyond the shipped `MovingAverage`/`MedianFilter` (true-RMS over a sliding/cycle window, plus std/var/min/max) — sensor health, envelope, condition monitoring | running-sum/window runtime in `filters.hpp`; pairs with `spectral.hpp` |
| Moving Average — variable-frequency only | The fixed boxcar `MovingAverage` is shipped (`filters.hpp`); add the variable-window (frequency-locked) variant — the DSC building block for PLLs | extends `filters.hpp` `MovingAverage` |
| LMS / NLMS adaptive filter | The lightweight adaptive filter to sit beside the shipped RLS — adaptive feedforward, active noise/vibration cancellation, adaptive line enhancer / notch | `filters.hpp` tapped-delay line + a normalized step update; mirrors `recursive_least_squares.hpp` |
| Direct-form FIR filter + `fir1` design | General tapped-delay FIR runtime, plus its light coefficient designer `fir1` (window-based: ideal-response × window, the companion every other filter runtime already has — biquads ship with `lowpass_*`/`notch`/Butterworth) | `filters.hpp` (delay line + dot product; `fir1` reuses a window function) |
| Integrator with wrapped state | Angle integrator that wraps to (−π, π] | trivial wrapper over the existing integrator |
| Variable-time-constant LP / variable-frequency 2nd-order filter | Runtime-retunable `filters.hpp` biquads (recompute coefficients each tick) | `filters.hpp` (small extension) |
| Foster / Cauer thermal models | Junction-temperature RC-ladder estimators for power-semiconductor thermal derating | discrete state-space / biquad chain |
| Second-order actuator model (linear + nonlinear) | ω²/(s²+2ζωs+ω²) actuator lag with rate and deflection/position saturation — a realistic actuator for sim/HIL (Aerospace Blockset `*Second-Order Actuator`) | `filters.hpp` 2nd-order core + `SlewLimiter`/bounds; distinct from the `actuator.hpp` command-mapping layer |
| Quaternion SLERP / log (+ mean, random rotation) | Attitude interpolation for attitude trajectories/commands — `geometry.hpp` has conjugate/inverse/normalize/rotate/axis-angle but no `slerp`/`log`/`meanrot` (the Aerospace/UAV `slerp`/`quatinterp`/`log`/`meanrot`/`randrot`). The exp map is already present as `from_axis_angle` (rotation-vector → quaternion); add a `log` inverse + `slerp` built on the exp/log pair | extends `geometry.hpp` `Quaternion`. (The ESKF does *not* need this — it carries only the error-state vector and the user injects δθ into the nominal quaternion via `from_axis_angle` externally.) |
| d-q Voltage Limiter | Clamp the dq voltage command to the SVPWM circle | `foc.hpp` `voltage_circle_radius` (→ #31) |

- Lower priority / niche: multiphase (5-/6-phase) Clarke-Park & decoupled transforms; PMU (three-phase phasor) and impedance-scan measurements (the SOGI-FLL/`analysis::impedance` already cover most of this).
- Out of scope (the bulk of Simscape Electrical): every acausal device model — semiconductors (diode/MOSFET/IGBT/BJT/JFET/thyristor/GTO/SPICE\*), op-amps/comparators/gate-drivers/optocouplers, CMOS logic-gate *device* models, passives (R/L/C/transformers/cables/transmission lines/magnetic cores), sources (V/I/programmable), switches/relays/breakers/fuses, machines-as-acausal-plants, energy storage (battery/fuel-cell/solar-cell/supercap/electrolyzer), power-system gear (SM AC/DC/ST excitation systems, governors, PSS, busbars, load-flow, harmonic filters), and sensor/MEMS/piezo/solenoid *device* models. (The machine models overlap **#31**'s Tier-2 motor plant builders where a simulation plant is actually wanted.)
- Acceptance (per planned piece): Smith predictor restores the delay-free loop's margin on a dead-time plant; MRAC converges the tracking error to zero on a first-order reference; RST reproduces a `place`-equivalent closed loop; RMS/moving-average match closed-form values on test signals; thermal RC ladder matches its step response; all embeddable pieces keep `make embedded-check` green.

### ☐ Fuzzy logic control (#33)

A compact, embeddable fuzzy inference engine — fuzzy controllers are deployed on MCUs across
appliance, HVAC, and automotive control, and the toolbox is self-contained (no new
dependency, all `constexpr`-friendly fixed-size tables). Audited against the Fuzzy Logic
Toolbox blocks.

- **Membership functions** (`fuzzy/membership.hpp`, pure `constexpr`): triangular, trapezoidal, Gaussian, gaussian2, generalized-bell, sigmoidal (+ diff/prod), S/Z/Pi-shaped, linear S/Z. Each a small functor `T → [0,1]`.
- **Inference engine** (`fuzzy/fis.hpp`, fixed-size, allocation-free): Mamdani and Sugeno FIS over `NIn` inputs / `NOut` outputs / `NRules` rules; min/prod implication, max/probabilistic-OR aggregation, centroid/weighted-average defuzzification. `FIS::evaluate(inputs) → outputs` runs on-target. A `FISTree` composes small FISes (the toolbox "FIS Tree").
- **Fuzzy PID** (Tier-2): the common 2-input (error, Δerror) → control map preset, the "Fuzzy PID Controller" block.
- References: Passino & Yurkovich, *Fuzzy Control*, Addison-Wesley, 1998; Mamdani & Assilian, IJMMS 7(1), 1975; Takagi & Sugeno, IEEE T-SMC 15(1), 1985.
- Acceptance: membership functions match their closed forms; a known Mamdani rule base reproduces reference surface values; fuzzy-PID reduces to a tuned PID on a linear plant; `make embedded-check` stays green.

### ☐ Constraint-enforcement safety filters (#34)

A minimally-invasive safety layer that sits between any controller and the actuator and
modifies the command as little as possible to keep the system inside a safe set — the
Simulink Control Design "Constraint Enforcement", "Control Barrier Function" (+ high-order),
and "Passivity Enforcement" blocks. Increasingly standard for deploying learned/aggressive
controllers safely; embeddable since each tick is a tiny fixed-size QP.

- **Action governor / constraint enforcement** `design::safety_filter(f, g, constraints)`: solve `min ‖u − u_des‖²` s.t. `A(x)·u ≤ b(x)` each tick — reuses the small dense QP from MPC (#14); for box/affine constraints the one-step projection is closed-form (no solver).
- **Control Barrier Function** (+ high-order CBF for relative-degree ≥ 2): enforce `ḣ(x,u) ≥ −α·h(x)` as the QP constraint so a barrier `h ≥ 0` (safe set) is forward-invariant.
- **Passivity enforcement**: constrain the command to keep the loop passive (input/output passivity index ≥ 0).
- References: Ames et al., "Control Barrier Functions: Theory and Applications," ECC 2019, https://doi.org/10.23919/ECC.2019.8796030; Gurriet et al., realizable safety filters.
- Acceptance: the filter is the identity when the desired command is already safe, and holds `h ≥ 0` / the constraint set invariant under a command that would otherwise violate it; per-tick cost is bounded (closed-form for box constraints).

### ☐ Rigid-body robot dynamics & model-based control (#35)

The kinematics module (`kinematics/serial_arm.hpp` — DH chains, geometric Jacobian, DLS-IK;
plus `scara.hpp`/`stewart.hpp`) has the geometry but **no dynamics**. Add the mass/Coriolis/
gravity terms and the model-based controllers they enable — the headline being computed-torque
and gravity-compensation control for arms. On-target embeddable (fixed N, allocation-free),
extends #22/#23. Audited against the Robotics System Toolbox dynamics blocks.

| Piece | Plan | Reuse |
|---|---|---|
| Inverse dynamics `τ = ID(q, q̇, q̈)` | Recursive Newton–Euler (RNEA) over the DH chain + per-link inertial params (mass, COM, inertia tensor) | `serial_arm.hpp` chain, `geometry.hpp` |
| Joint-space mass matrix `M(q)` | Composite-rigid-body algorithm (CRBA) | RNEA columns |
| Gravity torque `G(q)`, velocity-product `C(q,q̇)q̇` | RNEA with selected terms zeroed | RNEA |
| Forward dynamics `q̈ = FD(q,q̇,τ)` | Articulated-body algorithm, or `M⁻¹(τ − C − G)` via `solve` | `matrix/solve.hpp` |
| Computed-torque / inverse-dynamics control, gravity comp | feedback-linearizing law `τ = M(q)(q̈_d + Kₚe + K_d ė) + C + G` | the above + `PIDController` gains |
| Joint-space / task-space motion model | the FD as a simulation plant (joint or operational-space inputs) | `simulation/` |
| Generalized / constrained IK | Multi-constraint IK (pose + aiming + Cartesian/joint bounds) solved as a weighted nonlinear least-squares over the Jacobian — beyond the shipped single-target DLS IK; for redundant arms and task constraints (the Robotics System Toolbox `generalizedInverseKinematics` + `constraint*` family) | NLS core (#3) + `serial_arm` Jacobian |

- References: Featherstone, *Rigid Body Dynamics Algorithms*, Springer, 2008 (RNEA/CRBA/ABA); Craig, *Introduction to Robotics*, 3rd ed., 2005 (computed torque); Siciliano et al., *Robotics: Modelling, Planning and Control*, 2009.
- Acceptance: RNEA inverse dynamics round-trips against forward dynamics; `M(q)` is SPD; computed-torque linearizes a 2-link arm to the commanded error dynamics; gravity-comp holds a static pose with zero steady-state droop. Collision geometry/checking (the toolbox `Collision *` blocks) stays out of scope — that's the planning layer, not control.

### ☐ Mobile-robot kinematics & path following (#36)

Ground-vehicle kinematic models plus the classic geometric path-following controllers — small,
embeddable, and squarely control (the planning layer above them is not). Audited against the
Robotics System Toolbox / Navigation / Automated Driving steering blocks.

- **Kinematic models** (`kinematics/mobile.hpp`): unicycle, differential-drive, bicycle (front-steer), Ackermann, articulated-steer — each `state ← step(state, cmd, dt)` plus the inverse command map. Differential-drive shares the mixer already planned in the Tier-2 mixers item.
- **Pure Pursuit** — look-ahead geometric tracker → linear + angular (or curvature) command; the workhorse for diff-drive/unicycle.
- **Stanley** — cross-track + heading-error steering law (front-axle reference) for bicycle/Ackermann, the car-like path follower.
- **MPPI** (stretch) — Model-Predictive Path-Integral sampling control over the kinematic model; modern sampling-based alternative to the geometric trackers, heavier (Monte-Carlo rollouts), lower priority. Relates to the MPC work in #14 but is sampling- not QP-based.
- Out of scope: path *planning* / obstacle avoidance (VFH, Timed Elastic Band, A*/RRT, occupancy grids) — that's the navigation layer, a different concern from the tracking controller.
- References: Coulter, "Implementation of the Pure Pursuit Path Tracking Algorithm," CMU-RI-TR-92-01, 1992; Thrun et al., "Stanley: The robot that won the DARPA Grand Challenge," 2006; Williams et al., "Model Predictive Path Integral Control," IEEE T-RO, 2017.
- Acceptance: each kinematic model integrates a known command to the analytic pose; Pure Pursuit and Stanley track a reference path within a cross-track tolerance and converge from an offset start; commands respect curvature/steering limits.

### ☐ Hydraulic & electrohydraulic control (#37)

Valve, actuator, pump, and system-level control for hydraulic drives — a domain with its own
nonlinearities (square-root orifice flow, spool deadband/overlap, fluid compressibility,
asymmetric cylinders, load sensing) that generic loops handle badly. On-target embeddable; builds
on the shipped PID/SMC/ADRC, `Lut` (flow/efficiency maps), `deadband`/`expo`/`SlewLimiter`
(`utility/io.hpp`), and sensor scaling. The hydrostatic dual-path drive ties the Tier-2 mixers item.

| Layer | Plan | Reuse |
|---|---|---|
| Valve | Spool-position inner loop (proportional/servo valve); **deadband/overlap compensation**; **flow linearization** — invert `Q ∝ sgn(ΔP)·A(x)·√│ΔP│` so the outer loop commands ~linear flow across varying load pressure; dither for stiction | `Lut` (valve area/flow map), `io.hpp` deadband |
| Actuator (cylinder) | Position / velocity / force / chamber-pressure control; **cylinder-asymmetry (area-ratio) compensation**; load-pressure feedback | PID/SMC/ADRC, pressure-sensor scaling |
| Pump | Variable-displacement swashplate control; **load-sensing** (LS) margin control; pressure-compensated flow; torque/power limiting (engine anti-stall) | PID + `Lut` (displacement/efficiency), bounds |
| System | Pressure-compensated flow sharing; hydrostatic-transmission / dual-path drive control; anti-cavitation; mode switching | the Tier-2 drive mixers below |

- Plant models (Tier-2 sim, `models::`): orifice-flow + bulk-modulus cylinder/load, hydrostatic transmission — for testing the loops.
- References: Merritt, *Hydraulic Control Systems*, 1967; Jelali & Kroll, *Hydraulic Servo-systems*, 2003; Manring, *Hydraulic Control Systems*, 2005.
- Acceptance: flow linearization makes cylinder velocity ~linear in command across a ΔP sweep; deadband comp removes the dead zone; LS pump holds its pressure margin under load steps; dual-path drive preserves the straight/turn split without clip distortion.

### ☐ Grid-tied inverter & energy-storage control (#38)

Grid-connected conversion and hybrid energy-storage control — the application layer on top of the
converter/modulation in **#24**, reusing the shipped grid primitives: `SinglePhasePLL`/`ThreePhasePLL`/
DSOGI `Resonator`/`SogiFll` (sync), `pr.hpp` (resonant current), `transforms.hpp` (Clarke/Park +
symmetrical components), the FOC current-loop structure, and EKF/UKF (state estimation).

| Area | Plan | Reuse |
|---|---|---|
| Synchronization | Grid sync via PLL/DSOGI; positive/negative-sequence extraction for unbalance; soft-start / pre-synchronization | `pll.hpp`/`sogi.hpp`, sequence components |
| Grid-following | dq current control (P/Q → id/iq refs) + DC-bus outer loop; αβ PR/resonant current control; cross-decoupling + grid-voltage feedforward; current limiting | `pr.hpp`, FOC current loop, `transforms.hpp` |
| Grid-forming | **Droop** (P-f, Q-V); **virtual synchronous machine** (inertia/damping emulation); voltage/frequency control; seamless island ↔ grid transition | new on PR/PI + transforms |
| Protection / grid-code | Islanding detection (passive ROCOF/V/f; active freq/impedance); LVRT/FRT ride-through with reactive injection | analysis + logic |
| Energy storage / hybrid | **Battery SOC** (OCV–R–RC equivalent-circuit + EKF/UKF); **SOH** (capacity/resistance-fade tracking); DC-bus regulation; **power-sharing / energy management** across battery + PV + grid; charge/discharge limits + droop | EKF/UKF, `Lut` (OCV–SOC curve), #24 DC-DC |

- Scope note: this is battery **estimation + power-management control** (in scope), distinct from the battery *plant models* declined in the Powertrain pass.
- References: Teodorescu, Liserre & Rodríguez, *Grid Converters for Photovoltaic and Wind Power Systems*, 2011; Yazdani & Iravani, *Voltage-Sourced Converters in Power Systems*, 2010; Rocabert et al., "Control of Power Converters in AC Microgrids," IEEE T-PEL 27(11), 2012; Plett, *Battery Management Systems, Vol. 2: Equivalent-Circuit Methods*, 2015 (SOC/EKF).
- Acceptance: PLL locks under distortion/unbalance; grid-following tracks P/Q at commanded power factor and the DC-bus holds under load steps; droop shares load between parallel sources; SOC-EKF tracks coulomb-count within tolerance and corrects an initial-SOC error; islanding is flagged within the grid-code window.

### ☐ Tier-2 drive/control mixers (#TBD)

Stateless mixing functions that combine conditioned operator axes into actuator
commands, living alongside the tier-2 I/O appliances in `utility/io.hpp`
(`wet::io`). The primitives are already in place — `AxisInput` (cal → scaled
dead zone → `expo` → scale), `deadband`/`scaled_deadband`/`expo`, and
`SlewLimiter` for output rate-limiting — so this is the combiner layer on top.

**Remaining:**

- **Differential / tank drive:** `differential_drive(throttle, turn) → {left, right}`, with proper normalization so a combined command never clips past the actuator range (scale both outputs down by the overshoot rather than hard-clamping, which distorts the turn ratio). Dual-path hydrostatic drive is a target use case for the anti-clip normalization.
- **Arcade (single-stick):** `arcade_drive(x, y) → {left, right}` — same kinematics, one-stick ergonomics.
- **Mecanum / holonomic:** `mecanum_drive(x, y, rotation) → {fl, fr, rl, rr}` with joint normalization across all four wheels.
- **Elevon / aileron (flight, low priority):** combine pitch+roll into two control surfaces.
- Acceptance (per piece): straight-line command yields equal wheel outputs; a saturating combined command preserves the turn/throttle *ratio* after normalization (no clip distortion); zero input → zero output; mecanum recovers pure translation/rotation on the cardinal inputs.

### ☐ Model-order reduction (#28)

Reduce a high-order `StateSpace` to a low-order approximation that preserves the
input-output behavior — for controller/observer order reduction, fitting an
identified high-order model to a deployable size, and discarding weakly
controllable/observable modes. Host-side design (`toolbox.hpp` — allocates the
balancing transform; the reduced model is a plain `StateSpace`).

The Gramian foundation is **done**: `wet::lyap`/`dlyap` (Kronecker solve,
`design/lyapunov.hpp`) and `stability::controllability_gramian`/
`observability_gramian`, plus `analysis::norm_h2`/`norm_hinf` for measuring the
reduction error.

**Remaining:**

- **Balanced realization** `design::balreal(sys)` → balancing transform `T` such that
  `Wc = Wo = diag(σ)` (the Hankel singular values), via the Gramian Cholesky factors
  + an SVD of their cross product. Returns the balanced `StateSpace` and the σ vector.
- **Balanced truncation** `design::balred(sys, order)` → drop the states with the
  smallest Hankel singular values; the a-priori error bound is `‖G−Gᵣ‖∞ ≤ 2·Σ σ_k`
  over the discarded modes (computable from the σ vector). MATLAB `balred`.
- **Singular-perturbation (residualization)** `modred(sys, keep, 'matchdc')` as the
  DC-accurate alternative to plain truncation — exact at steady state, MATLAB `modred`.
- `hankelsv(sys)` to expose the Hankel singular values on their own (MATLAB `hsvd`),
  reusing the balreal machinery.
- Square-root / Cholesky-factor Gramian path (`lyapchol`/`dlyapchol` equivalent) only
  if plain-Gramian balancing proves numerically marginal — the standard reason to add
  the square-root solvers, otherwise skip them (they were categorized "not necessary"
  precisely because nothing but balancing wants them).
- References: Moore, "Principal Component Analysis in Linear Systems: Controllability,
  Observability, and Model Reduction," IEEE TAC 26(1), 1981,
  https://doi.org/10.1109/TAC.1981.1102568; Glover, "All optimal Hankel-norm
  approximations…," Int. J. Control 39(6), 1984; Antoulas, *Approximation of
  Large-Scale Dynamical Systems*, SIAM, 2005.
- Acceptance: `balred` of a known high-order plant recovers the dominant modes and the
  realized `‖G−Gᵣ‖∞` (via `norm_hinf`) stays within the `2·Σσ_k` bound; `modred`
  matches DC gain exactly; round-trip `balreal` preserves the transfer function.

### ☐ Lookup-table enhancements — cubic interpolation + 2-D extrapolation (#30)

The breakpoint tables in `toolbox/lookup.hpp` (`Lut1D` linear/nearest, `Lut2D` bilinear,
`Extrapolation` clamp/linear) cover gain scheduling, sensor linearization, and motor maps.
Two gaps worth closing — both thin reuse, not new math:

- **Cubic / spline `Lut1D`.** A smooth (C²) interpolant for tables where linear segments
  kink — `Lut1DCubic` (or a `Lut1D` interpolation-mode flag) that precomputes per-segment
  coefficients from `design::cubic_spline(xs, ys)` (already in `trajectory/spline.hpp`) and
  evaluates them in `operator()`. Adapter over the existing spline solver; no new solve.
- **`Lut2D` extrapolation policy.** `Lut2D` currently hard-clamps out-of-grid queries; give it
  the same `Extrapolation` (clamp / linear) knob `Lut1D` already exposes, applied per axis.
  ~10 lines mirroring the `Lut1D` out-of-bounds branch.
- **Adaptive lookup table** (the Simulink Design Optimization "Adaptive Lookup Table" blocks): a `Lut1D`/`Lut2D` whose cell values update online from `(x, measured)` samples (running-mean / stair-fit per cell) — for self-calibrating maps (sensor drift, motor LUT refinement). Mutable cells on top of the existing table; the breakpoints stay fixed.
- Acceptance: cubic LUT matches `design::cubic_spline` at and between breakpoints and stays C²;
  `Lut2D` with linear extrapolation reproduces the edge gradient outside the grid and clamp mode
  is unchanged; adaptive cells converge to the sampled surface; all stay `constexpr`-constructible / allocation-free. (Bicubic `Lut2D` and n-D (≥3) tables
  stay out of scope until a real case asks — YAGNI.)

### ☐ Worked examples for every library feature (#27)

Every shipped feature needs one clear, runnable usage example — the "how do I actually
call this" that turns the inventory into something a new user can adopt. Examples double
as living documentation and as exercise of the public API surface.

- **Two acceptable forms.** (1) A standalone program in `examples/` (`example_*.cpp`) for
  end-to-end / multi-block workflows. (2) An *example-grade* unit test in `tests/test_*.cpp`
  — named and commented so a reader learns the API from it. The bar: a commented
  build→design→run sequence counts; a bare `CHECK(gain == Approx(...))` assertion dump does not.
- **Coverage target:** every public `design::` factory and every runtime
  controller/filter/estimator/kinematics type has at least one worked example (standalone or
  example-grade test). Track the gap as a feature→example checklist against the README inventory.
- **Header `@code` blocks must not rot:** wire each Doxygen `@code` snippet into the
  build/test (compiled-snippet harness or extraction into example-grade tests) so a
  signature change that breaks an example fails the build.
- Relates to the documentation standard (Doxygen + math + MATLAB equivalent + reference +
  worked example) already applied per-header; this item makes that example coverage
  *complete and enforced* rather than per-review.
- Acceptance: a feature→example inventory with no gaps; every header `@code` block compiles
  in CI; `make` builds all of `examples/`.

### ☐ MATLAB Control System Toolbox API parity (#29)

The library's purpose is **quickly designing and implementing embedded control systems**.
This ledger therefore plans only the Control System Toolbox functions that serve that goal —
synthesis, model assembly for deployment, sampled-loop fidelity (delay/discretization), and
the analysis that gates a deployable design. Everything else (host-only analysis, GUIs,
plotting, sparse/array model machinery, tuning frameworks) is recorded as **not planned** so
the audit stays complete. Implemented functions already live in
[../matlab.hpp](inc/wet/matlab.hpp); only the gaps appear here.

Like the rest of `matlab::`, every planned entry is a **thin MATLAB-spelled wrapper over a
canonical `wet::` function or object** — the real logic lives in `design::`/`analysis::`/the
model types, and the wrapper only adapts names/arguments. If a planned entry has no canonical
`wet::` home yet, that underlying capability is the actual roadmap work (build it in the
proper module first, then alias it here).

#### Planned — design & synthesis

| Function | Purpose | Why it serves embedded control |
|---|---|---|
| `pidstd` / `pidstd2` | Standard-form (Kp·(1+1/Ti·s+Td·s)) PID constructor → shipped parallel runtime | Industrial PID is tuned in Ti/Td; let designers enter gains the way they think, deploy the existing runtime |
| `kalmd` | Discrete Kalman estimator from a continuous plant + noise | On-target estimators run discrete but are designed from a continuous physical model (estimator analog of the shipped `lqrd`) |
| `lqry` | LQR weighting outputs instead of states (`Q = CᵀQ_yC`) | Output weights are physical and easy to choose for a deployable regulator |
| `make1DOF` / `make2DOF` | Convert between 1-DOF and 2-DOF PID | Pick the form the target implements (the runtime already supports 2-DOF via `b`/`c`) |

#### Planned — model assembly & reduction for deployment

| Function | Purpose | Why it serves embedded control |
|---|---|---|
| `connect` | Assemble a control loop from named blocks | Build the exact closed loop you then discretize and deploy |
| `minreal` | Cancel uncontrollable/unobservable modes | Fewer states ⇒ less compute and RAM on-target |
| `canon` | Controllable/observable/modal canonical realization | Canonical forms map directly to efficient fixed-structure implementations (direct form, decoupled modal) |
| `prescale` | State-space diagonal scaling | Conditions the realization for fixed-point / limited-precision math on-target |

#### Planned — sampled-loop fidelity (delay & discretization)

| Function | Purpose | Why it serves embedded control |
|---|---|---|
| `pade` | Padé rational approximation of dead time | Fold sensor/compute/transport delay into a rational plant so the discrete design accounts for it (`fopdt` in #3 already uses Padé internally) |
| `thiran` | Fractional-delay (Thiran) filter | Implementable sub-sample delay element on-target |
| `d2d` / `upsample` | Resample / upsample a discrete model | Retime a discrete controller to the actual ISR rate; pairs with multi-rate sim (#18) |

#### Planned — design verification (gates a deployable controller)

| Function | Purpose | Why it serves embedded control |
|---|---|---|
| `allmargin` | All gain/phase/**delay** margins + crossovers (superset of shipped `margin`) | Delay margin directly bounds tolerable sampling + compute latency — the classic embedded failure mode |
| `isstable` | Stability predicate for a model | Gate a synthesized controller/observer before deploy (`is_stable_continuous` exists; generalize to discrete + `StateSpace`/`TF`) |
| `rlocus` | Root-locus loci vs. gain | Classical SISO pole-placement design workflow |
| `diskmargin` | Disk-based stability margin — simultaneous gain+phase margin from a disk inscribed in the Nyquist exclusion region; computed from the loop FRF (no uncertain model) | the modern robustness complement to `margin`/`allmargin`; reuses `analysis::bode`/`nyquist`. (The full uncertain-model `wcdiskmargin` stays out of scope per #13.) |
| `nichols` *(low)* | Nichols (gain-dB vs phase-deg) response; data is a thin reorder of `analysis::bode`, with a `plot::nicholsplot` chart | Classical loop-shaping; the chart slots beside the existing `bodeplot`/`nyquistplot`/`pzplot` in `plot_plotly.hpp` (add the M/N-circle grid there) |
| `sigma` *(low)* | Singular-value (MIMO gain) frequency response | Robustness check for multivariable embedded loops |

#### Planned — block-representation type

| Type | Covers blocks | Why it serves embedded control |
|---|---|---|
| **LTV / "Varying" wrapper** — state-space (and TF / observer-form) container holding swappable matrices via a schedule or per-tick update callback | *Varying State Space / Transfer Function / Observer Form*, *LTV System*, *Varying Delay* | Gain scheduling and time-varying loops run on-target; an explicit type beats ad-hoc mutation of a `StateSpace`. Pairs with the LPV model + gain-scheduling workflow in **#10** |

Already-covered block types (no work): *LTI System* → `StateSpace`/`TransferFunction`/`ZPK`;
*PID Controller* & *(2DOF)* → `PIDController` (`b`/`c` give 2-DOF, discrete runtime);
*Varying Lowpass/Notch* → `filters.hpp` biquads with per-tick coefficient updates;
*Kalman* / *EKF* / *UKF* → `kalman` / EKF / `UnscentedKalmanFilter` / ESKF.

#### Not planned — outside the embedded-design goal

| Group | Functions | Reason |
|---|---|---|
| Niche model types | `dss`/`dss2ss`/`gcare`/`gdare`, `drss`/`rss`, `filt`, `frd`, *Sparse Second Order* (`mechss`) | Descriptor/random/FRD/sparse models aren't part of an embedded design flow |
| Host analysis | `tzero`/`zero`, `ctrbf`/`obsvf`, `stabsep`/`freqsep`/`modalsep`/`modalsum`/`modsep`, `covar`, `getGainCrossover`, `abs`, passivity (`isPassive`/`getPassiveIndex`/`getSectorIndex`) | Offline analysis that doesn't change what gets deployed; do `minreal`/`allmargin` instead |
| Model bookkeeping | `append`/`lft`/`sminreal`/`ss2ss`/`ssequiv`, `augstate`/`augoffset`/`augdelay`, `xperm`/`xsort`/`xelim`, `compreal`/`modalreal` (keep `canon`), `delayss`/`exp`/`absorbDelay`/`delay2z`/`totaldelay`/`hasdelay`/`get`/`setDelayModel`, `inv`/`conj`/`ctranspose`/`repsys`, `order`/`dsort`/`esort`, `piddata*`/`getComponents`, `particleFilter`, `d2c` | Structural/array/accessor utilities or heavy estimators with no embedded payoff |
| Tooling layer | `slTuner`/`slLinearizer`, `systune`/`looptune`/`TuningGoal.*`/`tunable*`/`genss`/`realp`, GUIs (`controlSystemDesigner`/`Tuner`, `pidTuner`, `linearSystemAnalyzer`, `openloopeditor`), `*options`/grid/`view*`, sparse `sparss`/`spy`/`full`/`sminDAE`/`Sparse*`, model arrays (`stack`/`reshape`/`permute`/`size`/`ndims`/`nmodels`), `frd` accessors, property `get`/`set`, units, `codegen` | No Simulink/GUI/sparse/array/tunable layer here. (The host-side `plot_plotly.hpp` *does* render charts — `bodeplot`/`nyquistplot`/`pzplot` exist and `nicholsplot` is planned above — so individual `*plot` aliases land there as thin wrappers, not as a missing layer.) |

Model-order reduction (`balreal`/`balred`/`modred`/`hsvd`/`reducespec`/`getrom`/`BalancedTruncation`/`ModalTruncation`/POD/NCF) is tracked under **#28**, not here.

### ☐ General thermal-network modeling + Kalman observer (#33)

`power/thermal.hpp` already realizes RC thermal networks as `StateSpace`
(`design::foster_thermal_ss`, `cauer_thermal_ss`) and estimates a node temperature open-loop
(`JunctionEstimator`). The FET pieces (`FetLossModel`) are the right specialization; the rest is
general thermal modeling that should be promoted to a domain helper (`toolbox/thermal.hpp`),
leaving `power/` with the FET loss model and a thin junction alias.

- **Absolute temperatures with ambient as an input.** Model absolute node temps with ambient/
  coolant as a real input column (`ẋ = Ax + B_P·P + B_amb·T_amb`, per-node leak to ambient),
  instead of estimating *rise* and adding a boundary offset. Required for multi-boundary networks
  and a proper absolute-temperature estimator.
- **MIMO networks.** `NU>1` heat sources at arbitrary nodes (winding copper + stator iron loss;
  multiple FETs) and `NY>1` measured nodes (NTCs at several locations).
- **General netlist assembler.** Per-node capacitance + inter-node resistances + node-to-ambient
  resistances → `StateSpace`; Foster/Cauer become the 1-D special cases, arbitrary 3-D thermal
  graphs fall out.
- **Kalman thermal observer (highest value, nearly free).** Drop the discretized thermal SS into
  the existing `design::kalman(sys, Q, R)` + `KalmanFilter`: predict with known `u = [P_loss;
  T_ambient]`, update from the measured node(s). Estimates the unmeasured junction from a lagging
  case NTC + power input and corrects the open-loop model bias `JunctionEstimator` can't. The
  open-loop estimator becomes the no-measurement degenerate case. Needs the thermal SS's noise
  structure (`G`, process `Q` from R/C/loss uncertainty, sensor `R`).
- **`HeatSource` concept** generalizing `ThermalLossModel`; `FetLossModel`, copper-loss `I²R`,
  iron-loss, friction are instances.
- Model-order reduction for high-order thermal models is covered by the general balanced-
  truncation (`balred`) tool (operates on any `StateSpace`) — track there, not here.
- Acceptance: `toolbox/thermal.hpp` general core + `power/` FET specialization; a worked KF
  thermal-observer example showing junction estimation that survives a wrong nominal R/C.

### ☐ Two-mass drivetrain estimator (dual-encoder) (#34)

`power/mechanical_estimator.hpp` ships the rigid 1-DOF model `[θ_m, ω_m, τ_load]` — the right
default for a motor-only encoder, with optional sensorless-angle and load-accelerometer
measurement channels. The richer two-mass model (motor inertia + spring/damper coupling + load
inertia) captures shaft compliance and the torsional resonance, but its extra states are only
**observable with a load-side measurement** — so it's gated on that, not the default.

- **Model.** `design::two_mass_ss(J_m, J_l, k_shaft, d_shaft, b_m, b_l)` → continuous `StateSpace`
  over `[θ_m, ω_m, θ_l, ω_l]` (input `i_q`); the torsional mode `(θ_m − θ_l)` is the resonance.
- **Observability gate.** Needs a load encoder (`NY=2`: motor + load angle) **or** a load
  accelerometer (the existing accel channel can substitute — it senses load-side motion). From a
  motor encoder alone the load states are weakly observable; keep the rigid model there.
- **Load friction + gravity (only here).** Once the load side is observed, add load Coulomb/viscous
  friction and an optional gravity term `m·g·l·sin(θ_l)` (mildly nonlinear → EKF measurement or a
  scheduled term). These are identifiable only with the load-side observability, which is why they
  belong on this model, not the rigid one.
- Reuses the multi-measurement `KalmanFilter::update(y, C, D, R, u)` overload already added for the
  heterogeneous-sensor case.
- Acceptance: dual-encoder two-mass estimator with a worked example showing torsional-mode and
  load-friction estimation; rigid model stays the single-encoder default.

Tracked in [decisions.md](decisions.md): O1 (ESC perturbation policy for MPPT, #8),
O2 (anti-chatter update strategy, #9), O3 (error-reporting under the ETL backend, #21),
O4 (promote `geometry.hpp` out of `utility/`, #22).
