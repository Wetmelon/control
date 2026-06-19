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
- **Tier 2 model builders** (`models.hpp`, embeddable — construct universal types only): `models::single_mass`, `second_order`, `two_mass` → `StateSpace`, `fopdt` → `TransferFunction` (Padé dead-time). Downstream design never knows it came from a builder.
- **Host-side identification** (`analysis/identification.hpp`, `toolbox.hpp`): `tfest<Nn,Nd>` (output-error/PEM), `ssest<NX>` (N4SID subspace), `validate` (cross-validation), plus Tier-2 wrappers `identify_fopdt_from_step`, `identify_two_mass`, etc. — each returns both a named-parameter struct and the universal type. Reuse the model structs already in `estimation/identification.hpp`; online RLS (`estimation/recursive_least_squares.hpp`) stays embedded.
- **FRF estimation** (`analysis/frf.hpp`, `toolbox.hpp`): `frfest(time, u, y, freqs)` → magnitude/phase/coherence, Welch-style averaging; pairs with Chirp/MultiSine.
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
- References: Rugh & Shamma, "Research on Gain Scheduling," Automatica, 2000, https://doi.org/10.1016/S0005-1098(00)00058-3; Apkarian & Gahinet, IEEE TAC, 1995, https://doi.org/10.1109/9.384219.
- Acceptance: continuity across transitions; stability over the certified envelope.

### ☐ H-infinity output feedback (#13)

Output-feedback synthesis for weighted generalized plants. Targets: flexible structures
with modal uncertainty, flight control with structured uncertainty, active
suspension/vibration isolation. Host-only (`toolbox.hpp` — gamma search allocates/iterates,
[decisions.md](decisions.md) D16).

- Interface: `design::synthesize_hinf(augmented_plant, weighting_filters, gamma_search_bounds)` → `HInfResult` / `HInfArtifacts` + S/T analysis models. Workflow: coupled Riccati solves over gamma; select feasible controller.
- References: Doyle et al., "State-Space Solutions to Standard H2 and H-infinity Control Problems," IEEE TAC, 1989, https://doi.org/10.1109/9.29425; Glover & Doyle, Systems & Control Letters, 1988, https://doi.org/10.1016/0167-6911(88)90055-2.
- Acceptance: weighted robust-stability/performance; regression vs known references.

### ☐ Constrained MPC (#14)

Finite-horizon constrained control with a deterministic runtime iteration budget. Targets:
multivariable process control with limits, AV trajectory tracking with actuator limits,
power electronics with current/voltage bounds. Host-only (`toolbox.hpp` — QP solver
allocates/iterates, [decisions.md](decisions.md) D16).

- Interface: `design::synthesize_mpc(plant, horizon, weights, constraints, solver_budget)` → `MPCResult` / `MPCArtifacts` + fixed-iteration warm-start runtime.
- References: García, Prett & Morari, "Model Predictive Control: Theory and Practice — A Survey," Automatica, 1989, https://doi.org/10.1016/0005-1098(89)90002-2; Qin & Badgwell, Control Engineering Practice, 2003, https://doi.org/10.1016/S0967-0661(02)00186-7.
- Acceptance: deterministic per-tick runtime bound; constraint-satisfaction regression; tracking/regulation tests.

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
- **DC-DC:** duty feedforward for buck (`D = Vo/Vi`), boost, buck-boost; peak/average current-mode with slope compensation; MPPT (Perturb-&-Observe, Incremental-Conductance — note overlap with ESC #8); interleaved boost.
- **PFC / AC-DC:** boost PFC with average-current-mode (inner current shapes inductor current to a rectified-sine reference, outer DC-bus loop); totem-pole bridgeless PFC (HF GaN/SiC leg + line-frequency unfolding leg, zero-cross commutation sequencer).
- **Multilevel:** 3-level NPC (phase-disposition carriers or 3-level SVM + neutral-point balancing); ANPC as the loss-balancing extension (do NPC first).
- References: Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd ed., 2020 (DC-DC, PFC, current-mode); Holmes & Lipo, *Pulse Width Modulation for Power Converters*, IEEE Press, 2003 (inverter PWM, multilevel); Hava et al., IEEE T-PEL 14(1), 1999 (CPWM/DPWM/GDPWM).
- Acceptance (per piece): duty feedforward matches the CCM conversion ratio; center-aligned/interleaved carriers give the expected phase/ripple relationships; MPPT converges to the true MPP under irradiance steps (matches ESC on a smooth P-V curve); boost-PFC current tracks the rectified-sine template at unity displacement factor; NPC neutral-point stays balanced; ANPC switch-state selection equalizes per-device loss.

### ☐ Tier-2 drive/control mixers (#TBD)

Stateless mixing functions that combine conditioned operator axes into actuator
commands, living alongside the tier-2 I/O appliances in `utility/io.hpp`
(`wet::io`). The primitives are already in place — `AxisInput` (cal → scaled
dead zone → `expo` → scale), `deadband`/`scaled_deadband`/`expo`, and
`SlewLimiter` for output rate-limiting — so this is the combiner layer on top.

**Remaining:**

- **Differential / tank drive:** `differential_drive(throttle, turn) → {left, right}`, with proper normalization so a combined command never clips past the actuator range (scale both outputs down by the overshoot rather than hard-clamping, which distorts the turn ratio). *Owner note (Paul): has prior art from Rexroth dual-path hydrostatic drive control — wants to drive the normalization/anti-clip design here.*
- **Arcade (single-stick):** `arcade_drive(x, y) → {left, right}` — same kinematics, one-stick ergonomics.
- **Mecanum / holonomic:** `mecanum_drive(x, y, rotation) → {fl, fr, rl, rr}` with joint normalization across all four wheels.
- **Elevon / aileron (flight, low priority):** combine pitch+roll into two control surfaces.
- Acceptance (per piece): straight-line command yields equal wheel outputs; a saturating combined command preserves the turn/throttle *ratio* after normalization (no clip distortion); zero input → zero output; mecanum recovers pure translation/rotation on the cardinal inputs.

Tracked in [decisions.md](decisions.md): O1 (ESC perturbation policy for MPPT, #8),
O2 (anti-chatter update strategy, #9), O3 (error-reporting under the ETL backend, #21),
O4 (promote `geometry.hpp` out of `utility/`, #22).
