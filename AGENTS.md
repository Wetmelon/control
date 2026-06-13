# AGENTS.md

This file provides guidance to LLMs when working with code in this repository.

## Project

Header-only C++20 control systems library (`wet`) targeting embedded systems. Zero dynamic allocation, fully constexpr, fixed-size stack-allocated types throughout.

## Build and Test

Build system is [tup](https://gittup.org/tup/) with a Makefile convenience wrapper. Compiler path configured in `tup.config` (see `tup.config.default`).

```bash
make            # format + compiledb + build + run all tests
make tests      # build tests only, then run
tup --quiet tests   # compile tests without running
./tests/build/test_runner.exe   # run already-built tests
```

Regenerate `compile_commands.json` for IDE support (clangd, VS Code intellisense). This only refreshes the compilation database — it does **not** compile anything, so it is far faster than a full build. Run it after adding/moving files or includes when clangd's diagnostics go stale, instead of a full `make`:

```bash
tup --quiet compiledb
```

Run a single test suite or case with doctest filters:

```bash
./tests/build/test_runner.exe -ts="Matrix Functions"
./tests/build/test_runner.exe -tc="Matrix exponential - diagonal"
```

Compiler flags: `-O3 -std=c++20 -march=native -Wall -Wextra -Wdouble-promotion` (see `Tuprules.lua`). Minimum toolchains: GCC 10+, Clang 12+, MSVC 2022+ (C++20 with `std::numbers`, designated initializers, three-way comparison).

Format code: `clang-format -i $(find inc tests examples -name '*.hpp' -o -name '*.cpp')` (runs automatically on `make`).

Generate golden reference data for tests: `py -3` with `scipy` and `control` libraries.

**Tupfile output paths are relative to the Tupfile's directory**, not the repo root. E.g. `tests/Tupfile.lua` writes to `tests/build/`, so the test runner lives at `tests/build/test_runner.exe`. Same pattern for `examples/`. When a Tupfile rule names an output, that path is local — don't prepend the parent directory.

## Architecture

### Namespace layout

- `wet` — core types (`Matrix`, `StateSpace`, `ColVec`, controllers, rotations)
- `wet::design` — **constexpr** design functions and result structs (gains, coefficients, Riccati solutions)
- `wet::mat` — free functions for matrix operations (norms, determinant, expm, decompositions)
- `wet` — constexpr math primitives (`wet::sin`, `wet::sqrt`, `wet::conj`, `wet::abs`)
- `wet::stability` — stability analysis
- `wet::analysis` — controllability, observability, frequency response

Design functions are `constexpr` — they work at both compile time and runtime. Users enforce compile-time evaluation via `constexpr` variables or `constinit` on the controller. Typical workflow: `design::dlqr(...)` in `double`, then `.as<float>()` to convert for embedded deployment.

### Type system

- `Matrix<Rows, Cols, T=double>` — owning, stack-allocated via `std::array<T, Rows*Cols>`
- `ColVec<N, T>` / `RowVec<N, T>` — inherit from `Matrix<N,1,T>` / `Matrix<1,N,T>`. Vector-shaped operators are type-preserving: `ColVec ± ColVec`, `Matrix * ColVec`, and `scalar * ColVec` all return `ColVec` (see the explicit overloads in `colvec.hpp`). What does *not* preserve the type is anything that produces a plain `Matrix<N,1>` — chiefly `mat::solve` / `mat::lu_solve` (generic in the RHS) and products of `.block<>()` **views** (which miss the exact-`Matrix` LHS the ColVec `operator*` requires). To let those flow back into a vector, the **same-precision** `Matrix<N,1,T> → ColVec<N,T>` (and `Matrix<1,N,T> → RowVec`) constructor is **implicit** (a no-op — identical shape/storage); **cross-precision** (`float`↔`double`) stays **explicit** (use `.as<U>()`). Don't re-`explicit` the same-precision ctor — it adds no safety (shapes are identical) and turns `ColVec x = *mat::solve(...)` into a compile error that only surfaces on instantiation.
- `Block`, `Diagonal`, `RowView`, `ColView`, `TransposeView`, `UpperTriangle`, `LowerTriangle` — non-owning views
- `StateSpace<NX, NU, NY, NW=0, NV=0, T=double>` — LTI system with operator overloads (`*` series, `+` parallel, `-` differencing, `/` feedback). Block-matrix interconnection operators preserve the noise input matrices `G` (process, `NX × NW`) and `H` (measurement, `NY × NV`) through every composition, so `Q`/`R` covariance propagation stays consistent end-to-end.
- Result structs (`LQRResult`, `KalmanResult`, `LQGResult`, etc.) — all have `.as<U>()` and `bool success`
- Rotation types (`geometry.hpp`): `DCM<T>` (3×3), `Quaternion<T>` (with SLERP), `Euler<T, Order>`. Convention: `EulerZYX` = aerospace yaw-pitch-roll, `EulerXYZ` = robotics roll-pitch-yaw — don't conflate them.
- **Pose representation — translation-vector + quaternion, never 4×4.** Rigid-body pose is `Pose{ Translation3<T>, Quaternion<T> }`, and the kinematics/transform code composes poses as `(q, t)` pairs (`q = q₁⊗q₂`, `t = t₁ + q₁·rotate(t₂)`) — *not* by multiplying 4×4 `Transform4` matrices. Rationale: ~half the FLOPs per compose, 7 scalars vs 16, and a quaternion renormalizes cheaply where a 4×4 rotation block drifts from orthonormal. `Transform4<T>` is retained only as an interop/export convenience (textbook DH matrix, homogeneous point transforms), never the working representation. This is a cross-cutting commitment for the kinematics roadmap items (Stewart platform, serial arm, motion-system mappings) and any orientation-trajectory work — see `inc/wet/roadmap.md` item #22 for the full argument.

All types must support: `float`, `double`, `wet::complex<float>`, `wet::complex<double>`.

### Folder layout and umbrellas

All headers live under `inc/wet/` in domain folders, so consumers need a single `-I inc` and write namespaced includes:

```text
inc/wet/
  control.hpp   embeddable umbrella     toolbox.hpp   host superset
  math/  matrix/                        (linear algebra + constexpr math core)
  systems/  controllers/  estimation/  filters/  analysis/  simulation/  plotting/
  geometry.hpp · motor_control.hpp · iec61131.hpp · utility.hpp · matlab.hpp
```

Two entry points:

- **`wet/control.hpp`** — the embeddable subset. Nothing reachable from it allocates (`std::vector`) or pulls a third-party dependency. This is the contract; `make embedded-check` enforces it by failing if `<vector>` becomes reachable.
- **`wet/toolbox.hpp`** — `control.hpp` plus the host-only design/analysis tooling: `analysis/analysis.hpp` (Bode/Nyquist/margins/`linspace`/`logspace`), `simulation/{solver,simulate}.hpp`, `plotting/plot.hpp`, and `matlab.hpp`. These allocate, so they stay off the target.

**Host vs embedded.** A header is *host-only* if it (transitively) pulls `<vector>` or a third-party lib. Today that's `analysis.hpp`, `solver.hpp`, `simulate.hpp`, `plot.hpp`, `plot_plotly.hpp`, `matlab.hpp`. Everything else is embeddable and belongs in `control.hpp`. If you add a header that allocates, put it behind `toolbox.hpp`, not `control.hpp` — `make embedded-check` will catch a regression.

**Include style.** Cross-folder includes are `wet/`-qualified (`#include "wet/systems/state_space.hpp"`). Intra-folder includes stay bare and resolve relative to the including file — including `matrix.hpp`'s ordered bottom-include block.

`matrix.hpp` (`wet/matrix/matrix.hpp`) is the root of the linear algebra layer. It bottom-includes its dependencies in a specific order to satisfy template two-phase lookup:

```text
matrix.hpp  (defines Matrix<R,C,T>)
  └─ #include "block.hpp"            // intra-folder, bare
  └─ #include "cholesky.hpp"
  └─ #include "colvec.hpp"
  └─ #include "matrix_functions.hpp"
  └─ #include "rowvec.hpp"
  └─ #include "views.hpp"
```

These bottom-includes use `// IWYU pragma: keep`. Do not reorder them.

Higher-level headers (`wet/controllers/lqr.hpp`, `wet/estimation/kalman.hpp`, etc.) include `wet/matrix/matrix.hpp` and `wet/systems/state_space.hpp`.

### Error handling

Return `std::optional<T>` for operations that can fail (matrix inversion, Cholesky, DARE convergence). Result structs carry `bool success`. No exceptions.

`tl::expected` is available at `libs/expected.hpp` but not yet integrated.

### Key conventions

- **Constexpr-first**: all algorithms must support compile-time evaluation. Treat the heavy work — matrix decompositions / solves, Riccati and pole-placement solvers, eigen-analysis, full matrix–matrix products — as **compile-time design work**. The runtime path is limited to lightweight per-tick operations on already-designed gains: one state update, one matrix–vector multiply, scalar arithmetic. If a new algorithm needs an iterative solver at runtime, that's a smell — push it into a `design::` function.
- **Solve, don't invert**: whenever the goal is `A⁻¹·b` (or `A⁻¹·B`), call `mat::solve` / `mat::cholesky_solve` / `mat::lu_solve` instead of `.inverse()` then multiplying — it's more accurate and faster, and `solve` returning `nullopt` doubles as the singularity check. Only form an explicit `.inverse()` when the inverse matrix itself is the deliverable (e.g. a returned/reused inverse) or when it's a right-factor that a left-solve can't express cleanly (e.g. the similarity `XΛX⁻¹`). See *Numerical Implementation* below for the full rationale and the `Ax = b` rearrangement trick.
- **Failure handling**: `std::optional` for fallible operations, never exceptions
- **Tolerances**: use `default_tol<T>()` from `matrix_traits.hpp` (1e-6f for float, 1e-12 for double)
- **Complex support**: use `wet::conj()`, `wet::abs()`, `wet::sqrt()` — they are identity/passthrough for real types
- **Type literals**: use `T{1}` not `1.0` in templated code for float parity
- **Brackets always**: `if`/`for`/`while` always use braces, even single-line bodies
- **No backwards compatibility**: prefer clarity and correctness over API stability
- **Test organization**: one `test_*.cpp` per header, `TEST_SUITE` grouping, `static_assert` for constexpr checks + `CHECK(doctest::Approx(...).epsilon(...))` for runtime

## API Design Guide

This library serves two audiences simultaneously: PhD controls engineers who think in DARE and Bode plots, and students/hobbyists who are wiring up their first balance bot. Every API decision must satisfy both.

### The Three-Tier Pattern

Every control feature follows three tiers. This is the core architectural pattern of the library.

```text
Tier 1: Design functions          design::dlqr(A, B, Q, R)         →  constexpr, double precision
        ↓ returns
Tier 2: Result structs            design::LQRResult { K, S, e, success }  →  .as<float>() for deployment
        ↓ constructs
Tier 3: Runtime controllers       LQR controller(result);           →  controller.control(x) in ISR
```

**Tier 1 — Design functions** (`design::`) compute gains, solve Riccati equations, and run iterative algorithms. These are the heavy math. All are `constexpr`, supporting both compile-time and runtime evaluation.

**Tier 2 — Result structs** hold everything the design produced (gains, covariances, poles, success flag). They are pure data with `.as<U>()` for type conversion. A student can `static_assert(result.success)` and get a compile error instead of a silent wrong answer.

**Tier 3 — Runtime controllers** are lightweight objects that run in an ISR or RTOS loop. They store only what's needed (the gain matrix), expose a single `.control(x)` method, and do one matrix-vector multiply per call. No heap, no iteration, no failure modes.

When adding a new control algorithm, implement all three tiers.

### Naming

**Descriptive names are primary. Short aliases are provided for experts.**

```cpp
// Primary API — what students find and read
design::discrete_lqr(A, B, Q, R);
design::discrete_lqr_from_continuous(A, B, Q, R, Ts);

// Short aliases — what MATLAB users expect
design::dlqr(A, B, Q, R);
design::lqrd(A, B, Q, R, Ts);
```

Rules:

- Every public function must have a descriptive name that reads as English: `discrete_lqr`, `kalman_filter`, `cholesky_solve`, `closed_loop_poles`
- MATLAB-style short names (`dlqr`, `dare`, `ctrb`, `obsv`) are thin aliases that call the descriptive version
- Result types use `PascalCase` with `Result` suffix: `LQRResult`, `KalmanResult`, `PIDResult`
- Runtime controller classes use short `PascalCase`: `LQR`, `LQI`, `PIDController`
- Type aliases for common sizes: `Mat2`, `Vec3`, `Quatd` — always optional convenience, never required

### Error Reporting

Match the error mechanism to the audience:

| Layer | Mechanism | Why |
| ----- | --------- | --- |
| Dimensions wrong | Template constraints / `static_assert` | Caught at compile time. Student sees "no matching function" not a runtime crash. |
| Design didn't converge | `bool success` on result struct | Student writes `static_assert(result.success)` or `if (!result.success)`. Caught at compile time when result is `constexpr`. |
| Single operation failed | `std::optional<T>` | Matrix inversion, Cholesky decomposition. Caller must handle with `.value()` or `.has_value()`. |
| Runtime controller | No failure mode | Controllers operate on already-validated designs. `.control(x)` always returns a value. |

`tl::expected` (`libs/expected.hpp`) is reserved for cases where the *reason* for failure matters — e.g., "DARE didn't converge" vs. "matrix is singular" vs. "system is not stabilizable". Use it when `std::optional` would leave the student guessing *why* something failed.

### Function Signatures

```cpp
// Design function pattern
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr LQRResult<NX, NU, T> discrete_lqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}   // optional params last, with defaults
);
```

Rules:

- `[[nodiscard]]` on everything that returns a value
- Accept matrices by `const&`
- Return by value (rely on NRVO)
- Default `T = double` — students shouldn't need to think about types until deployment
- Optional parameters at the end with zero-initialized defaults
- Overloads that accept `StateSpace` should delegate to the matrix-argument version, not duplicate logic

### Member vs. Free Function

- **Member functions** for operations intrinsic to the type: `.transpose()`, `.inverse()`, `.norm()`, `.block<R,C>(r,c)`, `.control(x)`
- **`mat::` free functions** for operations that take a matrix and produce something else: `mat::expm(A)`, `mat::cholesky(A)`, `mat::det(A)`, `mat::rank(A)`
- **`design::` free functions** for control design: `design::discrete_lqr(...)`, `design::kalman(...)`
- **`stability::` / `analysis::`** for queries about systems: `stability::is_stable(A)`, `analysis::controllability(A, B)`

The test: if it feels like "doing something *to* a matrix," it's a `mat::` free function. If it feels like "asking a matrix *about itself*," it's a member.

### Type Safety for Embedded

The design-to-deployment pipeline must be type-safe end-to-end:

```cpp
// 1. Design in double (full precision for numerical algorithms)
constexpr auto result = design::discrete_lqr(A, B, Q, R);
static_assert(result.success);

// 2. Convert to float (embedded target type)
constexpr auto result_f = result.as<float>();

// 3. Instantiate controller (stores only what's needed for runtime) marked constinit
constinit LQR<2, 1, float> controller(result_f);

// 4. Runtime loop — one matrix-vector multiply, no allocations
ColVec<1, float> u = controller.control(x);
```

Rules:

- Every result struct and controller must support `.as<U>()` for type conversion
- Runtime controllers default to `T = float` (embedded default), design functions default to `T = double`
- `StateSpace` requires `std::is_floating_point_v<T>` — no complex state-space systems (complex appears only in eigenvalues and internal computations)
- Use `T{1}` not `1.0` in templated code — a `float` template instantiation must produce float arithmetic, not implicit double promotion

### Operator Overloads

Operators are reserved for operations with obvious mathematical meaning:

| Operator | Matrix meaning | StateSpace meaning |
| -------- | -------------- | ------------------ |
| `A + B` | Element-wise addition | Parallel connection |
| `A - B` | Element-wise subtraction | Differencing junction |
| `A * B` | Matrix multiplication | Series/cascade connection |
| `sys / fb` | — | Negative feedback loop |
| `-A` | Negation | — |
| `A * scalar` | Scalar multiplication | — |

Do not overload operators for non-obvious operations. If the meaning isn't immediately clear from the math, use a named function.

### Progressive Complexity

Simple cases must be simple. Advanced features are opt-in.

```cpp
// Beginner: I just want an LQR for my balance bot
LQR controller(design::discrete_lqr(A, B, Q, R));
auto u = controller.control(x);

// Intermediate: I want to check stability and convert types
constexpr auto result = design::discrete_lqr(A, B, Q, R);
static_assert(result.success);
static_assert(result.is_stable());
LQR controller(result.as<float>());

// Advanced: I want cross-term weighting and continuous-time design
constexpr auto result = design::discrete_lqr_from_continuous(sys, Q, R, Ts, N);
// Access Riccati solution, closed-loop poles, etc.
auto S = result.S;
auto poles = result.e;
```

Rules:

- Default arguments should make the common case trivial (e.g., `N = 0`, `Ts` implicit when possible)
- Don't require users to specify template parameters that can be deduced
- Static factories (`.zeros()`, `.identity()`) over constructors for non-obvious initialization
- Aggregate initialization with designated initializers for `StateSpace`: `StateSpace sys{.A = ..., .B = ..., .C = ...}`

### Documentation Standards

Every public function, type, and concept must be documented. The audience is a student who has never seen a Riccati equation, reading the header for the first time.

**Doxygen comments on all public API surfaces.** Use `@brief`, `@param`, `@return`, `@tparam`. Keep `@brief` to one sentence.

**What goes in the comment vs. what doesn't:**

```cpp
/**
 * @brief Discrete-time Linear-Quadratic Regulator design
 *
 * Computes the optimal state-feedback gain K that minimizes the cost function:
 *
 *     J = Σ [ xᵀQx + uᵀRu + 2xᵀNu ]
 *
 * subject to the discrete-time dynamics x[k+1] = Ax[k] + Bu[k].
 *
 * The gain is applied as u = -Kx. The solution is found via the Discrete
 * Algebraic Riccati Equation (DARE).
 *
 * @note Compare with MATLAB's dlqr(A, B, Q, R, N).
 *
 * @see dare() for the underlying Riccati solver
 * @see lqrd() to design from a continuous-time system
 * @see "Optimal Control" (Anderson & Moore, 1990), Chapter 4
 *
 * @param A  State transition matrix (NX × NX)
 * @param B  Control input matrix (NX × NU)
 * @param Q  State cost matrix (NX × NX, positive semidefinite)
 * @param R  Input cost matrix (NU × NU, positive definite)
 * @param N  Cross-term cost matrix (NX × NU, default: zero)
 * @return LQRResult with gain K, Riccati solution S, and closed-loop poles
 */
```

Rules:

- **State the math.** Write the equation being solved, the cost function being minimized, or the identity being exploited. Use Unicode math (`xᵀQx`, `Σ`, `∈`) — it renders in IDEs, tooltips, and Doxygen output.
- **State the MATLAB equivalent** with `@note Compare with MATLAB's ...` when one exists. This is how both students and PhDs will search for the function.
- **Link related functions** with `@see`. Always link the underlying solver, the convenience overload, and any design-to-controller path.
- **Document matrix requirements** in `@param` — say "positive semidefinite", "positive definite", "must be square", "must be invertible" when applicable. This is the single most common mistake students make.
- **Don't restate the signature.** `@param A State transition matrix` is useful; `@param A The A matrix` is noise.

**Result struct documentation:**

```cpp
/**
 * @brief LQR design result
 *
 * Contains the optimal gain, Riccati solution, and closed-loop stability
 * information. Use .as<float>() to convert for embedded deployment.
 *
 * @see "Optimal Control" (Anderson & Moore, 1990), §4.3
 */
template<size_t NX, size_t NU, typename T = double>
struct LQRResult {
    Matrix<NU, NX, T>           K{};       ///< Optimal gain: u = -Kx
    Matrix<NX, NX, T>           S{};       ///< Riccati equation solution (DARE)
    ColVec<NX, wet::complex<T>> e{};       ///< Closed-loop poles (eigenvalues of A - BK)
    bool                        success{false}; ///< true if DARE converged and K was computed
};
```

Use `///< inline` comments for struct members — they render in IDE hover tooltips and are easier to scan than a block comment above each field.

### Academic References

Every algorithm must cite its source. This serves two purposes: students learn where to read more, and reviewers can verify correctness against the original formulation.

**Reference format in Doxygen:**

```cpp
/// @see "Paper/Book Title" (Author, Year), §Section or Eq. Number
```

**Where to cite:**

- **Design functions**: cite the textbook or paper that defines the algorithm. E.g., DARE solver cites Anderson & Moore or Laub's original SDA paper.
- **Numerical methods**: cite the specific algorithm variant. E.g., Padé approximation for matrix exponential cites Higham (2005), Denman-Beavers for matrix square root cites Higham (2008).
- **Controller architectures**: cite the original formulation. E.g., ADRC cites Han (1999), ESKF cites Solà (2017).

**Canonical references for this library (use these when applicable):**

| Topic | Reference |
| ----- | --------- |
| LQR / DARE / Optimal Control | Anderson & Moore, "Optimal Control: Linear Quadratic Methods" (1990) |
| Kalman Filter | Simon, "Optimal State Estimation" (2006) |
| PID Tuning | Åström & Hägglund, "Advanced PID Control" (2006) |
| Matrix Computations | Golub & Van Loan, "Matrix Computations" (4th ed., 2013) |
| Matrix Functions (expm, sqrt, log) | Higham, "Functions of Matrices" (2008) |
| ESKF / Sensor Fusion | Solà et al., "Quaternion kinematics for the error-state Kalman filter" (2017) |
| Sliding Mode Control | Edwards & Spurgeon, "Sliding Mode Control" (1998) |
| ADRC | Han, "From PID to Active Disturbance Rejection Control" (2009) |
| SDA (Riccati solver) | Chu et al., "Structure-Preserving Algorithms for Periodic DRE" (2004) |
| Discretization (ZOH/Tustin) | Franklin, Powell & Emami-Naeini, "Feedback Control of Dynamic Systems" (2015) |

When implementing a new algorithm, add its canonical reference to this table.

### Examples

Every header must have at least one complete, self-contained example in its Doxygen `@file` comment or in `examples/`. "Complete" means a student can copy-paste it into `main()` and compile.

**In-header example (short, inline):**

```cpp
/**
 * @file lqr.hpp
 * @brief Linear-Quadratic Regulator design and runtime controller
 *
 * Example: LQR for a double integrator (position + velocity control)
 * @code
 * #include "wet/controllers/lqr.hpp"
 * #include "wet/systems/state_space.hpp"
 *
 * using namespace wet;
 *
 * // Double integrator: x = [position, velocity], u = [acceleration]
 * constexpr StateSpace sys{
 *     .A = Matrix<2,2>{{0.0, 1.0}, {0.0, 0.0}},
 *     .B = Matrix<2,1>{{0.0}, {1.0}},
 *     .C = Matrix<1,2>{{1.0, 0.0}}
 * };
 *
 * constexpr auto result = design::discrete_lqr_from_continuous(
 *     sys.A, sys.B,
 *     Matrix<2,2>::identity(),    // Q: equal weight on position and velocity
 *     Matrix<1,1>{{0.1}},        // R: penalize control effort
 *     0.01                        // Ts: 100 Hz sample rate
 * );
 * static_assert(result.success);
 *
 * // Deploy to embedded target
 * LQR controller(result.as<float>());
 * // In ISR: auto u = controller.control(x);
 * @endcode
 */
```

**Standalone examples (`examples/`)** should demonstrate realistic use cases:

- Give the system a name and physical context ("inverted pendulum", "DC motor", "quadrotor attitude")
- Show the full design → deploy pipeline, not just the function call
- Include comments that explain *why* each tuning parameter was chosen, not just what it is
- Print or plot results so the student can see it working

### Teaching Through the API

The library should teach control theory through use. A student who reads the examples and the Doxygen comments should come away understanding not just *how* to use the function but *when* and *why*.

**For beginners (students, hobbyists):**

- Every example should name the physical system and explain what the states, inputs, and outputs represent
- Comments should explain tuning: "Q penalizes state deviation — increase Q(0,0) to make position tracking tighter"
- Error messages (via `static_assert` or `tl::expected`) should suggest what went wrong: "DARE did not converge — check that (A, B) is stabilizable and Q is positive semidefinite"
- The `@see` links should point to both the academic reference and a simpler prerequisite: `@see discrete_lqr for the underlying design` on an `LQG` function

**For experts (PhD engineers, reviewers):**

- State the exact algorithm variant and cite it: "Solves DARE via the Structure-Preserving Doubling Algorithm (Chu et al., 2004)"
- Document numerical properties: convergence rate, conditioning sensitivity, what causes failure
- Expose intermediate results (Riccati solution `S`, closed-loop poles `e`) on the result struct — don't hide the math behind a convenience wrapper
- Use standard notation from the cited reference (match variable names where possible: `S` for Riccati, `K` for gain, `L` for Kalman gain, `P` for covariance)

### Numerical Implementation

These rules come from an external audit and reflect hard-won lessons from production numerical code.

**Never form explicit matrix inverses.** Use decomposition-based solves instead:

```cpp
// WRONG — numerically unstable, slow
auto K = B.transpose() * S * A * (R + B.transpose() * S * B).inverse();

// RIGHT — use Cholesky solve (R + BᵀSB is positive definite)
auto K = mat::cholesky_solve(R + B.transpose() * S * B, B.transpose() * S * A);
```

This applies everywhere: LQR gain, Kalman gain, ZOH discretization, DARE internals. The library has `mat::cholesky_solve()` and `mat::lu_solve()` — use them.

**Rearrange equations into `Ax = b` form** to use triangular solves. Example for Kalman gain `K = PCᵀS⁻¹`:

```text
K = PCᵀS⁻¹  →  KS = PCᵀ  →  SᵀKᵀ = CPᵀ  →  Kᵀ = S.solve(CP)  →  K = S.solve(CP)ᵀ
```

(Transposes on symmetric S and P drop out.) Use LLT (Cholesky) when the matrix is positive definite (e.g., R in LQR), LU otherwise.

**Constexpr math should delegate to `<cmath>` at runtime.** Use `std::is_constant_evaluated()`:

```cpp
constexpr T sqrt(T x) {
    if (!std::is_constant_evaluated()) {
        return std::sqrt(x);  // hardware-accelerated at runtime
    }
    // series expansion for compile-time
}
```

The compile-time series expansions are necessary for consteval but slower than hardware intrinsics. Don't pay that cost at runtime.

**Precondition checks on solvers:**

- Check Q is positive semidefinite via LDLT (inspect pivot signs)
- Check R is positive definite via LLT (fails on zero/negative pivot)
- Check Q and R are symmetric (LLT/LDLT can explode on non-symmetric input)
- Check (A, B) is stabilizable before DARE (not just stable — stabilizable is weaker)
- Provide no-precondition-check overloads for performance-critical paths where the developer knows their system is well-formed

**Use Unicode math in code comments.** The `unicodeit` Python package converts LaTeX to Unicode. At minimum, use `ᵀ` for transpose instead of `'` or `^T`.

**Use `wet::` for every transcendental, not `std::`.** `wet::sin/cos/atan2/sqrt/log/exp/asin/acos/atan/fmod/isfinite/...` dispatch through `MathBackend<T>` at runtime and through constexpr series/Newton/etc. at compile time. Calling `std::sin` directly inside the library breaks constexpr design paths and bypasses the user's chosen backend; `make embedded-check`-style audit greps will catch it (`grep -rnE 'std::(sin|cos|sqrt|...)' inc/wet | grep -v inc/wet/math/` must stay empty).

**Use the `wet::` aliases for every backed type, not `std::`.** The same rule applies to the container/utility facilities the backend layer remaps: `wet::array`, `wet::optional`, `wet::tuple`, `wet::pair`, `wet::clamp`, `wet::min`, `wet::max`, `wet::move`, `wet::forward`, `wet::numbers::*` (and `wet::swap`, `wet::make_tuple`, …). These resolve to either `std::` or `etl::` per the active backend profile; writing `std::pair`/`std::array`/`std::clamp` directly in `inc/wet` defeats the freestanding/ETL build (`make freestanding-check` exists to catch hosted leaks). Reach for a raw `std::` type only when there is genuinely no `wet::` alias for it.

**This holds in `tests/` too, with one carve-out.** Tests may use `std::` math (`std::sin`, `std::sqrt`, `std::abs`, …) deliberately as an *independent oracle* — checking `wet::`'s result against the stdlib's is the point. But there is no oracle reason to reach for `std::pair`/`std::array`/`std::optional`/`std::clamp`: use the `wet::` alias so tests also exercise the active backend. (If you catch a `std::pair` in a new test, that's the smell — swap it for `wet::pair`.)

### Floating-point and `-ffast-math` contract

The library splits floating-point semantics across two paths, and contributors must keep them straight:

- **Compile-time (`constexpr design::*` paths).** Evaluated by the compiler's constant evaluator, which follows the language's abstract IEEE-754 semantics regardless of optimizer flags. `-ffast-math` does not reach this path — the value a `static_assert` sees is the same one a strict-IEEE TU would compute. This is the library's contract.
- **Runtime (`MathBackend<T>` and user code).** Dispatches to the user-selected backend (default `std::`) and is governed by the *user's* compiler flags. Under their `-ffast-math`, `MathBackend::isfinite` may return `true` for `∞`, near-singular accuracy degrades, and algebraic cancellations may not cancel. That's the flag's contract, not a library bug.

The test runner defaults to building with `-ffast-math` so any future compiler that leaks fast-math into constant evaluation breaks the constexpr `static_assert`s and refuses to produce the runner (see `tests/Tupfile.lua`). To spot-check strict-IEEE runtime behavior or chase a precision regression, drop `-ffast-math` from `TEST_CXXFLAGS` locally.

Implications for new numerical code:

- **Constexpr branches must guard domain edges *before* the `is_constant_evaluated()` dispatch** so compile- and runtime behavior agree. The pattern is in `wet::asin`/`acos` (clamp `|x| ≤ 1`) and `wet::fmod` (guard `y == 0`). If you only guard the constexpr arm, runtime returns NaN where constexpr returns the clamped value — a silent two-mode bug.
- **Never rely on exact algebraic cancellation** (e.g., `(b0 + b1 + b2) == (1 + a1 + a2)` because terms `±2ζω₀k` happen to cancel). Under `-fassociative-math` the optimizer is free to reorder, and the cancellation no longer cancels. Restructure so the desired property is computed directly. `design::lowpass_2nd` was the known offender (now fixed: its numerator taps are derived from the unit-DC-gain identity by construction). The RBJ biquad designers were swept and verified non-fragile under `-ffast-math` (`test_biquad.cpp` checks each one's band-edge gain identity from the stored coefficients to 1e-9–1e-12); keep that pattern — add a stored-coefficient identity check when you introduce a new designer.
- **For NaN/Inf guards, only `wet::isfinite`'s constexpr branch is `-ffast-math`-safe.** Runtime guards on IEEE specials are at the user's mercy. If a divergence-detection check matters at runtime, lean on magnitude bounds (`wet::abs(x) > guard`) rather than `isfinite` alone.

### What Not To Do

- **No runtime polymorphism.** No `virtual`, no `std::function`, no type erasure. Templates and concepts only.
- **No heap allocation.** No `std::vector`, `std::string`, `new`, or `std::unique_ptr` in the library core.
- **No exceptions.** Use `std::optional`, `tl::expected`, or `bool success`.
- **No global state.** Every function is pure or operates on explicit state passed by argument.
- **No implicit conversions** between unrelated types. Use `.as<U>()` for explicit type conversion, explicit constructors for related types.
- **No configuration macros.** Behavior is controlled by template parameters and function arguments, not `#define`.
- **Don't reimplement generic embedded plumbing.** Stay in the controls/DSP lane (see *Scope* below).

## Scope: stay in the controls lane

This library is **controls and DSP**: controllers, estimators, filters, system/transform math, signal-conditioning blocks, interpolation tables, encoder/tach, motion planning. General-purpose embedded plumbing is **out of scope** — the [Embedded Template Library (ETL)](https://www.etlcpp.com) already covers it well and battle-tested, and is vendored at `libs/etl` (optional submodule) as the recommended companion.

Before writing a new utility, check whether ETL already provides it. **Do not reimplement** things ETL covers, e.g.:

- containers / fixed-capacity vectors, queues, stacks, maps → `etl::vector`, `etl::queue`, `etl::flat_map`, …
- ring / FIFO / circular buffers → `etl::queue_spsc_atomic`, `etl::queue_spsc_locked`, `etl::circular_buffer`
- CRC / checksums → `etl::crc*`, `etl::checksum`
- generic debounce, integer compile-time math → `etl::debounce`, `etl::sqrt<>`/`etl::log<>`
- `std`-replacement types for freestanding builds → `etl::array`/`optional`/`tuple`/… (under the ETL backend; see *Backends & dependencies* below)

Conversely, **keep** controls-specific value even when it superficially resembles a utility — e.g. the constexpr-float math backend (`wet::sqrt/exp/...`, which ETL has no equivalent for; ETL's math is integer-compile-time or runtime), `Lut1D`/`Lut2D` interpolation, the DSP blocks in `filters/blocks.hpp`, `QuadratureDecoder`/`Tachometer`.

### Backends & dependencies

**No mandatory third-party dependency; two backend configurations.** The embeddable core (`wet/control.hpp`) is intended to be usable **with the C++ stdlib** (default — also the "standalone, no third-party lib" case) or **with ETL** (for freestanding/embedded targets), chosen by a backend profile (`wet_profile.hpp`; default = stdlib) that maps the small set of `std`-replacement types the core needs — `array`/`optional`/`tuple`/`pair`/`clamp`/`numbers` — to either `std::` or `etl::`. We do **not** invent our own container/optional primitives; anything not from the stdlib comes from ETL. ETL is a *companion, not a dependency*.

Per-part dependencies:

- **Core** (`wet/control.hpp`) — a backend (stdlib or ETL) + the `wet::` math backend; no third-party lib required; ships on target.
- **Math backend** (`wet::sin/sqrt/exp/...`) — freestanding-capable; the default implementation uses `<cmath>`, swappable via `wet_profile.hpp`.
- **Host superset** (`wet/toolbox.hpp`: analysis, simulation, MATLAB-style aliases) — needs the full hosted stdlib (`<vector>`, `<string>`, …); plotting also needs plotlypp. Never on target.
- **Examples** need fmt (some need plotlypp); **tests** need doctest. Both host-only.

The freestanding (ETL-backed) build is planned work; the stdlib configuration is what exists today.

## Third-party libraries

Located in `libs/` as git submodules:

- **fmt** — formatting (compiled as static lib for tests/examples)
- **nlohmann/json** — JSON (header-only, tests/examples)
- **plotlypp** — plotting (header-only, examples)
- **expected.hpp** — `tl::expected` (header-only, available but not yet used in core)
- **etl** — Embedded Template Library (optional companion for generic embedded plumbing, and the freestanding backend; not a core dependency — see *Scope* and *Backends & dependencies* above)
