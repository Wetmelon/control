# Copilot Instructions

## Project Overview
- **Header-only C++20 library** for compile-time control design and linear algebra, targeting embedded systems with zero dynamic allocation
- **Core architecture**: Fixed-size `Matrix<N,M,T>`, `StateSpace<NX,NU,NY>`, rotation types (DCM/Quaternion/Euler), and MATLAB®-style control APIs
- **Two evaluation modes**: `design::` (consteval/compile-time only) and `online::` (constexpr/runtime-capable)
- **Focus**: Robotics/attitude control with arbitrarily large matrices at compile-time, extensive compile-time verification, and embedded suitability

## Key Types and Patterns
- **`Matrix<N, M, T>`**: Core fixed-size matrix with arithmetic (`+`, `-`, `*`), transpose, trace, Gauss-Jordan inversion (returns `std::optional`), block views via `.block<Rows,Cols>(r,c)`
- **`StateSpace<NX, NU, NY, NW, NV, T>`**: System representation with operator overloads for interconnections:
  - `sys1 * sys2` → series/cascade connection
  - `sys1 + sys2` → parallel/summing junction
  - `sys1 - sys2` → differencing junction
  - `sys1 / sys2` → negative feedback loop
- **Rotation types**: `DCM<T>` (3×3), `Quaternion<T>` (4×1 with SLERP), `Euler<T, Order>` (ZYX=yaw-pitch-roll, XYZ=roll-pitch-yaw)
- **Result types**: `LQRResult`, `LQIResult`, `KalmanResult`, `LQGResult`, `LQGIResult` - all support `.as<U>()` type conversion
- **Type aliases**: `Mat2/3/4`, `Vec2/3/4`, `Quatf/Quatd`, `DCMf/DCMd`, `EulerZYX*`, `EulerXYZ*` for readability

## Critical Developer Workflows
- **Build system**: Run `make` from repo root - auto-formats code, builds via tup, runs all tests and examples
- **Testing**: Doctest framework with dual verification:
  - `static_assert()` for compile-time guarantees
  - Runtime `CHECK()` with `doctest::Approx` for numerical validation
- **Golden reference data**: Use `py -3` with scipy/control libraries to generate reference values for numerical tests
- **Include paths**: Headers in `inc/` directory, build expects `-I.` and `-I../inc`
- **output folders**:  Output paths in Tupfiles are relative to the Tupfile location.  E.g. `tests/Tupfile` outputs to `tests/build`, so test_runner is at `tests/build/test_runner.exe`

## Project-Specific Conventions
**Constexpr-first design**: All algorithms must work at compile-time; prefer `consteval` for design functions. Heavy matrix operations (e.g., LQR gain computation) are expected to evaluate at compile-time, while simple operations on small matrices may occur at runtime (e.g., single matrix multiplication in a control loop)
**Failure handling**: Return `std::optional<T>` for operations that can fail (matrix inversion, DARE convergence, axis-angle conversion)
**Dimension safety**: Template constraints prevent invalid operations at compile-time
**Numerical precision**: Maintain machine precision accuracy for compile-time matrix math, equivalent to SciPy or MATLAB; use `doctest::Approx().epsilon(1e-12)` or appropriate tolerance in tests for double precision
**Euler angle conventions**: ZYX = yaw-pitch-roll (aerospace), XYZ = roll-pitch-yaw (robotics)
**Controller classes**: `LQR<NX,NU,T>`, `LQI<NX,NU,NY,T>`, `LQG<NX,NU,NY,T>`, `LQGI<NX,NU,NY,T>` for runtime use
**Test organization**: One `.cpp` file per header, descriptive `TEST_CASE` names, mix of unit and integration tests

**Type support**: All matrix and control algorithms must support `float`, `double`, `wet::complex<float>`, `wet::complex<double>`, and their `const` versions. Use type traits to ensure correct overloads and conversions. Always maintain and test for `wet::complex` support in all new features and bugfixes.

## Integration Points and Data Flow
- **System interconnections**: Block matrix augmentation preserves noise matrices (`G`, `H`) through all operations
- **Control design flow**: `StateSpace` → `design::function()` → `Result.as<float>()` → `ControllerClass()` → runtime control loop
- **Noise propagation**: Process noise `Q` and measurement noise `R` flow through discretization and interconnections
- **Type conversion chain**: Design in `double` precision → convert to `float` for embedded use → instantiate controller classes

## External Dependencies and Environment
- **Python ecosystem**: `scipy` and `control` libraries for generating golden test data
- **Build tools**: `tup` for dependency tracking, `clang-format` for code formatting
- **Compiler requirements**: GCC 10+, Clang 12+, MSVC 2022+ with C++20 support
- **Platform**: Windows development environment with PowerShell terminal

## Extending the Codebase
- **New control algorithms**: Follow `design::`/`online::` pattern, return result structs with `.as<U>()` conversion
- **Matrix operations**: Build on existing `Matrix` primitives, document dimensional constraints
- **Rotation utilities**: Express in terms of DCM/Quaternion/Euler, handle gimbal-lock cases
- **Test additions**: Use descriptive names, include both `static_assert` compile-time checks and runtime numerical validation
- **Examples**: Demonstrate compile-time design → runtime controller instantiation workflow
- **Backards compatibility**: Do *not* maintain backwards compatibility; prefer clarity and correctness.

## Human Factors
- **Changlog**: After each prompt, update the `COPILOT_LOG.md` with a 1-2 sentence summary of the change. This helps maintain a clear history of modifications and their rationale.  No need for headers, just use a bullet point for each entry.
- **Code style**: Follow the existing code style and formatting conventions; use `clang-format` for consistency. This includes naming conventions, indentation, and comment style.  Always use brackets after `if`, `for`, `while`, even for single-line bodies, to prevent bugs and improve readability.
