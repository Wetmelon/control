# Copilot Instructions

## Project overview
- Header-only C++20 linear algebra and rotation utilities. Core types live in [inc/matrix.hpp](inc/matrix.hpp), [inc/vector.hpp](inc/vector.hpp), and [inc/rotation.hpp](inc/rotation.hpp). Tests are in [tests/](tests).
- Everything is fixed-size and stack-allocated; no dynamic allocation or STL containers beyond `std::array`/`std::span` and friends.
- Numerical focus is on small matrices/vectors (up to 4x4) for robotics/attitude work; missing pieces like Kalman/state-space are placeholders in [inc/state_space.hpp](inc/state_space.hpp) and [inc/kalman.hpp](inc/kalman.hpp).

## Key types and patterns
- `Matrix<N, M, T>` provides arithmetic, transpose, trace, and multiplication; inversion is implemented only for 1x1 and 2x2 cases and returns `std::optional`. Construction supports initializer lists and spans; element type is constrained to arithmetic.
- `ColVec`/`RowVec` wrap `Matrix` for vectors with dot/cross (3D), norm, and normalization helpers; operations reuse matrix operators so follow the same semantics (e.g., integer division truncates).
- Rotation stack in [inc/rotation.hpp](inc/rotation.hpp):
  - `DCM` (3x3) with axis-specific constructors, axis-angle factory returning `std::optional`, and composition via matrix multiply.
  - `Euler<T, Order>` templated on `EulerOrder` (default ZYX yaw-pitch-roll). Provides order-aware accessors, DCM conversion, and gimbal-lock-aware `from_dcm` logic.
  - `Quaternion` wraps a 4x1 matrix; supports normalization (safe and in-place), conjugate/inverse, Hamilton product, vector rotation, DCM/Euler conversions, axis-angle creation, body-rate integration (first-order), and `slerp` with antipodal handling.
- Type aliases: `Mat2/3/4`, `Vec2/3/4`, `Quatf/Quatd`, `DCMf/DCMd`, `EulerZYX*` and `EulerXYZ*` for convenience; use them for readability in new code.
- Prefer readable code over micro-optimizations; the compiler should handle inlining and optimizations given the header-only nature.  This will mostly be used by EEs who are not experts in template metaprogramming.

## Build and test workflow
- Preferred workflow: `tup compiledb && tup` from repo root (or `make` which runs the same) to build `tests/build/test_runner.exe`. Compilation uses g++ with `-std=c++20 -Ofast -march=native -Wall -Wextra -Werror` and gc-sections (see [tests/Tupfile.lua](tests/Tupfile.lua)).
- Tests are header-only via doctest; `tests/test_runner.cpp` defines the main. Executable is auto-run by `make` (last step in [Makefile](Makefile)).
- Include paths expected by build scripts: current directory and `../inc`. Keep headers under `inc/` to avoid tweaking build rules.

## Conventions and cautions
- Favor `constexpr` implementations; most APIs are `constexpr` and used in tests that assume compile-time availability.
- Keep operations dimension-safe and type-safe; new functionality should stick to fixed sizes and avoid heap use.
- When adding math routines, mirror existing optional-return patterns for failure cases (e.g., singular matrices, zero-length axes) instead of throwing.
- Tests assert exact values for integers and `doctest::Approx` for floats; maintain numeric stability within ~1e-6.
- Be mindful of coordinate conventions: Euler ZYX is yaw-pitch-roll, XYZ is roll-pitch-yaw; conversions are wired accordingly.

## Extending the codebase
- New rotation/linear-algebra helpers should be expressed in terms of existing primitives (Matrix/Vec/Quaternion/DCM) to retain consistency and testability.
- If expanding inversion or adding decompositions, document dimensional constraints and return `std::optional` on singular cases.
- State-space/Kalman scaffolding is empty—match the header-only, allocation-free style and reuse `Matrix`/`ColVec` for system matrices.
