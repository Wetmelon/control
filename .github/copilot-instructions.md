<!-- Repository-specific Copilot instructions for Wetmelon/control -->

# Copilot / AI agent instructions

Purpose: give actionable, repository-specific guidance so an AI coding agent can be immediately productive.

- Quick Start
  - Build (convenience wrapper): `make all` — runs `tup --quiet compiledb`, `tup`, then runs the test binary.
  - Compile only: `make compile` or run `tup --quiet compiledb && tup`.
  - Run tests: `make test` or run the built test runner `./test/build/test_runner.exe` (Windows `.exe`).
  - Format code: `make format` (runs `clang-format -i` on headers and sources).
  - Run examples: `make examples`.
  - Launch Python GUI: `make gui` (runs `tup --quiet python` then `py -3 python/odrive_tuning_gui.py`).

- Where to look first
  - `source/` — primary C++ library. Header files (.hpp) define public API; .cpp files implement algorithms.
  - `test/` — unit tests using the single-header `doctest.h`. Tests produce `test/build/test_runner.exe`.
  - `examples/` — small example programs demonstrating public API usage (useful for integration-level examples).
  - `python/` — Python GUI and bindings (`pycontrol.cpp`), run via the `make gui` flow.
  - `libs/` — bundled third-party code (Eigen, fmt, matplotplusplus, etc.).
  - `compile_commands.json` — present for editor/clang tooling; useful for static analysis and code completion.

- High-level architecture (why/how)
  - This repo implements control-systems algorithms (LTI, transfer functions, state-space, LQR/LQI, Kalman, etc.) as a C++ library in `source/`.
  - Examples and tests exercise the public API rather than rely on big integration infra — prefer adding an `examples/example_xxx.cpp` or `test/test_xxx.cpp` when adding features.
  - The build is driven by Tup (Tupfiles in top-level and subfolders). The top-level `Makefile` wraps common flows for convenience on Windows.

- Project-specific patterns and conventions
  - Public API is declared in `.hpp` files in `source/` (e.g., [source/types.hpp](source/types.hpp) defines `using Matrix = Eigen::MatrixXd` and `ColVec/RowVec`).
  - Tests use `doctest::Approx(...).epsilon(1e-3)` for numeric comparisons; match that style for new floating-point checks.
  - Prefer adding small example programs under `examples/` to demonstrate cross-language usage (C++ → Python) or algorithm behavior.
  - Use `clang-format` (invoked by `make format`) to keep style consistent.

- Build & CI notes
  - Primary build tool: `tup`. The `Makefile` is the canonical wrapper on Windows; CI scripts may call `tup` directly.
  
- Testing and verification
  - Unit tests: build with `tup --quiet test` or `make test`, then execute `./test/build/test_runner.exe`.
  - Example checks: build and run binaries under `examples/` (use `make examples`).

- Integration points and external dependencies
  - Python GUI and bindings live in `python/` — C++ code exposes functionality to Python via `pycontrol.cpp` and is included in Tup rules.
  - Third-party libs are vendored under `libs/` — treat these as external dependencies and do not modify them unless necessary.

- Guidance for the AI agent (concise)
  - Do: Read `source/` and `test/` before making changes. Add tests in `test/` for new behavior.
  - Do: Use uniform initializer syntax `{}` for object construction (e.g., `ExtendedKalmanFilter ekf{...};`).
  - Do: Run `make format` before committing; run `make test` to verify changes on Windows.
  - Do: Keep `Eigen` types (see [source/types.hpp](source/types.hpp)) and `doctest` patterns consistent; use `doctest::Approx` for numeric asserts.
  - Do:  Always use robust numerical methods for control algorithms; prefer established techniques (e.g., Kalman filters, LQR) over ad-hoc solutions.
  - Don't: Change vendored `libs/` unless absolutely required. Don't assume CMake — this repo uses `tup` and `Makefile` wrappers.
  - Don't: Retain backwards compatibility for public API unless explicitly requested; prefer small, clear APIs.  The library has not shipped yet.
  - Don't: Add examples unless reqquested; focus on tests for verification.
  - Don't: "Simplify" numerical methods at the cost of stability or accuracy.

If anything below is unclear or you want coverage for additional areas (packaging, CI hooks, or contribution rules), tell me which area to expand.
