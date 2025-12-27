# Control Systems Library - AI Coding Guidelines

## Environment
- **Platform**: Windows with PowerShell
- **Python**: Use `py -3` instead of `python3` or `python`
- **Shell**: PowerShell commands, not bash
- **Build System**: Tup (not Make/CMake directly)
This is a modern C++ control systems library implementing Linear Time-Invariant (LTI) system analysis. Core components are `TransferFunction` and `StateSpace` classes supporting continuous/discrete systems with full frequency/time domain analysis.

## Architecture Patterns

### Core LTI System Classes
- **TransferFunction**: Polynomial numerator/denominator representation
- **StateSpace**: Matrix-based (A,B,C,D) representation
- Both inherit from abstract `LTI` base class with `Ts` for discrete timing

### Key Design Patterns
```cpp
// Matrix types use Eigen with custom wrappers
using Matrix = Eigen::MatrixXd;
struct ColVec : Eigen::VectorXd { /* initializer_list support */ };

// LTI inheritance hierarchy
class LTI { /* virtual interface */ };
class TransferFunction : public LTI { std::vector<double> num, den; };
class StateSpace : public LTI { Matrix A, B, C, D; };

// Template-based solver with concepts
template <typename IntegratorType>
    requires AdaptiveStepIntegrator<IntegratorType>
class AdaptiveStepSolver { /* ... */ };
```

## Build System & Development Workflow

### Tup Build System
- **Root Tuprules.lua**: Defines compiler flags, includes, optimization settings
- **Directory Tupfile.lua**: Each subdirectory (examples/, libs/, test/, python/) has build rules
- **Key commands**:
  ```powershell
  tup --quiet compiledb    # Generate compile_commands.json
  tup                      # Build all targets
  make test               # Run tests (via tup + test_runner.exe)
  make examples           # Build example programs
  ```

### Dependencies & Includes
```lua
INCLUDES = {
    '.', 'source', 'libs', 'libs/eigen',
    'libs/fmt/include', 'libs/matplotplusplus/source'
}
```
- **Eigen**: Linear algebra backbone
- **Matplot++**: Plotting/visualization
- **fmt**: String formatting
- **pybind11**: Python bindings in `python/` directory

## Code Style & Conventions

### Formatting (clang-format)
```yaml
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 0
AlignConsecutiveDeclarations: true
AlignConsecutiveAssignments: true
```
- Run `clang-format -i source/*.hpp examples/*.cpp` before commits

### Naming & Structure
- **Namespace**: `control`
- **Types**: PascalCase (TransferFunction, StateSpace, ColVec)
- **Methods**: snake_case (is_stable(), poles(), zeros())
- **Files**: snake_case with .hpp/.cpp extension
- **Headers**: `#pragma once` with IWYU pragmas

### Testing with doctest
```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Component Name") {
    SUBCASE("Specific test case") {
        // Test implementation
        CHECK(result == expected);
    }
}
```

## Common Patterns & Gotchas

### LTI System Creation
```cpp
// Transfer function: num/den coefficients
TransferFunction tf({1.0, 2.0}, {1.0, 3.0, 2.0});

// State space: matrices with optional Ts
StateSpace ss(A_matrix, B_matrix, C_matrix, D_matrix, Ts);

// Discrete systems: include Ts parameter
TransferFunction discrete_tf({1.0}, {1.0, -0.5}, 0.1);  // Ts = 0.1
```

### System Conversions
```cpp
// TF ↔ SS conversions
StateSpace ss = control::ss(transfer_func);
TransferFunction tf = control::tf(state_space);

// MIMO systems: specify input/output indices
TransferFunction tf_ij = control::tf(ss, output_idx, input_idx);
```

### LTI Arithmetic Operations
```cpp
// Series connection: sys1 * sys2
auto series_sys = controller * plant;

// Parallel connection: sys1 + sys2
auto parallel_sys = sys1 + sys2;

// Feedback: feedback(forward, feedback, sign)
auto closed_loop = feedback(open_loop, sensor, -1);
```

### Python Bindings (pybind11)
```cpp
// In python/pycontrol.cpp
PYBIND11_MODULE(pycontrol, m) {
    py::class_<TransferFunction>(m, "TransferFunction")
        .def(py::init<std::vector<double>, std::vector<double>>())
        .def("poles", &TransferFunction::poles)
        .def("step", &TransferFunction::step);
}
```

## Critical Implementation Notes

### Poles/Zeros Computation
- **TransferFunction**: Companion matrix eigenvalue computation
- **StateSpace**: Direct A matrix eigenvalues
- Both return `std::vector<Pole>` (complex<double>)

### Stability Analysis
- **Continuous**: All poles in left half-plane (negative real parts)
- **Discrete**: All poles inside unit circle
- Use `is_stable()` method or check `poles()` results

### Frequency Domain Analysis
- **Bode/Nyquist**: Log-spaced frequency points, 1000+ points typical
- **Margins**: Find gain/phase crossovers by searching response arrays
- **Discrete systems**: Limit frequency range to Nyquist frequency

### Solver Architecture
- **Template-based**: Integrators must satisfy concepts
- **Adaptive stepping**: For stiff/non-stiff ODEs
- **Exact LTI solver**: Direct matrix exponential for linear systems

## File Organization
```
source/          # Core library headers/sources
examples/        # Usage examples (Tupfile.lua builds executables)
test/           # doctest-based unit tests
python/         # pybind11 bindings + Python utilities
libs/           # External dependencies (Eigen, Matplot++, etc.)
```

## Development Workflow
1. **Edit code** in `source/` or add examples in `examples/`
2. **Format**: `clang-format -i` modified files
3. **Build**: `tup` (or `make compile`)
4. **Test**: `make test`
5. **Python**: `make gui` for tuning interface

## Common Pitfalls
- **Matrix dimensions**: Eigen matrices must be properly sized (A: n×n, B: n×m, C: p×n, D: p×m)
- **Discrete vs continuous**: Check `Ts.has_value()` and `isDiscrete()`
- **Normalization**: Transfer functions should have monic denominators
- **MIMO limitations**: Many methods (zeros(), tf() conversion) only work for SISO systems
</content>
<parameter name="filePath">c:\workspace\control\.github\copilot-instructions.md
