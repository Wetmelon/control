# Compile-Time Initialization Enforcement

## Overview

The control library enforces compile-time initialization for controllers created from `design::` namespace functions. This guarantees that controllers designed at compile-time are actually initialized at compile-time, providing zero runtime overhead and catching configuration errors early.

## Quick Example

See [examples/example.cpp](examples/example.cpp) for a complete demonstration of:
- **Compile-time LQR**: Linearizing a nonlinear pendulum at equilibrium, designing controller at compile-time
- **Runtime LQR**: Linearizing about current state, adapting gain as the operating point changes

The example shows how the same nonlinear system can be controlled with either approach depending on your needs.

## How It Works

The library uses C++20's `consteval` keyword combined with separate result types to enforce compile-time evaluation:

1. **Separate Result Types**:
   - `design::LQRResult`, `design::LQIResult`, etc. - returned by `design::` functions
   - `online::LQRResult`, `online::LQIResult`, etc. - returned by `online::` functions

2. **Overloaded Constructors**:
   - `consteval` constructors accept `design::` result types → **must** be compile-time
   - `constexpr` constructors accept `online::` result types → **can** be compile-time or runtime

## Usage Examples

### ✅ Compile-Time Design (Recommended for Embedded)

```cpp
// All constexpr - guaranteed compile-time initialization
constexpr StateSpace sys{
    .A = Matrix<2, 2>{{0.0, 1.0}, {0.0, 0.0}},
    .B = Matrix<2, 1>{{0.0}, {1.0}},
    .C = Matrix<1, 2>{{1.0, 0.0}}
};
constexpr auto Q = Matrix<2, 2>::identity();
constexpr auto R = Matrix<1, 1>{{0.1}};
constexpr auto Ts = 0.01;

// design::lqrd is consteval - forces compile-time
constexpr auto res = design::lqrd(sys.A, sys.B, Q, R, Ts);
constexpr LQR controller = res;  // Zero runtime overhead!
```

### ✅ Runtime Design (Flexible for Tuning)

```cpp
void design_controller(double sample_time) {
    StateSpace sys{
        .A = Matrix<2, 2>{{0.0, 1.0}, {0.0, 0.0}},
        .B = Matrix<2, 1>{{0.0}, {1.0}},
        .C = Matrix<1, 2>{{1.0, 0.0}}
    };
    auto Q = Matrix<2, 2>::identity();
    auto R = Matrix<1, 1>{{0.1}};
    
    // online::lqrd is constexpr - allows runtime
    auto res = online::lqrd(sys.A, sys.B, Q, R, sample_time);
    LQR controller = res;  // Calls constexpr constructor
}
```

### ❌ Runtime with design:: (Compile Error)

```cpp
void invalid_design() {
    // Runtime variables
    auto sys = StateSpace{...};
    auto Q = Matrix<2, 2>::identity();
    auto R = Matrix<1, 1>{{0.1}};
    auto Ts = 0.01;
    
    // ERROR: design::lqrd is consteval, can't use runtime variables
    auto res = design::lqrd(sys.A, sys.B, Q, R, Ts);
    //         ^^^^^^^^^^^ error: call to consteval function is not a constant expression
    
    LQR controller = res;
    //              ^^^ error: call to consteval constructor is not a constant expression
}
```

**Compiler Error Message:**
```
error: call to consteval function 'design::lqrd(...)' is not a constant expression
error: call to consteval function 'LQR(res)' is not a constant expression
note: the value of 'Ts' is not usable in a constant expression
note: 'Ts' was not declared 'constexpr'
```

**Fix:** Either make all inputs `constexpr`, or use `online::lqrd()` instead.

## When to Use Each

### Use `design::` Namespace (consteval)

✓ Embedded systems with fixed control laws  
✓ When you want compile-time validation of controller design  
✓ For zero-overhead controllers with known parameters  
✓ To catch configuration errors at compile time  

### Use `online::` Namespace (constexpr)

✓ When controller parameters depend on runtime inputs  
✓ For adaptive or gain-scheduled control  
✓ During prototyping and tuning  
✓ When you need flexibility to recompute controllers  

## Available Functions

### `design::` Namespace (consteval - compile-time only)
- `dlqr()` - Discrete LQR
- `lqrd()` - Discretize continuous LQR
- `lqi()` - LQR with integral action
- `kalman()` - Kalman filter design
- `lqg()` - LQG regulator
- `lqgtrack()` - LQG with tracking
- Plus helper functions: `discrete_lqr_from_continuous()`, `lqr_with_integral()`, etc.

### `online::` Namespace (constexpr - runtime capable)
- Same function signatures as `design::`, but returns `online::` result types
- Can be used at compile-time OR runtime
- Slightly more flexible, but loses compile-time enforcement

## Result Type Conversions

You **cannot** mix result types:
```cpp
constexpr auto design_res = design::lqrd(...);  // design::LQRResult
auto online_res = online::lqrd(...);             // online::LQRResult

// These don't work:
// LQR<2, 1> c1 = online_res;  // OK if at compile-time (constexpr constructor)
// LQR<2, 1> c2 = design_res;  // OK only at compile-time (consteval constructor)
```

The type system enforces correct usage through constructor overload resolution.

## Technical Details

### Constructor Implementations

```cpp
template<size_t NX, size_t NU, typename T = double>
class LQR {
    // Consteval: MUST be compile-time
    consteval LQR(const design::LQRResult<NX, NU, T>& result) 
        : K(result.K) {}
    
    // Constexpr: CAN be compile-time or runtime  
    constexpr LQR(const online::LQRResult<NX, NU, T>& result)
        : K(result.K) {}
};
```

### Result Type Definitions

Both `design::` and `online::` result types have identical structures:
```cpp
namespace design {
    template<size_t NX, size_t NU, typename T = double>
    struct LQRResult {
        Matrix<NU, NX, T> K{};              // Feedback gain
        Matrix<NX, NX, T> P{};              // DARE solution
        ColVec<NX, wet::complex<T>> poles{};// Closed-loop poles
        bool success = false;
    };
}

namespace online {
    // Identical structure, different namespace
    template<size_t NX, size_t NU, typename T = double>
    struct LQRResult { /* same as design:: */ };
}
```

The namespace separation enables constructor overload resolution while maintaining API compatibility.

## Benefits

1. **Compile-Time Safety**: Controllers designed with `design::` functions are guaranteed to be initialized at compile time
2. **Zero Overhead**: Compile-time controllers have no runtime cost
3. **Early Error Detection**: Configuration errors caught during compilation, not at runtime
4. **Clear Intent**: Function namespace indicates whether runtime evaluation is intended
5. **Flexibility**: Can still use `online::` functions when runtime design is needed

## Verification

A test file `test_consteval_check.cpp` demonstrates all usage patterns:
- ✅ Compile-time design with `design::` → works
- ✅ Runtime design with `online::` → works  
- ✅ Compile-time design with `online::` → works (constexpr allows it)
- ❌ Runtime design with `design::` → compile error (consteval forbids it)
