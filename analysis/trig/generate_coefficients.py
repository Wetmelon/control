"""
Generate float32 minimax polynomial coefficients for trig.hpp.

Fits three approximations via L-infinity (minimax) linear programming,
then iteratively rounds each coefficient to float32:

  sin/cos:  sin(pi*g) = g * (c0 + c1*u + c2*u^2 + c3*u^3),  u = g^2
  asin:     asin(x)   = pi/2 - sqrt(1-x) * P(x),             degree-5
  atan:     atan(x)   = P(x),                                 degree-7 on [0, 1]

Usage:
    python generate_coefficients.py
"""

import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Minimax LP solver
# ---------------------------------------------------------------------------

def _solve_minimax_lp(V, fvals, fixed_mask, fixed_vals):
    """
    Minimize max|V @ c - fvals| over coefficients not flagged in fixed_mask.

    Fixed coefficients contribute a known offset that is subtracted from the
    target before the LP runs.  Returns (coeffs, max_error).
    """
    n_grid = V.shape[0]
    free_idx = np.where(~fixed_mask)[0]
    n_free = len(free_idx)

    target = fvals - V[:, fixed_mask] @ fixed_vals[fixed_mask]
    V_free = V[:, free_idx]

    # Variables: [free_coeffs..., E].  Minimize E subject to |residual| <= E.
    nv = n_free + 1
    c_obj = np.zeros(nv)
    c_obj[-1] = 1.0

    A_ub = np.zeros((2 * n_grid, nv))
    b_ub = np.zeros(2 * n_grid)
    A_ub[:n_grid, :n_free] = V_free
    A_ub[:n_grid, -1] = -1.0
    b_ub[:n_grid] = target
    A_ub[n_grid:, :n_free] = -V_free
    A_ub[n_grid:, -1] = -1.0
    b_ub[n_grid:] = -target

    bounds = [(None, None)] * n_free + [(0, None)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    coeffs = fixed_vals.copy()
    coeffs[free_idx] = res.x[:n_free]
    return coeffs, res.x[-1]


def _iterative_round_to_f32(V, fvals):
    """
    Float32-aware minimax: solve in float64, then round coefficients to
    float32 one at a time (highest degree first), re-solving for the
    remaining free coefficients after each step.

    High-degree-first minimizes perturbation because those coefficients
    have the smallest magnitudes in a monomial basis.
    """
    n = V.shape[1]
    fixed_mask = np.zeros(n, dtype=bool)
    fixed_vals = np.zeros(n)

    coeffs, _ = _solve_minimax_lp(V, fvals, fixed_mask, fixed_vals)

    for k in range(n - 1, -1, -1):
        fixed_vals[k] = float(np.float32(coeffs[k]))
        fixed_mask[k] = True
        if fixed_mask.all():
            break
        coeffs, _ = _solve_minimax_lp(V, fvals, fixed_mask, fixed_vals)

    return np.array([float(np.float32(v)) for v in fixed_vals])


def fpminimax_poly(f, degree, a, b, n_grid=20000):
    """Float32-aware minimax on a standard monomial basis."""
    grid = np.linspace(a, b, n_grid)
    fvals = np.array([f(v) for v in grid])
    V = np.column_stack([grid ** j for j in range(degree + 1)])
    return _iterative_round_to_f32(V, fvals)


# ---------------------------------------------------------------------------
# Coefficient generators
# ---------------------------------------------------------------------------

def generate_sin_coeffs(n_terms=4, g_max=0.5, n_grid=20000):
    """
    Fit sin(pi*g) = g * P(g^2) by minimizing max|sin(pi*g) - g*P(g^2)|.

    The design matrix has columns g^1, g^3, g^5, ... (odd powers), so
    the LP directly weights errors by |g|.
    """
    grid = np.linspace(-g_max, g_max, n_grid)
    fvals = np.sin(np.pi * grid)
    V = np.column_stack([grid ** (2 * j + 1) for j in range(n_terms)])
    return _iterative_round_to_f32(V, fvals)


def generate_asin_coeffs(degree=5, n_grid=15000):
    """
    Fit P(x) for asin(x) = pi/2 - sqrt(1-x) * P(x).

    Target: P(x) = (pi/2 - asin(x)) / sqrt(1-x), with limiting value
    sqrt(2) at x -> 1.
    """
    def target(x):
        x = np.float64(x)
        if (1.0 - x) < 1e-2:
            return np.sqrt(2.0)
        return (np.pi / 2.0 - np.arcsin(x)) / np.sqrt(1.0 - x)

    return fpminimax_poly(target, degree, 0.0, 0.99, n_grid)


def generate_atan_coeffs(degree=7, n_grid=20000):
    """Fit atan(x) = c0 + c1*x + ... + c7*x^7 on [0, 1]."""
    return fpminimax_poly(np.arctan, degree, 0.0, 1.0, n_grid)


# ---------------------------------------------------------------------------
# Shipped coefficients (must match trig.hpp)
# ---------------------------------------------------------------------------

SIN_COEFFS = generate_sin_coeffs()
ASIN_COEFFS = generate_asin_coeffs()
ATAN_COEFFS = generate_atan_coeffs()


# ---------------------------------------------------------------------------
# Accuracy evaluation (float32, matching C++ evaluation order)
# ---------------------------------------------------------------------------

def eval_sin_error(coeffs):
    """Max |sin(pi*g) - g*P(g^2)| evaluated in float32."""
    g = np.linspace(-0.5, 0.5, 20000).astype(np.float32)
    c = np.array(coeffs, dtype=np.float32)
    u = g * g
    u2 = u * u
    p = c[0] + c[1] * u + c[2] * u2 + c[3] * u2 * u
    approx = (g * p).astype(np.float64)
    ref = np.sin(np.pi * g.astype(np.float64))
    return float(np.max(np.abs(approx - ref)))


def eval_asin_error(coeffs):
    """Max |asin(x) - (pi/2 - sqrt(1-x)*P(x))| evaluated in float32."""
    x = np.linspace(0.0, 0.999, 5000).astype(np.float32)
    c = np.array(coeffs, dtype=np.float32)
    p = c[-1]
    for ci in reversed(c[:-1]):
        p = ci + x * p
    sqrt_term = np.sqrt(1.0 - x.astype(np.float64))
    approx = np.float64(np.float32(np.pi / 2.0)) - sqrt_term * p.astype(np.float64)
    ref = np.arcsin(x.astype(np.float64))
    return float(np.max(np.abs(approx - ref)))


def eval_atan_error(coeffs):
    """Max |atan(x) - P(x)| evaluated in float32 on [0, 1]."""
    x = np.linspace(0.0, 1.0, 5000).astype(np.float32)
    c = np.array(coeffs, dtype=np.float32)
    p = c[-1]
    for ci in reversed(c[:-1]):
        p = ci + x * p
    approx = p.astype(np.float64)
    ref = np.arctan(x.astype(np.float64))
    return float(np.max(np.abs(approx - ref)))


# ---------------------------------------------------------------------------
# C++ output
# ---------------------------------------------------------------------------

def float_to_c(f):
    """Format float32 as a C decimal literal with enough digits for exact round-trip."""
    return f"{float(np.float32(f)):+.15e}f"


def print_cpp_constants(sin_c, asin_c, atan_c):
    print("\nC++ constants for trig.hpp:")
    print("=" * 60)

    sections = [
        ("sin_coeffs", "sin(pi*g) = g * (c0 + c1*g^2 + c2*g^4 + c3*g^6)",
         sin_c),
        ("asin_coeffs", "asin(x) = pi/2 - sqrt(1-x) * P(x)",
         asin_c),
        ("atan_coeffs", "atan(x) = c0 + c1*x + ... + c7*x^7  (x in [0,1])",
         atan_c),
    ]
    for name, comment, coeffs in sections:
        print(f"\n// {comment}")
        print(f"constexpr float {name}[] = {{")
        for c in coeffs:
            print(f"    {float_to_c(c)},")
        print("};")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating float32 minimax coefficients")
    print("=" * 60)

    sin_c = generate_sin_coeffs()
    print(f"\nsin:  max |err| = {eval_sin_error(sin_c):.6e}")
    for i, c in enumerate(sin_c):
        print(f"  c{i} = {float_to_c(c)}")

    asin_c = generate_asin_coeffs()
    print(f"\nasin: max |err| = {eval_asin_error(asin_c):.6e}")
    for i, c in enumerate(asin_c):
        print(f"  c{i} = {float_to_c(c)}")

    atan_c = generate_atan_coeffs()
    print(f"\natan: max |err| = {eval_atan_error(atan_c):.6e}")
    for i, c in enumerate(atan_c):
        print(f"  c{i} = {float_to_c(c)}")

    print_cpp_constants(sin_c, asin_c, atan_c)


if __name__ == "__main__":
    main()
