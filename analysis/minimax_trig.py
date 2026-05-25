"""
Minimax polynomial approximation for sin and cos.

Range reduction normalizes any input to g in [-0.5, 0.5] via:
    n = nearbyint(x / pi)
    g = x / pi - n          (g in half-periods)

Then:
    sin(x) = (-1)^n * sin(pi*g)     polynomial in g, odd  -> g*P(g^2)
    cos(x) = sin(x + pi/2)          reuse sin with phase shift

This script:
  1. Computes minimax coefficients for sin(pi*g)/g on [0, 0.25] (u = g^2)
  2. Compares accuracy against math.sin/cos and the TI ARM approximations
  3. Prints C++ constants ready to paste
  4. Generates accuracy plots and a PDF report

Usage:
    py -3 analysis/minimax_trig.py
"""

import struct
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Minimax via linear programming
# ---------------------------------------------------------------------------

def _solve_lp(V, fvals, fixed_mask, fixed_vals):
    """Solve the minimax LP over the *free* coefficients.

    Coefficients flagged in fixed_mask contribute a known offset (fixed_vals * V_col)
    that gets subtracted from the target.  The LP then optimizes the remaining
    free coefficients to minimize E.
    """
    n_grid = V.shape[0]
    free_idx = np.where(~fixed_mask)[0]
    n_free = len(free_idx)

    fixed_contrib = np.zeros(n_grid)
    for k in np.where(fixed_mask)[0]:
        fixed_contrib += fixed_vals[k] * V[:, k]
    target = fvals - fixed_contrib

    V_free = V[:, free_idx]
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


def minimax_poly(f, degree, a, b, n_grid=20000):
    """
    Compute degree-n float64 minimax polynomial approximation of f on [a, b].

    Returns (coeffs, max_error).  Coefficients are float64-optimal; rounding
    them to float32 will degrade the achieved error (use fpminimax_poly for
    float32-aware coefficients).
    """
    grid = np.linspace(a, b, n_grid)
    fvals = np.array([f(x) for x in grid])
    V = np.column_stack([grid ** j for j in range(degree + 1)])

    fixed_mask = np.zeros(degree + 1, dtype=bool)
    fixed_vals = np.zeros(degree + 1)
    return _solve_lp(V, fvals, fixed_mask, fixed_vals)


def minimax_sin_direct(n_terms, g_max=0.5, n_grid=20000):
    """
    Fit sin(pi*g) ~= g * P(g^2) directly, minimizing max |sin(pi*g) - g*P(g^2)|.

    This is different from minimax_poly(sin_over_g, ...) which fits the ratio
    sin(pi*g)/g.  Fitting the ratio over-penalizes small |g| (where errors
    barely affect the output) and under-penalizes large |g| (where they
    dominate).  This routine bakes the |g| weighting into the LP, giving
    tighter bounds on the actual sin error.

    Returns (P_coeffs, max_sin_error) where P_coeffs are the coefficients of
    P(u) in u = g^2, matching the polynomial form used in wet_trig.cpp.
    """
    grid = np.linspace(-g_max, g_max, n_grid)
    fvals = np.sin(np.pi * grid)

    # Design columns: g^1, g^3, g^5, ... (odd powers of g)
    # Coefficient j multiplies g^(2j+1); equivalently P(u) coefficient j times g.
    V = np.column_stack([grid ** (2 * j + 1) for j in range(n_terms)])

    fixed_mask = np.zeros(n_terms, dtype=bool)
    fixed_vals = np.zeros(n_terms)
    return _solve_lp(V, fvals, fixed_mask, fixed_vals)


def _iterative_round_lp(V, fvals):
    """
    Core float32-aware iterative rounding loop.

    Given design matrix V (n_grid x n_coeffs) and target fvals, solves the
    minimax LP, then iteratively rounds coefficients to float32 (highest
    column index first), re-solving for remaining free coefficients after
    each rounding.  Returns (coeffs, final LP-reported error).

    Why high-index first: in monomial bases ordered by ascending power,
    high-index coefficients have the smallest magnitudes, so rounding them
    perturbs the polynomial least.  Low-index coefficients get to absorb
    the rounding error in subsequent LP solves.
    """
    n_coeffs = V.shape[1]
    fixed_mask = np.zeros(n_coeffs, dtype=bool)
    fixed_vals = np.zeros(n_coeffs)

    coeffs, E = _solve_lp(V, fvals, fixed_mask, fixed_vals)

    for k in range(n_coeffs - 1, -1, -1):
        fixed_vals[k] = float(np.float32(coeffs[k]))
        fixed_mask[k] = True
        if fixed_mask.all():
            coeffs = fixed_vals.copy()
            break
        coeffs, E = _solve_lp(V, fvals, fixed_mask, fixed_vals)

    return coeffs, E


def fpminimax_poly(f, degree, a, b, n_grid=20000):
    """
    Float32-aware minimax for the sinc objective: fits f on [a, b] with
    iterative coefficient rounding.  See _iterative_round_lp for details.
    """
    grid = np.linspace(a, b, n_grid)
    fvals = np.array([f(x) for x in grid])
    V = np.column_stack([grid ** j for j in range(degree + 1)])
    return _iterative_round_lp(V, fvals)


def fpminimax_sin_direct(n_terms, g_max=0.5, n_grid=20000):
    """
    Float32-aware version of minimax_sin_direct.  Fits sin(pi*g) ~= g*P(g^2)
    with iterative coefficient rounding to float32-representable values.
    """
    grid = np.linspace(-g_max, g_max, n_grid)
    fvals = np.sin(np.pi * grid)
    V = np.column_stack([grid ** (2 * j + 1) for j in range(n_terms)])
    return _iterative_round_lp(V, fvals)


def evaluate_f32_error(coeffs, f, a, b, n_grid=20000):
    """Evaluate polynomial in float32 and report max error vs f."""
    grid = np.linspace(a, b, n_grid)
    fvals = np.array([f(x) for x in grid])
    coeffs_f32 = np.array(coeffs, dtype=np.float32)
    grid_f32 = grid.astype(np.float32)

    # Horner in float32
    p = np.full_like(grid_f32, coeffs_f32[-1])
    for c in reversed(coeffs_f32[:-1]):
        p = c + grid_f32 * p
    return float(np.max(np.abs(p.astype(np.float64) - fvals)))


# ---------------------------------------------------------------------------
# Target functions  (u = g^2, domain [0, 0.25])
# ---------------------------------------------------------------------------

def sin_over_g(u):
    """sin(pi*sqrt(u)) / sqrt(u) -- polynomial in u gives sin(pi*g)/g."""
    if u < 1e-30:
        return np.pi
    s = np.sqrt(u)
    return np.sin(np.pi * s) / s


# ---------------------------------------------------------------------------
# Inverse trig functions for ODrive-style optimization
# ---------------------------------------------------------------------------

def atan_poly_f32(x, atan_coeffs):
    """Evaluate atan via polynomial on [0, 1]"""
    x_f32 = x.astype(np.float32)
    coeffs_f32 = np.array(atan_coeffs, dtype=np.float32)

    # Horner evaluation
    p = coeffs_f32[-1]
    for c in reversed(coeffs_f32[:-1]):
        p = c + x_f32 * p

    return p


def wet_atan2_f32(y, x, atan_coeffs):
    """Python float32 simulation of wet::atan2."""
    y_f32 = y.astype(np.float32)
    x_f32 = x.astype(np.float32)
    ax = np.abs(x_f32)
    ay = np.abs(y_f32)
    lo = np.minimum(ax, ay)
    hi = np.maximum(ax, ay)
    ratio = (lo / hi).astype(np.float32)
    t = atan_poly_f32(ratio, atan_coeffs)
    pi_over_2 = np.float32(np.pi / 2.0)
    pi_f32 = np.float32(np.pi)
    r = np.where(ay > ax, pi_over_2 - t, t).astype(np.float32)
    r = np.where(x_f32 >= np.float32(0.0), r, pi_f32 - r).astype(np.float32)
    return np.copysign(r, y_f32).astype(np.float32)


def lut_atan2_f32(y, x):
    """Python float32 simulation of lut::atan2 (fast_atan2 from math_utils.cpp)."""
    y_f32 = y.astype(np.float32)
    x_f32 = x.astype(np.float32)
    abs_y = np.abs(y_f32)
    abs_x = np.abs(x_f32)
    flt_min = np.finfo(np.float32).tiny
    a = (np.minimum(abs_x, abs_y) / (np.maximum(abs_x, abs_y) + np.float32(flt_min))).astype(np.float32)
    s = (a * a).astype(np.float32)
    inner = ((np.float32(-0.0464964749) * s + np.float32(0.15931422)) * s - np.float32(0.327622764))
    r = (inner * s * a + a).astype(np.float32)
    r = np.where(abs_y > abs_x, np.float32(1.57079637) - r, r).astype(np.float32)
    r = np.where(x_f32 < np.float32(0.0), np.float32(3.14159274) - r, r).astype(np.float32)
    r = np.where(y_f32 < np.float32(0.0), -r, r).astype(np.float32)
    return r


# ---------------------------------------------------------------------------
# TI ARM reimplementation (for comparison)
# ---------------------------------------------------------------------------

PI_CONSTS = [1.5707963267, 3.1415926535, 4.7123889803, 6.2831853071]
SIN_CONSTS = [
    0.999996615908002773079325846913220383,
    -0.16664828381895056829366054140948866,
    0.00830632522715989396465411782615901079,
    -0.00018363653976946785297280224158683484,
]
COS_CONSTS = [
    0.999999953466670136306412430924463351,
    -0.49999905347076729097546897993796764,
    0.0416635846931078386653947196040757567,
    -0.00138537043082318983893723662479142648,
    0.0000231539316590538762175742441588523467,
]


def ti_sin(a):
    ar = a
    if a > PI_CONSTS[0]:
        ar = PI_CONSTS[1] - a
    if a > PI_CONSTS[2]:
        ar = a - PI_CONSTS[3]
    x2 = ar * ar
    x4 = x2 * x2
    return ar * (SIN_CONSTS[0] + SIN_CONSTS[1] * x2 +
                 SIN_CONSTS[2] * x4 + SIN_CONSTS[3] * x2 * x4)


def ti_cos(a):
    ar = a
    negate = False
    if a > PI_CONSTS[0]:
        ar = a - PI_CONSTS[1]
        negate = True
    if a > PI_CONSTS[2]:
        ar = ar - PI_CONSTS[1]
        negate = False
    x2 = ar * ar
    x4 = x2 * x2
    r = (COS_CONSTS[0] + COS_CONSTS[1] * x2 + COS_CONSTS[2] * x4 +
         COS_CONSTS[3] * x2 * x4 + COS_CONSTS[4] * x4 * x4)
    return -r if negate else r


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def float_to_hex(f):
    b = struct.pack(">f", float(f))
    bits = struct.unpack(">I", b)[0]
    sign = "-" if (bits >> 31) else ""
    exp_raw = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exp_raw == 0:
        return f"{f:.10e}f"
    exp = exp_raw - 127
    return f"{sign}0x1.{mantissa << 1:06x}p{exp:+d}f"


def ulp_error_arr(approx, ref):
    """Vectorized ULP error computation in float32."""
    a32 = np.float32(approx)
    r32 = np.float32(ref)
    spacing = np.spacing(np.abs(r32))
    with np.errstate(divide="ignore", invalid="ignore"):
        ulps = np.abs(a32.astype(np.float64) - r32.astype(np.float64)) / spacing.astype(np.float64)
    ulps[np.abs(r32) < 1e-6] = np.nan
    return ulps


# ---------------------------------------------------------------------------
# Evaluation helpers (mirroring wet_trig.cpp)
# ---------------------------------------------------------------------------

def horner(coeffs, u):
    """Evaluate polynomial via Horner's method: coeffs[0] + coeffs[1]*u + ..."""
    r = np.full_like(u, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        r = c + u * r
    return r


def estrin_f32(coeffs, u):
    """Float32 mirror of wet::detail::estrin_eval.

    Evaluates coeffs[0] + coeffs[1]*x + ... pairwise, squaring the power each
    level. Rounds to float32 at every step so the result matches the compiled
    kernel's rounding (not just the math). Horner and Estrin are algebraically
    equal but round differently -- this is what lets us compare their float32
    accuracy, while the critical-path depth (~log2(N) vs N) is read from the asm.
    """
    cf = [np.float32(c) for c in coeffs]
    uf = u.astype(np.float32)
    b = [np.full_like(uf, c) for c in cf]
    power = uf.copy()
    n = len(b)
    while n > 1:
        for i in range(n // 2):
            b[i] = (b[2 * i] + power * b[2 * i + 1]).astype(np.float32)
        if n % 2 == 1:
            b[n // 2] = b[n - 1]
        power = (power * power).astype(np.float32)
        n = (n + 1) // 2
    return b[0]


def horner_f32(coeffs, u):
    """Float32 Horner, for an apples-to-apples comparison against estrin_f32."""
    cf = [np.float32(c) for c in coeffs]
    uf = u.astype(np.float32)
    r = np.full_like(uf, cf[-1])
    for c in reversed(cf[:-1]):
        r = (np.float32(c) + uf * r).astype(np.float32)
    return r


# All computation is done in float32 to mirror the C++ implementation's
# precision exactly.  Reference values stay in float64 for honest error
# measurement.
INV_PI_F32 = np.float32(1.0 / np.pi)


def _reduce_f32(x_f32, shift_f32=np.float32(0.0)):
    """Range-reduce: t = x * inv_pi + shift,  g = t - nearbyint(t)."""
    t = x_f32 * INV_PI_F32 + shift_f32
    nf = np.rint(t).astype(np.float32)
    g = t - nf
    return g, nf.astype(np.int32)


def _sin_poly_f32(g, sin_coeffs_f32):
    """Evaluate sin(pi*g) via the += chain (same shape as C++ sin_poly)."""
    u = g * g
    u2 = u * u
    p = sin_coeffs_f32[0]
    p = p + sin_coeffs_f32[1] * u
    p = p + sin_coeffs_f32[2] * u2
    p = p + sin_coeffs_f32[3] * u2 * u
    return g * p


def wet_sin(x, sin_coeffs):
    """Mirrors wet::sin: reduce(x), sin_poly(g), conditional negate."""
    x_f32 = x.astype(np.float32)
    sc_f32 = np.array(sin_coeffs, dtype=np.float32)
    g, n = _reduce_f32(x_f32)
    s = _sin_poly_f32(g, sc_f32)
    odd = (n % 2) != 0
    return np.where(odd, -s, s).astype(np.float64)


def wet_cos(x, sin_coeffs):
    """Mirrors wet::cos: reduce(x) unshifted, sin_poly(0.5 - |g|), negate.

    Same identity as wet::sincos -- folding the 0.5 shift in before
    nearbyint (reduce(x, 0.5)) loses accuracy at large |x|."""
    x_f32 = x.astype(np.float32)
    sc_f32 = np.array(sin_coeffs, dtype=np.float32)
    g, n = _reduce_f32(x_f32)
    c = _sin_poly_f32(np.float32(0.5) - np.abs(g), sc_f32)
    odd = (n % 2) != 0
    return np.where(odd, -c, c).astype(np.float64)


# ---------------------------------------------------------------------------
# CMSIS-DSP style LUT (matches math_utils.cpp fast_sin_f32)
# ---------------------------------------------------------------------------

LUT_SIZE = 512
LUT_F32 = np.array(
    [np.sin(2.0 * np.pi * i / LUT_SIZE) for i in range(LUT_SIZE + 1)],
    dtype=np.float32,
)
TWO_PI_F32 = np.float32(2.0 * np.pi)


def lut_sin(x):
    """Mirrors fast_sin_f32 from math_utils.cpp: floor-wrap + 512-entry LUT
    with linear interpolation between adjacent entries."""
    x_f32 = x.astype(np.float32)
    in_val = x_f32 / TWO_PI_F32                              # x / (2*pi)
    in_val = in_val - np.floor(in_val).astype(np.float32)    # frac in [0, 1)

    findex = np.float32(LUT_SIZE) * in_val
    index = findex.astype(np.uint32)
    index = np.clip(index, 0, LUT_SIZE - 1)
    fract = findex - index.astype(np.float32)

    a = LUT_F32[index]
    b = LUT_F32[index + 1]
    return ((np.float32(1.0) - fract) * a + fract * b).astype(np.float64)


def lut_cos(x):
    """cos(x) = sin(x + pi/2) -- same code path with a phase shift."""
    return lut_sin(x + np.float32(np.pi / 2.0))


# ---------------------------------------------------------------------------
# Wrapped ti_arm: ti_arm::sin/cos are only valid on [0, 2pi], so in practice
# you'd wrap any input first.  Use the same nearbyint-based reduction as
# wet::reduce so the wrap noise is comparable.
# ---------------------------------------------------------------------------

INV_TWO_PI_F32 = np.float32(1.0 / (2.0 * np.pi))


def wrap_to_2pi_f32(x):
    """Nearbyint-based wrap to [0, 2pi).  ~4 extra instructions in C++:
    a VMUL, VRINTR, VSUB, VFMA (to undo the multiply)."""
    x_f32 = x.astype(np.float32)
    t = x_f32 * INV_TWO_PI_F32
    n = np.rint(t).astype(np.float32)
    frac = t - n                                          # in [-0.5, 0.5]
    frac = np.where(frac < 0, frac + np.float32(1.0), frac)  # shift to [0, 1)
    return (frac * TWO_PI_F32).astype(np.float64)


def ti_sin_wrapped(x):
    wrapped = wrap_to_2pi_f32(x)
    return np.array([ti_sin(float(a)) for a in wrapped])


def ti_cos_wrapped(x):
    wrapped = wrap_to_2pi_f32(x)
    return np.array([ti_cos(float(a)) for a in wrapped])


def wet_sincos(x, sin_coeffs):
    """Mirrors wet::sincos: one reduce, sin_poly(g) and sin_poly(0.5 - |g|).

    cos is even, so cos(pi*g) = sin(pi*(0.5 - |g|)) keeps the argument in
    [0, 0.5] where the polynomial is valid.
    """
    x_f32 = x.astype(np.float32)
    sc_f32 = np.array(sin_coeffs, dtype=np.float32)
    g, n = _reduce_f32(x_f32)
    s = _sin_poly_f32(g, sc_f32)
    c = _sin_poly_f32(np.float32(0.5) - np.abs(g), sc_f32)
    odd = (n % 2) != 0
    s = np.where(odd, -s, s).astype(np.float64)
    c = np.where(odd, -c, c).astype(np.float64)
    return s, c


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _envelope(x, y, n_bins=400):
    """Bin (x, y) into n_bins and return (bin_centers, max_y_per_bin).

    Plots the worst-case error in each region instead of every single sample.
    Preserves the minimax oscillation pattern while hiding float32 noise.
    """
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.searchsorted(bins, x) - 1, 0, n_bins - 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    max_per_bin = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = idx == i
        if mask.any():
            max_per_bin[i] = y[mask].max()
    return centers, max_per_bin


def plot_error_curves(ax, series, title, ylabel, n_bins=400):
    """series is a list of (x, y, label, color) tuples.  Each series is
    binned to its worst-case error per bin for a clean envelope view."""
    for x, y, label, color in series:
        ex, ey = _envelope(x, y, n_bins)
        ax.semilogy(ex, ey, linewidth=1.2, alpha=0.85, label=label, color=color)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("angle (rad)")
    ax.set_ylabel(ylabel + "  (max per bin)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_ulp_hist(ax, series, title):
    """series is a list of (ulps, label, color) tuples."""
    valid = [(u[~np.isnan(u)], lbl, col) for u, lbl, col in series]
    all_valid = np.concatenate([v[0] for v in valid])
    # Cap at 99th percentile so zero-crossing outliers don't crush the scale
    cap = np.percentile(all_valid, 99)
    bins = np.linspace(0, cap, 60)
    for u, lbl, col in valid:
        ax.hist(np.clip(u, 0, cap), bins=bins, alpha=0.55, label=lbl, color=col)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"ULP error (capped at 99th percentile = {cap:.0f})")
    ax.set_ylabel("count")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def generate_report(sin_coeffs, sin_err, atan_coeffs):
    # --- Test data ---
    N = 200000
    full_x = np.linspace(-4 * np.pi, 4 * np.pi, N)
    ref_sin = np.sin(full_x)
    ref_cos = np.cos(full_x)

    wet_s = wet_sin(full_x, sin_coeffs)
    wet_c = wet_cos(full_x, sin_coeffs)
    wet_sc_s, wet_sc_c = wet_sincos(full_x, sin_coeffs)
    lut_s = lut_sin(full_x)
    lut_c = lut_cos(full_x)

    # Polynomial-only error: evaluate sin_poly(g) directly on g in [-0.5, 0.5]
    # (no range reduction, no quadrant logic).  This matches how other libraries
    # often report their error -- isolating the polynomial fit from wrapping noise.
    g_grid = np.linspace(-0.5, 0.5, N).astype(np.float32)
    sc_f32 = np.array(sin_coeffs, dtype=np.float32)
    poly_only = _sin_poly_f32(g_grid, sc_f32).astype(np.float64)
    poly_ref = np.sin(np.pi * g_grid.astype(np.float64))
    poly_only_err = np.abs(poly_only - poly_ref)

    # TI ARM requires input in [0, 2pi] -- in practice you'd always wrap
    # first, so we evaluate ti_arm AFTER our nearbyint-based wrap.  Now
    # the comparison is apples-to-apples over the full [-4pi, 4pi] range.
    ti_s = ti_sin_wrapped(full_x)
    ti_c = ti_cos_wrapped(full_x)
    ti_ref_sin = ref_sin
    ti_ref_cos = ref_cos

    # Absolute errors
    wet_sin_err = np.abs(wet_s - ref_sin)
    wet_cos_err = np.abs(wet_c - ref_cos)
    wet_sincos_s_err = np.abs(wet_sc_s - ref_sin)
    wet_sincos_c_err = np.abs(wet_sc_c - ref_cos)
    lut_sin_err = np.abs(lut_s - ref_sin)
    lut_cos_err = np.abs(lut_c - ref_cos)
    ti_sin_err = np.abs(ti_s - ti_ref_sin)
    ti_cos_err = np.abs(ti_c - ti_ref_cos)

    # ULP errors (subsample for histogram)
    stride = max(1, N // 20000)
    wet_sin_ulps = ulp_error_arr(wet_s[::stride], ref_sin[::stride])
    wet_cos_ulps = ulp_error_arr(wet_c[::stride], ref_cos[::stride])
    wet_sc_s_ulps = ulp_error_arr(wet_sc_s[::stride], ref_sin[::stride])
    wet_sc_c_ulps = ulp_error_arr(wet_sc_c[::stride], ref_cos[::stride])
    lut_sin_ulps = ulp_error_arr(lut_s[::stride], ref_sin[::stride])
    lut_cos_ulps = ulp_error_arr(lut_c[::stride], ref_cos[::stride])
    ti_sin_ulps = ulp_error_arr(ti_s[::stride], ti_ref_sin[::stride])
    ti_cos_ulps = ulp_error_arr(ti_c[::stride], ti_ref_cos[::stride])

    # --- Summary stats ---
    def stats(name, err, ulps):
        valid = ulps[~np.isnan(ulps)]
        return {
            "name": name,
            "max_abs": np.max(err),
            "mean_abs": np.mean(err),
            "max_ulp": np.max(valid) if len(valid) else 0,
            "mean_ulp": np.mean(valid) if len(valid) else 0,
        }

    # ULP for polynomial-only (subsample like the others)
    poly_only_ulps = ulp_error_arr(poly_only[::stride], poly_ref[::stride])

    table = [
        stats("sin_poly only  g in [-1/2, 1/2]", poly_only_err, poly_only_ulps),
        stats("wet::sin       [-4pi, 4pi]",      wet_sin_err, wet_sin_ulps),
        stats("wet::cos       [-4pi, 4pi]",      wet_cos_err, wet_cos_ulps),
        stats("wet::sincos.s  [-4pi, 4pi]",      wet_sincos_s_err, wet_sc_s_ulps),
        stats("wet::sincos.c  [-4pi, 4pi]",      wet_sincos_c_err, wet_sc_c_ulps),
        stats("lut::sin       [-4pi, 4pi]",      lut_sin_err,  lut_sin_ulps),
        stats("lut::cos       [-4pi, 4pi]",      lut_cos_err,  lut_cos_ulps),
        stats("ti_arm::sin (wrapped) [-4pi, 4pi]", ti_sin_err, ti_sin_ulps),
        stats("ti_arm::cos (wrapped) [-4pi, 4pi]", ti_cos_err, ti_cos_ulps),
    ]

    # --- Print summary ---
    print("\n" + "=" * 72)
    print("Accuracy comparison")
    print("=" * 72)
    for s in table:
        print(f"\n  {s['name']}")
        print(f"    max |err|  = {s['max_abs']:.6e}")
        print(f"    mean |err| = {s['mean_abs']:.6e}")
        print(f"    max ULP    = {s['max_ulp']:.1f}")
        print(f"    mean ULP   = {s['mean_ulp']:.1f}")

    # --- Per-period plot data (one clean cycle, not 8 squashed periods) ---
    P = 50000
    period_x = np.linspace(0.0, 2 * np.pi, P)
    period_ref_sin = np.sin(period_x)
    period_ref_cos = np.cos(period_x)

    period_wet_sin_err = np.abs(wet_sin(period_x, sin_coeffs) - period_ref_sin)
    period_wet_cos_err = np.abs(wet_cos(period_x, sin_coeffs) - period_ref_cos)
    period_lut_sin_err = np.abs(lut_sin(period_x) - period_ref_sin)
    period_lut_cos_err = np.abs(lut_cos(period_x) - period_ref_cos)
    period_ti_sin_err  = np.abs(ti_sin_wrapped(period_x) - period_ref_sin)
    period_ti_cos_err  = np.abs(ti_cos_wrapped(period_x) - period_ref_cos)

    # --- PDF report ---
    pdf_path = "analysis/build/minimax_trig_report.pdf"
    png_path = "analysis/build/minimax_trig_report.png"

    fig, axes = plt.subplots(5, 1, figsize=(11, 18))
    fig.suptitle("Minimax trig approximation — wet:: vs TI ARM",
                 fontsize=15, y=0.995)

    # Row 0: sin error curve (one period)
    plot_error_curves(axes[0], [
        (period_x, period_wet_sin_err, "wet::sin", "C0"),
        (period_x, period_lut_sin_err, "lut::sin", "C2"),
        (period_x, period_ti_sin_err,  "ti_arm::sin", "C1"),
    ], "sin — absolute error over one period [0, 2pi]", "|approx − ref|")

    # Row 1: cos error curve (one period)
    plot_error_curves(axes[1], [
        (period_x, period_wet_cos_err, "wet::cos", "C0"),
        (period_x, period_lut_cos_err, "lut::cos", "C2"),
        (period_x, period_ti_cos_err,  "ti_arm::cos", "C1"),
    ], "cos — absolute error over one period [0, 2pi]", "|approx − ref|")

    # Row 2-3: ULP histograms (stats computed over the full range)
    plot_ulp_hist(axes[2], [
        (wet_sin_ulps, "wet",    "C0"),
        (lut_sin_ulps, "lut",    "C2"),
        (ti_sin_ulps,  "ti_arm", "C1"),
    ], "sin — ULP error distribution (full range)")

    plot_ulp_hist(axes[3], [
        (wet_cos_ulps, "wet",    "C0"),
        (lut_cos_ulps, "lut",    "C2"),
        (ti_cos_ulps,  "ti_arm", "C1"),
    ], "cos — ULP error distribution (full range)")

    # Row 4: summary table
    axes[4].axis("off")
    col_labels = ["", "max |err|", "mean |err|", "max ULP", "mean ULP"]
    cell_text = []
    for s in table:
        cell_text.append([
            s["name"],
            f"{s['max_abs']:.2e}",
            f"{s['mean_abs']:.2e}",
            f"{s['max_ulp']:.1f}",
            f"{s['mean_ulp']:.1f}",
        ])

    tbl = axes[4].table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.8)
    axes[4].set_title("Summary", fontsize=12, pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    import os
    os.makedirs("analysis/build", exist_ok=True)

    fig.savefig(png_path, dpi=150)
    print(f"\nSaved: {png_path}")

    # -------------------------------------------------------------------------
    # Page 2: atan2 error — wet:: vs lut:: vs np.arctan2 (float32)
    # -------------------------------------------------------------------------
    A = 100000
    theta = np.linspace(-np.pi, np.pi, A)
    # Inputs as float32 (as they would arrive from ADC / observer on embedded)
    y_f32 = np.sin(theta).astype(np.float32)
    x_f32 = np.cos(theta).astype(np.float32)
    # Double-precision reference
    ref_atan2 = np.arctan2(y_f32.astype(np.float64), x_f32.astype(np.float64))

    wet_a2   = wet_atan2_f32(y_f32, x_f32, atan_coeffs).astype(np.float64)
    lut_a2   = lut_atan2_f32(y_f32, x_f32).astype(np.float64)
    np_a2_f32 = np.arctan2(y_f32, x_f32).astype(np.float64)  # std::atan2f baseline

    wet_a2_err  = np.abs(wet_a2   - ref_atan2)
    lut_a2_err  = np.abs(lut_a2   - ref_atan2)
    np_a2_err   = np.abs(np_a2_f32 - ref_atan2)

    def abs_stats(name, err):
        return {
            "name":     name,
            "max_abs":  float(np.max(err)),
            "mean_abs": float(np.mean(err)),
        }

    atan2_table = [
        abs_stats("wet::atan2  [-pi, pi]", wet_a2_err),
        abs_stats("lut::atan2  [-pi, pi]", lut_a2_err),
        abs_stats("np.arctan2f [-pi, pi]", np_a2_err),
    ]

    print("\n" + "=" * 72)
    print("atan2 accuracy comparison (inputs: float32 sin/cos of swept angle)")
    print("=" * 72)
    for s in atan2_table:
        print(f"\n  {s['name']}")
        print(f"    max |err|  = {s['max_abs']:.6e} rad  ({np.degrees(s['max_abs']) * 1e3:.4f} mdeg)")
        print(f"    mean |err| = {s['mean_abs']:.6e} rad  ({np.degrees(s['mean_abs']) * 1e3:.4f} mdeg)")

    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 9))
    fig2.suptitle("atan2 accuracy — wet:: vs lut:: vs np.arctan2 (float32 inputs)",
                  fontsize=14, y=0.998)

    # Row 0: absolute error curve vs angle
    plot_error_curves(axes2[0], [
        (theta, wet_a2_err,  "wet::atan2",       "C0"),
        (theta, lut_a2_err,  "lut::atan2",       "C2"),
        (theta, np_a2_err,   "np.arctan2 (f32)", "C3"),
    ], "atan2 — absolute error vs angle, inputs = (sin θ, cos θ) in float32",
       "|approx − ref| (rad)")
    axes2[0].set_xlabel("angle θ (rad)")

    # Row 1: summary table
    axes2[1].axis("off")
    col_labels2 = ["", "max |err| (rad)", "max |err| (mdeg)", "mean |err| (rad)", "mean |err| (mdeg)"]
    cell_text2 = []
    for s in atan2_table:
        cell_text2.append([
            s["name"],
            f"{s['max_abs']:.2e}",
            f"{np.degrees(s['max_abs']) * 1e3:.4f}",
            f"{s['mean_abs']:.2e}",
            f"{np.degrees(s['mean_abs']) * 1e3:.4f}",
        ])
    tbl2 = axes2[1].table(
        cellText=cell_text2,
        colLabels=col_labels2,
        loc="center",
        cellLoc="center",
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(10)
    tbl2.scale(1.0, 2.5)
    axes2[1].set_title("Summary", fontsize=12, pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.992])

    fig2.savefig(png_path.replace(".png", "_atan2.png"), dpi=150)
    print(f"Saved: {png_path.replace('.png', '_atan2.png')}")

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
    print(f"Saved: {pdf_path} (2 pages)")

    plt.close(fig)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Minimax polynomial coefficients for sin/cos + inverse trig")
    print("=" * 72)

    # =========================================================================
    # SIN/COS COEFFICIENTS (4 terms)
    # =========================================================================
    n_terms = 4  # matches TI ARM's op count

    # --- Direct objective, float32-aware iterative rounding ---
    fp_coeffs, fp_err = fpminimax_sin_direct(n_terms, g_max=0.5)

    # Evaluate achieved error in float32 for each
    def eval_f32_sin(coeffs):
        N = 20000
        g_grid = np.linspace(-0.5, 0.5, N).astype(np.float32)
        sc = np.array(coeffs, dtype=np.float32)
        p = _sin_poly_f32(g_grid, sc).astype(np.float64)
        ref = np.sin(np.pi * g_grid.astype(np.float64))
        return float(np.max(np.abs(p - ref)))

    sin_actual = eval_f32_sin(fp_coeffs)
    sin_coeffs = fp_coeffs

    print(f"\nsin/cos: max error = {sin_actual:.6e}")
    print("  Coefficients for sin(pi*g) = g * (c0 + c1*u + c2*u^2 + c3*u^3), u = g^2:")
    for i, c in enumerate(sin_coeffs):
        print(f"    sin_coeff[{i}] = {float_to_hex(np.float32(c))};")

    # =========================================================================
    # ASIN/ACOS COEFFICIENTS - Using the form asin(x) = pi/2 - sqrt(1-x)*P(x)
    # =========================================================================
    print("\n" + "=" * 72)
    print("Generating asin/acos coefficients (x in [0, 1])")
    print("=" * 72)
    
    # Fit asin using the numerically better-conditioned form:
    # asin(x) = pi/2 - sqrt(1-x) * P(x)
    # 
    # We fit P(x) by solving for it:
    # P(x) = (pi/2 - asin(x)) / sqrt(1-x)
    #
    # To avoid numerical issues, we:
    # 1. Work in double precision for fitting
    # 2. Avoid x values too close to 1.0
    # 3. Use Taylor expansion for near x=1
    
    def asin_target_stable(x):
        """Target function P(x) for fitting: (pi/2 - asin(x)) / sqrt(1-x)"""
        x_np = np.asarray(x, dtype=np.float64)
        one_minus_x = 1.0 - x_np
        
        # For values very close to 1, use limiting behavior
        # Near x→1: asin(x) → pi/2, so (pi/2 - asin(x)) ~ sqrt(2(1-x))
        # Therefore P(x) ~ sqrt(2)
        
        result = np.zeros_like(x_np, dtype=np.float64)
        
        # For x < 0.99, use the formula directly
        safe_mask = (one_minus_x > 1e-2)
        if np.any(safe_mask):
            x_safe = x_np[safe_mask]
            sqrt_1_minus_x = np.sqrt(1.0 - x_safe)
            result[safe_mask] = (np.pi / 2.0 - np.arcsin(x_safe)) / sqrt_1_minus_x
        
        # For x >= 0.99, use limiting value sqrt(2)
        near_one_mask = ~safe_mask
        result[near_one_mask] = np.sqrt(2.0)
        
        return result.item() if np.isscalar(x) else result
    
    # Use fpminimax_poly on the stable target, fitting on [0, 0.99]
    asin_coeffs_opt, _ = fpminimax_poly(
        asin_target_stable,
        degree=5, 
        a=0.0, 
        b=0.99,
        n_grid=15000
    )
    
    # Evaluate error in float32: check |asin(x) - (pi/2 - sqrt(1-x)*P(x))|
    def eval_f32_asin_form(coeffs):
        N = 5000
        x_grid = np.linspace(0.0, 0.999, N).astype(np.float32)
        coeffs_f32 = np.array(coeffs, dtype=np.float32)
        
        # Horner evaluation for P(x)
        p = coeffs_f32[-1]
        for c in reversed(coeffs_f32[:-1]):
            p = c + x_grid * p
        
        # Compute asin approx = pi/2 - sqrt(1-x) * P(x)
        sqrt_term = np.sqrt(1.0 - x_grid.astype(np.float64))
        approx = np.float32(np.pi / 2.0) - sqrt_term * p.astype(np.float64)
        
        ref = np.arcsin(x_grid.astype(np.float64))
        return float(np.max(np.abs(approx - ref)))
    
    asin_actual = eval_f32_asin_form(asin_coeffs_opt)
    
    # Pad coefficients to 7 terms (one more than degree-5)
    asin_coeffs_final = list(asin_coeffs_opt) + [0.0] * (7 - len(asin_coeffs_opt))
    
    print(f"\nasin: max error = {asin_actual:.6e} (form: pi/2 - sqrt(1-x)*P(x))")
    print("  Coefficients for P(x) in pi/2 - sqrt(1-x) * P(x):")
    for i, c in enumerate(asin_coeffs_final):
        print(f"    asin_coeff[{i}] = {float_to_hex(np.float32(c))};")




    # =========================================================================
    # ATAN COEFFICIENTS (8 terms for x in [0, 1])
    # =========================================================================
    print("\n" + "=" * 72)
    print("Generating atan coefficients (x in [0, 1])")
    print("=" * 72)
    
    atan_coeffs_opt, _ = fpminimax_poly(np.arctan, degree=7, a=0.0, b=1.0, n_grid=20000)
    
    def eval_f32_atan(coeffs):
        N = 5000
        x_grid = np.linspace(0.0, 1.0, N).astype(np.float32)
        approx = atan_poly_f32(x_grid, coeffs).astype(np.float64)
        ref = np.arctan(x_grid.astype(np.float64))
        return float(np.max(np.abs(approx - ref)))
    
    atan_actual = eval_f32_atan(atan_coeffs_opt)
    print(f"\natan: max error = {atan_actual:.6e}")
    print("  Coefficients for P(x) = c0 + c1*x + c2*x^2 + ... + c7*x^7:")
    for i, c in enumerate(atan_coeffs_opt):
        print(f"    atan_coeff[{i}] = {float_to_hex(np.float32(c))};")

    # =========================================================================
    # PRINT ALL COEFFICIENTS FOR C++ HEADER
    # =========================================================================
    print("\n" + "=" * 72)
    print("C++ Header Constants (Ready to Paste)")
    print("=" * 72)
    print()
    print("// SIN/COS: sin(pi*g) = g * (c0 + c1*g^2 + c2*g^4 + c3*g^6)")
    print("constexpr float sin_coeffs[] = {")
    for i, c in enumerate(sin_coeffs):
        print(f"    {float_to_hex(np.float32(c))},  // c{i}")
    print("};")
    
    print()
    print("// ASIN/ACOS: asin(x) = pi/2 - sqrt(1-x) * (c0 + c1*x + ... + c6*x^6)")
    print("constexpr float asin_coeffs[] = {")
    for i, c in enumerate(asin_coeffs_opt):
        print(f"    {float_to_hex(np.float32(c))},  // c{i}")
    print("};")
    
    print()
    print("// ATAN: atan(x) = c0 + c1*x + c2*x^2 + ... + c7*x^7  (for x in [0,1])")
    print("constexpr float atan_coeffs[] = {")
    for i, c in enumerate(atan_coeffs_opt):
        print(f"    {float_to_hex(np.float32(c))},  // c{i}")
    print("};")

    generate_report(sin_coeffs, sin_actual, atan_coeffs_opt)





if __name__ == "__main__":
    main()
