"""
Plot wet:: trig approximation error against double-precision reference.

Python mirrors of the wet:: C++ code paths evaluate sin, cos, sincos, and
atan2 in float32, then measure absolute and ULP error vs numpy (float64).

Usage:
    python plot_accuracy.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_coefficients import SIN_COEFFS, ASIN_COEFFS, ATAN_COEFFS

SIN_COEFFS = np.array(SIN_COEFFS, dtype=np.float32)
ASIN_COEFFS = np.array(ASIN_COEFFS, dtype=np.float32)
ATAN_COEFFS = np.array(ATAN_COEFFS, dtype=np.float32)

INV_PI_F32 = np.float32(1.0 / np.pi)


# ---------------------------------------------------------------------------
# Python mirrors of wet:: C++ (float32 throughout)
# ---------------------------------------------------------------------------

def reduce_f32(x_f32):
    """Range reduce: frac = (x/pi) - nearbyint(x/pi), period_index = n."""
    t = x_f32 * INV_PI_F32
    n = np.rint(t).astype(np.float32)
    return (t - n), n.astype(np.int32)


def sin_poly_f32(g, sc=SIN_COEFFS):
    """sin(pi*g) = g * (c0 + c1*u + c2*u^2 + c3*u^3), u = g^2.

    Evaluation order matches the C++ sin_poly (precomputed u^2)."""
    u = g * g
    u2 = u * u
    p = sc[0]
    p = p + sc[1] * u
    p = p + sc[2] * u2
    p = p + sc[3] * u2 * u
    return g * p


def wet_sin(x):
    """Mirror of wet::sin."""
    x_f32 = x.astype(np.float32)
    g, n = reduce_f32(x_f32)
    s = sin_poly_f32(g)
    return np.where(n % 2 != 0, -s, s).astype(np.float64)


def wet_cos(x):
    """Mirror of wet::cos: cos(pi*g) = sin(pi*(0.5 - |g|))."""
    x_f32 = x.astype(np.float32)
    g, n = reduce_f32(x_f32)
    c = sin_poly_f32(np.float32(0.5) - np.abs(g))
    return np.where(n % 2 != 0, -c, c).astype(np.float64)


def wet_sincos(x):
    """Mirror of wet::sincos."""
    x_f32 = x.astype(np.float32)
    g, n = reduce_f32(x_f32)
    s = sin_poly_f32(g)
    c = sin_poly_f32(np.float32(0.5) - np.abs(g))
    odd = (n % 2) != 0
    return (np.where(odd, -s, s).astype(np.float64),
            np.where(odd, -c, c).astype(np.float64))


def _horner_f32(coeffs, x):
    """Horner evaluation in float32."""
    x = x.astype(np.float32)
    p = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        p = (c + x * p).astype(np.float32)
    return p


def wet_atan2(y, x, ac=ATAN_COEFFS):
    """Mirror of wet::atan2."""
    y_f32 = y.astype(np.float32)
    x_f32 = x.astype(np.float32)
    ax, ay = np.abs(x_f32), np.abs(y_f32)
    lo, hi = np.minimum(ax, ay), np.maximum(ax, ay)
    t = _horner_f32(ac, (lo / hi).astype(np.float32))
    pi_half = np.float32(np.pi / 2.0)
    pi_f32 = np.float32(np.pi)
    r = np.where(ay > ax, pi_half - t, t).astype(np.float32)
    r = np.where(x_f32 >= np.float32(0.0), r, pi_f32 - r).astype(np.float32)
    return np.copysign(r, y_f32).astype(np.float64)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def ulp_error(approx, ref):
    """ULP error in float32.  NaN where |ref| < 1e-6 (zero-crossing)."""
    a32, r32 = np.float32(approx), np.float32(ref)
    spacing = np.spacing(np.abs(r32)).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ulps = np.abs(a32.astype(np.float64) - r32.astype(np.float64)) / spacing
    ulps[np.abs(r32) < 1e-6] = np.nan
    return ulps


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _envelope(x, y, n_bins=400):
    """Bin y by x and return (centers, max_per_bin) for envelope plotting."""
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.searchsorted(bins, x) - 1, 0, n_bins - 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    maxes = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = idx == i
        if m.any():
            maxes[i] = y[m].max()
    return centers, maxes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N = 200000

    # --- Polynomial-only error (no range reduction) ---
    g = np.linspace(-0.5, 0.5, N).astype(np.float32)
    poly_approx = sin_poly_f32(g).astype(np.float64)
    poly_ref = np.sin(np.pi * g.astype(np.float64))
    poly_err = np.abs(poly_approx - poly_ref)

    # --- Full-pipeline sin/cos/sincos over [-4pi, 4pi] ---
    full_x = np.linspace(-4 * np.pi, 4 * np.pi, N)
    ref_sin, ref_cos = np.sin(full_x), np.cos(full_x)
    ws, wc = wet_sin(full_x), wet_cos(full_x)
    wsc_s, wsc_c = wet_sincos(full_x)

    # --- atan2 over unit circle ---
    A = 100000
    theta = np.linspace(-np.pi, np.pi, A)
    y_f32 = np.sin(theta).astype(np.float32)
    x_f32 = np.cos(theta).astype(np.float32)
    ref_a2 = np.arctan2(y_f32.astype(np.float64), x_f32.astype(np.float64))
    wet_a2 = wet_atan2(y_f32, x_f32)
    a2_err = np.abs(wet_a2 - ref_a2)

    # --- Summary table ---
    rows = [
        ("sin_poly  g in [-0.5, 0.5]", poly_err),
        ("wet::sin  [-4pi, 4pi]",      np.abs(ws - ref_sin)),
        ("wet::cos  [-4pi, 4pi]",      np.abs(wc - ref_cos)),
        ("wet::sincos.sin",            np.abs(wsc_s - ref_sin)),
        ("wet::sincos.cos",            np.abs(wsc_c - ref_cos)),
        ("wet::atan2 [-pi, pi]",       a2_err),
    ]
    print("wet:: accuracy vs float64 reference")
    print("=" * 55)
    for name, err in rows:
        print(f"  {name:35s}  max |err| = {np.max(err):.6e}")

    # --- Plots ---
    fig, axes = plt.subplots(4, 1, figsize=(11, 16))
    fig.suptitle("wet:: accuracy vs double-precision reference",
                 fontsize=14, y=0.995)

    # Row 0: polynomial-only
    ex, ey = _envelope(g.astype(np.float64), poly_err)
    axes[0].semilogy(ex, ey, linewidth=1.2, color="C0")
    axes[0].set_title("sin_poly(g) error, no range reduction", fontsize=12)
    axes[0].set_xlabel("g  (half-periods)")
    axes[0].set_ylabel("|approx - ref|")
    axes[0].grid(True, alpha=0.3)

    # Rows 1-2: sin/cos over one period
    P = 50000
    px = np.linspace(0, 2 * np.pi, P)
    pref_s, pref_c = np.sin(px), np.cos(px)

    for ax, err, name in [(axes[1], np.abs(wet_sin(px) - pref_s), "sin"),
                          (axes[2], np.abs(wet_cos(px) - pref_c), "cos")]:
        ex, ey = _envelope(px, err)
        ax.semilogy(ex, ey, linewidth=1.2, color="C0")
        ax.set_title(f"wet::{name} over [0, 2π]", fontsize=12)
        ax.set_xlabel("angle (rad)")
        ax.set_ylabel("|approx - ref|")
        ax.grid(True, alpha=0.3)

    # Row 3: atan2
    ex, ey = _envelope(theta, a2_err)
    axes[3].semilogy(ex, ey, linewidth=1.2, color="C0")
    axes[3].set_title("wet::atan2, inputs = (sin θ, cos θ) in float32",
                      fontsize=12)
    axes[3].set_xlabel("angle θ (rad)")
    axes[3].set_ylabel("|approx - ref| (rad)")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.985))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "wet_accuracy.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
