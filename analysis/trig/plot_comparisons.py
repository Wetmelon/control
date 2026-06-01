"""
Compare wet:: trig against TI ARM, CMSIS-DSP LUT, std::sin, and Cody-Waite.

Supplementary to the core accuracy analysis in plot_accuracy.py.  This file
is not required to verify the wet:: implementation itself.

Usage:
    python plot_comparisons.py
"""

import struct
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from plot_accuracy import (
    SIN_COEFFS, ATAN_COEFFS, INV_PI_F32,
    reduce_f32, sin_poly_f32, wet_sin, wet_cos, wet_sincos, wet_atan2,
    ulp_error, _envelope,
)


# ---------------------------------------------------------------------------
# TI ARM reimplementation (scalar, for comparison only)
# ---------------------------------------------------------------------------

_PI_C = [1.5707963267, 3.1415926535, 4.7123889803, 6.2831853071]
_SN = [
    0.999996615908002773079325846913220383,
    -0.16664828381895056829366054140948866,
    0.00830632522715989396465411782615901079,
    -0.00018363653976946785297280224158683484,
]
_CS = [
    0.999999953466670136306412430924463351,
    -0.49999905347076729097546897993796764,
    0.0416635846931078386653947196040757567,
    -0.00138537043082318983893723662479142648,
    0.0000231539316590538762175742441588523467,
]


def _ti_sin(a):
    ar = a
    if a > _PI_C[0]: ar = _PI_C[1] - a
    if a > _PI_C[2]: ar = a - _PI_C[3]
    x2 = ar * ar; x4 = x2 * x2
    return ar * (_SN[0] + _SN[1]*x2 + _SN[2]*x4 + _SN[3]*x2*x4)


def _ti_cos(a):
    ar, negate = a, False
    if a > _PI_C[0]: ar = a - _PI_C[1]; negate = True
    if a > _PI_C[2]: ar = ar - _PI_C[1]; negate = False
    x2 = ar * ar; x4 = x2 * x2
    r = _CS[0] + _CS[1]*x2 + _CS[2]*x4 + _CS[3]*x2*x4 + _CS[4]*x4*x4
    return -r if negate else r


INV_TWO_PI_F32 = np.float32(1.0 / (2.0 * np.pi))
TWO_PI_F32 = np.float32(2.0 * np.pi)


def _wrap_to_2pi(x):
    x_f32 = x.astype(np.float32)
    t = x_f32 * INV_TWO_PI_F32
    n = np.rint(t).astype(np.float32)
    frac = t - n
    frac = np.where(frac < 0, frac + np.float32(1.0), frac)
    return (frac * TWO_PI_F32).astype(np.float64)


def ti_sin_wrapped(x):
    return np.array([_ti_sin(float(a)) for a in _wrap_to_2pi(x)])

def ti_cos_wrapped(x):
    return np.array([_ti_cos(float(a)) for a in _wrap_to_2pi(x)])


# ---------------------------------------------------------------------------
# CMSIS-DSP style 512-entry LUT with linear interpolation
# ---------------------------------------------------------------------------

_LUT_SIZE = 512
_LUT = np.array(
    [np.sin(2.0 * np.pi * i / _LUT_SIZE) for i in range(_LUT_SIZE + 1)],
    dtype=np.float32,
)


def lut_sin(x):
    x_f32 = x.astype(np.float32)
    in_val = x_f32 / TWO_PI_F32
    in_val = in_val - np.floor(in_val).astype(np.float32)
    findex = np.float32(_LUT_SIZE) * in_val
    index = np.clip(findex.astype(np.uint32), 0, _LUT_SIZE - 1)
    fract = findex - index.astype(np.float32)
    a, b = _LUT[index], _LUT[index + 1]
    return ((np.float32(1.0) - fract) * a + fract * b).astype(np.float64)


def lut_cos(x):
    return lut_sin(x + np.float32(np.pi / 2.0))


def lut_atan2(y, x):
    y_f32, x_f32 = y.astype(np.float32), x.astype(np.float32)
    ay, ax = np.abs(y_f32), np.abs(x_f32)
    flt_min = np.finfo(np.float32).tiny
    a = (np.minimum(ax, ay) / (np.maximum(ax, ay) + np.float32(flt_min))).astype(np.float32)
    s = (a * a).astype(np.float32)
    inner = (np.float32(-0.0464964749) * s + np.float32(0.15931422)) * s - np.float32(0.327622764)
    r = (inner * s * a + a).astype(np.float32)
    r = np.where(ay > ax, np.float32(1.57079637) - r, r).astype(np.float32)
    r = np.where(x_f32 < np.float32(0.0), np.float32(3.14159274) - r, r).astype(np.float32)
    return np.where(y_f32 < np.float32(0.0), -r, r).astype(np.float64)


# ---------------------------------------------------------------------------
# Cody-Waite reduction variant (three-word pi)
# ---------------------------------------------------------------------------

def _split_pi():
    pi = np.float64(np.pi)
    bits = struct.unpack("<I", struct.pack("<f", np.float32(pi)))[0]
    hi = np.float32(struct.unpack("<f", struct.pack("<I", bits & 0xFFFFE000))[0])
    lo = np.float32(pi - np.float64(hi))
    lo2 = np.float32(pi - np.float64(hi) - np.float64(lo))
    return hi, lo, lo2

_PI_HI, _PI_LO, _PI_LO2 = _split_pi()


def wet_sin_codywaite(x):
    """wet::sin with explicit Cody-Waite (matches shipped wet_trig.hpp)."""
    x = x.astype(np.float32)
    n = np.rint(x * INV_PI_F32).astype(np.float32)
    r = (x - n * _PI_HI).astype(np.float32)
    r = (r - n * _PI_LO).astype(np.float32)
    r = (r - n * _PI_LO2).astype(np.float32)
    g = (r * INV_PI_F32).astype(np.float32)
    s = sin_poly_f32(g)
    return np.where(n.astype(np.int64) % 2 != 0, -s, s).astype(np.float64)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _envelope_log(x, y, nbins=350):
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), nbins + 1)
    idx = np.clip(np.searchsorted(bins, x) - 1, 0, nbins - 1)
    cx, cy = [], []
    for i in range(nbins):
        m = idx == i
        if m.any():
            cx.append(np.sqrt(bins[i] * bins[i + 1]))
            cy.append(y[m].max())
    return np.array(cx), np.array(cy)


def _plot_error_curves(ax, series, title, ylabel, n_bins=400):
    for x, y, label, color in series:
        ex, ey = _envelope(x, y, n_bins)
        ax.semilogy(ex, ey, linewidth=1.2, alpha=0.85, label=label, color=color)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("angle (rad)")
    ax.set_ylabel(ylabel + "  (max per bin)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)


def _plot_ulp_hist(ax, series, title):
    valid = [(u[~np.isnan(u)], lbl, col) for u, lbl, col in series]
    all_valid = np.concatenate([v[0] for v in valid])
    cap = np.percentile(all_valid, 99)
    bins = np.linspace(0, cap, 60)
    for u, lbl, col in valid:
        ax.hist(np.clip(u, 0, cap), bins=bins, alpha=0.55, label=lbl, color=col)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"ULP error (capped at 99th pctl = {cap:.0f})")
    ax.set_ylabel("count")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Multi-method sin/cos comparison (absolute + ULP)
    # -----------------------------------------------------------------------
    N = 200000
    full_x = np.linspace(-4 * np.pi, 4 * np.pi, N)
    ref_sin, ref_cos = np.sin(full_x), np.cos(full_x)

    ws, wc = wet_sin(full_x), wet_cos(full_x)
    ls, lc = lut_sin(full_x), lut_cos(full_x)
    ts, tc = ti_sin_wrapped(full_x), ti_cos_wrapped(full_x)

    stride = max(1, N // 20000)
    wet_sin_ulps = ulp_error(ws[::stride], ref_sin[::stride])
    wet_cos_ulps = ulp_error(wc[::stride], ref_cos[::stride])
    lut_sin_ulps = ulp_error(ls[::stride], ref_sin[::stride])
    lut_cos_ulps = ulp_error(lc[::stride], ref_cos[::stride])
    ti_sin_ulps  = ulp_error(ts[::stride], ref_sin[::stride])
    ti_cos_ulps  = ulp_error(tc[::stride], ref_cos[::stride])

    P = 50000
    px = np.linspace(0, 2 * np.pi, P)
    prs, prc = np.sin(px), np.cos(px)

    fig, axes = plt.subplots(4, 1, figsize=(11, 16))
    fig.suptitle("Trig comparison: wet:: vs TI ARM vs 512-entry LUT",
                 fontsize=14, y=0.995)

    _plot_error_curves(axes[0], [
        (px, np.abs(wet_sin(px) - prs), "wet::sin", "C0"),
        (px, np.abs(lut_sin(px) - prs), "lut::sin", "C2"),
        (px, np.abs(ti_sin_wrapped(px) - prs), "ti_arm::sin", "C1"),
    ], "sin absolute error [0, 2pi]", "|approx - ref|")

    _plot_error_curves(axes[1], [
        (px, np.abs(wet_cos(px) - prc), "wet::cos", "C0"),
        (px, np.abs(lut_cos(px) - prc), "lut::cos", "C2"),
        (px, np.abs(ti_cos_wrapped(px) - prc), "ti_arm::cos", "C1"),
    ], "cos absolute error [0, 2pi]", "|approx - ref|")

    _plot_ulp_hist(axes[2], [
        (wet_sin_ulps, "wet", "C0"),
        (lut_sin_ulps, "lut", "C2"),
        (ti_sin_ulps, "ti_arm", "C1"),
    ], "sin ULP distribution (full range)")

    _plot_ulp_hist(axes[3], [
        (wet_cos_ulps, "wet", "C0"),
        (lut_cos_ulps, "lut", "C2"),
        (ti_cos_ulps, "ti_arm", "C1"),
    ], "cos ULP distribution (full range)")

    plt.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(os.path.join(out_dir, "comparison_sincos.png"), dpi=150)
    print(f"Saved: {os.path.join(out_dir, 'comparison_sincos.png')}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 2. atan2 comparison
    # -----------------------------------------------------------------------
    A = 100000
    theta = np.linspace(-np.pi, np.pi, A)
    y_f32 = np.sin(theta).astype(np.float32)
    x_f32 = np.cos(theta).astype(np.float32)
    ref_a2 = np.arctan2(y_f32.astype(np.float64), x_f32.astype(np.float64))

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    _plot_error_curves(ax2, [
        (theta, np.abs(wet_atan2(y_f32, x_f32) - ref_a2), "wet::atan2", "C0"),
        (theta, np.abs(lut_atan2(y_f32, x_f32) - ref_a2), "lut::atan2", "C2"),
        (theta, np.abs(np.arctan2(y_f32, x_f32).astype(np.float64) - ref_a2),
         "np.arctan2f", "C3"),
    ], "atan2 absolute error, inputs = (sin t, cos t) in float32",
       "|approx - ref| (rad)")
    ax2.set_xlabel("angle t (rad)")
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, "comparison_atan2.png"), dpi=150)
    print(f"Saved: {os.path.join(out_dir, 'comparison_atan2.png')}")
    plt.close(fig2)

    # -----------------------------------------------------------------------
    # 3. Cody-Waite large-argument analysis (log-log)
    # -----------------------------------------------------------------------
    M = 3_000_000
    cx = np.linspace(0.1, 5000.0, M)
    cref = np.sin(cx)

    series = [
        ("std::sin",              np.abs(np.sin(cx.astype(np.float32)) - cref), "C2"),
        ("ti_arm (wrapped)",       np.abs(ti_sin_wrapped(cx) - cref),            "C1"),
        ("wet:: (single-step)",    np.abs(wet_sin(cx) - cref),                   "C0"),
        ("wet:: (Cody-Waite)",     np.abs(wet_sin_codywaite(cx) - cref),         "C3"),
        ("lut:: (512-entry)",      np.abs(lut_sin(cx) - cref),                   "C4"),
    ]

    xf32 = cx.astype(np.float32)
    input_floor = np.abs(np.cos(xf32.astype(np.float64)) * (xf32.astype(np.float64) - cx))

    fig3, ax3 = plt.subplots(figsize=(16, 9))
    for label, err, color in series:
        ex, ey = _envelope_log(cx, err)
        ax3.loglog(ex, ey, label=label, color=color, linewidth=1.4)

    fx, fy = _envelope_log(cx, input_floor)
    ax3.loglog(fx, fy, label="float32 input-quant floor (irreducible)",
               color="gray", ls="--", lw=1.3, alpha=0.8)
    ax3.axhline(8e-7, color="black", ls=":", lw=1, alpha=0.5,
                label="polynomial floor (~8e-7)")

    ax3.set_xlabel("|x| (rad)")
    ax3.set_ylabel("max abs error per bin")
    ax3.set_title("sin error vs argument magnitude: reduction strategies compared")
    ax3.grid(True, which="both", alpha=0.25)
    ax3.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig3.savefig(os.path.join(out_dir, "comparison_codywaite.png"), dpi=150)
    print(f"Saved: {os.path.join(out_dir, 'comparison_codywaite.png')}")
    plt.close(fig3)

    # Numeric summary at key magnitudes
    col_names = ["std", "ti(wrap)", "wet", "wet C-W", "lut", "input-flr"]
    print(f"\n  |x|  " + "".join(f"{n:>11}" for n in col_names))
    for c in [10, 100, 1000, 5000]:
        w = (cx > c * 0.98) & (cx < c * 1.02)
        row = [float(np.max(s[1][w])) for s in series] + [float(np.max(input_floor[w]))]
        print(f"  {c:>5}" + "".join(f"{v:>11.2e}" for v in row))


if __name__ == "__main__":
    main()
