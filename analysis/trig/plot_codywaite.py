"""
Error vs |x| over a large range: std vs TI (wrapped) vs wet:: (single-step
reduction, as-is) vs wet:: with Cody-Waite two-word-pi reduction.

Shows where Cody-Waite matters: the single-step reductions (wet as-is, TI
wrapped) lose ~1 ULP/rad, while Cody-Waite stays pinned at the polynomial floor.

    py -3 analysis/plot_codywaite.py
"""

import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis.trig.minimax_trig as m

# Shipped header coefficients for sin(pi*g) = g * P(g^2)
SIN_COEFFS = [3.141582250595093, -5.167152404785156,
              2.541962385177612, -0.5547642707824707]
SC = np.array(SIN_COEFFS, dtype=np.float32)
INV_PI = np.float32(1.0 / np.pi)


# --- Cody-Waite pi split: low mantissa bits zeroed so n*PI_HI is exact -------
def _split_pi():
    pi = np.float64(np.pi)
    bits = struct.unpack("<I", struct.pack("<f", np.float32(pi)))[0]
    hi = np.float32(struct.unpack("<f", struct.pack("<I", bits & 0xFFFFE000))[0])  # zero low 13 bits
    lo = np.float32(pi - np.float64(hi))
    lo2 = np.float32(pi - np.float64(hi) - np.float64(lo))
    return hi, lo, lo2


PI_HI, PI_LO, PI_LO2 = _split_pi()


def wet_sin_cw(x):
    """wet::sin with Cody-Waite reduction: r = x - n*pi in extended precision,
    then reuse the existing half-period polynomial via g = r/pi."""
    x = x.astype(np.float32)
    n = np.rint(x * INV_PI).astype(np.float32)
    r = (x - n * PI_HI).astype(np.float32)
    r = (r - n * PI_LO).astype(np.float32)
    r = (r - n * PI_LO2).astype(np.float32)
    g = (r * INV_PI).astype(np.float32)
    s = m._sin_poly_f32(g, SC)
    ni = n.astype(np.int64)
    return np.where(ni % 2 != 0, -s, s).astype(np.float64)


# --- vectorized TI sin on [0, 2pi] (float64 poly, matches minimax_trig.ti_sin) -
PI_C = m.PI_CONSTS
SN = m.SIN_CONSTS


def ti_sin_vec(a):
    a = a.astype(np.float64)
    ar = np.where(a > PI_C[0], PI_C[1] - a, a)
    ar = np.where(a > PI_C[2], a - PI_C[3], ar)
    x2 = ar * ar
    x4 = x2 * x2
    return ar * (SN[0] + SN[1] * x2 + SN[2] * x4 + SN[3] * x2 * x4)


def ti_sin_wrapped_vec(x):
    return ti_sin_vec(m.wrap_to_2pi_f32(x))


# --- odrv:: lookup table sin (512-entry, linear interpolation) ---------------
_ODRV_TABLE_SIZE = 512
# Regenerate the table using the same formula as math_utils.hpp:
#   sinTable_f32[i] = sin(2*pi*i / TABLE_SIZE)  for i in 0..512
_ODRV_SIN_TABLE = np.array(
    [np.sin(2.0 * np.pi * i / _ODRV_TABLE_SIZE) for i in range(_ODRV_TABLE_SIZE + 1)],
    dtype=np.float32,
)


def odrv_sin_vec(x):
    """odrv::fast_sin_f32 - 512-entry LUT with linear interpolation."""
    x = x.astype(np.float32)
    two_pi_f32 = np.float32(2.0 * np.pi)
    tsz_f32 = np.float32(_ODRV_TABLE_SIZE)

    # Scale to [0, 1) the same way the C++ does: divide then subtract floor
    in_ = (x / two_pi_f32).astype(np.float32)
    in_ = (in_ - np.floor(in_)).astype(np.float32)

    findex = (tsz_f32 * in_).astype(np.float32)
    index = findex.astype(np.int64)

    # Clamp edge case (index == 512 due to float rounding)
    wrap = index >= _ODRV_TABLE_SIZE
    index = np.where(wrap, np.int64(0), index)
    findex = np.where(wrap, findex - tsz_f32, findex).astype(np.float32)

    fract = (findex - index.astype(np.float32)).astype(np.float32)

    a = _ODRV_SIN_TABLE[index]
    b = _ODRV_SIN_TABLE[index + 1]
    return ((np.float32(1.0) - fract) * a + fract * b).astype(np.float64)


def envelope_lin(x, y, nbins=500):
    bins = np.linspace(x.min(), x.max(), nbins + 1)
    idx = np.clip(np.searchsorted(bins, x) - 1, 0, nbins - 1)
    cx, cy = [], []
    for i in range(nbins):
        mask = idx == i
        if mask.any():
            cx.append(0.5 * (bins[i] + bins[i + 1]))
            cy.append(y[mask].max())
    return np.array(cx), np.array(cy)


def envelope_log(x, y, nbins=350):
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), nbins + 1)
    idx = np.clip(np.searchsorted(bins, x) - 1, 0, nbins - 1)
    cx, cy = [], []
    for i in range(nbins):
        mask = idx == i
        if mask.any():
            cx.append(np.sqrt(bins[i] * bins[i + 1]))
            cy.append(y[mask].max())
    return np.array(cx), np.array(cy)


def main():
    N = 3_000_000
    x = np.linspace(0.1, 5000.0, N)
    xf = x.astype(np.float32)
    xfd = xf.astype(np.float64)
    # Algorithm error: reference is the exact sine of the *float32 input* the
    # function actually receives -- isolates reduction+poly from the unavoidable
    # float32 input quantization (which every method shares).
    ref = np.sin(x)

    series = [
        ("std::sinf",                 np.abs(np.sin(xf) - ref), "C2"),
        ("ti:: (single-step wrap)",   np.abs(ti_sin_wrapped_vec(x) - ref),         "C1"),
        # ("ti:: (no wrap)",            np.abs(ti_sin_vec(x) - ref),                 "C9"),
        ("wet:: as-is (single-step)", np.abs(m.wet_sin(x, SIN_COEFFS) - ref),      "C0"),
        ("wet:: Cody-Waite",          np.abs(wet_sin_cw(x) - ref),                 "C3"),
        ("odrv:: (512-entry LUT)",    np.abs(odrv_sin_vec(x) - ref),               "C4"),
    ]

    # Irreducible input-quantization floor: |cos(x)| * |x - float32(x)|, the
    # error from the input angle not being representable in float32 -- no
    # algorithm can beat this. Caps the real-world benefit of Cody-Waite.
    input_floor = np.abs(np.cos(xfd) * (xfd - x))

    fig, ax = plt.subplots(figsize=(16, 9))
    for label, err, color in series:
        ex, ey = envelope_log(x, err)
        ax.loglog(ex, ey, label=label, color=color, linewidth=1.4)

    fx, fy = envelope_log(x, input_floor)
    ax.loglog(fx, fy, label="float32 input-quant floor (irreducible)",
              color="gray", ls="--", lw=1.3, alpha=0.8)
    ax.axhline(8e-7, color="black", ls=":", lw=1, alpha=0.5,
               label="wet:: polynomial floor (~8e-7)")

    ax.set_xlabel("|x|  (radians)")
    ax.set_ylabel("max abs error per bin")
    ax.set_title("sin error vs argument magnitude — single-step reduction vs Cody-Waite vs odrv:: LUT")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    import os
    os.makedirs("analysis/build", exist_ok=True)
    out = "analysis/build/codywaite_error.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")

    # numeric summary at a few magnitudes
    print("\n  |x|      std        ti(wrap)   ti(no-wr)  wet as-is  wet C-W    odrv::LUT  input-floor")
    for c in [10, 100, 1000, 5000]:
        w = (x > c * 0.98) & (x < c * 1.02)
        row = [np.max(s[1][w]) for s in series] + [np.max(input_floor[w])]
        print(f"  {c:>5}   " + "  ".join(f"{v:.2e}" for v in row))

    # --- second plot: one full period [0, 2π] --------------------------------
    x2 = np.linspace(0.01, 2.0 * np.pi, N)
    xf2 = x2.astype(np.float32)
    xfd2 = xf2.astype(np.float64)
    ref2 = np.sin(xfd2)

    series2 = [
        ("std::sinf",                 np.abs(np.sin(xf2).astype(np.float64) - ref2), "C2"),
        # ("ti:: (single-step wrap)",   np.abs(ti_sin_wrapped_vec(x2) - ref2),         "C1"),
        ("ti:: (no wrap)",            np.abs(ti_sin_vec(x2) - ref2),                 "C9"),
        # ("wet:: as-is (single-step)", np.abs(m.wet_sin(x2, SIN_COEFFS) - ref2),      "C0"),
        ("wet:: Cody-Waite",          np.abs(wet_sin_cw(x2) - ref2),                 "C3"),
        ("odrv:: (512-entry LUT)",    np.abs(odrv_sin_vec(x2) - ref2),               "C4"),
    ]

    fig2, ax2 = plt.subplots(figsize=(16, 9))
    for label, err, color in series2:
        ex2, ey2 = envelope_lin(x2, err)
        ax2.semilogy(ex2, ey2, label=label, color=color, linewidth=1.4)

    ax2.axhline(8e-7, color="black", ls=":", lw=1, alpha=0.5,
                label="wet:: polynomial floor (~8e-7)")

    ax2.set_xlabel("|x|  (radians)")
    ax2.set_ylabel("max abs error per bin")
    ax2.set_title("sin error over one period [0, 2π]")
    ax2.set_xlim(0.01, 2.0 * np.pi)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="lower right", fontsize=9)

    out2 = "analysis/build/codywaite_error_oneperiod.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
