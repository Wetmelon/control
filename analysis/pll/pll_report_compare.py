#!/usr/bin/env python3
"""PLL loop comparison script using python-control.

This script mirrors analysis/pll/generate_pll_report.cpp and matlab.m:
- continuous-time loop metrics
- sampled-data trends vs sample rate
- delay sensitivity estimate
- PI gain sweep

It intentionally uses both API styles from python-control:
- Descriptive/pythonic style via `control` (ct)
- MATLAB short-name style via `control.matlab` (ml): tf, c2d, ctrb, obsv
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import control as ct
    from control import matlab as ml
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: python-control. Install with: pip install control scipy numpy"
    ) from exc

TWO_PI = 2.0 * math.pi


@dataclass
class LoopMetrics:
    phase_margin: Optional[tuple[float, float]]
    gain_margin: Optional[tuple[float, float]]
    bandwidth_hz: Optional[float]
    min_nyquist_distance: float
    peak_sensitivity_db: float


def _pair_or_na(pair: Optional[tuple[float, float]], p1: int = 2, p2: int = 2) -> str:
    if pair is None:
        return "N/A"
    v1, omega = pair
    return f"{v1:.{p1}f} @ {omega / TWO_PI:.{p2}f}"


def _value_or_na_hz(value: Optional[float], p: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{p}f}"


def _finite_pair(value: float, omega: float) -> Optional[tuple[float, float]]:
    if np.isfinite(value) and np.isfinite(omega) and omega > 0.0:
        return (float(value), float(omega))
    return None


def _bandwidth_from_mag_db(omega: Iterable[float], mag_db: list[float]) -> Optional[float]:
    if not mag_db:
        return None

    omega_list = [float(w) for w in omega]
    if len(omega_list) != len(mag_db) or len(omega_list) < 2:
        return None

    threshold = mag_db[0] - 3.0
    for i in range(1, len(mag_db)):
        prev_db = mag_db[i - 1]
        curr_db = mag_db[i]
        if curr_db < threshold:
            denom = curr_db - prev_db
            if abs(denom) < 1e-15:
                omega_bw = omega_list[i]
            else:
                frac = (threshold - prev_db) / denom
                omega_bw = omega_list[i - 1] + frac * (omega_list[i] - omega_list[i - 1])
            return omega_bw / TWO_PI

    return None


def loop_metrics(loop_sys: ct.TransferFunction, omega: Iterable[float]) -> LoopMetrics:
    gm, pm, w_cg, w_cp = ct.margin(loop_sys)

    pm_pair = _finite_pair(pm, w_cp)

    gm_db = 20.0 * math.log10(gm) if np.isfinite(gm) and gm > 0.0 else float("nan")
    gm_pair = _finite_pair(gm_db, w_cg)

    s_sys = ct.feedback(1.0, loop_sys)
    t_sys = ct.feedback(loop_sys, 1.0)

    dt = None
    if getattr(loop_sys, "dt", None) is not None and loop_sys.dt not in (0, True):
        dt = float(loop_sys.dt)

    min_nyq = float("inf")
    peak_s_db = -float("inf")
    t_mag_db: list[float] = []
    for w in omega:
        freq = float(w)
        eval_point = np.exp(1j * freq * dt) if dt is not None else (1j * freq)

        Lw = complex(ct.evalfr(loop_sys, eval_point))
        Sw = complex(ct.evalfr(s_sys, eval_point))
        Tw = complex(ct.evalfr(t_sys, eval_point))

        min_nyq = min(min_nyq, abs(1.0 + Lw))

        s_mag = max(abs(Sw), 1e-30)
        s_db = 20.0 * math.log10(s_mag)
        peak_s_db = max(peak_s_db, s_db)

        t_mag = max(abs(Tw), 1e-30)
        t_mag_db.append(20.0 * math.log10(t_mag))

    bw_hz = _bandwidth_from_mag_db(omega, t_mag_db)

    return LoopMetrics(
        phase_margin=pm_pair,
        gain_margin=gm_pair,
        bandwidth_hz=bw_hz,
        min_nyquist_distance=min_nyq,
        peak_sensitivity_db=peak_s_db,
    )


def _eval_point_for_system(sys: ct.TransferFunction, omega: float) -> complex:
    dt = None
    if getattr(sys, "dt", None) is not None and sys.dt not in (0, True):
        dt = float(sys.dt)
    if dt is None:
        return 1j * omega
    return complex(np.exp(1j * omega * dt))


def sample_loop_response(loop_sys: ct.TransferFunction, omega: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_sys = ct.feedback(1.0, loop_sys)
    t_sys = ct.feedback(loop_sys, 1.0)

    l_vals: list[complex] = []
    s_vals: list[complex] = []
    t_vals: list[complex] = []
    for w in omega:
        eval_point = _eval_point_for_system(loop_sys, float(w))
        l_vals.append(complex(ct.evalfr(loop_sys, eval_point)))
        s_vals.append(complex(ct.evalfr(s_sys, eval_point)))
        t_vals.append(complex(ct.evalfr(t_sys, eval_point)))

    return np.asarray(l_vals), np.asarray(s_vals), np.asarray(t_vals)


def write_plots(loop_sys: ct.TransferFunction, omega: np.ndarray, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nPlot export skipped: matplotlib is not installed.")
        print("Install with: pip install matplotlib")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    freq_hz = omega / TWO_PI
    l_vals, s_vals, t_vals = sample_loop_response(loop_sys, omega)

    l_mag_db = 20.0 * np.log10(np.maximum(np.abs(l_vals), 1e-30))
    l_phase_deg = np.unwrap(np.angle(l_vals)) * 180.0 / math.pi
    s_mag_db = 20.0 * np.log10(np.maximum(np.abs(s_vals), 1e-30))
    t_mag_db = 20.0 * np.log10(np.maximum(np.abs(t_vals), 1e-30))

    fig1, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6), constrained_layout=True)
    axes[0].semilogx(freq_hz, l_mag_db, color="#1f77b4", linewidth=1.8)
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].grid(True, which="both", alpha=0.35)

    axes[1].semilogx(freq_hz, l_phase_deg, color="#d62728", linewidth=1.8)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (deg)")
    axes[1].grid(True, which="both", alpha=0.35)

    fig1.suptitle("Open-Loop Bode (Shared Frequency Axis)")
    fig1.savefig(out_dir / "bode_shared.svg", format="svg")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax2.semilogx(freq_hz, t_mag_db, label="|T|", color="#1f77b4", linewidth=1.8)
    ax2.semilogx(freq_hz, s_mag_db, label="|S|", color="#d62728", linewidth=1.8)
    ax2.set_title("Closed-Loop T and Sensitivity S")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.grid(True, which="both", alpha=0.35)
    ax2.legend(loc="best")
    fig2.savefig(out_dir / "closed_loop_ts.svg", format="svg")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax3.plot(np.real(l_vals), np.imag(l_vals), color="#1f77b4", linewidth=1.8)
    ax3.plot([-1.0], [0.0], "ro", markersize=5)
    ax3.set_title("Nyquist L(jw)")
    ax3.set_xlabel("Re{L(jw)}")
    ax3.set_ylabel("Im{L(jw)}")
    ax3.grid(True, alpha=0.35)
    ax3.axis("equal")
    fig3.savefig(out_dir / "nyquist.svg", format="svg")
    plt.close(fig3)

    print(f"\nWrote SVG plots to {out_dir}")


def build_nominal_loop(kp: float, ki: float, f0_hz: float = 50.0, k_sogi: float = math.sqrt(2.0)) -> ct.TransferFunction:
    w0 = TWO_PI * f0_hz

    # Descriptive API style (control namespace).
    hq = ct.tf([k_sogi * w0 * w0], [1.0, k_sogi * w0, w0 * w0])
    pi = ct.tf([kp, ki], [1.0, 0.0])
    g_theta = ct.tf([TWO_PI], [1.0, 0.0])

    return ct.minreal(hq * pi * g_theta, verbose=False)


def matlab_wrapper_smoke_check(kp: float, ki: float, f0_hz: float = 50.0, k_sogi: float = math.sqrt(2.0)) -> None:
    """Sanity-check MATLAB wrapper short names from control.matlab."""
    w0 = TWO_PI * f0_hz

    hq_m = ml.tf([k_sogi * w0 * w0], [1.0, k_sogi * w0, w0 * w0])
    pi_m = ml.tf([kp, ki], [1.0, 0.0])
    g_theta_m = ml.tf([TWO_PI], [1.0, 0.0])

    loop_m = hq_m * pi_m * g_theta_m
    _ = ml.c2d(loop_m, 1.0 / 10_000.0, method="tustin")

    ss_hq = ct.ss(hq_m)
    a, b, c, _ = ct.ssdata(ss_hq)
    _ = ml.ctrb(a, b)
    _ = ml.obsv(a, c)


def print_report(kp: float, ki: float, write_svg_plots: bool, plots_dir: Path) -> None:
    f0 = 50.0
    w0 = TWO_PI * f0
    k_sogi = math.sqrt(2.0)

    # Use both styles: main analysis uses descriptive control API,
    # and a smoke-check uses control.matlab short aliases.
    matlab_wrapper_smoke_check(kp, ki, f0_hz=f0, k_sogi=k_sogi)

    loop = build_nominal_loop(kp, ki, f0_hz=f0, k_sogi=k_sogi)
    omega = np.logspace(math.log10(0.1), math.log10(2.0e4), 1200)
    cont = loop_metrics(loop, omega)

    print("PLL Analysis Metrics (Python/control)\n")
    print(
        f"Nominal: f0={f0:.2f} Hz, w0={w0:.3f} rad/s, k={k_sogi:.3f}, "
        f"Kp={kp:.3f}, Ki={ki:.3f}\n"
    )

    print("Continuous-time metrics")
    print(f"  PM [deg] @ fc [Hz]: {_pair_or_na(cont.phase_margin, 3, 3)}")
    print(f"  GM [dB]  @ f180 [Hz]: {_pair_or_na(cont.gain_margin, 3, 3)}")
    print(f"  BW [Hz]: {_value_or_na_hz(cont.bandwidth_hz, 3)}")
    print(f"  min|1+L|: {cont.min_nyquist_distance:.6f}")
    print(f"  Ms [dB]: {cont.peak_sensitivity_db:.3f}\n")

    print("Sampled-data trends")
    print("  Fs[Hz]   PM@fc[Hz]           GM@f180[Hz]        BW[Hz]   min|1+L|   Ms[dB]")

    fs_list = [5000.0, 10_000.0, 20_000.0, 40_000.0]
    for fs in fs_list:
        ts = 1.0 / fs

        # MATLAB-wrapper short names for c2d, mirroring matlab.m behavior.
        w0_local = TWO_PI * f0
        hq_m = ml.tf([k_sogi * w0_local * w0_local], [1.0, k_sogi * w0_local, w0_local * w0_local])
        pi_m = ml.tf([kp, ki], [1.0, 0.0])
        gth_m = ml.tf([TWO_PI], [1.0, 0.0])

        ld = ct.minreal(
            ml.c2d(hq_m, ts, method="tustin")
            * ml.c2d(pi_m, ts, method="tustin")
            * ml.c2d(gth_m, ts, method="tustin"),
            verbose=False,
        )

        w_max = 0.95 * math.pi / ts
        wd = np.logspace(math.log10(0.1), math.log10(w_max), 1000)
        md = loop_metrics(ld, wd)

        print(
            f"  {fs:<7.0f}"
            f" {_pair_or_na(md.phase_margin, 2, 2):<18}"
            f" {_pair_or_na(md.gain_margin, 2, 2):<18}"
            f" {_value_or_na_hz(md.bandwidth_hz, 2):>7}"
            f"  {md.min_nyquist_distance:8.4f}"
            f"  {md.peak_sensitivity_db:6.2f}"
        )

    print("\nDelay sensitivity")
    print("  Td[us]   Added lag[deg]   PM_est[deg]   PM fraction")

    delay_s = [25e-6, 50e-6, 100e-6, 200e-6, 400e-6]
    if cont.phase_margin is None:
        print("  N/A: no gain crossover found for nominal loop.\n")
    else:
        pm_deg, w_cp = cont.phase_margin
        for td in delay_s:
            lag_deg = w_cp * td * 180.0 / math.pi
            pm_est = pm_deg - lag_deg
            frac = lag_deg / pm_deg if pm_deg > 0.0 else float("nan")
            print(f"  {td * 1e6:<7.1f} {lag_deg:<15.2f} {pm_est:<12.2f} {frac:.3f}")
        print()

    print("PI sweep (continuous)")
    print("  Kp      Ki      PM@fc[Hz]           GM@f180[Hz]        BW[Hz]   Ms[dB]   min|1+L|")

    gain_scale = [0.25, 0.5, 1.0, 2.0, 4.0]
    for kp_s in gain_scale:
        for ki_s in gain_scale:
            kp_try = kp * kp_s
            ki_try = ki * ki_s
            loop_try = build_nominal_loop(kp_try, ki_try, f0_hz=f0, k_sogi=k_sogi)
            m = loop_metrics(loop_try, omega)
            print(
                f"  {kp_try:<7.3f} {ki_try:<7.3f}"
                f" {_pair_or_na(m.phase_margin, 2, 2):<18}"
                f" {_pair_or_na(m.gain_margin, 2, 2):<18}"
                f" {_value_or_na_hz(m.bandwidth_hz, 2):>7}"
                f"  {m.peak_sensitivity_db:6.2f}"
                f"  {m.min_nyquist_distance:8.4f}"
            )

    if write_svg_plots:
        write_plots(loop, omega, plots_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLL report comparison using python-control")
    parser.add_argument("Kp", nargs="?", type=float, default=10.0, help="PI proportional gain")
    parser.add_argument("Ki", nargs="?", type=float, default=100.0, help="PI integral gain")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing SVG plots",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
        help="Directory where SVG plots are written",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_report(args.Kp, args.Ki, write_svg_plots=not args.no_plots, plots_dir=args.plots_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
