# PLL Technical Report Generator

This folder computes SOGI-PLL analysis metrics and prints a paste-ready Markdown snippet to stdout (no Python dependency).

## Files

- `generate_pll_report.cpp`: compact analysis entry point and orchestration flow.
- `pll_report_support.hpp`: shared data structures and core analysis templates.
- `pll_report_text.cpp`: stdout Markdown section/table generation.
- `pll_report_plots.cpp`: Plotly HTML plus SVG artifact generation.
- `pll_report_compare.py`: Python comparison script using python-control (descriptive API + MATLAB wrapper aliases).
- `Makefile`: build and run targets.
- stdout from `build/pll_report.exe`: paste-ready metrics snippet for copy/paste into your hand-written report.

## Build and Generate

From this folder:

```bash
make report
```

Optional manual gain override:

```bash
./build/pll_report.exe <Kp> <Ki>
```

Python comparison run:

```bash
make pyreport
```

By default, the Python script also writes SVG plots to `analysis/pll/plots`:

- `bode_shared.svg`
- `closed_loop_ts.svg`
- `nyquist.svg`

Disable plot writing:

```bash
py -3 pll_report_compare.py --no-plots
```

Custom plot directory:

```bash
py -3 pll_report_compare.py --plots-dir ./plots_py
```

Optional Python gain override:

```bash
py -3 pll_report_compare.py <Kp> <Ki>
```

Output:

- stdout snippet containing continuous metrics, sampled-data trends, delay sensitivity, and PI gain sweep.
- Python script prints the same metric families for side-by-side comparison against C++/MATLAB outputs.
- Python plot export requires `matplotlib` (if missing, metrics still print and the script reports that plots were skipped).

## Scope of Computations

- Continuous-time open-loop and closed-loop frequency response
- Gain margin, phase margin, bandwidth, and Nyquist distance metrics
- Discrete-time margin trends versus sample rate
- Phase-delay sensitivity versus equivalent loop delay

## References

Use canonical links in Git history rather than storing paper PDFs in the repository.

1. M. Ciobotaru, R. Teodorescu, and F. Blaabjerg, "A New Software PLL Structure Based on Second Order Generalized Integrator," PESC 2006.
	Link: https://doi.org/10.1109/PESC.2006.1711988

2. P. Rodriguez, A. Luna, R. Teodorescu, I. Candela, and F. Blaabjerg, "A Stationary Reference Frame Grid Synchronization System for Three-Phase Grid-Connected Power Converters Under Adverse Grid Conditions," IEEE Transactions on Power Electronics, 2012.
	Link: https://doi.org/10.1109/TPEL.2011.2159242

3. L. R. Limongi, R. Bojoi, C. Pica, F. Profumo, and A. Tenconi, "Analysis and Comparison of Phase Locked Loop Techniques for Grid Utility Applications," PESC 2007.
	Link: https://doi.org/10.1109/PCCON.2007.373038

4. S. Golestan, J. M. Guerrero, and J. C. Vasquez, "Single-Phase PLLs: A Review of Recent Advances," IEEE Transactions on Power Electronics, 2017.
	Link: https://doi.org/10.1109/TPEL.2017.2653861
