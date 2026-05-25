# Single-Phase SOGI-PLL Small-Signal Analysis

**Abstract**
This report evaluates a single-phase SOGI-PLL linearized around lock using continuous-time and sampled-data frequency-domain analysis. For the nominal design (f0 = 50.00 Hz, k = 1.414, Kp = 10.000, Ki = 100.000), the loop achieves phase margin 60.02 @ 14.19 and gain margin 13.58 @ 48.86. Delay sensitivity and sampling-rate trends are included to support implementation decisions.

**Index Terms**: phase-locked loop, SOGI, sampled-data control, frequency response, stability margins.

## I. Linearized Loop Model

The analyzed signal path is input voltage -> SOGI quadrature extractor -> phase detector -> PI loop filter -> phase generator (NCO / frequency-to-phase integrator) with phase feedback.

<svg viewBox="0 0 960 240" width="100%" height="240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="SOGI-PLL linearized block diagram">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#1d3557"/>
    </marker>
  </defs>
  <rect x="30" y="88" width="120" height="52" rx="4" fill="#ffffff" stroke="#7a8691" stroke-width="1.2"/>
  <text x="90" y="118" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13">Input v_in</text>
  <rect x="190" y="88" width="150" height="52" rx="4" fill="#ffffff" stroke="#7a8691" stroke-width="1.2"/>
  <text x="265" y="111" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12.5">SOGI</text>
  <text x="265" y="127" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12">H_q(s)</text>
  <rect x="380" y="88" width="158" height="52" rx="4" fill="#ffffff" stroke="#7a8691" stroke-width="1.2"/>
  <text x="459" y="111" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12.5">Phase Detector</text>
  <text x="459" y="127" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12">e = v_in q</text>
  <rect x="575" y="88" width="130" height="52" rx="4" fill="#ffffff" stroke="#7a8691" stroke-width="1.2"/>
  <text x="640" y="111" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12.5">PI Filter</text>
  <text x="640" y="127" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12">F_PI(s)</text>
  <rect x="740" y="88" width="165" height="52" rx="4" fill="#ffffff" stroke="#7a8691" stroke-width="1.2"/>
  <text x="822" y="111" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12">Phase Generator (NCO)</text>
  <text x="822" y="127" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12">G_theta(s) = 2pi/s</text>
  <line x1="150" y1="114" x2="190" y2="114" stroke="#1d3557" stroke-width="1.6" marker-end="url(#arr)"/>
  <line x1="340" y1="114" x2="380" y2="114" stroke="#1d3557" stroke-width="1.6" marker-end="url(#arr)"/>
  <line x1="538" y1="114" x2="575" y2="114" stroke="#1d3557" stroke-width="1.6" marker-end="url(#arr)"/>
  <line x1="705" y1="114" x2="740" y2="114" stroke="#1d3557" stroke-width="1.6" marker-end="url(#arr)"/>
  <path d="M822 140 L822 198 L265 198 L265 140" fill="none" stroke="#1d3557" stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#arr)"/>
  <text x="545" y="190" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="11.5" fill="#2b2d42">Estimated phase feedback</text>
</svg>

The loop is modeled by

$$H_q(s)=\frac{k\omega_0^2}{s^2+k\omega_0 s+\omega_0^2},\quad F_{PI}(s)=K_p+\frac{K_i}{s},\quad G_{\theta}(s)=\frac{2\pi}{s}$$

$$L(s)=H_q(s)F_{PI}(s)G_{\theta}(s),\quad S(s)=\frac{1}{1+L(s)},\quad T(s)=\frac{L(s)}{1+L(s)}$$

with nominal parameters $f_0=50.00$ Hz, $\omega_0=314.159$ rad/s, $k=1.414$, $K_p=10.000$, and $K_i=100.000$.

## II. Analysis Method

Continuous-time frequency response is computed on $j\omega$, and sampled-data trends are computed from Tustin-discretized blocks evaluated on $z=e^{j\omega T_s}$. Primary metrics are phase margin, gain margin, closed-loop bandwidth, minimum Nyquist distance $\min|1+L|$, and peak sensitivity $M_s$.

## III. Results

### A. Continuous-Time Frequency-Domain Metrics

| Metric | Value |
|:---|:---|
| Phase margin [deg] @ gain crossover [Hz] | 60.025 @ 14.185 |
| Gain margin [dB] @ phase crossover [Hz] | 13.579 @ 48.862 |
| Closed-loop -3 dB bandwidth [Hz] | 27.884 |
| Min Nyquist distance min\|1 + L(jw)\| | 0.677991 |
| Peak sensitivity Ms [dB] | 3.376 |

### B. Interactive Plots

![Open-Loop Bode (Magnitude + Phase, Shared X Axis)](plots/bode_shared.svg)

![Closed-Loop T and Sensitivity S](plots/closed_loop_ts.svg)

![Nyquist L(jw)](plots/nyquist.svg)

The SVG plots are written to [analysis/pll/plots](plots).

### C. Sampled-Data Trends Versus Sampling Rate

Discrete models are obtained with Tustin discretization per block and evaluated on the unit circle z = exp(j*w*Ts). Frequency entries are reported in Hz.

| Fs [Hz] | PM [deg] @ fc [Hz] | GM [dB] @ f180 [Hz] | BW [Hz] | min\|1+L\| | Ms [dB] |
|---:|:---|:---|---:|---:|---:|
| 5000 | 60.02 @ 14.18 | 13.58 @ 48.85 | 27.88 | 0.6780 | 3.38 |
| 10000 | 60.02 @ 14.19 | 13.58 @ 48.86 | 27.88 | 0.6780 | 3.38 |
| 20000 | 60.02 @ 14.19 | 13.58 @ 48.86 | 27.88 | 0.6780 | 3.38 |
| 40000 | 60.02 @ 14.19 | 13.58 @ 48.86 | 27.88 | 0.6780 | 3.38 |

### D. Phase-Delay Sensitivity

Delay-induced phase loss at gain crossover is approximated by Delta_phi = -360 * f_c * Td.

Continuous-time crossover used for this estimate: f_c = 14.1851 Hz, PM = 60.025 deg.

| Delay Td [us] | Added phase lag [deg] | Estimated PM [deg] | Fraction of PM consumed |
|---:|---:|---:|---:|
| 25.0 | 0.13 | 59.90 | 0.002 |
| 50.0 | 0.26 | 59.77 | 0.004 |
| 100.0 | 0.51 | 59.51 | 0.009 |
| 200.0 | 1.02 | 59.00 | 0.017 |
| 400.0 | 2.04 | 57.98 | 0.034 |

## IV. Discussion

The nominal loop is well damped with practical robustness margins and low sensitivity peaking. Across the tested sampling rates, margin and bandwidth drift is negligible, indicating that Tustin discretization at these rates preserves the intended continuous-time behavior. Delay sensitivity remains modest at the evaluated crossover; nevertheless, implementation-specific delays should remain part of final validation.

## V. Conclusion

For the evaluated operating point, the SOGI-PLL design provides stable lock dynamics with robust margins and consistent sampled-data performance. This supports deployment using the present gain set, subject to confirmation with final hardware delay and quantization effects.

## References

1. M. Ciobotaru, R. Teodorescu, and F. Blaabjerg, "A New Software PLL Structure Based on Second Order Generalized Integrator," PESC 2006. DOI: [10.1109/PESC.2006.1711988](https://doi.org/10.1109/PESC.2006.1711988)
2. P. Rodriguez, A. Luna, R. Teodorescu, I. Candela, and F. Blaabjerg, "A Stationary Reference Frame Grid Synchronization System for Three-Phase Grid-Connected Power Converters Under Adverse Grid Conditions," IEEE Transactions on Power Electronics, 2012. DOI: [10.1109/TPEL.2011.2159242](https://doi.org/10.1109/TPEL.2011.2159242)
3. L. R. Limongi, R. Bojoi, C. Pica, F. Profumo, and A. Tenconi, "Analysis and Comparison of Phase Locked Loop Techniques for Grid Utility Applications," PESC 2007. DOI: [10.1109/PCCON.2007.373038](https://doi.org/10.1109/PCCON.2007.373038)
4. S. Golestan, J. M. Guerrero, and J. C. Vasquez, "Single-Phase PLLs: A Review of Recent Advances," IEEE Transactions on Power Electronics, 2017. DOI: [10.1109/TPEL.2017.2653861](https://doi.org/10.1109/TPEL.2017.2653861)

## Data and Code Availability

All figures and tables are generated from source in `analysis/pll` within this repository.
