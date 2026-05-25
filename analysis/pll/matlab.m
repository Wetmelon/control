% Minimal MATLAB version of analysis/pll/generate_pll_report.cpp
% Requires Control System Toolbox.

clear; clc;

f0 = 50;                % Hz
w0 = 2*pi*f0;           % rad/s
k  = sqrt(2);
Kp = 10;
Ki = 100;

s = tf('s');
Hq = (k*w0^2) / (s^2 + k*w0*s + w0^2);
PI = Kp + Ki/s;
Gth = (2*pi)/s;         % phase generator (frequency -> phase)
L = minreal(Hq*PI*Gth);

w = logspace(log10(0.1), log10(2e4), 1200);
[wcg, pm, wcp, gm_db, w180, bw_hz, min_nyq, ms_db] = loop_metrics(L, w);

fprintf('PLL Analysis Metrics (MATLAB)\n\n');
fprintf('Nominal: f0=%.2f Hz, w0=%.3f rad/s, k=%.3f, Kp=%.3f, Ki=%.3f\n\n', f0, w0, k, Kp, Ki);
fprintf('Continuous-time metrics\n');
fprintf('  PM [deg] @ fc [Hz]: %s\n', fmt_pair(pm, wcp));
fprintf('  GM [dB]  @ f180 [Hz]: %s\n', fmt_pair(gm_db, w180));
fprintf('  BW [Hz]: %.3f\n', bw_hz);
fprintf('  min|1+L|: %.6f\n', min_nyq);
fprintf('  Ms [dB]: %.3f\n\n', ms_db);

% Sampled-data trends (Tustin per block).
fs_list = [5000 10000 20000 40000];
fprintf('Sampled-data trends\n');
fprintf('  Fs[Hz]   PM@fc[Hz]           GM@f180[Hz]        BW[Hz]   min|1+L|   Ms[dB]\n');
for fs = fs_list
    Ts = 1/fs;
    Ld = minreal(c2d(Hq, Ts, 'tustin') * c2d(PI, Ts, 'tustin') * c2d(Gth, Ts, 'tustin'));
    wd = logspace(log10(0.1), log10(0.95*pi/Ts), 1000);
    [~, pm_d, wcp_d, gm_d, w180_d, bw_d, min_d, ms_d] = loop_metrics(Ld, wd);
    fprintf('  %-7.0f %-18s %-18s %7.2f  %8.4f  %6.2f\n', fs, fmt_pair(pm_d, wcp_d), fmt_pair(gm_d, w180_d), bw_d, min_d, ms_d);
end
fprintf('\n');

% Phase-delay sensitivity at continuous crossover.
delay_s = [25 50 100 200 400]*1e-6;
fprintf('Delay sensitivity\n');
fprintf('  Td[us]   Added lag[deg]   PM_est[deg]   PM fraction\n');
for td = delay_s
    lag_deg = wcp * td * 180/pi;
    pm_est  = pm - lag_deg;
    frac    = lag_deg / pm;
    fprintf('  %-7.1f %-15.2f %-12.2f %.3f\n', td*1e6, lag_deg, pm_est, frac);
end
fprintf('\n');

% PI gain sweep.
sc = [0.25 0.5 1 2 4];
fprintf('PI sweep (continuous)\n');
fprintf('  Kp      Ki      PM@fc[Hz]           GM@f180[Hz]        BW[Hz]   Ms[dB]   min|1+L|\n');
for kp_s = sc
    for ki_s = sc
        kp = Kp*kp_s;
        ki = Ki*ki_s;
        Ltry = minreal(Hq*(kp + ki/s)*Gth);
        [~, pm_t, wcp_t, gm_t, w180_t, bw_t, min_t, ms_t] = loop_metrics(Ltry, w);
        fprintf('  %-7.3f %-7.3f %-18s %-18s %7.2f  %6.2f  %8.4f\n', kp, ki, fmt_pair(pm_t, wcp_t), fmt_pair(gm_t, w180_t), bw_t, ms_t, min_t);
    end
end

% Plots (same three figures as C++ flow), using toolbox-native commands.
plot_dir = 'plots';
if ~exist(plot_dir, 'dir'), mkdir(plot_dir); end

S = feedback(1, L);
T = feedback(L, 1);

opts_hz = bodeoptions('cstprefs');
opts_hz.FreqUnits = 'Hz';
opts_hz.Grid = 'on';

f1 = figure('Color','w','Name','Open-Loop Bode (Shared Frequency Axis)');
bodeplot(L, w, opts_hz);
exportgraphics(f1, fullfile(plot_dir, 'bode_shared.svg'), 'ContentType', 'vector');

f2 = figure('Color','w','Name','Closed-Loop T and Sensitivity S');
bodemag(T, S, w, opts_hz);
legend('|T|','|S|','Location','best');
exportgraphics(f2, fullfile(plot_dir, 'closed_loop_ts.svg'), 'ContentType', 'vector');

f3 = figure('Color','w','Name','Nyquist L(jw)');
nyquist(L, w);
hold on; plot(-1, 0, 'ro', 'MarkerFaceColor', 'r'); hold off;
grid on;
exportgraphics(f3, fullfile(plot_dir, 'nyquist.svg'), 'ContentType', 'vector');

fprintf('\nWrote SVG plots to analysis/pll/plots\n');

function [wcg, pm, wcp, gm_db, w180, bw_hz, min_nyq, ms_db] = loop_metrics(L, w)
    [gm, pm, wcg, wcp] = margin(L);
    gm_db = 20*log10(gm);
    w180  = wcg;
    bw_hz = bandwidth(feedback(L,1))/(2*pi);

    respL = squeeze(freqresp(L, w));
    min_nyq = min(abs(1 + respL));

    S = feedback(1, L);
    respS = squeeze(freqresp(S, w));
    ms_db = max(20*log10(abs(respS)));
end

function s = fmt_pair(v1, w)
    if ~isfinite(v1) || ~isfinite(w) || w <= 0
        s = 'N/A';
    else
        s = sprintf('%.2f @ %.2f', v1, w/(2*pi));
    end
end
