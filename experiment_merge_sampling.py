"""
Merge fig2_sampling_rmse + figA2_sampling_cliff into a single 2x2 full-width figure.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from gsp_core import (
    load_power_grid_data, build_weighted_graph,
    laplacian_eigendecomposition
)
from experiment_sampling import (
    design_sampling_set, graph_sampling_reconstruction
)

OUTPUT_DIR = Path(__file__).parent / 'figures'

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'serif',
})

print("[1/3] Loading grid data...")
bus_topo, line_topo, trans_topo, _, sag_bus, _, _ = load_power_grid_data()
G, L_sp, bus_names, bus_idx = build_weighted_graph(
    bus_topo, line_topo, trans_topo, weight_mode='admittance')
L = L_sp.toarray()
eigvals, U = laplacian_eigendecomposition(L_sp, k=200)
n_eigs = min(200, len(eigvals))
eigvals = eigvals[:n_eigs]
U = U[:, :n_eigs]

# Build sag signals (event-based) -- VOLTAGE signal, matching paper's graph-signal definition (L168)
print("[2/3] Building sag signals...")
V_NOM = 1.0
sag_signals = {}
for evt_id in sorted(sag_bus['EVENT_ID'].unique()):
    evt = sag_bus[sag_bus['EVENT_ID'] == evt_id]
    x = np.ones(len(bus_names)) * V_NOM
    for _, row in evt.iterrows():
        if row['BUSNAME'] in bus_idx:
            x[bus_idx[row['BUSNAME']]] = row['VA']
    sag_signals[evt_id] = x

# ---- Panel (a,b): RMSE vs M (fig2) ----
print("[3/3] Generating merged figure...")
monitor_counts = [3, 5, 10, 15, 20, 30, 50]
bandwidths = [1, 3, 5, 10]
methods = ['random', 'max_degree']
n_trials = 20
test_evt = list(sag_signals.keys())[0]
true_signal = sag_signals[test_evt]

results = {method: {bw: {m: [] for m in monitor_counts} for bw in bandwidths}
           for method in methods}
for trial in range(n_trials):
    for M in monitor_counts:
        for method in methods:
            S = design_sampling_set(G, L, U, eigvals, M, method=method)
            y_S = true_signal[S]
            for bw in bandwidths:
                if bw > M or bw > n_eigs:
                    continue
                x_hat = graph_sampling_reconstruction(
                    L, U[:, :n_eigs], eigvals[:n_eigs], y_S, S, bw)
                rmse = np.sqrt(np.mean((x_hat - true_signal) ** 2))
                results[method][bw][M].append(rmse)

# ---- Panel (c,d): condition number + RMSE (figA2) ----
BWs = [2, 3, 5, 8]
Ms = [3, 4, 5, 6, 8, 10, 15, 20, 30, 40]
n_trials2 = 20
signal = true_signal

cond_results = np.full((len(BWs), len(Ms), n_trials2), np.nan)
rmse_results = np.full((len(BWs), len(Ms), n_trials2), np.nan)
for bi, BW in enumerate(BWs):
    for mi, M in enumerate(Ms):
        for trial in range(n_trials2):
            if BW > M:
                continue
            S = design_sampling_set(G, L, U, eigvals, M, method='random')
            M_S = np.zeros((M, len(bus_names)))
            for row, bus in enumerate(S):
                M_S[row, bus] = 1.0
            y_S = signal[S]
            Phi = M_S @ U[:, :BW]
            s_vals = np.linalg.svd(Phi, compute_uv=False)
            cond_results[bi, mi, trial] = (
                s_vals[0] / s_vals[-1] if s_vals[-1] > 0 else 1e10)
            x_hat = graph_sampling_reconstruction(
                L, U[:, :n_eigs], eigvals[:n_eigs], y_S, S, BW)
            rmse_results[bi, mi, trial] = np.sqrt(
                np.mean((x_hat - signal) ** 2))

cond_mean = np.nanmean(cond_results, axis=2)
rmse_mean = np.nanmean(rmse_results, axis=2)

# ---- Build 2x2 figure ----
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, 6))

# (a) random sampling RMSE
for bw in sorted(bandwidths):
    rmses = []
    for M in monitor_counts:
        vals = results['random'][bw][M]
        rmses.append(np.mean(vals) if vals else np.nan)
    axes[0, 0].plot(monitor_counts, rmses, 'o-', linewidth=1.2,
                    markersize=4, label=f'B={bw}', color=colors[bandwidths.index(bw)])
axes[0, 0].set_xlabel('Number of monitors (M)')
axes[0, 0].set_ylabel('RMSE (p.u.)')
axes[0, 0].set_title('(a) Random sampling')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# (b) max-degree sampling RMSE
for bw in sorted(bandwidths):
    rmses = []
    for M in monitor_counts:
        vals = results['max_degree'][bw][M]
        rmses.append(np.mean(vals) if vals else np.nan)
    axes[0, 1].plot(monitor_counts, rmses, 'o-', linewidth=1.2,
                    markersize=4, label=f'B={bw}', color=colors[bandwidths.index(bw)])
axes[0, 1].set_xlabel('Number of monitors (M)')
axes[0, 1].set_ylabel('RMSE (p.u.)')
axes[0, 1].set_title('(b) Max-degree sampling')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# (c) condition number
for bi, BW in enumerate(BWs):
    valid = [mi for mi, M in enumerate(Ms) if M >= BW]
    valid_m = [Ms[mi] for mi in valid]
    valid_cond = [cond_mean[bi, mi] for mi in valid]
    axes[1, 0].loglog(valid_m, valid_cond, 'o-', color=colors[bi],
                      linewidth=1.5, markersize=5, label=f'B={BW}')
axes[1, 0].axhline(y=10, color='red', linestyle=':', alpha=0.5,
                   linewidth=1, label='Ill-conditioning threshold')
axes[1, 0].set_xlabel('Number of monitors (M)')
axes[1, 0].set_ylabel(r'Condition number $\kappa(\Phi)$')
axes[1, 0].set_title('(c) Sampling matrix condition number')
axes[1, 0].legend(loc='upper left')
axes[1, 0].grid(True, alpha=0.3)

# (d) RMSE cliff
for bi, BW in enumerate(BWs):
    valid = [mi for mi, M in enumerate(Ms) if M >= BW]
    valid_m = [Ms[mi] for mi in valid]
    valid_rmse = [rmse_mean[bi, mi] for mi in valid]
    axes[1, 1].loglog(valid_m, valid_rmse, 's-', color=colors[bi],
                      linewidth=1.5, markersize=5, label=f'B={BW}')
axes[1, 1].set_xlabel('Number of monitors (M)')
axes[1, 1].set_ylabel('RMSE (p.u.)')
axes[1, 1].set_title('(d) Reconstruction error')
axes[1, 1].legend(loc='upper left')
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Sampling Reconstruction: Performance and Stability', fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_combined_sampling.pdf')
fig.savefig(OUTPUT_DIR / 'fig_combined_sampling.png')
print('  Saved fig_combined_sampling.pdf/png')
