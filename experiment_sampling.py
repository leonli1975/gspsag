"""
Experiment 1: Graph Sampling for Voltage Sag State Estimation (Direction 1)
Validates GSP sampling theory for recovering full-network sag from sparse monitors.
"""
import numpy as np
import networkx as nx
from scipy import sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from gsp_core import (
    load_power_grid_data, build_weighted_graph,
    laplacian_eigendecomposition, compute_gft,
    verify_bandlimitedness, graph_sampling_reconstruction,
    design_sampling_set
)

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 12, 'axes.titlesize': 16,
    'legend.fontsize': 12, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'serif',
})

SAVE_FIGS = True


def run_experiment():
    print('=' * 60)
    print('Experiment 1: GSP Sampling for Voltage Sag State Estimation')
    print('=' * 60)

    # 1. Load data
    print('\n[1/7] Loading power grid data...')
    bus_topo, line_topo, trans_topo, substation, sag_bus, sag_event, sag_index = load_power_grid_data()
    print(f'  Buses: {len(bus_topo)}, Lines: {len(line_topo)}, '
          f'Transformers: {len(trans_topo)}, Sag events: {len(sag_event)}')

    # 2. Build graph (only connected buses, includes transformer connections)
    print('\n[2/7] Building weighted graph (admittance-based, incl. transformer links)...')
    G, L, bus_names, bus_idx = build_weighted_graph(
        bus_topo, line_topo, trans_topo, weight_mode='admittance')
    N = len(bus_names)
    n_cc = nx.number_connected_components(G)
    print(f'  Graph: {N} nodes, {G.number_of_edges()} edges (incl. {G.number_of_edges() - len(line_topo)} transformer links)')
    print(f'  Connected components: {n_cc}')

    if n_cc > 1:
        comp_sizes = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
        print(f'  Component sizes: {comp_sizes[:5]}...')
        # Use the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        sorted_cc = sorted(largest_cc)
        old_to_new = {old: new for new, old in enumerate(sorted_cc)}

        G_sub = G.subgraph(largest_cc).copy()
        G_sub = nx.relabel_nodes(G_sub, old_to_new)

        bus_names_sub = [bus_names[old] for old in sorted_cc]
        bus_idx_sub = {name: i for i, name in enumerate(bus_names_sub)}

        W_sub = nx.adjacency_matrix(G_sub, weight='weight').astype(np.float64)
        degrees_sub = np.array(W_sub.sum(axis=1)).flatten()
        D_sub = sp.diags(degrees_sub)
        L = D_sub - W_sub
        G = G_sub
        bus_names = bus_names_sub
        bus_idx = bus_idx_sub
        N = len(bus_names)
        print(f'  Using largest connected component: {N} nodes, {G.number_of_edges()} edges')

    # 3. Eigendecomposition
    print('\n[3/7] Computing Laplacian eigendecomposition...')
    n_eigs = min(N, 200)
    eigenvalues, eigenvectors = laplacian_eigendecomposition(L, k=n_eigs)
    U = eigenvectors
    print(f'  Computed {n_eigs} eigenpairs')
    print(f'  Eigenvalue range: [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]')

    # 4. Verify bandlimitedness with real sag data
    print('\n[4/7] Verifying bandlimitedness of sag signals...')
    event_ids = sag_bus['EVENT_ID'].unique()
    print(f'  Total events: {len(event_ids)}')

    # Collect sag signals for buses in our connected component
    sag_signals = {}
    for evt_id in event_ids:
        evt_data = sag_bus[sag_bus['EVENT_ID'] == evt_id]
        signal = np.zeros(N)
        count = 0
        for _, row in evt_data.iterrows():
            bus_name = row['BUSNAME']
            if bus_name in bus_idx:
                signal[bus_idx[bus_name]] = row['VA']  # Phase A voltage
                count += 1
        if count > 0.3 * N:  # At least 30% coverage
            sag_signals[evt_id] = signal

    print(f'  Usable events (>=30% coverage): {len(sag_signals)}')

    # 5. Bandlimitedness analysis
    print('\n[5/7] Analyzing bandlimitedness...')
    cum_energy_curves = []
    for evt_id, signal in sag_signals.items():
        cum_energy, _ = verify_bandlimitedness(signal, U[:, :n_eigs], eigenvalues[:n_eigs], top_k=50)
        cum_energy_curves.append(cum_energy)

    cum_energy_mean = np.mean(cum_energy_curves, axis=0)
    cum_energy_std = np.std(cum_energy_curves, axis=0)

    # Find bandwidth where we recover 90% energy
    bandwidth_90 = np.argmax(cum_energy_mean >= 0.90) + 1
    bandwidth_95 = np.argmax(cum_energy_mean >= 0.95) + 1
    print(f'  Bandwidth for 90% energy: {bandwidth_90} ({bandwidth_90/N*100:.1f}%)')
    print(f'  Bandwidth for 95% energy: {bandwidth_95} ({bandwidth_95/N*100:.1f}%)')

    # Figure 1: Cumulative energy curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(1, len(cum_energy_mean) + 1)
    ax.plot(x, cum_energy_mean, 'b-', linewidth=1.5, label='Mean cumulative energy')
    ax.fill_between(x, cum_energy_mean - cum_energy_std, cum_energy_mean + cum_energy_std,
                     alpha=0.3, color='blue', label='±1 std')
    ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='90% energy')
    ax.axhline(y=0.95, color='red', linestyle=':', alpha=0.7, label='95% energy')
    ax.axvline(x=bandwidth_90, color='red', linestyle='--', alpha=0.4)
    ax.set_xlabel('Number of GFT coefficients (k)')
    ax.set_ylabel('Cumulative energy ratio')
    ax.set_title('Bandlimitedness of Voltage Sag Signals\n on the Power Grid Graph')
    ax.legend(loc='lower right')
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'fig1_bandlimitedness.pdf', dpi=300)
        fig.savefig(OUTPUT_DIR / 'fig1_bandlimitedness.png', dpi=300)
    print('  Saved fig1_bandlimitedness')

    # 6. Sampling and reconstruction experiment
    print('\n[6/7] Running sampling/reconstruction experiments...')
    monitor_counts = [3, 5, 10, 15, 20, 30, 50]
    bandwidths = [1, 3, 5, 10]
    methods = ['random', 'max_degree']
    n_trials = 20

    # Focus on one representative event for clarity
    if len(sag_signals) > 0:
        test_evt = list(sag_signals.keys())[0]
        true_signal = sag_signals[test_evt]

    results = {method: {bw: {m: [] for m in monitor_counts} for bw in bandwidths} for method in methods}

    for trial in range(n_trials):
        for M in monitor_counts:
            for method in methods:
                S = design_sampling_set(G, L, U, eigenvalues, M, method=method)
                y_S = true_signal[S]
                for bw in bandwidths:
                    if bw > M or bw > n_eigs:
                        continue
                    x_hat = graph_sampling_reconstruction(L, U[:, :n_eigs], eigenvalues[:n_eigs], y_S, S, bw)
                    rmse = np.sqrt(np.mean((x_hat - true_signal) ** 2))
                    results[method][bw][M].append(rmse)
        if (trial + 1) % 5 == 0:
            print(f'    Trial {trial + 1}/{n_trials}')

    # 7. Plot reconstruction results
    print('\n[7/7] Generating figures...')

    # Figure 2: RMSE vs number of monitors for different bandwidths
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        for bw in sorted(bandwidths):
            rmses = []
            for M in monitor_counts:
                vals = results[method][bw][M]
                if vals:
                    rmses.append(np.mean(vals))
                else:
                    rmses.append(np.nan)
            ax.plot(monitor_counts, rmses, 'o-', linewidth=1.2, markersize=4, label=f'Bandwidth={bw}')
        ax.set_xlabel('Number of monitors (M)')
        ax.set_ylabel('RMSE (p.u.)')
        ax.set_title(f'Sampling: {method.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle('GSP Sampling Reconstruction Error vs Monitor Count', fontsize=13)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'fig2_sampling_rmse.pdf', dpi=300)
        fig.savefig(OUTPUT_DIR / 'fig2_sampling_rmse.png', dpi=300)
    print('  Saved fig2_sampling_rmse')

    # Figure 3: Example reconstruction for a single event
    # Pick best method and parameters
    best_method = methods[0]
    best_bw_key = 5  # BW=5 gives good reconstruction
    best_M = 20  # M>BW for stable reconstruction

    S = design_sampling_set(G, L, U, eigenvalues, best_M, method=best_method)
    y_S = true_signal[S]
    x_hat = graph_sampling_reconstruction(L, U[:, :n_eigs], eigenvalues[:n_eigs], y_S, S, best_bw_key)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Scatter: true vs reconstructed
    ax = axes[0]
    ax.scatter(true_signal, x_hat, alpha=0.5, s=15, edgecolors='none')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect recovery')
    ax.set_xlabel('True voltage (p.u.)')
    ax.set_ylabel('Reconstructed voltage (p.u.)')
    ax.set_title(f'Reconstruction (M={best_M}, BW={best_bw_key})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Add RMSE text
    rmse_val = np.sqrt(np.mean((x_hat - true_signal) ** 2))
    ax.text(0.05, 0.95, f'RMSE = {rmse_val:.4f} p.u.', transform=ax.transAxes,
            verticalalignment='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # GFT coefficient magnitude
    ax = axes[1]
    x_hat_gft = compute_gft(true_signal, U[:, :n_eigs])
    ax.stem(np.arange(n_eigs), np.abs(x_hat_gft), linefmt='b-', markerfmt='bo', basefmt=' ')
    ax.axvline(x=best_bw_key, color='r', linestyle='--', linewidth=1, label=f'Bandwidth cutoff (k={best_bw_key})')
    ax.set_xlabel('GFT coefficient index')
    ax.set_ylabel('Magnitude |x̂(k)|')
    ax.set_title('Graph Fourier Transform of Sag Signal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'fig3_reconstruction_example.pdf', dpi=300)
        fig.savefig(OUTPUT_DIR / 'fig3_reconstruction_example.png', dpi=300)
    print('  Saved fig3_reconstruction_example')

    # Figure 4: Comparison with IDW interpolation baseline
    print('  Computing IDW baseline (precomputing all-pairs distances)...')
    # Precompute shortest path distances once
    all_pairs_dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    idw_by_M = {m: [] for m in monitor_counts}
    for trial in range(min(n_trials, 10)):
        for M in monitor_counts:
            S = np.random.choice(N, M, replace=False)
            y_S = true_signal[S]
            S_set = set(S)

            x_idw = np.zeros(N)
            for i in range(N):
                if i in S_set:
                    x_idw[i] = true_signal[i]
                else:
                    distances = []
                    values = []
                    for s in S:
                        d = all_pairs_dist[i].get(s, 1e6)
                        distances.append(d)
                        values.append(true_signal[s])
                    distances = np.array(distances)
                    values = np.array(values)
                    w = 1.0 / (distances ** 2 + 1e-10)
                    w = w / w.sum()
                    x_idw[i] = np.dot(w, values)
            rmse_idw = np.sqrt(np.mean((x_idw - true_signal) ** 2))
            idw_by_M[M].append(rmse_idw)
        if (trial + 1) % 3 == 0:
            print(f'    IDW trial {trial + 1}/10')

    idw_mean = [np.mean(idw_by_M[m]) for m in monitor_counts]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Use BW=5 as a good working point for comparison
    compare_bw = 5
    if compare_bw in bandwidths:
        compare_monitors = [m for m in monitor_counts if m >= compare_bw]
        gsp_rmses_plot = [np.mean(results['random'][compare_bw][m]) for m in compare_monitors]
        idw_subset = [idw_mean[monitor_counts.index(m)] for m in compare_monitors]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(compare_monitors, idw_subset, 's-', color='orange', linewidth=1.5, label='IDW Interpolation')
        ax.plot(compare_monitors, gsp_rmses_plot, 'o-', color='blue', linewidth=1.5,
                label=f'GSP Sampling (BW={compare_bw})')
        ax.set_xlabel('Number of monitors (M)')
        ax.set_ylabel('RMSE (p.u.)')
        ax.set_title('GSP Sampling vs. IDW Interpolation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if SAVE_FIGS:
            fig.savefig(OUTPUT_DIR / 'fig4_gsp_vs_idw.pdf', dpi=300)
            fig.savefig(OUTPUT_DIR / 'fig4_gsp_vs_idw.png', dpi=300)
        print('  Saved fig4_gsp_vs_idw')

    # Summary statistics
    print('\n' + '=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)
    print(f'  Network: {N} buses, {G.number_of_edges()} lines')
    print(f'  Sag events analyzed: {len(sag_signals)}')
    print(f'  Bandwidth for 90% energy: k = {bandwidth_90} ({bandwidth_90/N*100:.1f}%)')
    print(f'  Bandwidth for 95% energy: k = {bandwidth_95} ({bandwidth_95/N*100:.1f}%)')

    for method in methods:
        for bw in [1, 5]:
            rmse = np.mean(results[method][bw][30])
            print(f'  GSP ({method}, M=30, BW={bw}): RMSE = {rmse:.4f} p.u.')
        print()

    print(f'  IDW (M=30): RMSE = {idw_mean[monitor_counts.index(30)]:.4f} p.u.')
    print(f'\n  All figures saved to: {OUTPUT_DIR}/')

    return {
        'G': G, 'L': L, 'U': U, 'eigenvalues': eigenvalues,
        'sag_signals': sag_signals, 'bandwidth_90': bandwidth_90,
        'bandwidth_95': bandwidth_95, 'results': results,
        'idw_mean': idw_mean, 'bus_names': bus_names
    }


if __name__ == '__main__':
    run_experiment()
