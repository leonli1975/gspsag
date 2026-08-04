"""
Experiment 2: Graph Filtering for Voltage Sag Denoising (Direction 3)
Demonstrates graph low-pass filtering to remove measurement noise.
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
    laplacian_eigendecomposition, compute_gft, compute_igft,
    graph_lowpass_filter, graph_total_variation
)

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'serif',
})

SAVE_FIGS = True


def add_measurement_noise(signal, noise_level=0.02):
    """Add Gaussian measurement noise."""
    rng = np.random.RandomState(123)
    noise = noise_level * rng.randn(len(signal))
    return signal + noise


def run_experiment():
    print('=' * 60)
    print('Experiment 2: Graph Filtering for Voltage Sag Denoising')
    print('=' * 60)

    # Load and build graph
    print('\n[1/4] Loading data and building graph...')
    bus_topo, line_topo, trans_topo, substation, sag_bus, sag_event, _ = load_power_grid_data()
    G, L, bus_names, bus_idx = build_weighted_graph(
        bus_topo, line_topo, trans_topo, weight_mode='admittance')
    N = len(bus_names)

    # Use largest connected component
    n_cc = nx.number_connected_components(G)
    if n_cc > 1:
        largest_cc = max(nx.connected_components(G), key=len)
        sorted_cc = sorted(largest_cc)
        old_to_new = {old: new for new, old in enumerate(sorted_cc)}

        G_sub = G.subgraph(largest_cc).copy()
        G_sub = nx.relabel_nodes(G_sub, old_to_new)

        bus_names = [bus_names[old] for old in sorted_cc]
        bus_idx = {name: i for i, name in enumerate(bus_names)}

        W_sub = nx.adjacency_matrix(G_sub, weight='weight').astype(np.float64)
        degrees_sub = np.array(W_sub.sum(axis=1)).flatten()
        D_sub = sp.diags(degrees_sub)
        L = D_sub - W_sub
        G = G_sub
        N = len(bus_names)
    print(f'  Graph: {N} nodes, {G.number_of_edges()} edges')

    # Eigendecomposition
    print('[2/4] Computing eigendecomposition...')
    n_eigs = min(N, 200)
    eigenvalues, U = laplacian_eigendecomposition(L, k=n_eigs)
    print(f'  Computed {n_eigs} eigenpairs, λ1={eigenvalues[0]:.4f}, λ200={eigenvalues[-1]:.4f}')

    # Get a sag signal
    print('[3/4] Extracting sag signal...')
    signal_true = None
    for evt_id in sag_bus['EVENT_ID'].unique():
        evt_data = sag_bus[sag_bus['EVENT_ID'] == evt_id]
        signal = np.zeros(N)
        count = 0
        for _, row in evt_data.iterrows():
            if row['BUSNAME'] in bus_idx:
                signal[bus_idx[row['BUSNAME']]] = row['VA']
                count += 1
        if count > 0.3 * N:
            signal_true = signal
            break
    print(f'  Signal: {N} nodes, mean={signal_true.mean():.4f}, std={signal_true.std():.4f}')

    # Add noise and filter
    print('[4/4] Running filtering experiments...')
    noise_levels = [0.01, 0.02, 0.05, 0.10]
    cutoff_ratios = np.linspace(0.05, 0.50, 20)

    results = {}
    for nl in noise_levels:
        signal_noisy = add_measurement_noise(signal_true, noise_level=nl)
        snr_in = 10 * np.log10(np.var(signal_true) / np.var(signal_noisy - signal_true))
        rmse_noisy = np.sqrt(np.mean((signal_noisy - signal_true) ** 2))

        snr_outs = []
        rmses = []
        for ratio in cutoff_ratios:
            cutoff = max(1, int(ratio * n_eigs))
            signal_filtered = graph_lowpass_filter(U, signal_noisy, cutoff)
            rmse = np.sqrt(np.mean((signal_filtered - signal_true) ** 2))
            snr_out = 10 * np.log10(np.var(signal_true) / max(np.var(signal_filtered - signal_true), 1e-12))
            snr_outs.append(snr_out)
            rmses.append(rmse)

        results[nl] = {
            'snr_in': snr_in, 'rmse_noisy': rmse_noisy,
            'cutoff_ratios': cutoff_ratios, 'snr_outs': snr_outs, 'rmses': rmses
        }
    print('  Filtering complete.')

    # Figure 5: SNR improvement vs cutoff ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for (nl, res), color in zip(results.items(), colors):
        label = f'Noise {nl:.0%} (SNR_in={res["snr_in"]:.1f} dB)'
        ax1.plot(res['cutoff_ratios'] * 100, res['snr_outs'], '-', color=color, linewidth=1.5, label=label)
        ax1.axhline(y=res['snr_in'], color=color, linestyle=':', alpha=0.5)
    ax1.set_xlabel('Cutoff ratio (% of spectrum)')
    ax1.set_ylabel('Output SNR (dB)')
    ax1.set_title('SNR Improvement via Graph Low-Pass Filtering')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for (nl, res), color in zip(results.items(), colors):
        ax2.plot(res['cutoff_ratios'] * 100, res['rmses'], '-', color=color, linewidth=1.5,
                 label=f'Noise {nl:.0%}')
        ax2.axhline(y=res['rmse_noisy'], color=color, linestyle=':', alpha=0.5)
    ax2.set_xlabel('Cutoff ratio (% of spectrum)')
    ax2.set_ylabel('RMSE (p.u.)')
    ax2.set_title('Reconstruction Error vs Cutoff Ratio')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'fig5_filtering_snr.pdf', dpi=300)
        fig.savefig(OUTPUT_DIR / 'fig5_filtering_snr.png', dpi=300)
    print('  Saved fig5_filtering_snr')

    # Figure 6: GFT spectrum before/after filtering
    x_hat_true = compute_gft(signal_true, U)
    signal_noisy_demo = add_measurement_noise(signal_true, noise_level=0.05)
    x_hat_noisy = compute_gft(signal_noisy_demo, U)
    cutoff_demo = int(0.15 * n_eigs)
    signal_filtered_demo = graph_lowpass_filter(U, signal_noisy_demo, cutoff_demo)
    x_hat_filtered = compute_gft(signal_filtered_demo, U)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    titles = ['True GFT Spectrum', 'Noisy GFT Spectrum', 'Filtered GFT Spectrum']
    data = [x_hat_true, x_hat_noisy, x_hat_filtered]
    for ax, title, xh in zip(axes, titles, data):
        ax.stem(np.arange(len(xh)), np.abs(xh), linefmt='b-', markerfmt='bo', basefmt=' ')
        if 'Filtered' in title:
            ax.axvline(x=cutoff_demo, color='r', linestyle='--', linewidth=1.5, label=f'Cutoff (k={cutoff_demo})')
            ax.legend(fontsize=8)
        ax.set_xlabel('GFT coefficient index')
        ax.set_ylabel('Magnitude |x̂(k)|')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'fig6_gft_spectrum_filtering.pdf', dpi=300)
        fig.savefig(OUTPUT_DIR / 'fig6_gft_spectrum_filtering.png', dpi=300)
    print('  Saved fig6_gft_spectrum_filtering')

    # Figure 7: Graph total variation vs noise level
    fig, ax = plt.subplots(figsize=(7, 4.5))
    noise_range = np.linspace(0, 0.20, 21)
    tv_true = graph_total_variation(signal_true, L)
    tv_noisy = []
    tv_filtered = []
    for nl in noise_range:
        s_noisy = add_measurement_noise(signal_true, noise_level=nl)
        tv_noisy.append(graph_total_variation(s_noisy, L))
        s_filt = graph_lowpass_filter(U, s_noisy, int(0.15 * n_eigs))
        tv_filtered.append(graph_total_variation(s_filt, L))

    ax.plot(noise_range * 100, tv_noisy, 's-', color='red', linewidth=1.2, markersize=3, label='Noisy signal')
    ax.plot(noise_range * 100, tv_filtered, 'o-', color='blue', linewidth=1.2, markersize=3, label='Filtered signal')
    ax.axhline(y=tv_true, color='green', linestyle='--', linewidth=1.5, label=f'True signal (TV={tv_true:.3f})')
    ax.set_xlabel('Noise level (%)')
    ax.set_ylabel('Graph Total Variation (TV_G)')
    ax.set_title('Graph Total Variation vs Measurement Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'fig7_tv_vs_noise.pdf', dpi=300)
        fig.savefig(OUTPUT_DIR / 'fig7_tv_vs_noise.png', dpi=300)
    print('  Saved fig7_tv_vs_noise')

    # Summary
    print('\n' + '=' * 60)
    print('FILTERING RESULTS SUMMARY')
    print('=' * 60)
    for nl, res in results.items():
        best_idx = np.argmax(res['snr_outs'])
        best_ratio = res['cutoff_ratios'][best_idx]
        best_snr = res['snr_outs'][best_idx]
        improvement = best_snr - res['snr_in']
        print(f'  Noise {nl:.0%}: SNR_in={res["snr_in"]:.1f} dB, '
              f'Best SNR_out={best_snr:.1f} dB (cutoff={best_ratio:.0%}), '
              f'Improvement={improvement:.1f} dB')

    print(f'\n  All figures saved to: {OUTPUT_DIR}/')

    return results


if __name__ == '__main__':
    run_experiment()
