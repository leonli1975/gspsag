"""
Generate additional figures for paper enrichment:
- Fig A: GFT spectrum comparison by fault type (single-phase vs two-phase vs three-phase)
- Fig B: Sampling condition number cliff
- Fig C: Risk indicators (SARFI/VSINDEX) as graph signals
"""
import numpy as np
import networkx as nx
from scipy import sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from gsp_core import (
    load_power_grid_data, build_weighted_graph,
    laplacian_eigendecomposition, compute_gft,
    graph_sampling_reconstruction, design_sampling_set,
    verify_bandlimitedness,
)
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'serif',
})
SAVE_FIGS = True


def build_graph_and_eigendecomp():
    """Common setup: build graph, compute eigendecomposition."""
    bus_topo, line_topo, trans_topo, substation, sag_bus, sag_event, _ = load_power_grid_data()
    G, L, bus_names, bus_idx = build_weighted_graph(bus_topo, line_topo, trans_topo, weight_mode='admittance')

    largest_cc = max(nx.connected_components(G), key=len)
    sorted_cc = sorted(largest_cc)
    old_to_new = {old: new for new, old in enumerate(sorted_cc)}
    G_sub = G.subgraph(largest_cc).copy()
    G_sub = nx.relabel_nodes(G_sub, old_to_new)
    bus_names_sub = [bus_names[old] for old in sorted_cc]
    bus_idx_sub = {name: i for i, name in enumerate(bus_names_sub)}
    W_sub = nx.adjacency_matrix(G_sub, weight='weight').astype(np.float64)
    D_sub = sp.diags(np.array(W_sub.sum(axis=1)).flatten())
    L = D_sub - W_sub
    N = L.shape[0]

    eigvals, U = laplacian_eigendecomposition(L, k=min(N, 200))
    return L, U, eigvals, G_sub, bus_names_sub, bus_idx_sub, sag_bus, sag_event, N


def get_sag_signal(sag_bus, bus_idx, N, evt_id):
    """Extract a sag signal for a specific event."""
    evt_data = sag_bus[sag_bus['EVENT_ID'] == evt_id]
    signal = np.zeros(N)
    for _, row in evt_data.iterrows():
        if row['BUSNAME'] in bus_idx:
            signal[bus_idx[row['BUSNAME']]] = row['VA']
    return signal


# ============================================================
# Figure A1: GFT Spectrum Comparison by Fault Type
# ============================================================
def fig_fault_type_gft():
    print("Generating Figure A1: GFT Spectrum by Fault Type...")
    L, U, eigvals, G, bus_names, bus_idx, sag_bus, sag_event, N = build_graph_and_eigendecomp()

    # Group events by fault type
    type1_events = []
    type2_events = []
    for evt_id in sag_bus['EVENT_ID'].unique():
        evt_info = sag_event[sag_event['EVENT_ID'] == evt_id]
        if len(evt_info) == 0:
            continue
        ft = evt_info.iloc[0]['FAULT_TYPE']
        signal = get_sag_signal(sag_bus, bus_idx, N, evt_id)
        if np.sum(signal > 0) < 0.3 * N:
            continue
        if ft == 1:
            type1_events.append((evt_id, signal))
        else:
            type2_events.append((evt_id, signal))

    n_eigs = min(N, 200)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: GFT spectra side by side
    ax = axes[0]
    for evt_id, sig in type1_events[:2]:
        xh = np.abs(compute_gft(sig, U[:, :n_eigs]))
        ax.semilogy(np.arange(1, n_eigs+1), xh, linewidth=1.2,
                    label=f'Single-phase (Event {evt_id})', alpha=0.8)
    for evt_id, sig in type2_events[:3]:
        xh = np.abs(compute_gft(sig, U[:, :n_eigs]))
        ax.semilogy(np.arange(1, n_eigs+1), xh, linewidth=1.2,
                    label=f'Multi-phase (Event {evt_id})', linestyle='--', alpha=0.8)
    ax.set_xlabel('GFT coefficient index k')
    ax.set_ylabel('Magnitude |x̂(k)|')
    ax.set_title('GFT Spectra by Fault Type')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 30)

    # Panel 2: Cumulative energy comparison
    ax = axes[1]
    for evt_id, sig in type1_events:
        cum_e, _ = verify_bandlimitedness(sig, U[:, :n_eigs], eigvals[:n_eigs], top_k=20)
        ax.plot(np.arange(1, 21), cum_e, linewidth=1.5, label=f'1φ (Evt {evt_id})')
    for evt_id, sig in type2_events[:2]:
        cum_e, _ = verify_bandlimitedness(sig, U[:, :n_eigs], eigvals[:n_eigs], top_k=20)
        ax.plot(np.arange(1, 21), cum_e, '--', linewidth=1.5, label=f'2φ (Evt {evt_id})')
    ax.set_xlabel('Number of GFT coefficients k')
    ax.set_ylabel('Cumulative energy ratio')
    ax.set_title('Bandlimitedness by Fault Type')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.01)

    # Panel 3: High-frequency energy ratio
    ax = axes[2]
    categories = ['Single-phase', 'Multi-phase']
    hf_ratios_1p = []
    hf_ratios_2p = []
    for _, sig in type1_events:
        xh = compute_gft(sig, U[:, :n_eigs])
        e = np.abs(xh) ** 2
        hf_ratios_1p.append(np.sum(e[5:]) / np.sum(e[:5]) * 100)
    for _, sig in type2_events:
        xh = compute_gft(sig, U[:, :n_eigs])
        e = np.abs(xh) ** 2
        hf_ratios_2p.append(np.sum(e[5:]) / np.sum(e[:5]) * 100)

    positions = [1, 2]
    bp = ax.boxplot([hf_ratios_1p, hf_ratios_2p], positions=positions,
                     widths=0.4, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#2ca02c', '#d62728']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories)
    ax.set_ylabel('HF/LF energy ratio (%)')
    ax.set_title('Spatial Variation by Fault Type')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Fault Type Discrimination via Graph Fourier Spectrum', fontsize=14)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figA1_fault_type_gft.pdf', dpi=300)
        fig.savefig(OUTPUT_DIR / 'figA1_fault_type_gft.png', dpi=300)
    print('  Saved figA1_fault_type_gft')


# ============================================================
# Figure A2: Sampling Condition Number Cliff
# ============================================================
def fig_sampling_cliff():
    print("Generating Figure A2: Sampling Condition Number Cliff...")
    L, U, eigvals, G, bus_names, bus_idx, sag_bus, sag_event, N = build_graph_and_eigendecomp()

    n_eigs = min(N, 200)

    # Get a representative signal
    signal = None
    for evt_id in sag_bus['EVENT_ID'].unique():
        s = get_sag_signal(sag_bus, bus_idx, N, evt_id)
        if np.sum(s > 0) > 0.3 * N:
            signal = s
            break

    # Compute condition number for different M and BW
    Ms = [3, 5, 8, 10, 15, 20, 30, 50]
    BWs = [1, 2, 3, 5, 10, 20]
    n_trials = 20

    cond_results = np.zeros((len(BWs), len(Ms), n_trials))
    rmse_results = np.zeros((len(BWs), len(Ms), n_trials))

    for trial in range(n_trials):
        for mi, M in enumerate(Ms):
            S = np.random.choice(N, M, replace=False)
            M_S = np.zeros((M, N))
            for i, s in enumerate(S):
                M_S[i, s] = 1.0

            y_S = signal[S]
            for bi, BW in enumerate(BWs):
                if BW > M:
                    cond_results[bi, mi, trial] = np.nan
                    rmse_results[bi, mi, trial] = np.nan
                    continue
                Phi = M_S @ U[:, :BW]
                s_vals = np.linalg.svd(Phi, compute_uv=False)
                cond_results[bi, mi, trial] = s_vals[0] / s_vals[-1] if s_vals[-1] > 0 else 1e10

                x_hat = graph_sampling_reconstruction(
                    L, U[:, :n_eigs], eigvals[:n_eigs], y_S, S, BW)
                rmse_results[bi, mi, trial] = np.sqrt(np.mean((x_hat - signal) ** 2))

    cond_mean = np.nanmean(cond_results, axis=2)
    rmse_mean = np.nanmean(rmse_results, axis=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(BWs)))
    for bi, BW in enumerate(BWs):
        valid = [mi for mi, M in enumerate(Ms) if M >= BW]
        valid_m = [Ms[mi] for mi in valid]
        valid_cond = [cond_mean[bi, mi] for mi in valid]
        valid_rmse = [rmse_mean[bi, mi] for mi in valid]
        ax1.loglog(valid_m, valid_cond, 'o-', color=colors[bi], linewidth=1.5, markersize=5,
                   label=f'BW={BW}')
        ax1.set_xticks([3, 4, 6, 10, 20, 30, 40])
        ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax2.loglog(valid_m, valid_rmse, 's-', color=colors[bi], linewidth=1.5, markersize=5,
                   label=f'BW={BW}')
        ax2.set_xticks([3, 4, 6, 10, 20, 30, 40])
        ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax1.axhline(y=10, color='red', linestyle=':', alpha=0.5, linewidth=1, label='Ill-conditioning threshold')
    ax1.set_xlabel('Number of monitors M')
    ax1.set_ylabel('Condition number cond(Φ)')
    ax1.set_title('Sampling Matrix Condition Number')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Number of monitors M')
    ax2.set_ylabel('RMSE (p.u.)')
    ax2.set_title('Reconstruction Error')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('The Sampling Cliff: Why M >> BW is Required', fontsize=14)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figA2_sampling_cliff.pdf', dpi=300)
        fig.savefig(OUTPUT_DIR / 'figA2_sampling_cliff.png', dpi=300)
    print('  Saved figA2_sampling_cliff')


# ============================================================
# Figure A3: Risk Indicators as Graph Signals
# ============================================================
def fig_risk_indicators():
    print("Generating Figure A3: Risk Indicators as Graph Signals...")
    L, U, eigvals, G, bus_names, bus_idx, sag_bus, sag_event, N = build_graph_and_eigendecomp()

    n_eigs = min(N, 200)

    # Read SARFI and VSINDEX from the old platform
    risk_dir = Path('/home/lgc/rpvoltsag/data/暂降域及风险评估指标/风险评估指标')
    sarfi = pd.read_csv(risk_dir / 'F_SARFIC.csv', encoding='gbk')
    vsindex = pd.read_csv(risk_dir / 'F_VSINDEX.csv', encoding='gbk')

    # Map to graph nodes
    bus_to_idx = {}
    for _, row in sarfi.iterrows():
        name = row['BUSNAME']
        if isinstance(name, str) and name.endswith('.0'):
            name = name[:-2]
        if name in bus_idx:
            bus_to_idx[name] = bus_idx[name]

    # Build SARFI-SC signal
    sc_signal = np.full(N, np.nan)
    for _, row in sarfi.iterrows():
        name = row['BUSNAME']
        if isinstance(name, str) and name.endswith('.0'):
            name = name[:-2]
        if name in bus_idx:
            sc_signal[bus_idx[name]] = row['SC']

    valid = ~np.isnan(sc_signal)
    print(f'  SARFI coverage: {valid.sum()}/{N} buses')
    if valid.sum() > 100:
        # Fill missing with mean
        sc_filled = sc_signal.copy()
        sc_filled[~valid] = np.nanmean(sc_signal[valid])

        # GFT of risk signal
        xh_sc = np.abs(compute_gft(sc_filled, U[:, :n_eigs]))
        cum_e_sc, _ = verify_bandlimitedness(sc_filled, U[:, :n_eigs], eigvals[:n_eigs], top_k=50)

        # Get a sag event signal for comparison
        sag_sig = None
        for evt_id in sag_bus['EVENT_ID'].unique():
            s = get_sag_signal(sag_bus, bus_idx, N, evt_id)
            if np.sum(s > 0) > 0.3 * N:
                sag_sig = s
                break

        xh_sag = np.abs(compute_gft(sag_sig, U[:, :n_eigs]))
        cum_e_sag, _ = verify_bandlimitedness(sag_sig, U[:, :n_eigs], eigvals[:n_eigs], top_k=50)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        # Panel 1: SARFI-SC distribution on the graph
        ax = axes[0]
        degrees = np.array([G.degree(n) for n in range(N)])
        ax.scatter(degrees[valid], sc_signal[valid], alpha=0.4, s=10, c='blue', edgecolors='none')
        ax.set_xlabel('Node degree')
        ax.set_ylabel('SARFI-C (SC)')
        ax.set_title('SARFI-C vs Node Degree')
        ax.grid(True, alpha=0.3)

        # Panel 2: GFT spectrum comparison
        ax = axes[1]
        ax.semilogy(np.arange(1, 51), xh_sc[:50], 'b-', linewidth=1.5, label='SARFI-C (risk)')
        ax.semilogy(np.arange(1, 51), xh_sag[:50], 'r--', linewidth=1.5, label='Sag event (instant.)')
        ax.set_xlabel('GFT coefficient index k')
        ax.set_ylabel('Magnitude |x̂(k)|')
        ax.set_title('GFT: Risk vs. Instantaneous Signal')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 3: Bandlimitedness comparison
        ax = axes[2]
        ax.plot(np.arange(1, 51), cum_e_sc, 'b-', linewidth=1.5, label='SARFI-C (risk)')
        ax.plot(np.arange(1, 51), cum_e_sag, 'r--', linewidth=1.5, label='Sag event (instant.)')
        ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Number of GFT coefficients k')
        ax.set_ylabel('Cumulative energy ratio')
        ax.set_title('Bandlimitedness: Risk vs Instantaneous')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.8, 1.01)

        # Find bandwidth
        bw_sc = np.argmax(cum_e_sc >= 0.95) + 1
        bw_sag = np.argmax(cum_e_sag >= 0.95) + 1
        ax.text(0.5, 0.05, f'SARFI 95% BW={bw_sc}\nSag event 95% BW={bw_sag}',
                transform=ax.transAxes, fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle('Risk Indicators (SARFI) as Naturally Smooth Graph Signals', fontsize=14)
        fig.tight_layout()
        if SAVE_FIGS:
            fig.savefig(OUTPUT_DIR / 'figA3_risk_indicators.pdf', dpi=300)
            fig.savefig(OUTPUT_DIR / 'figA3_risk_indicators.png', dpi=300)
        print('  Saved figA3_risk_indicators')

    return valid.sum()


if __name__ == '__main__':
    fig_fault_type_gft()
    print()
    fig_sampling_cliff()
    print()
    coverage = fig_risk_indicators()
    print(f'\nAll figures generated. SARFI coverage: {coverage} buses.')
