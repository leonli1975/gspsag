"""
Large-scale bandlimitedness validation on the full 4112-event BPA dataset
(data/暂降域数据11-4/), plus GSP applicability boundary analysis.

Generates:
  figQ1_bandlimitedness_4112.pdf — cumulative GFT energy for 4112 events,
    split by fault type, vs the 6-event Regional set used in the paper.
  figQ2_gsp_applicability.pdf — GSP vs IDW advantage vs sag severity
    (fraction of buses below 0.85 p.u.), revealing the applicability boundary.
"""
import numpy as np
import sys
from pathlib import Path
import networkx as nx
from scipy import sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from gsp_core import (
    load_power_grid_data, build_weighted_graph,
    laplacian_eigendecomposition,
)

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'serif',
})

DATA4112 = Path('/home/lgc/rpvoltsag/data/暂降域数据11-4/sag_signals_1028.npz')
FT_ORDER = ['单相接地', '两相短路', '两相短路接地', '三相短路']
FT_LABEL = ['Single-phase', 'Two-phase', 'Two-phase-ground', 'Three-phase']


def load_graph():
    bus_topo, line_topo, trans_topo, _, _, _, _ = load_power_grid_data()
    G, L_sp, bus_names, bus_idx = build_weighted_graph(
        bus_topo, line_topo, trans_topo, weight_mode='admittance')
    largest_cc = max(nx.connected_components(G), key=len)
    sorted_cc = sorted(largest_cc)
    old_to_new = {old: new for new, old in enumerate(sorted_cc)}
    G = G.subgraph(largest_cc).copy()
    G = nx.relabel_nodes(G, old_to_new)
    bus_names_cc = [bus_names[old] for old in sorted_cc]
    bus_idx_cc = {name: i for i, name in enumerate(bus_names_cc)}
    N = len(bus_names_cc)
    W = nx.adjacency_matrix(G, weight='weight').astype(np.float64)
    D = sp.diags(np.array(W.sum(axis=1)).flatten())
    L_cc = D - W
    eigvals, U = laplacian_eigendecomposition(L_cc, k=100)
    return G, L_cc, eigvals, U, bus_names, bus_idx_cc, N


def load_4112_signals(bus_names, bus_idx_cc, N):
    data = np.load(DATA4112)
    sigs = data['signals']
    ft = data['fault_types']
    sig_cc = np.zeros((sigs.shape[0], N))
    for old_i, name in enumerate(bus_names):
        if name in bus_idx_cc:
            sig_cc[:, bus_idx_cc[name]] = sigs[:, old_i]
    return sig_cc, ft


def load_6_signals(bus_names, bus_idx_cc, N):
    _, _, _, _, sag_bus, _, _ = load_power_grid_data()
    sigs = []
    for evt in sorted(sag_bus['EVENT_ID'].unique()):
        ed = sag_bus[sag_bus['EVENT_ID'] == evt]
        sig = np.ones(N)
        for _, r in ed.iterrows():
            if r['BUSNAME'] in bus_idx_cc:
                sig[bus_idx_cc[r['BUSNAME']]] = r['VA']
        sigs.append(sig)
    return np.array(sigs)


def cum_energy(U, sigs, n_eigs=100):
    """Cumulative GFT energy for a batch of signals."""
    out = []
    for s in sigs:
        xhat = U[:, :n_eigs].T @ s
        e = np.cumsum(xhat**2) / np.sum(xhat**2)
        out.append(e)
    return np.array(out)


def main():
    print("[1/4] Loading graph and eigenvectors...")
    G, L_cc, eigvals, U, bus_names, bus_idx_cc, N = load_graph()
    print(f"  N={N} buses")

    print("[2/4] Loading signals...")
    sig_cc, ft = load_4112_signals(bus_names, bus_idx_cc, N)
    sigs6 = load_6_signals(bus_names, bus_idx_cc, N)
    print(f"  4112 events: {sig_cc.shape}, 6 events: {sigs6.shape}")

    print("[3/4] Computing bandlimitedness...")
    n_eigs = 100
    cum_4112 = cum_energy(U, sig_cc, n_eigs)
    cum_6 = cum_energy(U, sigs6, n_eigs)

    cum_4112_mean = cum_4112.mean(axis=0)
    cum_4112_std = cum_4112.std(axis=0)
    cum_6_mean = cum_6.mean(axis=0)
    cum_6_std = cum_6.std(axis=0)

    # Per-fault-type
    cum_by_ft = {}
    for ft_name in FT_ORDER:
        mask = ft == ft_name
        cum_by_ft[ft_name] = cum_energy(U, sig_cc[mask], n_eigs).mean(axis=0)

    # Severity metric for applicability analysis
    volts = sig_cc
    sev = (volts < 0.85).mean(axis=1)  # fraction of buses below 0.85

    print("  Bandlimitedness (4112 events):")
    print(f"    DC={cum_4112_mean[0]*100:.2f}%, B=5={cum_4112_mean[4]*100:.2f}%")
    print("  Bandlimitedness (6 events):")
    print(f"    DC={cum_6_mean[0]*100:.2f}%, B=5={cum_6_mean[4]*100:.2f}%")

    print("[4/4] GSP vs IDW by severity (applicability boundary)...")
    all_dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    dist_mat = np.full((N, N), 1e6)
    for i in range(N):
        for j, d in all_dist[i].items():
            dist_mat[i, j] = d

    rng = np.random.RandomState(0)
    M, B = 30, 5
    n_samp = 300
    idx = rng.choice(len(sig_cc), n_samp, replace=False)
    sig_sample = sig_cc[idx]
    sev_sample = sev[idx]

    gsp_rmse, idw_rmse = [], []
    for t in range(5):
        S = rng.choice(N, M, replace=False)
        dS = dist_mat[:, S]
        wS = 1.0 / (dS**2 + 1e-10)
        wS = wS / wS.sum(axis=1, keepdims=True)
        idw_rmse.append(np.sqrt(np.mean((wS @ sig_sample[:, S].T - sig_sample.T)**2, axis=0)))
        U_F = U[:, :B]
        Phi = U_F[S, :]
        alpha = np.linalg.pinv(Phi) @ sig_sample[:, S].T
        gsp_rmse.append(np.sqrt(np.mean((U_F @ alpha - sig_sample.T)**2, axis=0)))
    gsp_rmse = np.mean(gsp_rmse, axis=0)
    idw_rmse = np.mean(idw_rmse, axis=0)
    advantage = (idw_rmse - gsp_rmse) / idw_rmse * 100

    print(f"  GSP wins {np.sum(advantage>0)}/{n_samp}, "
          f"mean advantage {advantage.mean():+.1f}%")

    # ---- Figure 1: Bandlimitedness ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    k = np.arange(1, 21)

    ax = axes[0]
    ax.plot(k, cum_6_mean[:20]*100, 'o-', color='#d62728', linewidth=1.8,
            markersize=4, label='6-event Regional set')
    ax.fill_between(k, (cum_6_mean[:20]-cum_6_std[:20])*100,
                    (cum_6_mean[:20]+cum_6_std[:20])*100,
                    color='#d62728', alpha=0.15)
    ax.plot(k, cum_4112_mean[:20]*100, 's-', color='#1f77b4', linewidth=1.8,
            markersize=4, label='4112-event full set')
    ax.fill_between(k, (cum_4112_mean[:20]-cum_4112_std[:20])*100,
                    (cum_4112_mean[:20]+cum_4112_std[:20])*100,
                    color='#1f77b4', alpha=0.15)
    ax.axhline(95, color='gray', ls=':', alpha=0.6)
    ax.axhline(99, color='gray', ls=':', alpha=0.6)
    ax.set_xlabel('Number of GFT coefficients $k$')
    ax.set_ylabel('Cumulative energy (%)')
    ax.set_title('(a) Cumulative GFT energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors_ft = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    for ft_name, ft_lab, c in zip(FT_ORDER, FT_LABEL, colors_ft):
        cft = cum_by_ft[ft_name]
        ax.plot(k, cft[:20]*100, '-', color=c, linewidth=1.5,
                label=ft_lab)
    ax.axhline(95, color='gray', ls=':', alpha=0.6)
    ax.set_xlabel('Number of GFT coefficients $k$')
    ax.set_ylabel('Cumulative energy (%)')
    ax.set_title('(b) By fault type (4112 events)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Bandlimitedness of Voltage Sag Signals on the Regional Grid',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figQ1_bandlimitedness_4112.pdf')
    fig.savefig(OUTPUT_DIR / 'figQ1_bandlimitedness_4112.png')
    print('  Saved figQ1_bandlimitedness_4112')

    # ---- Figure 2: Applicability boundary ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(sev_sample, advantage, c='#1f77b4', s=18, alpha=0.6,
                    edgecolors='none')
    # binned means
    bins = np.linspace(0, sev_sample.max(), 8)
    bin_centers = []
    bin_means = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (sev_sample >= b0) & (sev_sample < b1)
        if m.sum() >= 5:
            bin_centers.append((b0+b1)/2)
            bin_means.append(advantage[m].mean())
    ax.plot(bin_centers, bin_means, 'o-', color='#d62728', linewidth=2,
            markersize=6, label='Binned mean')
    ax.axhline(0, color='black', linewidth=0.8)
    # Mark the 6-event point
    sev6 = (sigs6 < 0.85).mean(axis=1)
    # use event0701 severity: <0.85 fraction
    frac6 = [(v < 0.85).mean() for v in sigs6]
    adv6 = []
    for v6 in sigs6:
        g6, i6 = [], []
        for t in range(5):
            S = rng.choice(N, M, replace=False)
            dS = dist_mat[:, S]
            wS = 1.0/(dS**2+1e-10); wS = wS/wS.sum(axis=1, keepdims=True)
            i6.append(np.sqrt(np.mean((wS@v6[S]-v6)**2)))
            UF = U[:, :B]; Phi = UF[S, :]
            alpha = np.linalg.pinv(Phi) @ v6[S]
            g6.append(np.sqrt(np.mean((UF@alpha - v6)**2)))
        adv6.append((np.mean(i6)-np.mean(g6))/np.mean(i6)*100)
    for fr, av in zip(frac6, adv6):
        ax.plot(fr, av, 'D', color='#2ca02c', markersize=9,
                markeredgecolor='white', zorder=5,
                label='6-event set' if av == adv6[0] else None)
    ax.set_xlabel('Sag extent (fraction of buses below 0.85 p.u.)')
    ax.set_ylabel('GSP advantage over IDW (%)')
    ax.set_title('GSP Applicability vs Sag Severity (B=5, M=30)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figQ2_gsp_applicability.pdf')
    fig.savefig(OUTPUT_DIR / 'figQ2_gsp_applicability.png')
    print('  Saved figQ2_gsp_applicability')

    print("\n=== SUMMARY ===")
    print(f"  Bandlimitedness: 4112 events DC={cum_4112_mean[0]*100:.1f}%, "
          f"B=5={cum_4112_mean[4]*100:.1f}%")
    print(f"  6 events DC={cum_6_mean[0]*100:.1f}%, B=5={cum_6_mean[4]*100:.1f}%")
    print(f"  GSP wins {np.sum(advantage>0)}/{n_samp} on 4112 set, "
          f"mean adv {advantage.mean():+.1f}%")
    print(f"  6-event advantage: {np.mean(adv6):+.1f}% "
          f"(per-event: {[f'{a:+.0f}' for a in adv6]})")


if __name__ == '__main__':
    main()
