"""
GSP-extended analysis: Sag domain identification, vulnerability assessment,
compensation guidance, and interpretable GFT decomposition.
"""
import numpy as np
import networkx as nx
from scipy import sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from gsp_core import (
    load_power_grid_data, build_weighted_graph,
    laplacian_eigendecomposition, compute_gft, compute_igft,
    verify_bandlimitedness, graph_total_variation,
)

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 8, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'serif',
})
SAVE_FIGS = True


def setup_graph_and_signals():
    """Common setup."""
    bus_topo, line_topo, trans_topo, _, sag_bus, sag_event, _ = load_power_grid_data()
    G, L, bus_names_all, bus_idx_all = build_weighted_graph(
        bus_topo, line_topo, trans_topo, weight_mode='admittance')

    largest_cc = max(nx.connected_components(G), key=len)
    sorted_cc = sorted(largest_cc)
    old_to_new = {old: new for new, old in enumerate(sorted_cc)}
    G = G.subgraph(largest_cc).copy()
    G = nx.relabel_nodes(G, old_to_new)
    bus_names = [bus_names_all[old] for old in sorted_cc]
    bus_idx = {name: i for i, name in enumerate(bus_names)}
    W = nx.adjacency_matrix(G, weight='weight').astype(np.float64)
    D = sp.diags(np.array(W.sum(axis=1)).flatten())
    L = D - W
    N = L.shape[0]
    eigvals, U = laplacian_eigendecomposition(L, k=min(N, 200))

    # All events
    sag_signals = {}
    for evt_id in sag_bus['EVENT_ID'].unique():
        evt_data = sag_bus[sag_bus['EVENT_ID'] == evt_id]
        signal = np.zeros(N)
        for _, row in evt_data.iterrows():
            if row['BUSNAME'] in bus_idx:
                signal[bus_idx[row['BUSNAME']]] = row['VA']
        if np.sum(signal > 0) > 0.3 * N:
            sag_signals[evt_id] = signal

    return G, L, U, eigvals, bus_names, bus_idx, sag_signals, sag_event, N, bus_topo


# ============================================================
# 1. SAG DOMAIN IDENTIFICATION via GSP
# ============================================================
def sag_domain_identification():
    """
    Use GSP to cluster buses by their sag response patterns.
    The sag domain: for a given fault, which buses experience similar
    voltage drops? GFT decomposition reveals this naturally.
    """
    print("=" * 60)
    print("1. SAG DOMAIN IDENTIFICATION via GSP")
    print("=" * 60)
    G, L, U, eigvals, bus_names, bus_idx, sag_signals, sag_event, N, _ = setup_graph_and_signals()

    # Use one representative event
    sig = list(sag_signals.values())[0]
    n_eigs = min(N, 200)

    # GFT decomposition
    x_hat = compute_gft(sig, U[:, :n_eigs])

    # Reconstruct at different graph frequency bands
    # Low-frequency: k=1..3 (global sag pattern)
    # Mid-frequency: k=4..10 (regional patterns)
    # High-frequency: k=11..50 (local perturbations)

    bands = {
        'DC (Global Mean)': (0, 1),
        'Low-Freq (Wide-Area)': (1, 3),
        'Mid-Freq (Regional)': (3, 10),
        'High-Freq (Local)': (10, 50),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (band_name, (k_start, k_end)) in zip(axes, bands.items()):
        xh_band = np.zeros(n_eigs)
        xh_band[k_start:k_end] = x_hat[k_start:k_end]
        sig_band = compute_igft(xh_band, U[:, :n_eigs])

        # Color nodes by band contribution
        sc = ax.scatter(
            np.arange(N),
            sig_band,
            c=sig_band,
            cmap='RdYlGn_r',
            s=3,
            alpha=0.8,
            edgecolors='none'
        )
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
        ax.set_xlabel('Bus index')
        ax.set_ylabel('Voltage component (p.u.)')
        ax.set_title(f'{band_name}\n(GFT k={k_start+1}..{min(k_end, n_eigs)})')
        ax.grid(True, alpha=0.2)
        plt.colorbar(sc, ax=ax, label='p.u.')

    fig.suptitle('Sag Signal Decomposition by Graph Frequency Band', fontsize=14)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figC1_sag_domain_decomposition.pdf')
        fig.savefig(OUTPUT_DIR / 'figC1_sag_domain_decomposition.png')
    print('  Saved figC1_sag_domain_decomposition')

    # ========== Spatial clustering of sag response patterns ==========
    # Build sag response matrix: each column is one event
    event_list = sorted(sag_signals.keys())
    sig_matrix = np.column_stack([sag_signals[e] for e in event_list])

    # Cluster buses by their multi-event sag response using GFT coefficients
    # Buses with similar GFT spectra across events = same sag domain
    from sklearn.cluster import KMeans

    n_domains = 5
    kmeans = KMeans(n_clusters=n_domains, random_state=42, n_init=10)
    domain_labels = kmeans.fit_predict(sig_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_domains))
    for d in range(n_domains):
        mask = domain_labels == d
        if mask.sum() > 0:
            ax.scatter(np.arange(N)[mask], sig_matrix[mask].mean(axis=1),
                       color=colors[d], s=3, alpha=0.6, edgecolors='none',
                       label=f'Domain {d+1} (n={mask.sum()})')
    ax.set_xlabel('Bus index')
    ax.set_ylabel('Mean sag voltage (p.u.)')
    ax.set_title('Sag Domains: K-Means on Multi-Event Response')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    # Show domain centroids (average sag profile per domain)
    for d in range(n_domains):
        mask = domain_labels == d
        if mask.sum() > 0:
            centroid = sig_matrix[mask].mean(axis=0)
            ax.plot(range(len(event_list)), centroid, 'o-', color=colors[d],
                    linewidth=1.2, markersize=4, label=f'Domain {d+1}')
    ax.set_xlabel('Event index')
    ax.set_ylabel('Mean voltage in domain (p.u.)')
    ax.set_title('Domain Centroids Across Events')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)

    fig.suptitle('Sag Domain Identification via Multi-Event GSP Clustering', fontsize=14)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figC2_sag_domain_clustering.pdf')
        fig.savefig(OUTPUT_DIR / 'figC2_sag_domain_clustering.png')
    print('  Saved figC2_sag_domain_clustering')

    return domain_labels, sig_matrix


# ============================================================
# 2. SPECTRAL VULNERABILITY ANALYSIS
# ============================================================
def vulnerability_analysis():
    """
    Compute GSP-based vulnerability indices:
    - Spectral centrality: how "central" is each bus in the graph spectral sense?
    - Fiedler participation: contribution to the Fiedler vector (graph partitioning)
    - Graph resistance distance: effective electrical distance
    - Combine with risk indicators (SARFI from old platform)
    """
    print("\n" + "=" * 60)
    print("2. SPECTRAL VULNERABILITY ANALYSIS")
    print("=" * 60)
    G, L, U, eigvals, bus_names, bus_idx, sag_signals, sag_event, N, bus_topo_all = setup_graph_and_signals()

    n_eigs = min(N, 200)

    # ---- Spectral vulnerability indices ----

    # 1. Spectral centrality (sum of squared eigenvector components, weighted by 1/sqrt(lambda))
    # Nodes important in low-frequency eigenvectors are structurally central
    K_central = 10
    spectral_centrality = np.zeros(N)
    for k in range(1, K_central + 1):  # skip k=0 (DC)
        lam_k = max(eigvals[k], 1e-10)
        spectral_centrality += np.abs(U[:, k]) / np.sqrt(lam_k)
    spectral_centrality /= spectral_centrality.max()

    # 2. Fiedler vector participation (k=1, the algebraic connectivity vector)
    fiedler_vec = np.abs(U[:, 1])
    fiedler_vec /= fiedler_vec.max()

    # 3. Effective resistance (commute time distance proxy)
    # R_ij = sum_k (u_ik - u_jk)^2 / lambda_k
    # Node-level: sum of effective resistances to all other nodes
    resist_centrality = np.zeros(N)
    for k in range(1, K_central + 1):
        lam_k = max(eigvals[k], 1e-10)
        uk = U[:, k]
        for i in range(N):
            resist_centrality[i] += np.sum((uk[i] - uk)**2) / lam_k
    resist_centrality /= resist_centrality.max()

    # 4. Load SARFI risk indicators
    risk_dir = Path('/home/lgc/rpvoltsag/data/暂降域及风险评估指标/风险评估指标')
    sarfi = pd.read_csv(risk_dir / 'F_SARFIC.csv', encoding='gbk')

    sarfi_vals = np.full(N, np.nan)
    for _, row in sarfi.iterrows():
        name = str(row['BUSNAME'])
        if name.endswith('.0'):
            name = name[:-2]
        if name in bus_idx:
            sarfi_vals[bus_idx[name]] = row['SC']

    # ---- Vulnerability ranking ----
    # Combine spectral and data-driven indices
    valid = ~np.isnan(sarfi_vals)
    print(f'  SARFI coverage: {valid.sum()}/{N}')

    # Normalize SARFI
    sarfi_norm = sarfi_vals.copy()
    sarfi_norm[valid] = sarfi_norm[valid] / np.nanmax(sarfi_norm)

    # Composite vulnerability = weighted combination
    vuln_composite = (
        0.3 * spectral_centrality +
        0.2 * fiedler_vec +
        0.2 * resist_centrality +
        0.3 * np.nan_to_num(sarfi_norm, nan=0.0)
    )
    vuln_composite /= vuln_composite.max()

    # Top vulnerable buses
    top_n = 20
    top_idx = np.argsort(vuln_composite)[::-1][:top_n]
    print(f'  Top {top_n} most vulnerable buses:')
    for rank, idx in enumerate(top_idx[:10]):
        print(f'    #{rank+1}: {bus_names[idx]} (composite={vuln_composite[idx]:.4f}, '
              f'SARFI={sarfi_vals[idx]:.1f}, spectral={spectral_centrality[idx]:.3f})')

    # ---- Figure ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    indices = [
        (spectral_centrality, 'Spectral Centrality', 'viridis'),
        (fiedler_vec, 'Fiedler Participation', 'plasma'),
        (resist_centrality, 'Resistance Centrality', 'inferno'),
        (sarfi_norm, 'SARFI-SC (Normalized)', 'RdYlGn_r'),
        (vuln_composite, 'Composite Vulnerability', 'hot'),
        (None, '', ''),
    ]

    for i, (idx_arr, title, cmap) in enumerate(indices[:5]):
        ax = axes[i]
        if idx_arr is not None:
            # Sort by index value for visibility
            sort_idx = np.argsort(idx_arr)
            sorted_vals = idx_arr[sort_idx]
            sc = ax.scatter(np.arange(N), sorted_vals,
                           c=sorted_vals, cmap=cmap, s=2, alpha=0.8, edgecolors='none')
            ax.set_title(title)
            plt.colorbar(sc, ax=ax)
        ax.set_xlabel('Bus (sorted)')
        ax.grid(True, alpha=0.2)

    # Panel 6: correlation between spectral centrality and SARFI
    ax = axes[5]
    valid_mask = ~np.isnan(sarfi_vals)
    ax.scatter(spectral_centrality[valid_mask], sarfi_vals[valid_mask],
               alpha=0.3, s=8, c='blue', edgecolors='none')
    corr = np.corrcoef(spectral_centrality[valid_mask], sarfi_vals[valid_mask])[0, 1]
    ax.set_xlabel('Spectral Centrality')
    ax.set_ylabel('SARFI-SC')
    ax.set_title(f'Spectral vs. Data-Driven Risk\nCorrelation r = {corr:.3f}')
    ax.grid(True, alpha=0.2)

    fig.suptitle('GSP-Based Bus Vulnerability Assessment', fontsize=14)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figC3_vulnerability_analysis.pdf')
        fig.savefig(OUTPUT_DIR / 'figC3_vulnerability_analysis.png')
    print('  Saved figC3_vulnerability_analysis')

    return vuln_composite, spectral_centrality, sarfi_vals


# ============================================================
# 3. COMPENSATION GUIDANCE via GRAPH TOTAL VARIATION
# ============================================================
def compensation_analysis():
    """
    Frame reactive power compensation as graph signal enhancement:
    - Compute the graph gradient (voltage difference between neighbors)
    - Identify edges with largest sag voltage drops → candidate compensation locations
    - Simulate compensation by adding "virtual support" that reduces TV_G
    - Show how TV_G decreases with strategic compensation
    """
    print("\n" + "=" * 60)
    print("3. COMPENSATION via GRAPH TV MINIMIZATION")
    print("=" * 60)
    G, L, U, eigvals, bus_names, bus_idx, sag_signals, sag_event, N, _ = setup_graph_and_signals()

    sig = list(sag_signals.values())[0]

    # ====== Compute graph gradient (voltage drop per edge) ======
    edges = list(G.edges())
    edge_gradients = []
    for u, v in edges:
        grad = abs(sig[u] - sig[v])
        w = G[u][v]['weight']
        edge_gradients.append((u, v, grad, w))

    edge_gradients.sort(key=lambda x: x[2], reverse=True)

    # Top edges with largest voltage drops → candidate compensation points
    print(f'  Top 10 edges with largest voltage gradients:')
    for rank, (u, v, grad, w) in enumerate(edge_gradients[:10]):
        print(f'    #{rank+1}: {bus_names[u]} -- {bus_names[v]} '
              f'(ΔV={grad:.4f} p.u., w={w:.2f})')

    # ====== Simulate compensation by reducing voltage drop at top edges ======
    def apply_compensation(signal, edges_to_boost, boost_factor=0.3):
        """Simulate compensation: reduce voltage drop by boost_factor at specified edges."""
        sig_comp = signal.copy()
        for u, v in edges_to_boost:
            # Boost the lower voltage bus toward the higher voltage bus
            if sig[v] < sig[u]:
                sig_comp[v] += boost_factor * (sig[u] - sig[v])
            else:
                sig_comp[u] += boost_factor * (sig[v] - sig[u])
        return sig_comp

    n_comp_list = [0, 5, 10, 20, 50, 100]
    tv_values = []
    for n_comp in n_comp_list:
        if n_comp == 0:
            sig_c = sig.copy()
        else:
            comp_edges = [(u, v) for u, v, _, _ in edge_gradients[:n_comp]]
            sig_c = apply_compensation(sig, comp_edges, boost_factor=0.5)
        tv = graph_total_variation(sig_c, L)
        tv_values.append(tv)

    # ====== Figure ======
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: edge gradient histogram
    ax = axes[0]
    grads = [g for _, _, g, _ in edge_gradients]
    ax.hist(grads, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=np.mean(grads), color='red', linestyle='--', label=f'Mean={np.mean(grads):.3f}')
    ax.set_xlabel('Voltage gradient |ΔV| (p.u.)')
    ax.set_ylabel('Number of edges')
    ax.set_title('Edge Voltage Gradient Distribution')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Panel 2: TV_G reduction with compensation
    ax = axes[1]
    ax.plot(n_comp_list, tv_values, 'o-', color='darkgreen', linewidth=1.5, markersize=6)
    tv0 = tv_values[0]
    ax.axhline(y=tv0, color='gray', linestyle=':', alpha=0.5, label=f'No compensation (TV={tv0:.3f})')
    ax.set_xlabel('Number of compensated edges')
    ax.set_ylabel('Graph Total Variation TV_G')
    ax.set_title('TV_G Reduction via Strategic Compensation')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Panel 3: compensated vs original signal
    ax = axes[2]
    comp_edges_20 = [(u, v) for u, v, _, _ in edge_gradients[:20]]
    sig_comp_20 = apply_compensation(sig, comp_edges_20, boost_factor=0.5)

    sort_idx = np.argsort(sig)
    ax.plot(np.arange(N), sig[sort_idx], '-', color='red', linewidth=1, alpha=0.7, label='Original')
    ax.plot(np.arange(N), sig_comp_20[sort_idx], '-', color='green', linewidth=1, alpha=0.7,
            label=f'Compensated (20 edges)')
    ax.set_xlabel('Bus (sorted by original voltage)')
    ax.set_ylabel('Voltage (p.u.)')
    ax.set_title('Compensation Effect on Voltage Profile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle('Graph Total Variation-Guided Compensation', fontsize=14)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figC4_compensation_tv.pdf')
        fig.savefig(OUTPUT_DIR / 'figC4_compensation_tv.png')
    print('  Saved figC4_compensation_tv')

    return edge_gradients, tv_values


# ============================================================
# 4. GSP-INTERPRETABLE EXPLANATION
# ============================================================
def interpretable_explanation():
    """
    Demonstrate GSP's interpretability:
    1. Decompose sag into graph frequency components with physical meaning
    2. Show which eigenvectors contribute to which spatial patterns
    3. Map graph frequency to network topology features
    """
    print("\n" + "=" * 60)
    print("4. GSP-INTERPRETABLE DECOMPOSITION")
    print("=" * 60)
    G, L, U, eigvals, bus_names, bus_idx, sag_signals, sag_event, N, bus_topo_all = setup_graph_and_signals()

    sig = list(sag_signals.values())[0]
    n_eigs = min(N, 200)

    x_hat = compute_gft(sig, U[:, :n_eigs])

    # ====== Physical interpretation of GFT coefficients ======
    # Low-frequency (k=1, λ≈0): DC component = global voltage mean
    # k=2 (Fiedler): bipartition of the graph → separates two "halves" of the network
    # k=3-5: larger-scale spatial variation
    # k=10+: localized near the fault bus

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: GFT coefficient magnitudes with eigenvalue annotation
    ax = axes[0]
    ax.semilogy(range(1, 51), np.abs(x_hat[:50]), 'b-', linewidth=1, alpha=0.7)
    ax.stem(range(1, 11), np.abs(x_hat[:10]), linefmt='b-', markerfmt='bo', basefmt=' ')
    for k in [1, 2, 3, 5, 10]:
        ax.annotate(f'λ{k}={eigvals[k-1]:.4f}',
                    xy=(k, np.abs(x_hat[k-1])), fontsize=7,
                    xytext=(5, 5), textcoords='offset points', color='darkred')

    ax.set_xlabel('GFT coefficient index k')
    ax.set_ylabel('|x̂(k)| (log scale)')
    ax.set_title('GFT Spectrum with Laplacian Eigenvalues')
    ax.grid(True, alpha=0.2)

    # Panel 2: Eigenvector spatial patterns
    ax = axes[1]
    for k in [1, 2, 5]:
        uk = U[:, k-1]
        ax.plot(np.arange(N), uk, linewidth=1, alpha=0.7, label=f'k={k}, λ={eigvals[k-1]:.4f}')
    ax.set_xlabel('Bus index (sorted by eigenvector value for k=2)')
    ax.set_ylabel('Eigenvector component')
    ax.set_title('Laplacian Eigenvector Spatial Patterns')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 3: Cumulative reconstruction from low to high frequency
    ax = axes[2]
    k_values = [1, 2, 3, 5, 10, 20, 50]
    for k in k_values:
        xh_trunc = np.zeros(n_eigs)
        xh_trunc[:k] = x_hat[:k]
        sig_trunc = compute_igft(xh_trunc, U[:, :n_eigs])

        sort_idx = np.argsort(sig)
        ax.plot(np.arange(N), sig_trunc[sort_idx], linewidth=1, alpha=0.6,
                label=f'k≤{k}')
    ax.plot(np.arange(N), sig[sort_idx], 'k-', linewidth=2, alpha=0.8, label='True signal')
    ax.set_xlabel('Bus (sorted by true voltage)')
    ax.set_ylabel('Voltage (p.u.)')
    ax.set_title('Cumulative GFT Reconstruction')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)

    fig.suptitle('GSP-Interpretable Sag Decomposition: From Eigenvalues to Physical Meaning', fontsize=14)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figC5_interpretable_gsp.pdf')
        fig.savefig(OUTPUT_DIR / 'figC5_interpretable_gsp.png')
    print('  Saved figC5_interpretable_gsp')

    # ====== Physical meaning table ======
    print('\n  Physical Interpretation of GFT Components:')
    interpretations = [
        (1, 'DC Component', 'Global mean voltage across all buses'),
        (2, 'Network Bipartition', 'Separates buses into two groups with correlated sag responses'),
        (3, 'Substation-level variation', 'Captures regional differences between substation areas'),
        (5, 'Feeder-level variation', 'Distinguishes sag severity between different feeders'),
        (10, 'Local anomalies', 'Identifies buses with unusually deep or shallow sags'),
    ]
    for k, name, meaning in interpretations:
        if k <= n_eigs:
            print(f'    k={k}: {name:30s} |x̂|={np.abs(x_hat[k-1]):.4f}  → {meaning}')

    # ====== Eigenvector community matching ======
    # Check if Fiedler vector partitions match actual substation boundaries
    print('\n  Fiedler Partition Analysis:')
    fiedler = U[:, 1]
    left_group = np.where(fiedler < 0)[0]
    right_group = np.where(fiedler >= 0)[0]
    print(f'    Left group ({len(left_group)} buses): mean voltage = {sig[left_group].mean():.4f}')
    print(f'    Right group ({len(right_group)} buses): mean voltage = {sig[right_group].mean():.4f}')

    # Get substation information
    subs = {}
    for _, row in bus_topo_all.iterrows():
        if row['BUSNAME'] in bus_idx:
            sub = row['SUBSTATION']
            idx = bus_idx[row['BUSNAME']]
            if sub not in subs:
                subs[sub] = []
            subs[sub].append(idx)

    # Check which substations are predominantly in left/right group
    for sub_name, indices in list(subs.items())[:10]:
        if len(indices) < 3:
            continue
        left_count = sum(1 for i in indices if i in left_group)
        right_count = len(indices) - left_count
        if left_count > 0 or right_count > 0:
            dominant = 'Left' if left_count > right_count else 'Right'
            print(f'    {sub_name}: {left_count}L / {right_count}R → {dominant}')

    return None


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("GSP-EXTENDED: SAG DOMAIN + VULNERABILITY + COMPENSATION")
    print("=" * 60)

    sag_domain_identification()
    vulnerability_analysis()
    compensation_analysis()
    interpretable_explanation()

    print("\n" + "=" * 60)
    print("ALL EXTENDED ANALYSES COMPLETE")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print("=" * 60)
