"""
Independent PEGASE 1354 validation for GSP-SAG.

Generates synthetic sag events on the PEGASE 1354-bus system via the Zbus
method, then validates the paper's core claims independently:
  1. Bandlimitedness (B=5 captures >99% energy)
  2. GSP sampling reconstruction vs IDW (RMSE comparison)
  3. M >= 3B stability guideline (condition number analysis)

This provides a second, independent test network for the GSP-SAG framework.
"""
import numpy as np
import networkx as nx
from scipy import sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandapower.networks as pn
import pandapower as pp

sys.path.insert(0, str(Path(__file__).parent))
from gsp_core import (
    laplacian_eigendecomposition, verify_bandlimitedness,
    graph_sampling_reconstruction,
)

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'serif',
})
SAVE_FIGS = True


def build_graph_from_pp_network(net):
    """Build weighted graph from a pandapower network using real impedances."""
    N = len(net.bus)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    for _, line in net.line.iterrows():
        u = int(line['from_bus'])
        v = int(line['to_bus'])
        r = line['r_ohm_per_km'] * line['length_km']
        x = line['x_ohm_per_km'] * line['length_km']
        z = np.sqrt(r**2 + x**2)
        w = 1.0 / max(z, 1e-6)
        G.add_edge(u, v, weight=w)

    for _, trafo in net.trafo.iterrows():
        u = int(trafo['hv_bus'])
        v = int(trafo['lv_bus'])
        G.add_edge(u, v, weight=50.0)

    W = nx.adjacency_matrix(G, weight='weight').astype(np.float64)
    D = sp.diags(np.array(W.sum(axis=1)).flatten())
    L = D - W
    return G, L, N


def generate_sag_data_zbus(net, n_events=60):
    """Generate sag data using Zbus method with diverse fault parameters."""
    np.random.seed(42)
    N = len(net.bus)

    Ybus = np.zeros((N, N), dtype=complex)
    for _, line in net.line.iterrows():
        u = int(line['from_bus'])
        v = int(line['to_bus'])
        r = line['r_ohm_per_km'] * line['length_km']
        x = line['x_ohm_per_km'] * line['length_km']
        z = r + 1j * x
        if abs(z) < 1e-12:
            z = 1e-10
        y = 1.0 / z
        Ybus[u, v] -= y
        Ybus[v, u] -= y
        Ybus[u, u] += y
        Ybus[v, v] += y

    for _, trafo in net.trafo.iterrows():
        u = int(trafo['hv_bus'])
        v = int(trafo['lv_bus'])
        y = 1.0 / (0.001 + 1j * 0.05)
        Ybus[u, v] -= y
        Ybus[v, u] -= y
        Ybus[u, u] += y
        Ybus[v, v] += y

    Ybus[0, 0] += 1.0 / (0.01 + 1j * 0.1)
    Zbus = np.linalg.pinv(Ybus)

    sag_signals = {}
    # Diverse fault locations, types, and resistances
    n_fault_locs = 15
    fault_buses = np.linspace(0, N-1, n_fault_locs).astype(int)
    fault_types = ['3ph', '2ph', '1ph']
    r_faults = [0.01, 0.05, 0.1, 0.5]

    event_id = 0
    for fault_bus in fault_buses:
        for ft in fault_types:
            for rf in r_faults:
                if event_id >= n_events:
                    break
                z_f = rf + 1j * 0.0
                z_kk = Zbus[fault_bus, fault_bus]
                v_pre = 1.0

                if ft == '3ph':
                    z_total = z_kk + z_f
                elif ft == '2ph':
                    z_total = 2.0 * (z_kk + z_f)
                else:
                    z_total = 3.0 * (z_kk + z_f)

                voltages = np.ones(N)
                for i in range(N):
                    v_drop = (Zbus[i, fault_bus] / z_total) * v_pre
                    voltages[i] = abs(v_pre - v_drop)

                if np.min(voltages) < 0.99:
                    sag_signals[event_id] = voltages
                    event_id += 1
            if event_id >= n_events:
                break
        if event_id >= n_events:
            break

    return sag_signals


def main():
    print("=" * 60)
    print("Independent PEGASE 1354 Validation for GSP-SAG")
    print("=" * 60)

    # ---- 1. Load PEGASE network ----
    print("\n[1/5] Loading PEGASE 1354...")
    net1354 = pn.case1354pegase()
    pp.create_ext_grid(net1354, bus=0, vm_pu=1.0)
    G, L, N = build_graph_from_pp_network(net1354)
    print(f"  Buses: {N}, Edges: {G.number_of_edges()}, "
          f"Components: {nx.number_connected_components(G)}")

    # Largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    if len(largest_cc) < N:
        sorted_cc = sorted(largest_cc)
        old_to_new = {old: new for new, old in enumerate(sorted_cc)}
        G = G.subgraph(largest_cc).copy()
        G = nx.relabel_nodes(G, old_to_new)
        W = nx.adjacency_matrix(G, weight='weight').astype(np.float64)
        D = sp.diags(np.array(W.sum(axis=1)).flatten())
        L = D - W
        N = L.shape[0]
        print(f"  Using largest CC: {N} buses")

    n_eigs = min(N, 100)
    Ld = L.toarray()
    all_ev, U = np.linalg.eigh(Ld)
    eigvals = all_ev[:n_eigs]
    U = U[:, :n_eigs]
    print(f"  Eigendecomposition done (k={n_eigs})")

    # ---- 2. Generate independent sag events ----
    print("\n[2/5] Generating independent sag events (Zbus method)...")
    sigs = generate_sag_data_zbus(net1354, n_events=60)
    print(f"  Generated {len(sigs)} valid sag events")

    if len(sigs) < 10:
        print("  WARNING: too few events, retrying with more fault params")
        sigs = generate_sag_data_zbus(net1354, n_events=120)
        print(f"  Retry: {len(sigs)} events")

    # ---- 3. Bandlimitedness verification ----
    print("\n[3/5] Bandlimitedness verification...")
    cum_all = []
    for sig in sigs.values():
        cum_e, _ = verify_bandlimitedness(sig, U[:, :n_eigs], eigvals[:n_eigs], top_k=20)
        cum_all.append(cum_e)
    cum_mean = np.mean(cum_all, axis=0)
    cum_std = np.std(cum_all, axis=0)

    # Energy in first 5 coefficients
    e5 = cum_mean[4] if len(cum_mean) >= 5 else cum_mean[-1]
    e1 = cum_mean[0]
    bw_95 = np.argmax(cum_mean >= 0.95) + 1 if np.any(cum_mean >= 0.95) else n_eigs
    bw_99 = np.argmax(cum_mean >= 0.99) + 1 if np.any(cum_mean >= 0.99) else n_eigs
    print(f"  DC energy: {e1*100:.1f}%")
    print(f"  Energy in B=5: {e5*100:.1f}%")
    print(f"  BW(95%) = {bw_95}, BW(99%) = {bw_99}")

    # ---- 4. GSP sampling reconstruction vs IDW ----
    print("\n[4/5] GSP sampling reconstruction vs IDW...")

    # Precompute distance matrix (N x N) for IDW
    all_dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    dist_mat = np.full((N, N), 1e6)
    for i in range(N):
        di = all_dist[i]
        for j, d in di.items():
            dist_mat[i, j] = d
    del all_dist

    monitor_counts = [5, 10, 15, 20, 30, 50, 75]
    bandwidths = [3, 5]
    n_trials = 10
    sig_arr = np.array(list(sigs.values()))  # (n_sig, N)

    results = {bw: {'gsp': [], 'idw': [], 'M': []} for bw in bandwidths}

    U_F_cache = {bw: U[:, :bw] for bw in bandwidths}

    for M in monitor_counts:
        for bw in bandwidths:
            U_F = U_F_cache[bw]
            gsp_all = []
            idw_all = []
            for _ in range(n_trials):
                S = np.random.choice(N, M, replace=False)
                # Vectorized IDW weights: W_ij = 1/dist(i,s)^2 for all i, all s in S
                dS = dist_mat[:, S]              # (N, M)
                wS = 1.0 / (dS**2 + 1e-10)       # (N, M)
                wS = wS / wS.sum(axis=1, keepdims=True)  # (N, M)
                # IDW estimate for all signals at once: x_idw = W @ sig[:,S].T
                x_idw_all = wS @ sig_arr[:, S].T  # (N, n_sig)
                idw_rmse = np.sqrt(np.mean((x_idw_all - sig_arr.T)**2, axis=0))
                idw_all.extend(idw_rmse)

                # Vectorized GSP reconstruction for all signals
                Phi = U_F[S, :]                  # (M, bw)
                pinv_Phi = np.linalg.pinv(Phi)   # (bw, M)
                y_all = sig_arr[:, S].T          # (M, n_sig)
                alpha_all = pinv_Phi @ y_all     # (bw, n_sig)
                xh_all = U_F @ alpha_all         # (N, n_sig)
                gsp_rmse = np.sqrt(np.mean((xh_all - sig_arr.T)**2, axis=0))
                gsp_all.extend(gsp_rmse)

            results[bw]['gsp'].append(np.mean(gsp_all))
            results[bw]['idw'].append(np.mean(idw_all))
            results[bw]['M'].append(M)
            print(f"  M={M}, B={bw}: GSP={np.mean(gsp_all):.4f}, IDW={np.mean(idw_all):.4f}")

    # ---- 5. Condition number analysis (M >= 3B validation) ----
    print("\n[5/5] Condition number analysis...")
    BWs = [2, 3, 5, 8]
    Ms = [3, 5, 10, 15, 20, 30, 40]
    cond_mean = np.zeros((len(BWs), len(Ms)))

    for bi, BW in enumerate(BWs):
        for mi, M in enumerate(Ms):
            if BW > M:
                cond_mean[bi, mi] = np.nan
                continue
            conds = []
            for _ in range(n_trials):
                S = np.random.choice(N, M, replace=False)
                Phi = U[S, :BW]
                s_vals = np.linalg.svd(Phi, compute_uv=False)
                conds.append(s_vals[0] / s_vals[-1] if s_vals[-1] > 0 else 1e10)
            cond_mean[bi, mi] = np.mean(conds)

    print("  Condition numbers:")
    print(f"    {'M':>4}", end="")
    for M in Ms:
        print(f"{M:>7}", end="")
    print()
    for bi, BW in enumerate(BWs):
        print(f"    B={BW:<2}", end="")
        for mi, M in enumerate(Ms):
            v = cond_mean[bi, mi]
            print(f"{v:7.1f}" if not np.isnan(v) else f"{'--':>7}", end="")
        print()

    # ---- Figures ----
    print("\nGenerating figures...")

    # Figure 1: Bandlimitedness
    fig, ax = plt.subplots(figsize=(6, 4.5))
    k_range = np.arange(1, min(len(cum_mean), 20) + 1)
    ax.plot(k_range, cum_mean[:len(k_range)] * 100, 'o-', color='#1f77b4',
            linewidth=1.5, markersize=4, label='Mean')
    ax.fill_between(k_range, (cum_mean - cum_std)[:len(k_range)] * 100,
                    (cum_mean + cum_std)[:len(k_range)] * 100,
                    alpha=0.2, color='#1f77b4', label='±1 std')
    ax.axhline(y=95, color='red', linestyle=':', alpha=0.5)
    ax.axhline(y=99, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of GFT coefficients $k$')
    ax.set_ylabel('Cumulative energy (%)')
    ax.set_title(f'PEGASE 1354 Bandlimitedness (BW95={bw_95}, BW99={bw_99})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figP1_pegase_bandlimitedness.pdf')
        fig.savefig(OUTPUT_DIR / 'figP1_pegase_bandlimitedness.png')
    print('  Saved figP1_pegase_bandlimitedness')

    # Figure 2: GSP vs IDW
    fig, axes = plt.subplots(1, len(bandwidths), figsize=(11, 4.5))
    if len(bandwidths) == 1:
        axes = [axes]
    for ax, bw in zip(axes, bandwidths):
        ax.plot(results[bw]['M'], results[bw]['gsp'], 'o-', color='#1f77b4',
                linewidth=1.5, markersize=5, label='GSP')
        ax.plot(results[bw]['M'], results[bw]['idw'], 's--', color='#ff7f0e',
                linewidth=1.5, markersize=5, label='IDW')
        ax.axvline(x=3*bw, color='red', linestyle=':', alpha=0.5, label='M=3B')
        ax.set_xlabel('Number of monitors (M)')
        ax.set_ylabel('RMSE (p.u.)')
        ax.set_title(f'PEGASE 1354 (B={bw})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle('Independent PEGASE Validation: GSP vs IDW', fontsize=14)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figP2_pegase_gsp_vs_idw.pdf')
        fig.savefig(OUTPUT_DIR / 'figP2_pegase_gsp_vs_idw.png')
    print('  Saved figP2_pegase_gsp_vs_idw')

    # Figure 3: Condition number
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(BWs)))
    for bi, BW in enumerate(BWs):
        valid_m = [Ms[mi] for mi in range(len(Ms)) if M >= BW or M >= Ms[mi]]
        valid_m = [Ms[mi] for mi in range(len(Ms)) if not np.isnan(cond_mean[bi, mi])]
        valid_c = [cond_mean[bi, mi] for mi in range(len(Ms)) if not np.isnan(cond_mean[bi, mi])]
        ax.loglog(valid_m, valid_c, 'o-', color=colors[bi],
                  linewidth=1.5, markersize=5, label=f'B={BW}')
    ax.axhline(y=10, color='red', linestyle=':', alpha=0.5,
               label='Ill-conditioning')
    ax.set_xlabel('Number of monitors (M)')
    ax.set_ylabel(r'Condition number $\kappa(\Phi)$')
    ax.set_title('PEGASE 1354 Sampling Stability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figP3_pegase_condition.pdf')
        fig.savefig(OUTPUT_DIR / 'figP3_pegase_condition.png')
    print('  Saved figP3_pegase_condition')

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("PEGASE 1354 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Events: {len(sigs)}")
    print(f"  DC energy: {e1*100:.1f}%, Energy B=5: {e5*100:.1f}%")
    print(f"  BW(95%)={bw_95}, BW(99%)={bw_99}")
    print(f"\n  GSP vs IDW at M=30 (B=5):")
    i30 = results[5]['M'].index(30) if 30 in results[5]['M'] else -1
    if i30 >= 0:
        print(f"    GSP={results[5]['gsp'][i30]:.4f}, IDW={results[5]['idw'][i30]:.4f}, "
              f"Advantage={(results[5]['idw'][i30]-results[5]['gsp'][i30])/results[5]['idw'][i30]*100:+.1f}%")
    print(f"\n  M>=3B validation (condition number <= 10):")
    for bi, BW in enumerate(BWs):
        m3b = 3 * BW
        if m3b in Ms:
            mi = Ms.index(m3b)
            print(f"    B={BW}: M=3B={m3b}, cond={cond_mean[bi, mi]:.1f}")


if __name__ == '__main__':
    main()
