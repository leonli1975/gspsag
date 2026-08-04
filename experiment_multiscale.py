"""
Multi-scale Benchmark: Compare GSP across networks at equal sampling ratios.
Uses PEGASE 1354-bus alongside Chengdu 1025-bus.
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
    laplacian_eigendecomposition, compute_gft,
    verify_bandlimitedness, graph_sampling_reconstruction,
    load_power_grid_data, build_weighted_graph,
)

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 16, 'axes.labelsize': 16, 'axes.titlesize': 16,
    'legend.fontsize': 14, 'figure.dpi': 150, 'savefig.dpi': 300,
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


def generate_sag_data_zbus(net, n_events=30):
    """Generate sag data using Zbus method."""
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

    # Generate events at diverse fault locations
    sag_signals = {}
    fault_buses = np.linspace(0, N-1, min(N, 10)).astype(int)
    event_id = 0

    for fault_bus in fault_buses:
        for fault_type in ['3ph', '2ph', '1ph']:
            for r_fault in [0.01]:
                if event_id >= n_events:
                    break
                z_f = r_fault + 1j * 0.0
                z_kk = Zbus[fault_bus, fault_bus]
                v_pre = 1.0

                if fault_type == '3ph':
                    z_total = z_kk + z_f
                elif fault_type == '2ph':
                    z_total = 2.0 * (z_kk + z_f)
                else:
                    z_total = 3.0 * (z_kk + z_f)

                voltages = np.ones(N)  # real array
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


def run_gsp_at_ratios(G, L, U, eigvals, sag_signals, N, label):
    """Run GSP sampling at fixed ratios of nodes."""
    n_eigs = min(N, 100)
    # Pick the best sag signal (most spatial variation)
    if sag_signals:
        best_idx = max(sag_signals.keys(), key=lambda k: np.std(sag_signals[k]))
        test_sig = sag_signals[best_idx]
    else:
        return None

    # Sampling ratios
    ratios = [0.01, 0.02, 0.03, 0.05, 0.10]
    M_values = [max(2, int(r * N)) for r in ratios]  # at least 2 monitors
    BW = 3

    all_dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    n_trials = 15

    gsp_rmses = []
    idw_rmses = []
    for M in M_values:
        gsp_trials = []
        idw_trials = []
        for _ in range(n_trials):
            S = np.random.choice(N, M, replace=False)
            S_set = set(S)

            # GSP
            y_S = test_sig[S]
            xh = graph_sampling_reconstruction(L, U[:, :n_eigs], eigvals[:n_eigs], y_S, S, min(BW, n_eigs))
            gsp_trials.append(np.sqrt(np.mean((xh - test_sig)**2)))

            # IDW
            x_idw = np.zeros(N)
            for i in range(N):
                if i in S_set:
                    x_idw[i] = test_sig[i]
                else:
                    distances = np.array([all_dist[i].get(s, 1e6) for s in S])
                    values = np.array([test_sig[s] for s in S])
                    w = 1.0 / (distances**2 + 1e-10)
                    w = w / w.sum()
                    x_idw[i] = np.dot(w, values)
            idw_trials.append(np.sqrt(np.mean((x_idw - test_sig)**2)))

        gsp_rmses.append(np.mean(gsp_trials))
        idw_rmses.append(np.mean(idw_trials))

    # Bandlimitedness
    cum_mean = None
    bw_95 = None
    if sag_signals:
        cum_all = []
        for sig in sag_signals.values():
            cum_e, _ = verify_bandlimitedness(sig, U[:, :n_eigs], eigvals[:n_eigs], top_k=20)
            cum_all.append(cum_e)
        cum_mean = np.mean(cum_all, axis=0)
        bw_95 = np.argmax(cum_mean >= 0.95) + 1

    return {
        'ratios': ratios, 'M_values': M_values,
        'gsp': gsp_rmses, 'idw': idw_rmses,
        'bw_95': bw_95, 'N': N, 'label': label
    }


def main():
    print("=" * 60)
    print("Multi-Scale GSP Benchmark (Equal Sampling Ratios)")
    print("=" * 60)

    results = {}

    # ---- Chengdu Grid ----
    print("\n[1/4] Chengdu 1025-Bus Grid...")
    bus_topo, line_topo, trans_topo, _, sag_bus, _, _ = load_power_grid_data()
    G_cd, L_cd, bus_names_cd, bus_idx_cd = build_weighted_graph(
        bus_topo, line_topo, trans_topo, weight_mode='admittance')
    largest_cc = max(nx.connected_components(G_cd), key=len)
    sorted_cc = sorted(largest_cc)
    old_to_new = {old: new for new, old in enumerate(sorted_cc)}
    G_cd = G_cd.subgraph(largest_cc).copy()
    G_cd = nx.relabel_nodes(G_cd, old_to_new)
    bus_names_cd = [bus_names_cd[old] for old in sorted_cc]
    bus_idx_cd = {name: i for i, name in enumerate(bus_names_cd)}
    W_cd = nx.adjacency_matrix(G_cd, weight='weight').astype(np.float64)
    D_cd = sp.diags(np.array(W_cd.sum(axis=1)).flatten())
    L_cd = D_cd - W_cd
    N_cd = L_cd.shape[0]
    eigvals_cd, U_cd = laplacian_eigendecomposition(L_cd, k=100)

    cd_signals = {}
    for evt_id in sag_bus['EVENT_ID'].unique():
        evt_data = sag_bus[sag_bus['EVENT_ID'] == evt_id]
        signal = np.zeros(N_cd)
        for _, row in evt_data.iterrows():
            if row['BUSNAME'] in bus_idx_cd:
                signal[bus_idx_cd[row['BUSNAME']]] = row['VA']
        if np.sum(signal > 0) > 0.3 * N_cd:
            cd_signals[evt_id] = signal
    res_cd = run_gsp_at_ratios(G_cd, L_cd, U_cd, eigvals_cd, cd_signals, N_cd, 'Chengdu 1025')
    if res_cd:
        results['Regional (1025 buses)'] = res_cd
    print(f"  Done. BW(95%)={res_cd['bw_95']}, Events={len(cd_signals)}")

    # ---- IEEE 33-Bus ----
#    print("\n[2/4] IEEE 33-Bus...")
#    net33 = pn.case33bw()
#    pp.create_ext_grid(net33, bus=0, vm_pu=1.0)
#    G33, L33, N33 = build_graph_from_pp_network(net33)
#    eigvals33, U33 = laplacian_eigendecomposition(L33, k=33)
#    sigs33 = generate_sag_data_zbus(net33, n_events=30)
#    res33 = run_gsp_at_ratios(G33, L33, U33, eigvals33, sigs33, N33, 'IEEE 33')
#    if res33:
#        results['IEEE 33-Bus'] = res33
#    print(f"  Done. BW(95%)={res33['bw_95']}, Events={len(sigs33)}")

    # ---- PEGASE 1354-Bus ----
    print("\n[3/4] PEGASE 1354-Bus (comparable to Chengdu)...")
    net1354 = pn.case1354pegase()
    pp.create_ext_grid(net1354, bus=0, vm_pu=1.0)
    G1354, L1354, N1354 = build_graph_from_pp_network(net1354)
    n_eigs = min(N1354, 100)
    eigvals1354, U1354 = laplacian_eigendecomposition(L1354, k=n_eigs)
    sigs1354 = generate_sag_data_zbus(net1354, n_events=30)
    if sigs1354:
        res1354 = run_gsp_at_ratios(G1354, L1354, U1354, eigvals1354, sigs1354, N1354, 'PEGASE 1354')
        if res1354:
            results['PEGASE 1354-Bus'] = res1354
        print(f"  Done. BW(95%)={res1354['bw_95']}, Events={len(sigs1354)}")
    else:
        print("  No valid sag events generated!")
        # Try with more events
        sigs1354 = generate_sag_data_zbus(net1354, n_events=100)
        res1354 = run_gsp_at_ratios(G1354, L1354, U1354, eigvals1354, sigs1354, N1354, 'PEGASE 1354')
        if res1354:
            results['PEGASE 1354-Bus'] = res1354
        print(f"  Retry done. Events={len(sigs1354)}")

    # ---- Generate Figures ----
    print(f"\n[4/5] Generating comparison figures ({len(results)} networks)...")

    # Figure 1: RMSE vs sampling ratio for each network
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]

    colors_gsp = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728']
    colors_idw = ['#ff7f0e', '#bcbd22', '#8c564b', '#e377c2']

    for ax, (name, res), cg, ci in zip(axes, results.items(), colors_gsp, colors_idw):
        ratios_pct = [r * 100 for r in res['ratios']]
        ax.plot(ratios_pct, res['gsp'], 'o-', color=cg, linewidth=1.5, markersize=5, label='GSP (BW=3)')
        ax.plot(ratios_pct, res['idw'], 's--', color=ci, linewidth=1.5, markersize=5, label='IDW')
        ax.set_xlabel('Monitors (% of buses)')
        ax.set_ylabel('RMSE (p.u.)')
        ax.set_title(f'{name}\n(N={res["N"]}, BW95={res["bw_95"]})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate M values
        for r, m in zip(ratios_pct, res['M_values']):
            ax.annotate(f'M={m}', xy=(r, res['gsp'][ratios_pct.index(r)]),
                        fontsize=6, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points', color=cg)

    fig.suptitle('GSP vs IDW at Equal Sampling Ratios Across Network Scales', fontsize=14)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figA6_multiscale_comparison.pdf', dpi=300)
        fig.savefig(OUTPUT_DIR / 'figA6_multiscale_comparison.png', dpi=300)
    print('  Saved figA6_multiscale_comparison')

    # Figure 2: GSP/IDW improvement ratio vs network size at fixed 3% sampling
    improvement_data = []
    for name, res in results.items():
        # Find closest to 3% sampling
        target_ratio = 0.03
        idx = np.argmin([abs(r - target_ratio) for r in res['ratios']])
        gsp_val = res['gsp'][idx]
        idw_val = res['idw'][idx]
        if idw_val > 1e-6:
            improvement = (idw_val - gsp_val) / idw_val * 100
        else:
            improvement = 0
        improvement_data.append({
            'name': name, 'N': res['N'], 'improvement': improvement,
            'M': res['M_values'][idx], 'ratio': res['ratios'][idx] * 100,
            'gsp': gsp_val, 'idw': idw_val,
        })

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [d['name'] for d in improvement_data]
    Ns = [d['N'] for d in improvement_data]
    imps = [d['improvement'] for d in improvement_data]
    bars = ax.bar(range(len(names)), imps, color=['#d62728' if v < 0 else '#2ca02c' for v in imps],
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f'{n}\n(N={ns})' for n, ns in zip(names, Ns)], fontsize=9)
    ax.set_ylabel('GSP Improvement over IDW (%)')
    ax.set_title('GSP Advantage vs Network Size (at ~3% monitoring)')

    for i, (d, bar) in enumerate(zip(improvement_data, bars)):
        color = 'white' if abs(d['improvement']) > 15 else 'black'
        va = 'bottom' if d['improvement'] >= 0 else 'top'
        offset = -5 if d['improvement'] >= 0 else 5
        ax.text(i, d['improvement'] + offset,
                f'{d["improvement"]:+.0f}%\nM={d["M"]}/{d["N"]}', ha='center', va=va, fontsize=8, color=color)

    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figA7_scale_advantage.pdf', dpi=300)
        fig.savefig(OUTPUT_DIR / 'figA7_scale_advantage.png', dpi=300)
    print('  Saved figA7_scale_advantage')

    # Summary
    print("\n" + "=" * 60)
    print("MULTI-SCALE BENCHMARK SUMMARY")
    print("=" * 60)
    for d in improvement_data:
        print(f"  {d['name']} (N={d['N']}): GSP={d['gsp']:.4f}, IDW={d['idw']:.4f}, "
              f"Advantage={d['improvement']:+.1f}% at {d['ratio']:.0f}% monitors (M={d['M']})")


if __name__ == '__main__':
    main()
