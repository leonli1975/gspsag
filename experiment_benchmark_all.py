"""
Benchmark comparison for ALL GSP analyses across 3 networks:
IEEE 33-Bus, PEGASE 1354-Bus, Chengdu 1025-Bus.
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
    load_power_grid_data, build_weighted_graph,
    laplacian_eigendecomposition, compute_gft, compute_igft,
    verify_bandlimitedness, graph_sampling_reconstruction,
    graph_total_variation,
)

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'legend.fontsize': 7, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'serif',
})
SAVE_FIGS = True


def build_benchmark_graph(net_name):
    """Build graph and generate sag data for a benchmark network."""
    if net_name == 'IEEE33':
        net = pn.case33bw()
        pp.create_ext_grid(net, bus=0, vm_pu=1.0)
    elif net_name == 'PEGASE1354':
        net = pn.case1354pegase()
        pp.create_ext_grid(net, bus=0, vm_pu=1.0)
    else:
        raise ValueError(f'Unknown: {net_name}')

    N = len(net.bus)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    for _, line in net.line.iterrows():
        u, v = int(line['from_bus']), int(line['to_bus'])
        r, x = line['r_ohm_per_km']*line['length_km'], line['x_ohm_per_km']*line['length_km']
        z = np.sqrt(r**2 + x**2)
        G.add_edge(u, v, weight=1.0/max(z, 1e-6))

    for _, trafo in net.trafo.iterrows():
        G.add_edge(int(trafo['hv_bus']), int(trafo['lv_bus']), weight=50.0)

    W = nx.adjacency_matrix(G, weight='weight').astype(np.float64)
    D = sp.diags(np.array(W.sum(axis=1)).flatten())
    L = D - W
    eigvals, U = laplacian_eigendecomposition(L, k=min(N, 100))

    # Generate sag data via Zbus
    np.random.seed(42)
    Ybus = np.zeros((N, N), dtype=complex)
    for _, line in net.line.iterrows():
        u, v = int(line['from_bus']), int(line['to_bus'])
        r = line['r_ohm_per_km']*line['length_km']
        x = line['x_ohm_per_km']*line['length_km']
        z = r + 1j*x
        if abs(z) < 1e-12: z = 1e-10
        y = 1.0/z
        Ybus[u,v] -= y; Ybus[v,u] -= y
        Ybus[u,u] += y; Ybus[v,v] += y
    for _, trafo in net.trafo.iterrows():
        u, v = int(trafo['hv_bus']), int(trafo['lv_bus'])
        y = 1.0/(0.001 + 1j*0.05)
        Ybus[u,v] -= y; Ybus[v,u] -= y
        Ybus[u,u] += y; Ybus[v,v] += y
    Ybus[0,0] += 1.0/(0.01 + 1j*0.1)
    Zbus = np.linalg.pinv(Ybus)

    # Generate 10 sag events at diverse locations
    sigs = {}
    fault_buses = np.linspace(0, N-1, min(N, 10)).astype(int)
    eid = 0
    for fb in fault_buses:
        for ft, label in [('3ph', '3ph'), ('2ph', '2ph'), ('1ph', '1ph')]:
            if eid >= 10: break
            z_f = 0.01 + 1j*0
            z_kk = Zbus[fb, fb]
            if ft == '3ph': z_tot = z_kk + z_f
            elif ft == '2ph': z_tot = 2.0*(z_kk + z_f)
            else: z_tot = 3.0*(z_kk + z_f)
            v = np.ones(N)
            for i in range(N):
                vdrop = abs(Zbus[i,fb]/z_tot)
                v[i] = abs(1.0 - vdrop) if not np.iscomplex(vdrop) else abs(1.0 - vdrop)
            if np.min(v) < 0.99:
                sigs[eid] = v
                eid += 1
        if eid >= 10: break

    return G, L, U, eigvals, sigs, N


def setup_chengdu():
    """Setup Chengdu grid (shared across all analyses)."""
    bus_topo, line_topo, trans_topo, _, sag_bus, _, _ = load_power_grid_data()
    G, L, bus_names, bus_idx = build_weighted_graph(bus_topo, line_topo, trans_topo, weight_mode='admittance')
    cc = max(nx.connected_components(G), key=len)
    sc = sorted(cc)
    m = {o: n for n, o in enumerate(sc)}
    G = nx.relabel_nodes(G.subgraph(cc).copy(), m)
    bus_names = [bus_names[o] for o in sc]
    bus_idx = {n: i for i, n in enumerate(bus_names)}
    W = nx.adjacency_matrix(G, weight='weight').astype(np.float64)
    D = sp.diags(np.array(W.sum(axis=1)).flatten())
    L = D - W
    N = L.shape[0]
    eigvals, U = laplacian_eigendecomposition(L, k=min(N,200))
    sigs = {}
    for eid in sag_bus['EVENT_ID'].unique():
        ed = sag_bus[sag_bus['EVENT_ID'] == eid]
        s = np.zeros(N)
        for _, r in ed.iterrows():
            if r['BUSNAME'] in bus_idx:
                s[bus_idx[r['BUSNAME']]] = r['VA']
        if np.sum(s > 0) > 0.3 * N:
            sigs[eid] = s
    return G, L, U, eigvals, sigs, N, bus_names


# ===== 1. SAG DOMAIN IDENTIFICATION (all 3 networks) =====
def benchmark_sag_domain():
    print("=" * 60)
    print("1. SAG DOMAIN IDENTIFICATION — 2-NETWORK BENCHMARK")
    print("=" * 60)
    
    networks = {}
    print("  Building PEGASE 1354...")
    networks['PEGASE 1354-Bus'] = build_benchmark_graph('PEGASE1354')
    print("  Building Regional 1025...")
    networks['Regional 1025-Bus'] = setup_chengdu()[:6]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    all_fiedler_info = {}

    x = np.arange(4)  # DC, Low, Mid, High
    width = 0.35
    colors = {'PEGASE 1354-Bus': '#1f77b4', 'Regional 1025-Bus': '#ff7f0e'}

    for idx, (name, (G, L, U, eigvals, sigs, N)) in enumerate(networks.items()):
        if isinstance(N, tuple): N = N[0]
        sig = list(sigs.values())[0]
        n_eigs = min(N, len(eigvals)-1)
        
        # GFT decomposition
        xh = compute_gft(sig, U[:, :n_eigs+1])
        bands = {'DC': (0,1), 'Low': (1,3), 'Mid': (3,10), 'High': (10,min(50,n_eigs))}
        band_energies = {}
        for bname, (k0, k1) in bands.items():
            k1 = min(k1, n_eigs)
            if k1 > k0:
                band_energies[bname] = np.sum(np.abs(xh[k0:k1])**2)/np.sum(np.abs(xh)**2)*100
        
        vals = [band_energies[n] for n in ['DC', 'Low', 'Mid', 'High']]
        offset = width * (idx - 0.5)
        bars = ax.bar(x + offset, vals, width, label=f'{name} (N={N})',
                       color=colors[name], alpha=0.8, edgecolor='white')
        
        # Annotate values on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Fiedler partition info
        fiedler = U[:, 1]
        left = np.sum(fiedler < 0)
        right = N - left
        all_fiedler_info[name] = {'left': left, 'right': right, 'N': N}

    ax.set_xticks(x)
    ax.set_xticklabels(['DC', 'Low\n(2-3)', 'Mid\n(4-10)', 'High\n(>10)'], fontsize=10)
    ax.set_ylabel('Energy (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title('Sag Signal GFT Energy Distribution Across Network Scales', fontsize=14)
    ax.legend(fontsize=10)

    # Fiedler annotation
    f_text = 'Fiedler partition:\n ' + ' \n '.join(
        [f'{n.split("-")[0]}: {all_fiedler_info[n]["left"]}L/{all_fiedler_info[n]["right"]}R\n'
         for n in all_fiedler_info])
    ax.text(0.3, 0.5, f_text, transform=ax.transAxes, ha='left', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figE1_benchmark_domain.pdf')
        fig.savefig(OUTPUT_DIR / 'figE1_benchmark_domain.png')
    print('  Saved figE1_benchmark_domain')
    
    # Summary table
    print(f"\n  {'Network':<20} {'N':<8} {'DC%':<8} {'Low%':<8} {'Mid%':<8} {'Fiedler(L/R)':<15}")
    for name, (G, L, U, eigvals, sigs, N) in networks.items():
        if isinstance(N, tuple): N = N[0]
        sig = list(sigs.values())[0]
        n_eigs = min(N, len(eigvals)-1)
        xh = compute_gft(sig, U[:, :n_eigs+1])
        e_total = np.sum(np.abs(xh)**2)
        dc = np.sum(np.abs(xh[0:1])**2)/e_total*100
        low = np.sum(np.abs(xh[1:3])**2)/e_total*100
        mid = np.sum(np.abs(xh[3:10])**2)/e_total*100
        fi = all_fiedler_info[name]
        print(f"  {name:<20} {N:<8} {dc:.1f}%     {low:.1f}%     {mid:.1f}%     {fi['left']}/{fi['right']}")

    return networks


# ===== 2. SVI VULNERABILITY (all 3 networks) =====
def benchmark_svi(networks):
    print("\n" + "=" * 60)
    print("2. SVI VULNERABILITY — 3-NETWORK BENCHMARK")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    all_svi = {}

    for ax, (name, (G, L, U, eigvals, sigs, N)) in zip(axes, networks.items()):
        if isinstance(N, tuple): N = N[0]
        n_eigs = min(N, len(eigvals)-1)
        
        # SVI: L†[i,i] · σ_sag
        Lpinv_diag = np.zeros(N)
        for k in range(1, n_eigs):
            lam = max(eigvals[k], 1e-12)
            Lpinv_diag += U[:, k]**2 / lam
        
        sig_matrix = np.column_stack([sigs[e] for e in sorted(sigs.keys())])
        sig_std = np.std(sig_matrix, axis=1)
        
        SVI = (Lpinv_diag / np.median(Lpinv_diag)) * (sig_std / np.median(sig_std))
        SVI /= SVI.max()
        
        all_svi[name] = {'SVI': SVI, 'Lpinv_diag': Lpinv_diag, 'sig_std': sig_std}
        
        # Plot SVI distribution
        sort_idx = np.argsort(SVI)
        ax.plot(np.arange(N)/N*100, SVI[sort_idx], 'b-', linewidth=1)
        top5_idx = np.argsort(SVI)[::-1][:5]
        for idx in top5_idx:
            ax.axvline(x=np.where(sort_idx==idx)[0][0]/N*100, color='red', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Bus percentile (%)')
        ax.set_ylabel('SVI (normalized)')
        ax.set_title(f'{name}\nSVI Distribution')
        ax.grid(True, alpha=0.2)
        
        print(f"  {name:<20} N={N:<6} SVI max={SVI.max():.3f} SVI mean={SVI.mean():.3f}")

    fig.suptitle('Spectral Vulnerability Index Across Network Scales', fontsize=13)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figE2_benchmark_svi.pdf')
        fig.savefig(OUTPUT_DIR / 'figE2_benchmark_svi.png')
    print('  Saved figE2_benchmark_svi')

    return all_svi


# ===== 3. TV COMPENSATION (all 3 networks) =====
def benchmark_compensation(networks):
    print("\n" + "=" * 60)
    print("3. TV COMPENSATION — 3-NETWORK BENCHMARK")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    all_comp = {}

    for ax, (name, (G, L, U, eigvals, sigs, N)) in zip(axes, networks.items()):
        if isinstance(N, tuple): N = N[0]
        sig = list(sigs.values())[0]
        
        # Compute edge priorities
        edges = list(G.edges())
        priorities = []
        for u, v in edges:
            grad = abs(sig[u] - sig[v])
            w = G[u][v]['weight']
            priorities.append(grad * w)
        priorities.sort(reverse=True)
        
        # TV reduction vs compensation budget
        n_comp_options = list(range(0, min(51, len(edges)+1), max(1, len(edges)//20)))
        if len(n_comp_options) < 5:
            n_comp_options = list(range(0, len(edges)+1, max(1, len(edges)//10)))
        
        tv_values = []
        for n_c in n_comp_options:
            sig_c = sig.copy()
            for idx in range(min(n_c, len(edges))):
                u, v = edges[idx]
                if sig[v] < sig[u]: sig_c[v] += 0.3 * (sig[u] - sig[v])
                else: sig_c[u] += 0.3 * (sig[v] - sig[u])
            tv = graph_total_variation(sig_c, L)
            tv_values.append(tv)
        
        tv0 = tv_values[0]
        tv_reduction = [(tv0 - tv)/tv0*100 for tv in tv_values]
        
        ax.plot(n_comp_options, tv_reduction, 'o-', color='darkgreen', linewidth=1.2, markersize=4)
        ax.set_xlabel('Compensated edges')
        ax.set_ylabel('TV reduction (%)')
        ax.set_title(f'{name}\nTV Reduction')
        ax.grid(True, alpha=0.2)
        
        all_comp[name] = {'n_comp': n_comp_options, 'tv_red': tv_reduction}
        
        final_red = tv_reduction[-1]
        print(f"  {name:<20} N={N:<6} edges={len(edges):<6} TV red (max)={final_red:.1f}%")

    fig.suptitle('Graph TV-Guided Compensation Across Network Scales', fontsize=13)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figE3_benchmark_compensation.pdf')
        fig.savefig(OUTPUT_DIR / 'figE3_benchmark_compensation.png')
    print('  Saved figE3_benchmark_compensation')

    return all_comp


# ===== 4. FAULT TYPE DISCRIMINATION (all 3 networks) =====
def benchmark_fault_type():
    print("\n" + "=" * 60)
    print("4. FAULT TYPE DISCRIMINATION — 3-NETWORK BENCHMARK")
    print("=" * 60)
    
    # Generate events with explicit fault type labels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax_idx, net_name in enumerate(['PEGASE1354', 'Chengdu']):
        if net_name == 'Chengdu':
            G, L, U, eigvals, sigs, N, _ = setup_chengdu()
            from gsp_core import load_power_grid_data as lpgd
            _, _, _, _, _, sag_event, _ = lpgd()
            
            # Categorize by fault type
            type1_sigs = []
            type2_sigs = []
            for eid in sorted(sigs.keys()):
                evt_info = sag_event[sag_event['EVENT_ID'] == eid]
                if len(evt_info) > 0:
                    ft = evt_info.iloc[0]['FAULT_TYPE']
                    if ft == 1: type1_sigs.append(sigs[eid])
                    else: type2_sigs.append(sigs[eid])
            title = 'Chengdu 1025'
        else:
            G, L, U, eigvals, sigs, N = build_benchmark_graph(net_name)
            # For benchmarks, events 0,3,6 are 3ph; 1,4,7 are 2ph; 2,5,8 are 1ph
            type1_sigs = []
            type2_sigs = []
            for eid in sorted(sigs.keys()):
                if eid % 3 == 2: type1_sigs.append(sigs[eid])  # 1ph
                else: type2_sigs.append(sigs[eid])  # 3ph or 2ph
            title = f'{net_name}'
        
        n_eigs = min(N, len(U[0]))
        ax = axes[ax_idx]
        
        # HF/LF ratio for each event
        hf_lf_1p = []
        hf_lf_2p = []
        for sig in type1_sigs:
            xh = np.abs(compute_gft(sig, U[:, :n_eigs]))
            e = xh**2
            if np.sum(e[:5]) > 1e-10:
                hf_lf_1p.append(np.sum(e[5:])/np.sum(e[:5])*100)
        for sig in type2_sigs:
            xh = np.abs(compute_gft(sig, U[:, :n_eigs]))
            e = xh**2
            if np.sum(e[:5]) > 1e-10:
                hf_lf_2p.append(np.sum(e[5:])/np.sum(e[:5])*100)
        
        if hf_lf_1p and hf_lf_2p:
            bp = ax.boxplot([hf_lf_1p, hf_lf_2p],
                           patch_artist=True, widths=0.4)
            ax.set_xticklabels(['1\u03c6', '2\u03c6/3\u03c6'])
            for patch, color in zip(bp['boxes'], ['#2ca02c', '#d62728']):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_ylabel('HF/LF ratio (%)')
            ax.set_title(f'{title} (N={N})')
            ax.grid(True, alpha=0.2, axis='y')
            
            print(f"  {title:<20} 1φ HF/LF={np.mean(hf_lf_1p):.4f}%  "
                  f"2φ/3φ HF/LF={np.mean(hf_lf_2p):.4f}%  "
                  f"Ratio={np.mean(hf_lf_2p)/np.mean(hf_lf_1p):.1f}x")
    
    fig.suptitle('Fault Type Discrimination via GFT Across Network Scales', fontsize=13)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figE4_benchmark_fault_type.pdf')
        fig.savefig(OUTPUT_DIR / 'figE4_benchmark_fault_type.png')
    print('  Saved figE4_benchmark_fault_type')


# ===== 5. CROSS-EVENT + SAMPLING DESIGN (all 3 networks) =====
def benchmark_cross_event(networks):
    print("\n" + "=" * 60)
    print("5. CROSS-EVENT VALIDATION — 3-NETWORK BENCHMARK")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (name, (G, L, U, eigvals, sigs, N)) in zip(axes, networks.items()):
        if isinstance(N, tuple): N = N[0]
        n_eigs = min(N, len(eigvals)-1)
        
        M_ratio = min(10, N//3)  # ~30% of nodes
        n_trials = 5
        
        gsp_rmses = []
        idw_rmses = []
        
        all_dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        
        for eid in sorted(sigs.keys()):
            sig = sigs[eid]
            S = np.random.choice(N, M_ratio, replace=False)
            y_S = sig[S]
            xh = graph_sampling_reconstruction(L, U[:, :n_eigs], eigvals[:n_eigs], y_S, S, min(3, n_eigs))
            gsp_rmses.append(np.sqrt(np.mean((xh - sig)**2)))
            
            S_set = set(S)
            x_i = np.zeros(N)
            for i in range(N):
                if i in S_set:
                    x_i[i] = sig[i]
                else:
                    dists = np.array([all_dist[i].get(s, 1e6) for s in S])
                    vs = np.array([sig[s] for s in S])
                    w = 1.0/(dists**2 + 1e-10); w = w/w.sum()
                    x_i[i] = np.dot(w, vs)
            idw_rmses.append(np.sqrt(np.mean((x_i - sig)**2)))
        
        n_events = len(gsp_rmses)
        ax.bar(np.arange(n_events)-0.15, gsp_rmses, 0.3, label='GSP', color='#1f77b4', alpha=0.8)
        ax.bar(np.arange(n_events)+0.15, idw_rmses, 0.3, label='IDW', color='#ff7f0e', alpha=0.8)
        ax.set_xlabel('Event')
        ax.set_title(f'{name} (M={M_ratio}, {n_events} events)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis='y')
        
        gsp_win = sum(1 for g, i in zip(gsp_rmses, idw_rmses) if g < i)
        print(f"  {name:<20} M={M_ratio:<4} GSP wins {gsp_win}/{n_events}  "
              f"GSP avg={np.mean(gsp_rmses):.4f}  IDW avg={np.mean(idw_rmses):.4f}")
    
    fig.suptitle('Cross-Event Validation Across Network Scales', fontsize=13)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figE5_benchmark_crossevent.pdf')
        fig.savefig(OUTPUT_DIR / 'figE5_benchmark_crossevent.png')
    print('  Saved figE5_benchmark_crossevent')


# ===== 6. SUMMARY TABLE =====
def generate_summary_table(networks):
    print("\n" + "=" * 60)
    print("6. COMPREHENSIVE BENCHMARK SUMMARY TABLE")
    print("=" * 60)
    
    rows = []
    for name, (G, L, U, eigvals, sigs, N) in networks.items():
        if isinstance(N, tuple): N = N[0]
        n_eigs = min(N, len(eigvals)-1)
        sig = list(sigs.values())[0]
        xh = compute_gft(sig, U[:, :n_eigs+1])
        e = np.abs(xh)**2; etot = e.sum()
        
        # Bandwidth
        cum_e = np.cumsum(np.sort(e)[::-1]) / etot
        bw90 = np.argmax(cum_e >= 0.90) + 1
        bw95 = np.argmax(cum_e >= 0.95) + 1
        
        # SVI
        Lpinv = np.zeros(N)
        for k in range(1, n_eigs):
            Lpinv += U[:,k]**2 / max(eigvals[k], 1e-12)
        sig_matrix = np.column_stack([sigs[e] for e in sorted(sigs.keys())])
        SVI = (Lpinv/np.median(Lpinv)) * (np.std(sig_matrix, axis=1)/np.median(np.std(sig_matrix, axis=1)))
        
        # TV reduction
        edges = list(G.edges())
        comp_grads = sorted([abs(sig[u]-sig[v])*G[u][v]['weight'] for u,v in edges], reverse=True)
        
        # Fiedler
        fiedler = U[:,1]
        
        rows.append({
            'name': name, 'N': N, 'edges': G.number_of_edges(),
            'bw90': bw90, 'bw95': bw95,
            'dc_pct': e[0]/etot*100,
            'svi_range': f'{np.min(SVI):.2f}-{np.max(SVI):.2f}',
            'fiedler_split': f'{np.sum(fiedler<0)}/{np.sum(fiedler>=0)}',
            'tv_x_smoother': f'{1.0:.0f}x',
        })
    
    print(f"\n  {'Network':<20} {'N':<6} {'Edges':<7} {'BW90':<6} {'BW95':<6} {'DC%':<7} {'SVI range':<12} {'Fiedler':<12}")
    print(f"  {'-'*80}")
    for r in rows:
        print(f"  {r['name']:<20} {r['N']:<6} {r['edges']:<7} "
              f"{r['bw90']:<6} {r['bw95']:<6} {r['dc_pct']:.1f}%    "
              f"{r['svi_range']:<12} {r['fiedler_split']:<12}")
    
    return rows


if __name__ == '__main__':
    print("=" * 60)
    print("COMPREHENSIVE 3-NETWORK BENCHMARK FOR ALL GSP ANALYSES")
    print("=" * 60)
    
    networks = benchmark_sag_domain()
    benchmark_fault_type()
    all_svi = benchmark_svi(networks)
    all_comp = benchmark_compensation(networks)
    benchmark_cross_event(networks)
    summary = generate_summary_table(networks)
    
    print("\n" + "=" * 60)
    print("ALL BENCHMARK COMPARISONS COMPLETE")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print("=" * 60)
