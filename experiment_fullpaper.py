"""
Full-paper enrichment experiments for IEEE TPWRD (10-page target):
1. Kriging interpolation baseline
2. MLP/GCN baseline  
3. Graph weight ablation (binary vs admittance vs distance)
4. Cross-event validation (all 12 events)
5. Sampling set design comparison (random, max-degree, eigenvector, greedy)
6. Error bound analysis
"""
import numpy as np
import networkx as nx
from scipy import sparse as sp
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from gsp_core import (
    load_power_grid_data, build_weighted_graph,
    laplacian_eigendecomposition, compute_gft,
    verify_bandlimitedness, graph_sampling_reconstruction,
    design_sampling_set,
)

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 8, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'serif',
})
SAVE_FIGS = True


def setup_chengdu_graph():
    """Common setup: build Chengdu graph and get signals."""
    bus_topo, line_topo, trans_topo, _, sag_bus, sag_event, _ = load_power_grid_data()
    G, L, bus_names, bus_idx = build_weighted_graph(
        bus_topo, line_topo, trans_topo, weight_mode='admittance')
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

    # Get all valid sag signals
    sag_signals = {}
    for evt_id in sag_bus['EVENT_ID'].unique():
        evt_data = sag_bus[sag_bus['EVENT_ID'] == evt_id]
        signal = np.zeros(N)
        for _, row in evt_data.iterrows():
            if row['BUSNAME'] in bus_idx_sub:
                signal[bus_idx_sub[row['BUSNAME']]] = row['VA']
        if np.sum(signal > 0) > 0.3 * N:
            sag_signals[evt_id] = signal

    return G_sub, L, U, eigvals, bus_names_sub, bus_idx_sub, sag_signals, sag_bus, sag_event, N


# ============================================================
# 1. Kriging Baseline
# ============================================================
def run_kriging_baseline():
    print("=" * 60)
    print("1. KRIGING INTERPOLATION BASELINE")
    print("=" * 60)
    G, L, U, eigvals, bus_names, bus_idx, sag_signals, _, _, N = setup_chengdu_graph()

    # Use bus coordinates as spatial features
    bus_topo, _, _, _, _, _, _ = load_power_grid_data()
    coords = {}
    for _, row in bus_topo.iterrows():
        if row['BUSNAME'] in bus_idx:
            coords[bus_idx[row['BUSNAME']]] = (float(row['LONGITUDE']), float(row['LATITUDE']))

    # Build coordinate matrix
    coord_mat = np.zeros((N, 2))
    for i in range(N):
        if i in coords:
            coord_mat[i] = coords[i]
        else:
            coord_mat[i] = [104.0, 30.5]  # default center of Chengdu

    # Ordinary Kriging implementation
    def ordinary_kriging(X_train, y_train, X_test, variogram_range=0.5):
        """Simple ordinary Kriging with exponential variogram."""
        N_train = len(y_train)
        N_test = len(X_test)

        # Build variogram matrix
        dist = cdist(X_train, X_train)
        gamma = 1.0 - np.exp(-3 * dist / variogram_range)  # exponential model

        # Kriging system
        K = np.zeros((N_train + 1, N_train + 1))
        K[:N_train, :N_train] = gamma
        K[:N_train, N_train] = 1.0
        K[N_train, :N_train] = 1.0
        K[N_train, N_train] = 0.0

        y_aug = np.zeros(N_train + 1)
        y_aug[:N_train] = y_train

        try:
            weights = np.linalg.solve(K, y_aug)[:N_train]
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(K, y_aug, rcond=None)[0][:N_train]

        # Predict
        predictions = np.zeros(N_test)
        for j in range(N_test):
            dist_j = cdist(X_test[j:j+1], X_train).flatten()
            gamma_j = 1.0 - np.exp(-3 * dist_j / variogram_range)
            predictions[j] = np.dot(weights, gamma_j)

        return predictions

    # IDW as before
    all_dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    def idw_interpolate(test_sig, S, N):
        S_set = set(S)
        x_idw = np.zeros(N)
        for i in range(N):
            if i in S_set:
                x_idw[i] = test_sig[i]
            else:
                dists = np.array([all_dist[i].get(s, 1e6) for s in S])
                vals = np.array([test_sig[s] for s in S])
                w = 1.0 / (dists**2 + 1e-10)
                w = w / w.sum()
                x_idw[i] = np.dot(w, vals)
        return x_idw

    test_sig = list(sag_signals.values())[0]
    Ms = [3, 5, 10, 15, 20, 30, 50]
    n_trials = 10

    gsp_rmse = {bw: [] for bw in [1, 5]}
    idw_rmse = [[] for _ in Ms]
    krig_rmse_geo = [[] for _ in Ms]   # geographic coordinates
    krig_rmse_graph = [[] for _ in Ms]  # graph distance coordinates

    # Precompute graph Laplacian eigenmap coordinates (spectral embedding)
    spectral_coords = U[:, :5].copy()  # Use first 5 eigenvectors as latent coordinates

    for M in Ms:
        for _ in range(n_trials):
            S = np.random.choice(N, M, replace=False)
            y_S = test_sig[S]

            # GSP (BW=5)
            xh = graph_sampling_reconstruction(L, U, eigvals, y_S, S, 5)
            gsp_rmse[5].append(np.sqrt(np.mean((xh - test_sig)**2)))

            # IDW
            x_i = idw_interpolate(test_sig, S, N)
            idw_rmse[Ms.index(M)].append(np.sqrt(np.mean((x_i - test_sig)**2)))

            # Kriging - geographic
            try:
                x_k_g = ordinary_kriging(coord_mat[S], test_sig[S], coord_mat)
                krig_rmse_geo[Ms.index(M)].append(np.sqrt(np.mean((x_k_g - test_sig)**2)))
            except:
                krig_rmse_geo[Ms.index(M)].append(np.nan)

            # Kriging - spectral coordinates
            try:
                x_k_s = ordinary_kriging(spectral_coords[S], test_sig[S], spectral_coords)
                krig_rmse_graph[Ms.index(M)].append(np.sqrt(np.mean((x_k_s - test_sig)**2)))
            except:
                krig_rmse_graph[Ms.index(M)].append(np.nan)

    # Aggregate results
    gsp_vals = []
    for M in Ms:
        vals = []
        for _ in range(n_trials):
            S = np.random.choice(N, M, replace=False)
            y_S = test_sig[S]
            xh = graph_sampling_reconstruction(L, U, eigvals, y_S, S, 5)
            vals.append(np.sqrt(np.mean((xh - test_sig)**2)))
        gsp_vals.append(np.mean(vals))

    idw_mean = [np.nanmean(v) for v in idw_rmse]
    krig_geo_mean = [np.nanmean(v) for v in krig_rmse_geo]
    krig_spec_mean = [np.nanmean(v) for v in krig_rmse_graph]

    # Figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ms, gsp_vals, 'o-', color='#1f77b4', linewidth=1.5, label='GSP Sampling (BW=5)')
    ax.plot(Ms, krig_geo_mean, 'D--', color='#d62728', linewidth=1.2, label='Kriging (Geographic)')
    ax.plot(Ms, krig_spec_mean, '^--', color='#9467bd', linewidth=1.2, label='Kriging (Spectral)')
    ax.plot(Ms, idw_mean, 's:', color='#ff7f0e', linewidth=1.2, label='IDW (Graph Dist.)')
    ax.set_xlabel('Number of monitors (M)')
    ax.set_ylabel('RMSE (p.u.)')
    ax.set_title('Interpolation Baselines: GSP vs. Kriging vs. IDW')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figB1_kriging_baseline.pdf')
        fig.savefig(OUTPUT_DIR / 'figB1_kriging_baseline.png')
    print('  Saved figB1_kriging_baseline')

    return {'gsp': gsp_vals, 'kriging_geo': krig_geo_mean,
            'kriging_spec': krig_spec_mean, 'idw': idw_mean}


# ============================================================
# 2. MLP/GCN Baseline
# ============================================================
def run_mlp_gcn_baseline():
    print("\n" + "=" * 60)
    print("2. MLP/GCN BASELINE")
    print("=" * 60)
    G, L, U, eigvals, bus_names, bus_idx, sag_signals, sag_bus, sag_event, N = setup_chengdu_graph()

    # Use 70% of buses for training, 30% for testing
    train_frac = 0.7
    n_train = int(N * train_frac)
    n_events = len(sag_signals)

    # Build feature matrix: for each bus, use degree + graph embedding
    degrees = np.array([G.degree(n) for n in range(N)])
    spectral_features = U[:, :5]

    # Simple MLP trained on partial buses, tested on withheld
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    Ms = [3, 5, 10, 15, 20, 30, 50]
    n_trials = 5

    mlp_rmses = []
    gcn_rmses = []

    test_sig = list(sag_signals.values())[0]
    features = np.column_stack([degrees, spectral_features])

    for M in Ms:
        trial_rmse = []
        for _ in range(n_trials):
            train_idx = np.random.choice(N, M, replace=False)
            test_idx = np.setdiff1d(np.arange(N), train_idx)

            X_train = features[train_idx]
            y_train = test_sig[train_idx]
            X_test = features[test_idx]
            y_test = test_sig[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500,
                               random_state=42, early_stopping=False)
            mlp.fit(X_train_s, y_train)
            y_pred = mlp.predict(X_test_s)
            trial_rmse.append(np.sqrt(np.mean((y_pred - y_test)**2)))

        mlp_rmses.append(np.mean(trial_rmse))
        print(f"    MLP M={M}: RMSE={mlp_rmses[-1]:.4f}")

    # GSP comparison
    gsp_vals = []
    for M in Ms:
        vals = []
        for _ in range(n_trials):
            S = np.random.choice(N, M, replace=False)
            y_S = test_sig[S]
            xh = graph_sampling_reconstruction(L, U, eigvals, y_S, S, 5)
            vals.append(np.sqrt(np.mean((xh - test_sig)**2)))
        gsp_vals.append(np.mean(vals))

    # Also IDW
    all_dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    idw_vals = []
    for M in Ms:
        vals = []
        for _ in range(n_trials):
            S = np.random.choice(N, M, replace=False)
            S_set = set(S)
            x_idw = np.zeros(N)
            for i in range(N):
                if i in S_set:
                    x_idw[i] = test_sig[i]
                else:
                    dists = np.array([all_dist[i].get(s, 1e6) for s in S])
                    vs = np.array([test_sig[s] for s in S])
                    w = 1.0 / (dists**2 + 1e-10)
                    w = w / w.sum()
                    x_idw[i] = np.dot(w, vs)
            vals.append(np.sqrt(np.mean((x_idw - test_sig)**2)))
        idw_vals.append(np.mean(vals))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ms, gsp_vals, 'o-', color='#1f77b4', linewidth=1.5, label='GSP Sampling (BW=5)')
    ax.plot(Ms, mlp_rmses, 's--', color='#2ca02c', linewidth=1.2, label='MLP (spectral features)')
    ax.plot(Ms, idw_vals, 'D:', color='#ff7f0e', linewidth=1.2, label='IDW (graph dist.)')
    ax.set_xlabel('Number of monitors (M)')
    ax.set_ylabel('RMSE (p.u.)')
    ax.set_title('GSP vs. Data-Driven Baselines')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figB2_mlp_baseline.pdf')
        fig.savefig(OUTPUT_DIR / 'figB2_mlp_baseline.png')
    print('  Saved figB2_mlp_baseline')

    return {'gsp': gsp_vals, 'mlp': mlp_rmses}


# ============================================================
# 3. Graph Weight Ablation
# ============================================================
def run_weight_ablation():
    print("\n" + "=" * 60)
    print("3. GRAPH WEIGHT ABLATION")
    print("=" * 60)

    bus_topo, line_topo, trans_topo, _, sag_bus, _, _ = load_power_grid_data()

    weight_configs = {
        'Admittance (1/|Z|)': 'admittance',
        'Distance (exp(-d))': 'distance',
        'Binary (0/1)': 'binary',
    }

    results = {}
    for name, mode in weight_configs.items():
        G, L, bus_names, bus_idx = build_weighted_graph(
            bus_topo, line_topo, trans_topo, weight_mode=mode)

        largest_cc = max(nx.connected_components(G), key=len)
        sorted_cc = sorted(largest_cc)
        old_to_new = {old: new for new, old in enumerate(sorted_cc)}
        G_sub = G.subgraph(largest_cc).copy()
        G_sub = nx.relabel_nodes(G_sub, old_to_new)
        bus_names_sub = [bus_names[old] for old in sorted_cc]
        bus_idx_sub = {n: i for i, n in enumerate(bus_names_sub)}
        W_sub = nx.adjacency_matrix(G_sub, weight='weight').astype(np.float64)
        D_sub = sp.diags(np.array(W_sub.sum(axis=1)).flatten())
        L = D_sub - W_sub
        N = L.shape[0]
        eigvals, U = laplacian_eigendecomposition(L, k=min(N, 200))

        # Get signal
        signal = None
        for evt_id in sag_bus['EVENT_ID'].unique():
            evt_data = sag_bus[sag_bus['EVENT_ID'] == evt_id]
            sig = np.zeros(N)
            for _, row in evt_data.iterrows():
                if row['BUSNAME'] in bus_idx_sub:
                    sig[bus_idx_sub[row['BUSNAME']]] = row['VA']
            if np.sum(sig > 0) > 0.3 * N:
                signal = sig
                break

        # Bandlimitedness
        cum_e, _ = verify_bandlimitedness(signal, U, eigvals, top_k=50)
        bw_95 = np.argmax(cum_e >= 0.95) + 1

        # Sampling experiment
        Ms = [3, 5, 10, 20, 30]
        gsp_vals = []
        for M in Ms:
            vals = []
            for _ in range(10):
                S = np.random.choice(N, M, replace=False)
                y_S = signal[S]
                xh = graph_sampling_reconstruction(L, U, eigvals, y_S, S, min(5, min(N, 200)))
                vals.append(np.sqrt(np.mean((xh - signal)**2)))
            gsp_vals.append(np.mean(vals))

        results[name] = {'bw_95': bw_95, 'gsp': gsp_vals, 'Ms': Ms}
        print(f"  {name}: BW(95%)={bw_95}, GSP(M=20)={gsp_vals[Ms.index(20)]:.4f}")

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = ['#1f77b4', '#2ca02c', '#d62728']
    for (name, res), c in zip(results.items(), colors):
        ax1.plot(res['Ms'], res['gsp'], 'o-', color=c, linewidth=1.5, label=name)
    ax1.set_xlabel('Number of monitors (M)')
    ax1.set_ylabel('RMSE (p.u.)')
    ax1.set_title('GSP Performance by Weight Type')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    names = list(results.keys())
    bws = [results[n]['bw_95'] for n in names]
    rmses = [results[n]['gsp'][results[n]['Ms'].index(20)] for n in names]
    ax2.bar(range(len(names)), rmses, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([f'{n}\nBW95={bw}' for n, bw in zip(names, bws)], fontsize=8)
    ax2.set_ylabel('RMSE (p.u.) at M=20')
    ax2.set_title('Weight Imapct on Reconstruction')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Graph Weight Construction Ablation Study')
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figB3_weight_ablation.pdf')
        fig.savefig(OUTPUT_DIR / 'figB3_weight_ablation.png')
    print('  Saved figB3_weight_ablation')

    return results


# ============================================================
# 4. Cross-Event Validation
# ============================================================
def run_cross_event():
    print("\n" + "=" * 60)
    print("4. CROSS-EVENT VALIDATION (ALL EVENTS)")
    print("=" * 60)
    G, L, U, eigvals, bus_names, bus_idx, sag_signals, sag_bus, sag_event, N = setup_chengdu_graph()

    Ms = [5, 10, 20, 30, 50]
    n_trials = 10

    all_events = list(sag_signals.keys())
    event_gsp = {evt: {m: [] for m in Ms} for evt in all_events}
    event_idw = {evt: {m: [] for m in Ms} for evt in all_events}

    all_dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    for evt in all_events:
        sig = sag_signals[evt]
        for M in Ms:
            for _ in range(n_trials):
                S = np.random.choice(N, M, replace=False)
                S_set = set(S)
                y_S = sig[S]

                xh = graph_sampling_reconstruction(L, U, eigvals, y_S, S, min(5, len(eigvals)))
                event_gsp[evt][M].append(np.sqrt(np.mean((xh - sig)**2)))

                x_i = np.zeros(N)
                for i in range(N):
                    if i in S_set:
                        x_i[i] = sig[i]
                    else:
                        dists = np.array([all_dist[i].get(s, 1e6) for s in S])
                        vs = np.array([sig[s] for s in S])
                        w = 1.0 / (dists**2 + 1e-10)
                        w = w / w.sum()
                        x_i[i] = np.dot(w, vs)
                event_idw[evt][M].append(np.sqrt(np.mean((x_i - sig)**2)))

    # Aggregate
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    gsp_mean = {}
    idw_mean = {}
    for M in Ms:
        evt_gsp_vals = [np.mean(event_gsp[evt][M]) for evt in all_events]
        evt_idw_vals = [np.mean(event_idw[evt][M]) for evt in all_events]
        gsp_mean[M] = np.mean(evt_gsp_vals)
        idw_mean[M] = np.mean(evt_idw_vals)

        ax.plot([M]*len(all_events), evt_gsp_vals, 'o', color='#1f77b4', alpha=0.3, markersize=4)
        ax.plot([M]*len(all_events), evt_idw_vals, 's', color='#ff7f0e', alpha=0.3, markersize=4)

    ax.plot(Ms, [gsp_mean[m] for m in Ms], 'o-', color='#1f77b4', linewidth=2, label='GSP (mean)')
    ax.plot(Ms, [idw_mean[m] for m in Ms], 's-', color='#ff7f0e', linewidth=2, label='IDW (mean)')
    ax.set_xlabel('Number of monitors (M)')
    ax.set_ylabel('RMSE (p.u.)')
    ax.set_title(f'Cross-Event Validation ({len(all_events)} events)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bar chart: per-event GSP vs IDW at M=20
    ax = axes[1]
    x = np.arange(len(all_events))
    width = 0.35
    gsp_e20 = [np.mean(event_gsp[evt][20]) for evt in all_events]
    idw_e20 = [np.mean(event_idw[evt][20]) for evt in all_events]
    ax.bar(x - width/2, gsp_e20, width, label='GSP (M=20)', color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, idw_e20, width, label='IDW (M=20)', color='#ff7f0e', alpha=0.8)
    ax.set_xlabel('Event')
    ax.set_ylabel('RMSE (p.u.)')
    ax.set_title(f'Per-Event RMSE at M=20')
    ax.set_xticks(x)
    ax.set_xticklabels([f'E{str(e)[-4:]}' for e in all_events], fontsize=7, rotation=45)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Cross-Event Generalization')
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figB4_cross_event.pdf')
        fig.savefig(OUTPUT_DIR / 'figB4_cross_event.png')
    print('  Saved figB4_cross_event')

    # Print summary
    for M in Ms:
        gimp = [(np.mean(event_idw[evt][M]) - np.mean(event_gsp[evt][M])) / np.mean(event_idw[evt][M]) * 100
                for evt in all_events]
        print(f"  M={M}: GSP mean={gsp_mean[M]:.4f}, IDW mean={idw_mean[M]:.4f}, "
              f"GSP wins in {sum(1 for g in gimp if g > 0)}/{len(all_events)} events")

    return {'gsp_mean': gsp_mean, 'idw_mean': idw_mean}


# ============================================================
# 5. Sampling Set Design Comparison
# ============================================================
def run_sampling_design():
    print("\n" + "=" * 60)
    print("5. SAMPLING SET DESIGN COMPARISON")
    print("=" * 60)
    G, L, U, eigvals, bus_names, bus_idx, sag_signals, _, _, N = setup_chengdu_graph()
    test_sig = list(sag_signals.values())[0]

    # Greedy sampling: iteratively add the node that minimizes reconstruction error
    def greedy_sampling(G, L, U, eigvals, signal, M, n_eigs, bw=5):
        selected = []
        remaining = list(range(N))
        # Start with max-degree node
        degrees = np.array([G.degree(n) for n in range(N)])
        first = np.argmax(degrees)
        selected.append(first)
        remaining.remove(first)

        for _ in range(M - 1):
            best_node = None
            best_rmse = float('inf')
            # Sample a subset to test (too expensive to test all)
            candidates = np.random.choice(remaining, min(50, len(remaining)), replace=False)
            for cand in candidates:
                S_test = selected + [cand]
                y_S = signal[S_test]
                xh = graph_sampling_reconstruction(L, U[:, :n_eigs], eigvals[:n_eigs], y_S, S_test, bw)
                rmse = np.sqrt(np.mean((xh - signal)**2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_node = cand
            selected.append(best_node)
            remaining.remove(best_node)
        return np.array(selected)

    Ms = [3, 5, 8, 10, 15, 20, 30]
    methods = ['random', 'max_degree', 'eigenvector']
    n_trials = 10

    results = {m: {} for m in methods}
    n_eigs = min(N, 200)

    for method in methods:
        for M in Ms:
            vals = []
            for _ in range(n_trials):
                if method == 'greedy':
                    S = greedy_sampling(G, L, U, eigvals, test_sig, M, n_eigs, bw=5)
                else:
                    S = design_sampling_set(G, L, U, eigvals, M, method=method)
                if len(S) != M:
                    continue
                y_S = test_sig[S]
                xh = graph_sampling_reconstruction(L, U[:, :n_eigs], eigvals[:n_eigs], y_S, S, 5)
                vals.append(np.sqrt(np.mean((xh - test_sig)**2)))
            results[method][M] = np.mean(vals)
        print(f"  {method}: M=10 RMSE={results[method][10]:.4f}, M=20 RMSE={results[method][20]:.4f}")

    # Figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {'random': '#1f77b4', 'max_degree': '#2ca02c', 'eigenvector': '#d62728'}
    markers = {'random': 'o', 'max_degree': 's', 'eigenvector': '^'}
    for method in methods:
        valid_Ms = sorted(results[method].keys())
        valid_rmses = [results[method][m] for m in valid_Ms]
        ax.plot(valid_Ms, valid_rmses, f'{markers[method]}-', color=colors[method],
                linewidth=1.5, label=method.replace('_', ' ').title())

    ax.set_xlabel('Number of monitors (M)')
    ax.set_ylabel('RMSE (p.u.)')
    ax.set_title('Sampling Set Design Strategies (BW=5)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if SAVE_FIGS:
        fig.savefig(OUTPUT_DIR / 'figB5_sampling_design.pdf')
        fig.savefig(OUTPUT_DIR / 'figB5_sampling_design.png')
    print('  Saved figB5_sampling_design')

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("FULL-PAPER ENRICHMENT EXPERIMENTS")
    print("=" * 60)
    res1 = run_kriging_baseline()
    res2 = run_mlp_gcn_baseline()
    res3 = run_weight_ablation()
    res4 = run_cross_event()
    res5 = run_sampling_design()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print("=" * 60)
