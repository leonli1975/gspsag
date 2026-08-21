"""
W1: Paired significance test of GSP vs IDW on the PEGASE 60 wide-area deep-sag
events. For each event, computes paired RMSE (same sampling set for both
methods), then paired t-test + Wilcoxon signed-rank over the 60 events.
"""
import sys
from pathlib import Path
import numpy as np
import networkx as nx
from scipy import stats
import pandapower.networks as pn
import pandapower as pp

sys.path.insert(0, str(Path(__file__).parent))
from gsp_core import laplacian_eigendecomposition
from experiment_pegase_validation import build_graph_from_pp_network, generate_sag_data_zbus

print("[1/3] Loading PEGASE network...")
net = pn.case1354pegase()
pp.create_ext_grid(net, bus=0, vm_pu=1.0)
G, L, N = build_graph_from_pp_network(net)

print("[2/3] Eigendecomposition + sag generation...")
Ld = L.toarray()
ev, U = np.linalg.eigh(Ld)
eigvals = ev
n_eigs = 100
U = U[:, :n_eigs]
eigvals = eigvals[:n_eigs]
eigvals = np.clip(eigvals, 0, None)

sigs = generate_sag_data_zbus(net, n_events=60)
sig_arr = np.array(list(sigs.values()))  # (60, N)
n_evt = sig_arr.shape[0]
print(f"  {n_evt} events, N={N}")

# severity check
frac = (sig_arr < 0.85).mean(axis=1)
print(f"  <0.85 frac: median={np.median(frac):.1%}, min={frac.min():.1%}")

# Precompute distance matrix for IDW
print("[3/3] Paired GSP vs IDW at B=5, M=30...")
all_dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
dist_mat = np.full((N, N), 1e6)
for i in range(N):
    for j, d in all_dist[i].items():
        dist_mat[i, j] = d

B = 5
M = 30
U_F = U[:, :B]
n_trials = 10
rng = np.random.RandomState(42)

# For each event: mean GSP RMSE and mean IDW RMSE across trials
gsp_rmse_evt = np.zeros(n_evt)
idw_rmse_evt = np.zeros(n_evt)
for evt_i in range(n_evt):
    sig = sig_arr[evt_i]
    gsp_rs = []
    idw_rs = []
    for _ in range(n_trials):
        S = rng.choice(N, M, replace=False)
        # GSP
        Phi = U_F[S, :]
        pinv = np.linalg.pinv(Phi)
        y = sig[S]
        alpha = pinv @ y
        xh = U_F @ alpha
        gsp_rs.append(np.sqrt(np.mean((xh - sig)**2)))
        # IDW (same S)
        dS = dist_mat[:, S] + 1e-10
        wS = 1.0 / dS**2
        wS = wS / wS.sum(axis=1, keepdims=True)
        xidw = wS @ sig[S]
        idw_rs.append(np.sqrt(np.mean((xidw - sig)**2)))
    gsp_rmse_evt[evt_i] = np.mean(gsp_rs)
    idw_rmse_evt[evt_i] = np.mean(idw_rs)

diff = gsp_rmse_evt - idw_rmse_evt  # negative = GSP better
print("\n" + "=" * 60)
print("W1 PAIRED SIGNIFICANCE TEST (PEGASE 60 wide-area events, B=5, M=30)")
print("=" * 60)
print(f"  GSP mean RMSE: {gsp_rmse_evt.mean():.4f} (std {gsp_rmse_evt.std():.4f})")
print(f"  IDW mean RMSE: {idw_rmse_evt.mean():.4f} (std {idw_rmse_evt.std():.4f})")
print(f"  Mean improvement: {1 - gsp_rmse_evt.mean()/idw_rmse_evt.mean():.1%}")
print(f"  Events GSP wins: {(diff < 0).sum()}/{n_evt} ({(diff<0).mean():.1%})")

t_stat, p_t = stats.ttest_rel(gsp_rmse_evt, idw_rmse_evt)
print(f"  Paired t-test: t={t_stat:.2f}, p={p_t:.2e}")
w_stat, p_w = stats.wilcoxon(diff)
print(f"  Wilcoxon signed-rank: W={w_stat}, p={p_w:.2e}")

# Effect size (Cohen's d for paired)
d_cohen = diff.mean() / diff.std(ddof=1)
print(f"  Cohen's d (paired): {d_cohen:.2f}")

# also 4112 catalog internal: localised shallow sag => expect IDW wins
# (not part of W1 but confirm the boundary)
print("\n[Cross-check] Regional 6 wide-area events (from paper data):")
print("  paper reports 4/6 wins, mean RMSE 0.054 vs 0.077 (not re-run here)")
print("=" * 60)
