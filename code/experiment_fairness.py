"""Fairness check for the designed-sampling result.

Shows that GSP's advantage is not an artefact of giving GSP a better monitor set:
for the SAME D-optimal monitor set we evaluate both GSP and IDW, and we also compare
IDW on the designed set with IDW on random sets. If IDW on the designed set is no better
(or worse) than IDW on random sets, then designed placement does not unfairly favour IDW
in the reverse direction either, and GSP(designed) < IDW(designed) on the same monitors is
a genuine algorithmic advantage.

Saves results_fairness.json.
"""
import json
import os

import numpy as np
import scipy.sparse as sp

from electrical_model import build_network, DEFAULT_DATA
from experiment_graph_sensitivity import (
    build_operators, laplacian, load_detailed_events, distance_matrix)
from experiment_estimation_designed import greedy_d_optimal, gsp, idw, rmse

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'results_fairness.json')
N_DRAWS = 10


def main():
    net = build_network(
        os.path.join(BASE, 'BUSTOPO.csv'), os.path.join(BASE, 'LINETOPO.csv'),
        os.path.join(BASE, 'TRANSTOPO.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_LINE.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_TRANS.csv'))
    n = net['n']
    ops, kind = build_operators(net)
    ev = load_detailed_events(net)

    out = {'N': n, 'n_draws': N_DRAWS, 'operators': {}}
    for name, W in ops.items():
        L = sp.csr_matrix(laplacian(W))
        vals, U = np.linalg.eigh(L.toarray())
        D = distance_matrix(W)
        op = {}
        for M in [30, 100]:
            B = max(1, min(M // 3, 50))
            Sd = greedy_d_optimal(U, B, M)
            g_des, i_des, g_rand, i_rand = [], [], [], []
            for k, x in ev.items():
                g_des.append(rmse(gsp(U, x, Sd, B), x))
                i_des.append(rmse(idw(D, x, Sd), x))          # IDW on the SAME designed set
                gs, is_ = [], []
                for t in range(N_DRAWS):
                    Sr = np.random.RandomState(t).choice(n, M, replace=False)
                    gs.append(rmse(gsp(U, x, Sr, B), x))
                    is_.append(rmse(idw(D, x, Sr), x))
                g_rand.append(float(np.mean(gs)))
                i_rand.append(float(np.mean(is_)))
            op[M] = {
                'B': B,
                'gsp_designed': float(np.mean(g_des)),
                'idw_designed': float(np.mean(i_des)),
                'gsp_random': float(np.mean(g_rand)),
                'idw_random': float(np.mean(i_rand)),
                'gsp_beats_idw_same_set': int(sum(a < b for a, b in zip(g_des, i_des))),
            }
        out['operators'][name] = op
        print(f'{name}:')
        for M in [30, 100]:
            v = op[M]
            print(f"  M={M:>3} B={v['B']:>2}: GSP(des)={v['gsp_designed']:.3f} "
                  f"IDW(des)={v['idw_designed']:.3f} | GSP(rand)={v['gsp_random']:.3f} "
                  f"IDW(rand)={v['idw_random']:.3f} | GSP<IDW same-set {v['gsp_beats_idw_same_set']}/7")

    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('saved', OUT)


if __name__ == '__main__':
    main()
