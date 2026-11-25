"""Framework practicality: compare task-specific placements against common baselines.

For a fixed monitor budget, compares four placements:
  - designed   : greedy D-optimal on the first B eigenvectors (framework, estimation)
  - maxdegree  : highest-degree buses (common heuristic)
  - geospread  : farthest-point sampling on (lon, lat) (geographic spread / coverage)
  - random     : uniform random
on (i) estimation RMSE (7 detailed events, admittance graph) and (ii) localization accuracy and
coverage (4112-event catalogue, 5-fold). This shows that no single universal placement is best,
which is the practical case for the task-matched framework.
"""
import json
import os

import numpy as np
import scipy.sparse as sp

from electrical_model import build_network, DEFAULT_DATA
from experiment_graph_sensitivity import build_operators, laplacian, load_detailed_events
from experiment_localization import load_data, hop_index
from experiment_estimation_designed import greedy_d_optimal, gsp
from experiment_design_framework import eval_loc

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'results_framework_baselines.json')
N_RAND = 8


def maxdegree_placement(G, M):
    deg = np.array([G.degree(i) for i in range(G.number_of_nodes())])
    return np.argsort(-deg)[:M]


def geo_spread(coords, M, seed=0):
    n = len(coords)
    rng = np.random.RandomState(seed)
    sel = [int(rng.randint(n))]
    d = np.linalg.norm(coords - coords[sel[0]], axis=1)
    for _ in range(M - 1):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(coords - coords[i], axis=1))
    return np.array(sel)


def main():
    net = build_network(os.path.join(BASE, 'BUSTOPO.csv'), os.path.join(BASE, 'LINETOPO.csv'),
                        os.path.join(BASE, 'TRANSTOPO.csv'),
                        os.path.join(BASE, DEFAULT_DATA, 'DT_LINE.csv'),
                        os.path.join(BASE, DEFAULT_DATA, 'DT_TRANS.csv'))
    N = net['n']
    ops, kind = build_operators(net)
    W = ops['O2_admittance']
    L = sp.csr_matrix(laplacian(W)); vals, U = np.linalg.eigh(L.toarray())
    ev = load_detailed_events(net)
    D, fid, ft, _ = load_data(net)
    hop, G = hop_index(net)
    import pandas as pd
    bt = pd.read_csv(os.path.join(BASE, 'BUSTOPO.csv'), encoding='utf-8-sig')
    idx = net['bus_idx']
    coords = np.array([[float(bt.loc[idx[n], 'LONGITUDE']), float(bt.loc[idx[n], 'LATITUDE'])]
                       for n in net['bus_names']])

    out = {'N': N, 'n_rand': N_RAND, 'by_M': {}}
    for M in [30, 100]:
        B = min(M // 3, 50)
        Sd = greedy_d_optimal(U, B, M)
        Smd = maxdegree_placement(G, M)
        Sgeo = geo_spread(coords, M)
        placements = {'designed': Sd, 'maxdegree': Smd, 'geospread': Sgeo}

        row = {}
        for name, S in placements.items():
            est = np.mean([np.sqrt(np.mean((gsp(U, x, S, B) - x) ** 2)) for x in ev.values()])
            loc = eval_loc(D, fid, hop, N, S)
            row[name] = {'est_rmse': float(est), 'loc_le1': loc['le1'], 'loc_le3': loc['le3'],
                         'loc_coverage': loc['coverage'], 'loc_rank': loc['mean_rank']}
        # random baseline
        r = []
        for t in range(N_RAND):
            S = np.random.RandomState(t).choice(N, M, replace=False)
            est = np.mean([np.sqrt(np.mean((gsp(U, x, S, B) - x) ** 2)) for x in ev.values()])
            loc = eval_loc(D, fid, hop, N, S)
            r.append({'est_rmse': est, 'loc_le1': loc['le1'], 'loc_le3': loc['le3'], 'loc_coverage': loc['coverage'],
                      'loc_rank': loc['mean_rank']})
        row['random'] = {k: float(np.mean([x[k] for x in r])) for k in r[0]}
        out['by_M'][M] = {'B': B, 'placements': row}
        print(f'M={M} B={B}')
        for name in ['designed', 'maxdegree', 'geospread', 'random']:
            v = row[name]
            print(f"  {name:10s} est={v['est_rmse']:9.3f} le1={v['loc_le1']*100:5.1f}% le3={v['loc_le3']*100:5.1f}% cov={v['loc_coverage']*100:5.1f}%")
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('saved', OUT)


if __name__ == '__main__':
    main()
