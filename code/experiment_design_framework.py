"""E3: does one (estimation-optimised) monitor set also serve localization?

Compares localization accuracy using the greedy-D-optimal monitor set (chosen for
spectral observability of the first B eigenvectors) against random sets of the same size,
on the 4112-event catalogue with five-fold cross-validation. Also reports the spatial
coverage (fraction of faults with a monitor within 2 hops). This decides whether the framework
can recommend a single monitor set for both tasks or must allocate monitors per task.
"""
import json
import os

import numpy as np
import scipy.sparse as sp

from electrical_model import build_network, DEFAULT_DATA
from experiment_graph_sensitivity import build_operators, laplacian
from experiment_localization import load_data, hop_index
from experiment_estimation_designed import greedy_d_optimal

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'results_design_framework.json')
N_RAND = 10
N_FOLDS = 5


def eval_loc(D, fid, hop, N, S):
    rng = np.random.RandomState(0)
    order = rng.permutation(len(fid))
    folds = np.array_split(order, N_FOLDS)
    errs, ranks, vis = [], [], []
    for k in range(N_FOLDS):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
        db = np.unique(fid[train])
        Dm = np.zeros((N, len(db)))
        for c, b in enumerate(db):
            Dm[:, c] = D[train[fid[train] == b]].mean(0)
        DmS = Dm[S, :]
        DmSn = DmS / (np.linalg.norm(DmS, axis=0, keepdims=True) + 1e-12)
        for ti in test:
            y = D[ti, S]
            yn = y / (np.linalg.norm(y) + 1e-12)
            sim = DmSn.T @ yn
            pred = db[int(np.argmax(sim))]
            pos = np.where(db[np.argsort(-sim)] == fid[ti])[0]
            ranks.append(int(pos[0]) + 1 if len(pos) else len(db))
            errs.append(hop[pred].get(fid[ti], 999))
            vis.append(min(hop[fid[ti]][s] for s in S) <= 2)
    errs = np.array(errs)
    return {'le1': float(np.mean(errs <= 1)), 'le3': float(np.mean(errs <= 3)), 'mean_rank': float(np.mean(ranks)),
            'coverage': float(np.mean(vis))}


def main():
    net = build_network(
        os.path.join(BASE, 'BUSTOPO.csv'), os.path.join(BASE, 'LINETOPO.csv'),
        os.path.join(BASE, 'TRANSTOPO.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_LINE.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_TRANS.csv'))
    D, fid, ft, N = load_data(net)
    hop, G = hop_index(net)
    ops, kind = build_operators(net)
    W = ops['O2_admittance']
    L = sp.csr_matrix(laplacian(W))
    vals, U = np.linalg.eigh(L.toarray())

    out = {'N': N, 'n_events': int(len(fid)), 'n_rand': N_RAND, 'by_M': {}}
    for M in [30, 100, 300]:
        B = min(M // 3, 50)
        Sdes = greedy_d_optimal(U, B, M)
        des = eval_loc(D, fid, hop, N, Sdes)
        rand = []
        for t in range(N_RAND):
            S = np.random.RandomState(t).choice(N, M, replace=False)
            rand.append(eval_loc(D, fid, hop, N, S))
        agg = {k: float(np.mean([r[k] for r in rand])) for k in rand[0]}
        out['by_M'][M] = {'B': B, 'designed': des, 'random': agg}
        print(f"M={M:>3} B={B:>2}: designed le3={des['le3']*100:.1f}% rank={des['mean_rank']:.0f} "
              f"cov={des['coverage']*100:.1f}% | random le3={agg['le3']*100:.1f}% "
              f"rank={agg['mean_rank']:.0f} cov={agg['coverage']*100:.1f}%")
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('saved', OUT)


if __name__ == '__main__':
    main()
