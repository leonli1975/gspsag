"""GSP sag estimation with eigenbasis-designed sampling (positive result).

Motivation: the earlier negative result on physical operators used *random* monitor
placement and a small fixed bandwidth. The physical eigenbases are cluster-localized, so
random sampling is ill-conditioned. Here we design the sampling set on the physical
eigenbasis (greedy D-optimal: pick rows that span the first B eigenvectors) and set the
bandwidth to B ~ M/3. Under this design GSP reconstruction becomes competitive with, and
often better than, inverse-distance interpolation.

For each operator (O1 line-length, O2 admittance, O3 Zbus) and monitor count M:
  - GSP: greedy D-optimal sampling of M nodes for the first B=M//3 eigenvectors, pinv
    reconstruction (deterministic set, no seed).
  - IDW: 1/d^2 interpolation on random monitor sets (5 draws), same M, operator distance.
  - also GSP with RANDOM sampling, to quantify the value of the sampling design.

Saves results_estimation_designed.json.
"""
from __future__ import annotations
import json
import os

import numpy as np
import scipy.sparse as sp

from electrical_model import build_network, DEFAULT_DATA
from experiment_graph_sensitivity import (
    build_operators, laplacian, load_detailed_events, distance_matrix)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'results_estimation_designed.json')
N_IDW = 5


def greedy_d_optimal(U, B, M):
    """Greedy D-optimal row selection for the first B eigenvectors."""
    UF = U[:, :B].copy()
    sel = []
    resid = UF.copy()
    for _ in range(min(M, UF.shape[0])):
        sc = (resid ** 2).sum(1)
        if sel:
            sc[sel] = -1.0
        i = int(np.argmax(sc))
        sel.append(i)
        v = UF[i]
        proj = resid @ v / (v @ v + 1e-12)
        resid = resid - np.outer(proj, v)
    return np.array(sel)


def gsp(U, x, S, B):
    UF = U[:, :B]
    a = np.linalg.pinv(UF[S]) @ x[S]
    return UF @ a


def idw(D, x, S):
    d = D[:, S]
    w = 1.0 / (d ** 2 + 1e-10)
    w /= w.sum(1, keepdims=True)
    return (w * x[S]).sum(1)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    net = build_network(
        os.path.join(BASE, 'BUSTOPO.csv'), os.path.join(BASE, 'LINETOPO.csv'),
        os.path.join(BASE, 'TRANSTOPO.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_LINE.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_TRANS.csv'))
    n = net['n']
    ops, kind = build_operators(net)
    ev = load_detailed_events(net)
    print(f'N={n}, events={len(ev)}')

    res = {'N': n, 'n_idw_draws': N_IDW, 'operators': {}}
    for name, W in ops.items():
        L = sp.csr_matrix(laplacian(W))
        vals, U = np.linalg.eigh(L.toarray())
        D = distance_matrix(W)
        op = {'by_M': {}, 'by_event_M30': {}}
        for M in [20, 30, 50, 100, 200]:
            B = max(1, min(M // 3, 50))
            Sg = greedy_d_optimal(U, B, M)
            g_des, g_rand, idw_rand = [], [], []
            for k, x in ev.items():
                g_des.append(rmse(gsp(U, x, Sg, B), x))
                g_rand.append(np.mean([rmse(gsp(U, x, np.random.RandomState(t).choice(n, M, replace=False), B), x)
                                       for t in range(N_IDW)]))
                idw_rand.append(np.mean([rmse(idw(D, x, np.random.RandomState(100 + t).choice(n, M, replace=False)), x)
                                         for t in range(N_IDW)]))
            op['by_M'][M] = {
                'B': B,
                'gsp_designed': float(np.mean(g_des)),
                'gsp_random': float(np.mean(g_rand)),
                'idw': float(np.mean(idw_rand)),
                'designed_wins': int(sum(a < b for a, b in zip(g_des, idw_rand))),
                'random_wins': int(sum(a < b for a, b in zip(g_rand, idw_rand))),
            }
        # per-event at M=30
        M, B = 30, 10
        Sg = greedy_d_optimal(U, B, M)
        for k, x in ev.items():
            idw_v = np.mean([rmse(idw(D, x, np.random.RandomState(100 + t).choice(n, M, replace=False)), x)
                             for t in range(N_IDW)])
            op['by_event_M30'][str(k)] = {
                'n_affected': int((x < 0.85).sum()),
                'gsp_designed': rmse(gsp(U, x, Sg, B), x),
                'gsp_random': float(np.mean([rmse(gsp(U, x, np.random.RandomState(t).choice(n, M, replace=False), B), x)
                                             for t in range(N_IDW)])),
                'idw': float(idw_v),
            }
        res['operators'][name] = op
        rows = op['by_M']
        print(f'  {name}: ' + ' '.join(
            f"M{M}(B{rows[M]['B']}) d={rows[M]['gsp_designed']:.3f}/r={rows[M]['gsp_random']:.3f}/"
            f"i={rows[M]['idw']:.3f} w{rows[M]['designed_wins']}" for M in [30, 100]))

    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    print('saved', OUT)


if __name__ == '__main__':
    main()
