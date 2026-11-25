"""Graph-operator sensitivity of GSP voltage-sag estimation (L2 core experiment).

Three operators on the SAME 1025-bus regional network:
  O1_line        W = 1/L for lines, fixed W = 10 for transformers  (legacy practice)
  O2_admittance  W = series susceptance in siemens                 (physical)
  O3_zbus        W = |Zbus_ij|                                     (propagation kernel)

For each operator:
  - edge-weight structure (line vs transformer medians)
  - bandlimitedness of the deviation signal per detailed event (cumulative GFT energy)
  - reconstruction RMSE, GSP (B=5) vs IDW with 1/d^2 electrical weights,
    averaged over 20 random monitor sets at M in {10,20,30,50,100}

Outputs results_graph_sensitivity.json.
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from electrical_model import build_network, DEFAULT_DATA

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results_graph_sensitivity.json')
B_FIXED = 5
N_SEEDS = 20


def _read(p):
    for enc in ('utf-8-sig', 'gbk', 'gb2312', 'gb18030'):
        try:
            return pd.read_csv(p, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(p)


def build_operators(net):
    names, idx = net['bus_names'], net['bus_idx']
    n = len(names)
    lt = _read(os.path.join(BASE, 'LINETOPO.csv'))
    tt = _read(os.path.join(BASE, 'TRANSTOPO.csv'))
    W1 = np.zeros((n, n)); W2 = np.zeros((n, n))
    kind = np.zeros((n, n), dtype=int)
    lp = W1         # line-only adjacency for kind tagging
    for _, r in lt.iterrows():
        s, e = r['STARBUSNAME'], r['ENDBUSNAME']
        if s in idx and e in idx and s != e:
            L = float(r['LINELENGTH']) if np.isfinite(r['LINELENGTH']) else 1.0
            L = max(L, 0.01); i, j = idx[s], idx[e]
            W1[i, j] = W1[j, i] = 1.0 / L
            kind[i, j] = kind[j, i] = 1
    for i, j, yc, k in net['branches']:
        b = -np.imag(yc)
        if b <= 0:
            b = abs(yc)
        W2[i, j] += b; W2[j, i] += b
        if k == 'trans':
            kind[i, j] = kind[j, i] = 2
    for _, t in tt.iterrows():
        for c1, c2 in [('HIGBUSNAME', 'MIDBUSNAME'), ('HIGBUSNAME', 'LOWBUSNAME'),
                       ('MIDBUSNAME', 'LOWBUSNAME')]:
            bi, bj = t.get(c1), t.get(c2)
            if bi in idx and bj in idx and bi != bj:
                i, j = idx[bi], idx[bj]
                W1[i, j] = W1[j, i] = 10.0
                kind[i, j] = kind[j, i] = 2
    W3 = np.abs(net['Zbus']); np.fill_diagonal(W3, 0.0)
    return {'O1_line': W1, 'O2_admittance': W2, 'O3_zbus': W3}, kind


def laplacian(W):
    return np.diag(W.sum(1)) - W


def load_detailed_events(net):
    idx = net['bus_idx']; n = len(net['bus_names'])
    sag = _read(os.path.join(BASE, 'SAG_BUSVOLTAGE.csv'))
    events = {}
    for ev in sag['EVENT_ID'].unique():
        df = sag[sag['EVENT_ID'] == ev]
        x = np.ones(n)
        for _, r in df.iterrows():
            bn = str(r['BUSNAME']).strip()
            if bn in idx:
                x[idx[bn]] = float(r['VA'])
        events[str(ev)] = x
    return events


def distance_matrix(W):
    Wc = W.copy(); np.fill_diagonal(Wc, 0.0)
    C = np.full_like(Wc, np.inf)
    m = Wc > 0
    C[m] = 1.0 / Wc[m]
    np.fill_diagonal(C, 0.0)
    D = dijkstra(sp.csr_matrix(C), directed=False)
    D[~np.isfinite(D)] = 1e6
    return D


def gsp_reconstruct(U, x, S, B):
    UF = U[:, :B]
    a = np.linalg.pinv(UF[S]) @ x[S]
    return UF @ a


def idw_reconstruct(D, x, S):
    d = D[:, S]
    w = 1.0 / (d ** 2 + 1e-10)
    w /= w.sum(1, keepdims=True)
    return (w * x[S]).sum(1)


def main():
    net = build_network(
        os.path.join(BASE, 'BUSTOPO.csv'), os.path.join(BASE, 'LINETOPO.csv'),
        os.path.join(BASE, 'TRANSTOPO.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_LINE.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_TRANS.csv'))
    n = net['n']
    ops, kind = build_operators(net)
    events = load_detailed_events(net)
    print(f'N={n}, events={len(events)}')

    results = {'N': n, 'B_fixed': B_FIXED, 'n_seeds': N_SEEDS, 'operators': {}}
    for name, W in ops.items():
        L = laplacian(W)
        vals, U = np.linalg.eigh(L)
        D = distance_matrix(W)
        line_w = W[kind == 1]; trans_w = W[kind == 2]
        stats = {
            'lambda2': float(vals[1]), 'lambdaN': float(vals[-1]),
            'line_median': float(np.median(line_w[line_w > 0])) if (line_w > 0).any() else 0.0,
            'trans_median': float(np.median(trans_w[trans_w > 0])) if (trans_w > 0).any() else 0.0,
        }
        band = {}
        for ev, x in events.items():
            d = 1.0 - x
            c = d @ U; e = c ** 2
            cum = np.cumsum(e) / (e.sum() + 1e-15)
            band[ev] = {'k1': float(cum[0]), 'k5': float(cum[4]),
                        'k10': float(cum[9]), 'k20': float(cum[19]),
                        'n_affected': int((x < 0.85).sum())}
        stats['bandlimitedness'] = band
        est = {}
        for M in [10, 20, 30, 50, 100]:
            per_ev = {}
            for ev, x in events.items():
                gs, ds = [], []
                for s in range(N_SEEDS):
                    S = np.random.RandomState(s).choice(n, min(M, n), replace=False)
                    gs.append(np.sqrt(np.mean((gsp_reconstruct(U, x, S, min(B_FIXED, M)) - x) ** 2)))
                    ds.append(np.sqrt(np.mean((idw_reconstruct(D, x, S) - x) ** 2)))
                per_ev[ev] = {'gsp': float(np.mean(gs)), 'idw': float(np.mean(ds)),
                              'gsp_std': float(np.std(gs)), 'idw_std': float(np.std(ds))}
            est[M] = per_ev
        stats['estimation'] = est
        results['operators'][name] = stats
        print(f'  {name}: lambda2={stats["lambda2"]:.2e} line={stats["line_median"]:.3f} '
              f'trans={stats["trans_median"]:.3f}')

    with open(RESULTS, 'w') as f:
        json.dump(results, f, indent=2)
    print('saved', RESULTS)


if __name__ == '__main__':
    main()
