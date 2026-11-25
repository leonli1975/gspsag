"""Learned graph-filter (GCN-style) baseline vs training-free GSP.

Addresses reviewer R2.5 (compare with modern graph-learning methods). We train a linear
graph filter (a GCN without nonlinearity) that maps zero-filled monitor readings to the full
voltage profile,

    x_hat = sum_{k=0}^{K} theta_k (A_hat^k y0),

where y0 is the monitor vector zero-filled at unmonitored buses and A_hat a normalised
adjacency of the physical (admittance) graph. Theta is learned by ridge regression on
training events. Because the parameter count is K+1, the model is data-efficient yet genuinely
graph-based; this isolates "learned graph filtering" from "analytic GSP".

Benchmark: the 4112-event catalogue (local sags), 5-fold cross-validation, M=30, B=10,
designed monitor placement shared by all methods. Compared: GSP (training-free), IDW,
learned graph filter.
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from electrical_model import build_network, DEFAULT_DATA
from experiment_graph_sensitivity import build_operators, laplacian, distance_matrix
from experiment_estimation_designed import greedy_d_optimal, gsp, idw, rmse

BASE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(BASE, '..', 'data', '暂降域数据11-4', 'sag_signals_1028.npz')
OUT = os.path.join(BASE, 'results_learning_baseline.json')
K = 5
RIDGE = 1e-3


def load_voltage_signals(net):
    names, idx = net['bus_names'], net['bus_idx']
    n = len(names)
    bt = pd.read_csv(os.path.join(BASE, 'BUSTOPO.csv'), encoding='utf-8-sig')
    ball = bt['BUSNAME'].astype(str).tolist()
    d = np.load(CATALOG)
    sig, fb = d['signals'], d['fault_buses']
    ck = [j for j, b in enumerate(ball) if b in idx]
    gid = [idx[ball[j]] for j in ck]
    X = sig[:, ck][:, np.argsort(gid)]
    inc = np.array([idx.get(b, -1) >= 0 for b in fb])
    return X[inc], n


def normalised_adjacency(W):
    W = np.array(W, dtype=float)
    np.fill_diagonal(W, 0.0)
    deg = W.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    A = W * dinv[:, None] * dinv[None, :]
    return sp.csr_matrix(A)


def design_matrix(Ahat, y0):
    # column 0 = constant bias (captures the ~1.0 baseline), then Ahat^k y0 for k=0..K
    feats = np.zeros((len(y0), K + 2))
    feats[:, 0] = 1.0
    v = y0.copy()
    for k in range(K + 1):
        feats[:, k + 1] = v
        v = Ahat @ v
    return feats


def train_filter(Ahat, events, S):
    F = len(events)
    n = events.shape[1]
    M = np.zeros((K + 2, K + 2))
    b = np.zeros(K + 2)
    for x in events:
        y0 = np.zeros(n)
        y0[S] = x[S]
        Phi = design_matrix(Ahat, y0)
        M += Phi.T @ Phi
        b += Phi.T @ x
    theta = np.linalg.solve(M + RIDGE * np.eye(K + 2), b)
    return theta


def apply_filter(Ahat, theta, x, S):
    y0 = np.zeros(len(x))
    y0[S] = x[S]
    Phi = design_matrix(Ahat, y0)
    return Phi @ theta


def main():
    net = build_network(
        os.path.join(BASE, 'BUSTOPO.csv'), os.path.join(BASE, 'LINETOPO.csv'),
        os.path.join(BASE, 'TRANSTOPO.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_LINE.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_TRANS.csv'))
    n = net['n']
    X, _ = load_voltage_signals(net)
    ops, kind = build_operators(net)
    W = ops['O2_admittance']
    L = sp.csr_matrix(laplacian(W))
    vals, U = np.linalg.eigh(L.toarray())
    Ahat = normalised_adjacency(W)
    D = distance_matrix(W)

    M, B = 30, 10
    S = greedy_d_optimal(U, B, M)
    nev = len(X)
    rng = np.random.RandomState(0)
    folds = np.array_split(rng.permutation(nev), 5)

    metrics = {'gsp': [], 'idw': [], 'learned': []}
    for f in range(5):
        test = folds[f]
        train = np.concatenate([folds[j] for j in range(5) if j != f])
        theta = train_filter(Ahat, X[train], S)
        for ti in test:
            x = X[ti]
            metrics['gsp'].append(rmse(gsp(U, x, S, B), x))
            metrics['idw'].append(rmse(idw(D, x, S), x))
            metrics['learned'].append(rmse(apply_filter(Ahat, theta, x, S), x))

    out = {
        'N': n, 'n_events': int(nev), 'M': M, 'B': B, 'K': K,
        'gsp': float(np.mean(metrics['gsp'])),
        'idw': float(np.mean(metrics['idw'])),
        'learned': float(np.mean(metrics['learned'])),
        'learned_wins_vs_gsp': int(np.sum(np.array(metrics['learned']) < np.array(metrics['gsp']))),
        'gsp_wins_vs_idw': int(np.sum(np.array(metrics['gsp']) < np.array(metrics['idw']))),
    }
    print(json.dumps(out, indent=2))
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('saved', OUT)


if __name__ == '__main__':
    main()
