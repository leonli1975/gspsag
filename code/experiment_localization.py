"""Stable, reproducible voltage-sag source localization on the regional grid.

Protocol
--------
- Network: physical electrical model (electrical_model.build_network), 1025 buses.
- Data: 4112-event BPA catalog (1028 fault locations x 4 fault types).
- Signal: normalised deviation pattern d(i) = (1 - V_i) / max_j(1 - V_j), which
  removes the event amplitude and depends on fault location.
- Dictionary: for each candidate fault bus f, the *mean* normalised pattern of
  its training events (data-calibrated spatial fingerprint).
- Estimator: scale-invariant matched filter -- cosine similarity between the
  observed monitor pattern and each dictionary column; argmax = predicted bus.
- Evaluation: 5-fold cross-validation over events (fixed seed), reporting
  exact / <=1 hop / <=3 hop accuracy, mean rank of the true bus, and the
  size of the <=3-hop candidate region.

Baselines: deepest monitored bus, graph centroid of the affected set.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import networkx as nx

from electrical_model import build_network, DEFAULT_DATA

BASE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(BASE, '..', 'data', '暂降域数据11-4', 'sag_signals_1028.npz')
RESULTS = os.path.join(BASE, 'results_localization.json')


def load_data(net):
    names, idx = net['bus_names'], net['bus_idx']
    N = len(names)
    bt = pd.read_csv(os.path.join(BASE, 'BUSTOPO.csv'), encoding='utf-8-sig')
    ball = bt['BUSNAME'].astype(str).tolist()

    d = np.load(CATALOG)
    sig, fb, ft = d['signals'], d['fault_buses'], d['fault_types']

    ck = [j for j, b in enumerate(ball) if b in idx]
    gid = [idx[ball[j]] for j in ck]
    X = sig[:, ck][:, np.argsort(gid)]
    fid = np.array([idx.get(b, -1) for b in fb])
    inc = fid >= 0
    X, fid, ft = X[inc], fid[inc], ft[inc]

    dev = 1.0 - X
    mx = dev.max(axis=1, keepdims=True)
    mx[mx < 1e-9] = 1.0
    D = dev / mx          # normalised deviation in [0,1], fault bus ~1
    return D, fid, ft, N


def hop_index(net):
    G = nx.Graph()
    for i, j, yc, kind in net['branches']:
        G.add_edge(i, j)
    return dict(nx.all_pairs_shortest_path_length(G)), G


def evaluate(D, fid, hop, N, monitor_sets, n_folds=5, seed=0, hops_region=3):
    """5-fold CV localization for each monitor count in monitor_sets."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(fid))
    folds = np.array_split(order, n_folds)

    results = {}
    for M in monitor_sets:
        fold_metrics = []
        for k in range(n_folds):
            test = folds[k]
            train = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            # build data-calibrated dictionary on training events
            dict_buses = np.unique(fid[train])
            Dm = np.zeros((N, len(dict_buses)))
            for c, b in enumerate(dict_buses):
                rows = train[fid[train] == b]
                pat = D[rows].mean(axis=0)
                Dm[:, c] = pat
            # per-test monitor subset (fixed seed for pairing)
            mrng = np.random.RandomState(seed * 1000 + M)
            S = mrng.choice(N, size=min(M, N), replace=False)
            DmS = Dm[S, :]
            # normalise columns / rows for cosine
            DmS_n = DmS / (np.linalg.norm(DmS, axis=0, keepdims=True) + 1e-12)

            errs, ranks, cand_sizes = [], [], []
            deep_errs, cent_errs = [], []
            for ti in test:
                y = D[ti, S]
                yn = y / (np.linalg.norm(y) + 1e-12)
                sim = DmS_n.T @ yn
                order_c = np.argsort(-sim)
                pred_c = dict_buses[order_c[0]]
                true_b = fid[ti]
                # rank of the true bus among candidates
                pos = np.where(dict_buses[order_c] == true_b)[0]
                rank = int(pos[0]) + 1 if len(pos) else len(dict_buses)
                d = hop[pred_c].get(true_b, 999)
                errs.append(d)
                ranks.append(rank)
                # candidate region size: buses within hops_region of predicted bus
                cand_sizes.append(sum(1 for nb in hop[pred_c].values()
                                      if nb <= hops_region))
                # baseline: deepest monitored bus
                pred_deep = S[int(np.argmax(y))]
                deep_errs.append(hop[pred_deep].get(true_b, 999))
                # baseline: centroid (graph median) of monitored buses above 0.3
                aff = S[y > 0.3]
                if len(aff) == 0:
                    aff = S
                cent = min(S, key=lambda s: sum(hop[s].get(a, 999) for a in aff))
                cent_errs.append(hop[cent].get(true_b, 999))

            errs = np.array(errs); ranks = np.array(ranks)
            fold_metrics.append({
                'exact': float(np.mean(errs == 0)),
                'le1': float(np.mean(errs <= 1)),
                'le3': float(np.mean(errs <= 3)),
                'median_hop': float(np.median(errs)),
                'mean_rank': float(ranks.mean()),
                'top5': float(np.mean(ranks <= 5)),
                'top10': float(np.mean(ranks <= 10)),
                'cand_region': float(np.mean(cand_sizes)),
                'deep_le3': float(np.mean(np.array(deep_errs) <= 3)),
                'cent_le3': float(np.mean(np.array(cent_errs) <= 3)),
            })
        agg = {key: {'mean': float(np.mean([fm[key] for fm in fold_metrics])),
                     'std': float(np.std([fm[key] for fm in fold_metrics]))}
               for key in fold_metrics[0]}
        results[M] = agg
    return results


def main():
    net = build_network(
        os.path.join(BASE, 'BUSTOPO.csv'),
        os.path.join(BASE, 'LINETOPO.csv'),
        os.path.join(BASE, 'TRANSTOPO.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_LINE.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_TRANS.csv'),
    )
    D, fid, ft, N = load_data(net)
    hop, G = hop_index(net)
    print(f'network N={N}, events={len(fid)}, fault buses={len(np.unique(fid))}')

    # candidate region size for reference
    region3 = np.mean([sum(1 for v in hop[b].values() if v <= 3) for b in np.unique(fid)])
    print(f'mean <=3-hop region around a fault bus: {region3:.1f} buses ({region3/N*100:.1f}% of N)')

    monitor_sets = [10, 20, 30, 50, 100, 300, 1025]
    res = evaluate(D, fid, hop, N, monitor_sets, n_folds=5, seed=0)
    print('\nM     exact    <=1     <=3     medRank  candRegion  deep<=3  cent<=3')
    for M in monitor_sets:
        r = res[M]
        print(f"{M:>4}  {r['exact']['mean']*100:5.1f}  {r['le1']['mean']*100:5.1f}  "
              f"{r['le3']['mean']*100:5.1f}  {r['mean_rank']['mean']:6.1f}  "
              f"{r['cand_region']['mean']:8.1f}  {r['deep_le3']['mean']*100:6.1f}  "
              f"{r['cent_le3']['mean']*100:6.1f}")
    out = {'region3_buses': region3, 'N': N, 'n_events': int(len(fid)),
           'results': res}
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=2)
    print('\nsaved', RESULTS)


if __name__ == '__main__':
    main()
