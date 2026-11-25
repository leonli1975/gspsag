"""Figure 4 of stage03: actual vs estimated voltage (designed sampling, admittance graph)."""
import os

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from electrical_model import build_network, DEFAULT_DATA
from experiment_graph_sensitivity import build_operators, laplacian, load_detailed_events
from experiment_estimation_designed import greedy_d_optimal, gsp, rmse

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    net = build_network(
        os.path.join(BASE, 'BUSTOPO.csv'), os.path.join(BASE, 'LINETOPO.csv'),
        os.path.join(BASE, 'TRANSTOPO.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_LINE.csv'),
        os.path.join(BASE, DEFAULT_DATA, 'DT_TRANS.csv'))
    ops, kind = build_operators(net)
    ev = load_detailed_events(net)
    W = ops['O2_admittance']
    L = sp.csr_matrix(laplacian(W))
    vals, U = np.linalg.eigh(L.toarray())
    M, B = 30, 10
    S = greedy_d_optimal(U, B, M)

    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    for a, (k, title) in zip(ax, [('2019070701', '(a) wide-area (970 buses)'),
                                  ('2019070803', '(b) localized (10 buses)')]):
        x = ev[k]
        xh = gsp(U, x, S, B)
        a.scatter(x, xh, s=6, alpha=0.5, color='#1f77b4', edgecolors='none')
        a.plot([0, 1], [0, 1], 'k--', lw=0.8)
        a.set_xlabel('actual voltage (p.u.)')
        a.set_title(title)
        a.text(0.04, 0.88, f'RMSE={rmse(xh, x):.3f} p.u.', transform=a.transAxes)
        a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02); a.grid(True, alpha=0.2)
    ax[0].set_ylabel('estimated voltage (p.u.)')
    fig.tight_layout()
    out = os.path.join(BASE, 'figures')
    fig.savefig(os.path.join(out, 'fig_recon_scatter.pdf'))
    fig.savefig(os.path.join(out, 'fig_recon_scatter.png'), dpi=150)
    print('saved fig_recon_scatter')


if __name__ == '__main__':
    main()
