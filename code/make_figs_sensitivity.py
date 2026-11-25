"""Generate figures for the graph-operator sensitivity study (L2)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8})

COL = {'O1_line': '#1f77b4', 'O2_admittance': '#ff7f0e', 'O3_zbus': '#2ca02c'}
LAB = {'O1_line': 'Line-length (legacy)', 'O2_admittance': 'Branch admittance',
       'O3_zbus': 'Zbus kernel'}

res = json.load(open(os.path.join(BASE, 'results_graph_sensitivity.json')))
ops = list(res['operators'].keys())


def fig_edge_structure():
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(ops)); w = 0.36
    line = [res['operators'][o]['line_median'] for o in ops]
    trans = [res['operators'][o]['trans_median'] for o in ops]
    ax.bar(x - w/2, line, w, label='line', color='#4c72b0')
    ax.bar(x + w/2, trans, w, label='transformer', color='#c44e52')
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels([LAB[o] for o in ops], rotation=12)
    ax.set_ylabel('median edge weight (log)')
    ax.legend(frameon=False)
    ax.set_title('(a) Edge-weight structure')
    for i, (l, t) in enumerate(zip(line, trans)):
        ax.text(i - w/2, l*1.1, f'{l:.2f}', ha='center', fontsize=6)
        ax.text(i + w/2, t*1.1, f'{t:.2f}', ha='center', fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_ops_edges.pdf'))
    fig.savefig(os.path.join(FIG, 'fig_ops_edges.png'), dpi=150)
    plt.close(fig)


def fig_bandlimitedness():
    # cumulative energy curves: need per-k up to 20. We stored k1,k5,k10,k20 only.
    deep = ['2019070701', '2019071001']
    shallow = ['2019070803']
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ks = [1, 5, 10, 20]
    for o in ops:
        b = res['operators'][o]['bandlimitedness']
        for ev in b:
            if ev in deep:
                y = [b[ev]['k1'], b[ev]['k5'], b[ev]['k10'], b[ev]['k20']]
                ax.plot(ks, np.array(y)*100, 'o-', color=COL[o], alpha=0.9,
                        label=LAB[o] if ev == deep[0] else None, lw=1.2)
    ax.set_xlabel('number of GFT coefficients $k$')
    ax.set_ylabel('cumulative energy (%)')
    ax.set_ylim(0, 105)
    ax.set_title('(b) Bandlimitedness (deep events)')
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_ops_bandlimitedness.pdf'))
    fig.savefig(os.path.join(FIG, 'fig_ops_bandlimitedness.png'), dpi=150)
    plt.close(fig)


def fig_estimation():
    M = 30
    events = sorted(res['operators'][ops[0]]['estimation'][str(M)].keys())
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    x = np.arange(len(events)); w = 0.26
    for oi, o in enumerate(ops):
        est = res['operators'][o]['estimation'][str(M)]
        gsp = [est[e]['gsp'] for e in events]
        idw = [est[e]['idw'] for e in events]
        ax.bar(x + (oi - 1)*w, gsp, w*0.9, label=f'{LAB[o]} GSP',
               color=COL[o], alpha=0.85)
        ax.bar(x + (oi - 1)*w, idw, w*0.9, facecolor='none', edgecolor=COL[o],
               hatch='///', linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels([e[-4:] for e in events], rotation=0)
    ax.set_ylabel(f'RMSE (p.u.), M={M}')
    ax.set_title('(c) GSP (filled) vs IDW (hatched) by operator')
    # split legend: operator colours
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COL[o], label=LAB[o]) for o in ops],
              frameon=False, ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_ops_estimation.pdf'))
    fig.savefig(os.path.join(FIG, 'fig_ops_estimation.png'), dpi=150)
    plt.close(fig)


def fig_localization():
    p = os.path.join(BASE, 'results_localization.json')
    if not os.path.exists(p):
        return
    r = json.load(open(p))['results']
    Ms = sorted(int(m) for m in r.keys())
    le3 = [r[str(m)]['le3']['mean']*100 for m in Ms]
    le1 = [r[str(m)]['le1']['mean']*100 for m in Ms]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(Ms, le3, 'o-', color='#1f77b4', label='within 3 hops')
    ax.plot(Ms, le1, 's--', color='#ff7f0e', label='within 1 hop')
    ax.set_xscale('log'); ax.set_xlabel('number of monitors M')
    ax.set_ylabel('localization accuracy (%)')
    ax.set_ylim(0, 100); ax.legend(frameon=False)
    ax.set_title('(d) Localization vs monitoring')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_localization_vs_M.pdf'))
    fig.savefig(os.path.join(FIG, 'fig_localization_vs_M.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    fig_edge_structure()
    fig_bandlimitedness()
    fig_estimation()
    fig_localization()
    print('figures written to', FIG)
