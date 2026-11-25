"""Figure for the designed-sampling estimation result."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, 'figures')
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7})

r = json.load(open(os.path.join(BASE, 'results_estimation_designed.json')))
ops = ['O1_line', 'O2_admittance', 'O3_zbus']
LAB = {'O1_line': 'Line-length', 'O2_admittance': 'Admittance', 'O3_zbus': r'Z$_{bus}$'}

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
for ax, o in zip(axes, ops):
    bm = r['operators'][o]['by_M']
    Ms = sorted(int(m) for m in bm)
    des = [bm[str(m)]['gsp_designed'] for m in Ms]
    rnd = [bm[str(m)]['gsp_random'] for m in Ms]
    idw = [bm[str(m)]['idw'] for m in Ms]
    ax.plot(Ms, des, 'o-', color='#1f77b4', label='GSP (designed)')
    ax.plot(Ms, rnd, 's--', color='#d62728', label='GSP (random)')
    ax.plot(Ms, idw, '^:', color='#2ca02c', label='IDW')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(Ms); ax.set_xticklabels([str(m) for m in Ms], rotation=45, ha='right')
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0.02, 5)
    ax.set_xlabel('monitors $M$')
    ax.set_title(LAB[o])
    ax.grid(True, which='both', alpha=0.2)
axes[0].set_ylabel('RMSE (p.u.)')
axes[0].legend(frameon=False, loc='lower left')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_estimation_designed.pdf'))
fig.savefig(os.path.join(FIG, 'fig_estimation_designed.png'), dpi=150)
print('saved fig_estimation_designed')
