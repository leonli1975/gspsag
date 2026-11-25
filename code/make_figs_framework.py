"""Framework figures: (a) design map, (b) MODF procedure flowchart."""
import json
import os

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from electrical_model import build_network, DEFAULT_DATA
from experiment_localization import load_data, hop_index

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, 'figures')
plt.rcParams.update({'font.size': 8})

# ---------- coverage vs M (random placement) ----------
net = build_network(os.path.join(BASE, 'BUSTOPO.csv'), os.path.join(BASE, 'LINETOPO.csv'),
                    os.path.join(BASE, 'TRANSTOPO.csv'),
                    os.path.join(BASE, DEFAULT_DATA, 'DT_LINE.csv'),
                    os.path.join(BASE, DEFAULT_DATA, 'DT_TRANS.csv'))
D, fid, ft, N = load_data(net)
hop, G = hop_index(net)
Ms = [10, 20, 30, 50, 100, 300, 1025]
cov = []
for M in Ms:
    vals = []
    for t in range(10):
        S = np.random.RandomState(t).choice(N, M, replace=False)
        vals.append(np.mean([min(hop[f][s] for s in S) <= 2 for f in fid]))
    cov.append(np.mean(vals))

est = json.load(open(os.path.join(BASE, 'results_estimation_designed.json')))
loc = json.load(open(os.path.join(BASE, 'results_localization.json')))['results']
eM = sorted(int(m) for m in est['operators']['O2_admittance']['by_M'])
eRM = [est['operators']['O2_admittance']['by_M'][str(m)]['gsp_designed'] for m in eM]
lM = sorted(int(m) for m in loc)
lAcc = [loc[str(m)]['le3']['mean'] for m in lM]

# ---------- design map ----------
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
ax[0].plot(eM, eRM, 'o-', color='#1f77b4')
ax[0].axhline(0.05, color='gray', ls='--', lw=0.9)
ax[0].text(eM[0], 0.052, 'target 0.05 p.u.', fontsize=6, color='gray')
ax[0].set_xscale('log'); ax[0].set_xlabel('monitors $M$')
ax[0].set_xticks(eM); ax[0].set_xticklabels([str(m) for m in eM]); ax[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax[0].set_ylabel('estimation RMSE (p.u.)')
ax[0].set_title('(a) Estimation (admittance, designed)')
ax[0].grid(True, which='both', alpha=0.2)

ax[1].plot(lM, np.array(lAcc) * 100, 's-', color='#2ca02c', label='within 3 hops')
ax[1].plot(Ms, np.array(cov) * 100, '^--', color='#b35c00', label='faults with a monitor $\\leq$2 hops')
ax[1].axhline(50, color='gray', ls='--', lw=0.9)
ax[1].set_xscale('log'); ax[1].set_xlabel('monitors $M$')
ax[1].set_xticks(Ms); ax[1].set_xticklabels([str(m) for m in Ms]); ax[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax[1].set_ylabel('percent')
ax[1].set_title('(b) Localization and coverage')
ax[1].legend(frameon=False, fontsize=6)
ax[1].grid(True, which='both', alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_design_map.pdf'))
fig.savefig(os.path.join(FIG, 'fig_design_map.png'), dpi=160)
plt.close(fig)

# ---------- flowchart ----------
def box(ax, xy, w, h, text, fc, ec, fs=7.5, bold=False):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.10',
                                lw=1.0, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal')

def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=10, lw=1.0,
                                 color='#444444', shrinkA=1, shrinkB=1))

fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
box(ax, (2.0, 8.7), 6.0, 1.0,
    'Inputs: network + branch impedances; tasks; targets $(\\epsilon, k, p)$', '#e8f0fb', '#2b5fa8', bold=True)
box(ax, (2.0, 7.2), 6.0, 0.9, 'Build physical operator: admittance or $\\mathbf{Z}_{bus}$', '#eef2f7', '#4a6fa5')
box(ax, (0.3, 4.6), 4.3, 1.7,
    'Estimation rule\n$B\\approx10$; $M\\geq3B$\nD-optimal placement\nread RMSE$(M)$ from the design map',
    '#eaf6ea', '#2e7d32')
box(ax, (5.4, 4.6), 4.3, 1.7,
    'Localization rule\naccuracy and coverage from\nthe design map\nfingerprint matching',
    '#fdf1e6', '#b35c00')
box(ax, (1.2, 2.5), 7.6, 1.3,
    'Both tasks: the same D-optimal set serves both\nverify each target on the design maps',
    '#f3eaf7', '#6a3d9a')
box(ax, (2.0, 0.5), 6.0, 1.1,
    'Output: operator, monitor count, locations,\npredicted estimation error and localization resolution', '#e8f0fb', '#2b5fa8', bold=True)
arrow(ax, (5.0, 8.7), (5.0, 8.1))
arrow(ax, (4.0, 7.2), (2.5, 6.3))
arrow(ax, (6.0, 7.2), (7.5, 6.3))
arrow(ax, (2.5, 4.6), (4.0, 3.8))
arrow(ax, (7.5, 4.6), (6.0, 3.8))
arrow(ax, (5.0, 2.5), (5.0, 1.6))
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(FIG, 'fig_framework_flowchart.pdf'))
fig.savefig(os.path.join(FIG, 'fig_framework_flowchart.png'), dpi=160)
print('saved fig_design_map and fig_framework_flowchart')
