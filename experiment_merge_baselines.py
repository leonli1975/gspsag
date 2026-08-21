"""
Merge figB1_kriging + figB2_mlp into a single 1x2 figure (figB3).
Runs only the kriging and MLP baseline experiments, then plots side by side.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent))
from experiment_fullpaper import run_kriging_baseline, run_mlp_gcn_baseline

OUTPUT_DIR = Path(__file__).parent / 'figures'

import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'serif',
})

print("[1/2] Running Kriging baseline...")
res1 = run_kriging_baseline()
print("[2/2] Running MLP baseline...")
res2 = run_mlp_gcn_baseline()

M_list = [3, 5, 10, 15, 20, 30, 50]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

ax1.plot(M_list, res1['gsp'], 'o-', color='#1f77b4', linewidth=1.5,
         label='GSP Sampling (B=5)')
ax1.plot(M_list, res1['kriging_geo'], 'D--', color='#d62728', linewidth=1.2,
         label='Kriging (Geographic)')
ax1.plot(M_list, res1['kriging_spec'], '^--', color='#9467bd', linewidth=1.2,
         label='Kriging (Spectral)')
ax1.plot(M_list, res1['idw'], 's:', color='#ff7f0e', linewidth=1.2,
         label='IDW (Graph dist.)')
ax1.set_xlabel('Number of monitors (M)')
ax1.set_ylabel('RMSE (p.u.)')
ax1.set_title('(a) Kriging baselines')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.plot(M_list, res2['gsp'], 'o-', color='#1f77b4', linewidth=1.5,
         label='GSP Sampling (B=5)')
ax2.plot(M_list, res2['mlp'], 's--', color='#2ca02c', linewidth=1.2,
         label='MLP (spectral features)')
ax2.plot(M_list, res1['idw'], 'D:', color='#ff7f0e', linewidth=1.2,
         label='IDW (graph dist.)')
ax2.set_xlabel('Number of monitors (M)')
ax2.set_ylabel('RMSE (p.u.)')
ax2.set_title('(b) MLP baseline')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'figB3_combined_baselines.pdf')
fig.savefig(OUTPUT_DIR / 'figB3_combined_baselines.png')
print('  Saved figB3_combined_baselines.pdf/png')
