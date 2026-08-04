"""
Generate Theorem 1 sparsity validation figure
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Results from the experiment
events = ['0701', '0801', '0802', '0803', '0804', '1001']
sparsity = [167.0, 209.5, 94.1, 20.8, 150.0, 25.8]
concentration = [88.0, 98.3, 78.4, 100.0, 99.3, 33.4]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Sparsity ratio
colors_s = ['#2ca02c' if v > 100 else '#ff7f0e' if v > 50 else '#d62728' for v in sparsity]
bars1 = axes[0].bar(events, sparsity, color=colors_s, alpha=0.85, edgecolor='black', linewidth=0.5)
axes[0].axhline(y=100, color='green', ls='--', alpha=0.4, linewidth=1.5, label='Strong (100x)')
axes[0].axhline(y=50, color='orange', ls='--', alpha=0.4, linewidth=1.5, label='Moderate (50x)')
axes[0].set_ylabel('Sparsity Ratio', fontsize=11)
axes[0].set_title('(a) Sparsity: max |L$\\delta$| / mean |L$\\delta$|', fontsize=11)
axes[0].legend(fontsize=9, loc='lower left')
axes[0].set_yscale('log')
axes[0].grid(axis='y', alpha=0.2)

# Add values on bars
for bar, val in zip(bars1, sparsity):
    axes[0].text(bar.get_x() + bar.get_width()/2., val * 1.0,
                f'{val:.0f}$\\times$', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Concentration
colors_c = ['#2ca02c' if v > 80 else '#ff7f0e' if v > 50 else '#d62728' for v in concentration]
bars2 = axes[1].bar(events, [c/100 for c in concentration], color=colors_c, alpha=0.85, edgecolor='black', linewidth=0.5)
axes[1].axhline(y=0.8, color='green', ls='--', alpha=0.4, linewidth=1.5, label='High (80%)')
axes[1].axhline(y=0.5, color='orange', ls='--', alpha=0.4, linewidth=1.5, label='Moderate (50%)')
axes[1].set_ylabel('Fraction of $\\|\\mathbf{L}\\boldsymbol{\\delta}\\|^2_2$', fontsize=11)
axes[1].set_title('(b) Energy in Top-3 Buses', fontsize=11)
axes[1].legend(fontsize=9, loc='lower left')
axes[1].set_ylim(0, 1.15)
axes[1].grid(axis='y', alpha=0.2)

for bar, val in zip(bars2, concentration):
    axes[1].text(bar.get_x() + bar.get_width()/2., val/100 + 0.03,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Mean annotations
mean_s = np.mean(sparsity)
mean_c = np.mean(concentration)
axes[0].text(0.98, 0.05, f'Mean: {mean_s:.0f}$\\times$', transform=axes[0].transAxes,
            ha='right', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1].text(0.98, 0.05, f'Mean: {mean_c:.0f}%', transform=axes[1].transAxes,
            ha='right', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# x-axis label
for ax in axes:
    ax.set_xlabel('Event', fontsize=11)
plt.tight_layout()
plt.savefig('figures/fig_theorem1_validation_summary.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures/fig_theorem1_validation_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved.")
