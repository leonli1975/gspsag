"""Conceptual figure: monitor observability unifies estimation and localization."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'figures')
plt.rcParams.update({'font.size': 9})


def box(ax, xy, w, h, text, fc, ec, fs=9, bold=False):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                       linewidth=1.2, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', wrap=True)


def arrow(ax, p0, p1, color='#444444'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=12,
                                 linewidth=1.1, color=color, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

    # central concept
    box(ax, (3.5, 4.5), 3.0, 1.0, 'Monitor observability\n(what the monitors can see)',
        '#e8f0fb', '#2b5fa8', fs=9, bold=True)

    # two facets
    box(ax, (0.4, 2.2), 3.6, 1.6,
        'Estimation\n\nspectral observability\n'
        r'$\mathbf{M}_{\mathcal{S}}\mathbf{U}_{B}$ well conditioned',
        '#eaf6ea', '#2e7d32', fs=8)
    box(ax, (6.0, 2.2), 3.6, 1.6,
        'Localization\n\nspatial observability\nsampled signature must\ndiscriminate sources',
        '#fdf1e6', '#b35c00', fs=8)

    # actionable outcome
    box(ax, (2.6, 0.2), 4.8, 1.0,
        'A well-conditioned design\nserves both tasks',
        '#f3eaf7', '#6a3d9a', fs=8, bold=True)

    # arrows
    arrow(ax, (4.4, 4.5), (2.6, 3.8))
    arrow(ax, (5.6, 4.5), (7.4, 3.8))
    arrow(ax, (2.2, 2.2), (4.0, 1.2))
    arrow(ax, (7.8, 2.2), (6.0, 1.2))

    ax.text(3.0, 4.15, 'facet 1', fontsize=7, style='italic', color='#555555')
    ax.text(7.0, 4.15, 'facet 2', fontsize=7, style='italic', color='#555555')

    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, 'fig_concept_observability.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_concept_observability.png'), dpi=160)
    print('saved fig_concept_observability')


if __name__ == '__main__':
    main()
