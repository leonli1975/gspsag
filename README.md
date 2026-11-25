# Submission package — *Electrical Engineering* (Springer)

This directory bundles everything needed to inspect, reproduce, and submit the manuscript
*"A monitor-observability framework for measurement-network design in graph-based voltage-sag
estimation and localization"* (target: *Electrical Engineering*, Springer). It is self-contained
and can be zipped as is.

```
submission_package/
├── README.md            ← this file
├── code/                ← Python scripts that generate every table and figure
│   ├── *.py
│   ├── *.csv            ← regional-grid and BPA inputs used by the scripts
│   └── results_*.json   ← cached numeric results (so figures can be rebuilt quickly)
└── data/                ← additional inputs, in the layout the scripts expect
    ├── 历史数据表/数据表190908/{DT_LINE.csv, DT_TRANS.csv}
    └── 暂降域数据11-4/sag_signals_1028.npz
```

## Code (`code/`) — environment and layout

Python 3.10+ with: `numpy`, `scipy`, `pandas`, `networkx`, `matplotlib`, `pandapower`.

```bash
pip install numpy scipy pandas networkx matplotlib pandapower
```

The scripts expect to run **from this `code/` directory**; they read the CSVs next to them and the
inputs in `../data/` (the same relative layout as the original working directory). Each experiment
writes its numeric results to `results_*.json` and figures to `code/figures/`.

## Reproducing every table and figure

The cached `results_*.json` files let the `make_figs_*` scripts redraw the figures without re-running
the expensive experiments. To regenerate from scratch, run the experiment scripts first.

| Paper item | Script | Produces |
|-----------|--------|----------|
| Fig. 1 (concept) | `make_figs_concept.py` | `figures/fig_concept_observability.pdf` |
| Fig. 2 (edge weights) | `make_figs_sensitivity.py` | `figures/fig_ops_edges.pdf` |
| Fig. 3, Table 1 (bandlimitedness) | `experiment_graph_sensitivity.py`, then `make_figs_sensitivity.py` | `results_graph_sensitivity.json`, `figures/fig_ops_bandlimitedness.pdf` |
| Fig. 4 (flowchart), Fig. 5 (design maps) | `make_figs_framework.py` | `figures/fig_framework_flowchart.pdf`, `figures/fig_design_map.pdf` |
| Table 2 (fairness) | `experiment_fairness.py` | `results_fairness.json` |
| Table 3, Fig. 6 (designed vs random) | `experiment_estimation_designed.py`, then `make_figs_designed.py` | `results_estimation_designed.json`, `figures/fig_estimation_designed.pdf` |
| Fig. 7 (actual vs estimated) | `experiment_recon_scatter.py` | `figures/fig_recon_scatter.pdf` |
| Table 4 (learned filter) | `experiment_learning_baseline.py` | `results_learning_baseline.json` |
| Table 5, Fig. 8 (localization) | `experiment_localization.py`, then `make_figs_sensitivity.py` | `results_localization.json`, `figures/fig_localization_vs_M.pdf` |
| Table 6 (placement baselines) | `experiment_framework_baselines.py` | `results_framework_baselines.json` |
| Table 7 (second network, PEGASE) | `experiment_second_network.py` | `results_second_network.json` |
| Model-only baseline (inline in Sec. 4.2) | `experiment_model_baseline.py` | prints the model-only vs calibrated figures |

Typical order:

```bash
cd code
python experiment_graph_sensitivity.py
python experiment_estimation_designed.py
python experiment_fairness.py
python experiment_learning_baseline.py
python experiment_localization.py
python experiment_framework_baselines.py
python experiment_second_network.py     # needs pandapower; ~minutes (Zbus inverse)
python experiment_model_baseline.py
python experiment_recon_scatter.py
python make_figs_sensitivity.py
python make_figs_designed.py
python make_figs_framework.py
python make_figs_concept.py
```

`experiment_second_network.py` uses `pandapower.case1354pegase` and has no local data dependency;
the others read the CSVs in `code/` and the files in `../data/`.

## Data provenance

- **Regional grid**: `BUSTOPO.csv`, `LINETOPO.csv`, `TRANSTOPO.csv` (buses, lines, transformers),
  with per-unit impedances `DT_LINE.csv` / `DT_TRANS.csv`.
- **BPA short-circuit events**: `SAG_BUSVOLTAGE.csv`, `SAG_INDEX.csv` (the seven detailed events used
  for bandlimitedness and estimation).
- **4112-event catalog**: `sag_signals_1028.npz` (1028 fault locations × four fault types, mapped to
  the 1025-bus connected component), used for localization.

Per the manuscript's Data availability statement, the underlying BPA short-circuit event catalog is
available from the corresponding author on reasonable request; the code is also mirrored at
<https://github.com/leonli1975/gspsag>.

## Notes

- The derived 4112-event catalogue (`sag_signals_1028.npz`) is bundled in this local directory but is
  **not** committed to the repository, consistent with the data-availability statement (available from
  the corresponding author on request). If you clone the repository, request this file separately.
- Inputs that the scripts use are included here so the package runs offline, except `pandapower`'s
  built-in PEGASE case (fetched from the installed package at run time).
- Figures are regenerated into `code/figures/`; the copies used by the manuscript are in
  `manuscript/`.
- The scripts preserve the original relative paths (`code/../data/...`), so no path edits are needed.
