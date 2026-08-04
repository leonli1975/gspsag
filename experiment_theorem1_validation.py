"""
C1 Fix: Validate Theorem 1 — check whether sag data satisfies L·δ ≈ α·e_f
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gsp_core import load_power_grid_data, build_weighted_graph

# ── Load data ──────────────────────────────────────────────────────────────
bus_topo, line_topo, trans_topo, substation, sag_bus, sag_event, sag_index = load_power_grid_data()

# ── Build graph ────────────────────────────────────────────────────────────
G, L_sparse, bus_names, bus_idx = build_weighted_graph(bus_topo, line_topo, trans_topo)
L = L_sparse.toarray()
N = G.number_of_nodes()
print(f"Graph: {N} nodes, {G.number_of_edges()} edges")

# ── Inspect SAG data format ────────────────────────────────────────────────
print(f"SAG_BUS columns: {sag_bus.columns.tolist()}")
print(f"SAG_BUS sample events: {sag_bus.iloc[:5].to_dict('records')}")

# Find event column name
event_col = None
for col in ['EVENT_ID', '事件序号']:
    if col in sag_bus.columns:
        event_col = col
        break

if event_col is None:
    print("ERROR: Cannot find event column")
    exit(1)

event_ids = sorted(sag_bus[event_col].unique())
print(f"Events: {event_ids}")

# Find bus name column and voltage column
name_col = None
v_col = None
for col in ['BUSNAME', 'BUSNAME', '名称']:
    if col in sag_bus.columns:
        name_col = col
        break
for col in ['VA', '电压pu', 'VOLTAGE']:
    if col in sag_bus.columns:
        v_col = col
        break

print(f"Name column: {name_col}, Voltage column: {v_col}")
print(f"bus_idx has {len(bus_idx)} entries")
print(f"Sample SAG bus names: {sag_bus[name_col].iloc[:5].tolist()}")
print(f"Sample bus_idx keys: {list(bus_idx.keys())[:5]}")

# Check name matching
sag_names = set(str(n) for n in sag_bus[name_col].unique())
idx_names = set(str(n) for n in bus_idx.keys())
print(f"SAG names: {len(sag_names)}, bus_idx names: {len(idx_names)}")
print(f"Intersection: {len(sag_names & idx_names)}")
# Show mismatches
only_sag = sag_names - idx_names
only_idx = idx_names - sag_names
if only_sag:
    print(f"  In SAG but not bus_idx: {list(only_sag)[:5]}")
if only_idx:
    print(f"  In bus_idx but not SAG: {list(only_idx)[:5]}")

# ── Determine fault bus for each event ─────────────────────────────────────
event_fault_node = {}
for eid in event_ids:
    rows = sag_bus[sag_bus[event_col] == eid]
    if len(rows) == 0:
        continue
    worst_idx = rows[v_col].idxmin()
    fault_name = str(rows.loc[worst_idx, name_col])
    if fault_name in bus_idx:
        event_fault_node[eid] = bus_idx[fault_name]
    else:
        # Try to match by stripping suffixes
        fault_name_clean = fault_name.strip()
        if fault_name_clean in bus_idx:
            event_fault_node[eid] = bus_idx[fault_name_clean]

print(f"\nFault buses found: {len(event_fault_node)} of {len(event_ids)} events")

# ── Validate Lδ ≈ α·e_f ──────────────────────────────────────────────────
results = {}
for eid in event_ids[:6]:
    if eid not in event_fault_node:
        continue
    
    rows = sag_bus[sag_bus[event_col] == eid]
    x = np.ones(N)
    n_matched = 0
    for _, row in rows.iterrows():
        name = str(row[name_col])
        if name in bus_idx:
            x[bus_idx[name]] = float(row[v_col])
            n_matched += 1
    
    if n_matched < 10:
        continue
    
    delta = 1.0 - x  # voltage deviation
    Ld = L @ delta
    
    fault_node = event_fault_node[eid]
    exc = np.abs(Ld[fault_node])
    others = np.delete(np.abs(Ld), fault_node)
    max_other = np.max(others) if len(others) > 0 else 1e-15
    sparsity_ratio = exc / max_other
    
    # Fit linear model: Use only first B eigenpairs (bandlimited assumption)
    eigvals, U_all = np.linalg.eigh(L)
    B_fit = 5  # bandwidth per Theorem 1
    eigvals_k = eigvals[1:B_fit+1]  # skip λ_1=0
    U = U_all[:, 1:B_fit+1]
    
    Linv_col = np.zeros(N)
    for k_idx, k in enumerate(range(1, B_fit+1)):
        lam = eigvals_k[k_idx]
        if lam > 1e-15:
            Linv_col += U[fault_node, k_idx] * U[:, k_idx] / lam
    
    alpha = np.dot(Linv_col, delta) / (np.dot(Linv_col, Linv_col) + 1e-15)
    delta_pred = alpha * Linv_col
    ss_res = np.sum((delta - delta_pred)**2)
    ss_tot = np.sum((delta - np.mean(delta))**2)
    r_sq = 1 - ss_res/(ss_tot + 1e-15)
    
    results[eid] = {'fault_node': fault_node, 'sr': sparsity_ratio, 'r2': r_sq, 'alpha': alpha}
    print(f"  Event {eid}: SR={sparsity_ratio:.1f}x, R²={r_sq:.3f}, α={alpha:.4f}")

# ── Summary ────────────────────────────────────────────────────────────────
eids_sorted = sorted(results.keys())
r2_all = [results[e]['r2'] for e in eids_sorted]
sr_all = [results[e]['sr'] for e in eids_sorted]
mean_r2 = np.mean(r2_all)
mean_sr = np.exp(np.mean(np.log([v for v in sr_all if v > 0])))

print(f"\n{'='*60}")
print(f"SUMMARY: Mean R²={mean_r2:.3f}, GeoMean SR={mean_sr:.1f}x")
if mean_r2 > 0.8:
    print(f"STRONG validation: linear diffusion model confirmed")
elif mean_r2 > 0.5:
    print(f"MODERATE: model is reasonable approximation")
else:
    print(f"WEAK: model needs revision")

# ── Generate figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
labels = [str(e) for e in eids_sorted]

colors_s = ['#2ca02c' if v > 10 else '#ff7f0e' if v > 3 else '#d62728' for v in sr_all]
axes[0].bar(labels, sr_all, color=colors_s, alpha=0.8)
axes[0].axhline(y=10, color='green', ls='--', alpha=0.5, label='Strong (10×)')
axes[0].axhline(y=3, color='orange', ls='--', alpha=0.5, label='Moderate (3×)')
axes[0].set_ylabel('Sparsity Ratio')
axes[0].set_title('$|(\\mathbf{L}\\boldsymbol{\\delta})_f| / \\max_{i\\neq f}|(\\mathbf{L}\\boldsymbol{\\delta})_i|$')
axes[0].legend(loc='lower right')
axes[0].set_yscale('log')

colors_r = ['#2ca02c' if v > 0.8 else '#ff7f0e' if v > 0.5 else '#d62728' for v in r2_all]
axes[1].bar(labels, r2_all, color=colors_r, alpha=0.8)
axes[1].axhline(y=0.8, color='green', ls='--', alpha=0.5, label='Excellent (0.8)')
axes[1].axhline(y=0.5, color='orange', ls='--', alpha=0.5, label='Moderate (0.5)')
axes[1].set_ylabel('$R^2$ Score')
axes[1].set_title('$R^2$ of $\\boldsymbol{\\delta} = \\alpha\\mathbf{L}^\\dagger\\mathbf{e}_f$')
axes[1].legend(loc='lower right')
axes[1].set_ylim(0, 1.05)
axes[1].annotate(f'Mean $R^2$ = {mean_r2:.3f}', xy=(0.5, 0.05), xycoords='axes fraction',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle(r'Theorem 1 Model Validation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig_theorem1_validation_summary.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures/fig_theorem1_validation_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved to figures/fig_theorem1_validation_summary.pdf")
