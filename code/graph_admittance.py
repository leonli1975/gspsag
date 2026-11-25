"""Physical admittance-based graph construction for the voltage-sag GSP study.

Addresses reviewer concern (R3#1 / R2#3): the earlier graph used
W_ij = 1/line_length for lines and a fixed weight for transformers, which
(a) ignores per-unit base scaling across voltage levels and (b) makes
transformers the strongest edges, contrary to their role as bottlenecks.

Here every branch is converted to a *series susceptance in siemens*, a
physically absolute quantity that is directly comparable across voltage
levels:

  - Transmission lines: BPA DT_LINE gives R, X in ohms.
        y = 1/(R + jX),  b = -Im(y) = X/(R^2 + X^2)   [S]
  - Transformers: BPA DT_TRANS gives leakage reactance/resistance in per
    unit on the transformer's own MVA rating (verified: HV_LX * HV_S is
    approximately constant, cv ~ 0.3). Converted to ohms via the standard
    base Z_base = V_HV^2 / S_HV (V in kV, S in MVA):
        X_ohm = LX * V_HV^2 / S_HV,  R_ohm = RG * V_HV^2 / S_HV
        b = X_ohm/(R_ohm^2 + X_ohm^2)   [S]

The graph weight W_ij = b_ij (siemens). The Laplacian L = D - W is therefore
built from real branch susceptances; its eigenvectors (the GFT basis) are
invariant to a global rescaling, so only the *relative* branch strengths
matter -- which now correctly encode the physical network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp

S_BASE_MVA = 100.0  # fallback base if a transformer rating is missing


def _read(path, encodings=('utf-8-sig', 'gbk', 'gb2312', 'gb18030')):
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f'cannot decode {path}')


def _safe(v, default=np.nan):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def build_admittance_graph(bustopo_path, linetopo_path, transtopo_path,
                           dt_line_path, dt_trans_path,
                           keep_largest_component=True):
    """Return (G, L, bus_names, bus_idx) using physical series susceptances.

    G has integer node ids == BUSTOPO row order for the kept nodes; the
    returned bus_names list is parallel to G.nodes (sorted integer ids).
    """
    bus = _read(bustopo_path)
    lt = _read(linetopo_path)
    tt = _read(transtopo_path)
    dtl = _read(dt_line_path, ('gbk', 'gb2312', 'gb18030', 'utf-8-sig'))
    dtt = _read(dt_trans_path, ('gbk', 'gb2312', 'gb18030', 'utf-8-sig'))

    bus_names_all = bus['BUSNAME'].astype(str).tolist()
    bus_idx = {n: i for i, n in enumerate(bus_names_all)}

    edge_b = {}  # (i, j) -> susceptance (siemens), summed if parallel
    edge_kind = {}  # (i, j) -> 'line' or 'trans'

    def add_edge(i, j, b, kind):
        if i == j or not np.isfinite(b) or b <= 0:
            return
        key = (i, j) if i < j else (j, i)
        edge_b[key] = edge_b.get(key, 0.0) + b
        edge_kind[key] = kind

    # ---- transmission lines (R, X in ohms) ----
    dl = dtl.set_index('LINENAME')
    n_line = 0
    for _, row in lt.iterrows():
        s, e = row['STARBUSNAME'], row['ENDBUSNAME']
        if s not in bus_idx or e not in bus_idx:
            continue
        name = row['LINENAME']
        if name not in dl.index:
            continue
        r = _safe(dl.at[name, 'R'])
        x = _safe(dl.at[name, 'X'])
        if not np.isfinite(r) or not np.isfinite(x):
            continue
        z2 = r * r + x * x
        if z2 < 1e-12:
            continue
        b = x / z2  # series susceptance [S]
        add_edge(bus_idx[s], bus_idx[e], b, 'line')
        n_line += 1

    # ---- transformers (per-unit leakage on own MVA base) ----
    dti = dtt.set_index('TRANSNAME')
    n_trans = 0
    for _, trow in tt.iterrows():
        name = trow['TRANSNAME']
        if name not in dti.index:
            continue
        d = dti.loc[name]
        if isinstance(d, pd.DataFrame):   # duplicate names -> take first
            d = d.iloc[0]
        s_hv = _safe(d.get('HV_S'), S_BASE_MVA)
        if not np.isfinite(s_hv) or s_hv <= 0:
            s_hv = S_BASE_MVA

        pairs = []
        # (bus1 col, bus2 col, per-unit LX col, per-unit RG col, voltage col for base)
        if pd.notna(trow.get('HIGBUSNAME')) and pd.notna(trow.get('MIDBUSNAME')):
            pairs.append(('HIGBUSNAME', 'MIDBUSNAME', 'HV_LX', 'HV_RG', 'HVBUSVOLTAGE'))
        if pd.notna(trow.get('HIGBUSNAME')) and pd.notna(trow.get('LOWBUSNAME')):
            pairs.append(('HIGBUSNAME', 'LOWBUSNAME', 'HV_LX', 'HV_RG', 'HVBUSVOLTAGE'))
        if pd.notna(trow.get('MIDBUSNAME')) and pd.notna(trow.get('LOWBUSNAME')):
            pairs.append(('MIDBUSNAME', 'LOWBUSNAME', 'MV_LX', 'MV_RG', 'MVBUSVOLTAGE'))

        for c1, c2, lx_col, rg_col, v_col in pairs:
            bi, bj = trow.get(c1), trow.get(c2)
            if bi not in bus_idx or bj not in bus_idx:
                continue
            lx = _safe(d.get(lx_col))
            if not np.isfinite(lx) or lx <= 0:
                lx = 0.1  # conservative default leakage
            rg = _safe(d.get(rg_col), 0.0)
            if not np.isfinite(rg) or rg < 0:
                rg = 0.0
            v_kv = _safe(d.get(v_col))
            if not np.isfinite(v_kv) or v_kv <= 0:
                v_kv = max(_voltage_level(str(bi)), _voltage_level(str(bj)), 100.0)
            z_base = v_kv * v_kv / s_hv         # ohms, referred to HV side
            x_ohm = lx * z_base
            r_ohm = rg * z_base
            z2 = r_ohm * r_ohm + x_ohm * x_ohm
            if z2 < 1e-12:
                continue
            b = x_ohm / z2
            add_edge(bus_idx[bi], bus_idx[bj], b, 'trans')
            n_trans += 1

    # ---- assemble networkx graph ----
    G = nx.Graph()
    for (i, j), b in edge_b.items():
        G.add_edge(i, j, weight=float(b), kind=edge_kind[(i, j)])

    if keep_largest_component and G.number_of_nodes() > 0:
        comp = max(nx.connected_components(G), key=len)
        G = G.subgraph(comp).copy()

    nodes = sorted(G.nodes())
    remap = {old: new for new, old in enumerate(nodes)}
    G = nx.relabel_nodes(G, remap)
    bus_names = [bus_names_all[old] for old in nodes]
    new_idx = {name: k for k, name in enumerate(bus_names)}

    n = G.number_of_nodes()
    w = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    L = sp.diags(np.array([sum(d['weight'] for _, _, d in G.edges(nb, data=True))
                           for nb in G.nodes()])) - nx.to_scipy_sparse_array(G, weight='weight', format='csr')
    L = sp.csr_matrix(L)

    info = {
        'n_nodes': n,
        'n_edges': G.number_of_edges(),
        'n_line_edges': sum(1 for _, _, d in G.edges(data=True) if d['kind'] == 'line'),
        'n_trans_edges': sum(1 for _, _, d in G.edges(data=True) if d['kind'] == 'trans'),
        'b_line_median': float(np.median([d['weight'] for _, _, d in G.edges(data=True) if d['kind'] == 'line'])) if any(d['kind']=='line' for *_,d in G.edges(data=True)) else 0.0,
        'b_trans_median': float(np.median([d['weight'] for _, _, d in G.edges(data=True) if d['kind'] == 'trans'])) if any(d['kind']=='trans' for *_,d in G.edges(data=True)) else 0.0,
    }
    return G, L, bus_names, new_idx, info


def _voltage_level(name: str):
    parts = str(name).strip().split()
    if len(parts) >= 2:
        try:
            return float(parts[-1])
        except ValueError:
            return np.nan
    return np.nan


if __name__ == '__main__':
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    D = os.path.join(base, '..', 'data', '历史数据表', '数据表190908')
    G, L, names, idx, info = build_admittance_graph(
        os.path.join(base, 'BUSTOPO.csv'),
        os.path.join(base, 'LINETOPO.csv'),
        os.path.join(base, 'TRANSTOPO.csv'),
        os.path.join(D, 'DT_LINE.csv'),
        os.path.join(D, 'DT_TRANS.csv'),
    )
    print(info)
    import scipy.sparse.linalg as sla
    n = L.shape[0]
    vals = np.sort(np.linalg.eigvalsh(L.toarray()))
    print('eigenvalues 0..7:', np.round(vals[:8], 6))
    print('lambda2 = %.6f  lambdaN = %.4f' % (vals[1], vals[-1]))
