"""Physical electrical model (Ybus / Zbus) for the voltage-sag localization study.

Builds a complex bus admittance matrix from the BPA planning data:
  - transmission lines : DT_LINE R, X in ohms -> y = 1/(R + jX)
  - transformers       : DT_TRANS leakage LX,RG in per unit on the transformer's
                         own MVA rating -> Z_ohm = (RG + jLX) * V_HV^2 / S_HV

The bus impedance matrix Zbus = (Ybus + g I)^{-1} is the *voltage propagation
kernel*: for a fault at bus f with fault impedance Z_f, the per-unit voltage at
bus i is V_i = 1 - Z_{if}/(Z_{ff}+Z_f). Normalising by the fault-bus deviation
gives the spatial signature

    s_f(i) = Z_{if} / Z_{ff},

which is independent of Z_f. The columns s_f are the "steering vectors" of the
array-processing analogy: the observed normalised sag pattern is matched against
each candidate column to identify the fault location.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from graph_admittance import _read, _voltage_level

DEFAULT_DATA = '../data/历史数据表/数据表190908'


def _safe(v, default=np.nan):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def build_network(bustopo_path, linetopo_path, transtopo_path,
                  dt_line_path, dt_trans_path, keep_component=True,
                  shunt_pu=1e-3):
    """Return dict with bus_names, bus_idx, Ybus, Zbus, branch list."""
    import networkx as nx

    bus = _read(bustopo_path)
    lt = _read(linetopo_path)
    tt = _read(transtopo_path)
    dtl = _read(dt_line_path, ('gbk', 'gb2312', 'gb18030', 'utf-8-sig'))
    dtt = _read(dt_trans_path, ('gbk', 'gb2312', 'gb18030', 'utf-8-sig'))

    bus_names_all = bus['BUSNAME'].astype(str).tolist()
    idx_all = {n: i for i, n in enumerate(bus_names_all)}
    M = len(bus_names_all)

    Y = np.zeros((M, M), dtype=complex)
    branches = []  # (i, j, Ycomplex, kind)

    def add(i, j, yc, kind):
        nonlocal Y
        if i == j or not np.isfinite(abs(yc)) or abs(yc) == 0:
            return
        Y[i, i] += yc
        Y[j, j] += yc
        Y[i, j] -= yc
        Y[j, i] -= yc
        branches.append((i, j, yc, kind))

    dl = dtl.set_index('LINENAME')
    for _, row in lt.iterrows():
        s, e = row['STARBUSNAME'], row['ENDBUSNAME']
        if s not in idx_all or e not in idx_all:
            continue
        name = row['LINENAME']
        if name not in dl.index:
            continue
        r = _safe(dl.at[name, 'R']); x = _safe(dl.at[name, 'X'])
        if not np.isfinite(r) or not np.isfinite(x) or r * r + x * x < 1e-12:
            continue
        add(idx_all[s], idx_all[e], 1.0 / complex(r, x), 'line')

    dti = dtt.set_index('TRANSNAME')
    for _, trow in tt.iterrows():
        name = trow['TRANSNAME']
        if name not in dti.index:
            continue
        d = dti.loc[name]
        if isinstance(d, pd.DataFrame):
            d = d.iloc[0]
        s_hv = _safe(d.get('HV_S'), 100.0)
        if not np.isfinite(s_hv) or s_hv <= 0:
            s_hv = 100.0
        pairs = []
        if pd.notna(trow.get('HIGBUSNAME')) and pd.notna(trow.get('MIDBUSNAME')):
            pairs.append(('HIGBUSNAME', 'MIDBUSNAME', 'HV_LX', 'HV_RG'))
        if pd.notna(trow.get('HIGBUSNAME')) and pd.notna(trow.get('LOWBUSNAME')):
            pairs.append(('HIGBUSNAME', 'LOWBUSNAME', 'HV_LX', 'HV_RG'))
        if pd.notna(trow.get('MIDBUSNAME')) and pd.notna(trow.get('LOWBUSNAME')):
            pairs.append(('MIDBUSNAME', 'LOWBUSNAME', 'MV_LX', 'MV_RG'))
        for c1, c2, lc, rc in pairs:
            bi, bj = trow.get(c1), trow.get(c2)
            if bi not in idx_all or bj not in idx_all:
                continue
            lx = _safe(d.get(lc), 0.1)
            if not np.isfinite(lx) or lx <= 0:
                lx = 0.1
            rg = _safe(d.get(rc), 0.0)
            if not np.isfinite(rg) or rg < 0:
                rg = 0.0
            v = max(_voltage_level(str(bi)), _voltage_level(str(bj)))
            if not np.isfinite(v) or v <= 0:
                v = 110.0
            zb = v * v / s_hv
            add(idx_all[bi], idx_all[bj], 1.0 / complex(rg * zb, lx * zb), 'trans')

    # ---- keep the largest connected component (on the conductance skeleton) ----
    G = nx.Graph()
    for i, j, yc, kind in branches:
        G.add_edge(i, j)
    if keep_component and G.number_of_nodes() > 0:
        comp = max(nx.connected_components(G), key=len)
        nodes = sorted(comp)
    else:
        nodes = sorted(G.nodes())
    remap = {old: new for new, old in enumerate(nodes)}
    n = len(nodes)
    bus_names = [bus_names_all[old] for old in nodes]
    bus_idx = {name: k for k, name in enumerate(bus_names)}

    Yr = np.zeros((n, n), dtype=complex)
    br = []
    for i, j, yc, kind in branches:
        if i in remap and j in remap:
            a, b = remap[i], remap[j]
            Yr[a, a] += yc; Yr[b, b] += yc; Yr[a, b] -= yc; Yr[b, a] -= yc
            br.append((a, b, yc, kind))

    Ybus = Yr
    Zbus = np.linalg.inv(Ybus + shunt_pu * np.eye(n))

    return {
        'bus_names': bus_names,
        'bus_idx': bus_idx,
        'Ybus': Ybus,
        'Zbus': Zbus,
        'branches': br,
        'n': n,
        'shunt_pu': shunt_pu,
    }


def steering_dictionary(Zbus, buses=None):
    """Normalised |Zbus| columns: s_f(i) = |Z_if| / |Z_ff|.

    Rows are buses, columns are candidate fault buses. If `buses` is given,
    only those rows are returned (sparse-monitor sub-dictionary).
    """
    Z = Zbus
    diag = np.abs(np.diag(Z))
    diag[diag < 1e-12] = 1e-12
    S = np.abs(Z) / diag[None, :]     # s_f(i) = |Z_if|/|Z_ff|
    np.fill_diagonal(S, 1.0)
    if buses is not None:
        S = S[buses, :]
    return S


if __name__ == '__main__':
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    net = build_network(
        os.path.join(base, 'BUSTOPO.csv'),
        os.path.join(base, 'LINETOPO.csv'),
        os.path.join(base, 'TRANSTOPO.csv'),
        os.path.join(base, DEFAULT_DATA, 'DT_LINE.csv'),
        os.path.join(base, DEFAULT_DATA, 'DT_TRANS.csv'),
    )
    print('n =', net['n'], 'branches =', len(net['branches']))
    Z = net['Zbus']
    print('Zbus shape', Z.shape, '|Z| range', np.abs(Z).min(), np.abs(Z).max())
    S = steering_dictionary(Z)
    print('steering dict', S.shape, 'diag mean', np.diag(S).mean())
