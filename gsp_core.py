"""
GSP Core Module for Voltage Sag Analysis
Graph construction, GFT, sampling, and filtering for power distribution networks.
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.sparse import linalg as splinalg
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / '历史数据表' / '数据表190908'


def _read_csv(filename, encodings=None):
    """Read CSV with automatic encoding detection."""
    if encodings is None:
        encodings = ['utf-8-sig', 'gb2312', 'gbk', 'gb18030']
    for enc in encodings:
        try:
            return pd.read_csv(DATA_DIR / filename, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f'Cannot decode {filename} with any encoding')


def load_power_grid_data():
    """Load all topology and sag data from CSV files."""
    bus_topo = _read_csv('BUSTOPO.csv', ['utf-8-sig'])
    line_topo = _read_csv('LINETOPO.csv', ['utf-8-sig'])
    trans_topo = _read_csv('TRANSTOPO.csv', ['utf-8-sig'])
    substation = _read_csv('SUBSTATION.csv', ['utf-8-sig'])
    sag_bus = _read_csv('SAG_BUSVOLTAGE.csv', ['gb2312', 'gbk', 'gb18030'])
    sag_event = _read_csv('SAG_EVENT.csv', ['gb2312', 'gbk', 'gb18030'])
    sag_index = _read_csv('SAG_INDEX.csv', ['gb2312', 'gbk', 'gb18030'])
    return bus_topo, line_topo, trans_topo, substation, sag_bus, sag_event, sag_index


def build_weighted_graph(bus_topo, line_topo, trans_topo=None, weight_mode='admittance'):
    """
    Build a weighted undirected graph from power grid topology.

    Parameters
    ----------
    bus_topo : DataFrame with BUSNAME, LONGITUDE, LATITUDE
    line_topo : DataFrame with STARBUSNAME, ENDBUSNAME, LINELENGTH
    trans_topo : DataFrame with HIGBUSNAME, MIDBUSNAME, LOWBUSNAME (optional)
    weight_mode : str
        'admittance' - use 1/length as weight (shorter line = stronger coupling)
        'binary' - 0/1 weights based on connectivity
        'distance' - use exp(-length/mean_length) as weight

    Returns
    -------
    G : networkx.Graph
    L : scipy sparse Laplacian matrix (N x N)
    bus_names : list of bus name strings (ordered)
    bus_idx : dict mapping bus name -> index
    """
    bus_names = bus_topo['BUSNAME'].tolist()
    bus_idx = {name: i for i, name in enumerate(bus_names)}
    N = len(bus_names)

    G = nx.Graph()

    # Only add nodes that have connections (appear in lines or transformers)
    connected_nodes = set()

    # Collect edges from lines
    edges = []
    lengths = []
    for _, row in line_topo.iterrows():
        start = row['STARBUSNAME']
        end = row['ENDBUSNAME']
        length = float(row['LINELENGTH'])
        if start in bus_idx and end in bus_idx:
            u, v = bus_idx[start], bus_idx[end]
            connected_nodes.add(u)
            connected_nodes.add(v)
            edges.append((u, v, length))
            lengths.append(length)

    # Collect edges from transformers (virtual connections between voltage levels)
    # Transformer impedance is typically small, so use a fixed short length
    transformer_length = 0.1  # equivalent short length for low-impedance transformer
    trans_edges = 0
    if trans_topo is not None:
        for _, row in trans_topo.iterrows():
            buses_in_trans = []
            for col in ['HIGBUSNAME', 'MIDBUSNAME', 'LOWBUSNAME']:
                name = row.get(col, '')
                if pd.notna(name) and name.strip() and name in bus_idx:
                    buses_in_trans.append(bus_idx[name])
            # Connect all buses within the same transformer
            for i in range(len(buses_in_trans)):
                for j in range(i + 1, len(buses_in_trans)):
                    u, v = buses_in_trans[i], buses_in_trans[j]
                    connected_nodes.add(u)
                    connected_nodes.add(v)
                    edges.append((u, v, transformer_length))
                    trans_edges += 1

    if not edges:
        raise ValueError('No valid edges found. Check bus name matching.')

    # Build graph with only connected nodes
    connected_nodes = sorted(connected_nodes)
    old_to_new = {old: new for new, old in enumerate(connected_nodes)}
    N = len(connected_nodes)
    bus_names = [bus_names[old] for old in connected_nodes]
    bus_idx = {name: i for i, name in enumerate(bus_names)}

    G.add_nodes_from(range(N))
    mean_length = np.mean(lengths) if lengths else 1.0

    if weight_mode == 'admittance':
        for u_old, v_old, length in edges:
            w = 1.0 / max(length, 0.01)
            G.add_edge(old_to_new[u_old], old_to_new[v_old], weight=w)
    elif weight_mode == 'distance':
        for u_old, v_old, length in edges:
            w = np.exp(-length / mean_length)
            G.add_edge(old_to_new[u_old], old_to_new[v_old], weight=w)
    else:  # binary
        for u_old, v_old, _ in edges:
            G.add_edge(old_to_new[u_old], old_to_new[v_old], weight=1.0)

    # Build Laplacian matrix L = D - W
    W = nx.adjacency_matrix(G, weight='weight').astype(np.float64)
    degrees = np.array(W.sum(axis=1)).flatten()
    D = sparse.diags(degrees)
    L = D - W

    return G, L, bus_names, bus_idx


def laplacian_eigendecomposition(L, k=None):
    """
    Compute eigenvalue decomposition of graph Laplacian.
    Returns eigenvalues (ascending) and eigenvectors.
    """
    if k is None or k >= L.shape[0]:
        eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
    else:
        eigenvalues, eigenvectors = splinalg.eigsh(L, k=k, which='SM')
    # Sort by eigenvalue ascending
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def compute_gft(signal, U):
    """Compute Graph Fourier Transform of a signal x -> x_hat = U^T x."""
    return U.T @ signal


def compute_igft(coefficients, U):
    """Compute Inverse Graph Fourier Transform x = U x_hat."""
    return U @ coefficients


def verify_bandlimitedness(signal, U, eigenvalues, top_k=20):
    """
    Check if a signal is approximately bandlimited on the graph.
    Returns cumulative energy ratio for the first top_k frequencies.
    """
    x_hat = compute_gft(signal, U)
    energy = x_hat ** 2
    total_energy = np.sum(energy)
    sorted_idx = np.argsort(np.abs(x_hat))[::-1]
    cumulative = np.cumsum(energy[sorted_idx]) / total_energy
    return cumulative[:top_k], sorted_idx[:top_k]


def graph_sampling_reconstruction(L, U, eigenvalues, y_S, S, bandwidth):
    """
    Reconstruct full graph signal from partial samples using GSP sampling.

    Parameters
    ----------
    L : sparse Laplacian
    U : eigenvectors (N x N)
    eigenvalues : eigenvalues (N,)
    y_S : observed values at sampled nodes (M,)
    S : indices of sampled nodes (M,)
    bandwidth : number of lowest-frequency eigenvectors to use

    Returns
    -------
    x_hat : reconstructed full signal (N,)
    """
    N = L.shape[0]
    M = len(S)

    if bandwidth > M:
        bandwidth = M  # Can't reconstruct more than M measurements

    U_F = U[:, :bandwidth]  # N x |F|
    M_S = np.zeros((M, N))
    for i, s in enumerate(S):
        M_S[i, s] = 1.0

    # Reconstruction: x_hat = U_F * (M_S * U_F)^+ * y_S
    Phi = M_S @ U_F  # M x |F|
    pinv_Phi = np.linalg.pinv(Phi)
    alpha = pinv_Phi @ y_S
    x_hat = U_F @ alpha

    return x_hat


def graph_lowpass_filter(U, signal, cutoff_idx):
    """
    Apply graph low-pass filter: zero out GFT coefficients beyond cutoff_idx.
    Equivalent to polynomial filter H(L) where H(lambda_i) = 1 for i <= cutoff_idx, 0 otherwise.
    """
    x_hat = compute_gft(signal, U)
    x_hat_filtered = x_hat.copy()
    x_hat_filtered[cutoff_idx:] = 0.0
    return compute_igft(x_hat_filtered, U)


def graph_total_variation(signal, L):
    """Compute graph total variation TV_G(x) = x^T L x."""
    return signal.T @ L @ signal


def design_sampling_set(G, L, U, eigenvalues, M, method='max_degree'):
    """
    Design an optimal sampling set of M nodes.

    Methods:
    - 'max_degree': pick nodes with highest degree
    - 'random': random selection
    - 'eigenvector': pick nodes that maximize certain criteria from eigenvectors
    """
    N = L.shape[0]

    if M >= N:
        return np.arange(N)

    if method == 'max_degree':
        nodes = sorted(G.nodes())
        degrees = np.array([G.degree(n) for n in nodes])
        return np.argsort(degrees)[::-1][:M]
    elif method == 'eigenvector':
        # Use leverage scores of the first M eigenvectors
        U_M = U[:, :min(M, U.shape[1])]
        scores = np.sum(U_M ** 2, axis=1)
        return np.argsort(scores)[::-1][:M]
    else:
        rng = np.random.RandomState(42)
        return rng.choice(N, M, replace=False)
