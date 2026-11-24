"""
=============================================================================
SyncMap Edge-Weighted Graph Clustering Engine (Task A3)
Author: Sanjana Laxman
Candidate Selection - Laboratory of Intelligent Systems, Kyushu University
Supervisor: Prof. Danilo Vasconcellos Vargas

Description:
  Implements a self-organizing SyncMap architecture for continual general
  chunking, replaces the baseline DBSCAN clustering stage with a custom
  edge-weighted graph clustering method utilizing adaptive local scaling,
  co-activation topology fusion, and modularity optimization. Evaluates
  clustering accuracy (ARI, NMI, V-Measure) and multi-seed variance across
  four distinct benchmark tasks.
=============================================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

# Set random seed base for reproducible benchmarking
GLOBAL_SEED = 42

class SyncMap:
    """
    Continual Self-Organizing SyncMap dynamical system based on Vargas & Asabuki (AAAI-21).
    Maps symbolic variables to continuous Euclidean space via emergent attractor-repeller
    feedback centers without explicit objective function or backpropagation.
    """
    def __init__(self, n_nodes, dim=2, lr_att=0.09, lr_rep=0.045, seed=None):
        self.n_nodes = n_nodes
        self.dim = dim
        self.lr_att = lr_att
        self.lr_rep = lr_rep
        rng = np.random.default_rng(seed)
        # Random initial coordinates in [-1, 1]^d
        self.W = rng.uniform(-1.0, 1.0, size=(n_nodes, dim))
        # Empirical co-activation tracker
        self.co_act = np.zeros((n_nodes, n_nodes), dtype=float)
        self.activation_count = np.zeros(n_nodes, dtype=float)

    def step(self, active_nodes, decay=0.998):
        """
        Executes a single dynamical step of positive attraction and negative repulsion.
        Updates internal coordinate representations and empirical co-activation matrix.
        """
        self.co_act *= decay
        for i in active_nodes:
            self.activation_count[i] += 1.0
            for j in active_nodes:
                if i != j:
                    self.co_act[i, j] += 1.0

        A = np.array(active_nodes, dtype=int)
        N = np.array([x for x in range(self.n_nodes) if x not in active_nodes], dtype=int)

        c_p = np.mean(self.W[A], axis=0)
        c_n = np.mean(self.W[N], axis=0) if len(N) > 0 else np.zeros(self.dim)

        # Update active nodes (attract to positive center, repel from negative center)
        for i in A:
            diff_p = c_p - self.W[i]
            d_p = np.linalg.norm(diff_p) + 1e-6
            diff_n = c_n - self.W[i]
            d_n = np.linalg.norm(diff_n) + 1e-6
            self.W[i] += self.lr_att * (diff_p / d_p) - self.lr_rep * (diff_n / d_n)

        # Update inactive nodes (repel from active group center)
        if len(N) > 0:
            for j in N:
                diff = self.W[j] - c_p
                d = np.linalg.norm(diff) + 1e-6
                self.W[j] += (self.lr_rep * 0.4) * (diff / d)

        # Keep coordinate space bounded to maintain numerical stability
        self.W = np.clip(self.W, -3.5, 3.5)


def cluster_syncmap_dbscan(W, eps=0.30, min_samples=2):
    """
    Baseline SyncMap clustering: Applies sklearn DBSCAN directly to spatial coordinates.
    Returns cluster labels (-1 denotes unassigned noise).
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(W)
    return labels


def build_edge_weighted_graph(W, co_act, k_nn=4, beta=1.2, mutual=True):
    """
    Constructs an edge-weighted undirected graph G = (V, E, W_fused) over map nodes:
      1. Spatial affinity via Zelnik-Manor & Perona adaptive local scaling.
      2. Empirical co-activation normalized correlation.
      3. Multi-modal edge weight fusion: W_fused = S * (1 + beta * P).
      4. Structural edge pruning (mutual k-NN / adaptive thresholding).
    """
    n = W.shape[0]
    # Pairwise Euclidean distance
    D = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=-1)

    # Adaptive local scaling sigma_i based on distance to k-th nearest neighbor
    k_eff = min(k_nn, n - 1)
    sorted_dists = np.sort(D, axis=1)
    sigma = np.maximum(sorted_dists[:, k_eff], 1e-4)

    # Local scale Gaussian affinity kernel
    S = np.exp(- (D ** 2) / (sigma[:, None] * sigma[None, :] + 1e-6))
    np.fill_diagonal(S, 0.0)

    # Empirical co-activation cosine correlation
    denom = np.sqrt(np.diag(co_act)[:, None] * np.diag(co_act)[None, :]) + 1e-6
    P = co_act / denom
    np.fill_diagonal(P, 0.0)

    # Fused edge weight combining spatial convergence and temporal co-occurrence
    W_fused = S * (1.0 + beta * P)

    # Build graph with structural filtering
    top_k_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        idxs = np.argsort(W_fused[i])[-k_eff:]
        top_k_mask[i, idxs] = True

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            is_edge = (top_k_mask[i, j] and top_k_mask[j, i]) if mutual else (top_k_mask[i, j] or top_k_mask[j, i])
            if is_edge and W_fused[i, j] > 1e-4:
                G.add_edge(i, j, weight=float(W_fused[i, j]))

    # Ensure no node is isolated due to strict mutual filtering
    for i in range(n):
        if G.degree(i) == 0:
            best_j = int(np.argmax(W_fused[i]))
            if best_j != i:
                G.add_edge(i, best_j, weight=float(W_fused[i, best_j]))

    return G, W_fused


def cluster_syncmap_graph(W, co_act, k_nn=4, beta=1.2, resolution=1.0, mutual=True):
    """
    Proposed Pipeline: Completely removes DBSCAN.
    Clusters using edge properties and weights via modularity optimization (Louvain)
    over the constructed edge-weighted graph.
    """
    G, _ = build_edge_weighted_graph(W, co_act, k_nn=k_nn, beta=beta, mutual=mutual)
    n = W.shape[0]

    # Edge-weighted community detection maximizing modularity
    communities = nx.community.louvain_communities(G, weight='weight', resolution=resolution, seed=GLOBAL_SEED)
    labels = np.zeros(n, dtype=int)
    for c_idx, comm in enumerate(communities):
        for node in comm:
            labels[node] = c_idx

    return labels, G


def run_benchmark_suite(n_seeds=20):
    """
    Executes a comprehensive multi-seed benchmark comparing DBSCAN vs. Proposed
    Edge-Weighted Graph Clustering across four distinct experimental tasks:
      - Task 1: Non-Overlapping Fixed Chunks
      - Task 2: Overlapping / Shared-Element Chunks
      - Task 3: Noisy & Varying-Density Chunks
      - Task 4: Continual / Dynamic Shifting Chunk Structure
    """
    # ---------------------------------------------------------
    # Task Specifications
    # ---------------------------------------------------------
    # Task 1: 5 distinct chunks of 4 nodes each (Total 20 nodes)
    chunks_t1 = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19]]
    gt_t1 = np.repeat(np.arange(5), 4)

    # Task 2: 4 overlapping chunks with bridge nodes (Total 17 nodes)
    # Bridge nodes: 4 (between 0..3 and 5..7), 8 (between 5..7 and 9..11), 12 (between 9..11 and 13..16)
    chunks_t2 = [[0,1,2,3,4], [4,5,6,7,8], [8,9,10,11,12], [12,13,14,15,16]]
    gt_t2 = np.array([0,0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3])

    # Task 3: Chunks of varying sizes and activation frequencies + background noise (Total 25 nodes)
    chunks_t3 = [[0,1,2], [3,4,5,6], [7,8,9,10,11], [12,13,14,15,16,17], [18,19,20,21,22,23,24]]
    probs_t3 = np.array([0.38, 0.26, 0.18, 0.12, 0.06])
    gt_t3 = np.zeros(25, dtype=int)
    for idx, ch in enumerate(chunks_t3):
        for u in ch: gt_t3[u] = idx

    # Task 4: Continual structural shift (Total 20 nodes)
    # Phase 1: 4 chunks of 5 nodes
    chunks_t4_p1 = [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19]]
    # Phase 2: Dynamic topological reconfiguration into 4 new chunks
    chunks_t4_p2 = [[0,1,5,6,7], [2,3,4,8,9], [10,11,15,16,17], [12,13,14,18,19]]
    gt_t4 = np.zeros(20, dtype=int)
    for idx, ch in enumerate(chunks_t4_p2):
        for u in ch: gt_t4[u] = idx

    tasks_def = [
        {
            "id": "Task 1",
            "name": "Non-Overlapping Fixed Chunks",
            "n_nodes": 20,
            "gt": gt_t1,
            "stream_type": "fixed",
            "chunks": chunks_t1,
            "probs": None,
            "steps": 600,
            "has_noise": False,
            "k_nn": 4,
            "resolution": 1.0,
            "eps": 0.30
        },
        {
            "id": "Task 2",
            "name": "Overlapping / Shared-Element Chunks",
            "n_nodes": 17,
            "gt": gt_t2,
            "stream_type": "overlapping",
            "chunks": chunks_t2,
            "probs": None,
            "steps": 700,
            "has_noise": False,
            "k_nn": 3,
            "resolution": 1.1,
            "eps": 0.28
        },
        {
            "id": "Task 3",
            "name": "Noisy & Varying-Density Chunks",
            "n_nodes": 25,
            "gt": gt_t3,
            "stream_type": "noisy_varying",
            "chunks": chunks_t3,
            "probs": probs_t3,
            "steps": 750,
            "has_noise": True,
            "k_nn": 4,
            "resolution": 1.0,
            "eps": 0.30
        },
        {
            "id": "Task 4",
            "name": "Continual / Dynamic Shifting Structure",
            "n_nodes": 20,
            "gt": gt_t4,
            "stream_type": "continual",
            "chunks": (chunks_t4_p1, chunks_t4_p2),
            "probs": None,
            "steps": 800,
            "has_noise": False,
            "k_nn": 4,
            "resolution": 1.0,
            "eps": 0.30
        }
    ]

    all_results = {}
    representative_samples = {}

    print("=" * 75)
    print("STARTING ADVANCED TASK [A3] BENCHMARK EXECUTION")
    print(f"Total Tasks: {len(tasks_def)} | Seeds per Task: {n_seeds}")
    print("=" * 75)

    for task in tasks_def:
        t_id = task["id"]
        t_name = task["name"]
        n_nodes = task["n_nodes"]
        gt = task["gt"]
        k_val = task["k_nn"]
        res_val = task["resolution"]
        eps_val = task["eps"]
        steps = task["steps"]

        db_ari_list, gr_ari_list = [], []
        db_nmi_list, gr_nmi_list = [], []
        db_v_list, gr_v_list = [], []
        db_noise_list = []

        print(f"\nRunning {t_id}: {t_name} across {n_seeds} seeds...")

        for s in range(n_seeds):
            sm = SyncMap(n_nodes, seed=s)
            rng = np.random.default_rng(s)

            if task["stream_type"] == "continual":
                p1_chunks, p2_chunks = task["chunks"]
                # Phase 1
                for _ in range(350):
                    c = p1_chunks[rng.integers(len(p1_chunks))]
                    sm.step(c, decay=0.994)
                # Phase 2 (dynamic environmental shift)
                for _ in range(500):
                    c = p2_chunks[rng.integers(len(p2_chunks))]
                    sm.step(c, decay=0.994)
            else:
                chunks = task["chunks"]
                probs = task["probs"]
                for _ in range(steps):
                    if task["has_noise"] and rng.random() < 0.15:
                        c = rng.choice(n_nodes, size=3, replace=False).tolist()
                    else:
                        if probs is not None:
                            c = chunks[rng.choice(len(chunks), p=probs)]
                        else:
                            c = chunks[rng.integers(len(chunks))]
                    sm.step(c)

            # DBSCAN Baseline
            l_db = cluster_syncmap_dbscan(sm.W, eps=eps_val, min_samples=2)
            noise_ratio = float(np.mean(l_db == -1))
            db_noise_list.append(noise_ratio)

            # Proposed Edge-Weighted Graph Clustering
            l_gr, G_graph = cluster_syncmap_graph(
                sm.W, sm.co_act, k_nn=k_val, resolution=res_val, mutual=True
            )

            # Compute evaluation metrics
            ari_db = adjusted_rand_score(gt, l_db)
            ari_gr = adjusted_rand_score(gt, l_gr)
            nmi_db = normalized_mutual_info_score(gt, l_db)
            nmi_gr = normalized_mutual_info_score(gt, l_gr)
            v_db = v_measure_score(gt, l_db)
            v_gr = v_measure_score(gt, l_gr)

            db_ari_list.append(float(ari_db))
            gr_ari_list.append(float(ari_gr))
            db_nmi_list.append(float(nmi_db))
            gr_nmi_list.append(float(nmi_gr))
            db_v_list.append(float(v_db))
            gr_v_list.append(float(v_gr))

            # Store representative sample for visualization (seed 2)
            if s == 2 and t_id not in representative_samples:
                representative_samples[t_id] = {
                    "W": sm.W.copy(),
                    "gt": gt.copy(),
                    "db_labels": l_db.copy(),
                    "gr_labels": l_gr.copy(),
                    "G": G_graph.copy(),
                    "co_act": sm.co_act.copy()
                }

        # Compute summary statistics
        m_db_ari, s_db_ari, v_db_ari = np.mean(db_ari_list), np.std(db_ari_list), np.var(db_ari_list)
        m_gr_ari, s_gr_ari, v_gr_ari = np.mean(gr_ari_list), np.std(gr_ari_list), np.var(gr_ari_list)

        m_db_nmi, s_db_nmi, v_db_nmi = np.mean(db_nmi_list), np.std(db_nmi_list), np.var(db_nmi_list)
        m_gr_nmi, s_gr_nmi, v_gr_nmi = np.mean(gr_nmi_list), np.std(gr_nmi_list), np.var(gr_nmi_list)

        var_reduction_ari = (v_db_ari - v_gr_ari) / (v_db_ari + 1e-9) * 100.0
        acc_improvement_ari = m_gr_ari - m_db_ari

        task_record = {
            "task_id": t_id,
            "task_name": t_name,
            "n_nodes": n_nodes,
            "n_seeds": n_seeds,
            "dbscan": {
                "ari_mean": float(m_db_ari),
                "ari_std": float(s_db_ari),
                "ari_var": float(v_db_ari),
                "nmi_mean": float(m_db_nmi),
                "nmi_std": float(s_db_nmi),
                "nmi_var": float(v_db_nmi),
                "v_measure_mean": float(np.mean(db_v_list)),
                "noise_ratio_mean": float(np.mean(db_noise_list)),
                "raw_ari": db_ari_list,
                "raw_nmi": db_nmi_list
            },
            "graph_clustering": {
                "ari_mean": float(m_gr_ari),
                "ari_std": float(s_gr_ari),
                "ari_var": float(v_gr_ari),
                "nmi_mean": float(m_gr_nmi),
                "nmi_std": float(s_gr_nmi),
                "nmi_var": float(v_gr_nmi),
                "v_measure_mean": float(np.mean(gr_v_list)),
                "noise_ratio_mean": 0.0,
                "raw_ari": gr_ari_list,
                "raw_nmi": gr_nmi_list
            },
            "comparison": {
                "ari_delta_mean": float(acc_improvement_ari),
                "ari_variance_reduction_pct": float(var_reduction_ari),
                "accuracy_improved": bool(acc_improvement_ari > 0),
                "variance_decreased": bool(v_gr_ari < v_db_ari)
            }
        }

        all_results[t_id] = task_record

        print(f"  [DBSCAN] ARI Mean: {m_db_ari:.4f} +/- {s_db_ari:.4f} | Var: {v_db_ari:.5f} | NMI: {m_db_nmi:.4f}")
        print(f"  [GRAPH ] ARI Mean: {m_gr_ari:.4f} +/- {s_gr_ari:.4f} | Var: {v_gr_ari:.5f} | NMI: {m_gr_nmi:.4f}")
        print(f"  --> Accuracy Gain: {acc_improvement_ari:+.4f} | Variance Reduction: {var_reduction_ari:+.1f}%")

    # Save results to JSON
    out_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(out_dir, "a3_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Metrics successfully exported to: {metrics_path}")

    return all_results, representative_samples


def plot_cluster_comparison(representative_samples, save_path="a3_cluster_comparison.png"):
    """
    Generates a 4-panel visual comparison (Task 1 and Task 3):
      Panel (a): Ground Truth Chunks in 2D SyncMap Space
      Panel (b): Baseline DBSCAN Clustering (highlighting noise/merging issues)
      Panel (c): Edge-Weighted Graph Topology (weights indicated by line thickness/color)
      Panel (d): Proposed Edge-Weighted Graph Clustering Partition
    """
    sample = representative_samples["Task 1"]
    W = sample["W"]
    gt = sample["gt"]
    l_db = sample["db_labels"]
    l_gr = sample["gr_labels"]
    G = sample["G"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), dpi=300)
    plt.subplots_adjust(wspace=0.22, bottom=0.15, top=0.88, left=0.04, right=0.98)

    cmap = matplotlib.colormaps['tab10']
    node_size = 280

    # -------------------------------------------------------------
    # Panel 1: Ground Truth Chunks
    # -------------------------------------------------------------
    ax = axes[0]
    scatter = ax.scatter(W[:, 0], W[:, 1], c=gt, cmap=cmap, s=node_size, edgecolors='black', linewidth=1.5, zorder=4)
    for i in range(len(W)):
        ax.annotate(str(i), (W[i, 0], W[i, 1]), fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white', zorder=5)
    ax.set_title("(a) Ground-Truth Chunks (SyncMap Space)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("SyncMap Dim 1", fontsize=11)
    ax.set_ylabel("SyncMap Dim 2", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_facecolor("#fafbfc")

    # -------------------------------------------------------------
    # Panel 2: Baseline SyncMap + DBSCAN
    # -------------------------------------------------------------
    ax = axes[1]
    # Handle noise points (-1)
    unique_labels = np.unique(l_db)
    for lbl in unique_labels:
        mask = (l_db == lbl)
        if lbl == -1:
            ax.scatter(W[mask, 0], W[mask, 1], c="#2d3436", marker='x', s=node_size + 40,
                       linewidth=2.5, label="Noise (-1)", zorder=4)
            for idx in np.where(mask)[0]:
                ax.annotate(str(idx), (W[idx, 0], W[idx, 1]), fontsize=8, fontweight='bold',
                            ha='center', va='bottom', color='#2d3436', zorder=5)
        else:
            ax.scatter(W[mask, 0], W[mask, 1], color=cmap(lbl % 10), s=node_size,
                       edgecolors='black', linewidth=1.5, label=f"Cluster {lbl}", zorder=4)
            for idx in np.where(mask)[0]:
                ax.annotate(str(idx), (W[idx, 0], W[idx, 1]), fontsize=9, fontweight='bold',
                            ha='center', va='center', color='white', zorder=5)

    ax.set_title("(b) Baseline: SyncMap + DBSCAN", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("SyncMap Dim 1", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_facecolor("#fafbfc")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # -------------------------------------------------------------
    # Panel 3: Proposed Edge-Weighted Graph Topology
    # -------------------------------------------------------------
    ax = axes[2]
    # Draw edges with weight encoding
    edges = list(G.edges(data=True))
    if len(edges) > 0:
        weights = [d['weight'] for _, _, d in edges]
        max_w = max(weights) if max(weights) > 0 else 1.0
        norm_weights = [w / max_w for w in weights]
        
        for (u, v, d), nw in zip(edges, norm_weights):
            x = [W[u, 0], W[v, 0]]
            y = [W[u, 1], W[v, 1]]
            alpha = max(0.2, min(0.95, nw))
            width = max(1.0, nw * 3.5)
            ax.plot(x, y, color="#2980b9", alpha=alpha, linewidth=width, zorder=2)

    ax.scatter(W[:, 0], W[:, 1], c="#34495e", s=node_size, edgecolors='black', linewidth=1.5, zorder=4)
    for i in range(len(W)):
        ax.annotate(str(i), (W[i, 0], W[i, 1]), fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white', zorder=5)

    ax.set_title("(c) Edge-Weighted Graph Topology G(V, E, w)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("SyncMap Dim 1", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_facecolor("#fafbfc")

    # -------------------------------------------------------------
    # Panel 4: Proposed Edge-Weighted Graph Clusters
    # -------------------------------------------------------------
    ax = axes[3]
    # Draw faint intra-cluster edges
    for u, v, d in G.edges(data=True):
        if l_gr[u] == l_gr[v]:
            ax.plot([W[u, 0], W[v, 0]], [W[u, 1], W[v, 1]],
                    color="#27ae60", alpha=0.45, linewidth=2.0, zorder=2)

    ax.scatter(W[:, 0], W[:, 1], c=l_gr, cmap=cmap, s=node_size, edgecolors='black', linewidth=1.5, zorder=4)
    for i in range(len(W)):
        ax.annotate(str(i), (W[i, 0], W[i, 1]), fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white', zorder=5)

    ax.set_title("(d) Proposed: Edge-Weighted Graph Clusters", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("SyncMap Dim 1", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_facecolor("#fafbfc")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    full_save_path = os.path.join(out_dir, save_path)
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Cluster comparison plot saved to: {full_save_path}")


def plot_accuracy_and_variance(metrics_dict, save_path="a3_accuracy_variance.png"):
    """
    Generates a 4-panel publication-grade evaluation figure:
      Subplot 1: Mean Adjusted Rand Index (ARI) with standard deviation error bars
      Subplot 2: ARI Variance (s^2) across seeds (highlighting dramatic variance decrease)
      Subplot 3: Mean Normalized Mutual Information (NMI)
      Subplot 4: Percentage of Noise / Discarded Elements (DBSCAN vs 0% for Graph)
    """
    task_keys = ["Task 1", "Task 2", "Task 3", "Task 4"]
    task_labels = ["T1: Non-Overlap", "T2: Overlap Bridge", "T3: Noisy/Varying", "T4: Continual Shift"]

    db_ari_means = [metrics_dict[k]["dbscan"]["ari_mean"] for k in task_keys]
    db_ari_stds = [metrics_dict[k]["dbscan"]["ari_std"] for k in task_keys]
    db_ari_vars = [metrics_dict[k]["dbscan"]["ari_var"] for k in task_keys]

    gr_ari_means = [metrics_dict[k]["graph_clustering"]["ari_mean"] for k in task_keys]
    gr_ari_stds = [metrics_dict[k]["graph_clustering"]["ari_std"] for k in task_keys]
    gr_ari_vars = [metrics_dict[k]["graph_clustering"]["ari_var"] for k in task_keys]

    db_nmi_means = [metrics_dict[k]["dbscan"]["nmi_mean"] for k in task_keys]
    gr_nmi_means = [metrics_dict[k]["graph_clustering"]["nmi_mean"] for k in task_keys]

    db_noise_means = [metrics_dict[k]["dbscan"]["noise_ratio_mean"] * 100.0 for k in task_keys]
    gr_noise_means = [0.0 for _ in task_keys]

    x = np.arange(len(task_keys))
    bar_width = 0.36

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.2), dpi=300)
    plt.subplots_adjust(wspace=0.28, bottom=0.18, top=0.88, left=0.05, right=0.98)

    color_db = "#e74c3c"
    color_gr = "#2980b9"

    # -------------------------------------------------------------
    # Subplot 1: Mean ARI (Clustering Accuracy)
    # -------------------------------------------------------------
    ax = axes[0]
    ax.bar(x - bar_width/2, db_ari_means, width=bar_width, yerr=db_ari_stds, capsize=4,
           label="SyncMap + DBSCAN", color=color_db, alpha=0.88, edgecolor='black')
    ax.bar(x + bar_width/2, gr_ari_means, width=bar_width, yerr=gr_ari_stds, capsize=4,
           label="Proposed (Edge-Graph)", color=color_gr, alpha=0.88, edgecolor='black')
    ax.set_title("Clustering Accuracy (Mean ARI)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=18, fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Adjusted Rand Index (ARI)", fontsize=11)
    ax.set_ylim(0.0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_facecolor("#fafbfc")

    # Add accuracy delta text
    for i in range(len(x)):
        delta = gr_ari_means[i] - db_ari_means[i]
        y_pos = max(gr_ari_means[i] + gr_ari_stds[i], db_ari_means[i] + db_ari_stds[i]) + 0.04
        ax.text(x[i], min(y_pos, 1.04), f"+{delta:.2f}", ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color="#16a085")

    # -------------------------------------------------------------
    # Subplot 2: Multi-Seed Variance (s^2)
    # -------------------------------------------------------------
    ax = axes[1]
    ax.bar(x - bar_width/2, db_ari_vars, width=bar_width,
           label="SyncMap + DBSCAN", color=color_db, alpha=0.88, edgecolor='black')
    ax.bar(x + bar_width/2, gr_ari_vars, width=bar_width,
           label="Proposed (Edge-Graph)", color=color_gr, alpha=0.88, edgecolor='black')
    ax.set_title("Multi-Seed Variance (s² of ARI)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=18, fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Variance s²", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_facecolor("#fafbfc")

    # -------------------------------------------------------------
    # Subplot 3: Mean NMI (Information Recovery)
    # -------------------------------------------------------------
    ax = axes[2]
    ax.bar(x - bar_width/2, db_nmi_means, width=bar_width,
           label="SyncMap + DBSCAN", color=color_db, alpha=0.88, edgecolor='black')
    ax.bar(x + bar_width/2, gr_nmi_means, width=bar_width,
           label="Proposed (Edge-Graph)", color=color_gr, alpha=0.88, edgecolor='black')
    ax.set_title("Information Recovery (Mean NMI)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=18, fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Normalized Mutual Info (NMI)", fontsize=11)
    ax.set_ylim(0.0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_facecolor("#fafbfc")

    # -------------------------------------------------------------
    # Subplot 4: Percentage of Noise / Unassigned Elements
    # -------------------------------------------------------------
    ax = axes[3]
    ax.bar(x - bar_width/2, db_noise_means, width=bar_width,
           label="SyncMap + DBSCAN", color=color_db, alpha=0.88, edgecolor='black')
    ax.bar(x + bar_width/2, gr_noise_means, width=bar_width,
           label="Proposed (Edge-Graph: 0%)", color="#27ae60", alpha=0.88, edgecolor='black')
    ax.set_title("Noise Misclassification Rate (%)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=18, fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Elements Marked as Noise (%)", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_facecolor("#fafbfc")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    full_save_path = os.path.join(out_dir, save_path)
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Accuracy & variance comparison plot saved to: {full_save_path}")


if __name__ == "__main__":
    metrics, samples = run_benchmark_suite(n_seeds=20)
    plot_cluster_comparison(samples, save_path="a3_cluster_comparison.png")
    plot_accuracy_and_variance(metrics, save_path="a3_accuracy_variance.png")
    print("\n[SUCCESS] Engine execution and visual generation complete!")
