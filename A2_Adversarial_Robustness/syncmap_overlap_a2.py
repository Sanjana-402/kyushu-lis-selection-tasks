"""
syncmap_overlap_a2.py
Task [A2]: Continual General Chunking Problem and SyncMap
Candidate: Sanjana Laxman
Lab: Laboratory of Intelligent Systems, Kyushu University
Advisor: Prof. Danilo Vasconcellos Vargas

Reference repository mentioned for SyncMap: https://github.com/zweifel/SyncMap
This script provides a self-contained NumPy/scikit-learn reproduction and extension
of the CGCP (Overlap 1 and Overlap 2) dynamics rather than a direct copy of that repo.

Key optimisation: iterate over CHUNKS (each of length time_delay), not individual
timesteps. Within a chunk-window all inputs are identical (same node fires), so we
apply that update `time_delay` times in one shot -- reducing the Python loop from
25000 to 2500 iterations per model/seed.
"""

import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Sequence generators â€” return list of (node_index, repeat_count) events
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def gen_events_1(length=25000, time_delay=10):
    """Overlap 1: 8 nodes, chunks {0-4} and {3-7}"""
    n = 8
    true_labels = [0, 0, 0, 1, 1, 2, 2, 2]
    chunk_a = [0, 1, 2, 3, 4]
    chunk_b = [3, 4, 5, 6, 7]
    events = []
    total = 0
    while total < length:
        chunk = chunk_a if np.random.rand() < 0.5 else chunk_b
        for idx in np.random.permutation(len(chunk)):
            events.append((chunk[idx], time_delay))
            total += time_delay
    return events, true_labels, n


def gen_events_2(length=25000, time_delay=10):
    """Overlap 2: 10 nodes, chunks {0-4} and {0-9}"""
    n = 10
    true_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    chunk_a = list(range(5))
    chunk_b = list(range(10))
    events = []
    total = 0
    while total < length:
        chunk = chunk_a if np.random.rand() < 0.5 else chunk_b
        for idx in np.random.permutation(len(chunk)):
            events.append((chunk[idx], time_delay))
            total += time_delay
    return events, true_labels, n


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Baseline SyncMap â€” event-driven (one Python iter per chunk-element)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_baseline(n, events, lr_mult=0.001, space_size=10.0, eps=3.0, dims=2):
    lr = lr_mult * n
    smap = np.random.rand(n, dims) * space_size

    for node_idx, repeats in events:
        x = np.zeros(n)
        x[node_idx] = 1.0
        plus_mask  = x > 0.1          # only node_idx is True
        minus_mask = ~plus_mask

        n_plus  = int(plus_mask.sum())
        n_minus = int(minus_mask.sum())
        if n_plus == 0 or n_minus == 0:
            continue

        center_plus  = smap[plus_mask].mean(0)   # just smap[node_idx]
        center_minus = smap[minus_mask].mean(0)

        # Active node update (vectorised over repeats: apply lr*repeats at once)
        diff_a = center_plus  - smap[plus_mask]
        dist_a = np.linalg.norm(diff_a, axis=1, keepdims=True).clip(min=1e-9)
        diff_r = smap[plus_mask] - center_minus
        dist_r = np.linalg.norm(diff_r, axis=1, keepdims=True).clip(min=1e-9)
        update_plus = diff_a / dist_a - diff_r / dist_r   # unit force, weight=1

        # Inactive nodes
        diff_i = smap[minus_mask] - center_plus
        dist_i = np.linalg.norm(diff_i, axis=1, keepdims=True).clip(min=1e-9)
        update_minus = diff_i / dist_i

        # Apply repeats steps at once (approximation valid for small lr*repeats)
        effective_lr = lr * repeats
        new_smap = smap.copy()
        new_smap[plus_mask]  += effective_lr * update_plus
        new_smap[minus_mask] += effective_lr * update_minus

        mx = np.abs(new_smap).max()
        if mx > 1e-9:
            new_smap *= space_size / mx
        smap = new_smap

    labels = DBSCAN(eps=eps, min_samples=2).fit_predict(smap)
    return smap, labels


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Improved SyncMap: SymmetricalSyncMapMA â€” event-driven
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_improved(n, events, lr_mult=0.0025, space_size=10.0, eps=1.6, dims=2, cooc_decay=0.92, repel_weight=0.65, ema_decay=0.95):
    import numpy as np; from sklearn.cluster import DBSCAN
    lr=lr_mult*n; smap=np.random.rand(n,dims)*space_size; ema=smap.copy(); recent=np.zeros(n,dtype=float)
    for node_idx,repeats in events:
        recent*=(cooc_decay**repeats); recent[node_idx]=1.0
        w=recent.copy(); w[node_idx]=0.0
        c_cooc=(w[:,None]*smap).sum(0)/w.sum() if w.sum()>1e-9 else smap[node_idx]
        mm=np.ones(n,dtype=bool); mm[node_idx]=False
        nc=mm&(recent<0.25)
        if nc.sum()>0:
            rv=smap[node_idx]-smap[nc].mean(0); rv/=max(float(np.linalg.norm(rv)),1e-9)
        else:
            rv=np.zeros(dims)
        av=c_cooc-smap[node_idx]; av/=max(float(np.linalg.norm(av)),1e-9)
        ua=av+repel_weight*rv
        dm=smap[mm]-smap[node_idx]; dist=np.linalg.norm(dm,axis=1,keepdims=True).clip(min=1e-9); dir_m=dm/dist; wm=recent[mm,None]
        um=np.where(wm>=0.25,-0.5*wm*dir_m,0.35*repel_weight*dir_m)
        elr=lr*repeats; ns=smap.copy(); ns[node_idx]+=elr*ua; ns[mm]+=elr*um
        mx=np.abs(ns).max()
        if mx>1e-9: ns*=space_size/mx
        smap=ns; dr=ema_decay**repeats; ema=dr*ema+(1.0-dr)*smap
    return ema, DBSCAN(eps=eps,min_samples=2).fit_predict(ema)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Evaluation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def remap_noise(pred):
    ctr = (max(pred) if len(pred) > 0 else 0) + 1
    out = []
    for lbl in pred:
        if lbl == -1:
            out.append(ctr); ctr += 1
        else:
            out.append(lbl)
    return np.array(out)


def evaluate(run_fn, gen_fn, n_seeds, seq_len, time_delay, **kw):
    nmis, aris = [], []
    last_map = last_lbl = last_true = None
    for seed in range(n_seeds):
        np.random.seed(seed)
        events, true_labels, n = gen_fn(length=seq_len, time_delay=time_delay)
        smap, pred = run_fn(n, events, **kw)
        pred_rm = remap_noise(pred)
        nmi = normalized_mutual_info_score(true_labels, pred_rm)
        ari = adjusted_rand_score(true_labels, pred_rm)
        nmis.append(nmi); aris.append(ari)
        last_map, last_lbl, last_true = smap, pred, np.array(true_labels)
        print(f"    seed {seed:2d}: NMI={nmi*100:.1f}%  ARI={ari:.3f}")
        sys.stdout.flush()
    return np.array(nmis), np.array(aris), last_map, last_lbl, last_true


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Figure 1: 2D scatter (baseline vs improved, seed=last)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_map_comparison(viz, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle("SyncMap 2D Spatial Organisation (last seed)",
                 fontsize=14, fontweight="bold")
    panels = [
        ("Overlap 1 -- Baseline",    viz["ov1_base_map"], viz["ov1_base_lbl"], viz["ov1_true"]),
        ("Overlap 1 -- Improved",    viz["ov1_impr_map"], viz["ov1_impr_lbl"], viz["ov1_true"]),
        ("Overlap 2 -- Baseline",    viz["ov2_base_map"], viz["ov2_base_lbl"], viz["ov2_true"]),
        ("Overlap 2 -- Improved",    viz["ov2_impr_map"], viz["ov2_impr_lbl"], viz["ov2_true"]),
    ]
    cmap = matplotlib.colormaps.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "P", "*"]
    for ax, (title, smap, pred_lbl, true_lbl) in zip(axes.flat, panels):
        ax.set_title(title, fontsize=9, fontweight="bold")
        for i in range(len(true_lbl)):
            tc = cmap(int(true_lbl[i]) % 10)
            pl = int(pred_lbl[i])
            mk = markers[pl % len(markers)] if pl >= 0 else "x"
            ax.scatter(smap[i, 0], smap[i, 1], c=[tc], marker=mk,
                       s=140, edgecolors="k", linewidths=0.6, zorder=3)
            ax.annotate(f"n{i}", (smap[i, 0], smap[i, 1]), fontsize=7,
                        ha="center", va="bottom", xytext=(0, 5),
                        textcoords="offset points")
        patches = [mpatches.Patch(color=cmap(int(tl) % 10),
                                  label=f"Class {tl}") for tl in np.unique(true_lbl)]
        ax.legend(handles=patches, fontsize=7, loc="upper right")
        ax.set_xlabel("Dim 1", fontsize=8); ax.set_ylabel("Dim 2", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Figure 2: NMI summary
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_nmi_summary(metrics, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("SyncMap NMI: Baseline vs SymmetricalSyncMapMA",
                 fontsize=13, fontweight="bold")
    seeds = list(range(len(metrics["ov1_base_nmis"])))
    bc, ic = "#4878CF", "#6ACC65"

    for ax, ovk, title in [
        (axes[0, 0], "ov1", "Overlap 1: NMI per Seed"),
        (axes[0, 1], "ov2", "Overlap 2: NMI per Seed"),
    ]:
        ax.plot(seeds, [v*100 for v in metrics[f"{ovk}_base_nmis"]], "o--",
                color=bc, label="Baseline", linewidth=1.5)
        ax.plot(seeds, [v*100 for v in metrics[f"{ovk}_impr_nmis"]], "s-",
                color=ic, label="Improved",  linewidth=1.5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Seed"); ax.set_ylabel("NMI (%)")
        ax.set_ylim(0, 105); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    lbls = ["OV1\nBaseline", "OV1\nImproved", "OV2\nBaseline", "OV2\nImproved"]
    vals = [metrics["ov1_base_mean"]*100, metrics["ov1_impr_mean"]*100,
            metrics["ov2_base_mean"]*100, metrics["ov2_impr_mean"]*100]
    errs = [metrics["ov1_base_std"]*100,  metrics["ov1_impr_std"]*100,
            metrics["ov2_base_std"]*100,  metrics["ov2_impr_std"]*100]
    bclrs = [bc, ic, bc, ic]
    bars = ax.bar(lbls, vals, yerr=errs, capsize=5,
                  color=bclrs, edgecolor="k", linewidth=0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Mean NMI (+/- std)", fontweight="bold")
    ax.set_ylabel("NMI (%)"); ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]; ax.axis("off")
    ov1_abs = metrics["ov1_abs_gain_pp"]; ov1_rel = metrics["ov1_rel_gain_pct"]
    ov2_abs = metrics["ov2_abs_gain_pp"]; ov2_rel = metrics["ov2_rel_gain_pct"]
    p1 = "PASS" if ov1_abs > 5 and ov1_rel > 5 else "FAIL"
    p2 = "PASS" if ov2_abs > 5 and ov2_rel > 5 else "FAIL"
    rows = [
        ["Metric", "Overlap 1", "Overlap 2"],
        ["Baseline NMI",
         f"{metrics['ov1_base_mean']*100:.2f}% +/-{metrics['ov1_base_std']*100:.2f}",
         f"{metrics['ov2_base_mean']*100:.2f}% +/-{metrics['ov2_base_std']*100:.2f}"],
        ["Improved NMI",
         f"{metrics['ov1_impr_mean']*100:.2f}% +/-{metrics['ov1_impr_std']*100:.2f}",
         f"{metrics['ov2_impr_mean']*100:.2f}% +/-{metrics['ov2_impr_std']*100:.2f}"],
        ["Abs. Gain (pp)", f"+{ov1_abs:.2f}", f"+{ov2_abs:.2f}"],
        ["Rel. Gain (%)",  f"+{ov1_rel:.2f}%", f"+{ov2_rel:.2f}%"],
        [">5% Check",      f"[{p1}]",           f"[{p2}]"],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
        elif r == 5:
            txt = cell.get_text().get_text()
            cell.set_facecolor("#D5F5E3" if "PASS" in txt else "#FADBD8")
        elif r % 2 == 0:
            cell.set_facecolor("#F2F3F4")
    ax.set_title("Quantitative Summary", fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    N_SEEDS = 10; SEQ_LEN = 25000; TIME_DLY = 10; EPS = 3.0

    print("=" * 60)
    print("Task [A2] -- Continual General Chunking Problem & SyncMap")
    print("Candidate: Sanjana Laxman")
    print("=" * 60)
    sys.stdout.flush()

    print("\n[Overlap 1] Baseline ...")
    sys.stdout.flush()
    ov1_b_nmis, ov1_b_aris, ov1_b_map, ov1_b_lbl, ov1_true = evaluate(
        run_baseline, gen_events_1, N_SEEDS, SEQ_LEN, TIME_DLY, eps=EPS)
    print(f"  => {ov1_b_nmis.mean()*100:.2f}% +/- {ov1_b_nmis.std()*100:.2f}%"); sys.stdout.flush()

    print("\n[Overlap 1] Improved ...")
    sys.stdout.flush()
    ov1_i_nmis, ov1_i_aris, ov1_i_map, ov1_i_lbl, _ = evaluate(
        run_improved, gen_events_1, N_SEEDS, SEQ_LEN, TIME_DLY, eps=EPS)
    print(f"  => {ov1_i_nmis.mean()*100:.2f}% +/- {ov1_i_nmis.std()*100:.2f}%"); sys.stdout.flush()

    ov1_abs = (ov1_i_nmis.mean() - ov1_b_nmis.mean()) * 100
    ov1_rel = ov1_abs / (ov1_b_nmis.mean() * 100 + 1e-9) * 100

    print("\n[Overlap 2] Baseline ...")
    sys.stdout.flush()
    ov2_b_nmis, ov2_b_aris, ov2_b_map, ov2_b_lbl, ov2_true = evaluate(
        run_baseline, gen_events_2, N_SEEDS, SEQ_LEN, TIME_DLY, eps=EPS)
    print(f"  => {ov2_b_nmis.mean()*100:.2f}% +/- {ov2_b_nmis.std()*100:.2f}%"); sys.stdout.flush()

    print("\n[Overlap 2] Improved ...")
    sys.stdout.flush()
    ov2_i_nmis, ov2_i_aris, ov2_i_map, ov2_i_lbl, _ = evaluate(
        run_improved, gen_events_2, N_SEEDS, SEQ_LEN, TIME_DLY, eps=EPS)
    print(f"  => {ov2_i_nmis.mean()*100:.2f}% +/- {ov2_i_nmis.std()*100:.2f}%"); sys.stdout.flush()

    ov2_abs = (ov2_i_nmis.mean() - ov2_b_nmis.mean()) * 100
    ov2_rel = ov2_abs / (ov2_b_nmis.mean() * 100 + 1e-9) * 100

    # Save metrics
    metrics = {
        "task": "A2 -- Continual General Chunking Problem and SyncMap",
        "candidate": "Sanjana Laxman",
        "n_seeds": N_SEEDS, "seq_len": SEQ_LEN,
        "time_delay": TIME_DLY, "eps": EPS,
        "ov1_base_nmis": ov1_b_nmis.tolist(), "ov1_impr_nmis": ov1_i_nmis.tolist(),
        "ov1_base_aris": ov1_b_aris.tolist(), "ov1_impr_aris": ov1_i_aris.tolist(),
        "ov1_base_mean": float(ov1_b_nmis.mean()), "ov1_base_std": float(ov1_b_nmis.std()),
        "ov1_impr_mean": float(ov1_i_nmis.mean()), "ov1_impr_std": float(ov1_i_nmis.std()),
        "ov1_abs_gain_pp": float(ov1_abs), "ov1_rel_gain_pct": float(ov1_rel),
        "ov2_base_nmis": ov2_b_nmis.tolist(), "ov2_impr_nmis": ov2_i_nmis.tolist(),
        "ov2_base_aris": ov2_b_aris.tolist(), "ov2_impr_aris": ov2_i_aris.tolist(),
        "ov2_base_mean": float(ov2_b_nmis.mean()), "ov2_base_std": float(ov2_b_nmis.std()),
        "ov2_impr_mean": float(ov2_i_nmis.mean()), "ov2_impr_std": float(ov2_i_nmis.std()),
        "ov2_abs_gain_pp": float(ov2_abs), "ov2_rel_gain_pct": float(ov2_rel),
    }
    jp = os.path.join(base_dir, "a2_metrics.json")
    with open(jp, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved: {jp}"); sys.stdout.flush()

    # Figures
    print("\n[Figures] ...")
    sys.stdout.flush()
    viz = {
        "ov1_base_map": ov1_b_map, "ov1_base_lbl": ov1_b_lbl, "ov1_true": ov1_true,
        "ov1_impr_map": ov1_i_map, "ov1_impr_lbl": ov1_i_lbl,
        "ov2_base_map": ov2_b_map, "ov2_base_lbl": ov2_b_lbl, "ov2_true": ov2_true,
        "ov2_impr_map": ov2_i_map, "ov2_impr_lbl": ov2_i_lbl,
    }
    plot_map_comparison(viz, os.path.join(base_dir, "a2_attack_examples.png"))
    plot_nmi_summary(metrics,  os.path.join(base_dir, "a2_robustness_summary.png"))

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    for label, ag, rg in [("Overlap 1", ov1_abs, ov1_rel), ("Overlap 2", ov2_abs, ov2_rel)]:
        ok = ag > 5.0 and rg > 5.0
        print(f"  {label}: abs={ag:+.2f} pp  rel={rg:+.2f}%  [{'PASS' if ok else 'FAIL'}]")
    all_pass = ov1_abs > 5 and ov1_rel > 5 and ov2_abs > 5 and ov2_rel > 5
    print("\n  >>> ALL CHECKS PASSED <<<" if all_pass else "\n  !!! WARNING: FAILED !!!")
    print("=" * 60)
    sys.stdout.flush()

if __name__ == "__main__":
    main()

