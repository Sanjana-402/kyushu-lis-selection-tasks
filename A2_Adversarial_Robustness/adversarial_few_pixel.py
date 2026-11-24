"""
Advanced Task [A2]: Adversarial Attacks & Neural Network Robustness
Few-Pixel Black-Box Differential Evolution Attack & Lightweight Defense Evaluation

Candidate: Sanjana Laxman
Target: Laboratory of Intelligent Systems, Kyushu University
Advisor: Prof. Danilo Vasconcellos Vargas

This module implements:
1. Multi-class 16x16 synthetic visual pattern dataset (5 geometric classes).
2. Multi-Layer Perceptron (MLPClassifier) neural network classifier.
3. Black-box Differential Evolution few-pixel attack for pixel budgets d in {1, 3, 5}.
4. Lightweight spatial smoothing + bit-depth quantization preprocessing defense.
5. Export of a2_metrics.json, a2_attack_examples.png, and a2_robustness_summary.png.
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter, convolve
from scipy.optimize import differential_evolution

# Ensure deterministic reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CLASS_NAMES = [
    "Vertical Bar",
    "Horizontal Bar",
    "Diagonal Cross",
    "Center Block",
    "Corner Frame"
]

def generate_pattern_dataset(n_samples_per_class=350, seed=42):
    """
    Generates a reproducible 16x16 multi-class image dataset across 5 visual pattern classes.
    Each sample contains realistic sensor background noise and geometric structure variations.
    """
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    
    for _ in range(n_samples_per_class):
        for c in range(5):
            # Baseline background with subtle sensor noise
            img = rng.normal(0.24, 0.08, (16, 16))
            val = rng.uniform(0.34, 0.52)
            
            if c == 0:  # Vertical Bar
                col = rng.integers(5, 11)
                r_st = rng.integers(3, 5)
                img[r_st:r_st+9, col] += val
            elif c == 1:  # Horizontal Bar
                row = rng.integers(5, 11)
                c_st = rng.integers(3, 5)
                img[row, c_st:c_st+9] += val
            elif c == 2:  # Diagonal Cross
                for i in range(4, 12):
                    img[i, i] += val
                    img[i, 15 - i] += val
            elif c == 3:  # Center Block
                r0 = rng.integers(6, 8)
                c0 = rng.integers(6, 8)
                img[r0:r0+3, c0:c0+3] += val
            elif c == 4:  # Corner Frame
                img[2:4, 2:4] += val
                img[2:4, 12:14] += val
                img[12:14, 2:4] += val
                img[12:14, 12:14] += val
                
            images.append(np.clip(img, 0.0, 1.0))
            labels.append(c)
            
    X = np.array(images, dtype=np.float32).reshape(len(images), -1)
    y = np.array(labels, dtype=np.int64)
    return X, y

def apply_defense(img_2d, bit_depth=4):
    """
    Lightweight preprocessing defense:
    1. 3x3 spatial smoothing filter (uniform box filter with reflection padding).
    2. Bit-depth quantization (2^b levels) to neutralize high-frequency pixel spikes.
    """
    smoothed = uniform_filter(img_2d, size=3, mode='reflect')
    levels = (2 ** bit_depth) - 1
    quantized = np.round(smoothed * levels) / levels
    return np.clip(quantized, 0.0, 1.0)

def run_de_few_pixel_attack(orig_img_2d, true_label, classifier, d_budget, seed=42):
    """
    Black-box Differential Evolution few-pixel attack for pixel budget d.
    Optimizes d tuples of (row, col, intensity) using only predicted probabilities.
    
    Decision vector: theta in R^(3*d)
      row in [0, 15.999]
      col in [0, 15.999]
      val in [0.0, 1.0]
    """
    bounds = []
    for _ in range(d_budget):
        bounds.extend([(0, 15.999), (0, 15.999), (0.0, 1.0)])
        
    def objective(theta):
        im = orig_img_2d.copy()
        for k in range(d_budget):
            r = int(theta[3 * k])
            c = int(theta[3 * k + 1])
            v = float(np.clip(theta[3 * k + 2], 0.0, 1.0))
            im[r, c] = v
        prob = classifier.predict_proba(im.reshape(1, -1))[0]
        # Untargeted objective: minimize true class probability relative to runner-up
        other_probs = [prob[j] for j in range(5) if j != true_label]
        return prob[true_label] - max(other_probs)
        
    def early_stop_callback(xk, *args):
        im = orig_img_2d.copy()
        for k in range(d_budget):
            r = int(xk[3 * k])
            c = int(xk[3 * k + 1])
            v = float(np.clip(xk[3 * k + 2], 0.0, 1.0))
            im[r, c] = v
        pred = classifier.predict(im.reshape(1, -1))[0]
        return pred != true_label  # Halt once an adversarial perturbation is found

    t_start = time.time()
    res = differential_evolution(
        objective,
        bounds=bounds,
        callback=early_stop_callback,
        popsize=12,
        maxiter=25,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed
    )
    t_elapsed = time.time() - t_start
    
    # Reconstruct best adversarial candidate
    adv_img_2d = orig_img_2d.copy()
    for k in range(d_budget):
        r = int(res.x[3 * k])
        c = int(res.x[3 * k + 1])
        v = float(np.clip(res.x[3 * k + 2], 0.0, 1.0))
        adv_img_2d[r, c] = v
        
    adv_probs = classifier.predict_proba(adv_img_2d.reshape(1, -1))[0]
    adv_pred = int(np.argmax(adv_probs))
    is_success = (adv_pred != true_label)
    
    # Calculate perturbation norms
    diff = adv_img_2d - orig_img_2d
    l0_norm = int(np.count_nonzero(np.abs(diff) > 1e-4))
    l2_norm = float(np.linalg.norm(diff))
    linf_norm = float(np.max(np.abs(diff)))
    
    return {
        "success": is_success,
        "adv_img": adv_img_2d,
        "adv_pred": adv_pred,
        "adv_conf": float(adv_probs[adv_pred]),
        "orig_conf": float(classifier.predict_proba(orig_img_2d.reshape(1, -1))[0][true_label]),
        "l0": l0_norm,
        "l2": l2_norm,
        "linf": linf_norm,
        "nit": int(res.nit),
        "time": t_elapsed
    }

def main():
    print("=" * 70)
    print("Task [A2]: Adversarial Robustness & Few-Pixel Differential Evolution Attack")
    print("Candidate: Sanjana Laxman | Lab of Intelligent Systems, Kyushu University")
    print("=" * 70)
    
    # 1. Dataset Generation
    print("\n[Step 1] Generating synthetic visual pattern dataset (16x16, 5 classes)...")
    X, y = generate_pattern_dataset(n_samples_per_class=350, seed=RANDOM_SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    for c_id, c_name in enumerate(CLASS_NAMES):
        print(f"    Class {c_id}: {c_name:<16} (Train: {np.sum(y_train == c_id)}, Test: {np.sum(y_test == c_id)})")
        
    # 2. Train Neural Classifier (MLPClassifier)
    print("\n[Step 2] Training Neural Classifier (MLPClassifier)...")
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=300,
        random_state=RANDOM_SEED
    )
    clf.fit(X_train, y_train)
    clean_test_acc = float(clf.score(X_test, y_test))
    print(f"  Clean Test Accuracy: {clean_test_acc * 100:.2f}%")
    
    # 3. Evaluate Lightweight Defense on Clean Test Set
    print("\n[Step 3] Evaluating Preprocessing Defense (3x3 Smoothing + 4-bit Quantization)...")
    X_test_2d = X_test.reshape(-1, 16, 16)
    X_test_defended = np.array([apply_defense(im, bit_depth=4) for im in X_test_2d]).reshape(-1, 256)
    clean_defended_acc = float(clf.score(X_test_defended, y_test))
    print(f"  Clean Defended Accuracy: {clean_defended_acc * 100:.2f}%")
    
    # 4. Prepare Evaluation Samples for Few-Pixel Attack
    # Select 50 correctly classified test samples (10 per class) for balanced evaluation
    correct_test_indices = []
    for c in range(5):
        c_idxs = [i for i in range(len(X_test)) if y_test[i] == c and clf.predict(X_test[i:i+1])[0] == c]
        correct_test_indices.extend(c_idxs[:10])
        
    print(f"\n[Step 4] Running Black-Box Differential Evolution Few-Pixel Attack on {len(correct_test_indices)} samples...")
    print(f"  Pixel budgets evaluated: d in {{1, 3, 5}} pixels")
    
    budgets = [1, 3, 5]
    budget_results = {}
    attack_records = {d: [] for d in budgets}
    
    for d in budgets:
        print(f"\n  ---> Executing Attack for Pixel Budget d = {d} ...")
        t_d_start = time.time()
        successes = 0
        defended_correct = 0
        l0_list, l2_list, linf_list, nits, times = [], [], [], [], []
        class_vuln = {c: {"total": 0, "fooled": 0} for c in range(5)}
        
        for idx in correct_test_indices:
            orig_img = X_test[idx].reshape(16, 16)
            true_lbl = int(y_test[idx])
            class_vuln[true_lbl]["total"] += 1
            
            res = run_de_few_pixel_attack(
                orig_img_2d=orig_img,
                true_label=true_lbl,
                classifier=clf,
                d_budget=d,
                seed=RANDOM_SEED + idx * 7
            )
            
            # Defense evaluation
            adv_img = res["adv_img"]
            adv_def = apply_defense(adv_img, bit_depth=4)
            adv_def_probs = clf.predict_proba(adv_def.reshape(1, -1))[0]
            adv_def_pred = int(np.argmax(adv_def_probs))
            
            res["def_pred"] = adv_def_pred
            res["def_conf"] = float(adv_def_probs[adv_def_pred])
            res["true_label"] = true_lbl
            res["sample_idx"] = idx
            res["orig_img"] = orig_img
            res["def_img"] = adv_def
            attack_records[d].append(res)
            
            if res["success"]:
                successes += 1
                class_vuln[true_lbl]["fooled"] += 1
                if adv_def_pred == true_lbl:
                    defended_correct += 1
            else:
                # If attack failed, check if defense keeps it correct
                if adv_def_pred == true_lbl:
                    defended_correct += 1
                    
            l0_list.append(res["l0"])
            l2_list.append(res["l2"])
            linf_list.append(res["linf"])
            nits.append(res["nit"])
            times.append(res["time"])
            
        elapsed_d = time.time() - t_d_start
        n_eval = len(correct_test_indices)
        asr = (successes / n_eval) * 100.0
        post_attack_acc = ((n_eval - successes) / n_eval) * 100.0
        defended_post_attack_acc = (defended_correct / n_eval) * 100.0
        recovery_rate = (defended_correct - (n_eval - successes)) / successes * 100.0 if successes > 0 else 0.0
        
        # Calculate defense recovery among successful attacks
        recovered_attacks = sum(1 for r in attack_records[d] if r["success"] and r["def_pred"] == r["true_label"])
        recovery_rate_pct = (recovered_attacks / successes * 100.0) if successes > 0 else 0.0
        
        budget_results[d] = {
            "budget_d": d,
            "attack_success_rate_pct": float(round(asr, 2)),
            "post_attack_accuracy_pct": float(round(post_attack_acc, 2)),
            "defended_post_attack_accuracy_pct": float(round(defended_post_attack_acc, 2)),
            "defense_recovery_rate_pct": float(round(recovery_rate_pct, 2)),
            "successful_attacks": int(successes),
            "recovered_attacks": int(recovered_attacks),
            "total_attacked": int(n_eval),
            "mean_l0_norm": float(round(np.mean(l0_list), 2)),
            "mean_l2_norm": float(round(np.mean(l2_list), 4)),
            "mean_linf_norm": float(round(np.mean(linf_list), 4)),
            "mean_iterations": float(round(np.mean(nits), 2)),
            "mean_time_sec": float(round(np.mean(times), 3)),
            "class_vulnerability": {
                CLASS_NAMES[c]: {
                    "total": class_vuln[c]["total"],
                    "fooled": class_vuln[c]["fooled"],
                    "asr_pct": float(round((class_vuln[c]["fooled"] / class_vuln[c]["total"]) * 100, 1))
                } for c in range(5)
            }
        }
        
        print(f"    Budget d = {d}:")
        print(f"      Attack Success Rate (ASR):   {asr:.2f}% ({successes}/{n_eval})")
        print(f"      Post-Attack Accuracy:        {post_attack_acc:.2f}%")
        print(f"      Defended Accuracy:           {defended_post_attack_acc:.2f}%")
        print(f"      Defense Recovery Rate:       {recovery_rate_pct:.2f}% ({recovered_attacks}/{successes})")
        print(f"      Mean L0 Norm:                {np.mean(l0_list):.2f} pixels | Mean L2: {np.mean(l2_list):.4f}")
        print(f"      Execution Time:              {elapsed_d:.2f}s ({np.mean(times):.3f}s/sample)")

    # 5. Save Metrics JSON
    metrics_output = {
        "metadata": {
            "candidate": "Sanjana Laxman",
            "laboratory": "Laboratory of Intelligent Systems, Kyushu University",
            "faculty_advisor": "Prof. Danilo Vasconcellos Vargas",
            "task_id": "Task [A2] Few-Pixel Black-Box Differential Evolution & Robustness",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_resolution": [16, 16],
            "num_classes": 5,
            "class_names": CLASS_NAMES,
            "classifier_architecture": "MLP (64, 32) ReLU",
            "attack_method": "Differential Evolution (SciPy DE, Black-Box, Zero-Gradient)",
            "defense_method": "3x3 Spatial Box Smoothing + 4-bit Depth Quantization"
        },
        "model_performance": {
            "clean_test_accuracy_pct": float(round(clean_test_acc * 100, 2)),
            "clean_defended_accuracy_pct": float(round(clean_defended_acc * 100, 2)),
            "accuracy_drop_from_defense_pct": float(round((clean_test_acc - clean_defended_acc) * 100, 2))
        },
        "attack_and_defense_summary": {
            f"d_{d}": budget_results[d] for d in budgets
        }
    }
    
    with open("a2_metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    print("\n[Step 5] Saved 'a2_metrics.json' successfully.")
    
    # 6. Generate Attack Visual Examples Figure (a2_attack_examples.png)
    print("\n[Step 6] Generating Attack Diagnostics Figure 'a2_attack_examples.png'...")
    # Select 6 diverse successful attack samples across d=1, 3, 5
    selected_examples = []
    for d in [1, 3, 5]:
        succ_d = [r for r in attack_records[d] if r["success"]]
        if len(succ_d) >= 2:
            selected_examples.extend(succ_d[:2])
        elif len(succ_d) == 1:
            selected_examples.extend(succ_d)
            # Add one near-miss or candidate
            selected_examples.append(attack_records[d][0])
        else:
            selected_examples.extend(attack_records[d][:2])
            
    fig, axes = plt.subplots(len(selected_examples), 4, figsize=(11.5, 2.3 * len(selected_examples)), dpi=300)
    col_titles = [
        "Clean Input (x)",
        "Adversarial (x_adv)",
        "Perturbation Map |Delta|",
        "Defended Input (x_def)"
    ]
    
    for c_idx, title in enumerate(col_titles):
        axes[0, c_idx].set_title(title, fontsize=12, fontweight='bold', pad=10)
        
    for r_idx, ex in enumerate(selected_examples):
        d_val = ex["l0"]
        true_name = CLASS_NAMES[ex["true_label"]]
        adv_name = CLASS_NAMES[ex["adv_pred"]]
        def_name = CLASS_NAMES[ex["def_pred"]]
        
        # Clean image
        ax = axes[r_idx, 0]
        ax.imshow(ex["orig_img"], cmap='magma', vmin=0, vmax=1)
        ax.set_ylabel(f"Ex {r_idx+1} (d={d_val})", fontsize=11, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, -0.22, f"True: {true_name}\nConf: {ex['orig_conf']*100:.1f}%",
                transform=ax.transAxes, ha='center', fontsize=9, color='#1A365D', fontweight='semibold')
        
        # Adversarial image
        ax = axes[r_idx, 1]
        ax.imshow(ex["adv_img"], cmap='magma', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        status_color = '#C53030' if ex['success'] else '#2B6CB0'
        ax.text(0.5, -0.22, f"Pred: {adv_name}\nConf: {ex['adv_conf']*100:.1f}%",
                transform=ax.transAxes, ha='center', fontsize=9, color=status_color, fontweight='bold')
        
        # Perturbation map
        ax = axes[r_idx, 2]
        pert_map = np.abs(ex["adv_img"] - ex["orig_img"])
        im_pert = ax.imshow(pert_map, cmap='hot', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, -0.22, f"Mod Pixels: {ex['l0']}\nL2: {ex['l2']:.3f}, L-inf: {ex['linf']:.2f}",
                transform=ax.transAxes, ha='center', fontsize=9, color='#7B341E', fontweight='semibold')
        
        # Defended image
        ax = axes[r_idx, 3]
        ax.imshow(ex["def_img"], cmap='magma', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        def_color = '#22543D' if ex['def_pred'] == ex['true_label'] else '#C53030'
        def_status = "RECOVERED" if ex['def_pred'] == ex['true_label'] else "STILL FOOLED"
        ax.text(0.5, -0.22, f"Pred: {def_name} ({def_status})\nConf: {ex['def_conf']*100:.1f}%",
                transform=ax.transAxes, ha='center', fontsize=9, color=def_color, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, hspace=0.45)
    plt.savefig("a2_attack_examples.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved 'a2_attack_examples.png' successfully.")
    
    # 7. Generate Robustness Summary Figure (a2_robustness_summary.png)
    print("\n[Step 7] Generating Robustness Summary Figure 'a2_robustness_summary.png'...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    
    d_vals = budgets
    asrs = [budget_results[d]["attack_success_rate_pct"] for d in d_vals]
    post_accs = [budget_results[d]["post_attack_accuracy_pct"] for d in d_vals]
    def_post_accs = [budget_results[d]["defended_post_attack_accuracy_pct"] for d in d_vals]
    recs = [budget_results[d]["defense_recovery_rate_pct"] for d in d_vals]
    
    # Subplot A: Attack Success Rate vs Pixel Budget
    ax = axes[0, 0]
    ax.plot(d_vals, asrs, marker='o', markersize=8, color='#C53030', linewidth=2.5, label='Attack Success Rate (ASR)')
    for x_i, y_i in zip(d_vals, asrs):
        ax.annotate(f"{y_i:.1f}%", (x_i, y_i + 1.8), ha='center', fontsize=10, fontweight='bold', color='#C53030')
    ax.set_title("Panel A: Attack Success Rate (ASR) vs. Pixel Budget d", fontsize=11, fontweight='bold', color='#1A365D')
    ax.set_xlabel("Pixel Perturbation Budget (d)", fontsize=10)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=10)
    ax.set_xticks(d_vals)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True)
    
    # Subplot B: Accuracy Comparison Across Regimes
    ax = axes[0, 1]
    x_pos = np.arange(len(d_vals))
    width = 0.35
    ax.bar(x_pos - width/2, post_accs, width, label='Attacked (No Defense)', color='#E53E3E', alpha=0.85)
    ax.bar(x_pos + width/2, def_post_accs, width, label='Attacked (With 3x3+Quant Defense)', color='#2B6CB0', alpha=0.85)
    ax.axhline(clean_test_acc * 100, color='#2F855A', linestyle='--', linewidth=2, label=f'Clean Baseline ({clean_test_acc*100:.1f}%)')
    for i in x_pos:
        ax.text(i - width/2, post_accs[i] + 1.5, f"{post_accs[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        ax.text(i + width/2, def_post_accs[i] + 1.5, f"{def_post_accs[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
    ax.set_title("Panel B: Model Accuracy Across Defense Regimes", fontsize=11, fontweight='bold', color='#1A365D')
    ax.set_xlabel("Pixel Perturbation Budget (d)", fontsize=10)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"d = {d}" for d in d_vals])
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(loc='upper right', frameon=True)
    
    # Subplot C: Defense Recovery Rate
    ax = axes[1, 0]
    ax.bar([f"d = {d}" for d in d_vals], recs, color='#319795', width=0.55, edgecolor='#234E52', linewidth=1.2)
    for i, r_val in enumerate(recs):
        ax.text(i, r_val + 1.5, f"{r_val:.1f}%", ha='center', fontsize=10, fontweight='bold', color='#234E52')
    ax.set_title("Panel C: Defense Recovery Rate on Successful Attacks", fontsize=11, fontweight='bold', color='#1A365D')
    ax.set_xlabel("Pixel Perturbation Budget (d)", fontsize=10)
    ax.set_ylabel("Recovery Rate (% Foiled Attacks Restored)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Subplot D: Class-wise Vulnerability Breakdown (at d=5)
    ax = axes[1, 1]
    class_asrs_d5 = [budget_results[5]["class_vulnerability"][c_name]["asr_pct"] for c_name in CLASS_NAMES]
    colors = ['#4A5568', '#718096', '#DD6B20', '#3182CE', '#805AD5']
    y_pos = np.arange(len(CLASS_NAMES))
    ax.barh(y_pos, class_asrs_d5, color=colors, height=0.55, edgecolor='black', linewidth=0.8)
    for i, v in enumerate(class_asrs_d5):
        ax.text(v + 1.5, i, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold')
    ax.set_title("Panel D: Pattern Class Vulnerability Breakdown (d = 5)", fontsize=11, fontweight='bold', color='#1A365D')
    ax.set_xlabel("Attack Success Rate (%)", fontsize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.5, axis='x')
    
    plt.tight_layout()
    plt.savefig("a2_robustness_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved 'a2_robustness_summary.png' successfully.")
    
    print("\n" + "=" * 70)
    print("Execution of adversarial_few_pixel.py complete!")
    print("All artifacts generated: a2_metrics.json, a2_attack_examples.png, a2_robustness_summary.png")
    print("=" * 70)

if __name__ == "__main__":
    main()
