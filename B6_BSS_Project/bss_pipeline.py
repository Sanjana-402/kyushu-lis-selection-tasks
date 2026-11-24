"""
Blind Signal Separation (BSS) Pipeline for Task [B6]
Kyushu University - Laboratory of Intelligent Systems (Prof. Danilo V. Vargas)
Candidate: Sanjana Laxman

This script implements:
1. Synthetic audio source generation (3 distinct non-Gaussian waveforms).
2. Linear mixing simulating the Cocktail Party Problem (X = A * S).
3. Blind Signal Separation using FastICA (scikit-learn).
4. Ambiguity resolution (optimal permutation matching & sign alignment).
5. Quantitative evaluation (Pearson Correlation, SIR in dB, Kurtosis).
6. Multi-panel comparative visualization (Original, Mixed, Reconstructed).
7. Audio file export (original, mixed, separated WAV files).
8. 2-source and 3-source comparative benchmarks.
"""

import os
import json
import numpy as np
import scipy.signal
import scipy.optimize
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA


def generate_sources(fs=8000, duration=3.0):
    """
    Generate 3 distinct non-Gaussian synthetic audio signals:
    - Source 1: Pure Tone Sine Wave (261.63 Hz, Middle C)
    - Source 2: Sawtooth Wave (150 Hz)
    - Source 3: Linear Frequency-Modulated Chirp (100 Hz to 1200 Hz)
    """
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # 1. Sine wave (harmonically pure tone)
    f_sine = 261.63
    s1 = np.sin(2 * np.pi * f_sine * t)

    # 2. Sawtooth wave (rich linear harmonic spectrum, highly non-Gaussian)
    f_saw = 150.0
    s2 = scipy.signal.sawtooth(2 * np.pi * f_saw * t)

    # 3. Frequency-modulated chirp signal (non-stationary, wideband)
    s3 = scipy.signal.chirp(t, f0=100.0, t1=duration, f1=1200.0, method='linear')

    # Stack into source matrix S (shape: 3 x n_samples)
    S = np.vstack([s1, s2, s3])

    # Normalize each source to zero mean and unit variance for consistent scale
    S = (S - np.mean(S, axis=1, keepdims=True)) / np.std(S, axis=1, keepdims=True)

    names = [
        "Pure Sine Wave (262 Hz)",
        "Sawtooth Wave (150 Hz)",
        "FM Chirp (100-1200 Hz)"
    ]
    return S, t, fs, names


def mix_signals(S, A=None, seed=42):
    """
    Simulate the acoustic cocktail party problem via linear instantaneous mixing:
    X = A * S
    """
    n_sources = S.shape[0]
    if A is None:
        np.random.seed(seed)
        # Generate a random invertible mixing matrix with good conditioning
        for _ in range(100):
            candidate = np.random.uniform(0.3, 1.0, size=(n_sources, n_sources))
            # Ensure diagonal dominance for realistic acoustic sensor placement
            candidate += np.eye(n_sources) * 0.6
            if np.linalg.cond(candidate) < 10.0:
                A = candidate
                break
        if A is None:
            A = np.array([
                [0.65, 0.35, 0.25],
                [0.30, 0.70, 0.35],
                [0.20, 0.35, 0.75]
            ])[:n_sources, :n_sources]

    X = np.dot(A, S)
    return X, A


def separate_fastica(X, n_components=3, random_state=42):
    """
    Separate mixed signals using FastICA (Hyvarinen fixed-point algorithm).
    """
    ica = FastICA(
        n_components=n_components,
        algorithm='parallel',
        whiten='unit-variance',
        fun='logcosh',
        max_iter=1000,
        tol=1e-5,
        random_state=random_state
    )
    # FastICA expects shape (n_samples, n_features)
    Y_unmixed = ica.fit_transform(X.T).T
    W_estimated = ica.components_
    return Y_unmixed, W_estimated, ica


def resolve_ica_ambiguities(S, Y):
    """
    Resolve permutation and sign ambiguities inherent to ICA.
    Matches each recovered component to the original source using
    maximum absolute Pearson correlation (bipartite Hungarian matching).
    """
    n_sources = S.shape[0]
    corr_matrix = np.zeros((n_sources, n_sources))

    for i in range(n_sources):
        for j in range(n_sources):
            # Pearson correlation
            r = np.corrcoef(S[i], Y[j])[0, 1]
            corr_matrix[i, j] = r

    # Cost matrix for Hungarian matching: minimize negative absolute correlation
    cost_matrix = -np.abs(corr_matrix)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

    Y_aligned = np.zeros_like(Y)
    signs = np.zeros(n_sources)
    final_correlations = np.zeros(n_sources)

    for orig_idx, sep_idx in zip(row_ind, col_ind):
        r = corr_matrix[orig_idx, sep_idx]
        sign = 1.0 if r >= 0 else -1.0
        # Align sign to match positive phase
        Y_aligned[orig_idx] = sign * Y[sep_idx]
        # Match scale to unit variance
        Y_aligned[orig_idx] /= np.std(Y_aligned[orig_idx])
        signs[orig_idx] = sign
        final_correlations[orig_idx] = abs(r)

    return Y_aligned, final_correlations, col_ind, corr_matrix


def compute_sir(s_true, s_est):
    """
    Calculate the Signal-to-Interference Ratio (SIR) in dB.
    Uses optimal least-squares projection to account for arbitrary residual scale.
    s_est_scaled = alpha * s_est
    interference = s_true - s_est_scaled
    SIR = 10 * log10( ||s_true||^2 / ||interference||^2 )
    """
    alpha = np.dot(s_true, s_est) / (np.dot(s_est, s_est) + 1e-12)
    s_target = alpha * s_est
    e_interf = s_true - s_target
    power_signal = np.sum(s_true ** 2)
    power_noise = np.sum(e_interf ** 2) + 1e-12
    sir = 10.0 * np.log10(power_signal / power_noise)
    return sir


def compute_kurtosis(signal):
    """
    Compute sample excess kurtosis (measure of non-Gaussianity):
    kurt = E[(x - mu)^4] / sigma^4 - 3
    """
    return scipy.stats.kurtosis(signal, fisher=True)


def save_audio(filename, audio_data, fs):
    """
    Normalize audio data to [-0.95, 0.95] peak to prevent digital clipping,
    and write to standard 16-bit PCM WAV.
    """
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        norm_audio = 0.95 * (audio_data / max_val)
    else:
        norm_audio = audio_data
    sf.write(filename, norm_audio.astype(np.float32), fs, subtype='PCM_16')


def plot_bss_results(S, X, Y, t, fs, names, correlations, sirs, output_path="bss_results_plot.png"):
    """
    Generate a high-resolution, publication-grade multi-panel figure comparing:
    - Panel 1: Original Independent Sources (S)
    - Panel 2: Mixed Microphone Observations (X)
    - Panel 3: ICA Recovered Sources (Y)
    """
    n_sources = S.shape[0]
    fig, axes = plt.subplots(3, n_sources, figsize=(15, 9), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Time window to display for clear waveform inspection (first 40 ms)
    window_samples = int(fs * 0.04)
    t_win = t[:window_samples] * 1000.0  # in milliseconds

    colors_sources = ['#1f77b4', '#d62728', '#2ca02c']
    colors_mixed = ['#9467bd', '#8c564b', '#e377c2']
    colors_recovered = ['#0055ff', '#cc0000', '#009933']

    # Panel 1: Original Sources
    for i in range(n_sources):
        ax = axes[0, i]
        ax.plot(t_win, S[i, :window_samples], color=colors_sources[i], linewidth=1.8)
        ax.set_title(f"Source $s_{i+1}(t)$:\n{names[i]}", fontsize=11, fontweight='bold', pad=8)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim([-2.5, 2.5])
        if i == 0:
            ax.annotate("Original Sources (S)", xy=(-0.35, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', rotation=90, va='center', ha='center',
                        color='#1B365D')

    # Panel 2: Mixed Signals
    for i in range(n_sources):
        ax = axes[1, i]
        ax.plot(t_win, X[i, :window_samples], color=colors_mixed[i], linewidth=1.5)
        ax.set_title(f"Mixed Signal $x_{i+1}(t)$\n(Microphone {i+1})", fontsize=11, fontweight='bold', pad=8)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.annotate("Acoustic Mixtures (X)", xy=(-0.35, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', rotation=90, va='center', ha='center',
                        color='#800000')

    # Panel 3: Recovered Sources
    for i in range(n_sources):
        ax = axes[2, i]
        ax.plot(t_win, Y[i, :window_samples], color=colors_recovered[i], linewidth=1.8, linestyle='-')
        # Overlay original as faint dashed line to demonstrate reconstruction accuracy
        ax.plot(t_win, S[i, :window_samples], color='black', linewidth=1.0, linestyle='--', alpha=0.45, label="Original")
        ax.set_title(f"Recovered $\\hat{{s}}_{i+1}(t)$\n$r = {correlations[i]:.4f}$ | SIR = {sirs[i]:.1f} dB",
                     fontsize=10.5, fontweight='bold', pad=8, color='#004d00')
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim([-2.5, 2.5])
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        if i == 0:
            ax.annotate("ICA Recovered (Y)", xy=(-0.35, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', rotation=90, va='center', ha='center',
                        color='#006622')

    fig.suptitle("Blind Signal Separation (BSS) via FastICA — Cocktail Party Problem\n"
                 "Task [B6] Technical Validation | Kyushu University (Laboratory of Intelligent Systems)",
                 fontsize=14, fontweight='bold', y=0.98, color='#1B365D')

    plt.tight_layout(rect=[0.05, 0.03, 0.98, 0.93])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Multi-panel figure successfully saved to: {output_path}")


def run_pipeline(project_dir="C:\\Users\\lsanj\\OneDrive\\Desktop\\B6_BSS_Project"):
    """
    Execute the end-to-end BSS pipeline for 3 signals (and 2 signals benchmark).
    """
    os.chdir(project_dir)
    print("=" * 70)
    print(" BLIND SIGNAL SEPARATION (BSS) PIPELINE — FASTICA")
    print(" Kyushu University | Laboratory of Intelligent Systems")
    print(" Candidate: Sanjana Laxman")
    print("=" * 70)

    # 1. Source Generation
    print("\n[Step 1] Generating 3 distinct synthetic audio sources (fs = 8000 Hz, duration = 3.0 s)...")
    S, t, fs, names = generate_sources(fs=8000, duration=3.0)
    
    for i in range(3):
        orig_filename = f"original_{i+1}.wav"
        save_audio(orig_filename, S[i], fs)
        kurt = compute_kurtosis(S[i])
        print(f"  - Source {i+1}: {names[i]:<25} | Excess Kurtosis = {kurt:+.4f} (Non-Gaussian)")

    # 2. Linear Mixing
    print("\n[Step 2] Mixing signals using 3x3 mixing matrix A (Cocktail Party simulation)...")
    X, A = mix_signals(S, seed=42)
    print("  Mixing Matrix A:\n", np.array2string(A, precision=3, suppress_small=True, prefix="    "))
    print(f"  Matrix Condition Number: {np.linalg.cond(A):.2f}")

    for i in range(3):
        mix_filename = f"mixed_{i+1}.wav"
        save_audio(mix_filename, X[i], fs)
        # Compute correlation with original sources in mixture
        corr_orig = [abs(np.corrcoef(X[i], S[j])[0, 1]) for j in range(3)]
        print(f"  - Saved: {mix_filename:<14} | Corrs with S1,S2,S3: [{corr_orig[0]:.2f}, {corr_orig[1]:.2f}, {corr_orig[2]:.2f}]")

    # 3. Blind Signal Separation (FastICA)
    print("\n[Step 3] Performing FastICA (n_components=3, parallel algorithm, logcosh contrast)...")
    Y_raw, W_est, ica = separate_fastica(X, n_components=3, random_state=42)
    print(f"  FastICA converged in {ica.n_iter_} iterations.")

    # 4. Resolve Ambiguities and Align
    print("\n[Step 4] Resolving permutation and sign ambiguities via Hungarian matching...")
    Y_aligned, correlations, mapping, corr_matrix = resolve_ica_ambiguities(S, Y_raw)

    sirs = []
    for i in range(3):
        sep_filename = f"separated_{i+1}.wav"
        save_audio(sep_filename, Y_aligned[i], fs)
        sir_val = compute_sir(S[i], Y_aligned[i])
        sirs.append(sir_val)
        print(f"  - Recovered Source {i+1} ({names[i]}):")
        print(f"      Mapped ICA Component : {mapping[i] + 1}")
        print(f"      Pearson Correlation r: {correlations[i]:.5f}")
        print(f"      Signal-to-Interference: {sir_val:.2f} dB")
        print(f"      Saved to File        : {sep_filename}")

    # 5. Visualization
    print("\n[Step 5] Generating high-resolution publication plot...")
    plot_path = os.path.join(project_dir, "bss_results_plot.png")
    plot_bss_results(S, X, Y_aligned, t, fs, names, correlations, sirs, output_path=plot_path)

    # 6. Benchmark: 2-Source Separation
    print("\n[Step 6] Running comparative benchmark for 2-Source Separation...")
    S_2 = S[:2, :]
    X_2, A_2 = mix_signals(S_2, seed=42)
    Y_2_raw, _, ica_2 = separate_fastica(X_2, n_components=2, random_state=42)
    Y_2_aligned, corrs_2, _, _ = resolve_ica_ambiguities(S_2, Y_2_raw)
    sirs_2 = [compute_sir(S_2[i], Y_2_aligned[i]) for i in range(2)]

    print(f"  2-Source FastICA Converged in {ica_2.n_iter_} iterations.")
    print(f"  Source 1 ({names[0]}): r = {corrs_2[0]:.5f}, SIR = {sirs_2[0]:.2f} dB")
    print(f"  Source 2 ({names[1]}): r = {corrs_2[1]:.5f}, SIR = {sirs_2[1]:.2f} dB")

    # 7. Save metrics JSON for the PDF report generator
    metrics_data = {
        "candidate": "Sanjana Laxman",
        "laboratory": "Laboratory of Intelligent Systems, Kyushu University",
        "supervisor": "Prof. Danilo V. Vargas",
        "sample_rate": fs,
        "duration": 3.0,
        "num_samples": len(t),
        "three_source": {
            "sources": [
                {
                    "id": 1,
                    "name": names[0],
                    "type": "Pure Tone Sine",
                    "frequency": "262 Hz",
                    "kurtosis": float(compute_kurtosis(S[0])),
                    "mix_corr": float(np.max([abs(np.corrcoef(X[k], S[0])[0, 1]) for k in range(3)])),
                    "recovered_corr": float(correlations[0]),
                    "sir_db": float(sirs[0])
                },
                {
                    "id": 2,
                    "name": names[1],
                    "type": "Sawtooth Wave",
                    "frequency": "150 Hz",
                    "kurtosis": float(compute_kurtosis(S[1])),
                    "mix_corr": float(np.max([abs(np.corrcoef(X[k], S[1])[0, 1]) for k in range(3)])),
                    "recovered_corr": float(correlations[1]),
                    "sir_db": float(sirs[1])
                },
                {
                    "id": 3,
                    "name": names[2],
                    "type": "Linear FM Chirp",
                    "frequency": "100 - 1200 Hz",
                    "kurtosis": float(compute_kurtosis(S[2])),
                    "mix_corr": float(np.max([abs(np.corrcoef(X[k], S[2])[0, 1]) for k in range(3)])),
                    "recovered_corr": float(correlations[2]),
                    "sir_db": float(sirs[2])
                }
            ],
            "mixing_matrix": A.tolist(),
            "condition_number": float(np.linalg.cond(A)),
            "iterations": int(ica.n_iter_),
            "mean_correlation": float(np.mean(correlations)),
            "mean_sir": float(np.mean(sirs))
        },
        "two_source": {
            "iterations": int(ica_2.n_iter_),
            "correlations": [float(c) for c in corrs_2],
            "sirs_db": [float(s) for s in sirs_2],
            "mean_correlation": float(np.mean(corrs_2)),
            "mean_sir": float(np.mean(sirs_2))
        }
    }

    with open("bss_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
    print("\n[+] Experimental metrics exported to: bss_metrics.json")
    print("=" * 70)
    print(" BSS PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
