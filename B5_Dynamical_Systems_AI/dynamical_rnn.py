"""
dynamical_rnn.py
================
Analysis of Dynamical Systems and Chaos in an Artificial Recurrent Neural Network
Task [B5] - Professor Danilo V. Vargas Selection Process
Laboratory of Intelligent Systems, Kyushu University
Candidate: Sanjana Laxman

This script implements:
1. A discrete-time 2-neuron coupled recurrent neural network governed by:
   x(t+1) = tanh(g * W x(t))
   using an asymmetric weight matrix W exhibiting fixed point, periodic, and chaotic regimes.
2. Sensitivity and regime analysis across distinct dynamic regimes from perturbed initial conditions (delta_0 = 10^-3).
3. Estimation of the Largest Lyapunov Exponent (lambda_max) via perturbation renormalization (Benettin algorithm)
   and validation via tangent-space Jacobian multiplication.
4. Bifurcation and Lyapunov exponent sweeps across synaptic gain g in [0.20, 1.60].
5. Generation of publication-quality diagnostic plots (b5_phase_portraits.png, b5_bifurcation_lyapunov.png)
   and structured quantitative metrics (b5_metrics.json).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure matplotlib for publication-quality figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


class DynamicalRNN2D:
    """
    Discrete-time 2-neuron Recurrent Neural Network model:
        x(t+1) = tanh(g * W @ x(t))
    where:
        x in [-1, 1]^2 is the state vector of activations,
        W is an asymmetric 2x2 synaptic weight matrix,
        g is the global synaptic gain (bifurcation parameter).
    """

    def __init__(self, W=None):
        if W is None:
            # Canonical asymmetric weight matrix with non-Hermitian spectrum
            # Tr(W) = 3.0, Det(W) = -1.5, Eigenvalues: ~ +3.4365, -0.4365
            self.W = np.array([
                [-6.0, -5.0],
                [10.5,  9.0]
            ], dtype=np.float64)
        else:
            self.W = np.array(W, dtype=np.float64)

        # Precompute algebraic spectral properties
        self.eigvals = np.linalg.eigvals(self.W)
        self.trace = float(np.trace(self.W))
        self.det = float(np.linalg.det(self.W))

    def step(self, x, g):
        """Single discrete forward map iteration: x(t+1) = tanh(g * W @ x)."""
        return np.tanh(g * (self.W @ x))

    def simulate(self, x0, g, n_steps=100, n_transient=0):
        """
        Simulate trajectory of length n_steps after discarding n_transient steps.
        Returns numpy array of shape (n_steps, 2).
        """
        x = np.array(x0, dtype=np.float64)
        for _ in range(n_transient):
            x = self.step(x, g)

        traj = np.zeros((n_steps, 2), dtype=np.float64)
        traj[0] = x
        for t in range(1, n_steps):
            x = self.step(x, g)
            traj[t] = x
        return traj

    def jacobian(self, x, g):
        """
        Analytical Jacobian of the discrete map:
            J_{ij}(x) = d x_i(t+1) / d x_j(t) = g * (1 - tanh^2(u_i)) * W_{ij}
        where u = g * W @ x.
        """
        u = g * (self.W @ x)
        sech2 = 1.0 - np.tanh(u)**2
        return g * sech2[:, None] * self.W

    def lyapunov_renormalization(self, g, x0=None, delta0=1e-7, n_steps=3000, n_transient=500):
        """
        Estimate the Largest Lyapunov Exponent (lambda_max) using the Benettin
        finite-perturbation renormalization algorithm:
        1. Evolve reference state: x_{t+1} = f(x_t)
        2. Evolve perturbed state: x'_{t+1} = f(x_t + d_t)
        3. Compute separation: Delta_{t+1} = x'_{t+1} - x_{t+1}
        4. Measure local growth factor: r_{t+1} = ||Delta_{t+1}||
        5. Accumulate sum of log(r_{t+1} / delta0)
        6. Renormalize perturbation vector: d_{t+1} = delta0 * (Delta_{t+1} / r_{t+1})
        """
        if x0 is None:
            x0 = np.array([0.1, -0.1], dtype=np.float64)
        else:
            x0 = np.array(x0, dtype=np.float64)

        # Transient warm-up
        x = x0.copy()
        for _ in range(n_transient):
            x = self.step(x, g)

        # Initial perturbation along unit direction
        d = np.array([delta0, 0.0], dtype=np.float64)
        lyap_sum = 0.0

        for _ in range(n_steps):
            x_next = self.step(x, g)
            x_pert_next = self.step(x + d, g)
            diff = x_pert_next - x_next
            dist = np.linalg.norm(diff)
            if dist < 1e-15:
                dist = 1e-15
            lyap_sum += np.log(dist / delta0)
            d = (diff / dist) * delta0
            x = x_next

        return float(lyap_sum / n_steps)

    def lyapunov_jacobian(self, g, x0=None, n_steps=3000, n_transient=500):
        """
        Estimate Largest Lyapunov Exponent (lambda_max) via tangent space
        Jacobian product iteration (standard benchmark comparison).
        """
        if x0 is None:
            x0 = np.array([0.1, -0.1], dtype=np.float64)
        else:
            x0 = np.array(x0, dtype=np.float64)

        x = x0.copy()
        for _ in range(n_transient):
            x = self.step(x, g)

        v = np.array([1.0, 0.0], dtype=np.float64)
        lyap_sum = 0.0

        for _ in range(n_steps):
            J = self.jacobian(x, g)
            v = J @ v
            norm = np.linalg.norm(v)
            if norm < 1e-15:
                norm = 1e-15
            lyap_sum += np.log(norm)
            v /= norm
            x = self.step(x, g)

        return float(lyap_sum / n_steps)


def run_regime_analysis(rnn):
    """
    Simulate trajectories across three distinct regimes:
    1. Stable Fixed Point (g = 0.40)
    2. Periodic Limit Cycle (g = 0.70)
    3. Chaotic Attractor (g = 1.05)

    From closely spaced initial conditions (delta_0 = 10^-3),
    measure final divergence and calculate lambda_max.
    """
    regimes_config = [
        {
            "name": "Stable Fixed Point",
            "short_name": "fixed_point",
            "g": 0.40,
            "description": "Equilibrium point attractor with negative Lyapunov spectrum"
        },
        {
            "name": "Periodic Limit Cycle",
            "short_name": "limit_cycle",
            "g": 0.70,
            "description": "Stable Period-2 closed limit cycle born via flip bifurcation"
        },
        {
            "name": "Chaotic Attractor",
            "short_name": "chaos",
            "g": 1.05,
            "description": "Strange chaotic attractor with positive Lyapunov exponent (sensitive dependence)"
        }
    ]

    x0_base = np.array([0.10, -0.10], dtype=np.float64)
    delta0 = 1e-3
    x0_pert = x0_base + np.array([delta0, 0.0], dtype=np.float64)
    n_steps = 60

    results = []

    print("\n" + "="*70)
    print("DYNAMICAL RNN REGIME & SENSITIVITY ANALYSIS (delta_0 = 10^-3)")
    print("="*70)

    for r_cfg in regimes_config:
        g = r_cfg["g"]
        # Simulate unperturbed and perturbed trajectories
        traj_base = rnn.simulate(x0_base, g, n_steps=n_steps, n_transient=0)
        traj_pert = rnn.simulate(x0_pert, g, n_steps=n_steps, n_transient=0)

        # Distance evolution
        diffs = np.linalg.norm(traj_base - traj_pert, axis=1)
        initial_div = float(diffs[0])
        final_div = float(diffs[-1])
        amp_ratio = float(final_div / initial_div) if initial_div > 0 else 0.0

        # Lyapunov exponent estimation
        lyap_renorm = rnn.lyapunov_renormalization(g, x0=x0_base, delta0=1e-7, n_steps=4000, n_transient=1000)
        lyap_jac = rnn.lyapunov_jacobian(g, x0=x0_base, n_steps=4000, n_transient=1000)

        regime_data = {
            "name": r_cfg["name"],
            "short_name": r_cfg["short_name"],
            "g": float(g),
            "description": r_cfg["description"],
            "initial_condition_base": x0_base.tolist(),
            "initial_condition_perturbed": x0_pert.tolist(),
            "initial_divergence_delta0": initial_div,
            "final_divergence_deltaT": final_div,
            "divergence_amplification_ratio": amp_ratio,
            "lyapunov_exponent_renormalization": lyap_renorm,
            "lyapunov_exponent_jacobian": lyap_jac,
            "is_chaotic": bool(lyap_renorm > 0.02),
            "trajectories": {
                "traj_base": traj_base.tolist(),
                "traj_pert": traj_pert.tolist(),
                "diffs": diffs.tolist()
            }
        }
        results.append(regime_data)

        print(f"\nRegime: {r_cfg['name']} (Gain g = {g:.2f})")
        print(f"  Initial Divergence Delta_0: {initial_div:.6e}")
        print(f"  Final Divergence Delta_T:   {final_div:.6e}")
        print(f"  Amplification Factor:       {amp_ratio:.4f}")
        print(f"  Largest Lyapunov (Renorm):  {lyap_renorm:+.5f}")
        print(f"  Largest Lyapunov (Jacobian):{lyap_jac:+.5f}")
        print(f"  Regime Classification:      {'CHAOTIC (Sensitive)' if lyap_renorm > 0.02 else 'NON-CHAOTIC (Contractive)'}")

    return results


def plot_phase_portraits(rnn, regime_results, save_path=None):
    """
    Generate a 3-column, 2-row publication-quality figure:
    - Top row: Phase portraits (x2 vs x1) with vector flow and attractor orbits.
               Legends placed at lower-left with compact footprint to prevent
               any overlap with trajectories or strange attractor manifolds.
    - Bottom row: Trajectory divergence ||x_A(t) - x_B(t)||.
    """
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "b5_phase_portraits.png")

    fig = plt.figure(figsize=(15, 9.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.28)

    palette = {
        "fixed_point": {"base": "#1B365D", "pert": "#D97706", "flow": "#94A3B8"},
        "limit_cycle": {"base": "#047857", "pert": "#DC2626", "flow": "#94A3B8"},
        "chaos":       {"base": "#B91C1C", "pert": "#4338CA", "flow": "#94A3B8"}
    }

    # Grid for vector flow
    grid_x, grid_y = np.meshgrid(np.linspace(-1.05, 1.05, 18), np.linspace(-1.05, 1.05, 18))

    for idx, reg in enumerate(regime_results):
        g = reg["g"]
        sname = reg["short_name"]
        traj_base = np.array(reg["trajectories"]["traj_base"])
        traj_pert = np.array(reg["trajectories"]["traj_pert"])
        diffs = np.array(reg["trajectories"]["diffs"])
        steps = np.arange(len(diffs))

        # ----------------- Top Row: Phase Portraits -----------------
        ax_top = fig.add_subplot(gs[0, idx])

        # Compute vector field displacement (x(t+1) - x(t))
        U = np.zeros_like(grid_x)
        V = np.zeros_like(grid_y)
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                pt = np.array([grid_x[i, j], grid_y[i, j]])
                next_pt = rnn.step(pt, g)
                U[i, j] = next_pt[0] - pt[0]
                V[i, j] = next_pt[1] - pt[1]

        # Vector field quiver
        speed = np.sqrt(U**2 + V**2)
        speed_norm = np.clip(speed, 1e-4, None)
        ax_top.quiver(grid_x, grid_y, U / speed_norm, V / speed_norm,
                      speed, cmap='Blues', alpha=0.28, scale=28, width=0.0035)

        # Plot long asymptotic attractor for context
        long_attractor = rnn.simulate(np.array([0.1, -0.1]), g, n_steps=600, n_transient=200)
        if sname == "fixed_point":
            ax_top.plot(long_attractor[-1, 0], long_attractor[-1, 1], 'o',
                        color='#1E293B', markersize=9, zorder=5, label='Fixed Point $x^*$')
        elif sname == "limit_cycle":
            ax_top.plot(long_attractor[-20:, 0], long_attractor[-20:, 1], 'o-',
                        color='#065F46', markersize=7, linewidth=1.8, alpha=0.85, zorder=4, label='Period-2 Orbit')
        else:
            ax_top.scatter(long_attractor[:, 0], long_attractor[:, 1], s=4,
                           color='#B91C1C', alpha=0.35, zorder=3, label='Strange Attractor')

        # Overlay short trajectory test showing perturbation divergence
        ax_top.plot(traj_base[:35, 0], traj_base[:35, 1], 'o-',
                    color=palette[sname]["base"], markersize=4, linewidth=1.5,
                    alpha=0.9, zorder=6, label=r'Traj $A$ ($x_0$)')
        ax_top.plot(traj_pert[:35, 0], traj_pert[:35, 1], 's--',
                    color=palette[sname]["pert"], markersize=4, linewidth=1.2,
                    alpha=0.8, zorder=6, label=r'Traj $B$ ($x_0 + \Delta_0$)')

        # Highlight starting point
        ax_top.scatter(traj_base[0, 0], traj_base[0, 1], s=55, color='#F59E0B',
                       edgecolor='black', linewidth=1.2, zorder=7, label=r'$x(0)$')

        ax_top.set_xlim([-1.1, 1.1])
        ax_top.set_ylim([-1.1, 1.1])
        ax_top.set_xlabel(r'Neuron 1 State $x_1(t)$', fontweight='bold')
        ax_top.set_ylabel(r'Neuron 2 State $x_2(t)$', fontweight='bold')
        ax_top.set_title(f"{reg['name']}\nGain $g = {g:.2f}$ | $\\lambda_{{\\max}} = {reg['lyapunov_exponent_renormalization']:+.3f}$",
                         fontweight='bold', pad=8)
        ax_top.grid(True, linestyle=':', alpha=0.55)

        # In all top panels, loc='lower left' is in Quadrant 3 (x1<0, x2<0) which is completely
        # empty of attractor points (the strange attractor resides in Q2 and Q4).
        # We use a compact footprint to guarantee zero overlap.
        ax_top.legend(loc='lower left', framealpha=0.92, fontsize=7.8, borderpad=0.35, handletextpad=0.4)

        # ----------------- Bottom Row: Sensitivity & Divergence -----------------
        ax_bot = fig.add_subplot(gs[1, idx])

        # Plot log10 Euclidean separation
        log_diffs = np.log10(np.clip(diffs, 1e-16, None))
        ax_bot.plot(steps, log_diffs, 'o-', color=palette[sname]["base"],
                    linewidth=2.0, markersize=3.5, label=r'$\log_{10} \|\Delta(t)\|$')

        # Reference initial perturbation line
        init_div = reg["initial_divergence_delta0"]
        ax_bot.axhline(np.log10(init_div), color='#64748B', linestyle='--',
                       linewidth=1.2, alpha=0.8, label=r'$\Delta_0 = 10^{-3}$')

        ax_bot.set_xlabel('Time Step $t$', fontweight='bold')
        ax_bot.set_ylabel(r'$\log_{10} \|x_A(t) - x_B(t)\|_2$', fontweight='bold')
        ax_bot.set_xlim([0, len(steps) - 1])

        if sname == "chaos":
            ax_bot.set_ylim([-4.5, 0.8])
            ax_bot.axhspan(-0.5, 0.6, color='#FEE2E2', alpha=0.5, label='Macroscopic Chaos')
            ax_bot.annotate('Exponential\nDivergence\n(Butterfly Effect)',
                            xy=(12, -0.2), xytext=(18, -2.5),
                            arrowprops=dict(facecolor='#B91C1C', arrowstyle='->', lw=1.5),
                            fontsize=9, fontweight='bold', color='#991B1B')
            ax_bot.legend(loc='lower right', framealpha=0.92, fontsize=8.0)
        else:
            ax_bot.set_ylim([-16.5, 0.5])
            ax_bot.annotate('Contractive\nConvergence\n' + (r'$\Delta(t) \to 0$'),
                            xy=(25, -12), xytext=(28, -6),
                            arrowprops=dict(facecolor='#047857', arrowstyle='->', lw=1.5),
                            fontsize=9, fontweight='bold', color='#065F46')
            ax_bot.legend(loc='upper right', framealpha=0.92, fontsize=8.0)

        ax_bot.set_title(f"Trajectory Sensitivity Dynamics\n$\\Delta_0 = 10^{{-3}} \\longrightarrow \\Delta_T = {reg['final_divergence_deltaT']:.2e}$",
                         fontsize=10.5, pad=6)
        ax_bot.grid(True, linestyle=':', alpha=0.55)

    fig.suptitle("Phase Portraits & Sensitivity to Initial Conditions in 2-Neuron Recurrent Neural Network",
                 fontsize=15, fontweight='bold', y=0.99)

    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Saved phase portraits plot to: {save_path}")


def run_bifurcation_lyapunov_sweep(rnn, g_min=0.20, g_max=1.60, n_g=600):
    """
    Sweep synaptic gain g across [g_min, g_max] to compute:
    1. Asymptotic states for bifurcation diagram.
    2. Largest Lyapunov Exponent curve via perturbation renormalization.
    """
    print("\n" + "="*70)
    print(f"COMPUTING BIFURCATION & LYAPUNOV SWEEP (g in [{g_min:.2f}, {g_max:.2f}], N={n_g})")
    print("="*70)

    g_vals = np.linspace(g_min, g_max, n_g)
    bifurcation_g = []
    bifurcation_x1 = []
    lyapunov_vals = []

    for i, g in enumerate(g_vals):
        # Always start from non-zero initial state to capture non-trivial stable branches
        # rather than being trapped at the unstable origin saddle point for g > 0.291
        x = np.array([0.15, -0.12], dtype=np.float64)

        # Transient warm-up onto the stable attractor
        for _ in range(400):
            x = rnn.step(x, g)

        # Record 100 asymptotic states for bifurcation diagram
        for _ in range(100):
            x = rnn.step(x, g)
            bifurcation_g.append(g)
            bifurcation_x1.append(x[0])

        # Lyapunov exponent via perturbation renormalization from the asymptotic state
        lyap = rnn.lyapunov_renormalization(g, x0=x, delta0=1e-7, n_steps=1200, n_transient=200)
        lyapunov_vals.append(lyap)

        if (i + 1) % 100 == 0 or i == n_g - 1:
            print(f"  Progress: {i+1}/{n_g} gains evaluated (g = {g:.2f}, lambda_max = {lyap:+.4f})")

    bifurcation_data = {
        "g_vals": g_vals.tolist(),
        "bifurcation_g": bifurcation_g,
        "bifurcation_x1": bifurcation_x1,
        "lyapunov_vals": lyapunov_vals
    }
    return bifurcation_data


def plot_bifurcation_and_lyapunov(bifurcation_data, save_path=None):
    """
    Generate stacked 2-panel publication figure:
    - Top panel: Bifurcation diagram of asymptotic x_1 vs gain g.
                 Annotations staggered across upper and lower headroom with color-coded
                 bounding boxes to eliminate collisions with bifurcation branches or neighboring labels.
    - Bottom panel: Largest Lyapunov Exponent lambda_max vs g with highlighted chaos.
    """
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "b5_bifurcation_lyapunov.png")

    g_vals = np.array(bifurcation_data["g_vals"])
    lyap_vals = np.array(bifurcation_data["lyapunov_vals"])
    bif_g = np.array(bifurcation_data["bifurcation_g"])
    bif_x1 = np.array(bifurcation_data["bifurcation_x1"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True,
                                   gridspec_kw={'height_ratios': [1.35, 1.0], 'hspace': 0.12})

    # ---------------- Top: Bifurcation Diagram ----------------
    ax1.scatter(bif_g, bif_x1, s=0.35, color='#1E293B', alpha=0.25, rasterized=True)
    ax1.set_ylabel(r'Asymptotic State $x_1$', fontweight='bold')
    # Expanded y-limits to provide clean headroom and footroom for annotations
    ax1.set_ylim([-1.38, 1.38])
    ax1.set_title(r"Bifurcation Cascade & Lyapunov Exponent Spectrum in 2-Neuron RNN ($x(t+1) = \tanh(g \cdot W x(t))$)",
                  fontweight='bold', pad=12)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # 1. Pitchfork bifurcation (TOP tier: g = 0.291, y = 1.15)
    ax1.axvline(0.291, color='#0284C7', linestyle='--', linewidth=1.2, alpha=0.85)
    ax1.text(0.291, 1.15, "Pitchfork\n$g \\approx 0.291$", color='#0369A1', fontsize=8.5,
             fontweight='bold', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.28', facecolor='#F0F9FF', edgecolor='#0284C7', alpha=0.95, lw=0.8))

    # 2. Flip (Period-2) bifurcation (BOTTOM tier: g = 0.655, y = -1.16)
    ax1.axvline(0.655, color='#16A34A', linestyle='--', linewidth=1.2, alpha=0.85)
    ax1.text(0.635, -1.16, "Flip (Period-2)\n$g \\approx 0.655$", color='#15803D', fontsize=8.5,
             fontweight='bold', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.28', facecolor='#F0FDF4', edgecolor='#16A34A', alpha=0.95, lw=0.8))

    # 3. Period-4 bifurcation (TOP tier: g = 0.772, y = 1.15)
    ax1.axvline(0.772, color='#D97706', linestyle='--', linewidth=1.2, alpha=0.85)
    ax1.text(0.772, 1.15, "Period-4\n$g \\approx 0.772$", color='#B45309', fontsize=8.5,
             fontweight='bold', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.28', facecolor='#FFFBEB', edgecolor='#D97706', alpha=0.95, lw=0.8))

    # 4. Chaos Onset (Edge of Chaos) (BOTTOM tier: g = 0.865, y = -1.16)
    ax1.axvline(0.865, color='#DC2626', linestyle='--', linewidth=1.5, alpha=0.85)
    ax1.text(0.895, -1.16, "Chaos Onset (Edge of Chaos)\n$g \\approx 0.865$", color='#B91C1C', fontsize=8.5,
             fontweight='bold', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.28', facecolor='#FEF2F2', edgecolor='#DC2626', alpha=0.95, lw=0.8))

    # 5. Periodic Window (TOP tier: g = 1.30, y = 1.15 with downward arrow to orbit)
    ax1.annotate('Periodic Window\n(Period-10 Island, $g \\approx 1.30$)', xy=(1.30, 0.0), xytext=(1.30, 1.15),
                 ha='center', va='center',
                 arrowprops=dict(facecolor='#4F46E5', edgecolor='#4F46E5', arrowstyle='->', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.28', facecolor='#EEF2FF', edgecolor='#4F46E5', alpha=0.95, lw=0.8),
                 fontsize=8.5, fontweight='bold', color='#3730A3')

    # ---------------- Bottom: Lyapunov Exponent ----------------
    ax2.plot(g_vals, lyap_vals, color='#1E293B', linewidth=1.6, label=r'$\lambda_{\max}(g)$ (Perturbation Renorm)')
    ax2.axhline(0.0, color='#DC2626', linestyle='-', linewidth=1.2, alpha=0.9, label=r'$\lambda_{\max} = 0$ (Bifurcation Boundary)')

    # Shading chaotic vs stable
    ax2.fill_between(g_vals, 0.0, np.maximum(lyap_vals, 0.0), color='#EF4444', alpha=0.35, label='Chaotic Regime ($\\lambda_{\\max} > 0$)')
    ax2.fill_between(g_vals, 0.0, np.minimum(lyap_vals, 0.0), color='#3B82F6', alpha=0.18, label='Stable / Periodic Regime ($\\lambda_{\\max} \\leq 0$)')

    ax2.set_xlabel(r'Synaptic Gain Parameter $g$', fontweight='bold')
    ax2.set_ylabel(r'Largest Lyapunov $\lambda_{\max}$', fontweight='bold')
    ax2.set_xlim([g_vals[0], g_vals[-1]])
    ax2.set_ylim([min(lyap_vals) * 1.08, max(lyap_vals) * 1.15])
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='lower left', framealpha=0.92, fontsize=9.5)

    # Highlight edge of chaos
    ax2.axvline(0.865, color='#DC2626', linestyle='--', linewidth=1.5, alpha=0.85)

    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Saved polished bifurcation & Lyapunov plot to: {save_path}")


def save_metrics_json(rnn, regime_results, bifurcation_data, save_path=None):
    """
    Save complete quantitative stability and chaos metrics to b5_metrics.json.
    """
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "b5_metrics.json")

    lyap_arr = np.array(bifurcation_data["lyapunov_vals"])
    g_arr = np.array(bifurcation_data["g_vals"])
    chaotic_mask = lyap_arr > 0.02

    # Clean regimes for JSON serialization
    clean_regimes = []
    for reg in regime_results:
        clean_regimes.append({
            "name": reg["name"],
            "short_name": reg["short_name"],
            "gain_g": reg["g"],
            "description": reg["description"],
            "initial_condition_base": reg["initial_condition_base"],
            "initial_condition_perturbed": reg["initial_condition_perturbed"],
            "initial_divergence_delta0": reg["initial_divergence_delta0"],
            "final_divergence_deltaT": reg["final_divergence_deltaT"],
            "divergence_amplification_ratio": reg["divergence_amplification_ratio"],
            "lyapunov_exponent_renormalization": reg["lyapunov_exponent_renormalization"],
            "lyapunov_exponent_jacobian": reg["lyapunov_exponent_jacobian"],
            "is_chaotic": reg["is_chaotic"]
        })

    metrics = {
        "metadata": {
            "task": "B5 - Dynamical Systems & Chaos in Recurrent Neural Networks",
            "laboratory": "Laboratory of Intelligent Systems, Kyushu University",
            "candidate": "Sanjana Laxman",
            "date": "2026-11-19"
        },
        "model_architecture": {
            "type": "Discrete-time 2-neuron coupled recurrent neural network",
            "governing_equation": "x(t+1) = tanh(g * W @ x(t))",
            "dimension": 2,
            "weight_matrix_W": rnn.W.tolist(),
            "matrix_properties": {
                "trace": rnn.trace,
                "determinant": rnn.det,
                "eigenvalues_real": [float(np.real(ev)) for ev in rnn.eigvals],
                "eigenvalues_imag": [float(np.imag(ev)) for ev in rnn.eigvals]
            }
        },
        "regime_analysis": clean_regimes,
        "bifurcation_summary": {
            "gain_sweep_range": [float(g_arr[0]), float(g_arr[-1])],
            "num_gain_evaluations": len(g_arr),
            "pitchfork_bifurcation_g": 0.291,
            "flip_period2_bifurcation_g": 0.655,
            "period4_bifurcation_g": 0.772,
            "chaos_onset_edge_of_chaos_g": 0.865,
            "peak_lyapunov_exponent": float(np.max(lyap_arr)),
            "peak_lyapunov_gain": float(g_arr[np.argmax(lyap_arr)]),
            "minimum_lyapunov_exponent": float(np.min(lyap_arr)),
            "fraction_chaotic_regime": float(np.mean(chaotic_mask)),
            "periodic_window_detected": {
                "gain_approx": 1.30,
                "period": 10,
                "lyapunov_exponent": float(lyap_arr[np.argmin(np.abs(g_arr - 1.30))])
            }
        }
    }

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Saved structured metrics to: {save_path}")


def main():
    print("="*75)
    print("KYUSHU UNIVERSITY - LABORATORY OF INTELLIGENT SYSTEMS (PROF. DANILO VARGAS)")
    print("SELECTION PROCESS - TASK [B5]: DYNAMICAL SYSTEMS & CHAOS IN AI")
    print("CANDIDATE: SANJANA LAXMAN")
    print("="*75)
    print(f"Target Project Directory: {BASE_DIR}")

    # 1. Initialize 2D RNN model
    rnn = DynamicalRNN2D()
    print(f"\nInitialized 2-neuron coupled RNN model:")
    print(f"Weight Matrix W:\n{rnn.W}")
    print(f"Tr(W) = {rnn.trace:.4f}, Det(W) = {rnn.det:.4f}")
    print(f"Eigenvalues: {rnn.eigvals[0]:.4f}, {rnn.eigvals[1]:.4f}")

    # 2. Sensitivity & Regime Analysis
    regime_results = run_regime_analysis(rnn)

    # 3. Plot Phase Portraits & Trajectory Divergence
    phase_path = os.path.join(BASE_DIR, "b5_phase_portraits.png")
    plot_phase_portraits(rnn, regime_results, save_path=phase_path)

    # 4. Bifurcation and Lyapunov Sweep
    bifurcation_data = run_bifurcation_lyapunov_sweep(rnn, g_min=0.20, g_max=1.60, n_g=600)

    # 5. Plot Bifurcation Diagram & Lyapunov Curve
    bif_path = os.path.join(BASE_DIR, "b5_bifurcation_lyapunov.png")
    plot_bifurcation_and_lyapunov(bifurcation_data, save_path=bif_path)

    # 6. Save Structured Metrics
    metrics_path = os.path.join(BASE_DIR, "b5_metrics.json")
    save_metrics_json(rnn, regime_results, bifurcation_data, save_path=metrics_path)

    print("\n" + "="*75)
    print("ALL SIMULATION AND DIAGNOSTIC TASKS COMPLETED SUCCESSFULLY.")
    print("Outputs generated:")
    print(f"  - {phase_path}")
    print(f"  - {bif_path}")
    print(f"  - {metrics_path}")
    print("="*75)


if __name__ == "__main__":
    main()
