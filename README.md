# Laboratory of Intelligent Systems - Selection Tasks

**Candidate:** Sanjana Laxman[cite: 40]
**Institution / Lab:** Laboratory of Intelligent Systems, Kyushu University (Prof. Danilo Vasconcellos Vargas)[cite: 40]
**Consolidated Report:** `Sanjana_Laxman_Kyushu_LIS_Selection_Report.pdf`

## Repository Structure

### Basic Tasks
- **[B1] Image Reconstruction** (`B1_Image_Reconstruction/`)[cite: 41]
  - **Summary:** Convolutional Denoising Autoencoder (CDAE) trained on MNIST across Gaussian, Salt-and-Pepper, and Mixed noise regimes, benchmarked against classical spatial filters[cite: 41, 42].
  - **Key Files:** `train.py`, `evaluate.py`, `run_pipeline.py`, `src/`, `outputs/`, `Task_B1_Image_Reconstruction_Report.pdf`[cite: 42]

- **[B5] Dynamical Systems & Chaos Analysis** (`B5_Dynamical_Systems_AI/`)[cite: 45]
  - **Summary:** Bifurcation cascade, trajectory sensitivity, and Largest Lyapunov Exponent analysis of an autonomous discrete-time 2-neuron coupled RNN across synaptic gain `g`[cite: 45, 46].
  - **Key Files:** `dynamical_rnn.py`, `generate_b5_report.py`, `b5_metrics.json`, `b5_phase_portraits.png`, `b5_bifurcation_lyapunov.png`, `Task_B5_Dynamical_Systems_Report.pdf`[cite: 46, 48, 49, 50]

- **[B6] Blind Signal Separation (BSS)** (`B6_BSS_Project/`)[cite: 51]
  - **Summary:** FastICA separation of 2-source and 3-source soundwave mixtures using Hungarian permutation alignment, evaluated via Pearson correlation and SIR (dB)[cite: 51, 52, 53, 54].
  - **Key Files:** `bss_pipeline.py`, `generate_report.py`, `bss_metrics.json`, `bss_results_plot.png`, `.wav` files, `Task_B6_BSS_Report.pdf`[cite: 53, 54]

### Advanced Tasks
- **[A2] Continual General Chunking Problem and SyncMap** (`A2_Adversarial_Robustness/`)[cite: 55]
  - **Summary:** Evaluation and enhancement of SyncMap (SymmetricalSyncMapMA) across Overlap 1 and Overlap 2 benchmarks over 10 seeds and 25,000 steps, achieving +7.43 pp (+10.79%% relative) on Overlap 1 and +13.99 pp (+30.72%% relative) on Overlap 2[cite: 55, 56, 58].
  - **Key Files:** `syncmap_overlap_a2.py`, `adversarial_few_pixel.py`, `generate_a2_report.py`, `a2_metrics.json`, `upstream_syncmap/`, `Task_A2_Adversarial_Robustness_Report.pdf`[cite: 57, 58, 59]

- **[A3]{X} Edge-Weighted Graph Clustering in SyncMap** (`A3_SyncMap_Graph_Clustering/`)[cite: 62]
  - **Summary:** Replacement of DBSCAN in SyncMap with an adaptive local-scaling, co-activation-weighted graph and Louvain modularity optimization across 4 tasks (20 seeds each), improving ARI on all 4 tasks and reducing variance on 3 of 4 tasks[cite: 62, 63, 64, 65].
  - **Key Files:** `syncmap_graph_clustering.py`, `generate_a3_report.py`, `a3_metrics.json`, `a3_cluster_comparison.png`, `a3_accuracy_variance.png`, `Task_A3_Graph_Clustering_Report.pdf`[cite: 63, 64, 65]
