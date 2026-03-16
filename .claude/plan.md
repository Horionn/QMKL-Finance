# Plan: Étude QMKL Complète avec BO, Plus de Qubits/Kernels, et Nombreuses Figures

## Objectif
Créer un notebook `06_comprehensive_study.ipynb` qui pousse l'analyse plus loin : BO comme méthode phare, plus de qubits (jusqu'à 12), plus de kernels (20), et ~14 figures différentes.

## Fichiers à modifier

### 1. `src/kernels/feature_maps.py`
- Ajouter `get_extended_feature_map_library(n_qubits)` → 20 kernels
  - 3 alphas par type au lieu de 2
  - Ajouter entanglement "full" en plus de "linear"
  - Plus de diversité Pauli : Z, ZZ, pauli, pauli_XZ, pauli_YXX, pauli_YZX × {small, medium, large} α

### 2. `src/mkl/bayesian_optimizer.py`
- Ajouter `get_convergence_history()` → liste des scores à chaque itération
- Ajouter `get_search_trajectory()` → poids explorés par le GP

### 3. `src/evaluation/visualization.py`
- Ajouter 6 nouvelles fonctions de visualisation :
  - `plot_method_comparison_grouped()` : barres groupées (méthodes × datasets)
  - `plot_bo_convergence()` : courbe de convergence BO avec IC
  - `plot_weight_heatmap()` : heatmap méthodes × kernels
  - `plot_scaling_curve()` : n_qubits vs AUC avec bande CI95
  - `plot_concentration_scatter()` : concentration vs performance
  - `plot_radar_chart()` : radar multi-métrique (AUC, F1, Acc, Prec, Recall)

### 4. Nouveau `notebooks/06_comprehensive_study.ipynb`

#### Structure (10 sections, ~25 cellules) :

**§1 — Setup & Données** (cells 1-2)
- 3 datasets : german_credit, bank_marketing, synthetic
- N_SAMPLES = 250

**§2 — Bibliothèque étendue de kernels** (cells 3-4)
- 20 kernels via `get_extended_feature_map_library()`
- **Figure 1** : Grille 3×3 de heatmaps de kernels sélectionnés
- **Figure 2** : Bar chart des alignements kernel-target par kernel

**§3 — Comparaison des 6 méthodes** (cells 5-7)
- Single-Best, Average, Centered, SDP, Projection, **BO**
- 15 évaluations (3 rounds × 5-fold CV)
- **Figure 3** : Barres groupées (6 méthodes × 3 datasets)
- **Figure 4** : Radar chart multi-métriques (AUC, F1, Acc, Prec, Recall)

**§4 — Deep Dive Bayesian Optimization** (cells 8-10)
- Convergence BO pour chaque dataset
- Sensibilité : n_calls = [20, 50, 100]
- **Figure 5** : Courbes de convergence BO (3 datasets sur même plot)
- **Figure 6** : n_calls vs AUC final (diminishing returns)
- **Figure 7** : Poids BO vs poids Centered (scatter comparatif)

**§5 — Scaling Study : n_qubits** (cells 11-13)
- n_qubits = [2, 4, 6, 8, 10, 12]
- Comparer BO vs Centered vs Single-Best
- **Figure 8** : Courbe scaling avec CI95 band (3 méthodes)
- **Figure 9** : Heatmap concentration (n_qubits × kernel_name)

**§6 — Analyse des poids** (cells 14-16)
- Quels kernels sont sélectionnés par chaque méthode ?
- **Figure 10** : Heatmap poids (6 méthodes × 20 kernels)
- **Figure 11** : Violin plots des poids par kernel (stabilité)

**§7 — Fidelity vs Projected** (cells 17-18)
- Comparer les 2 types de kernel × 3 datasets
- **Figure 12** : Paired bar chart (fid vs proj, par dataset)

**§8 — Tableau récapitulatif** (cell 19)
- **Figure 13** : Win/Tie/Loss table (BO vs chaque autre méthode)
- Tests statistiques (Wilcoxon) avec p-values

**§9 — Figure de synthèse** (cell 20)
- **Figure 14** : Panel 2×3 combinant les résultats clés

## Notes de performance
- Tous les kernels précalculés au §2, cachés sur disque
- BO est lent (~2-5 min par dataset) → on cache les résultats dans un dict
- n_qubits=12 avec 20 kernels = ~2^12 × 250² = lourd → on utilise Aer multi-thread
- Estimation temps total : ~30-45 min en première exécution, <1 min ensuite (cache)
