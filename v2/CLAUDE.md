# CLAUDE.md — QMKL-v2

## Instructions de travail (à respecter impérativement)

- **Planification** : toujours utiliser l'agent Opus (`model: "opus"`) via l'outil Agent/Plan
- **Code** : toujours utiliser Sonnet (modèle par défaut) pour l'implémentation
- **Commandes shell** : toujours préfixer avec `rtk` (ex: `rtk git status`, `rtk python script.py`)
- **Skills** : plus de 150 skills disponibles — les identifier et les utiliser proactivement sans attendre que l'utilisateur le demande. Skills pertinents : `qiskit`, `scikit-learn`, `matplotlib`, `statistical-analysis`, `sympy`, `networkx`.

---

## Question de recherche centrale

**Comment assigner optimalement les features aux kernels quantiques via QUBO ?**

Deux usages QUBO :
1. **Sélection** (v1, fait) : quels kernels garder parmi M candidats ?
2. **Assignation** (v2, NOUVEAU) : quelles features → quel kernel ? Variables x_{k,m} ∈ {0,1}^{d×M}

**Motivation** : barren plateaus imposent Q ≤ 6 qubits par kernel. Avec d=20-30 features, l'assignation optimale permet de couvrir l'espace sans PCA qui perd l'information.

---

## Formulation QUBO (Formulation C)

```
min  -Σ_{m,k} a_{k,m} · x_{k,m}                          [qualité: alignments marginaux]
    + λ · Σ_k Σ_{m<m'} x_{k,m} · x_{k,m'}               [diversité: pénalise overlap]
    + μ₁ · Σ_m (Σ_k x_{k,m} - Q)²                        [contrainte: Q features/kernel]
```

- `a_{k,m}` = centered alignment du kernel m en utilisant seulement la feature k
- Variables : d × M binaires (ex: d=20, M=5 → 100 variables)
- Avantage quantique réel dès d=20, M=10 (brute force 2^200 → impossible)

---

## Datasets

| Dataset | Source | N | d | Cible | Statut |
|---|---|---|---|---|---|
| **FRED Recession** | Federal Reserve (FRED API) | ~600 | **19** | Récession vs expansion | **Principal** |
| German Credit | UCI / OpenML | 1000 | 20 | Risque crédit | Secondaire |
| Breast Cancer | sklearn | 569 | 30 | Tumeur | Benchmark |

### Features FRED (19 indicateurs macroéconomiques)
Unemployment, Nonfarm Payrolls, Fed Funds Rate, 10Y/2Y Treasury, Yield Spreads (10Y-2Y, 10Y-3M),
High Yield Spread, Baa Spread, TED Spread, Industrial Production, Retail Sales, Housing Starts,
Building Permits, CPI, M2, Oil Price WTI, Consumer Sentiment, S&P500.
Target : USREC (indicateur NBER, 0=expansion / 1=récession)

```python
# Avec clé FRED (gratuite sur https://fred.stlouisfed.org/)
export FRED_API_KEY=<votre_clé>

from src.data.loaders import load_fred_recession_data, load_fred_recession_synthetic
X, y, feat = load_fred_recession_data()         # données réelles
X, y, feat = load_fred_recession_synthetic()    # offline, pour tests
```

---

## Architecture des fichiers

```
Projet-QMKL-v2/
├── src/
│   ├── data/         — loaders.py (German Credit, Bank Marketing, Breast Cancer)
│   ├── preprocessing/— scaler.py (QuantumScaler)
│   ├── kernels/      — analytical.py (K_Z, K_ZZ numpy), subset_kernels.py (NOUVEAU)
│   ├── qubo/         — assignment_qubo.py (CORE), solvers.py (BF/SA/QAOA)
│   ├── mkl/          — alignment.py, combiner.py
│   └── evaluation/   — metrics.py
├── notebooks/
│   ├── 01_baseline_subsets.ipynb   — assignations naïves (référence)
│   ├── 02_qubo_formulation.ipynb   — construction QUBO, matrices, lambda sweep
│   ├── 03_classical_vs_qaoa.ipynb  — SA vs QAOA simulation
│   ├── 04_full_pipeline.ipynb      — pipeline complet, 3 datasets
│   ├── 05_ibm_hardware.ipynb       — QAOA sur IBM Torino (36 qubits)
│   └── 06_analysis.ipynb           — figures finales, tableaux, stats
├── results/
│   ├── figures/
│   ├── kernel_cache/
│   └── qubo_solutions/
└── tests/
```

---

## Résultats cibles

| Méthode | German Credit AUC | Breast Cancer AUC |
|---|---|---|
| Single kernel PCA Q=4 | ~0.69 | ~0.93 |
| QMKL PCA Q=4 (v1) | ~0.74 | ~0.99 |
| Random subsets Q=4, M=5 | ~0.72 ± 0.05 | ~0.96 ± 0.03 |
| Non-overlap fixe Q=4, M=5 | ~0.75 | ~0.97 |
| **QUBO-assignation Q=4, M=5** | **~0.78-0.82** | **~0.98-0.99** |
| RBF-SVM classique | ~0.80 | ~0.997 |

---

## IBM Torino — configuration cible

- Heron r2, 133 qubits
- Instance recommandée : d=12, M=3, Q=4 → **36 qubits QAOA**
- Brute force : 2^36 = 69 milliards → impossible classiquement

```python
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.backend("ibm_torino")
```

---

## Conventions style matplotlib

```python
plt.rcParams.update({'font.family': 'sans-serif', 'figure.dpi': 150, 'savefig.bbox': 'tight'})

COLORS = {
    'Z': '#3498db', 'ZZ': '#e74c3c', 'XZ': '#2ecc71',
    'YXX': '#f39c12', 'YZX': '#9b59b6', 'Pauli': '#1abc9c',
}
```
