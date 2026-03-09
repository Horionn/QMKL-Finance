# Quantum Multiple Kernel Learning (QMKL) pour la Classification Financière

Implémentation d'une pipeline complète QMKL basée sur les articles IBM/HSBC (2023), ICASSP 2024 et MLPRAE 2025.

## Installation

### 1. Créer et activer le venv

```bash
cd "Projet-QMKL-Finance"
source .venv/bin/activate
```

### 2. Vérifier l'installation

```bash
python -c "import qiskit; print(f'Qiskit {qiskit.__version__} OK')"
```

## Utilisation

### Option A : Jupyter Notebooks (recommandé pour explorer)

```bash
# Lancer Jupyter
jupyter notebook

# Dans Jupyter:
# 1. Naviguer vers notebooks/
# 2. Ouvrir 01_data_exploration.ipynb
# 3. En haut à droite, sélectionner le kernel: Kernel > Change Kernel > QMKL-Finance
# 4. Exécuter les cellules (Shift+Enter)
```

**Notebooks disponibles :**
- `01_data_exploration.ipynb` - Exploration des données financières
- `02_single_kernel_baseline.ipynb` - Baselines single kernel + concentration
- `03_mkl_experiments.ipynb` - Comparaison des 5 stratégies QMKL

### Option B : Script CLI

```bash
# Lancer une expérience avec la config par défaut
python scripts/run_experiment.py --config config/default.yaml

# Avec config spécifique
python scripts/run_experiment.py --config config/experiments/fraud_detection.yaml
```

### Option C : Tests

```bash
pytest tests/ -v
```

## Architecture

```
src/
├── preprocessing/      # Scaling [0,2π] + PCA
├── kernels/           # 12 feature maps (Z, ZZ, Pauli variants)
├── mkl/               # 4 stratégies d'alignement + Bayesian Opt
├── models/            # QSVM wrapper + pipeline complet
└── evaluation/        # Metrics + visualisations

config/
├── default.yaml       # Config par défaut (8 qubits, German Credit)
└── experiments/       # Configs spécialisées

data/
├── raw/              # Emplacement pour données brutes
└── processed/        # Données après préprocessing
```

## Datasets supportés

| Dataset | Classes | Features | Taille | Statut |
|---------|---------|----------|--------|--------|
| `german_credit` | 2 | 20 | 1000 | Automatique (OpenML) |
| `bank_marketing` | 2 | 16 | ~45k | Automatique (OpenML) |
| `iris_binary` | 2 | 4 | 150 | Automatique |
| `custom` | 2 | Variable | Vos données | Manuel (CSV) |

## Concepts clés

### QMKL (Quantum Multiple Kernel Learning)
Combine plusieurs quantum kernels avec des poids optimisés :
```
K = Σ w_i * K_i
```

### Stratégies d'optimisation des poids
1. **Average** - Poids égaux (baseline)
2. **Centered Alignment** - Kernel-target alignment centré
3. **SDP** - Semidefinite Programming
4. **Projection** - Alignement itératif par projection
5. **Bayesian** - BO via scikit-optimize (BO-MKQSVM)

### Feature Maps Quantiques
- **Z** - Rotations Z simples (α ∈ [1.4, 20])
- **ZZ** - Z + entanglement ZZ pairwise
- **Pauli** - Combinations de Pauli gates (X, Y, Z)

### Kernels Quantiques
- **Fidelity** - K(x,x') = |⟨ψ(x)|ψ(x')⟩|²
- **Projected** - RBF sur 1-RDMs (plus efficace, meilleur scaling)

## Exemple minimal

```python
from src.models import QMKLClassifier
from data.loaders import load_dataset
from sklearn.model_selection import train_test_split

# Charger données
X, y = load_dataset('german_credit', n_samples=200)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Créer et entraîner le classifieur
clf = QMKLClassifier(n_components=6, mkl_method='centered')
clf.fit(X_train, y_train)

# Prédire
y_pred = clf.predict(X_test)

# Afficher les poids
weights = clf.get_kernel_weights()
print(f'Top kernels: {weights.argsort()[-3:][::-1]}')
```

## Configuration

Éditer `config/default.yaml` pour modifier :
- `n_components` : nombre de qubits (= dimension PCA)
- `mkl.alignment_method` : stratégie d'alignement
- `feature_maps` : liste des feature maps à utiliser
- `kernel.types` : `fidelity` ou `projected`

## Résultats attendus

Sur German Credit (n=200) :
- **Single Kernel** : ROC-AUC ~ 0.65-0.75
- **QMKL-Average** : ROC-AUC ~ 0.70-0.80
- **QMKL-Centered** : ROC-AUC ~ 0.72-0.82
- **QMKL-BO** : ROC-AUC ~ 0.75-0.85 (meilleur)

(Résultats varient selon taille dataset et random seed)

## Dépendances clés

| Package | Version | Rôle |
|---------|---------|------|
| qiskit | ≥1.0 | Circuits quantiques |
| qiskit-machine-learning | ≥0.7 | Kernels quantiques |
| scikit-learn | ≥1.3 | SVM, métriques |
| scikit-optimize | ≥0.9 | Bayesian Optimization |
| cvxpy | ≥1.4 | SDP solver (CVX) |
| numpy, pandas, matplotlib | Latest | Calcul, visualisation |

## Références

1. **IBM/HSBC (2023)** - "Quantum Multiple Kernel Learning in Financial Classification Tasks"
   - Fidelity & Projected kernels
   - Centering alignment strategy
   - Error mitigation pipeline (hardware)

2. **ICASSP 2024** - "Exploiting QMKL for Low-Resource Spoken Command Recognition"
   - Multiple Gaussian kernels
   - Theoretical error bounds
   - Prototypical networks

3. **MLPRAE 2025** - "BO-MKQSVM: Bayesian Optimization for Multiple Kernel QSVM"
   - Qiskit 1.1 implementation
   - 3 feature maps (ZZ, Pauli, Z)
   - Bayesian hyperparameter optimization

## Prochaines étapes (Phase 5)

- [ ] Error mitigation sur hardware IBM (randomized compiling)
- [ ] Exécution sur IBM Quantum (ibm_auckland, ibm_nairobi)
- [ ] Comparaison simulateur vs hardware
- [ ] Benchmarking sur datasets financiers réels (fraud detection)
- [ ] Ablation studies sur nombre de kernels

## Troubleshooting

### "Kernel not found"
```bash
source .venv/bin/activate
python -m ipykernel install --user --name qmkl-finance --display-name "QMKL-Finance"
```

### Imports échouent dans notebook
Vérifier que le bon kernel est sélectionné : Kernel > Change Kernel > QMKL-Finance

### Qiskit slow
Les première exécution compile les circuits. Ensuite c'est plus rapide. Pour des tests, réduire `n_samples` et `n_components`.

## Contact

Pour questions : consulter les notebooks et la documentation inline des modules.
