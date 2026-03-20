# CLAUDE.md — QMKL-Finance

## Description du projet

**Quantum Multiple Kernel Learning (QMKL) pour la classification financière**

Implémentation de QMKL sur 3 datasets financiers réels :
- **German Credit** : 1000 clients, 20 variables, risque de crédit
- **Bank Marketing** : 45000 contacts, 16 variables, souscription dépôt
- **Breast Cancer** : 569 patients, 30 variables (benchmark médical)

Framework : Qiskit 1.x + qiskit-machine-learning, Python 3.11
Objectif futur : test sur IBM Quantum hardware (IBM Torino, Heron r2)

---

## Commandes de génération

```bash
# Figures de présentation
python gen_kernel_slides.py      # → pres_F_circuit.png + pres_F_kernels.png
python gen_qubo_slides.py        # → pres_qubo_complexity.png + pres_qubo_matrix.png

# Compilation LaTeX
pdflatex presentation.tex        # → presentation.pdf (22 slides)
pdflatex notes_slides_8_17.tex   # → notes_slides_8_17.pdf
```

---

## Figures de présentation (results/presentation/)

| Fichier | Slide | Contenu |
|---|---|---|
| `pres_F_circuit.png` | 6 | Circuit PauliFeatureMap ZZ, Q=4, reps=1, dessiné à la main |
| `pres_F_kernels.png` | 7 | 12 kernels : heatmaps K(x,x') + badge interaction + mini-barres AUC |
| `pres_F1_main_results.png` | 9 | QMKL vs classiques, 3 datasets |
| `pres_F2_qmkl_boost.png` | 15 | QMKL-Boost vs Average/Centered, 2 datasets |
| `pres_F5_qmkl_strategies.png` | 12 | 5 stratégies QMKL face à face |
| `pres_F6_german_ranking.png` | 10 | Classement détaillé German Credit |
| `pres_F8_shapley.png` | 14 | Valeurs de Shapley par kernel |
| `pres_F9_concentration.png` | 17 | Barren plateaus : variance vs Q |
| `pres_F10_lambda_sweep.png` | — | Lambda sweep QUBO (conservé, retiré de la présentation principale) |
| `pres_F11_learning_curves.png` | 11 | Courbes d'apprentissage, QMKL vs classiques |
| `pres_qubo_complexity.png` | 16b | Explosion combinatoire 2^M, seuil quantique M=25 |
| `pres_qubo_matrix.png` | 16 | Matrice Q 12×12 + tableau de décision QUBO |

---

## Résultats clés

### AUC par méthode et dataset (N=150, Q=4)

| Méthode | German Credit | Bank Marketing | Breast Cancer |
|---|---|---|---|
| QMKL-Centered Alignment | 0.743 | 0.861 | 0.993 |
| QMKL-Boost (QUBO, 2 kernels) | **0.823** | **0.767** | — |
| QMKL-Average | 0.782 | 0.691 | — |
| RBF-SVM | 0.800 | 0.875 | 0.997 |
| Forêt aléatoire | 0.845 | 0.890 | 0.998 |

Résultats = moyenne sur 20 répétitions avec splits aléatoires.
QUBO kernels sélectionnés : `ZZ α=2.5` + `XZ α=0.5`

### Test IBM Torino (N=30, Q=4, 1024 shots)

| Kernel | Sim | HW | Delta |
|---|---|---|---|
| ZZ α=1.0 | 0.670 | 0.735 | +6.5 pts (bruit bénéfique) |
| ZZ α=4.0 | 0.756 | <0.500 | Concentré, inutilisable |
| XZ α=0.5 | 0.721 | ~0.680 | Stable |

---

## Les 12 kernels quantiques

| # | Famille | Pauli | α | AUC (German) | Shapley |
|---|---|---|---|---|---|
| 1 | Z | Z | 1.0 | 0.698 | +0.005 |
| 2 | Z | Z | 3.0 | 0.712 | +0.012 |
| 3 | ZZ | Z,ZZ | 1.0 | 0.734 | +0.015 |
| 4 | ZZ | Z,ZZ | 4.0 | 0.756 | **+0.042** |
| 5 | XZ | X,Z | 0.5 | 0.721 | +0.031 |
| 6 | XZ | X,Z | 2.5 | 0.738 | +0.028 |
| 7 | YXX | Y,XX | 0.6 | 0.709 | +0.008 |
| 8 | YXX | Y,XX | 3.0 | 0.744 | +0.018 |
| 9 | YZX | Y,ZX | 0.6 | 0.702 | +0.003 |
| 10 | YZX | Y,ZX | 3.0 | 0.718 | +0.007 |
| 11 | Pauli | Z,ZZ | 0.6 | 0.729 | +0.011 |
| 12 | Pauli | Z,ZZ | 2.5 | 0.751 | +0.022 |

Niveau d'interaction : Z = local (1-corps) · ZZ, XZ = pairwise (2-corps) · YXX, YZX = 3-corps non-commutatif

---

## Complexité QUBO — Quand le quantique devient indispensable

| M (kernels) | Brute force 2^M | Heuristique M³ | Remarque |
|---|---|---|---|
| 12 | **4 096** | ~1 728 | Notre projet — classique OK |
| 25 | **33 millions** | ~15 625 | Limite classique |
| 35 | ~34 milliards | ~42 875 | Zone de transition |
| 50 | **~10^15** | ~125 000 | Quantique seul viable |

Note honnête : le recuit quantique (D-Wave, Pasqal) est **heuristique empiriquement polynomial** — pas de garantie théorique, mais en pratique O(M^1.5).

---

## Conventions de style matplotlib

```python
plt.rcParams.update({
    'font.family': 'sans-serif',
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

# Palette familles de kernels
COLORS = {
    'Z':     '#3498db',   # bleu
    'ZZ':    '#e74c3c',   # rouge
    'XZ':    '#2ecc71',   # vert
    'YXX':   '#f39c12',   # orange
    'YZX':   '#9b59b6',   # violet
    'Pauli': '#1abc9c',   # turquoise
}

# Régions QUBO
COLOR_CLASSICAL  = '#d5f5e3'  # vert pâle — M < 25
COLOR_TRANSITION = '#fdebd0'  # orange pâle — 25 ≤ M ≤ 35
COLOR_QUANTUM    = '#d6eaf8'  # bleu pâle — M > 35
```

---

## Architecture des fichiers source

```
Projet-QMKL-Finance/
├── src/
│   ├── kernels/      — calcul des matrices kernel (Qiskit + analytique)
│   ├── mkl/          — stratégies QMKL (Average, Centered, QUBO, BO, VQKL)
│   ├── models/       — SVM wrapper
│   └── preprocessing/— PCA, StandardScaler, MinMaxScaler
├── scripts/          — run_experiments.py, run_hardware.py
├── results/
│   ├── presentation/ — figures .png pour les slides
│   └── kernel_cache/ — matrices kernel sauvegardées (.npy)
├── gen_kernel_slides.py  — Figure 1 (circuit) + Figure 2 (heatmaps kernels)
├── gen_qubo_slides.py    — Figure A (complexité) + Figure B (matrice Q)
├── presentation.tex      — 22 slides Beamer
└── notes_slides_8_17.tex — Notes explicatives détaillées
```
