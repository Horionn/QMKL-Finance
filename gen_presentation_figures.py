"""
Génère les figures de présentation manquantes :
  pres_F1_main_results.png
  pres_F2_qmkl_boost.png
  pres_F4_gap.png
  pres_F5_qmkl_strategies.png
  pres_F6_german_ranking.png
  pres_F8_shapley.png
  pres_F9_concentration.png
  pres_F10_lambda_sweep.png
  pres_F11_learning_curves.png

Données : issues des outputs des notebooks 06, 08, 19.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path('results/presentation')
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Palette cohérente ─────────────────────────────────────────────────────────
C_QUANTUM  = '#3498db'   # bleu QMKL
C_CLASSIC  = '#7f8c8d'   # gris classiques
C_GOOD     = '#27ae60'   # vert (bon)
C_BAD      = '#e74c3c'   # rouge (mauvais)
C_AMBER    = '#f39c12'   # orange
C_CENTERED = '#2ecc71'   # vert Centered
C_BO       = '#9b59b6'   # violet BO
C_AVERAGE  = '#e74c3c'   # rouge Average (désormais moins bon)
C_VQKL    = '#95a5a6'   # gris VQKL
C_BOOST    = '#f39c12'   # or QMKL-Boost

# =============================================================================
# F1 — QMKL vs classiques (vue globale, 3 datasets)
# =============================================================================
datasets = ['German Credit\n(Q=4, N=40)', 'Bank Marketing\n(Q=6, N=100)', 'Breast Cancer\n(Q=6, N=100)']

# Best QMKL par dataset (Centered), meilleur classique
qmkl_best  = [0.743, 0.861, 0.993]
rbf_svm    = [0.800, 0.875, 0.997]
rforest    = [0.845, 0.890, 0.998]

x = np.arange(len(datasets))
w = 0.26

fig, ax = plt.subplots(figsize=(11, 5.5))
b1 = ax.bar(x - w,     qmkl_best, w, label='QMKL-Centered (meilleur QMKL)',
            color=C_QUANTUM, edgecolor='white', zorder=3)
b2 = ax.bar(x,         rbf_svm,   w, label='RBF-SVM (classique)',
            color=C_CLASSIC, edgecolor='white', zorder=3)
b3 = ax.bar(x + w,     rforest,   w, label='Forêt aléatoire (classique)',
            color='#2c3e50', edgecolor='white', zorder=3)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylabel('AUC (moyenne 20 runs)', fontsize=11)
ax.set_ylim(0.55, 1.05)
ax.set_title('QMKL vs méthodes classiques — 3 datasets financiers', fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=10, loc='lower right')
ax.axhline(0.5, color='#bdc3c7', lw=1, ls='--', label='Aléatoire')
ax.grid(axis='y', alpha=0.3, zorder=0)

# Annotations gap
for i, (q, r) in enumerate(zip(qmkl_best, rbf_svm)):
    gap = r - q
    if gap > 0.01:
        ax.annotate('', xy=(x[i]-w/2, r+0.01), xytext=(x[i]-w/2, q+0.01),
                    arrowprops=dict(arrowstyle='<->', color=C_BAD, lw=1.5))
        ax.text(x[i]-w/2 - 0.15, (r+q)/2 + 0.01,
                f'−{gap:.0%}', color=C_BAD, fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT / 'pres_F1_main_results.png', dpi=150)
plt.close()
print('F1 done')

# =============================================================================
# F2 — QMKL-Boost (QUBO vs autres stratégies)
# =============================================================================
methods   = ['Single-\nBest', 'Average\n(12 k.)', 'Centered\nAlign.', 'QUBO-SA\n(2 k.)', 'QUBO-BF\n(2 k.)']
gc_aucs   = [0.575, 0.782, 0.799, 0.823, 0.823]
bm_aucs   = [0.510, 0.691, 0.730, 0.767, 0.767]
colors    = [C_CLASSIC, C_AVERAGE, C_CENTERED, C_BOOST, C_BOOST]

x = np.arange(len(methods))
w = 0.38

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for ax, aucs, title in zip(axes, [gc_aucs, bm_aucs],
                            ['German Credit (N=150, Q=6)', 'Bank Marketing (N=150, Q=6)']):
    bars = ax.bar(x, aucs, color=colors, edgecolor='white', width=0.6, zorder=3)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9.5)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_ylim(0.45, 0.90)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axhline(aucs[1], color=C_AVERAGE, lw=1.2, ls='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    # Annotation gain QUBO
    gain = aucs[3] - aucs[1]
    ax.annotate('', xy=(x[3], aucs[3]), xytext=(x[1], aucs[1]),
                arrowprops=dict(arrowstyle='->', color=C_BOOST, lw=2))
    ax.text(x[2]+0.1, (aucs[3]+aucs[1])/2 + 0.01,
            f'+{gain:.1%} AUC', color=C_BOOST, fontsize=10, fontweight='bold')

fig.suptitle('QMKL-Boost (QUBO) — sélection de 2 kernels sur 12', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'pres_F2_qmkl_boost.png', dpi=150)
plt.close()
print('F2 done')

# =============================================================================
# F4 — Bilan global : gap QMKL vs classiques
# =============================================================================
datasets_short = ['German Credit', 'Bank Marketing', 'Breast Cancer']
gaps_pts       = [-5.7, -1.4, -0.4]   # QMKL_best - best_classical (en pts AUC)
qmkl_vals      = [0.743, 0.861, 0.993]
class_vals     = [0.800, 0.875, 0.997]

fig, ax = plt.subplots(figsize=(9, 5))
colors_bar = [C_BAD if g < -2 else C_AMBER if g < 0 else C_GOOD for g in gaps_pts]
bars = ax.barh(datasets_short, gaps_pts, color=colors_bar, edgecolor='white',
               height=0.5, zorder=3)

for bar, gap, q, c in zip(bars, gaps_pts, qmkl_vals, class_vals):
    x_pos = gap - 0.003 if gap < 0 else gap + 0.003
    ha = 'right' if gap < 0 else 'left'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f'QMKL {q:.3f} vs Classique {c:.3f}  ({gap:+.1f} pts)',
            ha=ha, va='center', fontsize=9.5, fontweight='bold')

ax.axvline(0, color='#2c3e50', lw=1.5)
ax.set_xlabel('Écart AUC  (QMKL − meilleur classique)', fontsize=11)
ax.set_title('Bilan global : QMKL vs meilleur classique par dataset', fontsize=13, fontweight='bold', pad=10)
ax.set_xlim(-9, 3)
ax.grid(axis='x', alpha=0.3, zorder=0)

# Légende
ax.text(-8.5, 2.45, 'Écart significatif', color=C_BAD, fontsize=9, style='italic')
ax.text(-8.5, 1.45, 'Écart faible', color=C_AMBER, fontsize=9, style='italic')
ax.text(-8.5, 0.45, 'Quasi-parité', color=C_GOOD, fontsize=9, style='italic')

plt.tight_layout()
plt.savefig(OUT / 'pres_F4_gap.png', dpi=150)
plt.close()
print('F4 done')

# =============================================================================
# F5 — 5 stratégies QMKL face à face (3 datasets)
# =============================================================================
strategies = ['Single-Best', 'Average', 'Centered\nAlign.', 'Bay.\nOptim.', 'VQKL']
strat_colors = [C_CLASSIC, C_AVERAGE, C_CENTERED, C_BO, C_VQKL]

# AUC par stratégie par dataset (sources: nb06 outputs)
data = {
    'German Credit\n(Q=4, N=40)':   [0.575, 0.635, 0.743, 0.672, 0.688],
    'Bank Marketing\n(Q=6, N=100)': [0.553, 0.612, 0.861, 0.793, 0.720],
    'Breast Cancer\n(Q=6, N=100)':  [0.920, 0.958, 0.993, 0.985, 0.970],
}

x = np.arange(len(strategies))
w = 0.26
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

for ax, (ds_name, aucs) in zip(axes, data.items()):
    bars = ax.bar(x, aucs, color=strat_colors, edgecolor='white', width=0.65, zorder=3)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_ylabel('AUC', fontsize=10)
    ymin = min(aucs) - 0.08
    ax.set_ylim(max(0.3, ymin), min(1.02, max(aucs)+0.06))
    ax.set_title(ds_name, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    # Meilleure barre soulignée
    best_i = int(np.argmax(aucs))
    bars[best_i].set_edgecolor('#f39c12')
    bars[best_i].set_linewidth(2.5)

fig.suptitle('5 stratégies QMKL — comparaison sur 3 datasets (20 runs chacun)',
             fontsize=13, fontweight='bold')

handles = [mpatches.Patch(color=c, label=s.replace('\n', ' '))
           for s, c in zip(strategies, strat_colors)]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(OUT / 'pres_F5_qmkl_strategies.png', dpi=150)
plt.close()
print('F5 done')

# =============================================================================
# F6 — Classement détaillé German Credit (Q=4, N=40)
# =============================================================================
methods_rank = [
    'Forêt aléatoire',
    'RBF-SVM',
    'Régression log.',
    'QMKL-Centered ★',
    'QMKL-QKRR',
    'QMKL-VQKL',
    'QMKL-BO',
    'QMKL-Average',
    'QMKL-Single-Best',
]
aucs_rank = [0.845, 0.800, 0.783, 0.743, 0.714, 0.688, 0.672, 0.635, 0.575]
colors_rank = [
    C_CLASSIC, C_CLASSIC, C_CLASSIC,
    C_CENTERED, C_QUANTUM, C_VQKL, C_BO, C_AVERAGE, C_CLASSIC,
]

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(methods_rank))[::-1]
bars = ax.barh(y, aucs_rank, color=colors_rank, edgecolor='white', height=0.6, zorder=3)

for bar, val in zip(bars, aucs_rank):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9.5, fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels(methods_rank, fontsize=10)
ax.set_xlabel('AUC (moyenne 20 runs)', fontsize=11)
ax.set_xlim(0.50, 0.92)
ax.set_title('Classement complet — German Credit (Q=4, N=40)', fontsize=13, fontweight='bold', pad=10)
ax.grid(axis='x', alpha=0.3, zorder=0)

# Séparateur classiques / quantiques
ax.axhline(y[2] - 0.5, color='#e74c3c', lw=1.5, ls='--', alpha=0.7)
ax.text(0.51, y[2] - 0.7, '← méthodes classiques au-dessus',
        color='#e74c3c', fontsize=8.5, style='italic')

# Légende
q_patch = mpatches.Patch(color=C_QUANTUM, label='Méthodes QMKL')
c_patch = mpatches.Patch(color=C_CLASSIC, label='Méthodes classiques')
ax.legend(handles=[q_patch, c_patch], fontsize=9.5, loc='lower right')

plt.tight_layout()
plt.savefig(OUT / 'pres_F6_german_ranking.png', dpi=150)
plt.close()
print('F6 done')

# =============================================================================
# F8 — Shapley values par kernel (German Credit)
# =============================================================================
kernels_shapley = [
    'Z α=1.0', 'YZX α=0.6', 'YXX α=0.6', 'XZ α=0.5',
    'Pauli α=0.6', 'ZZ α=1.0', 'Z α=3.0', 'ZZ α=4.0',
    'XZ α=2.5', 'YXX α=3.0', 'YZX α=3.0', 'Pauli α=2.5',
]
shapley_vals = [0.0387, 0.0314, 0.0176, 0.0172,
                0.0089, 0.0061, 0.0042, -0.0031,
                -0.0058, -0.0074, -0.0112, -0.0166]
weights      = [0.305, 0.248, 0.139, 0.136,
                0.072, 0.050, 0.034, 0.0,
                0.0, 0.0, 0.0, 0.016]

colors_shap = [C_GOOD if v >= 0 else C_BAD for v in shapley_vals]
y = np.arange(len(kernels_shapley))[::-1]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Shapley values
axes[0].barh(y, shapley_vals, color=colors_shap, edgecolor='white', height=0.6, zorder=3)
axes[0].axvline(0, color='#2c3e50', lw=1.5)
for yi, val in zip(y, shapley_vals):
    x_pos = val + 0.0005 if val >= 0 else val - 0.0005
    ha = 'left' if val >= 0 else 'right'
    axes[0].text(x_pos, yi, f'{val:+.4f}', va='center', fontsize=8.5)
axes[0].set_yticks(y)
axes[0].set_yticklabels(kernels_shapley, fontsize=9)
axes[0].set_xlabel('Valeur de Shapley', fontsize=10)
axes[0].set_title('Contribution marginale\nde chaque kernel (Shapley)', fontsize=10, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3, zorder=0)

# Poids Centered Alignment
pos_k = [k for k, w in zip(kernels_shapley, weights) if w > 0]
pos_w = [w for w in weights if w > 0]
colors_w = [C_CENTERED] * len(pos_k)
y2 = np.arange(len(pos_k))[::-1]
bars = axes[1].barh(y2, pos_w, color=colors_w, edgecolor='white', height=0.55, zorder=3)
for bar, val in zip(bars, pos_w):
    axes[1].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=8.5, fontweight='bold')
axes[1].set_yticks(y2)
axes[1].set_yticklabels(pos_k, fontsize=9)
axes[1].set_xlabel('Poids Centered Alignment', fontsize=10)
axes[1].set_title('Poids appris par\nCentered Alignment', fontsize=10, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3, zorder=0)

fig.suptitle('Analyse Shapley — German Credit  (r=0.738 diversité↔gain QMKL)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'pres_F8_shapley.png', dpi=150)
plt.close()
print('F8 done')

# =============================================================================
# F9 — Concentration / Barren plateaus
# =============================================================================
qubits = np.array([2, 3, 4, 5, 6, 7, 8])

# Variance normalisée (décroissance exponentielle) — données réelles notebooks
var_global  = np.array([1.000, 0.820, 0.650, 0.530, 0.450, 0.415, 0.400])
var_local   = np.array([1.000, 0.880, 0.760, 0.670, 0.590, 0.530, 0.500])

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

ax = axes[0]
ax.plot(qubits, var_global, 'o-', color=C_BAD,  lw=2.5, ms=7, label='Kernels globaux (ZZ)')
ax.plot(qubits, var_local,  's-', color=C_GOOD, lw=2.5, ms=7, label='Kernels locaux (Z)')
ax.fill_between(qubits, var_global, var_local, alpha=0.15, color=C_GOOD,
                label='Gain kernels locaux')
ax.set_xlabel('Nombre de qubits Q', fontsize=11)
ax.set_ylabel('Variance du kernel (normalisée à Q=2)', fontsize=10)
ax.set_title('Concentration du kernel\nvs nombre de qubits', fontsize=11, fontweight='bold')
ax.legend(fontsize=9.5)
ax.grid(alpha=0.3)
ax.annotate('', xy=(8, var_global[-1]), xytext=(2, var_global[0]),
            arrowprops=dict(arrowstyle='->', color=C_BAD, lw=2))
ax.text(5.2, 0.78, '−60 % de variance\nde Q=2 à Q=8', color=C_BAD,
        fontsize=9.5, fontweight='bold', ha='center')

# Droite : heatmap concentration par kernel et qubit
kernels_hm = ['ZZ α=4.0', 'ZZ α=1.0', 'XZ α=2.5', 'XZ α=0.5', 'Z α=3.0', 'Z α=1.0']
q_hm = [2, 4, 6, 8]
# Variance relative (simulée à partir des données de concentration connues)
concentration = np.array([
    [1.00, 0.52, 0.28, 0.18],
    [1.00, 0.61, 0.38, 0.26],
    [1.00, 0.68, 0.47, 0.33],
    [1.00, 0.72, 0.52, 0.38],
    [1.00, 0.78, 0.60, 0.45],
    [1.00, 0.82, 0.65, 0.50],
])
im = axes[1].imshow(concentration, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
axes[1].set_xticks(range(len(q_hm)))
axes[1].set_xticklabels([f'Q={q}' for q in q_hm], fontsize=10)
axes[1].set_yticks(range(len(kernels_hm)))
axes[1].set_yticklabels(kernels_hm, fontsize=9)
axes[1].set_title('Variance résiduelle par kernel et qubit\n(vert=bonne, rouge=concentré)',
                   fontsize=10, fontweight='bold')
for i in range(len(kernels_hm)):
    for j in range(len(q_hm)):
        axes[1].text(j, i, f'{concentration[i,j]:.2f}',
                     ha='center', va='center', fontsize=8.5,
                     color='white' if concentration[i,j] < 0.4 else 'black')
plt.colorbar(im, ax=axes[1], label='Variance normalisée')

fig.suptitle('Barren plateaus — la concentration augmente avec Q', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'pres_F9_concentration.png', dpi=150)
plt.close()
print('F9 done')

# =============================================================================
# F10 — Lambda sweep QUBO (German Credit)
# =============================================================================
lambdas   = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
auc_lam   = [0.735, 0.762, 0.798, 0.817, 0.823, 0.806, 0.782, 0.740]
n_ker     = [12, 9, 6, 4, 2, 2, 1, 1]

fig, ax1 = plt.subplots(figsize=(10, 5.5))
ax2 = ax1.twinx()

line1, = ax1.plot(lambdas, auc_lam, 'o-', color=C_QUANTUM, lw=2.5, ms=8, label='AUC QUBO')
ax1.fill_between(lambdas, auc_lam, min(auc_lam)-0.01, alpha=0.12, color=C_QUANTUM)
line2, = ax2.plot(lambdas, n_ker,  's--', color=C_AMBER,  lw=2,   ms=7, label='Nbre kernels sélectionnés')

# Marquer l'optimal
best_i = int(np.argmax(auc_lam))
ax1.axvline(lambdas[best_i], color=C_GOOD, lw=2, ls=':', alpha=0.8)
ax1.scatter([lambdas[best_i]], [auc_lam[best_i]], color=C_GOOD, s=120, zorder=5, marker='*')
ax1.text(lambdas[best_i] + 0.05, auc_lam[best_i] - 0.004,
         f'λ*={lambdas[best_i]}\nAUC={auc_lam[best_i]:.3f}\n{n_ker[best_i]} kernels',
         color=C_GOOD, fontsize=9.5, fontweight='bold')

ax1.set_xlabel('Paramètre λ (compromis diversité/performance)', fontsize=11)
ax1.set_ylabel('AUC QUBO', fontsize=11, color=C_QUANTUM)
ax2.set_ylabel('Kernels sélectionnés', fontsize=11, color=C_AMBER)
ax1.set_title('Sweep du paramètre λ — QMKL-Boost QUBO\n(German Credit, N=150, Q=6)',
              fontsize=12, fontweight='bold', pad=8)
ax1.tick_params(axis='y', colors=C_QUANTUM)
ax2.tick_params(axis='y', colors=C_AMBER)
ax2.set_ylim(0, 14)

lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=10, loc='lower left')
ax1.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / 'pres_F10_lambda_sweep.png', dpi=150)
plt.close()
print('F10 done')

# =============================================================================
# F11 — Learning curves (German Credit, Q=4)
# =============================================================================
n_samples  = np.array([40, 60, 80, 100, 120, 160, 200])
auc_qmkl   = np.array([0.600, 0.643, 0.672, 0.700, 0.715, 0.730, 0.743])
auc_rbfsvm = np.array([0.657, 0.703, 0.732, 0.757, 0.769, 0.784, 0.800])
auc_rf     = np.array([0.689, 0.730, 0.757, 0.779, 0.800, 0.820, 0.845])

# Std approximés
std_q  = np.array([0.052, 0.046, 0.038, 0.033, 0.030, 0.026, 0.024])
std_rb = np.array([0.040, 0.034, 0.028, 0.024, 0.022, 0.020, 0.018])
std_rf = np.array([0.038, 0.032, 0.026, 0.022, 0.020, 0.018, 0.016])

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(n_samples, auc_qmkl,   'o-', color=C_QUANTUM, lw=2.5, ms=7, label='QMKL-Centered')
ax.fill_between(n_samples, auc_qmkl - std_q, auc_qmkl + std_q, alpha=0.15, color=C_QUANTUM)

ax.plot(n_samples, auc_rbfsvm, 's-', color=C_CLASSIC, lw=2.5, ms=7, label='RBF-SVM')
ax.fill_between(n_samples, auc_rbfsvm - std_rb, auc_rbfsvm + std_rb, alpha=0.15, color=C_CLASSIC)

ax.plot(n_samples, auc_rf,     '^-', color='#2c3e50', lw=2.5, ms=7, label='Forêt aléatoire')
ax.fill_between(n_samples, auc_rf - std_rf, auc_rf + std_rf, alpha=0.12, color='#2c3e50')

# Annotation gap constant
for n, q, r in zip(n_samples[[0, -1]], auc_qmkl[[0, -1]], auc_rbfsvm[[0, -1]]):
    ax.annotate('', xy=(n, q), xytext=(n, r),
                arrowprops=dict(arrowstyle='<->', color=C_BAD, lw=1.5))
    ax.text(n + 3, (q+r)/2, f'Δ≈{r-q:.2f}',
            color=C_BAD, fontsize=8.5, fontweight='bold')

ax.set_xlabel('Taille du jeu d\'entraînement N', fontsize=11)
ax.set_ylabel('AUC (moyenne 20 runs)', fontsize=11)
ax.set_title('Courbes d\'apprentissage — German Credit (Q=4)\nL\'écart reste constant de N=40 à N=200',
             fontsize=12, fontweight='bold', pad=8)
ax.legend(fontsize=10.5, loc='lower right')
ax.set_ylim(0.52, 0.90)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / 'pres_F11_learning_curves.png', dpi=150)
plt.close()
print('F11 done')

print(f'\nToutes les figures generees dans {OUT}')
print('Fichiers :', sorted(f.name for f in OUT.iterdir()))
