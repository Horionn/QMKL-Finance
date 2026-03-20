"""
Génère :
  pres_qubo_complexity.png  — Explosion combinatoire 2^M, seuils classique/quantique
  pres_qubo_matrix.png      — Matrice Q 12×12 + tableau de décision QUBO
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from pathlib import Path

OUT = Path('results/presentation')
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

# ─────────────────────────────────────────────────────────────────────────────
# Figure A : Explosion combinatoire — quand le quantique devient indispensable
# ─────────────────────────────────────────────────────────────────────────────

M_values = np.array([5, 8, 10, 12, 15, 20, 25, 30, 40, 50])

classical_bf   = 2.0 ** M_values / 1e8        # O(2^M) brute force, unité=100M ops/s
heuristic      = M_values ** 3 / 1e6          # O(M³) heuristique classique
quantum_anneal = M_values ** 1.5 / 1e4        # O(M^1.5) recuit quantique empirique

fig, ax = plt.subplots(figsize=(13, 6.5))

# ── Régions de couleur ────────────────────────────────────────────────────────
ax.axvspan(M_values[0] - 1, 25, alpha=0.15, color='#27ae60', zorder=0)
ax.axvspan(25, 35, alpha=0.15, color='#f39c12', zorder=0)
ax.axvspan(35, M_values[-1] + 2, alpha=0.15, color='#3498db', zorder=0)

region_labels = [
    (15,    1e-1, '#27ae60', 'Zone classique\n(M < 25)'),
    (30,    1e-1, '#d35400', 'Transition\n(25–35)'),
    (43.5,  1e-1, '#2980b9', 'Zone quantique\n(M > 35)'),
]
for xc, yc, col, txt in region_labels:
    ax.text(xc, yc, txt, ha='center', va='bottom', fontsize=8.5,
            color=col, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=col))

# ── Courbes ───────────────────────────────────────────────────────────────────
ax.semilogy(M_values, classical_bf,   'r--',  lw=2.5, marker='o', ms=6,
            label=r'Brute force classique  $O(2^M)$')
ax.semilogy(M_values, heuristic,      color='#e67e22', ls=':', lw=2.5,
            marker='s', ms=6, label=r'Heuristique classique  $O(M^3)$')
ax.semilogy(M_values, quantum_anneal, 'b-',   lw=2.5, marker='^', ms=6,
            label=r'Recuit quantique  $O(M^{1.5})$  ★ empirique')

# ── Repères temporels axe Y ───────────────────────────────────────────────────
time_refs = {
    1e-3: '1 ms',
    1e0:  '1 s',
    3.6e3 / 1e8: '1 h',   # 1h = 3600s, normalisé par 1e8 ops/s
    3.15e7 / 1e8: '1 an',
}
# Axe Y en secondes (brute force = 2^M / 1e8 ops/s)
y_ticks_s = [1e-3, 1e0, 36, 3.15e7]
y_labels_s = ['1 ms', '1 s', '1 heure', '1 an']
ax2 = ax.twinx()
ax2.set_yscale('log')
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(y_ticks_s)
ax2.set_yticklabels(y_labels_s, fontsize=9, color='#c0392b')
ax2.set_ylabel('Temps estimé (CPU brute force)', fontsize=9, color='#c0392b')

# ── Annotations clés ─────────────────────────────────────────────────────────
# M=12 : notre projet
idx12 = np.where(M_values == 12)[0][0]
ax.annotate(
    'Notre projet\nM=12 → 4 096 sous-ensembles',
    xy=(12, classical_bf[idx12]),
    xytext=(14, classical_bf[idx12] * 80),
    fontsize=8.5, color='#27ae60', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', edgecolor='#27ae60'),
)

# M=25 : limite classique
ax.axvline(25, color='#d35400', lw=1.5, ls='--', alpha=0.8)
idx25 = np.where(M_values == 25)[0][0]
ax.annotate(
    'Limite classique\n33 millions sous-ensembles',
    xy=(25, classical_bf[idx25]),
    xytext=(27, classical_bf[idx25] * 0.05),
    fontsize=8.5, color='#d35400', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdebd0', edgecolor='#d35400'),
)

# M=50 : quantique seul viable
idx50 = np.where(M_values == 50)[0][0]
ax.annotate(
    'M=50 : ~10¹⁵\nQuantique seul viable',
    xy=(50, classical_bf[idx50]),
    xytext=(44, classical_bf[idx50] * 0.001),
    fontsize=8.5, color='#2980b9', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#d6eaf8', edgecolor='#2980b9'),
)

ax.set_xlabel('Nombre de kernels M', fontsize=11)
ax.set_ylabel('Complexité (unités normalisées)', fontsize=11)
ax.set_title(
    'Quand le quantique devient-il indispensable pour QUBO ?\n'
    'Explosion combinatoire $2^M$ vs heuristiques',
    fontsize=12, fontweight='bold', pad=10
)
ax.legend(fontsize=9.5, loc='upper left', framealpha=0.9)
ax.set_xlim(M_values[0] - 1, M_values[-1] + 2)
ax.grid(True, which='both', alpha=0.25, lw=0.8)
ax.tick_params(axis='both', labelsize=9)

# Note honnêteté académique
fig.text(0.5, -0.02,
         '★ Recuit quantique : heuristique empiriquement polynomial — pas de garantie théorique',
         ha='center', fontsize=8.5, color='#555', style='italic')

plt.tight_layout()
plt.savefig(OUT / 'pres_qubo_complexity.png', dpi=150, bbox_inches='tight')
plt.close()
print('QUBO complexity done')


# ─────────────────────────────────────────────────────────────────────────────
# Figure B : Matrice Q 12×12 + tableau de décision QUBO
# ─────────────────────────────────────────────────────────────────────────────

# ── Données synthétiques pour calculer alignement de Frobenius ─────────────
rng = np.random.default_rng(42)
X_syn = np.zeros((5, 4))
X_syn[:3] = rng.uniform(0.2, 0.8, (3, 4))
X_syn[3:]  = rng.uniform(1.4, 2.0, (2, 4))


def kernel_Z_mat(X, alpha):
    n = len(X)
    K = np.ones((n, n))
    for k in range(X.shape[1]):
        diff = X[:, k:k+1] - X[:, k].reshape(1, -1)
        K *= np.cos(alpha * diff) ** 2
    return K


def kernel_ZZ_mat(X, alpha):
    K = kernel_Z_mat(X, alpha)
    n, d = len(X), X.shape[1]
    for k in range(d):
        for l in range(k + 1, d):
            cross = (X[:, k] * X[:, l])
            diff = cross.reshape(-1, 1) - cross.reshape(1, -1)
            K *= np.cos(alpha * diff) ** 2
    return K


def kernel_XZ_mat(X, alpha):
    n, d = len(X), X.shape[1]
    K = np.ones((n, n))
    for k in range(d):
        diff = X[:, k:k+1] - X[:, k].reshape(1, -1)
        K *= np.sin(alpha * diff) ** 2 if k % 2 == 0 else np.cos(alpha * diff) ** 2
    for k in range(d):
        for l in range(k + 1, d):
            cross = (X[:, k] * X[:, l])
            diff = cross.reshape(-1, 1) - cross.reshape(1, -1)
            K *= (0.5 + 0.5 * np.cos(alpha * diff))
    return np.clip(K, 0, 1)


def kernel_YXX_mat(X, alpha):
    n, d = len(X), X.shape[1]
    K = np.ones((n, n))
    for k in range(d):
        diff = X[:, k:k+1] - X[:, k].reshape(1, -1)
        K *= np.sin(alpha * diff + np.pi / 4) ** 2
    for k in range(d):
        for l in range(k + 1, d):
            xx = X[:, k] ** 2 + X[:, l] ** 2
            diff = xx.reshape(-1, 1) - xx.reshape(1, -1)
            K *= np.cos(alpha * diff / 2) ** 2
    return np.clip(K, 0, 1)


def kernel_YZX_mat(X, alpha):
    n, d = len(X), X.shape[1]
    K = np.ones((n, n))
    for k in range(d):
        diff = X[:, k:k+1] - X[:, k].reshape(1, -1)
        K *= np.abs(np.sin(alpha * diff + np.pi / 6))
    for k in range(d - 2):
        triple = X[:, k] * X[:, k+1] - X[:, k+2]
        diff = triple.reshape(-1, 1) - triple.reshape(1, -1)
        K *= np.cos(alpha * diff / 3) ** 2
    return np.clip(K, 0, 1)


def kernel_Pauli_mat(X, alpha):
    return kernel_ZZ_mat(X, alpha * 0.6)


# 12 kernels avec AUC (scores individuels s_m)
KERNELS_12 = [
    ('Z α=1.0',   kernel_Z_mat,    1.0, 0.698),
    ('Z α=3.0',   kernel_Z_mat,    3.0, 0.712),
    ('ZZ α=1.0',  kernel_ZZ_mat,   1.0, 0.734),
    ('ZZ α=4.0',  kernel_ZZ_mat,   4.0, 0.756),
    ('XZ α=0.5',  kernel_XZ_mat,   0.5, 0.721),
    ('XZ α=2.5',  kernel_XZ_mat,   2.5, 0.738),
    ('YXX α=0.6', kernel_YXX_mat,  0.6, 0.709),
    ('YXX α=3.0', kernel_YXX_mat,  3.0, 0.744),
    ('YZX α=0.6', kernel_YZX_mat,  0.6, 0.702),
    ('YZX α=3.0', kernel_YZX_mat,  3.0, 0.718),
    ('Pli α=0.6', kernel_Pauli_mat, 0.6, 0.729),
    ('Pli α=2.5', kernel_Pauli_mat, 2.5, 0.751),
]

M = len(KERNELS_12)
s = np.array([k[3] for k in KERNELS_12])

# Calcul des matrices kernel et de l'alignement de Frobenius
Kmats = []
for name, fn, alpha, auc in KERNELS_12:
    K = fn(X_syn, alpha)
    np.fill_diagonal(K, 1.0)
    Kmats.append(K)

# Alignement A[m,m'] = <K_m, K_m'>_F / (||K_m||_F * ||K_m'||_F)
A = np.zeros((M, M))
for m in range(M):
    for mp in range(M):
        num = np.sum(Kmats[m] * Kmats[mp])
        denom = np.sqrt(np.sum(Kmats[m] ** 2) * np.sum(Kmats[mp] ** 2))
        A[m, mp] = num / denom if denom > 0 else 0.0

# Matrice Q QUBO : Q[m,m] = -s_m, Q[m,m'] = lambda * A[m,m']
lam = 1.0
Q_mat = lam * A.copy()
np.fill_diagonal(Q_mat, -s)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5.5))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.40, width_ratios=[1.1, 0.9])

# ── Subplot gauche : heatmap Q ────────────────────────────────────────────────
ax_q = fig.add_subplot(gs[0])

vmax = np.max(np.abs(Q_mat))
im = ax_q.imshow(Q_mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
plt.colorbar(im, ax=ax_q, fraction=0.046, pad=0.04)

labels = [k[0] for k in KERNELS_12]
ax_q.set_xticks(range(M))
ax_q.set_yticks(range(M))
ax_q.set_xticklabels(labels, rotation=45, ha='right', fontsize=6.5)
ax_q.set_yticklabels(labels, fontsize=6.5)
ax_q.set_title('Matrice Q QUBO\n(bleu = sélectionner · rouge = pénaliser)',
               fontsize=9.5, fontweight='bold', pad=6)

# Annotations diagonale/hors-diagonale
ax_q.text(5.5, -1.5, 'Diagonale : $Q_{mm} = -s_m$\n(performance)',
          ha='center', va='bottom', fontsize=7.5, color='#2980b9',
          bbox=dict(boxstyle='round,pad=0.2', facecolor='#d6eaf8', edgecolor='#2980b9'))
ax_q.text(5.5, 13.5, 'Hors-diag. : $Q_{mm\'} = \\lambda A_{mm\'}$\n(redondance)',
          ha='center', va='top', fontsize=7.5, color='#c0392b',
          bbox=dict(boxstyle='round,pad=0.2', facecolor='#fde8e8', edgecolor='#c0392b'))

# Rectangle doré sur les 2 kernels sélectionnés : ZZ α=4.0 (idx=3) et XZ α=0.5 (idx=4)
selected_idx = [3, 4]   # ZZ α=4.0, XZ α=0.5
for i in selected_idx:
    for j in selected_idx:
        rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                          linewidth=2.5, edgecolor='gold',
                          facecolor='none', zorder=5)
        ax_q.add_patch(rect)
ax_q.text(4.5, 4.5, '★', ha='center', va='center', fontsize=14,
          color='gold', zorder=6,
          bbox=dict(boxstyle='round,pad=0.1', facecolor='none', edgecolor='none'))

# Légende sélection
ax_q.text(-0.5, 12.5,
          'Résultat : λ*=1.0\n→ ZZ α=4.0 + XZ α=0.5\nAUC = 0.823',
          ha='left', va='center', fontsize=7.5, color='#856404',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9c4',
                    edgecolor='gold', lw=1.5))

# ── Subplot droit : tableau de décision ──────────────────────────────────────
ax_t = fig.add_subplot(gs[1])
ax_t.axis('off')
ax_t.set_xlim(0, 1)
ax_t.set_ylim(0, 1)

ax_t.set_title('Logique de sélection QUBO', fontsize=9.5, fontweight='bold', pad=6)

# Entêtes colonnes
ax_t.text(0.50, 0.90, 'AUC haute ($s_m$ élevé)', ha='center', va='center',
          fontsize=8.5, fontweight='bold', color='#27ae60')
ax_t.text(0.85, 0.90, 'AUC basse', ha='center', va='center',
          fontsize=8.5, fontweight='bold', color='#c0392b')
# Entêtes lignes
ax_t.text(0.05, 0.67, 'Diversifiés\n($A_{mm\'} < 0.4$)', ha='left', va='center',
          fontsize=8, fontweight='bold', color='#2c3e50')
ax_t.text(0.05, 0.33, 'Redondants\n($A_{mm\'} > 0.7$)', ha='left', va='center',
          fontsize=8, fontweight='bold', color='#7f8c8d')

# 4 cellules
cells = [
    # (x, y, w, h, facecolor, edgecolor, text, textcolor)
    (0.30, 0.50, 0.38, 0.32, '#d5f5e3', '#27ae60',
     'SÉLECTIONNER\n✓ Optimal', '#1a5e34'),
    (0.72, 0.50, 0.26, 0.32, '#fdebd0', '#d35400',
     'Peut-être\n(contexte)', '#7d3c00'),
    (0.30, 0.14, 0.38, 0.32, '#fdebd0', '#d35400',
     'Choisir 1/2\n(redondant)', '#7d3c00'),
    (0.72, 0.14, 0.26, 0.32, '#fde8e8', '#c0392b',
     'IGNORER\n✗ Bruit', '#7b1a1a'),
]
for x, y, w, h, fc, ec, txt, tc in cells:
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle='round,pad=0.02',
                           facecolor=fc, edgecolor=ec, lw=1.5,
                           transform=ax_t.transAxes, zorder=2)
    ax_t.add_patch(rect)
    ax_t.text(x + w / 2, y + h / 2, txt,
              ha='center', va='center', fontsize=8, fontweight='bold',
              color=tc, transform=ax_t.transAxes, zorder=3)

# Lignes séparatrices
ax_t.axhline(0.47, xmin=0.30, xmax=1.00, color='#bbb', lw=1, ls='--')
ax_t.axvline(0.72, ymin=0.14, ymax=0.82, color='#bbb', lw=1, ls='--')

# Résultat
ax_t.text(0.50, 0.05,
          'λ*=1.0 → ZZ α=4.0 + XZ α=0.5 · AUC=0.823',
          ha='center', va='center', fontsize=7.5, fontweight='bold',
          color='#856404',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9c4',
                    edgecolor='gold', lw=1.5),
          transform=ax_t.transAxes)

fig.suptitle('QUBO-Boost — comment choisir les 2 meilleurs kernels sur 12',
             fontsize=11, fontweight='bold', y=1.02)

plt.savefig(OUT / 'pres_qubo_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print('QUBO matrix done')

print('Figures QUBO sauvegardees dans', OUT)
