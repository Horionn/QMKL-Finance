"""
Génère :
  pres_F_circuit.png   — schéma du circuit PauliFeatureMap ZZ (Q=4, reps=1)
  pres_F_kernels.png   — présentation des 12 kernels organisés par famille
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from pathlib import Path

OUT = Path('results/presentation')
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 : Circuit PauliFeatureMap ZZ (Q=4, reps=1) dessiné à la main
# ─────────────────────────────────────────────────────────────────────────────

def gate_box(ax, x, y, label, color='#3498db', textcolor='white', w=0.55, h=0.40):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.03",
                         facecolor=color, edgecolor='black', lw=1.0, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=textcolor, zorder=4)

def cnot_control(ax, x, y, r=0.10):
    c = Circle((x, y), r, facecolor='#2c3e50', edgecolor='black', lw=1.0, zorder=3)
    ax.add_patch(c)

def cnot_target(ax, x, y, r=0.18):
    c = Circle((x, y), r, facecolor='white', edgecolor='#2c3e50', lw=1.5, zorder=3)
    ax.add_patch(c)
    ax.plot([x-r, x+r], [y, y], color='#2c3e50', lw=1.5, zorder=4)
    ax.plot([x, x], [y-r, y+r], color='#2c3e50', lw=1.5, zorder=4)

def vline(ax, x, y0, y1):
    ax.plot([x, x], [y0, y1], color='#2c3e50', lw=1.5, zorder=2)

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(-0.5, 13.5)
ax.set_ylim(-0.5, 4.2)
ax.axis('off')
ax.set_aspect('equal')

Q = 4
qubit_y = [3.5, 2.5, 1.5, 0.5]
labels_q = [r'$|0\rangle$', r'$|0\rangle$', r'$|0\rangle$', r'$|0\rangle$']
x_labels = [r'$x_0$', r'$x_1$', r'$x_2$', r'$x_3$']

# Lignes de qubits
x_start, x_end = 0.8, 13.0
for yq in qubit_y:
    ax.plot([x_start, x_end], [yq, yq], color='#555', lw=1.5, zorder=1)

# Labels qubit (gauche)
for i, (yq, lq) in enumerate(zip(qubit_y, labels_q)):
    ax.text(0.0, yq, lq, ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0.45, yq + 0.05, x_labels[i], ha='center', va='center',
            fontsize=8, color='#7f8c8d', style='italic')

# ── Bloc 1 : Hadamard ────────────────────────────────────────────────────────
x_h = 1.5
for yq in qubit_y:
    gate_box(ax, x_h, yq, 'H', color='#2ecc71', textcolor='white')

# ── Bloc 2 : RZ individuels (Z encoding) ─────────────────────────────────────
x_rz1 = 3.0
for i, yq in enumerate(qubit_y):
    gate_box(ax, x_rz1, yq, r'$R_z(2\alpha x_i)$', color='#3498db',
             textcolor='white', w=1.1)

# ── Bloc 3 : Entrelacement ZZ — paire (Q0, Q1) ───────────────────────────────
x_cx1a = 4.8
cnot_control(ax, x_cx1a, qubit_y[0])
cnot_target(ax, x_cx1a, qubit_y[1])
vline(ax, x_cx1a, qubit_y[1], qubit_y[0])

x_rzz1 = 6.0
gate_box(ax, x_rzz1, (qubit_y[0]+qubit_y[1])/2,
         r'$R_z(2\alpha x_i x_j)$', color='#8e44ad', textcolor='white', w=1.3, h=0.85)

x_cx1b = 7.2
cnot_control(ax, x_cx1b, qubit_y[0])
cnot_target(ax, x_cx1b, qubit_y[1])
vline(ax, x_cx1b, qubit_y[1], qubit_y[0])

# Lignes autour du bloc RZZ paire 0-1
for yq in [qubit_y[2], qubit_y[3]]:
    ax.plot([4.3, 7.7], [yq, yq], color='#555', lw=1.5, zorder=1)

# ── Bloc 4 : Entrelacement ZZ — paire (Q1, Q2) ───────────────────────────────
x_cx2a = 8.4
cnot_control(ax, x_cx2a, qubit_y[1])
cnot_target(ax, x_cx2a, qubit_y[2])
vline(ax, x_cx2a, qubit_y[2], qubit_y[1])

x_rzz2 = 9.6
gate_box(ax, x_rzz2, (qubit_y[1]+qubit_y[2])/2,
         r'$R_z(2\alpha x_i x_j)$', color='#8e44ad', textcolor='white', w=1.3, h=0.85)

x_cx2b = 10.8
cnot_control(ax, x_cx2b, qubit_y[1])
cnot_target(ax, x_cx2b, qubit_y[2])
vline(ax, x_cx2b, qubit_y[2], qubit_y[1])

# Lignes autour du bloc RZZ paire 1-2
for yq in [qubit_y[0], qubit_y[3]]:
    ax.plot([7.7, 11.3], [yq, yq], color='#555', lw=1.5, zorder=1)

# ── Sortie ────────────────────────────────────────────────────────────────────
ax.text(13.3, 2.0, r'$|\psi(\mathbf{x})\rangle$',
        ha='left', va='center', fontsize=13, color='#c0392b', fontweight='bold')

# ── Annotations des blocs ─────────────────────────────────────────────────────
def bracket(ax, x0, x1, y, label, color):
    dy = -0.32
    ax.annotate('', xy=(x0, y+dy), xytext=(x1, y+dy),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.5))
    ax.text((x0+x1)/2, y+dy-0.22, label, ha='center', va='top',
            fontsize=9, color=color, fontweight='bold')

bracket(ax, 1.2, 1.8, qubit_y[3], 'Superposition\n(Hadamard)', '#2ecc71')
bracket(ax, 2.4, 3.6, qubit_y[3], 'Encodage\nindividuel (Z)', '#3498db')
bracket(ax, 4.5, 11.2, qubit_y[3], 'Encodage des interactions par paires (ZZ)', '#8e44ad')

ax.set_title(
    'Circuit PauliFeatureMap ZZ  —  Q=4 qubits, reps=1, entanglement lineaire\n'
    r'Similarite : $K(x,x\prime)=|\langle\psi(x\prime)|\psi(x)\rangle|^2$',
    fontsize=12, fontweight='bold', pad=10
)

plt.tight_layout()
plt.savefig(OUT / 'pres_F_circuit.png', dpi=150)
plt.close()
print('Circuit done')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 : Les 12 kernels — heatmaps K(x,x') + niveaux d'interaction + AUC
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Données synthétiques 4D ────────────────────────────────────────────────
rng = np.random.default_rng(42)
# 3 points "proches" puis 2 points "lointains" → structure en blocs visible
X_syn = np.zeros((5, 4))
X_syn[:3] = rng.uniform(0.2, 0.8, (3, 4))   # cluster proche
X_syn[3:]  = rng.uniform(1.4, 2.0, (2, 4))   # cluster lointain


def kernel_Z(X, alpha):
    """K_ij = prod_k cos²(alpha*(x_ik - x_jk))"""
    n = len(X)
    K = np.ones((n, n))
    for k in range(X.shape[1]):
        diff = X[:, k:k+1] - X[:, k].reshape(1, -1)
        K *= np.cos(alpha * diff) ** 2
    return K


def kernel_ZZ(X, alpha):
    """ZZ ajoute termes croisés cos²(alpha*(x_i·x_j - x_k·x_l)) pour paires (k,l)"""
    K = kernel_Z(X, alpha)
    n, d = len(X), X.shape[1]
    for k in range(d):
        for l in range(k + 1, d):
            cross = (X[:, k] * X[:, l])
            diff = cross.reshape(-1, 1) - cross.reshape(1, -1)
            K *= np.cos(alpha * diff) ** 2
    return K


def kernel_XZ(X, alpha):
    """XZ : mixte X+Z — rotations dans plan XZ (sin pour X, cos pour Z)"""
    n, d = len(X), X.shape[1]
    K = np.ones((n, n))
    for k in range(d):
        diff = X[:, k:k+1] - X[:, k].reshape(1, -1)
        # alternance sin/cos selon parité
        if k % 2 == 0:
            K *= np.sin(alpha * diff) ** 2
        else:
            K *= np.cos(alpha * diff) ** 2
    for k in range(d):
        for l in range(k + 1, d):
            cross = (X[:, k] * X[:, l])
            diff = cross.reshape(-1, 1) - cross.reshape(1, -1)
            K *= (0.5 + 0.5 * np.cos(alpha * diff))
    return np.clip(K, 0, 1)


def kernel_YXX(X, alpha):
    """YXX : Y individuel + interactions XX — corrélations quadratiques"""
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


def kernel_YZX(X, alpha):
    """YZX : interactions à 3 corps non-commutatives"""
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


def kernel_Pauli(X, alpha):
    """Pauli (ZZ à régime faible bande) — similaire ZZ mais alpha réduit de 40%"""
    return kernel_ZZ(X, alpha * 0.6)


FAMILIES = [
    {
        'name': 'Famille Z',
        'color': '#3498db',
        'paulis': "paulis=['Z']",
        'badge': 'Local (1-corps)',
        'badge_color': '#2980b9',
        'interp': 'Effets individuels\nsans interaction',
        'kernels': [('α=1.0', 0.698), ('α=3.0', 0.712)],
        'fn': kernel_Z,
        'alphas': [1.0, 3.0],
    },
    {
        'name': 'Famille ZZ',
        'color': '#e74c3c',
        'paulis': "paulis=['Z','ZZ']",
        'badge': 'Pairwise (2-corps)',
        'badge_color': '#c0392b',
        'interp': 'Corrélations\npaires x·y',
        'kernels': [('α=1.0', 0.734), ('α=4.0', 0.756)],
        'fn': kernel_ZZ,
        'alphas': [1.0, 4.0],
    },
    {
        'name': 'Famille XZ',
        'color': '#2ecc71',
        'paulis': "paulis=['X','Z']",
        'badge': 'Pairwise (2-corps)',
        'badge_color': '#27ae60',
        'interp': 'Rotations dans\nplan XZ',
        'kernels': [('α=0.5', 0.721), ('α=2.5', 0.738)],
        'fn': kernel_XZ,
        'alphas': [0.5, 2.5],
    },
    {
        'name': 'Famille YXX',
        'color': '#f39c12',
        'paulis': "paulis=['Y','XX']",
        'badge': '3-corps non-comm.',
        'badge_color': '#d68910',
        'interp': 'Corrélations\nquadratiques',
        'kernels': [('α=0.6', 0.709), ('α=3.0', 0.744)],
        'fn': kernel_YXX,
        'alphas': [0.6, 3.0],
    },
    {
        'name': 'Famille YZX',
        'color': '#9b59b6',
        'paulis': "paulis=['Y','ZX']",
        'badge': '3-corps non-comm.',
        'badge_color': '#7d3c98',
        'interp': 'Interactions\nnon-commutatives',
        'kernels': [('α=0.6', 0.702), ('α=3.0', 0.718)],
        'fn': kernel_YZX,
        'alphas': [0.6, 3.0],
    },
    {
        'name': 'Famille Pauli',
        'color': '#1abc9c',
        'paulis': "paulis=['Z','ZZ']",
        'badge': 'Pairwise (2-corps)',
        'badge_color': '#17a589',
        'interp': 'ZZ à faible\nbande passante',
        'kernels': [('α=0.6', 0.729), ('α=2.5', 0.751)],
        'fn': kernel_Pauli,
        'alphas': [0.6, 2.5],
    },
]

# ── Layout : 6 colonnes, 3 sous-rangées par colonne ───────────────────────
fig = plt.figure(figsize=(17, 7.5))
fig.suptitle(
    '12 kernels quantiques — heatmap K(x,xʼ) + niveau d\'interaction',
    fontsize=13, fontweight='bold', y=1.00
)

outer = gridspec.GridSpec(1, 6, figure=fig, wspace=0.35)

for col_idx, fam in enumerate(FAMILIES):
    inner = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=outer[col_idx],
        hspace=0.08,
        height_ratios=[0.45, 0.25, 0.30],
    )

    # ── Rangée 1 : Heatmap kernel matrix 5×5 ─────────────────────────────
    ax_heat = fig.add_subplot(inner[0])
    K = fam['fn'](X_syn, fam['alphas'][0])
    np.fill_diagonal(K, 1.0)
    im = ax_heat.imshow(K, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax_heat.set_xticks(range(5))
    ax_heat.set_yticks(range(5))
    ax_heat.set_xticklabels(['x₁', 'x₂', 'x₃', 'x₄', 'x₅'], fontsize=6.5)
    ax_heat.set_yticklabels(['x₁', 'x₂', 'x₃', 'x₄', 'x₅'], fontsize=6.5)
    ax_heat.tick_params(length=2, pad=1)
    # Lignes de séparation des clusters
    ax_heat.axhline(2.5, color='white', lw=1.5, alpha=0.8)
    ax_heat.axvline(2.5, color='white', lw=1.5, alpha=0.8)
    # Valeurs dans les cases
    for i in range(5):
        for j in range(5):
            ax_heat.text(j, i, f'{K[i,j]:.2f}', ha='center', va='center',
                         fontsize=5.5, color='#333' if K[i,j] < 0.6 else 'white')
    ax_heat.set_title(fam['name'], fontsize=9, fontweight='bold',
                      color=fam['color'], pad=3)
    for spine in ax_heat.spines.values():
        spine.set_edgecolor(fam['color'])
        spine.set_linewidth(1.5)

    # ── Rangée 2 : Badge niveau + interprétation ─────────────────────────
    ax_badge = fig.add_subplot(inner[1])
    ax_badge.axis('off')
    # Badge FancyBboxPatch
    bbox = FancyBboxPatch((0.05, 0.55), 0.90, 0.35,
                           boxstyle="round,pad=0.05",
                           facecolor=fam['badge_color'], edgecolor='none',
                           transform=ax_badge.transAxes, zorder=3)
    ax_badge.add_patch(bbox)
    ax_badge.text(0.50, 0.73, fam['badge'],
                  ha='center', va='center', fontsize=7.5, fontweight='bold',
                  color='white', transform=ax_badge.transAxes, zorder=4)
    ax_badge.text(0.50, 0.20, fam['interp'],
                  ha='center', va='center', fontsize=7,
                  color='#555', style='italic',
                  transform=ax_badge.transAxes)
    ax_badge.text(0.50, -0.10, fam['paulis'],
                  ha='center', va='center', fontsize=6.5,
                  color='#777', transform=ax_badge.transAxes)

    # ── Rangée 3 : Mini-barres AUC ────────────────────────────────────────
    ax_bar = fig.add_subplot(inner[2])
    ks = [k[0] for k in fam['kernels']]
    vs = [k[1] for k in fam['kernels']]
    bars = ax_bar.bar([0, 1], vs, color=fam['color'], alpha=0.88,
                      edgecolor='white', width=0.55)
    for bar, val in zip(bars, vs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color=fam['color'])
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(ks, fontsize=7)
    ax_bar.set_ylim(0.65, 0.80)
    ax_bar.set_yticks([0.70, 0.75, 0.80])
    ax_bar.set_yticklabels(['0.70', '0.75', '0.80'], fontsize=6)
    if col_idx == 0:
        ax_bar.set_ylabel('AUC\n(German Credit)', fontsize=6.5)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax_bar.spines[spine].set_edgecolor('#aaa')
    ax_bar.tick_params(axis='both', length=2, pad=1)
    ax_bar.set_facecolor(fam['color'] + '15')

plt.savefig(OUT / 'pres_F_kernels.png', dpi=150, bbox_inches='tight')
plt.close()
print('Kernels (heatmaps) done')

print('Figures sauvegardees dans', OUT)
