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
# Figure 2 : Les 12 kernels — organisation par famille et alpha
# ─────────────────────────────────────────────────────────────────────────────
families = [
    {
        'name': 'Famille Z',
        'paulis': "paulis=['Z']",
        'desc': 'Encodage\nindividuel seul',
        'kernels': [('Z, α=1.0', 0.698), ('Z, α=3.0', 0.712)],
        'color': '#3498db',
        'capture': 'Effets individuels\n(monotone)',
    },
    {
        'name': 'Famille ZZ',
        'paulis': "paulis=['Z','ZZ']",
        'desc': 'Individuel +\ninteractions paires',
        'kernels': [('ZZ, α=1.0', 0.734), ('ZZ, α=4.0', 0.756)],
        'color': '#e74c3c',
        'capture': 'Correlations\nlinéaires x pairs',
    },
    {
        'name': 'Famille XZ',
        'paulis': "paulis=['X','Z']",
        'desc': 'Encodage\nmixte X+Z',
        'kernels': [('XZ, α=0.5', 0.721), ('XZ, α=2.5', 0.738)],
        'color': '#2ecc71',
        'capture': 'Rotations dans\nplan XZ',
    },
    {
        'name': 'Famille YXX',
        'paulis': "paulis=['Y','XX']",
        'desc': 'Y individuel +\ninteractions XX',
        'kernels': [('YXX, α=0.6', 0.709), ('YXX, α=3.0', 0.744)],
        'color': '#f39c12',
        'capture': 'Correlations\nquadratiques',
    },
    {
        'name': 'Famille YZX',
        'paulis': "paulis=['Y','ZX']",
        'desc': 'Interactions\nà 3 corps',
        'kernels': [('YZX, α=0.6', 0.702), ('YZX, α=3.0', 0.718)],
        'color': '#9b59b6',
        'capture': 'Interactions\nnon-commutatives',
    },
    {
        'name': 'Famille Pauli',
        'paulis': "paulis=['Z','ZZ']",
        'desc': 'ZZ à alpha\nplus faible',
        'kernels': [('Pauli, α=0.6', 0.729), ('Pauli, α=2.5', 0.751)],
        'color': '#1abc9c',
        'capture': 'Regime\nde faible bandwidth',
    },
]

fig, axes = plt.subplots(1, 6, figsize=(16, 5))
fig.suptitle(
    '12 kernels quantiques : 6 familles × 2 valeurs de α   '
    r'  —   $\alpha$ contrôle la largeur de bande du kernel',
    fontsize=13, fontweight='bold', y=1.01
)

for ax, fam in zip(axes, families):
    ks = [k[0] for k in fam['kernels']]
    vs = [k[1] for k in fam['kernels']]
    bars = ax.bar([0, 1], vs, color=fam['color'], alpha=0.85,
                  edgecolor='white', width=0.6)
    for bar, val in zip(bars, vs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels([r'$\alpha$ petit', r'$\alpha$ grand'], fontsize=8.5)
    ax.set_ylim(0.60, 0.82)
    ax.set_ylabel('AUC individuelle\n(German Credit)' if ax == axes[0] else '', fontsize=8)
    ax.set_title(fam['name'], fontsize=10.5, fontweight='bold',
                 color=fam['color'], pad=4)

    # Zone de fond colorée
    for spine in ax.spines.values():
        spine.set_edgecolor(fam['color'])
        spine.set_linewidth(1.8)
    ax.set_facecolor(fam['color'] + '12')

    # Circuit Pauli en bas
    ax.text(0.5, 0.62, fam['paulis'], ha='center', va='bottom',
            fontsize=7.5, color='#555', style='italic',
            transform=ax.get_xaxis_transform())

    # Ce que capture la famille
    ax.text(0.5, -0.22, fam['capture'], ha='center', va='top',
            fontsize=8, color='#444',
            transform=ax.transAxes)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / 'pres_F_kernels.png', dpi=150)
plt.close()
print('Kernels done')

print('Figures sauvegardees dans', OUT)
