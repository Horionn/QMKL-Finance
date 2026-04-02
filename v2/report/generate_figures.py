"""
Script de génération des figures pour rapport_qmkl.tex
Toutes les valeurs numériques sont calculées ici et vérifiables.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import time, json, os

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

C_Z   = '#3498db'
C_ZZ  = '#e74c3c'
C_FIT = '#2c3e50'

# ─────────────────────────────────────────────────────────────
# Noyaux analytiques (reproduits de src/kernels/analytical.py)
# ─────────────────────────────────────────────────────────────

def kernel_Z(X1, X2, alpha=1.0):
    """K_Z(x,x') = prod_k cos^2(alpha*(x_k - x'_k))"""
    diff = X1[:, None, :] - X2[None, :, :]   # (N1, N2, Q)
    return np.prod(np.cos(alpha * diff) ** 2, axis=-1)

def kernel_ZZ(X1, X2, alpha=1.0):
    """K_ZZ = K_Z * prod_{k<l} cos^2(alpha*(x_k*x_l - x'_k*x'_l))"""
    K = kernel_Z(X1, X2, alpha=alpha)
    Q = X1.shape[1]
    for k in range(Q):
        for l in range(k + 1, Q):
            prod1 = (X1[:, k] * X1[:, l])[:, None]
            prod2 = (X2[:, k] * X2[:, l])[None, :]
            K *= np.cos(alpha * (prod1 - prod2)) ** 2
    return K

def sigma_off(K):
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.std(K[mask]))

def mean_off(K):
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(K[mask]))

def quantum_scale(X_train, X_test=None):
    sc = Pipeline([("std", StandardScaler()),
                   ("mms", MinMaxScaler(feature_range=(0, 2 * np.pi)))])
    Xt = sc.fit_transform(X_train)
    if X_test is None:
        return Xt
    return Xt, sc.transform(X_test)

# ─────────────────────────────────────────────────────────────
# FIGURE 1 — Barren Plateaus
# ─────────────────────────────────────────────────────────────
print("=== Figure 1 : Barren Plateaus ===")

Q_vals = [2, 4, 6, 8, 10, 12]
N_BP   = 80
ALPHA  = 1.0
sigma_z, sigma_zz, mean_z, mean_zz = [], [], [], []

for Q in Q_vals:
    rng = np.random.RandomState(42)
    X = rng.uniform(0, 2 * np.pi, (N_BP, Q))
    Kz  = kernel_Z(X, X, alpha=ALPHA)
    Kzz = kernel_ZZ(X, X, alpha=ALPHA)
    sz  = sigma_off(Kz)
    szz = sigma_off(Kzz)
    sigma_z.append(sz)
    sigma_zz.append(szz)
    mean_z.append(mean_off(Kz))
    mean_zz.append(mean_off(Kzz))
    print(f"  Q={Q:2d}  σ_off(K_Z)={sz:.5f}  σ_off(K_ZZ)={szz:.5f}"
          f"  mean(K_Z)={mean_off(Kz):.5f}  mean(K_ZZ)={mean_off(Kzz):.5f}")

sigma_z  = np.array(sigma_z)
sigma_zz = np.array(sigma_zz)
Q_arr    = np.array(Q_vals, dtype=float)

# Ajustement exponentiel : f(Q) = A * exp(-b * Q)
def exp_decay(q, A, b):
    return A * np.exp(-b * q)

pz,  _ = curve_fit(exp_decay, Q_arr, sigma_z,  p0=[0.5, 0.3], maxfev=5000)
pzz, _ = curve_fit(exp_decay, Q_arr, sigma_zz, p0=[0.5, 0.4], maxfev=5000)
Q_fine = np.linspace(2, 12, 100)

print(f"\n  Fit K_Z  : A={pz[0]:.4f}, b={pz[1]:.4f}  "
      f"=> decay ~ exp(-{pz[1]:.3f}*Q), "
      f"2^(-Q) coeff = {np.log(2):.3f}")
print(f"  Fit K_ZZ : A={pzz[0]:.4f}, b={pzz[1]:.4f}")

# Calcul AUC vs Q (breast_cancer, PCA, K_Z uniquement)
print("\n  Calcul AUC vs Q (breast_cancer, N=200, 4-fold, 5 reps)...")
data = load_breast_cancer()
Xbc, ybc = data.data, data.target
rng0 = np.random.RandomState(42)
idx  = rng0.choice(len(Xbc), 200, replace=False)
Xbc200, ybc200 = Xbc[idx], ybc[idx]

auc_Q, auc_Q_std = [], []
for Q in Q_vals:
    aucs = []
    for rep in range(5):
        kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=rep * 7)
        for tr, te in kf.split(Xbc200, ybc200):
            Xtr, Xte = quantum_scale(Xbc200[tr], Xbc200[te])
            pca = PCA(n_components=Q, random_state=0)
            Xtr_q = pca.fit_transform(Xtr)
            Xte_q = pca.transform(Xte)
            Ktr = kernel_Z(Xtr_q, Xtr_q, alpha=ALPHA)
            Kte = kernel_Z(Xte_q, Xtr_q, alpha=ALPHA)
            clf = SVC(kernel='precomputed', C=1.0, probability=True)
            clf.fit(Ktr, ybc200[tr])
            prob = clf.predict_proba(Kte)[:, 1]
            aucs.append(roc_auc_score(ybc200[te], prob))
    auc_Q.append(np.mean(aucs))
    auc_Q_std.append(np.std(aucs))
    print(f"  Q={Q:2d}  AUC(K_Z)={auc_Q[-1]:.4f} ± {auc_Q_std[-1]:.4f}")

# ─── Tracé Figure 1 ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.semilogy(Q_arr, sigma_z,  'o-', color=C_Z,  lw=2, ms=7, label=r'$K_Z$')
ax.semilogy(Q_arr, sigma_zz, 's-', color=C_ZZ, lw=2, ms=7, label=r'$K_{ZZ}$')
ax.semilogy(Q_fine, exp_decay(Q_fine, *pz),  '--', color=C_Z,  alpha=0.6,
            label=fr'Fit $e^{{-{pz[1]:.2f}Q}}$')
ax.semilogy(Q_fine, exp_decay(Q_fine, *pzz), '--', color=C_ZZ, alpha=0.6,
            label=fr'Fit $e^{{-{pzz[1]:.2f}Q}}$')
# Référence 2^{-Q}
ref = 0.25 * np.exp(-np.log(2) * Q_fine)
ax.semilogy(Q_fine, ref, ':', color='gray', alpha=0.5, label=r'Référence $2^{-Q}$')
ax.set_xlabel('Nombre de qubits $Q$')
ax.set_ylabel(r'$\sigma_{\mathrm{off}}(K)$ (échelle log)')
ax.set_title('Concentration du noyau')
ax.set_xticks(Q_vals)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

ax = axes[1]
ax.errorbar(Q_vals, auc_Q, yerr=auc_Q_std,
            fmt='o-', color=C_Z, lw=2, ms=7, capsize=4,
            label=r'$K_Z$ (ACP + SVM)')
ax.axhline(0.5, color='gray', ls=':', lw=1.5, alpha=0.8, label='Niveau aléatoire')
ax.axvspan(6.5, 12.5, alpha=0.08, color='red', label='Régime barren plateau')
ax.set_xlabel('Nombre de qubits $Q$')
ax.set_ylabel('AUC ROC')
ax.set_title('Dégradation de la performance ($K_Z$, Breast Cancer)')
ax.set_xticks(Q_vals)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle('Barren Plateaus — Breast Cancer ($N=200$, $\\alpha=1.0$)',
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_barren_plateau.pdf"))
plt.close(fig)
print("  → fig_barren_plateau.pdf sauvegardé")

# ─────────────────────────────────────────────────────────────
# FIGURE 2 — Complexité : simulation classique vs circuit quantique
# ─────────────────────────────────────────────────────────────
print("\n=== Figure 2 : Complexité simulation vs circuit ===")

# Mémoire requise pour la simulation statevector : 2^Q amplitudes * 16 octets (complex128)
Q_theory = np.arange(2, 52, 2)
mem_gb   = (2.0 ** Q_theory) * 16 / 1e9   # en Go

print("  Mémoire statevector :")
for Q, m in zip(Q_theory, mem_gb):
    if m < 1e6:
        print(f"  Q={Q:2d}  mem={m:.3g} Go")

# Seuils RAM typiques
RAM_LIMITS = [(8, '8 Go (laptop)',       '--', '#7f8c8d'),
              (64, '64 Go (workstation)', ':',  '#e67e22'),
              (1e6, '1 Po (data center)', '-.', '#8e44ad')]
for gb, label, _, _ in RAM_LIMITS:
    q_star = np.log2(gb * 1e9 / 16)
    print(f"  Limite {label}: Q* = {q_star:.1f}")

# Ressources circuit quantique (ZZFeatureMap, reps=1)
# CNOT: Q*(Q-1)/2 paires × 2 CX = Q*(Q-1)
# Portes 1-qubit: 3Q  (Ry, Rz, Rz)
Q_gate    = np.arange(2, 52, 2)
n_cnot    = Q_gate * (Q_gate - 1)
n_single  = Q_gate * 3
gates_tot = n_cnot + n_single

print("\n  Portes ZZFeatureMap (reps=1) :")
for Q, g in zip(Q_gate[::2], gates_tot[::2]):
    print(f"  Q={Q:2d}  total={g:5d}  (CNOT={Q*(Q-1):4d}, 1q={3*Q:3d})")

# ─── Tracé Figure 2 ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# — Panneau gauche : mémoire simulation (exponentiel) —
ax1 = axes[0]
ax1.semilogy(Q_theory, mem_gb, 'o-', color='#e74c3c', lw=2, ms=5,
             label=r'Simulation statevector ($2^Q \times 16\,$o)')
for gb, label, ls, col in RAM_LIMITS:
    ax1.axhline(gb, color=col, ls=ls, alpha=0.75, lw=1.4, label=label)
    q_cross = np.log2(gb * 1e9 / 16)
    if 2 <= q_cross <= 50:
        ax1.axvline(q_cross, color=col, ls=':', alpha=0.35, lw=1)
        ax1.text(q_cross + 0.5, gb * 1.5, f'$Q={q_cross:.0f}$',
                 fontsize=7.5, color=col, va='bottom')
ax1.set_xlabel('Nombre de qubits $Q$')
ax1.set_ylabel('Mémoire requise (Go, échelle log)')
ax1.set_title('Simulation classique : $O(2^Q)$')
ax1.set_xticks(Q_theory[::2])
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, which='both', alpha=0.25)

# — Panneau droit : ressources circuit (polynomial) —
ax2 = axes[1]
ax2.plot(Q_gate, n_cnot,    's--', color='#e74c3c', ms=5, lw=1.5,
         label='CNOT  [$Q(Q-1)$]')
ax2.plot(Q_gate, n_single,  'o--', color='#3498db', ms=5, lw=1.5,
         label='1-qubit  [$3Q$]')
ax2.plot(Q_gate, gates_tot, 'D-',  color='#2c3e50', ms=5, lw=2,
         label='Total  [$Q^2+2Q$]')
ax2.set_xlabel('Nombre de qubits $Q$')
ax2.set_ylabel('Nombre de portes quantiques')
ax2.set_title('Circuit ZZFeatureMap : $O(Q^2)$')
ax2.set_xticks(Q_gate[::2])
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25)

fig.suptitle('Complexité : simulation classique (exponentielle) vs circuit quantique (polynomiale)',
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_complexity.pdf"))
plt.close(fig)
print("  → fig_complexity.pdf sauvegardé")

# ─────────────────────────────────────────────────────────────
# FIGURE 3 — Matrice QUBO
# ─────────────────────────────────────────────────────────────
print("\n=== Figure 3 : Matrice QUBO ===")

d_q, M_q, Q_q = 8, 3, 3
lam, mu = 0.5, 2.0
rng_q = np.random.RandomState(42)
a_mat = rng_q.uniform(0, 1, (d_q, M_q))

n_vars = d_q * M_q  # 24
Q_mat  = np.zeros((n_vars, n_vars))

for k in range(d_q):
    for m in range(M_q):
        i = k * M_q + m
        Q_mat[i, i] += -a_mat[k, m] + mu * (1 - 2 * Q_q)
        for k2 in range(k + 1, d_q):
            j = k2 * M_q + m
            Q_mat[i, j] += 2 * mu
        for m2 in range(m + 1, M_q):
            j = k * M_q + m2
            Q_mat[i, j] += lam

# Symétrise pour la visualisation
Q_sym = Q_mat + Q_mat.T - np.diag(np.diag(Q_mat))

print(f"  Diagonale : min={np.diag(Q_sym).min():.3f}, max={np.diag(Q_sym).max():.3f}")
print(f"  Hors-diag : min={Q_sym[Q_sym != np.diag(Q_sym)].min():.3f}, "
      f"max={Q_sym[Q_sym != np.diag(Q_sym)].max():.3f}")
print(f"  Termes size-constraint (off-diag same-kernel, =2*mu={2*mu:.1f}) "
      f": {(Q_sym == 2*mu).sum()//2} paires")
print(f"  Termes diversité (same-feature, ={lam:.1f}) "
      f": {(Q_sym == lam).sum()//2} paires")

fig, ax = plt.subplots(figsize=(7, 6))
vmax = max(abs(Q_sym.min()), abs(Q_sym.max()))
im = ax.imshow(Q_sym, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
fig.colorbar(im, ax=ax, label='Valeur $Q_{ij}$')

# Frontières noyaux (tous les M_q variables)
for pos in range(M_q, n_vars, M_q):
    ax.axhline(pos - 0.5, color='k', lw=0.8, alpha=0.5)
    ax.axvline(pos - 0.5, color='k', lw=0.8, alpha=0.5)

ticks = [k * M_q + M_q // 2 for k in range(d_q)]
labels_k = [f'$k={k}$' for k in range(d_q)]
ax.set_xticks(ticks); ax.set_xticklabels(labels_k, fontsize=8)
ax.set_yticks(ticks); ax.set_yticklabels(labels_k, fontsize=8)
ax.set_xlabel('Indice variable $i = k \\cdot M + m$')
ax.set_ylabel('Indice variable $j = k \\cdot M + m$')
ax.set_title(
    f'Matrice QUBO symétrisée — $d={d_q}$, $M={M_q}$, $Q={Q_q}$, '
    f'$\\lambda={lam}$, $\\mu={mu}$',
    fontsize=11)

# Annotation des blocs
ax.text(0.5, -0.12, 'Séparateurs = frontières features ($k$)',
        transform=ax.transAxes, ha='center', fontsize=8.5, color='gray')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_qubo_matrix.pdf"))
plt.close(fig)
print("  → fig_qubo_matrix.pdf sauvegardé")

# ─────────────────────────────────────────────────────────────
# FIGURE 4 — Convergence Hardware IBM Torino
# ─────────────────────────────────────────────────────────────
print("\n=== Figure 4 : Convergence IBM Torino ===")

json_path = os.path.join(
    os.path.dirname(__file__),
    "../results/qubo_solutions/qaoa_hw_d12_M3_Q4.json"
)
with open(json_path) as f:
    hw = json.load(f)

ev_hist = np.array(hw["ev_history"])
best_ev = hw["best_ev"]
n_iter  = len(ev_hist)
iters   = np.arange(n_iter)

# Running minimum
running_min = np.minimum.accumulate(ev_hist)

print(f"  {n_iter} valeurs EV : min={ev_hist.min():.4f}, max={ev_hist.max():.4f}")
print(f"  best_ev={best_ev:.4f}  (iter {np.argmin(ev_hist)})")
print(f"  Outlier positif à iter {np.argmax(ev_hist)} : EV={ev_hist.max():.4f}"
      f"  (bruit hardware)")

outlier_mask = ev_hist > 0
normal_mask  = ~outlier_mask

fig, ax = plt.subplots(figsize=(9, 4.5))

ax.plot(iters[normal_mask], ev_hist[normal_mask], 'o',
        color='#3498db', ms=5, zorder=3, label=r'$\langle H_C\rangle_\theta$ (IBM Torino)')
ax.plot(iters[normal_mask], ev_hist[normal_mask], '-',
        color='#3498db', lw=1.2, alpha=0.5, zorder=2)
ax.plot(iters[outlier_mask], ev_hist[outlier_mask], 'D',
        color='#e74c3c', ms=10, zorder=5,
        label=f'Erreur hardware (iter\u00a0{int(np.argmax(ev_hist))}, '
              f'EV\u00a0=\u00a0{ev_hist.max():.1f})')
ax.plot(iters, running_min, '--', color='#2ecc71', lw=2, alpha=0.9,
        label='Minimum courant')
ax.axhline(best_ev, color='#2c3e50', ls=':', lw=1.5,
           label=f'$E^* = {best_ev:.2f}$ (iter\u00a0{int(np.argmin(ev_hist))})')

ax.set_xlabel('Itération COBYLA')
ax.set_ylabel(r'$\langle H_C\rangle_\theta$')
ax.set_title(
    'QAOA sur IBM Torino — données expérimentales réelles\n'
    r'$d=12$, $M=3$, $Q=4$ $\Rightarrow$ 36 qubits, backend\,: ibm\_torino (Heron r2)',
    fontsize=11)
ax.set_xticks(iters)
ax.xaxis.set_tick_params(labelsize=8)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

# Annotation quota épuisé
ev_last = ev_hist[normal_mask][-1]
ax.annotate('Quota épuisé\n(22 / 50\u00a0iter.)', xy=(21, ev_last),
            xytext=(17.5, ev_last + 4),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
            fontsize=8.5, color='#555555')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_hw_convergence.pdf"))
plt.close(fig)
print("  → fig_hw_convergence.pdf sauvegardé")

# ─────────────────────────────────────────────────────────────
# FIGURE 5 — AUC comparatif v2 (breast_cancer proxy)
# ─────────────────────────────────────────────────────────────
print("\n=== Figure 5 : AUC comparatif assignations (breast_cancer, Q=4, M=3) ===")

Q_v2, M_v2 = 4, 3
N_v2 = 200

rng5 = np.random.RandomState(42)
idx5 = rng5.choice(len(Xbc), N_v2, replace=False)
X5, y5 = Xbc[idx5], ybc[idx5]

def compute_auc_strategy(X, y, get_assignment_fn, Q, M, n_splits=4, n_reps=5):
    aucs = []
    for rep in range(n_reps):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep * 7)
        for tr, te in kf.split(X, y):
            Xtr_s, Xte_s = quantum_scale(X[tr], X[te])
            assignment = get_assignment_fn(Xtr_s, y[tr], Q, M)
            K_tr = np.zeros((len(tr), len(tr)))
            K_te = np.zeros((len(te), len(tr)))
            for feats in assignment.values():
                Ktr_m = kernel_Z(Xtr_s[:, feats], Xtr_s[:, feats], alpha=ALPHA)
                Kte_m = kernel_Z(Xte_s[:, feats], Xtr_s[:, feats], alpha=ALPHA)
                K_tr += Ktr_m / M
                K_te += Kte_m / M
            clf = SVC(kernel='precomputed', C=1.0, probability=True)
            clf.fit(K_tr, y[tr])
            prob = clf.predict_proba(K_te)[:, 1]
            aucs.append(roc_auc_score(y[te], prob))
    return float(np.mean(aucs)), float(np.std(aucs))

# Stratégie 1 : sous-ensembles aléatoires
def assign_random(X, y, Q, M):
    d = X.shape[1]
    rng_r = np.random.RandomState(np.random.randint(1000))
    assignment = {}
    for m in range(M):
        assignment[m] = list(rng_r.choice(d, Q, replace=False))
    return assignment

# Stratégie 2 : non-chevauchant fixe
def assign_nonoverlap(X, y, Q, M):
    assignment = {}
    for m in range(M):
        assignment[m] = list(range(m * Q, (m + 1) * Q))
    return assignment

# Stratégie 3 : greedy sur alignements marginaux
def align_marginal(K, Ky_c):
    n = K.shape[0]
    one = np.ones((n, n)) / n
    Kc = K - one @ K - K @ one + one @ K @ one
    Ky_c_norm = np.linalg.norm(Ky_c, 'fro')
    if Ky_c_norm < 1e-10:
        return 0.0
    Kc_norm = np.linalg.norm(Kc, 'fro')
    if Kc_norm < 1e-10:
        return 0.0
    return float(np.sum(Kc * Ky_c) / (Kc_norm * Ky_c_norm))

def assign_qubo_greedy(X, y, Q, M):
    d = X.shape[1]
    n = X.shape[0]
    Ky = (y[:, None] == y[None, :]).astype(float)
    one = np.ones((n, n)) / n
    Ky_c = Ky - one @ Ky - Ky @ one + one @ Ky @ one
    # Alignements marginaux a[k, m]
    a = np.zeros((d, M))
    for k in range(d):
        for m in range(M):
            X_sub = X[:, [k] * Q]
            Km = kernel_Z(X_sub, X_sub, alpha=ALPHA)
            a[k, m] = align_marginal(Km, Ky_c)
    # Greedy : pour chaque noyau m, sélectionner Q features avec a[k,m] max
    assignment = {}
    used = set()
    for m in range(M):
        scores = [(a[k, m], k) for k in range(d) if k not in used]
        scores.sort(reverse=True)
        chosen = [k for _, k in scores[:Q]]
        assignment[m] = chosen
        used.update(chosen)
    # Si pas assez de features disponibles, on réutilise
    for m in range(M):
        while len(assignment[m]) < Q:
            assignment[m].append(np.argmax(a[:, m]))
    return assignment

# Calcul AUC pour chaque stratégie
print("  Calcul des AUC (peut prendre ~60s)...")
methods = {
    'Aléatoire': assign_random,
    'Blocs fixes': assign_nonoverlap,   # features 0..Q-1 → K0, Q..2Q-1 → K1, etc.
    'QUBO-Greedy': assign_qubo_greedy,
}
results = {}
for name, fn in methods.items():
    mu_auc, std_auc = compute_auc_strategy(X5, y5, fn, Q_v2, M_v2)
    results[name] = (mu_auc, std_auc)
    print(f"  {name:20s}  AUC = {mu_auc:.4f} ± {std_auc:.4f}")

# RBF-SVM de référence
rbf_aucs = []
for rep in range(5):
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=rep * 7)
    for tr, te in kf.split(X5, y5):
        sc_rbf = StandardScaler()
        Xtr_rbf = sc_rbf.fit_transform(X5[tr])
        Xte_rbf = sc_rbf.transform(X5[te])
        clf_rbf = SVC(kernel='rbf', C=1.0, probability=True)
        clf_rbf.fit(Xtr_rbf, y5[tr])
        prob = clf_rbf.predict_proba(Xte_rbf)[:, 1]
        rbf_aucs.append(roc_auc_score(y5[te], prob))
results['RBF-SVM'] = (float(np.mean(rbf_aucs)), float(np.std(rbf_aucs)))
print(f"  {'RBF-SVM':20s}  AUC = {results['RBF-SVM'][0]:.4f} ± {results['RBF-SVM'][1]:.4f}")

# ─── Tracé Figure 5 ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
names = list(results.keys())
means = [results[n][0] for n in names]
stds  = [results[n][1] for n in names]
colors_bar = [C_Z, '#2ecc71', '#f39c12', C_ZZ]
x = np.arange(len(names))
bars = ax.bar(x, means, yerr=stds, color=colors_bar, width=0.55,
              capsize=5, edgecolor='white', linewidth=0.5, alpha=0.88)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.002,
            f'{m:.3f}', ha='center', va='bottom', fontsize=9.5)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel('AUC ROC')
ax.set_ylim(0.90, 1.03)
ax.set_title(
    f'AUC comparatif — Breast Cancer ($N={N_v2}$, $Q={Q_v2}$, $M={M_v2}$)',
    fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.axhline(results['RBF-SVM'][0], color=C_ZZ, ls=':', lw=1.5, alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_auc_comparison.pdf"))
plt.close(fig)
print("  → fig_auc_comparison.pdf sauvegardé")

# ─────────────────────────────────────────────────────────────
# RÉCAPITULATIF NUMÉRIQUE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VALEURS NUMÉRIQUES VÉRIFIÉES")
print("=" * 60)
print("\nσ_off(K) vs Q :")
print(f"{'Q':>4}  {'σ_off(K_Z)':>12}  {'σ_off(K_ZZ)':>13}  {'AUC K_Z':>10}")
for i, Q in enumerate(Q_vals):
    print(f"{Q:>4}  {sigma_z[i]:>12.5f}  {sigma_zz[i]:>13.5f}  {auc_Q[i]:>10.4f}")
print(f"\nFit K_Z  : σ ~ {pz[0]:.4f}·exp(-{pz[1]:.4f}·Q)")
print(f"Fit K_ZZ : σ ~ {pzz[0]:.4f}·exp(-{pzz[1]:.4f}·Q)")
print(f"\nMémoire statevector (Go) :")
for Q, m in zip(Q_theory[::4], mem_gb[::4]):
    print(f"  Q={Q:2d}  mem={m:.3g} Go")
print(f"\nAUC comparatif assignations (Q={Q_v2}, M={M_v2}) :")
for n, (m, s) in results.items():
    print(f"  {n:22s}  {m:.4f} ± {s:.4f}")
print(f"\nHardware IBM Torino : {n_iter} itérations, best_ev={best_ev:.4f}")
print("=" * 60)
print("\nTous les fichiers PDF générés dans :", OUT_DIR)
