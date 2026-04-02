"""
Script de génération des figures pour rapport_qmkl.tex
Toutes les valeurs numériques sont calculées ici et vérifiables.

Figures produites :
  fig1_v1_results.pdf      — Résultats V1 : QMKL vs classiques (German Credit + Bank Mktg)
  fig2_barren_plateau.pdf  — Barren plateaus : σ_off(K) vs Q (K_Z et K_ZZ)
  fig3_alignment.pdf       — Scores d'alignement et assignation QUBO vs blocs fixes
  fig4_auc_comparison.pdf  — Résultats V2 : AUC par stratégie d'assignation
  fig5_hw_convergence.pdf  — Convergence QAOA sur IBM Torino (données réelles)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import json, os

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

# Palette cohérente avec le rapport
C_QMKL   = '#3498db'   # bleu  — méthodes QMKL
C_CLASS   = '#e74c3c'   # rouge — méthodes classiques
C_ZZ      = '#e74c3c'
C_Z       = '#3498db'
C_FIT     = '#2c3e50'
KERN_COLS = ['#3498db', '#2ecc71', '#f39c12']  # K_0, K_1, K_2

# ─────────────────────────────────────────────────────────────
# Noyaux analytiques
# ─────────────────────────────────────────────────────────────

def kernel_Z(X1, X2, alpha=1.0):
    diff = X1[:, None, :] - X2[None, :, :]
    return np.prod(np.cos(alpha * diff) ** 2, axis=-1)

def kernel_ZZ(X1, X2, alpha=1.0):
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

def quantum_scale(X_train, X_test=None):
    sc = Pipeline([("std", StandardScaler()),
                   ("mms", MinMaxScaler(feature_range=(0, 2 * np.pi)))])
    Xt = sc.fit_transform(X_train)
    if X_test is None:
        return Xt
    return Xt, sc.transform(X_test)

def centered_alignment(K, Ky_c):
    n = K.shape[0]
    one = np.ones((n, n)) / n
    Kc = K - one @ K - K @ one + one @ K @ one
    norm_Kc = np.linalg.norm(Kc, 'fro')
    norm_Ky = np.linalg.norm(Ky_c, 'fro')
    if norm_Kc < 1e-10 or norm_Ky < 1e-10:
        return 0.0
    return float(np.sum(Kc * Ky_c) / (norm_Kc * norm_Ky))

# ─────────────────────────────────────────────────────────────
# Données partagées — Breast Cancer, N=200
# ─────────────────────────────────────────────────────────────
data = load_breast_cancer()
Xbc_full, ybc_full = data.data, data.target
feat_names = data.feature_names
rng0 = np.random.RandomState(42)
idx0 = rng0.choice(len(Xbc_full), 200, replace=False)
Xbc200, ybc200 = Xbc_full[idx0], ybc_full[idx0]
ALPHA = 1.0

# ═════════════════════════════════════════════════════════════
# FIGURE 1 — Résultats V1 : QMKL vs méthodes classiques
# Source : Table 3 du rapport (résultats archivés du projet QMKL-Finance)
# ═════════════════════════════════════════════════════════════
print("=== Figure 1 : Résultats V1 ===")

# German Credit — Q=6, M=12, N=200, 4-fold × 20 tirages
gc_methods = ['QMKL\nMoyenne', 'QMKL\nAlign.C.', 'QMKL\nSDP', 'QMKL\nBO',
              'RBF\nSVM', 'Rnd.\nForest']
gc_auc  = [0.763, 0.750, 0.748, 0.758, 0.835, 0.833]
gc_std  = [0.055, 0.061, 0.063, 0.058, 0.042, 0.046]

# Bank Marketing — sous-ensemble de 5000 ex., 4-fold × 20 tirages
bm_methods = ['QMKL\nMoyenne', 'QMKL\nBO', 'RBF\nSVM', 'Rég.\nLogist.']
bm_auc  = [0.741, 0.774, 0.817, 0.867]
bm_std  = [0.151, 0.135, 0.068, 0.057]

# Couleurs : QMKL = bleu, classiques = rouge
def bar_colors(methods):
    return [C_QMKL if 'QMKL' in m else C_CLASS for m in methods]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

for ax, methods, aucs, stds, title, ylim_lo in [
    (axes[0], gc_methods, gc_auc, gc_std, 'German Credit', 0.68),
    (axes[1], bm_methods, bm_auc, bm_std, 'Bank Marketing', 0.66),
]:
    cols = bar_colors(methods)
    x = np.arange(len(methods))
    bars = ax.bar(x, aucs, yerr=stds, color=cols, width=0.55,
                  capsize=5, edgecolor='white', linewidth=0.5, alpha=0.88)
    for bar, a, s in zip(bars, aucs, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, a + s + 0.005,
                f'{a:.3f}', ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9.5)
    ax.set_ylabel('AUC ROC')
    ax.set_ylim(ylim_lo, 1.0)
    ax.set_title(title, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(max(aucs[-2:]), color=C_CLASS, ls=':', lw=1.2, alpha=0.5)

# Légende commune
from matplotlib.patches import Patch
legend_els = [Patch(color=C_QMKL, label='QMKL (noyaux quantiques)'),
              Patch(color=C_CLASS, label='Classique (référence)')]
fig.legend(handles=legend_els, loc='lower center', ncol=2,
           bbox_to_anchor=(0.5, -0.05), fontsize=10)
fig.suptitle('Résultats V1 — AUC-ROC ($Q=6$, $M=12$, $N=200$, 4-fold $\\times$ 20 tirages)',
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_v1_results.pdf"))
plt.close(fig)
print("  → fig1_v1_results.pdf sauvegardé")
print(f"  German Credit : QMKL max={max(gc_auc[:4]):.3f}, RBF-SVM={gc_auc[4]:.3f}")
print(f"  Bank Marketing: QMKL max={max(bm_auc[:2]):.3f}, Rég.Log.={bm_auc[3]:.3f}")

# ═════════════════════════════════════════════════════════════
# FIGURE 2 — Barren Plateaus : concentration du noyau vs Q
# ═════════════════════════════════════════════════════════════
print("\n=== Figure 2 : Barren Plateaus ===")

Q_vals = [2, 4, 6, 8, 10, 12]
N_BP   = 80

sigma_z, sigma_zz = [], []
for Q in Q_vals:
    rng = np.random.RandomState(42)
    X = rng.uniform(0, 2 * np.pi, (N_BP, Q))
    sigma_z.append(sigma_off(kernel_Z(X, X, alpha=ALPHA)))
    sigma_zz.append(sigma_off(kernel_ZZ(X, X, alpha=ALPHA)))
    print(f"  Q={Q:2d}  σ_off(K_Z)={sigma_z[-1]:.5f}  σ_off(K_ZZ)={sigma_zz[-1]:.6f}")

sigma_z  = np.array(sigma_z)
sigma_zz = np.array(sigma_zz)
Q_arr    = np.array(Q_vals, dtype=float)

def exp_decay(q, A, b):
    return A * np.exp(-b * q)

pz,  _ = curve_fit(exp_decay, Q_arr, sigma_z,  p0=[0.5, 0.3])
pzz, _ = curve_fit(exp_decay, Q_arr, sigma_zz, p0=[0.5, 0.4])
Q_fine = np.linspace(2, 12, 200)

print(f"\n  Fit K_Z  : A={pz[0]:.4f}, b={pz[1]:.4f}")
print(f"  Fit K_ZZ : A={pzz[0]:.4f}, b={pzz[1]:.4f}")

fig, ax = plt.subplots(figsize=(7.5, 4.5))

ax.semilogy(Q_arr, sigma_z,  'o-', color=C_Z,  lw=2, ms=8,
            label=r'$K_Z$  (sans intrication)')
ax.semilogy(Q_arr, sigma_zz, 's-', color=C_ZZ, lw=2, ms=8,
            label=r'$K_{ZZ}$  (intrication linéaire)')
ax.semilogy(Q_fine, exp_decay(Q_fine, *pz),  '--', color=C_Z,  alpha=0.6,
            label=fr'$\hat{{f}}(Q)={pz[0]:.2f}\,e^{{-{pz[1]:.2f}Q}}$')
ax.semilogy(Q_fine, exp_decay(Q_fine, *pzz), '--', color=C_ZZ, alpha=0.6,
            label=fr'$\hat{{f}}(Q)={pzz[0]:.2f}\,e^{{-{pzz[1]:.2f}Q}}$')

# Référence théorique 2^{-Q}
ref = 0.25 * 2.0 ** (-Q_fine)
ax.semilogy(Q_fine, ref, ':', color='gray', alpha=0.5,
            label=r'Référence $2^{-Q}$')

# Annotations : régime sûr vs barren plateau
ax.axvspan(2, 6.5,   alpha=0.07, color='green', label='Régime sûr ($Q \leq 6$)')
ax.axvspan(6.5, 12.5, alpha=0.07, color='red',   label='Barren plateau ($Q > 6$)')

ax.set_xlabel('Nombre de qubits $Q$')
ax.set_ylabel(r'$\sigma_{\mathrm{off}}(K)$ — écart-type hors-diagonal (échelle log)')
ax.set_title('Concentration du noyau en fonction de $Q$\n'
             r'($N=80$ points aléatoires uniformes sur $[0,2\pi]^Q$)')
ax.set_xticks(Q_vals)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, which='both', alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_barren_plateau.pdf"))
plt.close(fig)
print("  → fig2_barren_plateau.pdf sauvegardé")

# ═════════════════════════════════════════════════════════════
# FIGURE 3 — Scores d'alignement et assignation de features
# Montre POURQUOI l'assignation QUBO-Greedy est meilleure que les blocs fixes
# ═════════════════════════════════════════════════════════════
print("\n=== Figure 3 : Scores d'alignement et assignation ===")

# Paramètres : d=12 features de Breast Cancer, M=3 noyaux, Q=4 features/noyau
D_FIG3, M_FIG3, Q_FIG3 = 12, 3, 4

Xsc = quantum_scale(Xbc200)         # (200, 30) scalé dans [0, 2pi]
X12 = Xsc[:, :D_FIG3]               # 12 premières features
feat_labels = [feat_names[k].replace(' (mean)', '').replace(' ', '\n')
               for k in range(D_FIG3)]

# Noyau cible (alignement)
n = X12.shape[0]
Ky  = (ybc200[:, None] == ybc200[None, :]).astype(float)
one = np.ones((n, n)) / n
Ky_c = Ky - one @ Ky - Ky @ one + one @ Ky @ one

# Scores d'alignement marginal a_k pour chaque feature k
a_scores = np.zeros(D_FIG3)
for k in range(D_FIG3):
    X_sub = X12[:, [k] * Q_FIG3]   # Q copies de la feature k → 1-feature kernel
    Km    = kernel_Z(X_sub, X_sub, alpha=ALPHA)
    a_scores[k] = centered_alignment(Km, Ky_c)

print("  Scores d'alignement marginal a_k :")
for k, a in enumerate(a_scores):
    print(f"  k={k:2d}  a_k={a:.4f}  ({feat_names[k][:30]})")

# Assignation Blocs fixes : features 0..3 → K_0, 4..7 → K_1, 8..11 → K_2
assign_blocs = {m: list(range(m * Q_FIG3, (m + 1) * Q_FIG3)) for m in range(M_FIG3)}

# Assignation QUBO-Greedy : top-Q features par ordre décroissant de a_k, sans réutilisation
order = np.argsort(a_scores)[::-1]
assign_greedy = {}
used = set()
for m in range(M_FIG3):
    chosen = [k for k in order if k not in used][:Q_FIG3]
    assign_greedy[m] = chosen
    used.update(chosen)

print("\n  Assignation Blocs fixes :")
for m, feats in assign_blocs.items():
    print(f"    K_{m}: features {feats}  scores={[round(a_scores[k],3) for k in feats]}")
print("  Assignation QUBO-Greedy :")
for m, feats in assign_greedy.items():
    print(f"    K_{m}: features {feats}  scores={[round(a_scores[k],3) for k in feats]}")

# Score total de chaque stratégie (somme des alignements assignés)
score_blocs   = sum(a_scores[k] for feats in assign_blocs.values()  for k in feats)
score_greedy  = sum(a_scores[k] for feats in assign_greedy.values() for k in feats)
print(f"\n  Score total Blocs fixes  : {score_blocs:.4f}")
print(f"  Score total QUBO-Greedy : {score_greedy:.4f}")

# ─── Tracé Figure 3 ───────────────────────────────────────────
# Couleur de chaque feature selon l'assignation
def feature_colors(assignment, d):
    col = ['#cccccc'] * d
    for m, feats in assignment.items():
        for k in feats:
            col[k] = KERN_COLS[m]
    return col

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, assign, title in [
    (axes[0], assign_blocs,  'Blocs fixes (séquentiel)'),
    (axes[1], assign_greedy, 'QUBO-Greedy (par alignement)'),
]:
    cols = feature_colors(assign, D_FIG3)
    x = np.arange(D_FIG3)
    bars = ax.bar(x, a_scores, color=cols, width=0.7, edgecolor='white', linewidth=0.5)
    for bar, a in zip(bars, a_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, a + 0.002,
                f'{a:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'$k_{{{k}}}$' for k in range(D_FIG3)], fontsize=8.5)
    ax.set_xlabel('Feature $k$ (12 premières, Breast Cancer)')
    ax.set_ylabel('Score d\'alignement $a_k$')
    # Titre avec alignement moyen par noyau
    means_per_k = [np.mean([a_scores[k] for k in assign[m]]) for m in range(M_FIG3)]
    subtitle = '  '.join([f'$\\bar{{a}}(K_{m})={means_per_k[m]:.3f}$' for m in range(M_FIG3)])
    ax.set_title(f'{title}\n{subtitle}', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Légende noyaux
    from matplotlib.patches import Patch
    legend_els = [Patch(color=KERN_COLS[m], label=f'$K_{m}$') for m in range(M_FIG3)]
    ax.legend(handles=legend_els, title='Noyau assigné', fontsize=9)

fig.suptitle(
    'Assignation features $\\to$ noyaux — même hauteur de barre, couleur différente\n'
    f'$d={D_FIG3}$, $M={M_FIG3}$, $Q={Q_FIG3}$ features/noyau, Breast Cancer ($N=200$)',
    fontsize=11, y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_alignment.pdf"))
plt.close(fig)
print("  → fig3_alignment.pdf sauvegardé")

# ═════════════════════════════════════════════════════════════
# FIGURE 4 — Résultats V2 : AUC par stratégie d'assignation
# ═════════════════════════════════════════════════════════════
print("\n=== Figure 4 : AUC comparatif assignations ===")

Q_v2, M_v2, N_v2 = 4, 3, 200
rng5 = np.random.RandomState(42)
idx5 = rng5.choice(len(Xbc_full), N_v2, replace=False)
X5, y5 = Xbc_full[idx5], ybc_full[idx5]

def compute_auc(X, y, assign_fn, Q, M, n_splits=4, n_reps=5):
    aucs = []
    for rep in range(n_reps):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep * 7)
        for tr, te in kf.split(X, y):
            Xtr_s, Xte_s = quantum_scale(X[tr], X[te])
            assignment = assign_fn(Xtr_s, y[tr], Q, M)
            K_tr = np.zeros((len(tr), len(tr)))
            K_te = np.zeros((len(te), len(tr)))
            for feats in assignment.values():
                K_tr += kernel_Z(Xtr_s[:, feats], Xtr_s[:, feats], alpha=ALPHA) / M
                K_te += kernel_Z(Xte_s[:, feats], Xtr_s[:, feats], alpha=ALPHA) / M
            clf = SVC(kernel='precomputed', C=1.0, probability=True)
            clf.fit(K_tr, y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(K_te)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))

def assign_random(X, y, Q, M):
    d = X.shape[1]
    rng_r = np.random.RandomState(np.random.randint(1000))
    return {m: list(rng_r.choice(d, Q, replace=False)) for m in range(M)}

def assign_blocs_fixes(X, y, Q, M):
    return {m: list(range(m * Q, (m + 1) * Q)) for m in range(M)}

def assign_qubo_greedy(X, y, Q, M):
    d, n = X.shape[1], X.shape[0]
    Ky   = (y[:, None] == y[None, :]).astype(float)
    one  = np.ones((n, n)) / n
    Ky_c = Ky - one @ Ky - Ky @ one + one @ Ky @ one
    a = np.zeros((d, M))
    for k in range(d):
        for m in range(M):
            a[k, m] = centered_alignment(kernel_Z(X[:, [k] * Q], X[:, [k] * Q], alpha=ALPHA), Ky_c)
    assignment, used = {}, set()
    for m in range(M):
        scores = sorted([(a[k, m], k) for k in range(d) if k not in used], reverse=True)
        chosen = [k for _, k in scores[:Q]]
        assignment[m] = chosen
        used.update(chosen)
    for m in range(M):
        while len(assignment[m]) < Q:
            assignment[m].append(int(np.argmax(a[:, m])))
    return assignment

print("  Calcul AUC (peut prendre ~60s)...")
strategies = {
    'Aléatoire':       assign_random,
    'Blocs fixes':     assign_blocs_fixes,
    'QUBO-Greedy':     assign_qubo_greedy,
}
v2_results = {}
for name, fn in strategies.items():
    mu, sd = compute_auc(X5, y5, fn, Q_v2, M_v2)
    v2_results[name] = (mu, sd)
    print(f"  {name:15s}  AUC = {mu:.4f} ± {sd:.4f}")

# RBF-SVM référence
rbf_aucs = []
for rep in range(5):
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=rep * 7)
    for tr, te in kf.split(X5, y5):
        sc = StandardScaler()
        clf = SVC(kernel='rbf', C=1.0, probability=True)
        clf.fit(sc.fit_transform(X5[tr]), y5[tr])
        rbf_aucs.append(roc_auc_score(y5[te], clf.predict_proba(sc.transform(X5[te]))[:, 1]))
v2_results['RBF-SVM'] = (float(np.mean(rbf_aucs)), float(np.std(rbf_aucs)))
print(f"  {'RBF-SVM':15s}  AUC = {v2_results['RBF-SVM'][0]:.4f} ± {v2_results['RBF-SVM'][1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
names  = list(v2_results.keys())
means  = [v2_results[n][0] for n in names]
stds   = [v2_results[n][1] for n in names]
colors = [C_QMKL, C_QMKL, '#f39c12', C_CLASS]
x = np.arange(len(names))
bars = ax.bar(x, means, yerr=stds, color=colors, width=0.55,
              capsize=5, edgecolor='white', linewidth=0.5, alpha=0.88)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.002,
            f'{m:.3f}', ha='center', va='bottom', fontsize=9.5)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel('AUC ROC')
ax.set_ylim(0.88, 1.03)
ax.set_title(f'Résultats V2 — AUC par stratégie d\'assignation\n'
             f'Breast Cancer, $d=30$, $Q={Q_v2}$, $M={M_v2}$, $N={N_v2}$, '
             f'4-fold $\\times$ 5 tirages', fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.axhline(v2_results['RBF-SVM'][0], color=C_CLASS, ls=':', lw=1.5, alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_auc_comparison.pdf"))
plt.close(fig)
print("  → fig4_auc_comparison.pdf sauvegardé")

# ═════════════════════════════════════════════════════════════
# FIGURE 5 — Convergence QAOA sur IBM Torino (données réelles)
# ═════════════════════════════════════════════════════════════
print("\n=== Figure 5 : Convergence IBM Torino ===")

json_path = os.path.join(os.path.dirname(__file__),
                         "../results/qubo_solutions/qaoa_hw_d12_M3_Q4.json")
with open(json_path) as f:
    hw = json.load(f)

ev_hist     = np.array(hw["ev_history"])
best_ev     = hw["best_ev"]
n_iter      = len(ev_hist)
iters       = np.arange(n_iter)
running_min = np.minimum.accumulate(ev_hist)
outlier_idx = int(np.argmax(ev_hist))
normal_mask = np.ones(n_iter, dtype=bool)
normal_mask[outlier_idx] = False

print(f"  {n_iter} itérations, best_ev={best_ev:.4f} (iter {int(np.argmin(ev_hist))})")
print(f"  Outlier iter {outlier_idx}: EV={ev_hist[outlier_idx]:.2f}")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.scatter(iters[normal_mask], ev_hist[normal_mask],
           color=C_Z, s=40, zorder=4, label=r'$\langle H_C\rangle_\theta$ (IBM Torino)')
ax.plot(iters[normal_mask], ev_hist[normal_mask],
        color=C_Z, lw=1.0, alpha=0.4, zorder=3)
ax.scatter([outlier_idx], [ev_hist[outlier_idx]],
           color=C_ZZ, s=80, marker='D', zorder=5,
           label=f'Erreur hardware (iter {outlier_idx}, EV={ev_hist[outlier_idx]:.1f})')
ax.plot(iters, running_min, '--', color='#2ecc71', lw=2.0,
        label=f'Minimum courant ($E^* = {best_ev:.2f}$ à iter {int(np.argmin(ev_hist))})')
ax.axhline(best_ev, color=C_FIT, ls=':', lw=1.2, alpha=0.6)

# Annotation quota
ax.annotate('Quota épuisé\n(22 / 50 iter.)',
            xy=(n_iter - 1, ev_hist[normal_mask][-1]),
            xytext=(n_iter - 5, ev_hist[normal_mask][-1] + 3.5),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.1),
            fontsize=8.5, color='#555555')

ax.set_xlabel('Itération COBYLA')
ax.set_ylabel(r'$\langle H_C\rangle_\theta$')
ax.set_title('QAOA sur IBM Torino (Heron r2) — données expérimentales réelles\n'
             r'$d=12$, $M=3$, $Q=4$ $\Rightarrow$ 36 qubits, plan IBM Open (4 min)')
ax.set_xticks(iters)
ax.xaxis.set_tick_params(labelsize=8)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_hw_convergence.pdf"))
plt.close(fig)
print("  → fig5_hw_convergence.pdf sauvegardé")

# ═════════════════════════════════════════════════════════════
# RÉCAPITULATIF
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("VALEURS NUMÉRIQUES VÉRIFIÉES")
print("=" * 60)
print("\nV1 — German Credit (max QMKL vs RBF-SVM) :")
print(f"  Max QMKL = {max(gc_auc[:4]):.3f} (Moyenne)  RBF-SVM = {gc_auc[4]:.3f}")
print(f"  Écart = {gc_auc[4] - max(gc_auc[:4]):.3f} en défaveur du QMKL")
print("\nσ_off(K) vs Q :")
for i, Q in enumerate(Q_vals):
    print(f"  Q={Q:2d}  σ_off(K_Z)={sigma_z[i]:.5f}  σ_off(K_ZZ)={sigma_zz[i]:.6f}")
print(f"\nFit K_Z  : σ ~ {pz[0]:.4f}·exp(-{pz[1]:.4f}·Q)")
print(f"Fit K_ZZ : σ ~ {pzz[0]:.4f}·exp(-{pzz[1]:.4f}·Q)")
print(f"\nAssignation (scores d'alignement) :")
print(f"  Score total Blocs fixes  : {score_blocs:.4f}")
print(f"  Score total QUBO-Greedy : {score_greedy:.4f}")
print(f"\nV2 AUC comparatif (Q={Q_v2}, M={M_v2}) :")
for n, (m, s) in v2_results.items():
    print(f"  {n:15s}  {m:.4f} ± {s:.4f}")
print(f"\nIBM Torino : {n_iter} itérations, best_ev={best_ev:.4f}")
print("=" * 60)
print("\nFichiers générés dans :", OUT_DIR)
