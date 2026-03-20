"""
Generates presentation.pdf  — QMKL Finance Beamer-style slides.
Uses reportlab (pure Python).  Run: .venv/Scripts/python compile_slides.py
"""

from reportlab.lib.pagesizes import landscape
from reportlab.lib.units   import cm, mm
from reportlab.lib          import colors
from reportlab.pdfgen       import canvas
from reportlab.lib.utils    import ImageReader
from reportlab.pdfbase      import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
import os, textwrap, io

# ── Page setup ──────────────────────────────────────────────────────────────
W, H = landscape((33.87 * cm, 19.05 * cm))   # 16:9 at 96dpi equivalent

# ── Palette ─────────────────────────────────────────────────────────────────
NAVY      = colors.HexColor("#0D1B2A")
NAVY_L    = colors.HexColor("#142840")
CYAN      = colors.HexColor("#00B4D8")
AMBER     = colors.HexColor("#F4A261")
GREEN     = colors.HexColor("#52B788")
RED       = colors.HexColor("#E63946")
GOLD      = colors.HexColor("#FFD700")
GREY      = colors.HexColor("#8ECAE6")
WHITE     = colors.white
PURPLE    = colors.HexColor("#C77DFF")

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Canvas wrapper ───────────────────────────────────────────────────────────
class Slide:
    def __init__(self, c: canvas.Canvas, n: int, total: int):
        self.c = c
        self.n = n
        self.total = total
        self._draw_bg()

    def _draw_bg(self):
        self.c.setFillColor(NAVY)
        self.c.rect(0, 0, W, H, fill=1, stroke=0)

    def accent_bar(self, color=CYAN, x=0, w=0.35*cm):
        self.c.setFillColor(color)
        self.c.rect(x, 0, w, H, fill=1, stroke=0)

    def hline(self, y, color=CYAN, x0=1.2*cm, x1=None):
        x1 = x1 or (W - 1.2*cm)
        self.c.setStrokeColor(color)
        self.c.setLineWidth(0.5)
        self.c.line(x0, y, x1, y)

    def rect_fill(self, x, y, w, h, color):
        self.c.setFillColor(color)
        self.c.rect(x, y, w, h, fill=1, stroke=0)

    def text(self, txt, x, y, size=11, color=WHITE, bold=False, align="left", wrap_w=None):
        if not txt:
            return
        self.c.setFillColor(color)
        face = "Helvetica-Bold" if bold else "Helvetica"
        self.c.setFont(face, size)
        if wrap_w and len(txt) * size * 0.55 > wrap_w:
            chars_per_line = max(1, int(wrap_w / (size * 0.55)))
            lines = []
            for raw in txt.split('\n'):
                if raw == "":
                    lines.append("")
                else:
                    lines += textwrap.wrap(raw, chars_per_line) or [""]
            line_h = size * 0.045 * cm + 1
            for i, line in enumerate(lines):
                self._draw_text_line(line, x, y - i * line_h, size, align)
        else:
            for i, line in enumerate(txt.split('\n')):
                line_h = size * 0.045 * cm + 0.5
                self._draw_text_line(line, x, y - i * line_h, size, align)

    def _draw_text_line(self, line, x, y, size, align):
        if align == "center":
            self.c.drawCentredString(x, y, line)
        elif align == "right":
            self.c.drawRightString(x, y, line)
        else:
            self.c.drawString(x, y, line)

    def frametitle(self, title, color=CYAN):
        # Header bar
        self.rect_fill(0, H - 1.35*cm, W, 1.35*cm, NAVY_L)
        self.c.setFillColor(color)
        self.c.rect(0, H - 1.35*cm, 0.35*cm, 1.35*cm, fill=1, stroke=0)
        self.text(title, 0.65*cm, H - 0.82*cm, size=17, bold=True, color=WHITE)
        self.footline()

    def footline(self):
        self.rect_fill(0, 0, W, 0.55*cm, NAVY_L)
        self.text("Quantum Multiple Kernel Learning · Finance · 2026",
                  0.5*cm, 0.17*cm, size=7, color=GREY)
        self.text(f"{self.n} / {self.total}",
                  W - 0.5*cm, 0.17*cm, size=7, color=GREY, align="right")

    def bullet(self, items, x, y, size=10, spacing=0.55*cm, color=WHITE, dot=CYAN):
        for item in items:
            self.c.setFillColor(dot)
            self.c.circle(x + 0.18*cm, y + size * 0.018 * cm, 0.07*cm, fill=1, stroke=0)
            self.text(item, x + 0.38*cm, y, size=size, color=color, wrap_w=W/2 - x - 0.5*cm)
            y -= spacing
        return y

    def block(self, title, body_lines, x, y, w, color=CYAN, body_color=NAVY_L, text_size=10):
        title_h = 0.55*cm
        body_h  = len(body_lines) * (text_size * 0.045 * cm + 0.8) + 0.3*cm
        total_h = title_h + body_h
        # title bar
        self.rect_fill(x, y - total_h + title_h, w, title_h, color)
        self.text(title, x + 0.15*cm, y - total_h + title_h * 0.35, size=10, bold=True, color=NAVY)
        # body
        self.rect_fill(x, y - total_h, w, body_h, body_color)
        ty = y - total_h + body_h - 0.22*cm
        for line in body_lines:
            self.text(line, x + 0.15*cm, ty, size=text_size, color=WHITE, wrap_w=w - 0.3*cm)
            ty -= text_size * 0.045 * cm + 0.8
        return total_h

    def image(self, path, x, y, w, h, anchor="sw"):
        """Draw image, fitting within box while preserving aspect ratio."""
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            self.rect_fill(x, y, w, h, NAVY_L)
            self.text(f"[{os.path.basename(path)}]", x + 0.2*cm, y + h/2, size=9, color=GREY)
            return
        try:
            img   = PILImage.open(full)
            iw, ih = img.size
            ratio  = iw / ih
            if w / h > ratio:
                nw, nh = h * ratio, h
            else:
                nw, nh = w, w / ratio
            ox = x + (w - nw) / 2
            oy = y + (h - nh) / 2
            self.c.drawImage(full, ox, oy, nw, nh, preserveAspectRatio=True)
        except Exception as e:
            self.rect_fill(x, y, w, h, NAVY_L)
            self.text(str(e)[:50], x + 0.1*cm, y + h/2, size=7, color=GREY)

    def tag(self, text, x, y, color=CYAN, tcolor=NAVY):
        w = len(text) * 0.17 * cm + 0.35*cm
        h = 0.40*cm
        self.rect_fill(x, y - h, w, h, color)
        self.text(text, x + 0.12*cm, y - h + 0.10*cm, size=9, bold=True, color=tcolor)


# ── Slide builder ────────────────────────────────────────────────────────────
TOTAL = 13

def new_slide(c, n):
    if n > 1:
        c.showPage()
    return Slide(c, n, TOTAL)


def slide_title(c):
    s = new_slide(c, 1)
    s.rect_fill(0, 0, 0.35*cm, H, CYAN)
    # Decorative circles
    c.setStrokeColor(colors.HexColor("#00B4D855"))
    c.setLineWidth(0.6)
    c.circle(W * 0.72, H * 0.55, 3.5*cm, stroke=1, fill=0)
    c.circle(W * 0.88, H * 0.25, 2.0*cm, stroke=1, fill=0)
    s.footline()

    s.text("QUANTUM MACHINE LEARNING · FINANCE · ÉMULATION STATEVECTOR",
           0.7*cm, H - 1.4*cm, size=8, bold=True, color=CYAN)
    s.text("Quantum Multiple\nKernel Learning",
           0.7*cm, H - 2.5*cm, size=36, bold=True, color=WHITE)
    s.text("pour la Classification Financière",
           0.7*cm, H - 4.7*cm, size=20, color=AMBER)
    s.hline(H - 5.1*cm, CYAN, 0.7*cm, W * 0.72)
    s.text("17 notebooks  ·  12 kernels quantiques  ·  3 datasets financiers  ·  6 qubits simulés",
           0.7*cm, H - 5.5*cm, size=11, color=GREY)
    s.text("Mars 2026", 0.7*cm, H - 6.1*cm, size=11, color=GREY)


def slide_context(c):
    s = new_slide(c, 2)
    s.frametitle("Contexte et question centrale")
    y0 = H - 1.9*cm
    lw = W * 0.52

    s.text("Une question ouverte en QML", 0.65*cm, y0, size=12, bold=True, color=CYAN)
    s.text("Les kernels quantiques mesurent la similarité entre deux points de données",
           0.65*cm, y0 - 0.55*cm, size=10, color=WHITE)
    s.text("via la fidélité entre états quantiques encodés sur Q qubits :",
           0.65*cm, y0 - 1.0*cm, size=10, color=WHITE)

    # Formula box
    s.rect_fill(0.65*cm, y0 - 2.4*cm, lw - 0.65*cm - 0.3*cm, 1.15*cm, NAVY_L)
    s.text("K(x, x') = |<ψ(x')|ψ(x)>|²",
           lw/2, y0 - 1.85*cm, size=13, bold=True, color=GOLD, align="center")

    s.text("Multiple Kernel Learning (MKL)", 0.65*cm, y0 - 2.8*cm, size=12, bold=True, color=CYAN)
    s.text("Combiner M kernels quantiques distincts avec des poids appris :",
           0.65*cm, y0 - 3.35*cm, size=10, color=WHITE)
    s.rect_fill(0.65*cm, y0 - 4.55*cm, lw - 0.65*cm - 0.3*cm, 0.95*cm, NAVY_L)
    s.text("K_w(x,x') = Σ wm · Km(x,x')",
           lw/2, y0 - 4.1*cm, size=13, bold=True, color=GOLD, align="center")

    # Right block
    rx = lw + 0.1*cm
    rw = W - rx - 0.5*cm
    s.block("La question centrale",
            ["Les kernels quantiques apportent-ils un avantage mesurable",
             "sur des données financières réelles ?"],
            rx, y0 + 0.1*cm, rw, color=CYAN)

    s.text("Notre démarche :", rx, y0 - 2.3*cm, size=11, bold=True, color=GREY)
    s.bullet(["Émulation exacte par simulation statevector",
              "Benchmark honnête contre les méthodes classiques",
              "Diagnostic des causes de succès / échec",
              "Contributions algorithmiques originales"],
             rx, y0 - 2.9*cm, size=10, spacing=0.52*cm, color=WHITE)


def slide_setup(c):
    s = new_slide(c, 3)
    s.frametitle("Démarche expérimentale", color=AMBER)

    col_w = (W - 1.3*cm) / 3
    cols  = [
        ("3 Datasets financiers", CYAN, [
            "German Credit",
            "1000 clients · 20 features → 6 (PCA)",
            "30% mauvais payeurs",
            "",
            "Bank Marketing",
            "45k appels · 16 features → 6 (PCA)",
            "12% souscriptions (déséquilibre fort)",
            "",
            "Breast Cancer",
            "569 patients · 30 features → 6 (PCA)",
            "Référence ML classique",
        ]),
        ("12 Kernels quantiques", AMBER, [
            "6 familles Pauli :",
            "Z, ZZ, XZ, YXX, YZX, Pauli",
            "",
            "2 bandwidths par famille :",
            "α ∈ {0.5–0.6,  2.5–4.0}",
            "",
            "Simulation statevector exacte",
            "6 qubits  →  2^6 = 64 dimensions",
            "Gram matrix NxN (N=200)",
        ]),
        ("4 Stratégies MKL", GREEN, [
            "Average  —  wm = 1/M",
            "",
            "Centered Alignment",
            "Solution analytique sur KTA",
            "",
            "Bayesian Optimisation",
            "gp_minimize, 25 appels",
            "",
            "Protocole : 20 runs",
            "Split 67/33 stratifié",
        ]),
    ]

    y_top = H - 1.7*cm
    for i, (title, color, lines) in enumerate(cols):
        x = 0.65*cm + i * col_w
        # Column header
        s.rect_fill(x, y_top - 0.5*cm, col_w - 0.15*cm, 0.5*cm, color)
        s.text(title, x + 0.15*cm, y_top - 0.36*cm, size=11, bold=True, color=NAVY)
        # Column body
        s.rect_fill(x, y_top - 5.9*cm, col_w - 0.15*cm, 5.4*cm, NAVY_L)
        s.rect_fill(x, y_top - 5.9*cm, 0.15*cm, 5.4*cm, color)
        ty = y_top - 1.2*cm
        for line in lines:
            if line == "":
                ty -= 0.25*cm
                continue
            bold = line.endswith(":") or (i == 0 and line in ["German Credit","Bank Marketing","Breast Cancer"])
            tc = color if bold else WHITE
            s.text(line, x + 0.35*cm, ty, size=9, bold=bold, color=tc)
            ty -= 0.38*cm

    # Pipeline strip
    s.rect_fill(0.65*cm, 0.6*cm, W - 1.3*cm, 0.68*cm, NAVY_L)
    steps = ["Données brutes", "→", "PCA (6 dims)", "→", "QuantumScaler [0,2π]",
             "→", "PauliFeatureMap", "→", "Gram matrix", "→", "SVM précalculé"]
    tw = (W - 1.3*cm) / len(steps)
    for i, step in enumerate(steps):
        col = CYAN if step != "→" else GREY
        s.text(step, 0.65*cm + i * tw + tw/2, 0.85*cm, size=8,
               color=col, bold=(step != "→"), align="center")


def slide_results(c):
    s = new_slide(c, 4)
    s.frametitle("Résultat principal : QMKL vs méthodes classiques", color=RED)

    s.image("results/11/11_F2_qmkl_vs_classical.png",
            0.5*cm, 0.7*cm, W * 0.52, H - 2.2*cm)

    rx = W * 0.54
    rw = W - rx - 0.4*cm
    y  = H - 2.1*cm

    # Table
    rows = [
        ("[Q] Average",      "0.763", "0.995", CYAN,    False),
        ("[Q] Centered",     "0.750", "0.995", CYAN,    False),
        ("[Q] Bay. Opt.",    "0.758", "0.995", CYAN,    False),
        ("[C] RBF-SVM",      "0.835", "0.996", GREEN,   True),
        ("[C] Rnd Forest",   "0.833", "0.992", AMBER,   False),
    ]
    rh = 0.55*cm
    # Header
    s.rect_fill(rx, y, rw, rh, CYAN)
    s.text("Méthode",         rx + 0.1*cm,        y + 0.14*cm, size=9, bold=True, color=NAVY)
    s.text("German Cr.",      rx + rw * 0.58,      y + 0.14*cm, size=9, bold=True, color=NAVY)
    s.text("Breast C.",       rx + rw * 0.80,      y + 0.14*cm, size=9, bold=True, color=NAVY)
    y -= rh
    for label, gc, bc, tc, hl in rows:
        bg_c = NAVY_L if not hl else colors.HexColor("#0A3D2E")
        s.rect_fill(rx, y, rw, rh, bg_c)
        s.text(label, rx + 0.1*cm, y + 0.14*cm, size=9, color=tc, bold=hl)
        gcv, bcv = float(gc), float(bc)
        gc_c = GREEN if gcv > 0.83 else (WHITE if gcv > 0.77 else RED)
        bc_c = GREEN if bcv > 0.99 else WHITE
        if hl:
            gc_c = bc_c = GOLD
        s.text(gc, rx + rw * 0.61, y + 0.14*cm, size=9, bold=hl, color=gc_c)
        s.text(bc, rx + rw * 0.83, y + 0.14*cm, size=9, bold=hl, color=bc_c)
        y -= rh

    y -= 0.3*cm
    s.text("Écart meilleur QMKL − RBF-SVM :", rx, y, size=10, bold=True, color=WHITE)
    y -= 0.55*cm
    items = [
        ("German Credit",  "−7 pts AUC",       RED),
        ("Bank Marketing", "−4 pts AUC",        AMBER),
        ("Breast Cancer",  "−0.1 pt (non sig.)",GREEN),
    ]
    for ds, gap, col in items:
        s.rect_fill(rx, y - 0.5*cm, rw, 0.5*cm, NAVY_L)
        s.text(ds,  rx + 0.1*cm,      y - 0.33*cm, size=9,  color=GREY)
        s.text(gap, rx + rw * 0.48,   y - 0.33*cm, size=10, bold=True, color=col)
        y -= 0.56*cm


def slide_barren(c):
    s = new_slide(c, 5)
    s.frametitle("Les barren plateaux — première carte 2D (Q × α)", color=AMBER)

    s.image("results/13/13_T1_barren_plateau_map.png",
            0.4*cm, 0.7*cm, W * 0.60, H - 2.1*cm)

    rx  = W * 0.63
    rw  = W - rx - 0.4*cm
    y   = H - 2.1*cm

    s.text("Qu'est-ce qu'un barren plateau ?",
           rx, y, size=11, bold=True, color=AMBER)
    y -= 0.5*cm
    s.text("La matrice de Gram K tend vers l'identité quand Q augmente :",
           rx, y, size=9, color=WHITE, wrap_w=rw)
    y -= 0.45*cm
    s.text("toutes les paires semblent également similaires,",
           rx, y, size=9, color=WHITE)
    y -= 0.42*cm
    s.text("le kernel perd sa capacité discriminante.",
           rx, y, size=9, color=WHITE)
    y -= 0.65*cm

    s.text("Métriques mesurées :", rx, y, size=10, bold=True, color=CYAN)
    y -= 0.48*cm
    s.bullet(["Concentration : σ des valeurs hors-diagonale",
              "Expressivité : entropie spectrale H/H_max",
              "Performance : AUC SVM"],
             rx, y, size=9, spacing=0.45*cm)
    y -= 1.7*cm

    # Alert box
    s.rect_fill(rx, y - 1.1*cm, rw, 1.1*cm, colors.HexColor("#3D1A1A"))
    s.rect_fill(rx, y - 1.1*cm, rw, 0.40*cm, RED)
    s.text("Sweet spot", rx + 0.15*cm, y - 1.1*cm + 0.14*cm, size=10, bold=True, color=NAVY)
    s.text("Q=5, α=2.0  →  AUC = 0.83", rx + 0.15*cm, y - 0.7*cm, size=10, color=WHITE, bold=True)
    s.text("Concentration = 0.098", rx + 0.15*cm, y - 1.05*cm, size=9, color=GREY)
    y -= 1.35*cm

    s.hline(y, GREY, rx)
    y -= 0.35*cm
    s.text("La concentration chute de 60% entre Q=2 et Q=8.",
           rx, y, size=9, color=GREY, wrap_w=rw)
    y -= 0.42*cm
    s.text("Première carte 2D de ce type pour QMKL finance.",
           rx, y, size=9, color=GREY, wrap_w=rw)


def slide_hilbert(c):
    s = new_slide(c, 6)
    s.frametitle("Espace de Hilbert quantique vs espace gaussien (RBF)")

    s.image("results/12/12_F1_quantum_rbf_alignment.png",
            0.4*cm, 0.7*cm, W * 0.56, H - 2.1*cm)

    rx = W * 0.59
    rw = W - rx - 0.4*cm
    y  = H - 2.1*cm

    s.text("Alignement de Frobenius", rx, y, size=12, bold=True, color=CYAN)
    y -= 0.55*cm
    s.text("On mesure dans quelle mesure chaque kernel", rx, y, size=9, color=WHITE, wrap_w=rw)
    y -= 0.42*cm
    s.text("quantique K_q reproduit la structure du RBF :", rx, y, size=9, color=WHITE, wrap_w=rw)
    y -= 0.6*cm
    s.rect_fill(rx, y - 0.55*cm, rw, 0.55*cm, NAVY_L)
    s.text("align(Kq, Krbf) = <Kq, Krbf>_F / ||Kq||·||Krbf||",
           rx + 0.15*cm, y - 0.34*cm, size=9, bold=True, color=GOLD)
    y -= 0.85*cm

    s.text("Résultats :", rx, y, size=10, bold=True, color=WHITE)
    y -= 0.5*cm
    s.bullet(["Alignement moyen : 0.642",
              "7/12 kernels distincts du RBF (< 0.70)",
              "Aucun kernel quasi-identique (> 0.85)"],
             rx, y, size=9, spacing=0.48*cm, dot=AMBER)
    y -= 1.9*cm

    s.hline(y, GREY, rx)
    y -= 0.35*cm
    s.text("Les kernels quantiques ne sont pas de simples",
           rx, y, size=9, color=GREY, wrap_w=rw)
    y -= 0.42*cm
    s.text("surrogates du RBF — mais cette distinctivité",
           rx, y, size=9, color=GREY, wrap_w=rw)
    y -= 0.42*cm
    s.text("ne suffit pas à améliorer les performances",
           rx, y, size=9, color=GREY, wrap_w=rw)
    y -= 0.42*cm
    s.text("sur données tabulaires.",
           rx, y, size=9, color=GREY, wrap_w=rw)


def slide_spectra(c):
    s = new_slide(c, 7)
    s.frametitle("Richesse spectrale et mitigation de la concentration")

    s.image("results/17/17_A5_spectra.png",
            0.4*cm, 0.7*cm, W * 0.60, H - 2.1*cm)

    rx = W * 0.63
    rw = W - rx - 0.4*cm
    y  = H - 2.1*cm

    s.text("Spectre des valeurs propres de Gram", rx, y, size=11, bold=True, color=CYAN)
    y -= 0.5*cm
    s.text("Spectre plat  =  kernel expressif (riche).", rx, y, size=9, color=WHITE, wrap_w=rw)
    y -= 0.42*cm
    s.text("Spectre décroissant  =  kernel concentré.", rx, y, size=9, color=WHITE, wrap_w=rw)
    y -= 0.65*cm

    s.text("Rang effectif (Q=8, Breast Cancer) :", rx, y, size=10, bold=True, color=WHITE)
    y -= 0.5*cm
    rows = [
        ("Global (baseline)", RED),
        ("Local  p=2",        AMBER),
        ("Multi-échelles",    GREEN),
    ]
    for label, col in rows:
        s.rect_fill(rx, y - 0.4*cm, rw, 0.4*cm, NAVY_L)
        s.rect_fill(rx, y - 0.4*cm, 0.2*cm, 0.4*cm, col)
        s.text(label, rx + 0.35*cm, y - 0.25*cm, size=9, color=col, bold=True)
        y -= 0.46*cm

    y -= 0.4*cm
    # Mitigation block
    s.rect_fill(rx, y - 1.35*cm, rw, 1.35*cm, colors.HexColor("#0A2E1E"))
    s.rect_fill(rx, y - 1.35*cm, rw, 0.40*cm, GREEN)
    s.text("Mitigation (NB17 / arXiv 2602.16097)",
           rx + 0.15*cm, y - 1.35*cm + 0.13*cm, size=9, bold=True, color=NAVY)
    s.text("Patches locaux + multi-échelles", rx + 0.15*cm, y - 0.98*cm, size=9, color=WHITE)
    s.text("préservent la richesse spectrale quand Q↑.", rx + 0.15*cm, y - 1.28*cm, size=9, color=GREY)
    y -= 1.6*cm

    s.text("H/H_max quantique = 0.65  vs  RBF = 0.81",
           rx, y, size=9, color=GREY, wrap_w=rw)


def slide_qkrr(c):
    s = new_slide(c, 8)
    s.frametitle("Résultat positif : Quantum Kernel Ridge Regression", color=GREEN)

    s.image("results/13/13_T2_QKRR_comparison.png",
            0.4*cm, 0.7*cm, W * 0.56, H - 2.1*cm)

    rx = W * 0.59
    rw = W - rx - 0.4*cm
    y  = H - 2.1*cm

    s.text("Alternative analytique au SVM", rx, y, size=12, bold=True, color=GREEN)
    y -= 0.55*cm
    s.text("Au lieu du SVM (problème QP), on utilise la", rx, y, size=9, color=WHITE, wrap_w=rw)
    y -= 0.42*cm
    s.text("Kernel Ridge Regression — solution directe :", rx, y, size=9, color=WHITE, wrap_w=rw)
    y -= 0.62*cm
    s.rect_fill(rx, y - 0.55*cm, rw, 0.55*cm, NAVY_L)
    s.text("α* = (K + λI)⁻¹ y",
           rx + rw/2, y - 0.32*cm, size=13, bold=True, color=GOLD, align="center")
    y -= 0.9*cm

    s.text("Comparaison sur German Credit :", rx, y, size=10, bold=True, color=WHITE)
    y -= 0.5*cm
    data = [("QKRR",     "0.756", GREEN),
            ("QSVM",     "0.686", RED),
            ("RBF-KRR",  "0.774", AMBER)]
    for label, val, col in data:
        s.rect_fill(rx, y - 0.42*cm, rw, 0.42*cm, NAVY_L)
        s.text(label, rx + 0.15*cm, y - 0.27*cm, size=9, color=GREY)
        s.text(val,   rx + rw * 0.55, y - 0.27*cm, size=11, bold=True, color=col)
        if label == "QKRR":
            s.text("+7 pts vs QSVM", rx + rw * 0.72, y - 0.27*cm, size=8, color=GREEN)
        y -= 0.48*cm

    y -= 0.35*cm
    s.hline(y, GREY, rx)
    y -= 0.35*cm
    s.text("La régularisation ridge est mieux adaptée aux", rx, y, size=9, color=GREY, wrap_w=rw)
    y -= 0.42*cm
    s.text("kernels concentrés que la marge dure du SVM.", rx, y, size=9, color=GREY, wrap_w=rw)
    y -= 0.42*cm
    s.text("Résultat robuste sur 20 runs.", rx, y, size=9, color=GREY, wrap_w=rw)


def slide_diversity(c):
    s = new_slide(c, 9)
    s.frametitle("Diversité des kernels et gain MKL")

    s.image("results/14/14_D_kernel_diversity.png",
            0.4*cm, 0.7*cm, W * 0.60, H - 2.1*cm)

    rx = W * 0.63
    rw = W - rx - 0.4*cm
    y  = H - 2.1*cm

    s.text("Pourquoi combiner des kernels ?", rx, y, size=11, bold=True, color=CYAN)
    y -= 0.55*cm
    s.text("Plus deux kernels sont distincts l'un de l'autre", rx, y, size=9, color=WHITE, wrap_w=rw)
    y -= 0.42*cm
    s.text("(alignement croisé faible), plus leur combinaison", rx, y, size=9, color=WHITE, wrap_w=rw)
    y -= 0.42*cm
    s.text("apporte un gain marginal en classification.", rx, y, size=9, color=WHITE, wrap_w=rw)
    y -= 0.75*cm

    # Big stat box
    s.rect_fill(rx, y - 1.1*cm, rw, 1.1*cm, colors.HexColor("#0A2E38"))
    s.rect_fill(rx, y - 1.1*cm, rw, 0.40*cm, CYAN)
    s.text("Corrélation clé", rx + 0.15*cm, y - 1.1*cm + 0.13*cm, size=9, bold=True, color=NAVY)
    s.text("Diversité ↔ gain LOO :", rx + 0.15*cm, y - 0.72*cm, size=10, color=WHITE)
    s.text("r = 0.738", rx + rw/2, y - 1.02*cm, size=18, bold=True, color=GOLD, align="center")
    y -= 1.4*cm

    s.text("En pratique :", rx, y, size=10, bold=True, color=WHITE)
    y -= 0.5*cm
    s.bullet(["ZZ α=4.0 : diversité max → gain max",
              "Z α=1.0 : proche RBF → gain marginal",
              "Sélectionner les kernels les plus distincts"],
             rx, y, size=9, spacing=0.45*cm, dot=CYAN)
    y -= 1.6*cm

    s.hline(y, GREY, rx)
    y -= 0.35*cm
    s.text("Premier résultat quantifié de ce type pour", rx, y, size=9, color=GREY, wrap_w=rw)
    y -= 0.42*cm
    s.text("QMKL finance. Critère de sélection actionnable.", rx, y, size=9, color=GREY, wrap_w=rw)


def slide_phase(c):
    s = new_slide(c, 10)
    s.frametitle("Phase diagram — dans quelles conditions QMKL peut-il battre les classiques ?", color=PURPLE)

    s.image("results/15/15_P_phase_diagram.png",
            0.4*cm, 0.7*cm, W * 0.54, H - 2.1*cm)

    rx = W * 0.57
    rw = W - rx - 0.4*cm
    y  = H - 2.1*cm

    s.text("Phase diagram de l'avantage quantique", rx, y, size=11, bold=True, color=PURPLE)
    y -= 0.55*cm
    s.text("Grille systématique :", rx, y, size=9, bold=True, color=WHITE)
    s.text("séparabilité × niveau de bruit", rx + 3.0*cm, y, size=9, color=GREY)
    y -= 0.45*cm
    s.text("sur données synthétiques (structure financière).", rx, y, size=9, color=WHITE, wrap_w=rw)
    y -= 0.75*cm

    # Big number
    s.rect_fill(rx, y - 1.3*cm, rw, 1.3*cm, colors.HexColor("#3D0A0A"))
    s.rect_fill(rx, y - 1.3*cm, rw, 0.40*cm, RED)
    s.text("Résultat", rx + 0.15*cm, y - 1.3*cm + 0.13*cm, size=9, bold=True, color=NAVY)
    s.text("0 / 20", rx + rw/2, y - 0.85*cm, size=28, bold=True, color=RED, align="center")
    s.text("configurations avec avantage quantique", rx + rw/2, y - 1.22*cm,
           size=8, color=WHITE, align="center")
    y -= 1.6*cm

    s.bullet(["Rouge foncé = écart sévère (−30 pts régime difficile)",
              "Meilleure config : sep=1.5, bruit=0.05 → Δ≈0",
              "Aucune configuration favorable à QMKL"],
             rx, y, size=9, spacing=0.45*cm, dot=RED)
    y -= 1.6*cm

    s.hline(y, GREY, rx)
    y -= 0.35*cm
    s.text("Les données tabulaires n'ont pas la structure", rx, y, size=9, color=GREY, wrap_w=rw)
    y -= 0.42*cm
    s.text("d'entanglement que les PauliFeatureMaps ciblent.", rx, y, size=9, color=GREY, wrap_w=rw)


def slide_discussion(c):
    s = new_slide(c, 11)
    s.frametitle("Discussion — Réflexions sur les méthodes")

    mid = W / 2 - 0.1*cm
    lw  = mid - 0.65*cm - 0.15*cm
    rw2 = W - mid - 0.1*cm - 0.4*cm
    y0  = H - 2.0*cm

    # Left: limites
    s.text("Limites identifiées", 0.65*cm, y0, size=12, bold=True, color=RED)
    issues = [
        ("Barren plateaux",
         "La concentration augmente exponentiellement avec Q. À Q=6, le kernel est déjà\npartiellement dégénéré — 84% des instances deviennent des support vectors."),
        ("Inadéquation structurelle",
         "Les PauliFeatureMaps (reps=1) encodent des corrélations d'ordre 2.\nLes données financières tabulaires n'ont pas cette structure native."),
        ("Simulation vs hardware",
         "La simulation statevector exacte est idéale mais ne reflète pas les contraintes\ndu hardware quantique réel (bruit, connectivité limitée, decoherence)."),
        ("Concept drift",
         "Les poids MKL optimaux varient fortement d'une fenêtre temporelle à l'autre.\nCoût moyen si non réentraîné : −11 pts AUC."),
    ]
    y = y0 - 0.55*cm
    for title, text in issues:
        s.rect_fill(0.65*cm, y - 1.05*cm, lw, 1.05*cm, NAVY_L)
        s.rect_fill(0.65*cm, y - 1.05*cm, 0.2*cm, 1.05*cm, RED)
        s.text(title, 0.95*cm, y - 0.22*cm, size=10, bold=True, color=RED)
        s.text(text,  0.95*cm, y - 0.58*cm, size=8, color=WHITE, wrap_w=lw - 0.35*cm)
        y -= 1.15*cm

    # Right: contributions
    rx = mid + 0.1*cm
    s.text("Ce que cette étude apporte", rx, y0, size=12, bold=True, color=GREEN)
    contribs = [
        ("QKRR",
         "La KRR analytique surpasse le QSVM de +7 pts sur German Credit.\nSolution fermée : α* = (K + λI)⁻¹y — plus rapide et plus robuste."),
        ("Diversité → gain",
         "r = 0.738 entre diversité et gain marginal. Critère de sélection\nactionnable : privilégier les kernels les plus distincts."),
        ("Mitigation concentration",
         "Patches locaux + multi-échelles (NB17) préservent la richesse spectrale.\nLe gain en AUC dépend de l'alignement avec les labels (CKA)."),
        ("Sweet spot",
         "Q=5, α=2.0 maximise l'AUC (0.83) avec concentration contenue (0.098).\nPremière carte 2D (Q, α) pour QMKL finance."),
    ]
    y = y0 - 0.55*cm
    for title, text in contribs:
        s.rect_fill(rx, y - 1.05*cm, rw2, 1.05*cm, NAVY_L)
        s.rect_fill(rx, y - 1.05*cm, 0.2*cm, 1.05*cm, GREEN)
        s.text(title, rx + 0.3*cm, y - 0.22*cm, size=10, bold=True, color=GREEN)
        s.text(text,  rx + 0.3*cm, y - 0.58*cm, size=8, color=WHITE, wrap_w=rw2 - 0.35*cm)
        y -= 1.15*cm

    s.hline(0.65*cm, GREY)
    s.text("Le vrai avantage quantique nécessitera des données intrinsèquement quantiques, "
           "des circuits plus profonds (reps≥2), ou un encodage spécifique à la structure financière.",
           0.65*cm, 0.58*cm, size=8, color=GREY, wrap_w=W - 1.3*cm)


def slide_conclusions(c):
    s = new_slide(c, 12)
    s.frametitle("Conclusions et perspectives")

    # Verdict block
    s.rect_fill(0.65*cm, H - 3.15*cm, W - 1.3*cm, 1.1*cm, colors.HexColor("#3D0A0A"))
    s.rect_fill(0.65*cm, H - 3.15*cm, 0.25*cm, 1.1*cm, RED)
    s.text("Bilan", 0.95*cm, H - 2.2*cm, size=10, bold=True, color=RED)
    s.text("QMKL sous-performe les classiques de −4 à −7 pts AUC sur données financières tabulaires.",
           0.95*cm, H - 2.6*cm, size=10, color=WHITE, wrap_w=W - 1.6*cm)
    s.text("Causes : concentration exponentielle, alignement avec RBF (0.642), structure tabulaire inadaptée.",
           0.95*cm, H - 3.0*cm, size=9, color=GREY, wrap_w=W - 1.6*cm)

    mid = W / 2 + 0.2*cm
    lw  = mid - 0.65*cm - 0.2*cm
    rw2 = W - mid - 0.4*cm
    y0  = H - 3.5*cm

    # Left: positives
    s.text("Résultats positifs", 0.65*cm, y0, size=12, bold=True, color=GREEN)
    rows = [
        ("QKRR",      "+7 pts vs QSVM",   GREEN),
        ("Diversité", "r = 0.738",         CYAN),
        ("VQKL",      "+1.5 pts",          AMBER),
        ("Sweet spot","Q=5, α=2.0",        GOLD),
    ]
    y = y0 - 0.55*cm
    for label, val, col in rows:
        s.rect_fill(0.65*cm, y - 0.42*cm, lw, 0.42*cm, NAVY_L)
        s.text(label, 0.85*cm, y - 0.26*cm, size=10, color=GREY)
        s.text(val, 0.65*cm + lw * 0.55, y - 0.26*cm, size=11, bold=True, color=col)
        y -= 0.48*cm

    # Right: future work
    rx = mid
    s.text("Pistes futures", rx, y0, size=12, bold=True, color=CYAN)
    s.bullet(["Circuits plus profonds (reps = 2–3)",
              "Kernels locaux / multi-échelles (NB17)",
              "Données genuinement quantiques",
              "QKRR comme alternative principale au QSVM",
              "Sélection de kernels par critère de diversité"],
             rx, y0 - 0.55*cm, size=10, spacing=0.50*cm, dot=CYAN)

    # Bottom strip
    s.rect_fill(0.65*cm, 0.65*cm, W - 1.3*cm, 0.55*cm, NAVY_L)
    s.text("17 notebooks  ·  14 contributions originales  ·  "
           "Python 3.14 · Qiskit 2.3.1 · Simulation statevector exacte",
           W/2, 0.83*cm, size=9, color=GREY, align="center")


def slide_end(c):
    s = new_slide(c, 13)
    s.rect_fill(0, 0, 0.35*cm, H, CYAN)
    # circles
    c.setStrokeColor(colors.HexColor("#00B4D855"))
    c.setLineWidth(0.6)
    c.circle(W * 0.82, H * 0.55, 3.0*cm, stroke=1, fill=0)
    c.circle(W * 0.93, H * 0.22, 1.5*cm, stroke=1, fill=0)
    s.footline()

    s.text("QUANTUM MULTIPLE KERNEL LEARNING · FINANCE",
           0.7*cm, H - 1.5*cm, size=9, bold=True, color=CYAN)
    s.text("Merci", 0.7*cm, H - 3.2*cm, size=52, bold=True, color=WHITE)
    s.hline(H - 3.9*cm, CYAN, 0.7*cm, W * 0.7)
    details = [
        "17 notebooks  ·  14 contributions originales",
        "Python 3.14 · Qiskit 2.3.1 · Statevector exact",
        "German Credit · Bank Marketing · Breast Cancer",
        "arXiv 2602.16097 — Mitigation concentration (NB17)",
    ]
    y = H - 4.4*cm
    for d in details:
        s.text("·  " + d, 0.7*cm, y, size=11, color=GREY)
        y -= 0.5*cm


# ── Main ─────────────────────────────────────────────────────────────────────
out = os.path.join(ROOT, "presentation.pdf")
c = canvas.Canvas(out, pagesize=(W, H))
c.setTitle("Quantum Multiple Kernel Learning pour la Finance")
c.setAuthor("QMKL-Finance 2026")

slide_title(c)
slide_context(c)
slide_setup(c)
slide_results(c)
slide_barren(c)
slide_hilbert(c)
slide_spectra(c)
slide_qkrr(c)
slide_diversity(c)
slide_phase(c)
slide_discussion(c)
slide_conclusions(c)
slide_end(c)

c.save()
print(f"Saved -> {out}")
print(f"Slides : {TOTAL}")
