# -*- coding: utf-8 -*-
"""
EDA del dataset de vigas (dataset.json)
Genera figuras en notebooks/output/eda/
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter

# ── Configuración visual ────────────────────────────────────────────────────
BG     = "#1a1a2e"
PANEL  = "#16213e"
ACCENT = "#4ade80"   # verde
C2     = "#60a5fa"   # azul
C3     = "#f472b6"   # rosa
C4     = "#fbbf24"   # amarillo
GRAY   = "#6b7280"
WHITE  = "#e5e7eb"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRAY,
    "axes.labelcolor":  WHITE,
    "xtick.color":      WHITE,
    "ytick.color":      WHITE,
    "text.color":       WHITE,
    "grid.color":       GRAY,
    "grid.alpha":       0.3,
    "font.family":      "monospace",
    "axes.titlesize":   11,
    "axes.labelsize":   9,
})

OUT = Path("notebooks/output/eda")
OUT.mkdir(parents=True, exist_ok=True)

# ── Carga ────────────────────────────────────────────────────────────────────
print("Cargando dataset...")
with open("data/dataset.json", "r") as f:
    raw = json.load(f)
N = len(raw)
print(f"  {N:,} registros cargados")

# ── Extracción de features ───────────────────────────────────────────────────
lengths   = np.array([d["inputs"]["length_m"]    for d in raw])
widths    = np.array([d["inputs"]["b_m"]          for d in raw])
heights   = np.array([d["inputs"]["h_m"]          for d in raw])
fcs       = np.array([d["inputs"]["fc_kg_cm2"]    for d in raw])
loads     = np.array([d["inputs"]["q_tonf_m"]     for d in raw])
supports  = [d["inputs"]["support_type"]           for d in raw]

# Momento máximo positivo y mínimo negativo por viga
M_max = np.array([max(d["outputs"]["M_tonf_m"])  for d in raw])
M_min = np.array([min(d["outputs"]["M_tonf_m"])  for d in raw])
M_abs = np.maximum(np.abs(M_max), np.abs(M_min))

# Relación de esbeltez L/h
slenderness = lengths / heights

# ── 1. Resumen estadístico (texto) ───────────────────────────────────────────
print("\n── Estadísticas descriptivas ──────────────────────")
for name, arr in [("length_m", lengths), ("b_m", widths), ("h_m", heights),
                  ("fc kg/cm²", fcs), ("q tonf/m", loads),
                  ("M_max tonf·m", M_max), ("M_min tonf·m", M_min),
                  ("L/h", slenderness)]:
    print(f"  {name:>15}: min={arr.min():.3f}  med={np.median(arr):.3f}"
          f"  mean={arr.mean():.3f}  max={arr.max():.3f}  std={arr.std():.3f}")

cnt = Counter(supports)
print("\n  support_type:", dict(cnt))
print(f"  fc valores únicos: {sorted(np.unique(fcs).tolist())}")

# ════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Distribuciones geométricas
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("EDA — Distribuciones Geométricas del Dataset de Vigas",
             fontsize=13, color=WHITE, y=1.01)

def hist(ax, data, label, color, unit="", bins=40):
    ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor="none")
    ax.axvline(np.mean(data),   color="white", lw=1.5, ls="--", label=f"media={np.mean(data):.2f}")
    ax.axvline(np.median(data), color=ACCENT,  lw=1.5, ls=":",  label=f"mediana={np.median(data):.2f}")
    ax.set_xlabel(f"{label} [{unit}]" if unit else label)
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")

hist(axes[0,0], lengths,     "Longitud (L)",         C2,     "m")
hist(axes[0,1], widths*100,  "Ancho (b)",             ACCENT, "cm")
hist(axes[0,2], heights*100, "Altura (h)",            C3,     "cm")
hist(axes[1,0], slenderness, "Esbeltez (L/h)",        C4)
hist(axes[1,1], loads,       "Carga dist. (q)",       C2,     "tonf/m")

# fc — discreta: barplot
ax = axes[1,2]
fc_vals = sorted(np.unique(fcs).tolist())
fc_counts = [np.sum(fcs == v) for v in fc_vals]
bars = ax.bar([str(int(v)) for v in fc_vals], fc_counts,
              color=[ACCENT, C2, C3], edgecolor="none", alpha=0.9)
for bar, cnt_v in zip(bars, fc_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{cnt_v/N*100:.1f}%", ha="center", va="bottom", fontsize=9, color=WHITE)
ax.set_xlabel("f'c [kg/cm²]")
ax.set_ylabel("Frecuencia")
ax.set_title("Resistencia del concreto")
ax.grid(True, axis="y")

plt.tight_layout()
out1 = OUT / "1_distribuciones_geometricas.png"
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nGuardado: {out1}")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Distribución de momentos y cargas
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("EDA — Momentos y Cargas", fontsize=13, color=WHITE, y=1.01)

hist(axes[0,0], M_max, "M_máx positivo",  ACCENT, "tonf·m")
hist(axes[0,1], M_min, "M_mín negativo",  C3,     "tonf·m")
hist(axes[1,0], M_abs, "M_máx absoluto",  C2,     "tonf·m")
hist(axes[1,1], loads, "Carga distribuida (q)", C4, "tonf/m")

plt.tight_layout()
out2 = OUT / "2_momentos_cargas.png"
fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Guardado: {out2}")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — Tipo de apoyo y correlaciones
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 6))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# 3a — Pie de tipos de apoyo
ax_pie = fig.add_subplot(gs[0])
labels_s  = list(cnt.keys())
sizes_s   = list(cnt.values())
colors_s  = [ACCENT, C2, C3]
wedges, texts, autotexts = ax_pie.pie(
    sizes_s, labels=None, autopct="%1.1f%%",
    colors=colors_s, startangle=90,
    wedgeprops=dict(edgecolor=BG, linewidth=2)
)
for at in autotexts:
    at.set_color(WHITE); at.set_fontsize(10)
label_map = {"FF": "Empotrado-Empotrado", "FP": "Empotrado-Apoyado", "SS": "Simplemente Apoyado"}
ax_pie.legend(
    [f"{label_map.get(l, l)} ({s:,})" for l, s in zip(labels_s, sizes_s)],
    loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=9, frameon=False
)
ax_pie.set_title("Tipos de apoyo")

# 3b — Boxplot M_abs por support_type
ax_box = fig.add_subplot(gs[1])
support_order = ["SS", "FP", "FF"]
data_box  = [M_abs[np.array(supports) == s] for s in support_order]
box = ax_box.boxplot(data_box, patch_artist=True, widths=0.5,
                     medianprops=dict(color="white", lw=2),
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(color=GRAY),
                     capprops=dict(color=GRAY),
                     flierprops=dict(marker=".", color=GRAY, alpha=0.3, ms=2))
for patch, color in zip(box["boxes"], [ACCENT, C2, C3]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax_box.set_xticklabels(support_order)
ax_box.set_xlabel("Tipo de apoyo")
ax_box.set_ylabel("|M_máx| [tonf·m]")
ax_box.set_title("Momento máximo absoluto por tipo de apoyo")
ax_box.grid(True, axis="y")

fig.suptitle("EDA — Tipos de Apoyo", fontsize=13, color=WHITE)
plt.tight_layout()
out3 = OUT / "3_apoyos_y_momentos.png"
fig.savefig(out3, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Guardado: {out3}")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# FIGURA 4 — Scatter plots: relaciones entre variables
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("EDA — Relaciones entre Variables", fontsize=13, color=WHITE, y=1.01)

def scatter(ax, x, y, xlabel, ylabel, color=ACCENT, alpha=0.05, s=1):
    ax.scatter(x, y, c=color, alpha=alpha, s=s, linewidths=0)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    # Línea de tendencia
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, p(xs), color="white", lw=1.2, ls="--", alpha=0.8)
    r = np.corrcoef(x, y)[0, 1]
    ax.set_title(f"r = {r:.3f}", fontsize=9)
    ax.grid(True)

scatter(axes[0,0], loads,      M_abs,      "q [tonf/m]",    "|M_máx| [tonf·m]", ACCENT)
scatter(axes[0,1], lengths,    M_abs,      "L [m]",         "|M_máx| [tonf·m]", C2)
scatter(axes[1,0], slenderness,M_abs,      "L/h",           "|M_máx| [tonf·m]", C3)
scatter(axes[1,1], loads,      lengths,    "q [tonf/m]",    "L [m]",            C4)

plt.tight_layout()
out4 = OUT / "4_scatter_relaciones.png"
fig.savefig(out4, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Guardado: {out4}")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# FIGURA 5 — Muestra de diagramas de momento (9 vigas aleatorias)
# ════════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(42)
# 3 vigas por cada tipo de apoyo
sample_indices = []
for stype in ["SS", "FP", "FF"]:
    idxs = [i for i, s in enumerate(supports) if s == stype]
    sample_indices += rng.choice(idxs, size=3, replace=False).tolist()

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle("EDA — Muestra de Diagramas de Momento (9 vigas)", fontsize=13, color=WHITE)

for ax, idx in zip(axes.flat, sample_indices):
    d      = raw[idx]
    x      = np.array(d["outputs"]["x_m"])
    M      = np.array(d["outputs"]["M_tonf_m"])
    stype  = d["inputs"]["support_type"]
    color  = {"SS": ACCENT, "FP": C2, "FF": C3}[stype]

    ax.plot(x, M, color=color, lw=1.5)
    ax.fill_between(x, M, 0, where=M >= 0, alpha=0.25, color=color)
    ax.fill_between(x, M, 0, where=M <  0, alpha=0.25, color=C4)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_title(
        f"id={d['id']}  {stype}  L={d['inputs']['length_m']:.1f}m\n"
        f"q={d['inputs']['q_tonf_m']:.1f}t/m  f'c={int(d['inputs']['fc_kg_cm2'])}",
        fontsize=8
    )
    ax.set_xlabel("x [m]", fontsize=7)
    ax.set_ylabel("M [tonf·m]", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True)

plt.tight_layout()
out5 = OUT / "5_muestra_diagramas_momento.png"
fig.savefig(out5, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Guardado: {out5}")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# FIGURA 6 — Heatmap: M_abs promedio por (fc, support_type)
# ════════════════════════════════════════════════════════════════════════════
fc_vals_u = sorted(np.unique(fcs).tolist())
sup_vals  = ["SS", "FP", "FF"]
matrix    = np.zeros((len(fc_vals_u), len(sup_vals)))
for i, fc in enumerate(fc_vals_u):
    for j, st in enumerate(sup_vals):
        mask = (fcs == fc) & (np.array(supports) == st)
        matrix[i, j] = M_abs[mask].mean() if mask.sum() > 0 else 0

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(matrix, cmap="YlGn", aspect="auto")
ax.set_xticks(range(len(sup_vals)));  ax.set_xticklabels(sup_vals)
ax.set_yticks(range(len(fc_vals_u))); ax.set_yticklabels([f"{int(v)}" for v in fc_vals_u])
ax.set_xlabel("Tipo de apoyo"); ax.set_ylabel("f'c [kg/cm²]")
ax.set_title("|M_máx| promedio [tonf·m] por f'c y tipo de apoyo")
for i in range(len(fc_vals_u)):
    for j in range(len(sup_vals)):
        ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center",
                fontsize=11, color="black", fontweight="bold")
plt.colorbar(im, ax=ax, label="|M_máx| [tonf·m]")
plt.tight_layout()
out6 = OUT / "6_heatmap_momento_fc_apoyo.png"
fig.savefig(out6, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Guardado: {out6}")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# FIGURA 7 — Matriz de correlación
# ════════════════════════════════════════════════════════════════════════════
support_enc = np.array([{"SS": 0, "FP": 1, "FF": 2}[s] for s in supports])
matrix_data = np.column_stack([lengths, widths*100, heights*100, fcs,
                                loads, slenderness, M_abs, support_enc])
feat_labels = ["L (m)", "b (cm)", "h (cm)", "f'c", "q", "L/h", "|M|", "apoyo"]
corr = np.corrcoef(matrix_data.T)

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(feat_labels))); ax.set_xticklabels(feat_labels, rotation=30, ha="right")
ax.set_yticks(range(len(feat_labels))); ax.set_yticklabels(feat_labels)
ax.set_title("Matriz de correlación de Pearson")
for i in range(len(feat_labels)):
    for j in range(len(feat_labels)):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                fontsize=9, color="white" if abs(corr[i,j]) > 0.5 else "black")
plt.colorbar(im, ax=ax)
plt.tight_layout()
out7 = OUT / "7_correlacion_pearson.png"
fig.savefig(out7, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Guardado: {out7}")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
print("\n── Figuras generadas ──────────────────────────────")
for p in sorted(OUT.glob("*.png")):
    print(f"  {p}")
print("\nEDA completado.")
