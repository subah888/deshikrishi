"""Publication figures (300 dpi PNG) and table images for the DeshiKrishi paper.

Run from the repo root after the Step 3 notebook has saved its results:
    python research/make_figures.py
Reads research/results/*.csv and data/raw/bbs_wholesale_monthly_clean.csv, writes research/figures/.
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRE, CRI = "Pre-crisis (Jul 2023–Jun 2024)", "Crisis (Jul 2024–Jul 2026)"
COL = {PRE: "#257de0", CRI: "#f47230"}
INK, INK2, GRID = "#191919", "#666666", "#e9e9e9"
MODELS = ["Naive", "Seasonal change", "Random Forest", "Gradient Boosting"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#bbbbbb",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "research" / "results", ROOT / "research" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(OUT)
acc = pd.read_csv(RES / "results_accuracy.csv").rename(columns={"Unnamed: 0": "window", "Unnamed: 1": "model"})
dv = pd.read_csv(RES / "results_decision_value_by_series.csv")
raw = pd.read_csv(ROOT / "data" / "raw" / "bbs_wholesale_monthly_clean.csv", parse_dates=["month"])


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)


def grouped(ax, cats, series, fmt, ylabel, zero=False):
    x = np.arange(len(cats)); w = .38
    for i, (name, vals) in enumerate(series.items()):
        bars = ax.bar(x + (i - .5) * w, vals, w - .03, color=COL[name], label=name.split(" (")[0] + " (" + name.split(" (")[1])
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            y = b.get_height()
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, y), xytext=(0, 2 if y >= 0 else -9),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, cats); ax.set_ylabel(ylabel); style(ax)
    if zero:
        ax.axhline(0, color=INK, lw=.8)


# ---------- Figure 1: prices ----------
crops = ["Rice (Coarse)", "Rice (Medium)", "Masur", "Onion", "Potato"]
names = {"Rice (Coarse)": "Coarse rice", "Rice (Medium)": "Medium rice", "Masur": "Masur (red lentil)",
         "Onion": "Onion", "Potato": "Potato"}
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
for ax, c in zip(axs.flat, crops):
    w = raw[raw.commodity == c].pivot(index="month", columns="town", values="price_tk_per_kg")
    for t in w:
        ax.plot(w.index, w[t].interpolate(limit=2, limit_area="inside"), color="#c9cfd6", lw=.8)
    ax.plot(w.index, w.mean(axis=1), color="#257de0", lw=1.8)
    ax.axvspan(pd.Timestamp("2024-07-01"), w.index.max(), color="#f2a600", alpha=.13, lw=0)
    ax.set_title(names[c], loc="left", fontsize=9, color=INK); ax.set_ylim(bottom=0); style(ax)
    ax.tick_params(labelsize=7)
axs[0, 0].set_ylabel("Tk per kg"); axs[1, 0].set_ylabel("Tk per kg")
leg = axs.flat[5]; leg.axis("off")
leg.plot([], [], color="#257de0", lw=1.8, label="Average of six towns")
leg.plot([], [], color="#c9cfd6", lw=.8, label="Individual town")
leg.fill_between([], [], color="#f2a600", alpha=.25, label="Crisis test window\n(Jul 2024 onward)")
leg.legend(loc="center", frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("Fig1_prices.png", dpi=300, facecolor="white"); plt.close(fig)

# ---------- Figure 2: accuracy ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
p = acc.pivot(index="model", columns="window", values="MAE (Tk/kg)").loc[MODELS]
grouped(a1, ["Naive", "Seasonal", "Random\nForest", "Gradient\nBoosting"], {PRE: p[PRE].values, CRI: p[CRI].values},
        "{:.2f}", "Mean absolute error (Tk/kg)")
a1.set_title("(a) Forecast error (lower is better)", loc="left", fontsize=9, color=INK)
d = acc.pivot(index="model", columns="window", values="Direction correct (%)").loc[MODELS[1:]]
grouped(a2, ["Seasonal", "Random\nForest", "Gradient\nBoosting"], {PRE: d[PRE].values, CRI: d[CRI].values},
        "{:.0f}%", "Direction correct (%)")
a2.axhline(50, color=INK2, lw=.8, ls="--"); a2.text(-0.48, 51, "coin flip", fontsize=7, color=INK2, ha="left")
a2.set_ylim(0, 70); a2.set_title("(b) Up/down direction correct", loc="left", fontsize=9, color=INK)
h, l = a1.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig("Fig2_accuracy.png", dpi=300, facecolor="white"); plt.close(fig)

# ---------- Figure 3: decision value (2% storage; 1% and 5% in Table 2) ----------
M3 = ["Seasonal change", "Random Forest", "Gradient Boosting", "Perfect foresight"]
lab3 = ["Seasonal", "Random Forest", "Gradient Boosting", "Perfect foresight"]
g = dv.groupby(["storage", "window", "model"]).gain_pct.mean()
fig, ax = plt.subplots(figsize=(7.2, 3.2))
grouped(ax, lab3, {PRE: [g[(0.02, PRE, m)] for m in M3], CRI: [g[(0.02, CRI, m)] for m in M3]},
        "{:+.2f}", "Revenue vs. selling now (%)", zero=True)
ax.set_ylim(-1.6, 1.7)
ax.set_title("Storage cost 2% per month (1% and 5%: see Table 2)", loc="left", fontsize=9, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("Fig3_decision_value.png", dpi=300, facecolor="white"); plt.close(fig)

# ---------- Figure 4: by crop, crisis, 2% ----------
M4 = ["Seasonal change", "Random Forest", "Gradient Boosting", "Perfect foresight"]
C4 = {"Seasonal change": "#bfc5cc", "Random Forest": "#00b894", "Gradient Boosting": "#257de0", "Perfect foresight": "#f2a600"}
sub = dv[(dv.window == CRI) & (dv.storage == 0.02)].groupby(["commodity", "model"]).gain_pct.mean().unstack()
fig, ax = plt.subplots(figsize=(7.2, 3.4))
x = np.arange(len(crops)); w = .2
for i, m in enumerate(M4):
    vals = sub.loc[crops, m].values
    bars = ax.bar(x + (i - 1.5) * w, vals, w - .02, color=C4[m], label=m.replace(" change", ""))
    for b, v in zip(bars, vals):
        if abs(v) >= .5:
            ax.annotate(f"{v:+.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2 if v >= 0 else -9),
                        textcoords="offset points", ha="center", fontsize=7, color=INK)
ax.axhline(0, color=INK, lw=.8); ax.set_xticks(x, [names[c] for c in crops]); style(ax)
ax.set_ylabel("Revenue vs. selling now (%)")
ax.set_title("Crisis window (Jul 2024–Jul 2026), storage cost 2% per month", loc="left", fontsize=9, color=INK)
ax.set_ylim(-4.9, 3.3)
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.12))
fig.tight_layout(); fig.savefig("Fig4_by_crop.png", dpi=300, facecolor="white"); plt.close(fig)

# ---------- Table images ----------
def table_png(df, fname, title, colw=None):
    fig, ax = plt.subplots(figsize=(7.2, .26 * (len(df) + 2)))
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, loc="upper center", cellLoc="center", colLoc="center",
                 colWidths=colw)
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.3)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_text_props(weight="bold", color=INK); cell.set_facecolor("#f3f5f7")
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(fname, dpi=300, facecolor="white", bbox_inches="tight"); plt.close(fig)


t1 = acc.copy()
t1["window"] = t1.window.str.split(" ").str[0]
t1["Direction correct (%)"] = t1["Direction correct (%)"].map(lambda v: "n/a" if v == 0 else f"{v:.1f}")
t1 = t1.rename(columns={"window": "Window", "model": "Model"})[["Window", "Model", "MAE (Tk/kg)", "MAPE (%)", "Direction correct (%)"]]
t1.to_csv("Table1_accuracy.csv", index=False)
table_png(t1, "Table1_accuracy.png", "Table 1. Forecast accuracy, rolling-origin evaluation (30 series)",
          [.16, .24, .18, .16, .26])

from scipy.stats import wilcoxon
rows = []
for wnd in (PRE, CRI):
    for c in (0.01, 0.02, 0.05):
        pv = dv[(dv.window == wnd) & (dv.storage == c)].pivot(index="series", columns="model", values="gain_pct")
        for m in ["Seasonal change", "Random Forest", "Gradient Boosting", "Perfect foresight"]:
            v = pv[m]
            pval = wilcoxon(v).pvalue if m != "Perfect foresight" and (v != 0).any() else np.nan
            rows.append([wnd.split(" ")[0], f"{int(c*100)}%", m, f"{v.mean():+.2f}", f"{(v > 1e-9).sum()}/30",
                         "" if np.isnan(pval) else (f"{pval:.3f}" + (" *" if pval < .05 else ""))])
t2 = pd.DataFrame(rows, columns=["Window", "Storage", "Strategy", "Mean gain (%)", "Series with gain", "p (vs sell now)"])
t2.to_csv("Table2_decision_value.csv", index=False)
table_png(t2, "Table2_decision_value.png",
          "Table 2. Revenue vs. always selling now (Wilcoxon signed-rank across 30 series; * p < 0.05)",
          [.14, .1, .24, .16, .16, .2])
print("done")
