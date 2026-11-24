"""Builds the figures used in the CV report, the NLP report and the IEEE paper.

Every number comes from results/tables (the locked-test run). Run from the repository root:
    python deliverables/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables")
OUT = os.path.join(ROOT, "deliverables", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"cv": "#1f5fa8", "nlp": "#2e8b57", "fusion": "#7b4fa0", "simple": "#c0703a", "tech": "#8c8c8c", "prior": "#d0d0d0"}
H = [5, 10, 20, 40]


def tab(name):
    return pd.read_csv(os.path.join(TAB, name + ".csv"))


# ---------------------------------------------------------------- 1. CV and fusion against the numbers-only baselines (test)
pan = tab("phase_a_panel_test").set_index("model")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
series = [("cv_top3", "CV top-3", C["cv"]), ("fusion_late_top3", "CV+NLP late fusion", C["fusion"]),
          ("baseline_simple", "simple numbers", C["simple"]), ("tech_gbm", "technical LightGBM", C["tech"]),
          ("prior", "prior", C["prior"])]
titles = {"dir": "Direction (macro-F1)", "vol": "Volatility (R$^2$)", "dd": "Drawdown >10% (AUC)"}
x = np.arange(4)
w = 0.16
for ax, k in zip(axes, ["dir", "vol", "dd"]):
    for i, (m, lab, col) in enumerate(series):
        vals = [pan.loc[m, f"{k}_{h}"] for h in H]
        ax.bar(x + (i - 2) * w, vals, w, color=col, label=lab, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in H])
    ax.set_title(titles[k])
    ax.axhline(0 if k == "vol" else (0.5 if k == "dd" else 0), color="black", lw=0.6)
    if k == "dd":
        ax.set_ylim(0.45, 0.75)
    if k == "dir":
        ax.set_ylim(0.15, 0.40)
axes[0].set_ylabel("locked test, 2024-2026")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(OUT, "fig_benchmarks_test.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 2. Event study on the locked test
ev = tab("event_study_by_form_test")
order = ["no filing (control)", "8-K", "10-K", "10-Q"]
ev = ev.set_index("group").loc[order].reset_index()
fig, ax = plt.subplots(figsize=(3.5, 2.2))
cols = [C["prior"], C["nlp"], C["nlp"], C["nlp"]]
ax.barh(range(4), ev["mean_abs_z"], xerr=[ev["mean_abs_z"] - ev["ci_lo"], ev["ci_hi"] - ev["mean_abs_z"]], color=cols,
        error_kw={"elinewidth": 0.8, "capsize": 2})
ax.set_yticks(range(4))
ax.set_yticklabels([f"{g}  (n={n:,})" for g, n in zip(ev["group"], ev["n"])])
ax.set_xlabel("mean |abnormal 2-day move| (normal-day units)")
for i, v in enumerate(ev["mean_abs_z"]):
    ax.text(v + 0.18, i, f"{v:.2f}", va="center", fontsize=8)
ax.set_xlim(0, 2.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_event_study_test.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 3. Language models against no-text baselines on the same filings (test)
evt = tab("phase_a_event_test").set_index("model")
models = ["tfidf_lr", "finbert_ft", "secbert_ft", "roberta_l_ft", "longformer_ft", "deberta_l_ft", "fingpt_frozen",
          "company_event", "simple_event", "tech_event", "prior_event"]
names = {"tfidf_lr": "TF-IDF + LR", "finbert_ft": "FinBERT", "secbert_ft": "SEC-BERT", "roberta_l_ft": "RoBERTa-large",
         "longformer_ft": "Longformer", "deberta_l_ft": "DeBERTa-v3-large", "fingpt_frozen": "FinGPT (frozen)",
         "company_event": "company base rates", "simple_event": "simple numbers", "tech_event": "technical LightGBM",
         "prior_event": "prior"}
tasks = [f"{k}_{h}" for k in ("dir", "vol", "dd") for h in H]
M = evt.loc[models, tasks].to_numpy(float)
fig, ax = plt.subplots(figsize=(7.0, 3.1))
# colour each block of columns on its own scale so direction, R^2 and AUC are comparable within a target
show = np.zeros_like(M)
for j in range(0, 12, 4):
    blk = M[:, j:j + 4]
    lo, hi = np.nanmin(blk), np.nanmax(blk)
    show[:, j:j + 4] = (blk - lo) / (hi - lo + 1e-9)
ax.imshow(show, cmap="Greens", aspect="auto", vmin=-0.15, vmax=1.15)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if show[i, j] > 0.72 else "black")
ax.set_xticks(range(12))
ax.set_xticklabels([t.replace("_", " ") + "d" for t in tasks], rotation=0, fontsize=7)
ax.set_yticks(range(len(models)))
ax.set_yticklabels([names[m] for m in models], fontsize=7.5)
ax.axhline(6.5, color="black", lw=1.0)
for j in (3.5, 7.5):
    ax.axvline(j, color="white", lw=2)
ax.set_title("Filing events, locked test: direction macro-F1 | volatility R$^2$ | drawdown AUC (no-text baselines below the line)")
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_nlp_events_test.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 4. Hypothesis outcomes against the strong baselines
reg = tab("hypothesis_register").set_index("id")
rows = [("H1c", "CV vs numbers-only"), ("H2c", "NLP vs no-text (same filings)"), ("H3c", "CV+NLP fusion vs numbers-only"),
        ("H9a", "CV combination vs best single CV"), ("H8", "technical + CV + NLP vs technical"),
        ("H8f", "text beyond calendar + company")]
import re
better, same, worse = [], [], []
for hid, _ in rows:
    s = reg.loc[hid, "outcome"]
    b = int(re.search(r"better (\d+)", s).group(1))
    n = int(re.search(r"no clear difference (\d+)", s).group(1))
    wv = int(re.search(r"worse (\d+)", s).group(1))
    better.append(b); same.append(n); worse.append(wv)
fig, ax = plt.subplots(figsize=(3.5, 2.3))
y = np.arange(len(rows))[::-1]
ax.barh(y, better, color=C["nlp"], label="better")
ax.barh(y, same, left=better, color="#cfcfcf", label="no clear difference")
ax.barh(y, worse, left=np.add(better, same), color="#c44e52", label="worse")
ax.set_yticks(y)
ax.set_yticklabels([f"{h}: {t}" for h, t in rows], fontsize=7)
ax.set_xlabel("tasks out of 12 (Holm-corrected)")
ax.set_xlim(0, 12)
ax.legend(loc="lower center", bbox_to_anchor=(0.35, -0.55), ncol=3, frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_hypotheses.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 5. Per-network test skill for the CV models (direction, all four horizons)
cvl = tab("cv_leaderboard_test")
nets = [("jkx_cnn_ft", "Custom CNN (scratch)"), ("resnet50_ft", "ResNet-50"), ("effnet_b0_ft", "EfficientNet-B0"),
        ("convnext_t_ft", "ConvNeXt-T"), ("vit_b16_ft", "ViT-B/16"), ("swin_t_ft", "Swin-T"), ("yolov8_cls_ft", "YOLOv8s-cls"),
        ("resnet50_frozen_shuffled", "shuffled-label control")]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
for ax, k in zip(axes, ["dir", "vol", "dd"]):
    mat = []
    for m, _ in nets:
        mat.append([cvl[(cvl.model == m) & (cvl.task == f"{k}_{h}")]["primary"].iloc[0] for h in H])
    mat = np.array(mat)
    ax.imshow(mat, cmap="Blues", aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(4):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > np.nanmean(mat) + 0.6 * np.nanstd(mat) else "black")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{h}d" for h in H])
    ax.set_title(titles[k])
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_yticks(range(len(nets)))
axes[0].set_yticklabels([n for _, n in nets], fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_cv_networks_test.png"), bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------- 6. System architecture (data -> models -> combination -> outputs)
def box(ax, x, y, w, h, text, fc, fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec="#333333", lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333"))


fig, ax = plt.subplots(figsize=(7.2, 2.9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.2)
ax.axis("off")
box(ax, 0.1, 2.55, 1.75, 1.2, "Price bars\n1W, 1D, 1H\n50 stocks\n2016-2026", "#dbe7f5")
box(ax, 0.1, 0.45, 1.75, 1.2, "SEC EDGAR\n10-K, 10-Q, 8-K\n10,269 filings\npoint-in-time", "#dcefe3")
box(ax, 2.2, 2.55, 1.75, 1.2, "Redraw charts\n3 x 224 x 224\n52 / 63 / 70\ncandles", "#dbe7f5")
box(ax, 2.2, 0.45, 1.75, 1.2, "MD&A / 8-K body\n[form][section]\ntext, chunked", "#dcefe3")
box(ax, 4.3, 2.55, 1.85, 1.2, "7 vision networks\nCNN, ResNet, EffNet,\nConvNeXt, ViT, Swin,\nYOLOv8-cls", "#c6dbef")
box(ax, 4.3, 0.45, 1.85, 1.2, "7 text models\nTF-IDF, FinBERT,\nSEC-BERT, RoBERTa,\nDeBERTa, Longformer,\nFinGPT-7B", "#c7e6d0", fs=6.6)
box(ax, 6.5, 1.5, 1.6, 1.2, "Top-3 per target\nchosen on 2022-23,\nblend, calibrate", "#e3d7ee")
box(ax, 8.4, 2.55, 1.5, 1.2, "12 outputs\ndirection, volatility,\ndrawdown x\n5/10/20/40 days", "#f3e3cf", fs=6.8)
box(ax, 8.4, 0.45, 1.5, 1.2, "Monte Carlo\n20-day cone\n4,000 paths", "#f3e3cf")
for y in (3.15, 1.05):
    arrow(ax, 1.85, y, 2.2, y)
    arrow(ax, 3.95, y, 4.3, y)
arrow(ax, 6.15, 3.15, 6.5, 2.4)
arrow(ax, 6.15, 1.05, 6.5, 1.8)
arrow(ax, 8.1, 2.3, 8.4, 3.0)
arrow(ax, 8.1, 1.9, 8.4, 1.2)
ax.text(5.0, 4.05, "Each network has one shared multi-task head: 4 horizons x (3 direction logits, log-volatility, drawdown logit)",
        ha="center", fontsize=7.2, style="italic")
ax.text(5.0, 0.08, "Locked test 2024-01 to 2026-08, sealed with a hashed freeze before it was opened", ha="center", fontsize=7.2,
        style="italic")
fig.savefig(os.path.join(OUT, "fig_architecture.png"), bbox_inches="tight")
plt.close(fig)

print("figures written to", OUT, sorted(os.listdir(OUT)))
