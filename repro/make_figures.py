"""
Reproducible figure generation for the VCIBA manuscript (Linux, portable paths).

Non-destructive: regenerates data-driven figures into repro/figures_regen/ for
parity checking, and writes three NEW publication-quality figures directly into
paper/figures/ that support the corrected, defensible narrative:

  fig_region_correlation.png   - inter-region peak-pressure correlation heatmap
                                 (the basis of the r=-0.34 claim) + raw contrast
  fig_classification_cv.png    - 5-fold CV weighted-F1, RF vs XGB, by region
  fig_temp_distribution.png    - cleaned plantar-temperature distributions by region

The existing DTW / SHAP / trend figures in paper/figures/ are left untouched.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BL = ROOT / "Step_Segmentation_Metrics" / "Baselines"
REGEN = ROOT / "repro" / "figures_regen"
PAPER_FIG = ROOT / "paper" / "figures"
REGEN.mkdir(parents=True, exist_ok=True)

PRESSURE = ["pData", "pData_2", "pData_3", "pData_4"]
TEMP = ["tData", "tData_2", "tData_3", "tData_4"]
LAB = {"pData": "R1 Toe", "pData_2": "R2 Mid-foot", "pData_3": "R3 Lateral", "pData_4": "R4 Heel"}
TLAB = {"tData": "R1 Toe", "tData_2": "R2 Mid-foot", "tData_3": "R3 Lateral", "tData_4": "R4 Heel"}
TEMP_MIN, TEMP_MAX = 15.0, 45.0
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "savefig.bbox": "tight"})

stats = json.loads((ROOT / "repro" / "canonical_stats.json").read_text())


def fig_foot_regions():
    """Overlay labelled sensor zones on the original foot-outline art so the
    four regions (and which ones are the metatarsal/forefoot sensors) are
    explicit. Reads the preserved raw outline; never reads its own output."""
    import matplotlib.image as mpimg
    from matplotlib.patches import FancyBboxPatch
    raw = PAPER_FIG / "fig_foot_outline_raw.png"
    if not raw.exists():
        print(f"  [skip] {raw} not found; cannot annotate foot regions")
        return
    img = mpimg.imread(raw)
    h, w = img.shape[0], img.shape[1]
    # sensor centres as fractions of (width, height); image origin is top-left
    zones = [
        ("R1", "Toe (metatarsal M1)", 0.45, 0.13, "#1b9e77"),
        ("R2", "Mid-foot (metatarsal M2)", 0.37, 0.42, "#d95f02"),
        ("R3", "Lateral (metatarsal M3)", 0.61, 0.40, "#7570b3"),
        ("R4", "Heel", 0.50, 0.84, "#e7298a"),
    ]
    label_xy = {  # text-box anchor (fraction) and arrow target side
        "R1": (1.18, 0.10), "R2": (-0.32, 0.34),
        "R3": (1.18, 0.42), "R4": (1.18, 0.84),
    }
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.imshow(img, extent=[0, w, h, 0])
    for tag, name, fx, fy, col in zones:
        cx, cy = fx * w, fy * h
        ax.scatter([cx], [cy], s=520, facecolor=col, edgecolor="white",
                   linewidth=2, zorder=5, alpha=0.92)
        ax.text(cx, cy, tag, color="white", ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=6)
        lx, ly = label_xy[tag]
        ha = "left" if lx > 1 else "right"
        ax.annotate(f"{tag}: {name}", xy=(cx, cy), xycoords="data",
                    xytext=(lx * w, ly * h), textcoords="data",
                    ha=ha, va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc=col, ec="none", alpha=0.15),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
    ax.set_xlim(-0.40 * w, 1.95 * w)
    ax.set_ylim(1.06 * h, -0.06 * h)
    ax.axis("off")
    ax.text(0.5, -0.04,
            "Regions R1\u2013R3 are the forefoot \"metatarsal\" sensors (M1\u2013M3); "
            "R4 is the heel sensor.\nAll four are FlexiForce A401 units in a custom smart insole.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#333333")
    for out in (REGEN / "fig_foot_regions.png", PAPER_FIG / "fig_foot_regions.png"):
        fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_region_correlation():
    combined = pd.DataFrame()
    for r in PRESSURE:
        combined[LAB[r]] = pd.read_csv(BL / f"{r}_stepwise_baseline.csv")["peak_pressure"].fillna(0)
    corr = combined.corr()
    raw = stats["raw_descriptive"]["raw_sample_correlation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, mat, title in [
        (axes[0], corr.values, "Step-aligned baseline peak pressure\n(exploratory; zero-padded)"),
        (axes[1], np.array([[raw[a][b] for b in PRESSURE] for a in PRESSURE]),
         "Raw per-sample peak pressure\n(near-zero linear coupling)")]:
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([LAB[r] for r in PRESSURE], rotation=45, ha="right")
        ax.set_yticklabels([LAB[r] for r in PRESSURE])
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=9)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Inter-region pressure correlation: method matters", fontsize=12)
    fig.tight_layout()
    for out in (REGEN / "fig_region_correlation.png", PAPER_FIG / "fig_region_correlation.png"):
        fig.savefig(out)
    plt.close(fig)


def fig_classification_cv():
    regions = [r for r in PRESSURE if stats["classification"][r].get("rf_cv_f1_mean") is not None]
    rf = [stats["classification"][r]["rf_cv_f1_mean"] for r in regions]
    rfe = [stats["classification"][r]["rf_cv_f1_std"] for r in regions]
    xg = [stats["classification"][r]["xgb_cv_f1_mean"] for r in regions]
    xge = [stats["classification"][r]["xgb_cv_f1_std"] for r in regions]
    x = np.arange(len(regions)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w / 2, rf, w, yerr=rfe, capsize=4, label="Random Forest", color="#2c7fb8")
    ax.bar(x + w / 2, xg, w, yerr=xge, capsize=4, label="XGBoost", color="#f03b20")
    ax.set_xticks(x); ax.set_xticklabels([LAB[r] for r in regions])
    ax.set_ylabel("Weighted F1 (5-fold stratified CV)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Anomaly-boundary recovery by region (R1 excluded: only 7 steps)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for out in (REGEN / "fig_classification_cv.png", PAPER_FIG / "fig_classification_cv.png"):
        fig.savefig(out)
    plt.close(fig)


def fig_temp_distribution():
    filt = pd.read_csv(ROOT / "filtered_patients_25_steps.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for c in TEMP:
        d = filt[c].replace([np.inf, -np.inf], np.nan).dropna()
        d = d[(d >= TEMP_MIN) & (d <= TEMP_MAX)]
        ax.hist(d, bins=40, histtype="step", linewidth=1.8, density=True, label=TLAB[c])
    ax.set_xlabel("Plantar temperature (degrees C, cleaned 15-45)")
    ax.set_ylabel("Density")
    ax.set_title("Cleaned plantar-temperature distribution by region")
    ax.legend(); ax.grid(alpha=0.3)
    for out in (REGEN / "fig_temp_distribution.png", PAPER_FIG / "fig_temp_distribution.png"):
        fig.savefig(out)
    plt.close(fig)


def fig_peak_distributions():
    # parity regeneration (non-destructive) of regional peak-pressure distributions
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for r in PRESSURE:
        pp = pd.read_csv(BL / f"{r}_stepwise_baseline.csv")["peak_pressure"].dropna()
        ax.hist(pp, bins=30, histtype="step", linewidth=1.6, label=f"{LAB[r]} (mean {pp.mean():.0f})")
    ax.set_xlabel("Peak pressure (baseline steps)")
    ax.set_ylabel("Count")
    ax.set_title("Regional stepwise-baseline peak-pressure distributions")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(REGEN / "fig_peak_distributions.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_foot_regions()
    fig_region_correlation()
    fig_classification_cv()
    fig_temp_distribution()
    fig_peak_distributions()
    print("Figures written to:")
    print(f"  {PAPER_FIG}/  (fig_region_correlation, fig_classification_cv, fig_temp_distribution)")
    print(f"  {REGEN}/  (parity copies + fig_peak_distributions)")
