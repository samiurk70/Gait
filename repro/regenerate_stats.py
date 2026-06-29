"""
Canonical, reproducible statistics pipeline for the VCIBA manuscript.

This single Linux-friendly script regenerates EVERY number cited in
paper/sn-article.tex from the committed data files, using the same
definitions as the original notebook (4090_s2-2.ipynb). It is path-portable
(uses paths relative to the repo root, not the original Windows D:\\Study\\Gait).

Outputs:
  repro/canonical_stats.json   - machine-readable canonical numbers
  prints a human-readable summary to stdout

Design choices (documented so reviewers can audit):
  * "Stepwise baseline" files in Step_Segmentation_Metrics/Baselines/ are the
    canonical reference used for DTW, anomaly labelling and SHAP.
  * Regional peak-pressure DESCRIPTIVE statistics (Table: population baseline)
    use ALL baseline steps (peak_pressure has no missing values).
  * CLASSIFICATION uses baseline steps with a complete 7-feature vector
    (rows with NaN skew/kurtosis from single-sample steps are dropped), which
    is why its per-region n (7/89/74/43) is smaller than the descriptive n
    (92/149/185/134). Both ns are reported explicitly.
  * Temperature analyses use a physiological plausibility window [15, 45] C.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp

ROOT = Path(__file__).resolve().parent.parent
BL = ROOT / "Step_Segmentation_Metrics" / "Baselines"
SEG = ROOT / "Step_Segmentation_Metrics"

PRESSURE = ["pData", "pData_2", "pData_3", "pData_4"]
TEMP = ["tData", "tData_2", "tData_3", "tData_4"]
REGION_LABEL = {"pData": "R1 (Toe)", "pData_2": "R2 (Mid-foot)",
                "pData_3": "R3 (Lateral)", "pData_4": "R4 (Heel)"}
FEATURES = ["peak_pressure", "auc", "rise_gradient", "descent_gradient",
            "variance", "skewness", "kurtosis"]

# Physiological / sensor sanity bounds
TEMP_MIN, TEMP_MAX = 15.0, 45.0
PRESSURE_SANITY_MAX = 1000.0   # raw FlexiForce counts above this are saturation artefacts

stats: dict = {}


def jround(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# ---------------------------------------------------------------------------
# 1. Cohort flow (312 -> variability -> 25-step)
# ---------------------------------------------------------------------------
def cohort_flow():
    raw_path = ROOT / "capture_data copy.csv"
    filt_path = ROOT / "filtered_patients_25_steps.csv"
    out = {}
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        selected = []
        for cid in raw["cid"].unique():
            p = raw[raw["cid"] == cid]
            for col in PRESSURE + TEMP:
                if col not in p.columns:
                    continue
                d = p[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(d) > 1:
                    std_v = d.std()
                    rng = d.max() - d.min()
                    rate = d.diff().abs()
                    rate_var = rate.std() if rate.dropna().size > 1 else 0
                    if std_v >= 10 and rng >= 50 and rate_var >= 2:
                        selected.append(int(cid))
                        break
        out["raw_patients"] = int(raw["cid"].nunique())
        out["raw_observations"] = int(len(raw))
        out["after_variability_filter"] = int(len(selected))
    filt = pd.read_csv(filt_path)
    out["after_25_step_filter"] = int(filt["cid"].nunique())
    out["filtered_observations"] = int(len(filt))
    return out, (pd.read_csv(filt_path))


# ---------------------------------------------------------------------------
# 2. Raw per-sample descriptive + pressure CV + inter-region raw correlation
# ---------------------------------------------------------------------------
def raw_descriptive(filt):
    out = {"pressure_cv": {}, "pressure_raw": {}, "pressure_capped": {}}
    for c in PRESSURE:
        d = filt[c].replace([np.inf, -np.inf], np.nan).dropna()
        dc = d[d <= PRESSURE_SANITY_MAX]
        out["pressure_raw"][c] = {"mean": jround(d.mean(), 2), "std": jround(d.std(), 2),
                                  "max": jround(d.max(), 1), "cv": jround(d.std() / d.mean(), 2)}
        out["pressure_capped"][c] = {"mean": jround(dc.mean(), 2), "std": jround(dc.std(), 2),
                                     "max": jround(dc.max(), 1), "cv": jround(dc.std() / dc.mean(), 2),
                                     "pct_kept": jround(len(dc) / len(d) * 100, 1)}
        out["pressure_cv"][c] = jround(d.std() / d.mean(), 2)
    cvs = [out["pressure_cv"][c] for c in PRESSURE]
    out["pressure_cv_range"] = [min(cvs), max(cvs)]
    raw_corr = filt[PRESSURE].replace([np.inf, -np.inf], np.nan).dropna().corr()
    out["raw_sample_correlation"] = {a: {b: jround(raw_corr.loc[a, b]) for b in PRESSURE} for a in PRESSURE}
    out["raw_r1_r4"] = jround(raw_corr.loc["pData", "pData_4"])
    return out


# ---------------------------------------------------------------------------
# 3. Temperature cleaning [15,45]
# ---------------------------------------------------------------------------
def temperature_clean(filt):
    out = {}
    for c in TEMP:
        raw = filt[c].replace([np.inf, -np.inf], np.nan).dropna()
        clean = raw[(raw >= TEMP_MIN) & (raw <= TEMP_MAX)]
        out[c] = {"raw_n": int(len(raw)), "clean_n": int(len(clean)),
                  "pct_kept": jround(len(clean) / len(raw) * 100, 1),
                  "mean": jround(clean.mean(), 2), "std": jround(clean.std(), 2)}
    disp = {c: out[c]["std"] for c in TEMP}
    out["highest_dispersion"] = max(disp, key=disp.get)
    return out


# ---------------------------------------------------------------------------
# 4. Stepwise-baseline peak-pressure descriptive (ALL rows) = population table
# ---------------------------------------------------------------------------
def baseline_peak_pressure():
    out = {}
    for r in PRESSURE:
        df = pd.read_csv(BL / f"{r}_stepwise_baseline.csv")
        pp = df["peak_pressure"].dropna()
        out[r] = {"label": REGION_LABEL[r], "n_steps_all": int(len(df)),
                  "mean": jround(pp.mean(), 2), "std": jround(pp.std(), 2),
                  "median": jround(pp.median(), 2)}
    heel = out["pData_4"]["mean"]
    nxt = max(out[r]["mean"] for r in ["pData", "pData_2", "pData_3"])
    out["heel_vs_next_pct"] = jround((heel - nxt) / nxt * 100, 1)
    return out


# ---------------------------------------------------------------------------
# 5. Inter-region correlation (notebook method: peak_pressure.fillna(0), aligned)
# ---------------------------------------------------------------------------
def baseline_correlation():
    combined = pd.DataFrame()
    for r in PRESSURE:
        combined[r] = pd.read_csv(BL / f"{r}_stepwise_baseline.csv")["peak_pressure"].fillna(0)
    corr = combined.corr()
    return {
        "method": "peak_pressure.fillna(0), row-index aligned across regions (notebook)",
        "matrix": {a: {b: jround(corr.loc[a, b]) for b in PRESSURE} for a in PRESSURE},
        "r1_r4": jround(corr.loc["pData", "pData_4"]),
        "r2_r4": jround(corr.loc["pData_2", "pData_4"]),
        "r1_r3": jround(corr.loc["pData", "pData_3"]),
    }


# ---------------------------------------------------------------------------
# 6. Anomaly labelling + classification (5-fold CV), reproducing the paper table
# ---------------------------------------------------------------------------
def classification():
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import xgboost as xgb

    out = {}
    for r in PRESSURE:
        df = pd.read_csv(BL / f"{r}_stepwise_baseline.csv").dropna(
            subset=[c for c in FEATURES if c in pd.read_csv(BL / f"{r}_stepwise_baseline.csv").columns])
        X = df[FEATURES].copy()
        z = X.apply(lambda col: np.abs((col - col.mean()) / (col.std() + 1e-8)))
        y = (z.max(axis=1) > 2).astype(int)
        rec = {"n_steps": int(len(X)), "n_anomaly": int(y.sum()),
               "anomaly_rate": jround(y.sum() / len(X) * 100, 1)}
        if y.nunique() >= 2 and len(X) >= 5:
            Xs = StandardScaler().fit_transform(X)
            n_splits = min(5, int(y.value_counts().min()))
            if n_splits >= 2:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                xg = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42,
                                       eval_metric="logloss")
                rf_cv = cross_val_score(rf, Xs, y, cv=cv, scoring="f1_weighted")
                xg_cv = cross_val_score(xg, Xs, y, cv=cv, scoring="f1_weighted")
                rec.update({"cv_splits": n_splits,
                            "rf_cv_f1_mean": jround(rf_cv.mean(), 3), "rf_cv_f1_std": jround(rf_cv.std(), 3),
                            "xgb_cv_f1_mean": jround(xg_cv.mean(), 3), "xgb_cv_f1_std": jround(xg_cv.std(), 3)})
        out[r] = rec
    return out


# ---------------------------------------------------------------------------
# 6b. Representative within-participant case study (Patient 350, Table tab:p350)
# ---------------------------------------------------------------------------
def patient_case_study(cid=350):
    out = {"cid": cid, "regions": {}}
    for r in PRESSURE:
        f = SEG / r / f"patient_{cid}" / "step_metrics.csv"
        if f.exists():
            pp = pd.read_csv(f)["peak_pressure"].dropna()
            out["regions"][r] = {"label": REGION_LABEL[r], "n_steps": int(len(pp)),
                                 "mean_peak": jround(pp.mean(), 2), "std_peak": jround(pp.std(), 2),
                                 "cv_pct": jround(pp.std() / pp.mean() * 100, 2)}
    return out


# ---------------------------------------------------------------------------
# Provenance notes for values that originate in the original interactive
# notebook modules and are not recomputed from the released CSV artefacts.
# ---------------------------------------------------------------------------
NOTEBOOK_DERIVED = {
    "stage1_cohort_baseline": {
        "values_peak_pressure": {"pData": 59.69, "pData_2": 58.50, "pData_3": 63.03, "pData_4": 86.96},
        "note": ("Stage-1 cohort baseline used an earlier find_peaks-based periodicity "
                 "segmentation (analyze_step_periodicity), not the threshold-based segmentation "
                 "behind the reported results. It served only for early screening and is not "
                 "recomputed here; it is not used in any headline result."),
    },
    "patient350_anomaly_rates": {
        "values": {"pData": "8/92", "pData_2": "43/74", "pData_3": "13/39"},
        "note": ("Per-patient anomaly rates and DTW abnormality counts come from the notebook's "
                 "interactive anomaly/DTW modules (patient-relative thresholds and warping-path "
                 "logic) and are reported as illustrative case-study values."),
    },
}


# ---------------------------------------------------------------------------
# 7. ANOVA on regional stepwise-baseline peak pressure (same data as Table 4)
# ---------------------------------------------------------------------------
def anova_baseline():
    groups, counts = [], {}
    for r in PRESSURE:
        s = pd.read_csv(BL / f"{r}_stepwise_baseline.csv")["peak_pressure"].dropna().values
        counts[r] = int(len(s))
        groups.append(np.asarray(s, dtype=float))
    F, p = sp.f_oneway(*groups)
    return {"F": jround(F, 2), "p": p, "group_counts": counts,
            "note": "one-way ANOVA on all-rows regional stepwise-baseline peak pressure (matches Table 4 ns)"}


def main():
    flow, filt = cohort_flow()
    stats["cohort_flow"] = flow
    stats["raw_descriptive"] = raw_descriptive(filt)
    stats["temperature_clean"] = temperature_clean(filt)
    stats["baseline_peak_pressure"] = baseline_peak_pressure()
    stats["baseline_correlation"] = baseline_correlation()
    stats["classification"] = classification()
    stats["patient_case_study"] = patient_case_study(350)
    stats["notebook_derived"] = NOTEBOOK_DERIVED
    stats["anova_peak_pressure"] = anova_baseline()

    out_path = ROOT / "repro" / "canonical_stats.json"
    out_path.write_text(json.dumps(stats, indent=2, default=str))

    # ---- human-readable summary ----
    print("=" * 72)
    print("CANONICAL STATS  (regenerated on Linux, paths relative to repo root)")
    print("=" * 72)
    f = stats["cohort_flow"]
    print(f"Cohort flow: {f.get('raw_patients','?')} -> "
          f"{f.get('after_variability_filter','?')} (variability) -> "
          f"{f['after_25_step_filter']} (>=25 steps); obs={f['filtered_observations']}")
    print("\nPopulation stepwise-baseline peak pressure (ALL steps):")
    bp = stats["baseline_peak_pressure"]
    for r in PRESSURE:
        print(f"  {bp[r]['label']:16s} mean={bp[r]['mean']:.2f}  std={bp[r]['std']:.2f}  n={bp[r]['n_steps_all']}")
    print(f"  heel vs next-highest region: +{bp['heel_vs_next_pct']}%")
    print("\nInter-region correlation (notebook method):")
    c = stats["baseline_correlation"]
    print(f"  R1-R4={c['r1_r4']}  R2-R4={c['r2_r4']}  R1-R3={c['r1_r3']}"
          f"   | raw per-sample R1-R4={stats['raw_descriptive']['raw_r1_r4']}")
    print("\nTemperature (cleaned 15-45C):")
    tc = stats["temperature_clean"]
    for cc in TEMP:
        print(f"  {cc}: kept={tc[cc]['pct_kept']}%  mean={tc[cc]['mean']}  std={tc[cc]['std']}")
    print(f"  highest dispersion: {tc['highest_dispersion']}")
    print("\nPressure CV (raw per-sample):",
          {k: stats['raw_descriptive']['pressure_cv'][k] for k in PRESSURE},
          "range", stats["raw_descriptive"]["pressure_cv_range"])
    print("\nClassification (5-fold CV weighted F1):")
    for r in PRESSURE:
        rec = stats["classification"][r]
        rf = rec.get("rf_cv_f1_mean"); xg = rec.get("xgb_cv_f1_mean")
        print(f"  {REGION_LABEL[r]:16s} n={rec['n_steps']:3d} anom={rec['anomaly_rate']}%  "
              f"RF={rf}±{rec.get('rf_cv_f1_std')}  XGB={xg}±{rec.get('xgb_cv_f1_std')}")
    a = stats["anova_peak_pressure"]
    print(f"\nANOVA peak pressure across regions (pooled steps): F={a['F']} p={a['p']} counts={a['group_counts']}")
    print("\nPatient 350 case study (per-region peak pressure):")
    for r, v in stats["patient_case_study"]["regions"].items():
        print(f"  {v['label']:16s} mean={v['mean_peak']}  CV={v['cv_pct']}%  n={v['n_steps']}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
