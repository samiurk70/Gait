# Paper Numbers Audit

All values regenerated on Linux by `repro/regenerate_stats.py` from the committed
data files (paths relative to the repo root; no Windows-specific paths). Compare
against `paper/sn-article.tex`.

Legend: OK = reproduces; FIX = manuscript value must change; CAVEAT = reproduces
but needs a methodological note.

| # | Claim in manuscript | Paper value | Reproduced value | Verdict |
|---|---------------------|-------------|------------------|---------|
| 1 | Cohort flow | 312 -> 92 -> 67, 58,807 obs | 312 -> 92 -> 67, 58,807 | OK |
| 2 | Heel baseline peak pressure | 112.58 | 112.58 (mean of 134 baseline steps) | OK |
| 3 | Region means (Table 4) | 80.52 / 57.78 / 45.28 / 112.58 | identical | OK |
| 4 | Region **std** (Table 4) | 24.71 / 24.49 / 29.67 / 39.12 | **18.80 / 30.44 / 33.95 / 46.49** | FIX |
| 5 | Baseline step counts (Table 4) | 92 / 149 / 185 / 134 | identical | OK |
| 6 | Heel vs next region | "~40% higher" | +39.8% (vs R1 80.52) | OK |
| 7 | R1-R4 correlation | -0.34 | -0.336 (step-aligned, zero-padded) | CAVEAT |
| 8 | R2-R4 correlation | 0.29 | 0.290 | CAVEAT |
| 9 | Raw per-sample R1-R4 corr | (not stated) | 0.0065 (near zero) | ADD contrast |
| 10 | Temperature kept % | 80-84% (R1-3), 62% (R4) | 84.2/80.8/83.5/62.4 | OK |
| 11 | Temperature means | 23.95-25.04 | 25.04/24.87/24.75/23.95 | OK |
| 12 | Temperature std | 3.82-4.17 | 3.82/4.17/3.89/4.09 | OK |
| 13 | Highest temp dispersion | "Regions 2 and 3" | R2 (4.17) then R3 (3.89) | OK |
| 14 | Pressure CV range (abstract) | "4.6 to 8.3" | **4.83 to 18.32** | FIX |
| 15 | ANOVA peak pressure | F=206.61, p<0.0001 | **F=107.61, p<1e-54** (Table 4 data) | FIX |
| 16 | Classification CV F1 (R2 RF) | 0.953 +/- 0.042 | 0.953 +/- 0.042 | OK |
| 17 | Classification CV F1 (R3 RF) | 0.900 +/- 0.072 | 0.900 +/- 0.072 | OK |
| 18 | Classification CV F1 (R4 RF) | 0.883 +/- 0.117 | 0.883 +/- 0.117 | OK |
| 19 | Classification CV F1 (R1) | 0.867 +/- 0.267 | **not computable** (1 anomaly / 7 steps) | FIX -> footnote |
| 20 | Hold-out accuracy = 1.000 (R1, R4) | reported in table | test sets of ~2-9 samples | FIX -> remove |
| 21 | Anomaly rates | 14.3/20.2/24.3/23.3 | identical | OK |
| 22 | Patient 350 mean peak (Table p350) | 65.83 / 69.23 / 63.74 / 138.43 | identical | OK |
| 23 | Patient 350 CV (Table p350) | 7.19 / 13.05 / 9.12 / 29.17 % | identical | OK |
| 24 | Stage-1 cohort baseline peaks | 59.69 / 58.50 / 63.03 / 86.96 | not recomputed (see note A) | CAVEAT |
| 25 | Patient 350 anomaly rates | 8/92, 43/74, 13/39 | not recomputed (see note B) | CAVEAT |
| 26 | Patient 350 DTW abnormal counts | 1/2/1/10 | not recomputed (see note B) | CAVEAT |

### Provenance notes (now emitted in `canonical_stats.json` under `notebook_derived`)
- **Note A (Stage-1):** the Stage-1 cohort baseline used an earlier `find_peaks`-based periodicity
  segmentation, not the threshold-based segmentation behind the reported results. It served only
  for early screening / z-score visualisation and is **not used in any headline result**. The
  manuscript text has been softened to present it as a coarse screening range (order 60--90 counts)
  rather than four exact decimals, removing the reproducibility liability.
- **Note B (per-patient anomaly / DTW):** Patient-350 anomaly rates and DTW abnormality counts
  come from the notebook's interactive anomaly/DTW modules (patient-relative thresholds and
  warping-path logic). They are framed in the paper as illustrative single-patient case-study
  values. The reproducible part of the Patient-350 case (per-region mean peak pressure and CV,
  rows 22-23) now regenerates exactly via `patient_case_study(350)`.

## Required manuscript edits
- **FIX #4**: replace Table 4 std column with 18.80 / 30.44 / 33.95 / 46.49.
- **FIX #14**: abstract pressure CV "4.6 to 8.3" -> "4.8 to 18.3".
- **FIX #15**: ANOVA "F=206.61" -> "F=107.61" on regional baseline peak pressure.
- **FIX #19/#20**: drop hold-out accuracy columns; move R1 to a footnote (not CV-evaluable).
- **CAVEAT #7/#8/#9**: state the -0.34/0.29 are step-index-aligned baseline correlations
  (zero-padded across unequal-length regions); note raw per-sample correlation is ~0.006,
  and that DTW temporal offsets (not linear r) are the substantive cross-region signal.
- **CAVEAT #24 (DONE)**: Stage-1 sentence reworded to a screening-only coarse range; exact
  decimals removed.
- **CAVEAT #25/#26**: Patient-350 anomaly/DTW counts retained as illustrative case-study values
  (notebook-derived); reproducible peak/CV portion verified (rows 22-23).

## Reproduce
```
python3 -m venv .venv_linux && .venv_linux/bin/pip install pandas numpy scikit-learn scipy matplotlib seaborn xgboost shap dtaidistance
.venv_linux/bin/python repro/regenerate_stats.py   # -> repro/canonical_stats.json
.venv_linux/bin/python repro/make_figures.py       # -> paper/figures/fig_region_correlation.png, etc.
```
