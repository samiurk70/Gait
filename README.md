# Gait Cycle Dynamics Analysis Using Explainable AI and Dynamic Time Warping

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Award](https://img.shields.io/badge/Award-Top%205%20MSc%20Thesis%20MDX%202025-gold)
![License](https://img.shields.io/badge/License-Private-red)
![University](https://img.shields.io/badge/Middlesex%20University%20London-MSc%20Data%20Science-red)

> **Top 5 MSc Data Science Thesis — Middlesex University London, 2025**
> Supervisor: Clifford De Raffaele | Student: Samiur Rahman Khan

---

## Abstract

This study presents an advanced approach to gait analysis by integrating **Dynamic Time Warping (DTW)**, **machine learning**, and **step-wise feature extraction** to evaluate foot pressure and temperature patterns across different regions of the foot during walking. By segmenting steps into meaningful units, key metrics such as peak pressure, descent gradient, and maximum temperature were analysed across the metatarsal and heel regions.

Results revealed that the heel consistently exhibited higher peak pressures with lower variability, reflecting its primary role in impact absorption, whereas metatarsal regions demonstrated greater variability in both pressure and temperature, indicating sensitivity to different phases of gait. Machine learning models enhanced with **Explainable AI (XAI)** methods — specifically SHAP values — identified descent gradient and temperature as the most significant features influencing anomaly detection, adding transparency to the clinical decision-making process.

**Keywords:** Dynamic Time Warping · Random Forest · Gait Analysis · Explainable AI · Biomedical Pattern Recognition · Foot Pressure · Plantar Temperature

---

## Research Questions

1. Can DTW-based alignment effectively detect step-level gait deviations across patients with varying walking speeds and patterns?
2. Which foot pressure regions are most predictive of gait abnormalities when combined with temperature data?
3. How do XAI methods (SHAP) improve clinical interpretability of machine learning models applied to biomedical gait data?

---

## Dataset

Data was collected using a **custom smart insole system** embedded with FlexiForce A401 pressure sensors and temperature sensors across four distinct foot regions:

| Column | Region | Type |
|---|---|---|
| `pData` | Metatarsal 1 (Toe) | Pressure |
| `pData_2` | Metatarsal 2 (Mid-foot) | Pressure |
| `pData_3` | Metatarsal 3 (Lateral) | Pressure |
| `pData_4` | Heel | Pressure |
| `tData` | Metatarsal 1 (Toe) | Temperature |
| `tData_2` | Metatarsal 2 (Mid-foot) | Temperature |
| `tData_3` | Metatarsal 3 (Lateral) | Temperature |
| `tData_4` | Heel | Temperature |

- **Participants:** 312 patients
- **Format:** MySQL → CSV (via SQLite conversion)
- **Sampling:** High temporal resolution with timestamps per reading
- **Data note:** Dataset is proprietary and not publicly available. Code available upon request.

---

## Methodology

```
Raw Sensor Data (MySQL)
        │
        ▼
┌─────────────────────────┐
│  Preprocessing           │  ← ffill/bfill, inf→NaN, rate-of-change variability
│  & Patient Filtration    │     Threshold: std≥10, range≥50, rate variability≥2
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Step Segmentation       │  ← scipy.signal.find_peaks, threshold=5
│  & Metric Extraction     │     Peak pressure, AUC, rise/descent gradient,
└──────────┬──────────────┘     variance, skewness, kurtosis
           │
           ├──────────────────────────────────┐
           ▼                                  ▼
┌─────────────────────┐           ┌───────────────────────┐
│  Baseline Model 1   │           │  Baseline Model 2      │
│  (Top 5 patients    │           │  (Step-wise metrics,   │
│   by data volume)   │           │   min 25 steps/patient)│
└──────────┬──────────┘           └───────────┬───────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐           ┌───────────────────────┐
│  Random Forest      │           │  DTW Alignment         │
│  Classifier         │           │  (dtaidistance)        │
│  (per region)       │           │  Anomaly Detection     │
└──────────┬──────────┘           └───────────┬───────────┘
           │                                  │
           └──────────────┬───────────────────┘
                          ▼
              ┌─────────────────────┐
              │  XAI Integration    │
              │  (SHAP values)      │
              │  Temperature + DTW  │
              └─────────────────────┘
```

---

## Key Findings

### 1. Temperature Uniformity
Temperature data exhibited uniform step-wise rises and falls across all patients, in stark contrast to the non-uniform, spike-heavy nature of pressure data. This distinction made temperature a complementary rather than redundant feature for anomaly detection.

### 2. Metatarsal Region 2 — Highest Pressure Frequency & Fluctuation
Pressure distribution frequency was highest in Metatarsal Region 2, indicating that patients sustain prolonged pressure in this region — a potential indicator of future musculoskeletal stress. Standard deviation analysis confirmed Region 2 as having the broadest variance in step pressure, aligning with known gait biomechanics literature.

### 3. Heel (Region 4) — Highest Peak Pressure
The heel region consistently demonstrated significantly higher peak pressures (mean: 138.43) compared to all metatarsal regions (means: 65–69), confirming its dominant role in impact absorption during heel strike.

### 4. Toe-Oriented Gait → Elevated Heel Abnormality
Patients with toe-dominant gait patterns showed markedly higher anomaly rates in the heel region, suggesting compensatory loading mechanisms that could increase injury risk over time.

### 5. Region 1 & 4 Negative Correlation
A strong negative correlation (r = -0.34) was found between Metatarsal 1 (toe) and Heel peak pressures — when one increases, the other decreases — indicating active pressure redistribution during gait phases.

### 6. XAI Feature Importance
SHAP analysis identified **descent gradient** and **temperature** as the two most influential features in anomaly detection, with mean |SHAP| values of 0.26 and 0.17 respectively — providing clinically interpretable evidence for model decisions.

### 7. Region 3 — Highest Temperature Dispersion
Metatarsal Region 3 exhibited the most diverse temperature variations during foot lift-off, potentially indicating vascular or inflammatory responses in the lateral midfoot.

---

## Tech Stack

```
Data Processing:     pandas · numpy · tqdm · scipy.stats
Signal Processing:   scipy.signal.find_peaks · interp1d
Machine Learning:    scikit-learn (RandomForest) · XGBoost · joblib
Clustering:          KMeans · DBSCAN
Time-Series:         dtaidistance (DTW) · dtw_visualisation
Explainable AI:      SHAP · LIME
Visualisation:       matplotlib · plotly · seaborn · pandas.plotting
Database:            MySQL → SQLite → CSV
Environment:         Python 3.12.8 · VS Code
```

---

## Repository Structure

```
Gait/
├── Baseline_Metrics/               # First baseline — top 5 patients by data volume
├── Baselines_Stepwise/             # Second baseline — step-segmented metrics
├── Individual_Baseline_Metrics/    # Per-patient baseline profiles
├── Metric_Distributions/           # Distribution visualisations across regions
├── Metrics_Baselines/              # Aggregated baseline metric CSVs
├── Models/                         # Saved Random Forest models (joblib)
├── Pdata_Model/                    # Region-specific pressure models
├── Pdata_baseline/                 # Pressure baseline data
├── Random_Forest_Models/           # Per-region RF classifiers
├── Step_Segmentation_Metrics/      # Extracted step-level metrics
├── Step_Segmentation_Metrics_tdata/# Temperature step metrics
├── Stepwise_Model/                 # DTW + step-wise anomaly detection
├── 4090_s2.ipynb                   # Main analysis notebook
├── baseline_metrics.csv            # Aggregated baseline metrics
├── baseline_metrics_pdata[1-4].csv # Region-wise baseline metrics
├── capture_data.csv                # Raw sensor data (anonymised)
├── cleaned_capture_data.csv        # Preprocessed dataset
└── README.md
```

---

## Pipeline Walkthrough

### Step 1 — Preprocessing
```python
def preprocess_patient_data(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.ffill().bfill()
    return data
```

### Step 2 — Step Segmentation
```python
def segment_steps_with_metrics(data, column):
    threshold = 5  # Minimum pressure to register a step
    steps = []
    current_step = []
    for i, value in enumerate(data[column]):
        if value > threshold:
            current_step.append(value)
        elif current_step:
            steps.append(current_step)
            current_step = []
    # Extract: peak_pressure, AUC, rise_gradient,
    #          descent_gradient, variance, skewness, kurtosis
```

### Step 3 — DTW Alignment
```python
from dtaidistance import dtw, dtw_visualisation as dtwvis
distance = dtw.distance(patient_sequence, baseline_sequence)
path = dtw.warping_path(patient_sequence, baseline_sequence)
```

### Step 4 — SHAP Explainability
```python
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)
```

---

## Results Summary

| Finding | Region | Key Metric | Clinical Implication |
|---|---|---|---|
| Highest pressure frequency | Metatarsal 2 | Frequency distribution | Prolonged mid-foot loading |
| Highest peak pressure | Heel (Region 4) | Mean peak = 138.43 | Dominant shock absorption |
| Highest pressure fluctuation | Metatarsal 2 | Std dev = 0.85–0.90 | Mid-foot instability |
| Highest temperature dispersion | Metatarsal 3 | Scatter spread | Potential vascular stress |
| Negative cross-region pressure | Region 1 & 4 | r = -0.34 | Compensatory redistribution |
| Top XAI features | All regions | SHAP = 0.26, 0.17 | Descent gradient & temperature |

---

## Limitations

- Sensor variability and calibration inconsistencies across patients
- 300+ patients in raw data but majority filtered out due to <25 steps threshold
- DTW computationally expensive at scale — not yet real-time capable
- Baseline selection bias toward patients with most data rather than most typical gait
- Temperature and pressure combined but no concurrent EMG or kinematic data

---

## Citation

```bibtex
@mastersthesis{khan2025gait,
  author    = {Samiur Rahman Khan},
  title     = {An Analogy on Gait Cycle Dynamics Utilizing Explainable AI
               and Dynamic Time Warping for Pattern Recognition},
  school    = {Middlesex University London},
  year      = {2025},
  type      = {MSc Thesis},
  note      = {Top 5 MSc Data Science Project, CST4090 Individual Project},
  supervisor= {Clifford De Raffaele}
}
```

---

## Related Publications

- **Khan, S.**, Al-Amin, M., Hossain, H., Noor, N., & Sadik, M. W. (2020). A pragmatical study on blockchain empowered decentralized application development platform. *ICCA 2020, ACM*. [Cited 14×]
- **Sohan, M. F. A. A., Khan, S. R.**, et al. (2022). Towards a secured smart IoT using lightweight blockchain. *arXiv:2206.06925*. [Cited 13×]
- **Khan, S. R.**, Al-Amin, M. (2023). Towards a Novel Identity Check Using W3C Standards & Hybrid Blockchain for Paperless Verification. *IJIEEB, Vol.15, No.4*. DOI:10.5815/ijieeb.2023.04.02

---

## Author

**Samiur Rahman Khan**

MSc Data Science (Distinction) · Middlesex University London · 2025 

MSc Computer Network & Architecture (Summa Cum Laude) · American Internation University Bangladesh · 2022

[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Profile-blue)](https://scholar.google.co.uk/citations?user=ddQa7D4AAAAJ&hl)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-samiurk70-blue)](https://linkedin.com/in/samiurk70)
[![GitHub](https://img.shields.io/badge/GitHub-samiurk70-black)](https://github.com/samiurk70)

---

> **Note:** Dataset and full codebase are private due to data privacy agreements with participating patients. Code is available upon request for academic collaboration. Contact: samiurk70@gmail.com
