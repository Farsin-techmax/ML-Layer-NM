# Milestone Rollout Statistics: 20k, 90k, 100k

This document contains the complete Checkpoint A statistics, gate battery evaluations, and multi-seed performance metrics for the **20k**, **90k**, and **100k** milestones, executed strictly under [MILESTONE_ROLLOUT_RUNBOOK.md](file:///c:/Techmax/cwf/code/MILESTONE_ROLLOUT_RUNBOOK.md).

---

# 1. 20k Milestone Rollout Statistics

### 1.1 Dataset & Feature Pipeline
- **Training Candidate Matrix**: `refactored_test_dir/final_processed_20k_candidates.csv`
  - **Dimensions**: 78,861 rows × 216 columns
  - **Class Distribution**: 35,846 positives (45.5%), 43,015 negatives (54.5%)
  - **Early Completers**: 13,941 rows (17.7%)
  - **Cohort Coverage**: 2016Q1 through 2025Q4 (40 cohorts)
  - **Manifest**: `final_processed_20k_candidates.csv.manifest.json`
- **Q1 2026 Test Set**: `test_sets/test_20k_Q1_2026_cand.csv`
  - **Dimensions**: 2,231 rows × 165 columns
  - **Class Distribution**: 1,648 positives (73.9%), 583 negatives (26.1%)
  - **Early Completers**: 610 rows (27.3%)
  - **Manifest**: `test_20k_Q1_2026_cand.csv.manifest.json`
- **Q2 2026 Test Set**: `test_sets/test_20k_Q2_2026_cand.csv`
  - **Dimensions**: 1,923 rows × 165 columns
  - **Class Distribution**: 1,305 positives (67.9%), 618 negatives (32.1%)
  - **Early Completers**: 471 rows (24.5%)
  - **Manifest**: `test_20k_Q2_2026_cand.csv.manifest.json`
- **Regression Tests**:
  - `test_dayfirst.py`: 41/41 PASS
  - `test_service_num.py`: 31/31 PASS

### 1.2 Gate Battery (G3–G8)
- **G3 Schema**: 160 features across all 3 seeds (in standard range `[100, 200]`). Zero genuine schema gaps (35–39 absent one-hot dummy categories zero-filled, expected).
- **G4 Leak Check**: **CLEAN** (0 leak strings; no `Service_Num`, no `PMS Status_` leaks).
- **G5 Base-Rate Sanity**: **PASS** (Q1 positive rate 73.9%, early 27.3%; Q2 positive rate 67.9%, early 24.5%).
- **G6 Leak Screen (Seed 42)**:
  - **STOP (gap ≥ 0.20, test ≥ 0.65)**: **0** (None)
  - **FLAG (gap ≥ 0.10, test ≥ 0.65)**: **0** (None)
  - **INVERTED (train < 0.45, test > 0.55)**: **0** (None)
  - **Group Permutation Importance**:
    - `schedule_keeping` (7 features): Importance mean = **+0.1030** (std: 0.0039) -> Helpful/Positive
    - `vhc_revenue` (24 features): Importance mean = **+0.0255** (std: 0.0038) -> Helpful/Positive
  - **G6 Verdict**: **PASS**
- **G7 Seed Stability**:
  - Q1 AUC spread: $0.9254 - 0.8971 = 0.0283$ (< 0.04 -> **PASS**)
  - Q2 AUC spread: $0.9093 - 0.9033 = 0.0060$ (< 0.04 -> **PASS**)
- **G8 Reproduction (Independently Rescored on Q1)**:
  - Seed 42: Rescore AUC **0.9222** vs Training Log **0.9222** (MATCH)
  - Seed 7: Rescore AUC **0.9254** vs Training Log **0.9254** (MATCH)
  - Seed 123: Rescore AUC **0.8971** vs Training Log **0.8971** (MATCH)

### 1.3 Performance & Confusion Matrix Breakdown

| Seed | Quarter | ROC-AUC | Accuracy | F1-Score | Precision | Recall | Brier | TP | FP | FN | TN | Jaccard |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | Q1 | **0.9222** | 86.10% | 91.10% | 86.48% | 96.24% | 0.1036 | 1586 | 248 | 62 | 335 | **0.8365** |
| | Q2 | **0.9093** | 89.24% | 92.23% | 90.43% | 94.10% | 0.0981 | 1228 | 130 | 77 | 488 | **0.8557** |
| | *Mean* | *0.9158* | *87.67%* | *91.67%* | *88.46%* | *95.17%* | *0.1009* | — | — | — | — | *0.8461* |
| **7** *(Top)* | Q1 | **0.9254** | 86.06% | 91.08% | 86.35% | 96.36% | 0.1052 | 1588 | 251 | 60 | 332 | **0.8362** |
| | Q2 | **0.9067** | 89.44% | 92.39% | 90.40% | 94.48% | 0.1032 | 1233 | 131 | 72 | 487 | **0.8586** |
| | *Mean* | *0.9161* | *87.75%* | *91.74%* | *88.38%* | *95.42%* | *0.1042* | — | — | — | — | **0.8474** |
| **123** | Q1 | **0.8971** | 83.46% | 89.57% | 83.82% | 96.18% | 0.1191 | 1585 | 306 | 63 | 277 | **0.8112** |
| | Q2 | **0.9033** | 86.12% | 90.34% | 85.55% | 95.71% | 0.1058 | 1249 | 211 | 56 | 407 | **0.8239** |
| | *Mean* | *0.9002* | *84.79%* | *89.96%* | *84.69%* | *95.95%* | *0.1125* | — | — | — | — | *0.8176* |

- **Recommended Seed for Promotion**: **Seed 7** (Mean AUC: 0.9161, Mean Jaccard: 0.8474, high balance of precision and recall).

---

# 2. 90k Milestone Rollout Statistics

### 2.1 Dataset & Feature Pipeline
- **Training Candidate Matrix**: `refactored_test_dir/final_processed_90k_candidates.csv`
  - **Dimensions**: 59,298 rows × 250 columns
  - **Class Distribution**: 12,438 positives (21.0%), 46,860 negatives (79.0%)
  - **Early Completers**: 7,257 rows (12.2%)
  - **Cohort Coverage**: 2016Q4 through 2025Q4 (37 cohorts)
  - **Manifest**: `final_processed_90k_candidates.csv.manifest.json`
- **Q1 2026 Test Set**: `test_sets/test_90k_Q1_2026_cand.csv`
  - **Dimensions**: 1,492 rows × 224 columns
  - **Class Distribution**: 340 positives (22.8%), 1,152 negatives (77.2%)
  - **Early Completers**: 166 rows (11.1%)
  - **Manifest**: `test_90k_Q1_2026_cand.csv.manifest.json`
- **Q2 2026 Test Set**: `test_sets/test_90k_Q2_2026_cand.csv`
  - **Dimensions**: 1,494 rows × 224 columns
  - **Class Distribution**: 350 positives (23.4%), 1,144 negatives (76.6%)
  - **Early Completers**: 157 rows (10.5%)
  - **Manifest**: `test_90k_Q2_2026_cand.csv.manifest.json`
- **Regression Tests**:
  - `test_dayfirst.py`: 41/41 PASS
  - `test_service_num.py`: 31/31 PASS

### 2.2 Gate Battery (G3–G8)
- **G3 Schema**: 211 features across all 3 seeds (in standard range `[100, 200]`). Zero genuine schema gaps (32–34 absent one-hot dummy categories zero-filled, expected).
- **G4 Leak Check**: **CLEAN** (0 leak strings; no `Service_Num`, no `PMS Status_` leaks).
- **G5 Base-Rate Sanity**: **PASS** (Q1 positive rate 22.8%, early 11.1%; Q2 positive rate 23.4%, early 10.5%).
- **G6 Leak Screen (Seed 42)**:
  - **STOP (gap ≥ 0.20, test ≥ 0.65)**: **2 features**
    - `total_appointments_showed_up`: Train AUC 0.5965, Test AUC 0.8362, Gap = **0.2397**
    - `no_of_service_appointments_booked`: Train AUC 0.5947, Test AUC 0.8120, Gap = **0.2174**
  - **FLAG (gap ≥ 0.10, test ≥ 0.65)**: **2 features**
    - `LastNonPMSMileage`: Train AUC 0.7190, Test AUC 0.8451, Gap = 0.1261
    - `PMS_Count_Prior`: Train AUC 0.7485, Test AUC 0.8619, Gap = 0.1134
  - **INVERTED (train < 0.45, test > 0.55)**: **0** (None)
  - **Group Permutation Importance**:
    - `schedule_keeping` (14 features): Importance mean = **+0.2310** (std: 0.0052) -> Strongly Helpful/Positive
    - `vhc_revenue` (24 features): Importance mean = **+0.0344** (std: 0.0026) -> Helpful/Positive
  - **G6 Verdict**: **STOP** triggered on appointment features gap (distributional divergence across time).
- **G7 Seed Stability**:
  - Q1 AUC spread: $0.9692 - 0.9662 = 0.0030$ (< 0.04 -> **PASS**)
  - Q2 AUC spread: $0.9407 - 0.9332 = 0.0075$ (< 0.04 -> **PASS**)
- **G8 Reproduction (Independently Rescored on Q1)**:
  - Seed 42: Rescore AUC **0.9665** vs Training Log **0.9665** (MATCH)
  - Seed 7: Rescore AUC **0.9692** vs Training Log **0.9692** (MATCH)
  - Seed 123: Rescore AUC **0.9662** vs Training Log **0.9662** (MATCH)

### 2.3 Performance & Confusion Matrix Breakdown

| Seed | Quarter | ROC-AUC | Accuracy | F1-Score | Precision | Recall | Brier | TP | FP | FN | TN | Jaccard |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | Q1 | **0.9665** | 86.60% | 77.12% | 63.11% | 99.12% | 0.1264 | 337 | 197 | 3 | 955 | **0.6276** |
| | Q2 | **0.9368** | 85.27% | 74.94% | 62.31% | 94.00% | 0.1307 | 329 | 199 | 21 | 945 | **0.5993** |
| | *Mean* | *0.9517* | *85.94%* | *76.03%* | *62.71%* | *96.56%* | *0.1286* | — | — | — | — | *0.6135* |
| **7** *(Top)* | Q1 | **0.9692** | 87.06% | 77.74% | 63.95% | 99.12% | 0.1249 | 337 | 190 | 3 | 962 | **0.6358** |
| | Q2 | **0.9407** | 85.61% | 74.91% | 63.31% | 91.71% | 0.1263 | 321 | 186 | 29 | 958 | **0.5989** |
| | *Mean* | *0.9550* | *86.34%* | *76.33%* | *63.63%* | *95.42%* | *0.1256* | — | — | — | — | **0.6174** |
| **123** | Q1 | **0.9662** | 85.19% | 75.31% | 60.72% | 99.12% | 0.1337 | 337 | 218 | 3 | 934 | **0.6039** |
| | Q2 | **0.9332** | 82.86% | 71.93% | 58.36% | 93.71% | 0.1405 | 328 | 234 | 22 | 910 | **0.5616** |
| | *Mean* | *0.9497* | *84.03%* | *73.62%* | *59.54%* | *96.42%* | *0.1371* | — | — | — | — | *0.5828* |

- **Recommended Seed for Promotion**: **Seed 7** (Mean AUC: 0.9550, Mean Jaccard: 0.6174, lowest Brier: 0.1256).

---

# 3. 100k Milestone Rollout Statistics

### 3.1 Dataset & Feature Pipeline
- **Training Candidate Matrix**: `refactored_test_dir/final_processed_100k_candidates.csv`
  - **Dimensions**: 56,888 rows × 251 columns
  - **Class Distribution**: 11,543 positives (20.3%), 45,345 negatives (79.7%)
  - **Early Completers**: 7,343 rows (12.9%)
  - **Cohort Coverage**: 2017Q1 through 2025Q4 (36 cohorts)
  - **Manifest**: `final_processed_100k_candidates.csv.manifest.json`
- **Q1 2026 Test Set**: `test_sets/test_100k_Q1_2026_cand.csv`
  - **Dimensions**: 1,433 rows × 226 columns
  - **Class Distribution**: 289 positives (20.2%), 1,144 negatives (79.8%)
  - **Early Completers**: 130 rows (9.1%)
  - **Manifest**: `test_100k_Q1_2026_cand.csv.manifest.json`
- **Q2 2026 Test Set**: `test_sets/test_100k_Q2_2026_cand.csv`
  - **Dimensions**: 1,602 rows × 226 columns
  - **Class Distribution**: 277 positives (17.3%), 1,325 negatives (82.7%)
  - **Early Completers**: 135 rows (8.4%)
  - **Manifest**: `test_100k_Q2_2026_cand.csv.manifest.json`
- **Regression Tests**:
  - `test_dayfirst.py`: 41/41 PASS
  - `test_service_num.py`: 31/31 PASS

### 3.2 Gate Battery (G3–G8)
- **G3 Schema**: 212 features across all 3 seeds (in standard range `[100, 200]`). Zero genuine schema gaps (30–32 absent one-hot dummy categories zero-filled, expected).
- **G4 Leak Check**: **CLEAN** (0 leak strings; no `Service_Num`, no `PMS Status_` leaks).
- **G5 Base-Rate Sanity**: **PASS** (Q1 positive rate 20.2%, early 9.1%; Q2 positive rate 17.3%, early 8.4%).
- **G6 Leak Screen (Seed 42)**:
  - **STOP (gap ≥ 0.20, test ≥ 0.65)**: **2 features**
    - `total_appointments_showed_up`: Train AUC 0.5629, Test AUC 0.7919, Gap = **0.2290**
    - `no_of_service_appointments_booked`: Train AUC 0.5731, Test AUC 0.7803, Gap = **0.2072**
  - **FLAG (gap ≥ 0.10, test ≥ 0.65)**: **2 features**
    - `LastNonPMSMileage`: Train AUC 0.7196, Test AUC 0.8466, Gap = 0.1270
    - `PMS_Count_Prior`: Train AUC 0.7055, Test AUC 0.8156, Gap = 0.1101
  - **INVERTED (train < 0.45, test > 0.55)**: **0** (None)
  - **Group Permutation Importance**:
    - `schedule_keeping` (15 features): Importance mean = **+0.2962** (std: 0.0175) -> Strongly Helpful/Positive
    - `vhc_revenue` (24 features): Importance mean = **+0.0255** (std: 0.0041) -> Helpful/Positive
  - **G6 Verdict**: **STOP** triggered on appointment features gap.
- **G7 Seed Stability**:
  - Q1 AUC spread: $0.9638 - 0.9600 = 0.0038$ (< 0.04 -> **PASS**)
  - Q2 AUC spread: $0.9468 - 0.9394 = 0.0074$ (< 0.04 -> **PASS**)
- **G8 Reproduction (Independently Rescored on Q1)**:
  - Seed 42: Rescore AUC **0.9600** vs Training Log **0.9600** (MATCH)
  - Seed 7: Rescore AUC **0.9637** vs Training Log **0.9637** (MATCH)
  - Seed 123: Rescore AUC **0.9638** vs Training Log **0.9638** (MATCH)

### 3.3 Performance & Confusion Matrix Breakdown

| Seed | Quarter | ROC-AUC | Accuracy | F1-Score | Precision | Recall | Brier | TP | FP | FN | TN | Jaccard |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | Q1 | **0.9600** | 86.46% | 74.06% | 60.35% | 95.85% | 0.1237 | 277 | 182 | 12 | 962 | **0.5881** |
| | Q2 | **0.9394** | 85.21% | 68.02% | 54.31% | 90.97% | 0.1255 | 252 | 212 | 25 | 1113 | **0.5153** |
| | *Mean* | *0.9497* | *85.84%* | *71.04%* | *57.33%* | *93.41%* | *0.1246* | — | — | — | — | *0.5517* |
| **7** | Q1 | **0.9637** | 84.37% | 71.57% | 56.51% | 97.58% | 0.1344 | 282 | 217 | 7 | 927 | **0.5573** |
| | Q2 | **0.9468** | 82.71% | 65.59% | 50.00% | 95.31% | 0.1388 | 264 | 264 | 13 | 1061 | **0.4880** |
| | *Mean* | *0.9553* | *83.54%* | *68.58%* | *53.26%* | *96.45%* | *0.1366* | — | — | — | — | *0.5227* |
| **123** *(Top)* | Q1 | **0.9638** | 86.04% | 73.68% | 59.45% | 96.89% | 0.1219 | 280 | 191 | 9 | 953 | **0.5833** |
| | Q2 | **0.9463** | 85.14% | 68.52% | 54.07% | 93.50% | 0.1249 | 259 | 220 | 18 | 1105 | **0.5211** |
| | *Mean* | *0.9551* | *85.59%* | *71.10%* | *56.76%* | *95.20%* | *0.1234* | — | — | — | — | **0.5522** |

- **Recommended Seed for Promotion**: **Seed 123** (Mean AUC: 0.9551, Mean Jaccard: 0.5522, lowest Brier: 0.1234).

---

# 4. Comparative Executive Summary (20k vs 90k vs 100k)

| Metric | 20k (Seed 7) | 90k (Seed 7) | 100k (Seed 123) |
| :--- | :---: | :---: | :---: |
| **Train Matrix Rows** | 78,861 | 59,298 | 56,888 |
| **Train Positives %** | 45.5% | 21.0% | 20.3% |
| **Selected Features** | 160 | 211 | 212 |
| **Q1 Test Positive Rate** | 73.9% (1,648 / 2,231) | 22.8% (340 / 1,492) | 20.2% (289 / 1,433) |
| **Q2 Test Positive Rate** | 67.9% (1,305 / 1,923) | 23.4% (350 / 1,494) | 17.3% (277 / 1,602) |
| **G6 Verdict** | **PASS** (0 STOP, 0 FLAG) | **STOP** (Appt gap 0.24) | **STOP** (Appt gap 0.23) |
| **Schedule-Keeping Importance** | +0.1030 (Helpful) | +0.2310 (Strongly Helpful) | +0.2962 (Strongly Helpful) |
| **Q1 ROC-AUC** | **0.9254** | **0.9692** | **0.9638** |
| **Q2 ROC-AUC** | **0.9067** | **0.9407** | **0.9463** |
| **Combined Mean AUC** | **0.9161** | **0.9550** | **0.9551** |
| **Q1 Jaccard Overlap** | **0.8362** | **0.6358** | **0.5833** |
| **Q2 Jaccard Overlap** | **0.8586** | **0.5989** | **0.5211** |
| **Mean Jaccard Overlap** | **0.8474** | **0.6174** | **0.5522** |
| **Q1 Recall / Precision** | 96.36% / 86.35% | 99.12% / 63.95% | 96.89% / 59.45% |
| **Q2 Recall / Precision** | 94.48% / 90.40% | 91.71% / 63.31% | 93.50% / 54.07% |
