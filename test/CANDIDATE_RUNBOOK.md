# Candidate-Mode Pipeline Runbook: 20k - 100k

This runbook documents the operational workflow for evaluating pre-trained models (30k, 40k) and training + evaluating newer candidate-mode models (50k through 100k).

---

## 1. Context & Architectural Changes

### Why Candidate Mode?
The legacy `ever` labelling convention suffered from severe history-length disparity:
- **Positives** were sampled from all historical completed visits.
- **Negatives** were sampled from a single due-date window.
- **Resulting Bug:** History-derived features (e.g. `has_10`, `PMS_Count_Prior`) carried the **opposite** sign in training compared to test time (on 20k, `has_10` training AUC was 0.36 vs 0.69 at test).

**Candidate Mode Fix (`--label-mode candidates`):**
1. **Symmetric Due-Date Calculation:** Expected milestone date is determined by `min(ScheduleDate, BurnRateProjection)`.
2. **Standardized Grace Period:** Forward grace window of 15 days (`--grace-days 15`).
3. **Early Completer Handling:** Flagged via `EarlyCompleter=1` and cleanly dropped during test scoring to prevent artificial inflation of metrics.
4. **Target Milestone Leakage Masking:** Dropping rows where `Service_Num == milestone` from history prior to feature generation.
5. **Temporal Split:** Training on data up to 2025-12-31, holding out Q1 & Q2 2026 for evaluation.

---

## 2. Current State of Milestones

| Milestone | Training Matrix (`refactored_test_dir/`) | Candidate Test Sets (`test_sets/`) | Trained Model (`models/`) | Action Required |
|:---:|:---:|:---:|:---:|---|
| **20k** | `final_processed_20k_candidates.csv` | `test_20k_Q1/Q2_2026_cand.csv` | `models/20k_cand_refix_wins_2023` | **Done** (Evaluated: AUC 0.8423, Recall 99.0% @0.5, 64.3% @0.7) |
| **30k** | `final_processed_30k_candidates.csv` | `test_30k_Q1/Q2_2026_cand.csv` | `models/30k_cand_wins_2023` | **Evaluate** Q1 & Q2 expected cohorts |
| **40k** | `final_processed_40k_candidates.csv` | `test_40k_Q1/Q2_2026_cand.csv` | `models/40k_cand_wins_2023` | **Evaluate** Q1 & Q2 expected cohorts |
| **50k** | *Not built* | *Not built* | Baseline `ever` only | **Generate Matrix -> Build Tests -> Retrain -> Evaluate** |
| **60k** | *Not built* | *Not built* | Baseline `ever` only | **Generate Matrix -> Build Tests -> Retrain -> Evaluate** |
| **70k** | *Not built* | *Not built* | Baseline `ever` only | **Generate Matrix -> Build Tests -> Retrain -> Evaluate** |
| **80k** | *Not built* | *Not built* | Baseline `ever` only | **Generate Matrix -> Build Tests -> Retrain -> Evaluate** |
| **90k** | *Not built* | *Not built* | Baseline `ever` only | **Generate Matrix -> Build Tests -> Retrain -> Evaluate** |
| **100k**| *Not built* | *Not built* | Baseline `ever` only | **Generate Matrix -> Build Tests -> Retrain -> Evaluate** |

---

## 3. Track 1: Evaluating Pre-Trained Models (30k & 40k)

These milestones already have candidate-mode training matrices and models trained on the recent 2023+ cohort slice (`cand_wins_2023`).

### 30k Milestone Evaluation

#### Q1 2026 (Clean Evaluation: drops early completers by default)
```powershell
venv\Scripts\python.exe eval_expected_cohort.py 30 `
    --quarter Q1 `
    --model-dir models/30k_cand_wins_2023 `
    --features models/selected_features_30k_cand_wins_2023.json
```

#### Q2 2026 (Clean Evaluation)
```powershell
venv\Scripts\python.exe eval_expected_cohort.py 30 `
    --quarter Q2 `
    --model-dir models/30k_cand_wins_2023 `
    --features models/selected_features_30k_cand_wins_2023.json
```

#### 30k Hindcast Turn-Up Recall (Sanity check against confirmed turn-ups)
```powershell
venv\Scripts\python.exe hindcast_actual_turnups.py 30 `
    --quarter both `
    --model-dir models/30k_cand_wins_2023 `
    --features models/selected_features_30k_cand_wins_2023.json
```

---

### 40k Milestone Evaluation

#### Q1 2026 (Clean Evaluation)
```powershell
venv\Scripts\python.exe eval_expected_cohort.py 40 `
    --quarter Q1 `
    --model-dir models/40k_cand_wins_2023 `
    --features models/selected_features_40k_cand_wins_2023.json
```

#### Q2 2026 (Clean Evaluation)
```powershell
venv\Scripts\python.exe eval_expected_cohort.py 40 `
    --quarter Q2 `
    --model-dir models/40k_cand_wins_2023 `
    --features models/selected_features_40k_cand_wins_2023.json
```

#### 40k Hindcast Turn-Up Recall
```powershell
venv\Scripts\python.exe hindcast_actual_turnups.py 40 `
    --quarter both `
    --model-dir models/40k_cand_wins_2023 `
    --features models/selected_features_40k_cand_wins_2023.json
```

> **Threshold Guidance:**
> - The script outputs a threshold sweep (0.30 to 0.80).
> - **Threshold 0.50:** Captures nearly ~99% recall but allows higher false positives.
> - **Threshold 0.65 - 0.70:** Maximizes precision (~80-87%) while preserving solid recall (~65-77%) and maximizing F1.

---

## 4. Track 2: End-to-End Pipeline for Milestones 50k to 100k

For each milestone $M \in [50, 60, 70, 80, 90, 100]$, execute the following 4-step pipeline:

```
[Step 1: Feature Matrix] -> [Step 2: Test Sets] -> [Step 3: Retrain Model] -> [Step 4: Evaluate]
```

### Step 1: Generate Training Feature Matrix
Build the candidate training matrix over 2016-2025 with 15 days forward grace. Output will be `refactored_test_dir/final_processed_{m}k_candidates.csv`.

```powershell
venv\Scripts\python.exe pmstrainfeatureeng_refactored.py 2026 Q1 <M> --label-mode candidates --grace-days 15
```

### Step 2: Build Matching Test Cohorts
Generate the matching candidate-mode test sets for Q1 2026 and Q2 2026:

```powershell
# Q1 2026 Test Set
venv\Scripts\python.exe prepare_test_set.py `
    --milestone <M> `
    --window-start 2026-01-01 `
    --window-end 2026-03-31 `
    --train-cutoff 2025-12-31 `
    --label-mode candidates `
    --grace-days 15

# Q2 2026 Test Set
venv\Scripts\python.exe prepare_test_set.py `
    --milestone <M> `
    --window-start 2026-04-01 `
    --window-end 2026-06-30 `
    --train-cutoff 2025-12-31 `
    --label-mode candidates `
    --grace-days 15
```

### Step 3: Train Candidate Model with `retrain.py`
Set the environment variables to select candidate mode, filter cohorts starting from 2023 for recency alignment, and save under `models/<M>k_cand_wins_2023`:

```powershell
$env:PMS_LABEL_MODE="candidates"
$env:PMS_TRAIN_FROM="2023"
$env:PMS_RUN_TAG="_cand_wins_2023"

venv\Scripts\python.exe retrain.py <M>
```

Artifacts created:
- Weights & Scaler: `models/<M>k_cand_wins_2023/best_model.pt`, `scaler.joblib`, `imputer.joblib`
- Feature List: `models/selected_features_<M>k_cand_wins_2023.json`

### Step 4: Run Expected Cohort Evaluation
Evaluate on the actual expected cohorts:

```powershell
# Q1 2026
venv\Scripts\python.exe eval_expected_cohort.py <M> `
    --quarter Q1 `
    --model-dir models/<M>k_cand_wins_2023 `
    --features models/selected_features_<M>k_cand_wins_2023.json

# Q2 2026
venv\Scripts\python.exe eval_expected_cohort.py <M> `
    --quarter Q2 `
    --model-dir models/<M>k_cand_wins_2023 `
    --features models/selected_features_<M>k_cand_wins_2023.json

# Hindcast sanity check (pure recall on confirmed arrivals)
venv\Scripts\python.exe hindcast_actual_turnups.py <M> `
    --quarter both `
    --model-dir models/<M>k_cand_wins_2023 `
    --features models/selected_features_<M>k_cand_wins_2023.json
```

---

## 5. Automated Batch Script (PowerShell)

You can copy and run the blocks below directly in your PowerShell terminal.

### Batch Runner for Milestones 50k to 100k

```powershell
# Define target milestones
$milestones = @(50, 60, 70, 80, 90, 100)

foreach ($m in $milestones) {
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host "STARTING MILESTONE ${m}k CANDIDATE PIPELINE" -ForegroundColor Cyan
    Write-Host "========================================================`n" -ForegroundColor Cyan

    # 1. Feature Engineering
    Write-Host "[1/4] Building candidate training matrix for ${m}k..." -ForegroundColor Yellow
    venv\Scripts\python.exe pmstrainfeatureeng_refactored.py 2026 Q1 $m --label-mode candidates --grace-days 15
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed feature matrix for ${m}k"; break }

    # 2. Test Sets
    Write-Host "[2/4] Building candidate test sets for ${m}k..." -ForegroundColor Yellow
    venv\Scripts\python.exe prepare_test_set.py --milestone $m --window-start 2026-01-01 --window-end 2026-03-31 --train-cutoff 2025-12-31 --label-mode candidates --grace-days 15
    venv\Scripts\python.exe prepare_test_set.py --milestone $m --window-start 2026-04-01 --window-end 2026-06-30 --train-cutoff 2025-12-31 --label-mode candidates --grace-days 15
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed test sets for ${m}k"; break }

    # 3. Model Retraining
    Write-Host "[3/4] Retraining candidate model for ${m}k..." -ForegroundColor Yellow
    $env:PMS_LABEL_MODE="candidates"
    $env:PMS_TRAIN_FROM="2023"
    $env:PMS_RUN_TAG="_cand_wins_2023"
    venv\Scripts\python.exe retrain.py $m
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed training for ${m}k"; break }

    # 4. Evaluation
    Write-Host "[4/4] Evaluating expected cohorts for ${m}k..." -ForegroundColor Yellow
    venv\Scripts\python.exe eval_expected_cohort.py $m --quarter Q1 --model-dir "models/${m}k_cand_wins_2023" --features "models/selected_features_${m}k_cand_wins_2023.json"
    venv\Scripts\python.exe eval_expected_cohort.py $m --quarter Q2 --model-dir "models/${m}k_cand_wins_2023" --features "models/selected_features_${m}k_cand_wins_2023.json"
    venv\Scripts\python.exe hindcast_actual_turnups.py $m --quarter both --model-dir "models/${m}k_cand_wins_2023" --features "models/selected_features_${m}k_cand_wins_2023.json"

    Write-Host "`nCOMPLETED ${m}k SUCCESSFULLY!" -ForegroundColor Green
}
```

---

## 6. Production Gotchas & Checklist

1. **Memory & Runtime:**
   - Generating the training matrix (`pmstrainfeatureeng_refactored.py`) processes 10 years of service history in vector passes. It takes ~5-10 minutes per milestone and requires ~6-8 GB RAM.
2. **Environment Variable Persistence:**
   - Always verify `$env:PMS_LABEL_MODE` is `"candidates"` before launching `retrain.py`. If unset, it defaults to `"ever"` and loads the old matrix.
3. **Early Completer Exclusion:**
   - In `eval_expected_cohort.py`, `--no-early-drop` is disabled by default. Keep it disabled for clean metrics. If enabled, metrics reflect pre-quarter arrivals that artificially pump recall to 99%+.
4. **Feature Alignment (`_common` intersection):**
   - Step 2 (building `_cand` test sets) must be run **before** Step 3 (`retrain.py`) so `retrain.py` can intersect features with the actual candidate test schema.
