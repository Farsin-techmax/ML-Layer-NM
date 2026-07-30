# PMS Turn-Up Prediction — ML Layer

Automotive after-sales ML for a Nissan dealership network.  
For each mileage milestone (20k / 30k / … / 100k km), predict whether a vehicle will turn up for that Periodic Maintenance Service (PMS).  
One PyTorch binary classifier per milestone.

---

## Repository layout

```
.
├── features.py                        # ~60 shared feature-derivation functions
├── pmstrainfeatureeng_refactored.py   # Feature-engineering orchestrator (CLI)
├── retrain.py                         # Per-milestone model trainer (CLI)
├── milestone_features.py              # Extra per-VIN interval features (months / mileage between milestones)
├── prepare_test_set.py                # Build a held-out test cohort for a given quarter window
├── eval_test_metrics.py               # Evaluate retrained models on Q1/Q2 test sets
├── eval_alan_tests.py                 # Evaluate legacy models on tests_alan/ test sets
├── run_all.py                         # Orchestrator — trains + plots all milestones (⚠️ partially broken)
├── prepare_scripts.py                 # One-shot codegen helper (⚠️ source templates deleted)
├── fix_features.py                    # One-shot patcher for features.py (patches likely applied)
│
├── pmstrainfeatureEng_6.py            # Legacy monolith — training feature-eng
├── predservicemil_4.py                # Legacy monolith — prediction feature-eng (executes at import)
├── training_run.py                    # Legacy scratch — unstructured 70k trainer (no main())
├── prediction_run.py                  # Legacy scratch — unstructured 70k predictor (no main())
├── train_milestone.py                 # Legacy argv trainer over monolith output
│
├── features.py                        # Shared module imported by refactored + legacy scripts
├── requirements.txt                   # Core Python deps (incomplete — see below)
├── CLAUDE.md                          # Full project context & gotchas
├── DATA.md                            # Data file vintage documentation
├── AGENTS.md                          # Points to CLAUDE.md
│
├── data/                  (gitignored) Raw CSV / XLSX input files
├── models/                (gitignored) Trained model artifacts
├── validatecode/          (gitignored) Intermediate debug CSVs
├── predictions/           (gitignored) Prediction output CSVs
├── cache/                              Processing cache
├── legacy/                (gitignored) Earliest script copies
├── refactored_test_dir/   (gitignored) Refactored pipeline outputs & analysis
├── test_sets/                          Prepared test cohort CSVs
├── tests_alan/                         Legacy test data (Q1/Q2 per milestone)
├── venv/                  (gitignored) In-repo virtual environment
└── feature/pytorch-refactor/           Experimental PyTorch refactor work
```

---

## Target definition

**`TargetFlag == 1`** = the vehicle's last PMS *is* the target milestone (it completed it, ever).  
**`TargetFlag == 0`** = pending (`Service_Num < milestone`).  
No date condition on positives — the quarter only selects which pending cohort is sampled as zeros. Vehicles already past the milestone are dropped by the `dfafter` filter.

---

## Two pipelines

### Current pipeline (`features.py`-based)

Uses the shared `features.py` module (~60 pure feature-derivation functions). Both the feature-engineering and training scripts `from features import *`.

#### 1. Feature engineering

```bash
python pmstrainfeatureeng_refactored.py <year> <quarter> <milestone> [--train-cutoff YYYY-MM-DD]
```

Key orchestration:
- `prepare_pms_datasets()` — labels; tunes the date window until positive rate ≈ 0.35–0.45
- `process_service_data()` — merges per-VIN feature tables, one-hot encodes, runs `IterativeImputer(Lasso)`
- `feature_sel(cumulative_threshold=0.85)` — mutual-information selection + hard-drops leaky columns

**Output:** `refactored_test_dir/final_processed_{m}k.csv` + selected features list.

#### 2. Train one milestone

```bash
python retrain.py <milestone>
# Example: python retrain.py 20
```

Pipeline:
1. Loads `traindataq1_q2/finalmerged{m}k{Q1,Q2…}.csv`
2. Temporal cutoff: `Next{m}K_Due <= 2025-12-31`
3. Adds `has_x` milestone completion flags (derived from `Service_Num`)
4. Drops leak columns (`Service_Num`, `PMS_Delay`, `Vehicle Age`, `Current Age`, etc.)
5. Intersects train features with test-set schema to avoid zero-fill at inference
6. Drops unstable Gower cluster one-hots (`*_Cluster_*`)
7. Conditionally drops RFM segment one-hots (for milestone 60k — see Gotchas)
8. `IsolationForest(contamination=0.005)` outlier removal
9. Chronological 80/20 train/val split
10. `SimpleImputer(median)` → `RobustScaler`
11. `BinaryClassifier` (64→32→1, BatchNorm, ReLU, Dropout 0.2)
12. BCE(pos_weight) + 0.5·Brier + label smoothing (0.1)
13. AdamW (lr=3e-4, weight_decay=1e-2), ReduceLROnPlateau, early stopping (patience 10)

**Artifacts → `models/models_alan/{m}k/`:**
- `best_model.pt` — trained PyTorch state dict
- `scaler.joblib` — fitted RobustScaler
- `imputer.joblib` — fitted SimpleImputer
- `selected_features.json` — feature list used for training
- `metrics_{m}k_Q1_Test.json`, `metrics_{m}k_Q2_Test.json` — eval metrics

#### 3. Extra milestone interval features

```bash
python milestone_features.py <milestone>
# Example: python milestone_features.py 40
```

`MilestoneHistory(m).features_for(vins, cutoffs)` — computes months and mileage between consecutive completed milestones from raw service history, each row cut off at its own `Next{m}K_Due`.

#### 4. Build a test set

```bash
python prepare_test_set.py --milestone 20 --window-start 2026-01-01 --window-end 2026-03-31 --train-cutoff 2025-12-31
```

Identifies vehicles due for the milestone in the target window. Assigns ground truth from actual service history.

#### 5. Evaluate models

```bash
# All milestones, Q1 + Q2
python eval_test_metrics.py

# Single milestone
python eval_test_metrics.py 30
```

Mirrors `retrain.py`'s `evaluate_test()` exactly (same feature derivation, same `has_x` flags, 0.5 threshold). Prints confusion matrix breakdown with TP/TN/FP/FN, writes `retrained_test_metrics_final.csv`.

---

### Legacy / monolith pipeline

Pre-`features.py` scripts with heavy code duplication.

| Script | Role | Notes |
|--------|------|-------|
| `pmstrainfeatureEng_6.py` | Training feature-eng monolith | Has `main()`, ~124k bytes |
| `predservicemil_4.py` | Prediction feature-eng monolith | **Executes at import** — no `main()` guard |
| `train_milestone.py` | argv trainer over monolith output | `python train_milestone.py <milestone>` |
| `training_run.py` | Unstructured 70k trainer scratch | No `main()`, hardcoded 70k paths |
| `prediction_run.py` | Unstructured 70k predictor scratch | No `main()`, hardcoded 70k paths |
| `eval_alan_tests.py` | Legacy model evaluator | Uses `ServicePredictionNN` (dropout 0.5), `StandardScaler` |
| `legacy/` | Earliest copies of scripts | gitignored |

Legacy model architecture: `ServicePredictionNN` (64→32→1, dropout 0.5, `StandardScaler`).  
Artifacts in `models/{m}k/training/` (`best_nn_model.pt`, `imputer.pkl`, `scaler.pkl`).

---

## Feature families (from `features.py`)

| Family | Key functions | Description |
|--------|--------------|-------------|
| **Parsing & Utilities** | `extract_k1`, `extract_kk`, `extract_k`, `derive_servcode` | Parse service descriptions → numeric service numbers |
| **PMS / NPMS Date** | `derive_pms_features1`, `derive_npms_features` | Months since first/last PMS and NPMS per VIN |
| **Mileage** | `derive_pms_mileage_features`, `derive_npms_mileage_features` | Average mileage intervals with single-record adjustments |
| **Service Intervals** | `derive_pms_service_intervals`, `adjust_service_intervals` | Monthly intervals normalized by mileage, multiplied by distance-to-target |
| **NPMS Aggregates** | `derive_npms_features2` | NPMS counts, revenue, frequency metrics |
| **Branch** | `branch_visit_features`, `branch_diversity_features`, `calculate_bodyshop_count` | Top-N branch pivots, unique branch count, bodyshop visits |
| **Revenue** | `compute_service_features`, `calculate_revenue_spend` | PMS/NonPMS revenue aggregates per VIN |
| **Appointments** | `map_appointments_to_services`, `derive_appointment_show_features`, `compute_late_appointment_metrics` | Show/no-show, late, median delay, last status |
| **Complaints** | `get_non_pms_events`, `get_last_nonpms_mileage` | OHE of last non-PMS event type, revenue, mileage |
| **VHC** | `vhcpreparation`, `map_vhc_history`, `backfill_vhc_leakage_safe` | Parts/revenue/criticality KPIs, conversion & risk metrics |
| **Clustering** | `hierarchical_gower_clustering`, `map_cluster` | Gower-distance clustering for high-cardinality categoricals |
| **Imputation** | `IterativeImputer(Lasso)` in orchestrator | Final numeric matrix production |

---

## Data files

All data lives in `data/` (gitignored). See [DATA.md](DATA.md) for full inventory.

**Two snapshots exist side-by-side:**

| Snapshot | Example files | Used by |
|----------|--------------|---------|
| **Legacy** | `EDA - Q3 2025.csv`, `Service History Q1 - 2026.csv` | Monolith scripts |
| **New (Q2-2026)** | `EDA_Q2-2026.csv`, `Service_History_Q2-2026.csv` | Refactored pipeline (partially — see warning) |

> [!WARNING]
> The refactored pipeline currently **mixes vintages**: `pmstrainfeatureeng_refactored.py` hardcodes legacy EDA + Service History but new RFM/Appointments/Digital/VHC. Check which vintage each hardcoded path resolves to before trusting feature-eng output.

Other common inputs: `Service Code Desc.csv`, `Appoinment - Showed Up.csv`, `No Show VINs.csv`.

---

## Environment setup

```bash
cd c:\Techmax\cwf\code
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> [!IMPORTANT]
> `requirements.txt` is **incomplete**. Also install:
> ```bash
> pip install torch joblib matplotlib shap openpyxl fastparquet
> ```

**Always run from repo root** (`c:\Techmax\cwf\code`) — all data paths are hardcoded relative to cwd.

`requirements.txt` contains:
```
pandas, numpy, scikit-learn, scipy, gower, duckdb, kneed, python-dateutil
```

---

## Quick-start workflow

```bash
# 1. Feature engineering for a milestone
python pmstrainfeatureeng_refactored.py 2026 Q1 40

# 2. Train that milestone
python retrain.py 40

# 3. Evaluate on held-out test sets
python eval_test_metrics.py 40

# 4. (Optional) Build a fresh test cohort
python prepare_test_set.py --milestone 40 --window-start 2026-01-01 --window-end 2026-03-31 --train-cutoff 2025-12-31
```

---

## Known gotchas

> [!CAUTION]
> **`Service_Num` is an exact label leak**: `Service_Num == milestone` ⟺ `TargetFlag == 1`, zero exceptions. It and related columns (`PMS_Delay`, `Vehicle Age`, `Current Age`, `Vehicle Lifetime in Months`, `Vehicle_Key_Actual_Service`) are dropped in both `feature_sel()` and `retrain.py`. **Keep those guards when editing.**

- **`has_x` flags inherit the leak**: Derived from `Service_Num` (stamped only after turn-up), so every training positive gets `has_x = 1`. Real prediction sets never have `Service_Num == milestone`, so `has_x`-driven test metrics **will not transfer** to production. Known & accepted. `HAS_X_EXCLUDE = {80}` because 80k's Q2 file records `Service_Num = 0` for all 651 positives.

- **Train/test schema mismatch** is the dominant cause of bad metrics — missing columns get zero-filled. `retrain.py` guards this by intersecting training columns with test-set columns.

- **Gower cluster one-hots (`*_Cluster_*`)** are not stable across feature-eng runs (cluster "4" in train ≠ cluster "4" in test). `retrain.py` drops them.

- **RFM segments encode cohort vintage, not behaviour** — recency buckets from a single fixed snapshot. Training positives land in Lost/Hibernating, test positives in Potential Loyalist/Promising. Caused 60k's 92.0 (Q1) vs 73.7 (Q2) split. `DROP_RFM = {60}`.

- **Broken orchestrators**: `prepare_scripts.py` expects `retrain_20k.py` (deleted) and `run_all.py` calls `generate_eval_plots.py` (deleted). Edit `retrain.py` directly; do not trust those orchestrators.

- **Data directory expectations**: `traindataq1_q2/` and `testdataq1_q2/` may sit under `tests_alan/` but `retrain.py` / `eval_test_metrics.py` expect them at repo root. Check/symlink before running.

- **Legacy eval-on-train quirk**: The `train_nn_50k…100k` scripts pointed `TEST_DATA_PATH` at their own training CSV — those 50k+ metrics are **not** held-out.

- **Hardcoded filenames everywhere**: Both legacy and refactored scripts use many hardcoded CSV paths. Adapt paths or ensure the working directory contains the expected files.

- **Performance**: Groupby/merge operations and the Gower distance matrix (O(n²)) can be slow on large VIN counts. Run on a machine with sufficient memory.

- **Leakage guards**: The code avoids leakage by filtering with `Service_Num < target` or `Service_Date <= filter_date`. **Preserve these rules if refactoring.**

---

## Gitignored content

`data/`, `models/`, `validatecode/`, `predictions/`, `legacy/`, `refactored_test_dir/`, `venv/`, `*.csv`, `*.parquet`, `*.pkl`, `*.log` — only source code and documentation are tracked.

---

## Related documentation

| File | Contents |
|------|----------|
| [CLAUDE.md](CLAUDE.md) | Full project context, architecture, gotchas (canonical reference) |
| [DATA.md](DATA.md) | Data file vintages and known mismatches |
| [AGENTS.md](AGENTS.md) | Points to CLAUDE.md |
| `refactored_test_dir/mi_scores_analysis.md` | Feature-selection / MI scores analysis |
