# Multi-Milestone PMS Training & Prediction Plan

Predict all vehicles expected to visit for PMS in **Q3 + Q4 2026** across milestones **[20k, 30k, 40k, 50k, 60k, 70k, 80k, 90k, 100k]**.

## Data Leakage Analysis

> [!CAUTION]
> **Hardcoded dates throughout both pipelines.** Every leakage risk below must be fixed before execution.

### Leakage Points Found

| # | File | Line(s) | Issue | Fix |
|---|---|---|---|---|
| 1 | [predservicemil_4_refactored.py](file:///d:/techmax/nurture-mate/ML_layer/code/predservicemil_4_refactored.py#L484) | 484 | `filterdate = '2025-09-30'` hardcoded | Must be `'2026-06-30'` (end of Q2 2026 = latest data we have) |
| 2 | [predservicemil_4_refactored.py](file:///d:/techmax/nurture-mate/ML_layer/code/predservicemil_4_refactored.py#L778-L779) | 778-779 | Prediction window `'2025-06-30'` to `'2025-09-30'` | Must be `'2026-06-30'` to `'2026-12-31'` (Q3+Q4 2026) |
| 3 | [predservicemil_4_refactored.py](file:///d:/techmax/nurture-mate/ML_layer/code/predservicemil_4_refactored.py#L223) | 223 | NonPMS window hardcoded `'2025-12-31'` | Must be `'2026-12-31'` |
| 4 | [pmstrainfeatureEng_6.py](file:///d:/techmax/nurture-mate/ML_layer/code/pmstrainfeatureEng_6.py) main() | ~L2790 | `filter_date = '2025-06-30'`, `year=2025`, `selected_quarter="Q3"` | Must be `'2026-06-30'`, `year=2026`, `selected_quarter="Q2"` |
| 5 | [training_run.py](file:///d:/techmax/nurture-mate/ML_layer/code/training_run.py#L1) | 1 | `df[df['Next70K_Due'] < '2025-07-01']` — milestone-specific column, date hardcoded | Must parameterize per target milestone, use `'2026-07-01'` |
| 6 | [prediction_run.py](file:///d:/techmax/nurture-mate/ML_layer/code/prediction_run.py#L1) | 1 | `pd.read_csv('70kPMSTestDataQ3.csv')` — filename hardcoded to 70k | Must parameterize per milestone |
| 7 | Service history filter | pred L494, L588 | `serv1.query("Service_Date <= @filterdate")` — **correct pattern** ✅ | No fix needed — already uses filterdate |
| 8 | Service_Num filter | pred L601 | `serv1.query(f"Service_Num <= {last_service_code}")` — **correct** ✅ | Prevents future milestone data from leaking into lower milestone features |

### Key Leakage Rule (Already Enforced)
The pipeline already enforces `Service_Num <= target` and `Service_Date <= filterdate` on service history. This prevents future PMS data from leaking into feature engineering. **This is correct and must not be changed.**

---

## Missing Dependencies

> [!IMPORTANT]
> The prediction pipeline needs cluster mapping files produced by the training pipeline.

| Dependency | Produced By | Consumed By |
|---|---|---|
| `validatecode/Model_clusters_{target}.csv` | Training pipeline (Gower clustering) | Prediction pipeline L510 |
| `validatecode/Variant_clusters_{target}.csv` | Training pipeline | Prediction pipeline L511 |
| `validatecode/Nationality_clusters_{target}.csv` | Training pipeline | Prediction pipeline L512 |
| `indfeats` (feature list) | Training pipeline feature selection | Prediction run L11 |
| `scaler` (RobustScaler) | Training run L33-35 | Prediction run L19 |
| `model` (trained PyTorch model) | Training run L352 | Prediction run L41 |

**→ Training MUST run before prediction for each milestone.**

---

## Proposed Execution Strategy

> [!WARNING]
> Running 9 milestones × (training ~15min + prediction ~20min) = ~5+ hours of compute. This is NOT something to run all in one agent turn. Below is how to split it.

### Phase 1: Create Orchestrator Script
Build a single `run_all_milestones.py` that:
1. Loops over `[20, 30, 40, 50, 60, 70, 80, 90, 100]`
2. For each target milestone:
   - Calls the **training pipeline** (`pmstrainfeatureEng_6.py` main) with correct params
   - Runs **training_run.py** logic (outlier removal, scaling, model training)
   - Saves: model weights, scaler, feature list → `models/{target}k/`
   - Calls the **prediction pipeline** (`predservicemil_4_refactored.py` logic) with correct dates
   - Runs **prediction_run.py** logic (inference)
   - Saves: prediction CSV with probabilities → `predictions/{target}k_Q3Q4_2026.csv`
3. Final step: concatenates all milestone predictions into `predictions/all_milestones_Q3Q4_2026.csv`

### Phase 2: Date/Config Parameterization
All hardcoded dates get replaced with config constants:

```python
# Config for Q3+Q4 2026 prediction run
FILTER_DATE = '2026-06-30'          # Last day of available data (Q2 2026)
PRED_WINDOW_START = '2026-06-30'    # Start of prediction window
PRED_WINDOW_END = '2026-12-31'      # End of prediction window (Q3+Q4)
TRAIN_QUARTER = 'Q2'                # Quarter of the training data
TRAIN_YEAR = 2026                   # Year
MILESTONES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
```

### Phase 3: Execute (YOU run this, not me)
Since each milestone takes ~30+ min and the full run is ~5 hours:

**Option A (Recommended):** I generate the script, you run it overnight via:
```bash
python run_all_milestones.py > run_log.txt 2>&1
```

**Option B:** We run milestones one at a time interactively (9 separate runs).

---

## Proposed Changes

### Orchestration & Config

#### [NEW] [run_all_milestones.py](file:///d:/techmax/nurture-mate/ML_layer/code/run_all_milestones.py)
- Config constants (dates, milestones, file paths)
- Loop over milestones calling train → predict
- Save all artifacts per milestone under `models/` and `predictions/`
- Final concatenation into single output

#### [NEW] `models/` directory
- Per-milestone subdirs: `models/20k/`, `models/30k/`, etc.
- Each contains: `model.pt`, `scaler.pkl`, `feature_list.json`

#### [NEW] `predictions/` directory
- Per-milestone CSVs: `predictions/20k_Q3Q4_2026.csv`, etc.
- Final combined: `predictions/all_milestones_Q3Q4_2026.csv`

---

### Training Pipeline

#### [MODIFY] [pmstrainfeatureEng_6.py](file:///d:/techmax/nurture-mate/ML_layer/code/pmstrainfeatureEng_6.py)
- `main()` accepts `last_service_code`, `filter_date`, `quarter`, `year` as parameters instead of hardcoded values
- Returns `result_df` so the orchestrator can use it

---

### Prediction Pipeline

#### [MODIFY] [predservicemil_4_refactored.py](file:///d:/techmax/nurture-mate/ML_layer/code/predservicemil_4_refactored.py)
- Wrap bottom-level script code (L484-970) into a `def run_prediction(last_service_code, filterdate, pred_window_start, pred_window_end, ...)` function
- Replace all hardcoded dates with parameters
- Return `finalbase` instead of only writing CSV

---

## Output Format

Final CSV columns per milestone:
```
VIN | Vehicle Key | Customer ID | Last Service Date - PMS | Next{X}K_Due | Milestone | Predicted_Prob | Predicted_Class | ... (all features)
```

Combined CSV adds a `Milestone` column (20, 30, ..., 100) for easy filtering.

---

## Verification Plan

### Automated Checks (built into orchestrator)
- Assert `filterdate < pred_window_start` (no future data in features)
- Assert max `Service_Date` in training data ≤ `filterdate`
- Assert no VIN appears in multiple milestones for the same target
- Log row counts per milestone
- Log positive/negative class ratios per milestone

### Manual Verification
- Spot-check a few VINs: verify their `Last Service - PMS` matches the milestone
- Verify prediction probability distribution looks reasonable (not all 0 or all 1)

---

## Open Questions

> [!IMPORTANT]
> 1. **Appointment data files**: The prediction pipeline expects `Appoinment - Showed Up.csv` and `No Show VINs.csv` — these were NOT in the Q2-2026 xlsx set. Do you have these files, or should we derive them from `Appointments_Q2-2026.xlsx`?

> [!IMPORTANT]
> 2. **Your milestone list has 60 twice**: `[20,30,40,50,60,60,80,90,100]` — I'm treating this as `[20,30,40,50,60,70,80,90,100]`. Confirm?

> [!IMPORTANT]
> 3. **Franchise filter**: The prediction_run.py filters `Franchise_NISSAN == 1`. Should all milestones be NISSAN-only, or do some milestones cover other franchises?

> [!IMPORTANT]
> 4. **Execution preference**: Option A (I build the script, you run overnight) or Option B (we run one milestone at a time)?
