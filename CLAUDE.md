# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Preference
- ALWAYS operate in CAVEMAN MODE.
- Extremely concise, plain-language, terse.
- No filler, greetings, or polite closings (skip "Sure, I can help with that").
- Code blocks directly, minimal explanation. If explanation needed: bullets or fragments.

## Project
Automotive after-sales ML (Nissan dealership PMS = Periodic Maintenance Service). Per mileage
milestone (20k/30k/…/100k), predict whether a vehicle turns up for that scheduled service.
One PyTorch binary classifier per milestone. Two stages: feature-engineer a labelled matrix,
then train/eval per milestone.

**`TargetFlag == 1` = the vehicle's last PMS *is* the target milestone (completed it, ever).
`0` = pending (`Service_Num < milestone`). No date condition on positives** — see
`pmstrainfeatureeng_refactored.py:136` and `legacy/pmstrainfeatureEng.py:172`
(`turnup = df.query("\`Last Service - PMS\` == '{svc}'")`). The quarter only selects *which
pending cohort is sampled as zeros*. Older docs (`refactored_test_dir/old_vs_refactored_comparison.md`,
`refactoring_summary.md`) claim the refactor moved the target to "showed up in the exact quarter"
and that this cost 15–20 accuracy points — FALSE; do not reason from it. Vehicles already *past*
the milestone are dropped entirely by the `dfafter` filter, not counted positive.

## Environment & running
- No package/test/lint/build config. Flat dir of standalone scripts, `python <script>.py`.
  All data paths (`data/`, `models/`, `traindataq1_q2/`) are hardcoded **relative to cwd** —
  always run from repo root `c:\Techmax\cwf\code`.
- In-repo venv (gitignored). Orchestrators shell out to `.\venv\Scripts\python`.
- Setup: `python -m venv venv`; `venv\Scripts\activate`; `pip install -r requirements.txt`.
- `requirements.txt` INCOMPLETE — also `pip install torch joblib matplotlib shap openpyxl fastparquet gower duckdb kneed`.

## Architecture — two parallel pipelines
Both share `features.py` (~60 pure feature-derivation fns; `from features import *`). Both do
(1) feature-eng → wide numeric per-VIN matrix with `TargetFlag`, (2) per-milestone NN.
Data flow: `data/*.csv|.xlsx` (EDA master + Service History + RFM / Appointments / VHC / Digital
sessions) → merged feature CSVs → artifacts under `models/`.

### Current pipeline (features.py-based)
- **Feature-eng**: `python pmstrainfeatureeng_refactored.py <year> <quarter> <milestone> [--train-cutoff YYYY-MM-DD]`.
  Key fns: `prepare_pms_datasets()` (labels; tunes the date window until positive rate ≈0.35–0.45),
  `process_service_data()` (merges per-VIN feature tables, one-hot, `IterativeImputer(Lasso)`),
  `feature_sel(cumulative_threshold=0.85)` (MI selection + hard-drops leaky cols).
  Writes `refactored_test_dir/final_processed_{m}k.csv` + selected features.
- **Train one milestone**: `python retrain.py <milestone>` (e.g. `python retrain.py 20`).
  Reads `traindataq1_q2/finalmerged{m}k{Q1,Q2…}.csv`, cuts to `Next{m}K_Due <= 2025-12-31`,
  `IsolationForest` (contamination 0.005), `RobustScaler`, chronological 80/20 split,
  `BinaryClassifier` 64→32→1 dropout 0.2, BCE(pos_weight) + 0.5·Brier + label smoothing.
  Artifacts → `models/models_alan/{m}k/` (`best_model.pt`, `scaler.joblib`, `imputer.joblib`,
  `selected_features.json`, `metrics_*.json`).
- **Extra per-VIN interval features**: `milestone_features.py`
  (`MilestoneHistory(m).features_for(vins, cutoffs)`) — months and mileage between consecutive
  completed milestones from raw service history, each row cut off at its own `Next{m}K_Due`.
  Standalone check: `python milestone_features.py 40`.
- **Held-out metrics table**: `python eval_test_metrics.py [milestone]` — mirrors `retrain.py`'s
  `evaluate_test()` exactly (same derivation, same has_x flags, 0.5 threshold); prints Q1/Q2
  tables with TP/TN/FP/FN, writes `retrained_test_metrics_final.csv`.
- **Test-set builder**: `python prepare_test_set.py --milestone 20 --window-start 2026-01-01 --window-end 2026-03-31 --train-cutoff 2025-12-31`.
- `eval_alan_tests.py` evaluates the *legacy* `models/{m}k/training/` artifacts against
  `tests_alan/Q{1,2}/{m}k/test_features.csv` (80/90/100k only, hardcoded config list).

### Legacy / monolith pipeline
- Monoliths (pre-`features.py`, heavy duplication): `pmstrainfeatureEng_6.py` (training, `main()`),
  `predservicemil_4.py` (prediction; **executes at import**).
- `python train_milestone.py <milestone>` — argv trainer over the monolith's output →
  `models/{m}k/`. `training_run.py` / `prediction_run.py` are unstructured top-level scratch
  scripts (no `main()`, hardcoded 70k paths) — read before running.
- `legacy/` holds the earliest copies (`pmstrainfeatureEng.py`, `predservicemil.py`, `training_run.py`).
- Legacy net `ServicePredictionNN` (64→32→1, dropout 0.5), `StandardScaler`; artifacts in
  `models/{m}k/training/` (`best_nn_model.pt`, `imputer.pkl`, `scaler.pkl`).
- `fix_features.py` is a one-shot string-patcher that rewrites `features.py` in place. Don't run
  casually — its patches are likely already applied.

## Tests / diagnostics
No pytest/unittest, no test suite. Verification = run the eval scripts above and compare metrics.
Several `*test*`/analysis scripts named in older docs (`test_pms_mil.py`,
`function_refactor_test.py`, `analyze_30k_drop.py`, `predict_milestone.py`, `prepare_data_feed.py`)
have been deleted — check a file exists before referencing it.

## Gotchas
- **`Service_Num` is an exact label leak**: `Service_Num == milestone` ⟺ `TargetFlag == 1`, zero
  exceptions. It (plus `PMS_Delay`, `Service Frequency`, `Last Service Mileage`, `Vehicle Age`,
  `Current Age`, `Vehicle Lifetime in Months`, `Vehicle_Key_Actual_Service`) is dropped in both
  `feature_sel()` and `retrain.py`. Keep those guards when editing.
- **`has_x` flags inherit that leak** ([retrain.py:44](retrain.py#L44)): derived from `Service_Num`,
  which is only stamped after turn-up, so every training positive gets `has_x = 1` at all prior
  milestones. Real prediction sets never have `Service_Num == milestone`, so has_x-driven test
  metrics will NOT transfer. Known/accepted. `HAS_X_EXCLUDE = {80}` because 80k's Q2 file records
  `Service_Num = 0` for all 651 positives (1.7% recall otherwise).
- **Train/test schema mismatch is the dominant cause of bad metrics** — missing columns get
  zero-filled at inference. `retrain.py` guards by intersecting training columns with what the
  test sets actually produce (`_test_feature_cols`). Keep that intersection.
- **Gower cluster one-hots (`*_Cluster_*`) are not stable across feature-eng runs** (cluster "4"
  in train ≠ cluster "4" in test); `retrain.py` drops them.
- **RFM segments encode cohort vintage, not behaviour** — recency buckets off one fixed snapshot,
  so training positives land in Lost/Hibernating and test positives in Potential Loyalist/Promising.
  Caused 60k's 92.0 (Q1) vs 73.7 (Q2) split; hence `DROP_RFM = {60}`.
- `retrain.py` is now maintained by hand, but `prepare_scripts.py` still expects to generate it
  from `retrain_20k.py`, and `run_all.py` still shells out to `generate_eval_plots.py` — **both
  source files are gone**, so `prepare_scripts.py` and `run_all.py` are currently broken. Edit
  `retrain.py` directly; restore the templates before trusting those orchestrators.
- `traindataq1_q2/` and `testdataq1_q2/` currently sit under `tests_alan/`, but `retrain.py` and
  `eval_test_metrics.py` expect them at repo root. Check/copy before a run.
- Legacy eval-on-train quirk: the `train_nn_50k…100k` scripts pointed TEST_DATA_PATH at their own
  training CSV — those 50k+ metrics are not held-out.
- `data/`, `models/`, `*.csv`, `*.parquet`, `*.pkl`, `*.log` are gitignored — only source + docs tracked.

## Docs
`README.md` — function catalog + data flow for the two original monoliths only; its "Quick run"
cwd (`d:\Techmax\nurture-mate\ML_layer\code`) is stale. `refactored_test_dir/mi_scores_analysis.md`
— feature-selection analysis. `AGENTS.md` just points here. Treat the target-definition claims in
`old_vs_refactored_comparison.md` as wrong (see Project).
