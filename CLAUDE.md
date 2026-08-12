# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Preference
- Work at an intermediate/staff software-engineer level, in simple mode.
- With each change or modification, make the user aware of it and teach them the steps.
- Ask clarifying questions often, to confirm the user has the context.
- No filler, greetings, or polite closings (skip "Sure, I can help with that").
- Before committing: describe the changes in plain language and commit only on approval.
- (Supersedes the old CAVEMAN MODE rule. `AGENTS.md` carries the same instructions.)

## Project
Automotive after-sales ML (Nissan dealership PMS = Periodic Maintenance Service). Per mileage
milestone (20k/30k/…/100k), predict whether a vehicle turns up for that scheduled service.
One PyTorch binary classifier per milestone. Two stages: feature-engineer a labelled matrix,
then train/eval per milestone.

### Labelling: intended vs actual — they differ. Know which one you are talking about.
**Intended (the goal, per `AGENTS.md`)**: `1` = the vehicle turned up for the milestone *in the
quarter it was expected*; `0` = it did not. Date-conditioned.

**Actual (what the code does today)**: `TargetFlag == 1` = the vehicle's last PMS *is* the target
milestone — it completed it, ever. `0` = pending (`Service_Num < milestone`). **No date condition
on positives** — see `pmstrainfeatureeng_refactored.py:136` and `legacy/pmstrainfeatureEng.py:172`
(`turnup = df.query("\`Last Service - PMS\` == '{svc}'")`). The quarter only selects *which pending
cohort is sampled as zeros*. Vehicles already *past* the milestone are dropped entirely by the
`dfafter` filter, not counted positive.

The gap is a known defect, not yet fixed — no code change has been made for it. It is also the root
of the `Service_Num` / `has_x` leak below: `Service_Num` is only stamped once a vehicle turns up, so
an "ever completed" positive carries its own label in its features. Moving to the intended
quarter-based rule would change every training matrix and every published metric; do not do it
silently as part of another task.

Older docs (`refactored_test_dir/old_vs_refactored_comparison.md`, `later/refactoring_summary.md`)
claim the refactor *already* moved the target to "showed up in the exact quarter" and that this cost
15–20 accuracy points — FALSE as a description of the code; do not reason from it.

## Environment & running
- No package/test/lint/build config. Flat dir of standalone scripts, `python <script>.py`.
  All data paths (`data/`, `models/`, `traindataq1_q2/`) are hardcoded **relative to cwd** —
  always run from repo root `c:\Techmax\cwf\code`.
- In-repo venv (gitignored). Orchestrators shell out to `.\venv\Scripts\python`.
- Setup: `python -m venv venv`; `venv\Scripts\activate`; `pip install -r requirements.txt`.
- `requirements.txt` INCOMPLETE — also `pip install torch joblib matplotlib shap openpyxl fastparquet gower duckdb kneed`.
- `features.py` writes a progress trace to `debug.txt` only when `PMS_DEBUG` is set
  (`$env:PMS_DEBUG=1`). The file is untracked; before 2026-08-12 it was written unconditionally and
  dirtied the working tree on every feature-eng run.

## Architecture — two parallel pipelines
Both share `features.py` (~60 pure feature-derivation fns; `from features import *`). Both do
(1) feature-eng → wide numeric per-VIN matrix with `TargetFlag`, (2) per-milestone NN.
`pms_model.py` holds everything the *current* train/score/eval scripts share: `BinaryClassifier`,
`add_milestone_flags()`, `derive_inference_features()`, `build_inference_matrix()`,
`load_artifacts()`, `predict_proba()`, `resolve_path()`, plus `LEAK_COLS` / `DERIVED_FEATURES` /
`HAS_X_EXCLUDE` / `DROP_RFM`. Edit the policy there, not in the three callers.
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
  `selected_features.json`, `metrics_*.json`). `resolve_path()` (in `pms_model.py`) falls back to
  `tests_alan/<path>` when a data path is missing at root.
  Test files: `testdataq1_q2/{m}kPMSTestDataQ{1,2}*.csv` — both must be the FULL
  `process_service_data()` output, because the training columns are intersected with every test
  file that exists. A narrow test file silently shrinks the model (see gotcha below).
- **Score a cohort**: `python score_milestone.py <milestone> --test <cohort.csv> [--out ... --threshold ...]`.
  Same derivation as `evaluate_test()`; loads `models/models_alan/{m}k/`, uses that dir's
  `threshold.json` (f1-tuned, e.g. 0.87) if present else 0.5 — note `retrain.py`/`eval_test_metrics.py`
  always use 0.5, so their numbers differ from `score_milestone.py`'s at the same model. Writes
  per-VIN `prob_turnup` to `predictions/{m}k/`, prints metrics only if the input has a real
  `TargetFlag`, and **warns with the count of zero-filled features** — read that warning, it is the
  main failure mode (below).
- **Extra per-VIN interval features**: `milestone_features.py`
  (`MilestoneHistory(m).features_for(vins, cutoffs)`) — months and mileage between consecutive
  completed milestones from raw service history, each row cut off at its own `Next{m}K_Due`.
  Standalone check: `python milestone_features.py 40`.
- **Held-out metrics table**: `python eval_test_metrics.py [milestone]` — mirrors `retrain.py`'s
  `evaluate_test()` exactly (same derivation, same has_x flags, 0.5 threshold); prints Q1/Q2
  tables with TP/TN/FP/FN, writes `retrained_test_metrics_final.csv`.
- **Test-set builder**: `python prepare_test_set.py --milestone 20 --window-start 2026-01-01 --window-end 2026-03-31 --train-cutoff 2025-12-31`
  → `test_sets/test_{m}k_Q1_2026.csv`. Labels from raw service history (not EDA's leaky
  `Last Service - PMS`), drops VINs that completed the milestone before the cutoff. **Its output is
  currently unusable as-is** — see the selected_features gotcha.
- `eval_alan_tests.py` evaluates the *legacy* `models/{m}k/training/` artifacts against
  `tests_alan/Q{1,2}/{m}k/test_features.csv` (80/90/100k only, hardcoded config list).

### Legacy / monolith pipeline
- Monoliths (pre-`features.py`, heavy duplication): `pmstrainfeatureEng_6.py` (training, `main()`),
  `predservicemil_4.py` (prediction; **executes at import**).
- `python train_milestone.py <milestone>` — argv trainer over the monolith's output →
  `models/{m}k/`. `training_run.py` / `prediction_run.py` are unstructured top-level scratch
  scripts (no `main()`, hardcoded 70k paths) — read before running.
- `legacy/` holds the earliest copies (`pmstrainfeatureEng.py`, `predservicemil.py`, `training_run.py`)
  — but it is really a whole old working copy: ~921 MB of duplicated `data/` files live there too.
  `legacy/parked/` is where the dead root scripts went on 2026-08-12 (its `training_run.py` is a
  *different, newer* file than `legacy/training_run.py`).
- Legacy net `ServicePredictionNN` (64→32→1, dropout 0.5), `StandardScaler`; artifacts in
  `models/{m}k/training/` (`best_nn_model.pt`, `imputer.pkl`, `scaler.pkl`).
- `fix_features.py` is a one-shot string-patcher that rewrites `features.py` in place. Don't run
  casually — its patches are likely already applied.

## Tests / diagnostics
No pytest/unittest, no test suite. Verification = run the eval scripts above and compare metrics.
`eval_test_metrics.py` is the quickest full check (all 9 milestones, Q1+Q2, ~2 min). Until
2026-08-12 it printed `No results.` and exited silently because it looked for `testdataq1_q2/` at
the repo root; if it ever prints that again, the test files moved.
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
- **A narrow test file silently shrinks the model.** `retrain.py` intersects the training columns
  with *every* test file that exists, so pointing `TEST_Q1` at a truncated file cuts the model's
  feature set. This already happened: 50k was retrained on 2026-08-04 against
  `prepare_test_set.py`'s 35-column output and collapsed to **32 features / 63.1% Q1 accuracy**
  (peers: 113–136 features, ~87%). Fixed 2026-08-12 — `TEST_Q1` is back on
  `testdataq1_q2/{m}kPMSTestDataQ1.csv` and 50k retrained to 118 features / 88.5%. Keep both test
  paths on the full-width files.
- **Two different `selected_features.json`, not interchangeable**: `models/{m}k/` = the
  `feature_sel()` MI shortlist (~33 cols); `models/models_alan/{m}k/` = what the trained model
  actually consumes (~113–136 cols, post test-schema intersection). `prepare_test_set.py` used to
  filter its output down to the **33** list, zero-filling 67–78% of every model's inputs; it now
  writes the full `process_service_data()` matrix. Never re-add a `selected_features` filter there
  — the models also need columns derived at inference time, which that list does not contain.
  Numbers per milestone: `REFACTOR_LOSSES.md`.
- **Gower cluster one-hots (`*_Cluster_*`) are not stable across feature-eng runs** (cluster "4"
  in train ≠ cluster "4" in test); `retrain.py` drops them.
- **RFM segments encode cohort vintage, not behaviour** — recency buckets off one fixed snapshot,
  so training positives land in Lost/Hibernating and test positives in Potential Loyalist/Promising.
  Caused 60k's 92.0 (Q1) vs 73.7 (Q2) split; hence `DROP_RFM = {60}`.
- `retrain.py` is maintained by hand. The orchestrators that used to generate it
  (`prepare_scripts.py`, `run_all.py`) were broken — their source templates no longer exist — and
  were parked in `legacy/parked/` on 2026-08-12 along with `training_run.py`, `prediction_run.py`
  and `fix_features.py`. See `legacy/parked/README.md`.
- `traindataq1_q2/` and `testdataq1_q2/` currently sit under `tests_alan/`, not at repo root.
  `retrain.py`, `score_milestone.py` and `eval_test_metrics.py` handle it via
  `resolve_path()` (`pms_model.py`); `milestone_features.py:117` still hardcodes the root path —
  copy/symlink before running it.
- `feature/pytorch-refactor` is a **git submodule/gitlink pinned at commit `dd31462`** — a stale
  nested clone of this same repo. Grep and glob hit it and return duplicate matches. Never edit
  there; it is not the working copy.
- Legacy eval-on-train quirk: the `train_nn_50k…100k` scripts pointed TEST_DATA_PATH at their own
  training CSV — those 50k+ metrics are not held-out.
- `data/`, `models/`, `predictions/`, `refactored_test_dir/`, `legacy/`, `validatecode/`, `*.csv`,
  `*.parquet`, `*.pkl`, `*.log` are gitignored — only source + docs tracked. So most dirs below
  exist on disk only; don't assume a fresh clone has them.

## Untracked dirs on disk (what's in them)
- `test_sets/` — `prepare_test_set.py` output; only 20/30/40/50k exist.
- `predictions/{m}k/` — `score_milestone.py` output (`scored_*.csv`) + legacy prediction datasets.
- `tests_alan/` — `Q1/`,`Q2/` legacy test features + the real `traindataq1_q2/`, `testdataq1_q2/`.
- `refactored_test_dir/` — current feature-eng output (`final_processed_{m}k.csv`, MI scores).
- `validatecode/` — legacy monolith intermediates; `predservicemil_4.py:2307` *reads*
  `Model_clusters_{m}.csv` / `Variant_clusters_{m}.csv` / `Nationality_clusters_{m}.csv` from here,
  so it is an input dir, not just debug dumps.
- `later/` — newer-vintage data (Appointments/VHC Q1-2027, Digital Sessions Q1-2026) not wired into
  any script, plus `refactoring_summary.md`.
- `legacy/` — old full working copy: earliest script versions, `parked/`, and ~921 MB of duplicated
  `data/` files. Nothing reads from it; it is an archive, not an input.
- `cache/`, `models/{m}k/` (legacy artifacts), `models/models_alan/{m}k/` (current).

## Docs
`README.md` — function catalog + data flow for the two original monoliths only; its "Quick run"
cwd (`d:\Techmax\nurture-mate\ML_layer\code`) is stale. `DATA.md` — which `data/*.csv` files are
legacy vs the new Q2-2026 snapshot, and where scripts currently mix vintages (unresolved).
`refactored_test_dir/mi_scores_analysis.md` — feature-selection analysis.
`REFACTOR_LOSSES.md` — line-referenced audit of `prepare_test_set.py` vs legacy `predservicemil_4.py`;
source of the zero-fill percentages and the two-`selected_features.json` finding. Trust it.
`S3_MIGRATION_GUIDE.md` — 8-stage plan to move data I/O to S3 (`s3io.py`, `stage_raw.py`,
`pmsfeatureeng_s3.py`, `retrain_s3.py`). **Aspirational — none of those files exist yet.**
`AGENTS.md` just points here. Treat the target-definition claims in `old_vs_refactored_comparison.md`
and `later/refactoring_summary.md` as wrong (see Project).
