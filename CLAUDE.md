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

### Labelling: "ever completed" — the POC convention, decided 2026-08-12
`TargetFlag == 1` = the vehicle has completed the target milestone **at any point** up to the latest
data. `TargetFlag == 0` = it was expected for the milestone and is **still pending past its expected
date** (`Service_Num < milestone`). **No date condition on positives.** The quarter argument only
selects *which pending cohort is sampled as the zeros*. Vehicles already *past* the milestone are
dropped by the `dfafter` filter, not counted positive.

This applies to **both training and test**, and it is what the code already does in both pipelines:
- `pmstrainfeatureeng_refactored.py:136` — `turnup = df.query("\`Last Service - PMS\` == '{svc}'")`
- legacy `pmstrainfeatureEng_6.py:229-234` and `legacy/pmstrainfeatureEng.py:172` — identical
- `prepare_test_set.py:88-94` — `Service_Num == milestone` anywhere in service history

The user chose this deliberately over a quarter-conditioned rule. Earlier revisions of this file and
of `AGENTS.md` called it an unfixed defect; that framing is **superseded**. Do not "fix" it —
switching to a quarter rule invalidates every training matrix and every published metric, and is a
decision, not a cleanup.

Three consequences, all accepted for the POC:
1. `Service_Num` leaks the label (only stamped on turn-up) — hence the hard drops in Gotchas below.
2. A `0` becomes a `1` in a later extract once the customer finally turns up. Labels are a function
   of the extract date, so always record which service-history file produced a metric.
3. Positives are graded generously. Measured on `test_sets/test_20k_Q1_2026.csv`: 751/951 rows
   positive (79%); only 366 turned up inside the Q1 window, 385 turned up after it (latest
   2026-12-06). Under this convention all 751 are correct — but a 79% base rate makes accuracy a
   weak headline number. Quote precision/recall and the confusion matrix instead.

One deliberate train/test asymmetry: a vehicle that completed the milestone **before the feature
cutoff** is a `1` in training but is dropped from the test cohort (`prepare_test_set.py:76-82`),
because its outcome is already known at scoring time.

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
  `feature_sel(cumulative_threshold=0.85)` (MI scores — **inspection only** since 2026-08-12; the
  block that wrote a competing feature list and `training_features.csv` is commented out).
  Writes `refactored_test_dir/final_processed_{m}k.csv` + `models/{m}k/mi_scores.csv`.
- **Train one milestone**: `python retrain.py <milestone>` (e.g. `python retrain.py 20`).
  Reads the ONE matrix `refactored_test_dir/final_processed_{m}k.csv`, cuts to
  `Next{m}K_Due <= 2025-12-31`, sorts by due date, `IsolationForest` (contamination 0.005),
  `RobustScaler`, chronological 80/20 split, `BinaryClassifier` 64→32→1 dropout 0.2,
  BCE(pos_weight) + 0.5·Brier + label smoothing. Seeded via `PMS_SEED` (default 42).
  Artifacts → `models/{m}k/` (`best_model.pt`, `scaler.joblib`, `imputer.joblib`,
  `threshold.json`, `metrics_*.json`) + `models/selected_features_{m}k.json`.
  `resolve_path()` (in `pms_model.py`) falls back to `tests_alan/<path>` when a data path is
  missing at root.
  Test files: `test_sets/test_{m}k_Q{1,2}_2026.csv` — both must be the FULL
  `process_service_data()` output, because the training columns are intersected with every test
  file that exists. A narrow test file silently shrinks the model (see gotcha below). A missing
  test file is skipped with a warning, not a crash.
- **Score a cohort**: `python score_milestone.py <milestone> --test <cohort.csv> [--out ... --threshold ...]`.
  Same derivation as `evaluate_test()`; loads `models/{m}k/` + `models/selected_features_{m}k.json`,
  uses that dir's `threshold.json` if present else 0.5 — `retrain.py`/`eval_test_metrics.py` always
  use 0.5, so a mismatched `threshold.json` makes their numbers disagree at the same model (20k was
  0.87 and scored 64.35% vs 96.32% at 0.5; retuned to 0.5 on 2026-08-12, the other eight are still
  stale). Writes
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
  → `test_sets/test_{m}k_Q1_2026.csv`. The filename is derived from the window by `window_label()`
  (an exact calendar quarter → `Q2_2026`, anything else → `2026-01-15_2026-02-28`). Until
  2026-08-12 it was hardcoded to `Q1_2026` while the window came from argv, so building a Q2 cohort
  silently overwrote the Q1 file under the Q1 name.
  Labels from raw service history (not EDA's `Last Service - PMS`), drops VINs that completed the
  milestone before the cutoff. Writes the full `process_service_data()` matrix — never re-add a
  feature filter.
- `eval_alan_tests.py` — **RETIRED 2026-08-12**, evaluation loop commented out. It read the legacy
  `models/{m}k/training/` artifacts and fell back to the now-removed
  `models/{m}k/selected_features.json`. Use `eval_test_metrics.py`.

### Legacy / monolith pipeline
- Monoliths (pre-`features.py`, heavy duplication): `pmstrainfeatureEng_6.py` (training, `main()`),
  `predservicemil_4.py` (prediction; **executes at import**).
- `python train_milestone.py <milestone>` — argv trainer over the monolith's output →
  `models/{m}k/`. `training_run.py` / `prediction_run.py` are unstructured top-level scratch
  scripts (no `main()`, hardcoded 70k paths) — read before running.
### Terminology: "legacy" means the `/legacy` DIRECTORY
When the user says *legacy*, they mean **`/legacy`** — not the legacy artifacts that still sit in
`models/{m}k/`. Keep the two apart in writing:
- **`/legacy`** — the archive directory. Contains exactly: `pmstrainfeatureEng.py`,
  `predservicemil.py`, `training_run.py` (the earliest copies), `parked/`,
  `validatecode/serv_filteredpmsapp.csv`, its own `venv/`, and data files.
  It holds **no model artifacts, no feature lists and no metrics** — no legacy metric can be
  reproduced from it.
  Its data files sit at the **root of `/legacy`**, not in `legacy/data/`, and they are **not
  duplicates of `data/`** — they are older, smaller snapshots (`EDA - Q3 2025.csv` 48.07 MB vs
  54.4 MB at root; `Service History Q1 - 2026.csv` 108.81 MB vs 123.9 MB). Any comparison against
  root `data/` must account for that.
  `legacy/parked/` is where the dead root scripts went on 2026-08-12 (its `training_run.py` is a
  *different, newer* file than `legacy/training_run.py`).
- **"legacy artifacts in `models/{m}k/`"** — always say it in full. `model.pt`, `feature_list.json`
  (188 features), `scaler.pkl`, `training/`. This is what the old root-level scripts produced.
- **`/legacy` has never been run — it has no outputs, logs or metrics, and none can be attributed
  to it.** Do not connect anything in `models/`, `models_alan/` or any eval output to `/legacy`;
  those artifacts came from the root-level scripts, not from the archive. Read `/legacy` as source
  code only.
- **Nothing in `/legacy` could produce a held-out metric anyway.** `legacy/training_run.py` has no
  evaluation at all: the test split and every metric line are commented out (lines 2, 8-10, 51-52),
  it early-stops on **training** loss (line 222), and there is no validation split. Features are
  `X_clean.iloc[:, 5:]` — drop five ID columns positionally, feed in everything else, no leak
  filtering. So any "legacy accuracy" quoted from memory or old docs cannot have come from this
  code, and is not a bar for the current pipeline to reach.
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
  exceptions — verified 2026-08-12 on the 20k and 60k training matrices (100.00% of rows).
- **`PMS_Delay` smuggles that leak back in, and it is NOT dropped. The refactor caused this.**
  The two pipelines compute it differently, and only one of them is a real feature:
  - `/legacy` ([legacy/pmstrainfeatureEng.py:2573](legacy/pmstrainfeatureEng.py#L2573)):
    `PMS_Delay = Vehicle_Key_ExpectedServices - Vehicle_Key_Actual_Service` — expected is a
    **per-vehicle** EDA column (22 distinct values, 0-21, mean 12.18, std 6.75), so the feature
    genuinely measures "how far behind schedule is this customer".
  - current ([features.py:1450](features.py#L1450)): `compute_pms_delay()` replaces that first term
    with the **constant** `target_milestone / 10` (2 for 20k, 6 for 60k).

  Consequence, measured 2026-08-12: current `PMS_Delay` correlates **-1.0000** with
  `Vehicle_Key_Actual_Service` — it *is* that column, negated and shifted — and only **0.4346** with
  the legacy definition. `LEAK_COLS` drops `Vehicle_Key_Actual_Service`, then `PMS_Delay` carries it
  straight back in. Both pipelines drop the `Vehicle_Key_ExpectedServices` source column at the same
  point (`legacy:2675` / `refactored:807`), but legacy computes `PMS_Delay` *before* that drop; the
  refactor no longer reads it at all.
  **FIXED 2026-08-13**: `compute_pms_delay()` now uses the per-vehicle
  `Vehicle_Key_ExpectedServices`, matching legacy, and warns loudly if that column is absent rather
  than silently falling back to the constant. `PMS_Delay` is computed at
  `pmstrainfeatureeng_refactored.py:635`, before the source column is dropped at :807 — keep that
  ordering. Measured effect on 20k:

  | | constant (old) | per-vehicle (now) |
  |---|---|---|
  | corr with `Vehicle_Key_Actual_Service` | -1.0000 | -0.1620 train / -0.7624 test |
  | single-feature AUC on test | 0.9849 | 0.8362 |
  | `PMS_Delay <= 0` reproduces TargetFlag | 98.84% | 83.91% (base 78.97%) |
  | full-model AUC on Q1 test | 0.9914 | 0.8145 |

  Deleting the feature instead gives AUC 0.6041, so the fixed version carries **real** signal —
  fix it, do not drop it. Residual caveat: inside a single due-quarter the expected term barely
  varies (7 distinct values, -4..2, vs 22 spanning -1..20 in training), so it partly collapses back
  toward a constant at test time. That is why test AUC alone is still 0.836.
  The numbers in the rest of this section were measured with the OLD constant version.
- **`PMS_Delay` is now selectable, four ways — default `legacy`, recommended `derivedB`**
  (2026-08-16). `compute_pms_delay()` emits all four candidates into every matrix and test set;
  `select_pms_delay_variant()` in `pms_model.py` collapses them to one column *always named*
  `PMS_Delay`, so `selected_features_{m}k.json` is byte-identical across variants and an A/B
  compares values, not schema. Switch with `$env:PMS_DELAY_VARIANT=derivedB`.

  | variant | formula |
  |---|---|
  | `legacy` (default) | `EDAexp - EDAact` |
  | `doc` | `legacy - TargetFlag` |
  | `derivedA` | `EDAexp - sum(has_x)` |
  | `derivedB` | `2*Years_Since_First_PMS - sum(has_x)` |

  **Both terms of the legacy formula are `LEAK_COLS` members in disguise.**
  `Vehicle_Key_ExpectedServices == 2 × Vehicle Age` — 53.2% exact, **99.90% within 1** (a
  semi-annual schedule; `Vehicle Age` is an integer 0-10). And
  `Vehicle_Key_Actual_Service == sum(has_x) + TargetFlag` to three decimals on both matrices, i.e.
  the EDA count includes the target service for turn-ups **and only for turn-ups**. So the legacy
  formula already carries an implicit `-1` on positives.
  The expected term is also measured at **extract**, not at the cutoff: reproducing `Vehicle Age`
  as `floor((ref - Invoice date)/365.25)` matches **4.88%** at ref=2025-12-31 (the training cutoff)
  but **90.44%** at ref=2026-12-06 (the extract). Re-deriving from sale date is not viable —
  `Invoice date` parses for only 60,411/130,759 EDA rows (46%) — but `Years_Since_First_PMS`
  (`derive_pms_frequency()`, anchored on `filterdate`) is 100% populated, hence `derivedB`.
- **The doc's turn-up `-1` rule is REJECTED — measured, not assumed** (2026-08-16). Given the
  identity above it is really a `-2` on positives, and it is label-conditioned by construction.
  Single-feature AUC train → Q1 test: **0.5507 → 0.9755** (20k), **0.5094 → 0.9934** (60k); the
  rule `PMS_Delay_Doc <= 0` reproduces `TargetFlag` on 95.06% / 97.51% of test rows. It scores that
  well because `prepare_test_set.py` builds labelled cohorts, so the test column subtracts the test
  label too — in production there is no `TargetFlag` and the column silently falls back to
  `PMS_Delay`, a different feature. It is also *useless*: 20k full-model AUC **0.4753**, worse than
  chance and the worst of the four. `/legacy` never implemented it
  ([legacy/pmstrainfeatureEng.py:2573](legacy/pmstrainfeatureEng.py#L2573), plain
  `expected - actual`, same on the prediction side at
  [predservicemil_4.py:2131](predservicemil_4.py#L2131)). Keep it available for demonstration only.
- **Variant results, seed 42, same regenerated matrices** (2026-08-16). `AUCtr`/`AUCte` are
  `PMS_Delay` alone; the rest is the full model on the Q1 test set.

  | | | 20k (base 78.97%) | | | | 60k (base 82.44%) | | |
  |---|---|---|---|---|---|---|---|---|
  | variant | AUCtr | AUCte | model AUC | best acc | AUCtr | AUCte | model AUC | best acc |
  | `legacy` | 0.5036 | 0.8362 | 0.5508 | 84.12 | 0.5534 | 0.9629 | 0.9506 | 90.51 |
  | `doc` | 0.5507 | 0.9755 | 0.4753 | 81.81 | 0.5094 | 0.9934 | 0.9593 | 91.46 |
  | `derivedA` | 0.5602 | 0.6002 | 0.5976 | 84.23 | 0.5969 | 0.8869 | 0.9462 | 90.15 |
  | **`derivedB`** | 0.5855 | 0.5817 | **0.6191** | **84.75** | 0.5062 | 0.8997 | 0.9530 | 90.87 |

  **`AUCte >> AUCtr` is the leak signature.** `derivedB` is the only variant without it on 20k
  (0.5855 → 0.5817), wins 20k on every full-model measure, and on 60k gives the best F1 (77.0) and
  recall (87.2) while trailing `legacy` by 0.003 AUC. It is also the only variant whose correlation
  with the EDA expected term is not near-unity (+0.666 / +0.712 vs +0.92…+1.00).
- **`Service Frequency` is now the largest remaining leak** (2026-08-13). It is an EDA
  current-state field, NOT in `LEAK_COLS`, and it reaches every model. Evidence:
  single-feature AUC **0.9449 on test vs 0.8300 on train** (stronger on test = post-outcome
  information), correlation with `Vehicle_Key_Actual_Service` 0.31 train / **0.77 test**, and it is
  one of the 7 EDA columns that changed between the Q3-2025 and Q2-2026 snapshots (86.9% of rows),
  i.e. recomputed per extract rather than a static attribute. Identical profile to
  `Vehicle Lifetime in Months`, which IS already dropped. Removing it costs 20k AUC 0.8145 -> 0.6445.
- **`Vehicle Age` has no hidden signal, at least per-quarter cohort.** Restored to the feature-eng
  output on 2026-08-13 (legacy `rem` list at :2520-2523 never dropped it; the refactor added it).
  Measured: **constant** across the 20k Q1 test cohort — zero variance, because "due for 20k in
  Q1 2026" pins vehicle age by construction — and 0.5399 AUC alone on train. `Current Age` is
  similar (0.5159 train / 0.5638 test) but genuinely NOT leaky (corr 0.07 / 0.02 with actual
  services); legacy kept `Current Age` and dropped `Vehicle Age` positionally via
  `feature_sel`'s `iloc[:,7:-1]`. Both are still in `LEAK_COLS`.
- **20k honest baseline, all measured on the same regenerated data, seed 42** (base rate 78.97%,
  so majority-class accuracy is 78.97):

  | config | feats | AUC | acc@0.5 | best acc (tuned thr) |
  |---|---|---|---|---|
  | all features, fixed `PMS_Delay` | 165 | 0.8145 | 66.67 | 87.49 @ 0.22 |
  | minus `Service Frequency` | 164 | 0.6445 | 85.28 | 85.91 @ 0.73 |
  | minus `PMS_Delay`+`Service Frequency`+`Last Service Mileage` | 162 | 0.6041 | 84.65 | 84.86 @ 0.58 |

  Do **not** compare these against models trained before the `PMS_Delay` fix: `PMS_Delay` changed
  meaning, so an old model scored on a new test set is a train/test semantic mismatch (it reads
  0.8054, which is meaningless). `LEAK_COLS` contains only 5 columns; `PMS_Delay`, `Service Frequency` and
  `Last Service Mileage` are **not** among them and DO reach the model (present in every
  `selected_features_{m}k.json`). An earlier revision of this file claimed all 8 were dropped in
  `feature_sel()` and `retrain.py` — false for those three.
  On the 20k Q1 test set the hand-written rule `PMS_Delay <= 0` reproduces `TargetFlag` on
  **98.84%** of rows (base rate 78.97%), beating the trained model. Single-feature AUC on test:
  `PMS_Delay` 0.985 (20k) / 0.982 (60k), vs the full model's 0.9746.
- **`has_x` was built correctly and then OVERWRITTEN. Fixed 2026-08-16.** Two definitions existed:
  - [pmstrainfeatureeng_refactored.py:343-348](pmstrainfeatureeng_refactored.py#L343-L348) builds
    them from `serv1`, already cut at `Service_Date <= filterdate` (:332) — exactly the intended
    "by the cutoff, had this vehicle done its 50k?". **Not leaky.**
  - `add_milestone_flags()` in `pms_model.py` then replaced them with `(Service_Num >= x)`, where
    `Service_Num` comes from EDA's post-outcome `Last Service - PMS`. **Leaky.**

  `retrain.py` and `build_inference_matrix()` both called it, so the good version never reached a
  model. Measured on 60k, AUC as built → after the overwrite:

  | | built | overwritten | | | built | overwritten |
  |---|---|---|---|---|---|---|
  | `has_10` | 0.6197 | 0.5000 | | `has_40` | 0.7108 | 0.8313 |
  | `has_20` | 0.5119 | 0.6128 | | `has_50` | 0.7879 | **0.9223** |
  | `has_30` | 0.5617 | 0.6907 | | | | |

  `has_50`'s positive-class rate went 0.732 → **1.000** while negatives held at 0.155: the overwrite
  forced the flag on for every training positive. It injected a leak *and* destroyed real signal
  (`has_10` collapsed to constant 1). `add_milestone_flags()` now **skips any flag already in the
  frame** and warns when it has to derive one; the `Service_Num` path survives only as a fallback
  for inference frames built outside `process_service_data()`. Post-fix `has_50` reads pos 0.732 /
  neg 0.156 — verify that number after any regeneration.
  `HAS_X_EXCLUDE = {80}` because 80k's Q2 file records `Service_Num = 0` for all 651 positives
  (1.7% recall otherwise) — a defect of the *EDA-derived* path, so it is probably unnecessary now.
  Untested; 80k was out of scope for the fix.
- **`LowMileageFreqUsers` had the label ANDed into its own formula. Fixed 2026-08-13/16.** The
  definition carried `& (pmsnew["TargetFlag"]==1)`, so the flag could only ever be 1 for a positive
  — on the 60k matrix all 208 flagged rows were positive. `/legacy` had already fixed this
  ([legacy/pmstrainfeatureEng.py:2621-2625](legacy/pmstrainfeatureEng.py#L2621-L2625)); the clause
  is now gone from `pmstrainfeatureeng_refactored.py` too. Post-fix 20k crosstab: 1,040 negatives
  and 103 positives carry the flag, and it became the **#1 permutation-importance feature on 20k**
  (+0.1018). `pmstrainfeatureEng_6.py:2650` still has the bug — legacy monolith, not on the current
  path. Note the two pipelines disagree on the mileage term (training `Last Service Mileage`,
  `predservicemil_4.py:2680` `Last PMS Mileage`) — unresolved.
- **`--label-mode window` makes training match the test cohorts** (added 2026-08-16, `ever` is
  still the default). `python pmstrainfeatureeng_refactored.py 2026 Q1 20 --train-cutoff 2025-12-31
  --label-mode window --label-start 2018-01-01` → `refactored_test_dir/final_processed_20k_window.csv`;
  `retrain.py` picks it up via `$env:PMS_LABEL_MODE=window`. **The test sets need no change** —
  `create_test_cohort()` ([prepare_test_set.py:66-70](prepare_test_set.py#L66-L70)) already builds a
  due-in-window cohort, so only training was lopsided.
  In window mode the positives are cut to the same `Expected{svc}Date` window as the negatives, and
  the ratio-tuning `while` loop is **skipped** — the window now defines both classes, so sliding it
  to hit 0.35-0.45 would be manufacturing the base rate. `--label-start` widens the shared window;
  the get_quarter_dates() default (Jan 1 of the previous year) is far too narrow once positives are
  windowed too. Cohort sizes by start date:

  | start | 20k | 30k | 40k | 60k |
  |---|---|---|---|---|
  | 2025-01-01 (default) | 1,343 / 66.1% | 2,212 / 61.6% | 2,391 / 28.5% | 2,613 / 25.0% |
  | 2018-01-01 (used) | **7,042 / 54.9%** | **12,669 / 49.3%** | **18,532 / 35.5%** | **23,114 / 21.5%** |

  **Result: `window` did NOT beat `ever` — it lost AUC almost everywhere** (measured 2026-08-16,
  seed 42, PMS_Delay held at `legacy`, evaluated on Q1 *and* Q2 2026):

  | | 20k Q1 | 20k Q2 | 30k Q1 | 30k Q2 | 40k Q1 | 40k Q2 | 60k Q1 | 60k Q2 |
  |---|---|---|---|---|---|---|---|---|
  | `ever` | 0.5549 | 0.4154 | 0.9405 | 0.9290 | 0.9169 | 0.8711 | 0.9506 | 0.9581 |
  | `window` | 0.5665 | 0.4541 | 0.8715 | 0.8684 | 0.7839 | 0.7279 | 0.9233 | 0.9374 |
  | lift | +0.012 | +0.039 | -0.069 | -0.061 | **-0.133** | **-0.143** | -0.027 | -0.021 |

  It *did* fix calibration — accuracy at the default 0.5 threshold jumped (20k Q1 42.5→77.6,
  60k Q2 76.5→89.8) because the training base rate now matches the test one. Better calibrated,
  worse at ranking. Do not adopt it as-is; the two causes below have to be fixed first.
- **`Next{m}K_Due` is FABRICATED for `ever`-mode positives** (2026-08-16). At
  [pmstrainfeatureeng_refactored.py:189](pmstrainfeatureeng_refactored.py#L189) `Expected{svc}Date`
  is back-filled from `Last Service Date - PMS`. Positives never had it computed, so a positive's
  "due date" **is the day it turned up**; negatives get a genuine forecast. Measured months from due
  to actual turn-up for positives: median **0.0** and p90 **0.0** on both 20k and 60k — exactly
  zero at the 90th percentile is the tell. It is dropped by `id_cols()` so it does not feed the
  model, but `retrain.py` sorts by it for the chronological 80/20 split and applies the
  `<= 2025-12-31` cutoff to it, so the two classes are ordered on different clocks.
- **The expected-date formula is miscalibrated by up to a year** (2026-08-16). `window` mode
  computes `Expected{svc}Date = FirstSrvDate + 6 months x (milestone/10)` honestly for both classes,
  and that exposed it. Months from that due date to the actual turn-up, positives only:

  | cohort | median | p90 | max |
  |---|---|---|---|
  | 20k train `window` | -2.5 | +8.7 | +67.0 |
  | 60k train `window` | **-12.7** | +8.1 | +61.8 |
  | 20k / 60k **test** Q1-2026 | +0.8 | +5.6 / +5.2 | +10.5 |

  Customers do their 60k about **a year earlier** than the formula predicts, so window mode filtered
  on the wrong dates — which is why 40k and 60k lost the most AUC. The test cohorts are tight
  (+0.8 median, +10.5 max) not because of a better date but because `create_test_cohort()`
  **excludes vehicles that completed the milestone before the cutoff** ([prepare_test_set.py:76-82](prepare_test_set.py#L76-L82)),
  so its positives complete *after* the window opens by construction.
- **The mileage projection is WORSE than the time-based date — do not switch to it** (2026-08-16).
  `create_test_cohort(use_mileage_projection=True)` extrapolates the vehicle's lifetime km/day to
  when it hits `milestone * 1000`. The CLI flag `--enable-mileage-projection` defaults it OFF, so
  every test set built so far used the plain time-based date. Measured lag to the actual turn-up:

  | | median | p10 | p90 | IQR |
  |---|---|---|---|---|
  | 20k time-based | -3.9 | -11.5 | +4.7 | **7.9** |
  | 20k mileage-projected | **-17.8** | -55.8 | -0.1 | **29.9** |
  | 60k time-based | -14.8 | -30.8 | +3.3 | **17.6** |
  | 60k mileage-projected | -16.1 | -52.3 | 0.0 | **27.4** |

  It breaks for vehicles already past the milestone mileage: `Miles_Remaining` goes negative,
  `Days_Remaining` is clipped to NaN then `fillna(0)`, so `Projected_Date` collapses onto the last
  service date. Leave the flag off.
- **`--date-model calibrated` — built, measured, and it did NOT win** (2026-08-16).
  `features.expected_milestone_months()` is shared by both builders; `MILESTONE_MONTHS_CALIBRATED`
  holds medians measured over 15k-48k vehicles per milestone. The schedule assumes 6 months per 10k;
  reality is 3.25-4.40 and drifts down with the milestone (20k 12→9, 40k 24→16, 60k 36→22,
  100k 60→32). Switch with `--date-model calibrated` on both scripts and `$env:PMS_DATE_MODEL`
  on `retrain.py`; a calibrated build gets its own `_cal` matrix and `test_{m}k_Q{1,2}_2026_cal.csv`
  because it selects a **different population**, so metrics are NOT comparable across date models.

  Full four-arm AUC, seed 42, PMS_Delay held at `legacy`:

  | | 20k Q1 | 20k Q2 | 30k Q1 | 30k Q2 | 40k Q1 | 40k Q2 | 60k Q1 | 60k Q2 |
  |---|---|---|---|---|---|---|---|---|
  | schedule / **ever** | 0.5549 | 0.4154 | **0.9405** | **0.9290** | **0.9169** | **0.8711** | **0.9506** | **0.9581** |
  | schedule / window | **0.5665** | 0.4541 | 0.8715 | 0.8684 | 0.7839 | 0.7279 | 0.9233 | 0.9374 |
  | calibrated / ever | 0.4013 | 0.4012 | 0.8629 | 0.7119 | 0.8969 | 0.8676 | 0.9195 | 0.9313 |
  | calibrated / window | 0.5025 | **0.4925** | 0.8822 | 0.7649 | 0.8312 | 0.6895 | 0.9116 | 0.8887 |

  **`schedule` + `ever` — the current production default — wins 6 of 8.** The one real signal:
  under calibrated dates `window` beats `ever` on 20k and 30k (it loses under schedule dates), which
  is weak confirmation that better dates help window mode relatively. Not enough to adopt.
- **The calibration is biased early — it was fitted on COMPLETERS only** (2026-08-16). This is why
  the calibrated cohorts ballooned and their base rates collapsed (60k Q1: 843 rows / 17.6% positive
  → 1,666 rows / 10.0%): an earlier due date sweeps in vehicles that are not due yet.

  | | eligible | ever completed | completion rate |
  |---|---|---|---|
  | 20k | 85,775 | 52,443 | 61.1% |
  | 40k | 82,201 | 43,061 | 52.4% |
  | 60k | 77,896 | 30,116 | **38.7%** |

  Only 39% of 60k-eligible vehicles ever do the service, so a median over completers answers "when
  do completers complete", not "when is it due" — censored data needs a survival/hazard estimate
  (Kaplan-Meier), not a median. Fixing that is the prerequisite before `--date-model calibrated` is
  worth retrying.
- **The schedule-keeping features point the WRONG WAY between train and test, and on 20k they
  actively hurt** (2026-08-16). This is what `--label-mode window` above exists to fix. Raw single-feature
  AUC (not folded above 0.5 — `<0.5` means the relationship is inverted):

  | feature | 20k train | 20k test | 60k train | 60k test |
  |---|---|---|---|---|
  | `has_10` | 0.3602 | **0.6885** | 0.3803 | **0.5340** |
  | `PMS_Count_Prior` | 0.3792 | **0.7224** | 0.6187 | 0.8824 |
  | `Years_Since_First_PMS` | 0.4099 | **0.5435** | 0.5693 | **0.4026** |
  | `LowMileageFreqUsers` | 0.3747 | 0.2791 | 0.4267 | **0.5926** |

  In 20k training, "has done more prior services" predicts **not** turning up; in the test cohort it
  predicts the opposite. Group-permutation on the Q1 test set confirms the cost — shuffling
  `{PMS_Delay, Years_Since_First_PMS, PMS_Count_Prior, PMS_Freq_PerYear, has_*}` together **raises**
  20k test AUC by **0.19** (0.6191 → ~0.81). The 20k model would be better with that whole family
  removed.
  **Cause is the "ever completed" convention, not a bug.** Positives are drawn from all time,
  negatives from one due-date window, so the two cohorts differ systematically in service-history
  depth and any history-derived feature inherits the difference with the wrong sign. On 60k the
  milestones nearest the target (`has_30`/`has_40`/`has_50`) stay consistent, which is why 60k still
  trains to AUC 0.95 while 20k sits at 0.62. Do not "fix" this by dropping features — it is the
  labelling convention, and changing that is the user's decision (see Project).
- **Permutation importance under-credits redundant features — read groups, not columns**
  (2026-08-16). The model encodes "is this customer keeping up with services" across `has_10..has_50`,
  `PMS_Count_Prior`, `PMS_Freq_PerYear`, `PMS_Delay` and `PMS Status_*`. Permuting one column at a
  time leaves the others to cover for it, so each scores near zero — `PMS_Delay` alone measures
  **+0.0005** (20k) / **-0.0006** (60k) despite a 0.90 single-feature test AUC on 60k. That is why
  `Total VHC Revenue` ranked #1: it is the strongest feature with **no duplicate**, not the strongest
  feature. Grouped permutation on 60k: VHC/revenue family **+0.1047** (11 cols) vs schedule-keeping
  **+0.0195** (9 cols) — so on 60k VHC really does dominate *this* model, because the
  schedule-keeping family has been broken by the inversion above. On 20k the ordering is the
  expected one: `LowMileageFreqUsers` **+0.1018** at #1, VHC family **-0.0057**.
  SHAP (legacy 70k) and permutation importance are **not comparable** here — the legacy model had no
  `has_x` at all, so `PMS_Delay` had no competitor and dominated its SHAP as it should.
- **`load_artifacts()` ignored `PMS_RUN_TAG`. Fixed 2026-08-16** — it always read
  `models/selected_features_{m}k.json`, so scoring a tagged experiment dir loaded the untagged
  feature list: a `state_dict` size mismatch when the counts differ, or a silent
  wrong-column-order load when they happen to match. Pass `features_file=` for any tagged run.
- **Ablation switches** (added 2026-08-12, defaults reproduce production exactly):
  `$env:PMS_USE_HAS_X=0`, `$env:PMS_DROP="PMS_Delay,Service Frequency"`,
  `$env:PMS_RUN_TAG=_nohasx` → artifacts to `models/{m}k_nohasx/`. Use these to A/B rather than
  editing `retrain.py`.
- **Train/test schema mismatch is the dominant cause of bad metrics** — missing columns get
  zero-filled at inference. `retrain.py` guards by reconciling training columns with what the test
  sets actually produce (`_test_feature_cols`). Keep that reconcile.
- **…but a blanket intersection over-corrects, and it was costing real features.** Per-category
  columns are only emitted for categories present in the frame, so a 951-row test cohort cannot
  produce a dummy for a rare part that 8,609 training rows do. The old intersection deleted those
  from *training* too. Fixed 2026-08-12: a train-only column is kept when the test set has a sibling
  from the same family (`family_of()` in `pms_model.py`, shared with the reporting side), dropped
  when the whole family is absent. On 20k that recovered **36** of 41 columns —
  `DeferredPart__*`, `InvoicedPart__*`, `LostPart__*`, `Franchise_*` — taking the model from 129 to
  165 features.
  Do **not** re-add a binary 0/1 test for "is this a dummy": several of those columns are
  per-category **counts**, and a count of 0 for a category that never occurred is equally correct.
  That mistake cost 10 of the 36 on the first attempt.
  Still deliberately dropped: `PMS Status_*` (its family IS present, but it describes the last PMS
  visit and restates the label) and two dealer dummies with no detectable family.
  **Known edge case:** `family_of()` splits on the last `_`, so a category value containing an
  underscore is mis-familied — `last_appointment_status_NO_SHOW` resolves to family
  `last_appointment_status_NO` and does not match `last_appointment_status`, so it is dropped
  as a false "gap". Costs 1 column on 20k.
- **Zero-fills are now reported in two buckets** — `split_missing()` separates absent categories
  (harmless, 0 is correct) from genuine pipeline gaps (invalidates the metrics). `eval_test_metrics.py`
  reports them as `ZeroFill` (real gaps, must be 0) and `DummyFill` (informational). Before this,
  36 harmless zero-fills would have buried a real one.
- **A narrow test file silently shrinks the model.** `retrain.py` intersects the training columns
  with *every* test file that exists, so pointing `TEST_Q1` at a truncated file cuts the model's
  feature set. This already happened: 50k was retrained on 2026-08-04 against
  `prepare_test_set.py`'s 35-column output and collapsed to **32 features / 63.1% Q1 accuracy**
  (peers: 113–136 features, ~87%). Fixed 2026-08-12 — `TEST_Q1` is back on
  `testdataq1_q2/{m}kPMSTestDataQ1.csv` and 50k retrained to 118 features / 88.5%. Keep both test
  paths on the full-width files.
- **One feature list per model, since 2026-08-12**: `models/selected_features_{m}k.json`
  (113–136 cols, post test-schema intersection), written by `retrain.py` and read by every scoring
  path via `features_path()` in `pms_model.py`. The two same-named, incompatible files that used to
  coexist are gone: `models/{m}k/selected_features.json` (the `feature_sel()` MI shortlist, ~33
  cols) is no longer written — that block is commented out in `pmstrainfeatureeng_refactored.py` —
  and the old copies were renamed `_retired_mi_shortlist.json`. Filtering a test set by the
  shortlist used to zero-fill 67–78% of every model's inputs and still run clean; per-milestone
  numbers in `REFACTOR_LOSSES.md`. Never re-add a feature filter to `prepare_test_set.py` — the
  models also need columns derived at inference time, which no stored list contains.
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
- `legacy/` — the archive. Earliest script versions, `parked/`, `validatecode/`, its own `venv/`,
  and **older, smaller** data snapshots at the directory root (not `legacy/data/`, and not copies of
  `data/`). No model artifacts, no feature lists, no metrics. Nothing reads from it. See the
  terminology note in the Legacy pipeline section — "legacy" always means this directory.
- `cache/`, `models/{m}k/` (current artifacts + `mi_scores.csv`; legacy leftovers alongside them in
  `training/`, `_retired_mi_shortlist.json`, `_legacy_threshold.json`),
  `models/selected_features_{m}k.json` (the feature lists),
  `models/models_alan/` (pre-2026-08-12 layout, kept as a backup — nothing reads it).
- `c:\Techmax\cwf\backups\` — outside the repo. `2026-08-12_pre-q2-regen/` holds `models/`,
  `refactored_test_dir/` and `test_sets/` as they stood before the Q2-2026 data switch (344 MB).

## Docs
`README.md` — function catalog + data flow for the two original monoliths only; its "Quick run"
cwd (`d:\Techmax\nurture-mate\ML_layer\code`) is stale. `DATA.md` — which `data/*.csv` files are
legacy vs the new Q2-2026 snapshot. **Its "scripts mix vintages, unresolved" warning is now stale**:
both feature-eng scripts read the Q2-2026 set as of 2026-08-12 (`EDA_Q2-2026.csv`,
`Service_History_Q2-2026.csv`, `Service_Code_Desc.csv`, RFM / Appointments / Digital Sessions / VHC).
Verified before switching: `Service_History_Q2-2026.csv` is byte-identical to the old
`Service History Q1 - 2026.csv`, and the two EDA snapshots share all 100 columns, all 129,636 VINs
and an **identical `Last Service - PMS`** — only 7 feature columns differ (`Service Frequency` 86.9%
of rows, `Vehicle Lifetime in Years` 43.8%, `Target Revenue` 19.2%, the rest &lt;0.5%). So the switch
refreshes feature values without moving a single label. Note `Service_History_Q2-2026.csv` actually
runs to **2026-12-06**, not Q2 — the name understates its coverage.
`refactored_test_dir/mi_scores_analysis.md` — feature-selection analysis.
`REFACTOR_LOSSES.md` — line-referenced audit of `prepare_test_set.py` vs legacy `predservicemil_4.py`;
source of the zero-fill percentages and the two-`selected_features.json` finding. Trust it.
`S3_MIGRATION_GUIDE.md` — 8-stage plan to move data I/O to S3 (`s3io.py`, `stage_raw.py`,
`pmsfeatureeng_s3.py`, `retrain_s3.py`). **Aspirational — none of those files exist yet.**
`AGENTS.md` just points here. Treat the target-definition claims in `old_vs_refactored_comparison.md`
and `later/refactoring_summary.md` as wrong (see Project).
