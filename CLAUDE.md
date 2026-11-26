# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Preference
- Work at a beginner software-engineer level, in simple mode.
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

> **Read this before the `ever` section below.** Every model in `production_models/` (all nine,
> 20k–100k) was built with **`--label-mode candidates`**, not `ever`. The "ever completed"
> convention documented next is the POC/baseline lineage and still the *code* default
> (`--label-mode ever`, `PMS_LABEL_MODE=ever`), which is why the older Gotchas are written against
> it — but it is **not what ships**. See `### Production labelling: --label-mode candidates`.

### Labelling: "ever completed" — the POC convention, decided 2026-08-12
`TargetFlag == 1` = the vehicle has completed the target milestone **at any point** up to the latest
data. `TargetFlag == 0` = it was expected for the milestone and is **still pending past its expected
date** (`Service_Num < milestone`). **No date condition on positives.** The quarter argument only
selects *which pending cohort is sampled as the zeros*. Vehicles already *past* the milestone are
dropped by the `dfafter` filter, not counted positive.

This applies to **both training and test**, and it is what the code already does in both pipelines:
- `pmstrainfeatureeng_refactored.py:136` — `turnup = df.query("\`Last Service - PMS\` == '{svc}'")`
- legacy `monoliths/pmstrainfeatureEng_6.py:229-234` and `legacy/pmstrainfeatureEng.py:172` — identical
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

### Production labelling: `--label-mode candidates` (cohort-v3) — what every promoted model uses
`ever` and `window` are the two modes the Gotchas below argue about; **`candidates` is the one that
won and shipped.** All nine `production_models/{m}k/` were trained with:

```
python pmstrainfeatureeng_refactored.py 2026 Q1 <M> --label-mode candidates --grace-days 15
```

→ `refactored_test_dir/final_processed_{M}k_candidates.csv`, and matched test cohorts from
`prepare_test_set.py --label-mode candidates --grace-days 15` → `test_sets/test_{M}k_Q{1,2}_2026_cand.csv`.
`retrain.py` picks the matrix up via `$env:PMS_LABEL_MODE='candidates'`.

How it differs from the other modes (`build_candidates_cohort()` / `run_candidates_mode()`):
- **Cohort membership** = vehicles whose *projected* due date lands in the quarter, where the due
  date is `earliest(6-months-per-10k schedule, cumulative burn-rate projection)` —
  `due_date.earliest_date()` / `burn_rate_date()`. Not EDA's back-filled `Expected{svc}Date`, which
  is the fabricated-date problem documented in Gotchas.
- **Requires a genuine EDA `Invoice date`** (the invoice guard; `--strict-invoice-guard` tightens it
  to per-VIN).
- **Spans many due-quarters** — `--quarters-from` / `--quarters-to` (default `2016Q1`..`2025Q4`),
  so a matrix is ~57k–75k rows over ~36 quarterly cohorts, not one cohort.
- **Early completers are POSITIVES**, flagged `EarlyCompleter`, rather than dropped. Deliberate —
  do not "fix" it in training. `stale_completion_filter.py` removes them from *forward* lists only.
- `--grace-days` is **forward-only** here: any early completion counts, a late one counts only
  within the grace band. The rollout standard is **15**, not the CLI default of 45.
- The positive-rate tuning `while` loop does not apply; base rates land ~20–25%.

Other modes still reachable: `ever` (default), `window`, `all`, `history`. `history` is the
direct ancestor of `candidates` — same per-quarter cohort construction, but with the plain
schedule due date and no invoice guard.

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

### Project layout (2026-11-26)
Shared modules and the main entry scripts stay at the **root** — about 50 scripts import them by bare
name (`features`, `date_utils`, `due_date`, `pms_model`, `pmstrainfeatureeng_refactored`,
`build_unionstale_20k`), and every documented command assumes it. Everything in a sub-folder is run
**from the repo root with `$env:PYTHONPATH = "."`** so its imports of root modules resolve:

| folder | what lives there |
|---|---|
| root | shared modules; production builders and scorers (`pmstrainfeatureeng_refactored.py`, `prepare_test_set.py`, `retrain.py`, `score_milestone.py`, `eval_test_metrics.py`, `leak_screen.py`, `write_manifest.py`, `stale_completion_filter.py`, `tidy.py`) |
| `dprime_v2/` | the D′ v2 pipeline — current best model (±45-day list + tree) |
| `tests/` | gate G2 regression scripts (`test_dayfirst.py`, `test_service_num.py`) |
| `analysis/` | reconciliation / overlap reports (`reconcile_quarter.py`, `q3_eval_report.py`, `july_*`) |
| `monoliths/` | the pre-`features.py` monoliths + `train_milestone.py` — NOT the `/legacy` archive |
| `q1_due_rule_experiment/`, `noshow_feature_review/`, `Q2/`, `test/` | experiment scripts and runbooks behind numbers in this file |
| `production_models/`, `models/`, `refactored_test_dir/`, `test_sets/`, `data/`, `Q3/` | artifacts and data — gitignored |

## Architecture — two parallel pipelines
Both share `features.py` (~60 pure feature-derivation fns; `from features import *`). Both do
(1) feature-eng → wide numeric per-VIN matrix with `TargetFlag`, (2) per-milestone NN.
`pms_model.py` holds everything the *current* train/score/eval scripts share: `BinaryClassifier`,
`add_milestone_flags()`, `derive_inference_features()`, `build_inference_matrix()`,
`load_artifacts()`, `predict_proba()`, `resolve_path()`, plus `LEAK_COLS` / `DERIVED_FEATURES` /
`HAS_X_EXCLUDE` / `DROP_RFM`. Edit the policy there, not in the three callers.
`due_date.py` owns cohort dating: `first_srv_date()`, `burn_rate_date()`, `earliest_date()`.
Data flow: `data/*.csv|.xlsx` (EDA master + Service History + RFM / Appointments / VHC / Digital
sessions) → merged feature CSVs → artifacts under `models/`.

### Two silent-corruption traps — always use the shared parsers
Both produced wrong numbers that shipped into docs before being caught. Neither raises.

- **Dates: `date_utils.parse_dates()`, never bare `pd.to_datetime`.** The raw files are *mixed
  convention* — EDA / Service History / VHC are **day-first** (`12/11/2024` = 12 Nov), Appointments
  and anything round-tripped through `to_csv` are **ISO**. Parsing day-first data without
  `dayfirst=True` silently swaps day and month whenever the day is ≤12; passing `dayfirst=True` to
  ISO-with-time breaks it the other way (on pandas 3.0.3,
  `to_datetime('2026-08-09 00:00:00', format='mixed', dayfirst=True)` → 2026-09-08). `parse_dates()`
  sniffs per column, which is the only thing that actually knows. This bug is what produced the
  phantom "history runs to 2026-12-06" claim — it really ends **2026-06-29**.
- **Service number: `features.resolve_service_num()`, never `int()` on `Description`.** The same
  20k service is spelled `'11-20'` (July extract), `'Nov-20'` (Excel date damage, most of the main
  history) and `'11_20'` (a manual repair, Q1 2026 only, 1,967 rows). PEP 515 makes `int('11_20')`
  return **1120**, so Q1 2026 — the exact window every test set is built on — had 481 of 951
  vehicles mislabelled "did not turn up". `Service_Code` is now the source of truth and
  `Description` only the fallback. **The `Service_Num == 0` ⇒ non-PMS convention is load-bearing**
  (12 sites in `features.py` test it); a naive "just use `Service_Code`" change empties every
  non-PMS feature — `tests/test_service_num.py` check B exists to catch exactly that.

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
  **Until 2026-10-16 the saved "best" model was really the last epoch** (state_dict kept by
  reference; now deep-copied) — every promoted model carries that. See `### D′ v2 retrain`.
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
- `eval_alan_tests.py` — **RETIRED 2026-08-12**, evaluation,file removed in the commit on 14/09/2026 loop commented out. It read the legacy
  `models/{m}k/training/` artifacts and fell back to the now-removed
  `models/{m}k/selected_features.json`. Use `eval_test_metrics.py`.

### Legacy / monolith pipeline
- Monoliths (pre-`features.py`, heavy duplication): `monoliths/pmstrainfeatureEng_6.py` (training, `main()`),
  `monoliths/predservicemil_4.py` (prediction; **executes at import**).
- `python monoliths/train_milestone.py <milestone>` — argv trainer over the monolith's output →
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

### Regression tests — run these before training anything (gate G2)
No pytest. Two plain scripts, run from repo root, **exit 0 = pass**, each prints PASS/FAIL with the
observed value per check:

```
$env:PYTHONPATH = "."
python tests/test_dayfirst.py      # expect "41/41 passed"  -- every date column in every data file
python tests/test_service_num.py   # expect "31/31 passed"  -- Service_Num across all three spellings
```

Each guards one of the two silent-corruption traps above, in four layers: unit (hand-written values
with a known answer) → file boundary (every real data file, every date column) → code path (the
actual call sites) → invariant (the headline fact the bug falsified — history ends 2026-06-29;
`Service_Num == 0` still means non-PMS; 20k Q1 positive rate back near ~82%). **If either fails,
stop and report — do not train**; that is Step 3 of `MILESTONE_ROLLOUT_RUNBOOK.md`.
They read the full data files, so they take a couple of minutes.

### Leak screen (gate G6)
```
python leak_screen.py <M> --train refactored_test_dir/final_processed_<M>k_candidates.csv \
  --test test_sets/test_<M>k_Q1_2026_cand.csv \
  --model-dir models/<M>k_cand_v3 --features models/selected_features_<M>k_cand_v3.json \
  --out models/<M>k_cand_v3/leak_screen_Q1.csv
```
Two read-only diagnostics: per-feature train-vs-test single-feature AUC (**AUC is RAW, not folded
above 0.5 — below 0.5 means inverted**), and grouped permutation importance by feature family.
Gate: `test - train >= 0.20` with `test >= 0.65` → STOP, `>= 0.10` → flag. Omit `--model-dir`/
`--features` for the single-feature screen alone. **Read the G6 BLIND SPOT gotcha before acting on
any STOP** — coverage drift mimics the signature. Results land at repo root as
`leak_screen_{m}k_q1*.csv`.

### Provenance
`write_manifest.py <artifact> [--inputs ...] [--cmd "..."]` writes `<artifact>.manifest.json` with
SHA256 of the artifact and every input, git HEAD, the command line, and all `PMS_*` env vars.
Exists because the 2026-08-19 label regression was a **silently swapped data file**, which git
history could not catch. `provenance_freeze.json` is the 2026-09-08 hash freeze of the seven
`data/*.csv` inputs; gate G1 checks the current files against it. Manifest every matrix, test set
and `best_model.pt` — the runbook has the exact invocations.

### Explainability — retired 2026-11-26
`pms_explain.py` / `explain_server.py` (SHAP over the old `production_models/{m}k/` NN, plus a
localhost viewer) were intermediate tools for finding data anomalies. Moved with `add_vehicle_key.py`
and `q1_due_rule_experiment/avg_interval_no_invoice.py` to
`c:\Techmax\cwf\backups\2026-11-26_cleanup\08_anomaly_scripts\`. What carries over: SHAP of a
`BinaryClassifier` is in **log-odds** (`base_value + sum(shap) == logit`), and contributions read
relative to the cohort used as background — change the cohort and the numbers legitimately move.

### Cohort-vs-reality validation
`analysis/july_candidates_overlap.py` / `analysis/july_overlap_diagnose.py` — score-free check of the candidate
*list* against a July 2026 service log that was **not** in the base history (which stops
2026-06-29). The first reports recall/precision of the cohort per milestone, raw and
"open" (minus already-completed); the second explains the misses, splitting **reachability** (the
`Last Service - PMS == '-'` and invoice-date filters cap recall before any date is computed) from
**timing** (due-date bias/noise, and what a wider window would buy). Both run with no arguments.

### Other
Verification otherwise = run the eval scripts above and compare metrics.
`eval_test_metrics.py` is the quickest full check (all 9 milestones, Q1+Q2, ~2 min). Until
2026-08-12 it printed `No results.` and exited silently because it looked for `testdataq1_q2/` at
the repo root; if it ever prints that again, the test files moved.
Several `*test*`/analysis scripts named in older docs (`test_pms_mil.py`,
`function_refactor_test.py`, `analyze_30k_drop.py`, `predict_milestone.py`, `prepare_data_feed.py`)
have been deleted — check a file exists before referencing it.

Expected cohort reconciliation test cases sit in `test/` (`eval_expected_cohort.py`,
`eval_q1_expected_cohort.py`, `hindcast_actual_turnups.py`, `convert_and_eval_milestones.ps1`) to check
reconciliation for the present codebase against Q1 and Q2 sets.

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
  [monoliths/predservicemil_4.py:2131](monoliths/predservicemil_4.py#L2131)). Keep it available for demonstration only.
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
  (+0.1018). `monoliths/pmstrainfeatureEng_6.py:2650` still has the bug — legacy monolith, not on the current
  path. Note the two pipelines disagree on the mileage term (training `Last Service Mileage`,
  `monoliths/predservicemil_4.py:2680` `Last PMS Mileage`) — unresolved.
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
- **COMPANY RULE: one VIN can hold several `Vehicle_Key`s — on resale the key's trailing sequence
  is bumped and EDA carries a row per key. Always anchor on the EARLIEST evidence across all of
  them. FIXED 2026-09-09** (`due_date.first_srv_date()`), surfaced while auditing the **30k Q3
  2026** candidate cohort.

  When a vehicle changes owner the group re-issues the key with the sequence incremented
  (`<pre>-<base>-0` → `<pre>-<base>-1`) — a **company-wide identifier convention, not a data error**.
  EDA keeps both rows, and each carries its own `First Service Date`, which for the new key is the
  *new owner's* first visit, not the vehicle's.

  **A BARE trailing dash (`<pre>-<base>-`, no digit) = sold by an EXTERNAL dealer but serviced with
  our client.** 70,214 of 130,759 EDA rows (53.70%); 69,700 VINs carry only bare keys. Nothing
  special about them — **do not prioritise or special-case them**. They are the plain majority
  case. Structurally the key always matches `<pre>-<base>-<seq>` (130,759/130,759 conform);
  `seq` runs `<empty>` 53.70%, `0` 41.8%, `1` 4.2%, `2` 0.3%, `3` 32 rows, `4` 2 rows. A bare
  `seq` and an explicit `0` NEVER share a base (0 cases), and where a bare key sits alongside a
  numbered one on the same base (443 groups) the bare row is always the older record
  (260/260 on both First and Last Service Date) — so sorting `''` as 0 orders them correctly.

  **Insight from one traced resold vehicle (keys `-0` → `-1`):** the old per-row rule dated its 30k
  from each key separately — 2027-08-07 from the new key's invoice, 2026-10-19 from the old key's
  `First Service Date` (really its 10k visit) — while its true first service (2024-07-03) gives
  **2026-01-03**; it came in 2026-02-10, so the old rule missed by 8 and 20 months and listed the
  same vehicle **twice, in two different quarters**.

  **Picking the earlier EDA row is NOT enough** — the older key's `First Service Date` (2025-04-19)
  is that vehicle's *code-10* visit, not its first service (2024-07-03). Raw service history has to
  be read directly. But service history alone is also wrong: for **21.8%** of VINs a genuine
  `Invoice date` legitimately precedes the first service and is the better zero-point. So the rule
  is the min over **all** evidence:

  ```
  FirstSrvDate = min( every key's Invoice date,
                      every key's First Service Date,
                      min(Service_Date) in raw service history )
  ```

  Scale: **1,102 VINs** in `EDA_Q2-2026` have >1 row (**329** survive the `Last Service - PMS != '-'`
  filter). Months from due date to actual turn-up, completers, schedule model:

  | | 20k | 30k | 40k | 60k |
  |---|---|---|---|---|
  | median old → new | -4.2 → **-3.4** | -6.5 → **-5.8** | -9.3 → **-8.4** | -15.1 → **-14.1** |
  | p10 old → new | -10.3 → **-8.6** | -14.1 → **-12.9** | -22.8 → **-19.4** | -30.7 → **-29.2** |

  Two "which row wins" questions, answered **differently** in `_apply_first_srv_date()`
  ([pmstrainfeatureeng_refactored.py](pmstrainfeatureeng_refactored.py)): the **anchor** takes the
  earliest date across keys, the **feature row** takes the *newest* key (`sort_values("Last Service
  Date").drop_duplicates(keep="last")`) because after a resale only the new key carries live
  `Last Service - PMS` / `Last Service Mileage` — on the VIN above the new key reads 30k while the
  old still reads a stale 20k. The previous `.drop_duplicates(subset="VIN")` kept whichever row came
  first in the CSV, which is neither. `--strict-invoice-guard` is now per-VIN too: a VIN is kept when
  **any** of its keys has a sale date.
  `prepare_test_set.py` never deduped at all, so test cohorts carried duplicate VIN rows (3 in 30k
  Q3, 6 in 20k Q1); it now calls the same helper. **Only the `candidates` path is converted** —
  `create_test_cohort()`'s `ever`/`history` branches still use the old per-row rule, so every
  published `ever`-mode metric stays reproducible.
  Effect on 30k Q3 2026 (`--grace-days 15`): 2,216 rows → **2,133**, duplicates 3 → **0**,
  519 dropped / 439 added / 1,694 retained (**23% churn**), positives 728 (32.9%) → **482 (22.6%)**.
  Any cohort built before 2026-09-09 must be rebuilt before its metrics are compared to a new one.
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
  shortlist used to zero-fill 67–78% of every model's inputs and still run clean (the per-milestone
  breakdown lived in `REFACTOR_LOSSES.md`, deleted 2026-10-07 — the 67–78% range is the finding worth
  keeping). Never re-add a feature filter to `prepare_test_set.py` — the
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
- **Cleanup 2026-11-26.** 467 finished-experiment items (1.1 GB) were MOVED, not deleted, to
  `c:\Techmax\cwf\backups\2026-11-26_cleanup\`: experiment matrices, ~75 model families,
  `_dprime_v3` / `_cal` / `_cand_b45` / `_h45` test sets, `dprime_v2/v3/`, the 20k Q3 cohort folders,
  `reconcile_q3_2026/`, `cache/`, `frames_20k.pkl`, root diag/log files. Its `README.txt` says what
  each group served; `moved_items.csv` lists original paths — restore = move back to the same path.
  Paths in this file that point at those items now live there. Kept: `data/`, `Q2/`, `Q3/`, `later/`,
  `legacy/`, `salman/`, `predictions/`, production + plain + `_cand_v3` + D′ v2 matrices/models/test
  sets. `.gitignore` now also covers `production_models/`, `test_sets/`, `cache/`,
  `**/pruned_*.json`, `.agents/`, `.vibe-wise/` and `q1_due_rule_experiment/*.json`.
- **The GitHub repo is PUBLIC.** Never commit customer identifiers (VINs, `Vehicle_Key`s with visit
  dates). The `q1_due_rule_experiment/*.json` files hold them and are ignored for that reason.
- `test_sets/` — `prepare_test_set.py` output; only 20/30/40/50k exist.
- `predictions/{m}k/` — `score_milestone.py` output (`scored_*.csv`) + legacy prediction datasets.
- `tests_alan/` — `Q1/`,`Q2/` legacy test features + the real `traindataq1_q2/`, `testdataq1_q2/`.
- `refactored_test_dir/` — current feature-eng output (`final_processed_{m}k.csv`, MI scores).
- `validatecode/` — legacy monolith intermediates; `monoliths/predservicemil_4.py:2307` *reads*
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
refreshes feature values without moving a single label. **`Service_History_Q2-2026.csv` runs to
2026-06-29 — the name is accurate** (only data up to June was loaded). Every date column in both
files maxes there: `Last Service Date`, `Last Service Date - PMS`, `Invoice date`,
`First Service Date`. An earlier revision of this file claimed it "actually runs to 2026-12-06, not
Q2 — the name understates its coverage"; that is **FALSE** (checked 2026-09-09 under both day-first
and month-first parsing) and the "latest 2026-12-06" figure in the labelling section above inherits
the same error. Consequence: **Q1 2026 is the only fully-observed outcome window**, Q2's
`--grace-days 15` band runs 16 days past the data, and Q3/Q4 have no outcome coverage at all —
which is by design, see the note below.

### Q3/Q4 cohorts are FORWARD PROJECTIONS — do not score them
A cohort whose window opens after the data ends is a **candidate list**: "these vehicles are
projected due for the milestone next quarter." That is the deliverable, not a test set. The
builders still stamp a `TargetFlag` on them, and on a future window that column can only ever mean
"already completed before the window opened" — 100% of positives are `EarlyCompleter` by
construction. Measured 2026-09-09 on the 30k candidates cohorts:

| cohort | rows | pos | early | genuine in-window |
|---|---|---|---|---|
| Q1 2026 | 1,998 | 1,291 (64.6%) | 43.7% | 727 |
| Q2 2026 | 2,403 | 1,526 (63.5%) | 55.0% | 686 |
| Q3 2026 | 2,133 | 482 (22.6%) | **100%** | **0** |
| Q4 2026 | 1,845 | 311 (16.9%) | **100%** | **0** |

Never quote a metric off a Q3/Q4 file — read `prob_turnup` from it instead. The 43.7%/55.0% on the
observed quarters is the normal rate; a 100% reading is the tell that the window outran the data.

## Production rollout — state as of 2026-09-10

### `production_models/{m}k/` is the promotion target
**The rollout is complete — all nine milestones (20k–100k) are promoted**, every one built
`--label-mode candidates --grace-days 15` at git HEAD `60b6372`, on the `provenance_freeze.json`
data. `salman/expected_cohort_{m}k_Q{3,4}_2026.csv` exists for all nine. Each dir holds exactly
`model.pt`,
`scaler.joblib`, `imputer.joblib`, `selected_features.json`, `threshold.json`, `metadata.json`
(+ `model.pt.manifest.json`). `metadata.json` carries full provenance, validated Q1/Q2 metrics with
confusion matrices, the 3-seed comparison, all eight gate results and known caveats — **read the
target milestone's `metadata.json` before touching it**, the caveats are milestone-specific.

| milestone | seed | Q1 AUC | Q2 AUC | features | gates |
|---|---|---|---|---|---|
| 20k | 123 | 0.9290 | 0.9093 | 160 | seed 42 excluded (G7 spread 0.077) |
| 30k | 42 | 0.9446 | 0.9595 | 191 | all clean |
| 40k | 7 | 0.9109 | 0.9166 | 195 | **G6 STOP overridden — see below** |
| 50k | 7 | 0.9442 | 0.9456 | 200 | all clean |
| 60k | 42 | 0.9289 | 0.9309 | 202 | 2 G6 flags (coverage drift, see below) |
| 70k | 42 | 0.9322 | 0.9383 | 204 | 3 G6 flags; **F1 peaks at 0.65, ships 0.5** |
| 80k | 7 | 0.9453 | 0.9291 | 209 | **G6 STOP overridden — false positive, measured** |
| 90k | 7 | 0.9692 | 0.9407 | 211 | G6 STOP overridden (coverage drift); G7 tightest in rollout |
| 100k | 123 | 0.9638 | 0.9463 | 212 | G6 STOP overridden; **weakest precision/overlap — see below** |

**90k/100k caveats, from their `metadata.json`:** threshold 0.5 ships **unswept** for both (the
sweep that established 0.5 covered 20k/30k only, and 70k already peaked at 0.65), and neither has a
VIN-level audit. 100k is the weakest promoted model at the operating point — precision 0.59/0.54
against base rates 20.2%/17.3%, mean Jaccard 0.552 vs 20k's 0.847 — so its *ranking* is excellent
(AUC 0.96/0.95) but the 0.5 cut point is badly placed. Sweep before either list drives customer
contact. 90k's Q1 recall of 0.9912 (3 FN of 340) reproduces on independent re-score and is G6-clean
after era-matching, but confirm it against real Q3 2026 outcomes.

Feature counts drift above the documented `[100,200]` G3 band from 60k up (202/204/209/211/212).
Treated as
passing — that band guards against the sub-80 collapse mode, not as a ceiling — but recorded in each
`metadata.json` rather than rounded away.
**Seed selection is a two-factor rule** (best AUC across *both* quarters + best Jaccard overlap on
*both*). When the two disagree, compare each margin against the measured G7 seed spread: on 80k,
seed 42 led AUC by 0.0034/0.0022 while the seed spread was 0.0034/0.0076 — i.e. inside noise — so
seed 7's 0.038/0.025 overlap lead (114 fewer false positives) decided it.

- **NEVER let a `best_model.pt` exist in a `production_models/{m}k/` dir.** `load_artifacts()`
  prefers `best_model.pt` over `model.pt` whenever both are present, so a leftover copy silently
  scores a **different, superseded model with no error**. This actually happened on 30k
  (2026-09-09): a stale round-1 `best_model.pt` shadowed the freshly promoted `model.pt` and the
  Q3/Q4 lists were scored with the wrong weights. Caught only because an audit subagent's numbers
  didn't reconcile. After any promotion, `ls production_models/{m}k/` and confirm.
- `score_milestone.py` and `pms_model.py:load_artifacts()` accept **either** filename
  (`best_model.pt` for `models/` staging, falling back to `model.pt` for production) since
  2026-09-09 — so `--model-dir production_models/{m}k` works directly, no wrapper script.
- Deliverables for the business live in `salman/` as
  `expected_cohort_{m}k_Q{3,4}_2026.csv`, 8 columns: `Predicted Quarter, Predicted Year,
  Predicted Service Mileage, Predicted Service_Code, NextServiceDate, VIN, predicted_prob,
  predicted_flag`. `NextServiceDate` must be plain `YYYY-MM-DD` — the raw `Next{M}K_Due` column
  carries full timestamps (`2026-12-19 05:35:59.306525404`), so always
  `pd.to_datetime(..., format='mixed').dt.strftime('%Y-%m-%d')`.
- When reshaping a scored file, **use the scored output's own `Next{M}K_Due` column**. Merging it
  against the test set's copy of the same column produces a silent `_x`/`_y` suffix collision and a
  `KeyError` (or worse, a wrong column). Hit twice on 2026-09-09.

### `MILESTONE_ROLLOUT_RUNBOOK.md` — the step-by-step for a new milestone
Exact commands for matrix → test sets → 3 seeds → gate battery → promotion → Q3/Q4 → delivery, with
three checkpoints where a human decides (seed selection, standalone verification, filter results).
Written so the milestone owner can run it without burning agent context. Keep it in sync with any
pipeline change.

### Forward quarters use a LATER feature cutoff than training — `--train-cutoff 2026-06-30`
Standard as of 2026-09-09. Training and the Q1/Q2 test sets stay at `2025-12-31`; **Q3/Q4 test sets
use `2026-06-30`** (the latest real service data). Rationale: training computes each cohort's
features just before *its own* due window, so a 2025-12-31 cutoff on a Q3/Q4 window is a 6–12 month
train/test mismatch, not a safety margin. Cohort membership is unchanged either way — only feature
values move. Measured on 30k Q4: `has_20` went **8.7% → 47.2%**, `PMS_Count_Prior` 1.01 → 2.09, and
flagged positives **103 → 355**; 30k Q3 went 239 → 728.
**Q1/Q2 must NEVER move to a later cutoff** — their outcomes fall inside 2026 H1, so that is a real
leak. Q4 stays ~3 months stale by data limit (history ends 2026-06-29) and will understate until
newer history lands.

### `stale_completion_filter.py` — forward lists only, never training
`python stale_completion_filter.py <milestone> <window_start> <expected_cohort.csv>` drops any VIN
whose raw history already shows a `Service_Num == milestone` record dated **before the window
opens**, overwrites in place, and writes a sibling `_removed_stale_completions.csv` audit trail.
Applies **only** to the forward Q3/Q4 deliverables. Training/eval keep `EarlyCompleter` rows as
positives — that is the deliberate candidates-mode design, do not "fix" it there.
It removes 12–26% of rows but **40–54% of the flagged positives**, because the model was trained to
score already-completed vehicles highly. That is expected, not a bug.

### Q3 2026 outcomes landed — the first genuinely forward test, and every model degrades
Measured 2026-10-06 by `analysis/reconcile_quarter.py` against `Q3/Service_History_Q3-2026.csv` (20,726 rows,
16,028 VINs, 2026-07-01..2026-09-29). The Q3 lists were built with `--train-cutoff 2026-06-30`,
**before Q3 happened**, so this is the first time any promoted model has been scored on outcomes that
were not in the data when it was built.

| ms | Q1 AUC | Q2 AUC | **Q3 AUC** | drop |
|---|---|---|---|---|
| 20k | 0.9290 | 0.9093 | **0.6191** | −0.31 |
| 30k | 0.9446 | 0.9595 | **0.6728** | −0.27 |
| 40k | 0.9109 | 0.9166 | **0.6722** | −0.24 |
| 50k | 0.9442 | 0.9456 | **0.8044** | −0.14 |
| 60k | 0.9289 | 0.9309 | **0.8449** | −0.08 |
| 70k | 0.9322 | 0.9383 | **0.8516** | −0.08 |
| 80k | 0.9453 | 0.9291 | **0.8554** | −0.09 |
| 90k | 0.9692 | 0.9407 | **0.8980** | −0.07 |
| 100k | 0.9638 | 0.9463 | **0.8909** | −0.07 |

Q1/Q2 showed a flat ~0.91–0.97 across all nine; Q3 shows a clear gradient — worst at low milestones,
mildest at high. **Treat the Q3 column as the honest estimate of forward performance** and the
`metadata.json` Q1/Q2 figures as optimistic until the hypothesis below is tested.

**Leading hypothesis, NOT yet proven — early completers inflated Q1/Q2.** Those cohorts are 43.7% /
55.0% `EarlyCompleter`: vehicles that had already completed the milestone before the window opened,
so the completion sat inside the history the features were computed from. Impossible on a forward
quarter by construction. **To test: re-measure Q1/Q2 AUC on the not-early subset only** and see
whether it lands near the Q3 figure. This decides whether the published metrics need retracting.
Still open.
**Partly tested 2026-10-14 (20k only):** the production model on the NOT-early Q1/Q2 rows still reads
AUC **0.903 / 0.891** vs Q3 **0.619** — so early completers are NOT the main inflator. Stronger
candidate: label-dependent feature code, see `### 20k no-show feature review` below.

**Where recall is actually lost (three-bucket funnel, per milestone).** Recall of the delivered lists
was **32–47%**:
- **Structural ceiling (outside the EDA gate): 6–41 vehicles**, except 20k at 159 (9.3%). Small.
- **Cohort error (in gate, due, never listed): 268–945 per milestone** — roughly half of all real
  completers. The due-date projection is the bottleneck. No threshold change recovers these.
- **Milestone assignment is NOT the problem.** Any-PMS precision barely beats exact-match (20k 64.7%
  vs 61.1%), so only ~47 listed vehicles came in for a *different* service. The list is not off by
  one milestone — it is missing people entirely.

Lift over each list's own base rate rises monotonically with milestone, 20k **1.05×** → 100k
**3.46×**: at 20k the flag barely beats contacting the whole list, while 100k flags too many (221 FP
against 110 TP). **The model earns its keep at HIGH milestones, not low ones.**

`analysis/reconcile_quarter.py` takes `--cohort-dir`, `--milestones`, `--out-dir`; it reads delivered lists and
raw history only — no model, no feature engineering. The EDA gate deliberately stays
`data/EDA_Q2-2026.csv`: gating on the Q3 snapshot would use post-quarter state to decide who was ever
eligible. `Q3/EDA_Q3-2026.csv` is **Vehicle_Key-keyed with no VIN column** — cross-check only, never
a join key.

### 20k no-show feature review on real Q3 outcomes (2026-10-14) — findings only, nothing changed
Goal: find features that push listed no-shows DOWN the ranking. Population: the delivered 20k Q3 list
with real outcomes (`test_sets/test_20k_Q3_2026_REAL.csv`, 1,309 rows, 800 came / 509 did not); also
the not-early Q1/Q2 rows and the training rows (`Next20K_Due <= 2025-12-31`). Only Q3 is a fair
forward test.

**1. Label-dependent feature code — the real finding.** `process_service_data()` builds
`LastNonPMSMileage` / `Last PMS Mileage` through **different code paths for `TargetFlag==1` and
`==0`** (`get_last_nonpms_before_targetpms` vs `get_last_nonpms_mileage`,
[pmstrainfeatureeng_refactored.py:845-850](pmstrainfeatureeng_refactored.py#L845-L850), and the
`TargetFlag == 1` overwrite at [:1000-1022](pmstrainfeatureeng_refactored.py#L1000-L1022));
`derive_pms_service_intervals(serv, …)` at [:858](pmstrainfeatureeng_refactored.py#L858) gets the
**uncut** history (`serv`, not `serv1`); `PMS_Delay` (legacy) is extract-time EDA. Test sets run the
same code with their own `TargetFlag`, so Q1/Q2 inherit it. A forward list has no outcome yet, so every
row takes the `TargetFlag==0` path. Single-feature AUC (raw; <0.5 = higher value → more no-shows):

| feature | train | Q1 | Q2 | **Q3** |
|---|---|---|---|---|
| `PMS_Delay` | 0.25 | 0.28 | 0.20 | **0.50** |
| `Last Service Mileage` | 0.17 | 0.19 | 0.15 | **0.40** |
| `Avg_Service_Interval_PMS` | 0.69 | 0.59 | 0.64 | **0.50** |

**Do NOT drop them.** Removing all 9 (`PMS_Delay`, `Last Service Mileage`, `LastNonPMSMileage`, the
four `Avg_Service_Interval_PMS*`, `months_to_10k`, `months_base_to_10`) costs Q3 AUC **0.691 → 0.626**
(paired bootstrap 95% −0.09…−0.03) — together they still carry forward signal. The fix is one code
path for every row, cut at the cutoff, then rebuild + retrain. **Not done**; it changes every
published 20k number.

**2. Own-rhythm ratio** = (quarter end − last PMS) ÷ the vehicle's own usual gap (avg-interval leg).
Best single no-show signal on Q3: AUC **0.77**, also 0.77 inside the flagged rows (best existing
feature 0.66). But only 0.58 in training and it **flips on Q2** (0.42 — 3-month-stale cutoff, the
same set-up the Q4 lists use). As a model feature: +0.007 Q3 AUC (CI 0.002–0.012), +1.3pp precision
(CI spans 0) — training barely sees it. As a post-score rule (drop flagged rows with ratio < 1.25):

| | flagged → kept | came in | real visits lost |
|---|---|---|---|
| Q3 | 600 → 531 | 64.3% → **69.9%** | 15 of 386 |
| Q1 | 1,232 → 1,131 | 79.6% → 81.9% | 55 of 981 |
| Q2 | 903 → 897 | no change | 4 |

1.25 is a judgement line read off Q1/Q3, not tuned. Rows with no rhythm (no avg-interval leg, ~20%)
are left alone.

**3. Learner side finding — bigger than any feature change.** Same 160 features in an untuned
`HistGradientBoostingClassifier` (3 seeds) vs the production NN, Q3: AUC **0.619 → 0.691**, precision
at the same call volume **64.3% → 70.7%** (bootstrap NN − HGB: −0.10…−0.04 AUC, −9…−3pp). Not a
promotion candidate yet — no gates, no seed selection.

Mileage-projection, key-count and missing-step candidates added nothing. Sources:
`noshow_feature_review/noshow_features_20k.py` (feature build + AUC tables), `noshow_ab_20k.py`
(A/B, bootstrap, rhythm rule), CSVs alongside, `frames_20k.pkl` (gitignored cache, 123 MB).

### Threshold 0.5 — optimal on Q1/Q2, WRONG on forward data (corrected 2026-10-06)
The original finding (2026-09-09) swept 0.50→0.20 on the labelled **not-early Q1/Q2** population and
found **F1 peaks at exactly 0.50** for 20k (0.881) and 30k (0.885). That measurement stands for that
population.

**It does not transfer to a forward quarter.** Re-swept against real Q3 2026 outcomes: the 20k model
**under-flags badly at 0.5**. The earlier sweep was not wrong, it was measured on an easier
population — the same confound that makes Q1/Q2 AUC read 0.93 against Q3's 0.62.

Threshold chosen on a random half of the cohort and reported on the held-out half, because picking a
threshold on Q3 outcomes and then scoring it on the same Q3 outcomes is in-sample selection and
flatters the number by an unstated amount:

| holdout, 1,177 rows, base rate 0.483 | flagged | acc | prec | rec | F1 |
|---|---|---|---|---|---|
| shipped 0.50 | 532 | 0.540 | 0.526 | 0.492 | 0.509 |
| **0.12** | 992 | **0.583** | 0.539 | **0.940** | **0.685** |
| contact everyone | 1,177 | 0.483 | 0.483 | 1.000 | 0.652 |

Quote **0.685**, not the 0.704 the full cohort reports at 0.12 — that one is in-sample.
Note the flag only modestly beats contacting everyone (F1 +0.033, lift 1.12× vs 1.00×), so at a ~50%
base rate the list's value is mostly in *who is on it*, not in the flag.

**The ranking is informative only at the extremes.** Precision by probability decile on the 20k Q3
union+±45d cohort:

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| precision | **0.114** | 0.362 | 0.609 | 0.587 | 0.593 | 0.455 | 0.417 | 0.574 | 0.532 | **0.767** |

Top/bottom ratio 6.70×, but deciles 3–9 are flat and **non-monotonic** (0.42–0.61) — that is why AUC
is only 0.628 despite the spread. Operationally: work the top decile first, safely deprioritise the
bottom two, and treat the middle 70% as **unordered** — a probability gap inside it is not signal.

`production_models/{m}k/threshold.json` still ships 0.5 and was deliberately NOT changed; promoting a
new threshold is a promotion decision (see `MILESTONE_ROLLOUT_RUNBOOK.md`). **Only 20k has been
re-swept** — 30k–100k remain unexamined against forward outcomes.

### The cohort under-catches real demand — PARTLY SOLVED 2026-10-06 (Phase 4.2 finally run)
~3,831 vehicles completed 20k in 2026 H1 (~1,900/quarter), but the delivered 20k Q3 list held only
1,309 candidates post-filter. A vehicle only enters the list if its *projected* due date lands inside
the quarter, and that projection has real error (median |err| ~42 days against a 91-day quarter).
Threshold tuning cannot substitute for fixing that — confirmed, see the threshold section above.

**Two cohort-rule changes were measured against real Q3 2026 outcomes. Both help; neither touches the
model, the feature list or the weights.**

| arm | listed | overlap w/ real completers | list recall | AUC | extra rows per extra completer |
|---|---|---|---|---|---|
| production (`earliest`, pad 0) | 1,309 | 800 | 46.6% | 0.619 | — |
| union only | 1,478 | 879 | 51.2% | 0.607 | **2.14** |
| ±45d only | 2,180 | 1,114 | 64.9% | **0.642** | 2.77 |
| union + ±45d | 2,353 | **1,179** | **68.7%** | 0.628 | 2.66 |

1. **Union the two due-date legs instead of taking the minimum.** `due_date.earliest_date()` is a
   min, so it **discards** the other candidate: a vehicle whose schedule leg lands in Q3 but whose
   burn-rate leg lands in Q2 never reaches the Q3 list, despite one of its own two estimates putting
   it there. On the 20k Q3 cohort, **2,702 vehicles qualified on the schedule leg alone and 1,687 on
   the burn-rate leg alone — only 864 on both.** Listing on either leg is the cheapest change
   available (2.14 extra rows per extra completer) and the only arm whose F1 beats production.
2. **Widen membership to ±45 days** around the quarter. Sized against the ~42-day median error.
   Biggest absolute gain (+314 of the +379). **Label the outcome on the STRICT quarter, not the
   padded window** — otherwise arms stop being comparable and the reconciliation moves with the rule.

Together the "in-universe, due, never listed" bucket **halves, 758 → 379**.

**AUC RISES (0.619 → 0.628), which is the signal that these rows are legitimate.** Contrast the
rejected gate-dropping arm below, where AUC inverted to 0.447. Added vehicles that are genuinely due
but misdated remain rankable; already-completed vehicles do not.

The stale filter does much heavier lifting once padded — it removed **2,900 of 5,253** pre-filter
rows (55%), because padding pulls in many May/June-due vehicles who already completed. **Keep it on**;
without it this repeats the gate-dropping failure.

Cost is calls, not model quality: 2,353 contacts to reach 1,179 versus 1,309 to reach 800 — 2.00 per
completer instead of 1.64, with a marginal rate of 2.75 on the extra 379. That is a business call.

**Only 20k has been measured. 30k–100k are untouched**, and neither
`pmstrainfeatureeng_refactored.py` nor `prepare_test_set.py` has been changed — the arms were run
from a throwaway harness, so adopting this means adding `--due-rule union` / `--window-pad N` to both
builders (they hardcode `earliest_date` and no padding) and rebuilding the Q1/Q2 test sets under the
same rule, or there is no comparable baseline. See the `--date-model calibrated` precedent: a
different due-date rule selects a **different population**, so metrics are not comparable across rules.

### Dropping the cohort gates — MEASURED AND REJECTED (2026-10-06)
The filters were the other suspect for low recall. They are not the cause. Dropping
`Last Service - PMS != '-'`, the `Vehicle Service Status` filter and `stale_completion_filter.py`
took the 20k Q3 list from 1,309 to 2,426 rows and the overlap from 800 to **809** — **+1,117 rows for
+9 real completers**, with 755 extra false positives.

| | delivered | ungated |
|---|---|---|
| list recall / precision | 46.6% / 61.1% | 47.1% / **33.3%** |
| acc / prec / rec / F1 | .520 / .643 / .483 / .551 | .427 / .286 / .480 / .358 |
| AUC | 0.619 | **0.447** |

**AUC below 0.5 means the ranking is inverted on that population — no threshold recovers it.** Cause:
the model scored the added rows *higher* (mean prob **0.641** vs 0.468) because **439 of them had
already completed the milestone**, and it is trained to score completed vehicles highly. Of the 969
false positives: 439 stale completers, 316 with no PMS record at all, 214 neither.

Individual filter costs, for the record:

| filter | where | effect on 20k Q3 |
|---|---|---|
| `Last Service - PMS != '-'` | `prepare_test_set.py:43` | +549 rows, **+7 completers** |
| `Vehicle Service Status` | `prepare_test_set.py:86` | **no-op** — only excludes `No Services`, already removed by the filter above |
| invoice-date guard | `pmstrainfeatureeng_refactored.py:508` | **already relaxed** by default since 2026-09-01 |
| stale completion filter | `stale_completion_filter.py` | +451 rows, **+1 completer** |

The Q3 funnel counted 159 out-of-gate 20k completers, which is what made the gate look guilty — but
only **8** of them get a projected due date landing in Q3. The gate was never what excluded them.
`extract_k1('-')` returns 0, the existing non-PMS convention, so dropping the EDA filter does not
break `prepare_test_set.py:393` — that was the main technical risk and it is clear.
**Do not re-litigate relaxing these filters.** The due-date projection is the constraint.

### 40k's G6 STOP was overridden on evidence — the one non-clean promotion
`LastNonPMSMileage` scored train AUC **0.4173** vs test **0.6640** (gap 0.2467) — the documented
leak signature, sign-flipped between train and test. Two tests were run before overriding:
1. **Ablation** (`PMS_DROP="LastNonPMSMileage,Last Service Mileage"` — **both**, because
   `derive_inference_features()` rebuilds `Last Service Mileage` *from* `LastNonPMSMileage`, so
   dropping one alone leaves the same signal under the other name): leak screen went fully clean,
   but AUC fell to 0.8770/0.8719 — deltas **-0.034 / -0.045**, both over the 0.03 gate.
2. **Head-to-head on real outcomes**: where the two models disagree on Q1/Q2, the with-mileage
   model's unique flags were right **31.9%** (Q1) / **72.0%** (Q2) vs the ablated model's 6.4% /
   22.7%. Net correct calls **+114** vs **-53**.

Conclusion: the feature carries genuine signal; removing it is strictly worse. **But the
instability is real** — marginal precision swung 31.9%→72.0% between quarters, and in Q1 that was
*below* the 54.7% base rate. On the forward lists the two models overlap only 0.67 (Q3) / 0.44 (Q4)
Jaccard, disagreement correlating **-0.77** with `LastNonPMSMileage`. **Re-validate 40k as soon as
Q3 2026 outcomes land.** Same feature is an informational flag at 50k (gap 0.1738) and clean at 30k.

### G6 has a BLIND SPOT: feature-availability drift reads as a leak (measured 2026-09-10 on 80k)
`leak_screen.py` compares single-feature train-vs-test AUC. It does **not** account for a feature
whose data collection *began part-way through* the 2016-2025 training span — such a feature is a
structural zero on old cohorts, which dilutes its train AUC and manufactures a large gap against a
fully-covered 2026 test cohort. **That is not leakage.**

Caught on 80k, where `total_appointments_showed_up` triggered a hard STOP (train 0.5560 / test
0.7708, gap **0.2148**). Nonzero rate by training cohort year:

| 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | test Q1/Q2 |
|---|---|---|---|---|---|---|---|---|
| 0.0% | 0.0% | 0.1% | 21.8% | 74.4% | 82.8% | 91.1% | 96.6% | **96.7% / 100%** |

The appointments system did not exist before ~2021, so **63% of training rows carry a zero meaning
"not tracked yet"**, not "customer booked nothing". Re-measured era-matched (training cohorts 2022+
only):

| feature | raw gap | era-matched gap |
|---|---|---|
| `total_appointments_showed_up` | +0.2148 (STOP) | **+0.0413** |
| `no_of_service_appointments_booked` | +0.1866 (FLAG) | **+0.0127** |

Both fall below the 0.10 FLAG line, let alone the 0.20 STOP line.

**How to tell a real leak from this artifact:** a real leak *flips sign* — 40k's
`LastNonPMSMileage` reads train AUC **0.4173**, i.e. below 0.5, pointing the opposite way on train
vs test. Coverage drift does not flip sign (80k's reads 0.5560, above 0.5); it only weakens.
**Before treating any future G6 STOP as real, check the feature's nonzero rate by cohort year and
re-measure on the era-matched subset.**

This also retroactively explains the **appointment-feature flags at 60k (gap 0.1263) and 70k (0.1611,
0.1338)** — same two columns, same artifact. So the apparent "G6 flags rise with milestone"
(30k 0 → 40k 1 → 50k 1 → 60k 2 → 70k 3 → 80k 10) is **partly an artifact of the screen**, not
proof of degrading models. Recommended fix: have `leak_screen.py` print coverage-by-year next to
each AUC gap.

### The 10k-anchored due-date leg — MEASURED AND REJECTED FOR NOW (2026-09-10)
Business rule proposed: a customer who did their 10k six months ago is due for their 20k, so add
`10k_date + 6 months` as a third leg to `earliest(schedule, burn_rate)`. Measured end-to-end
against the **3,831 vehicles that actually completed their 20k in H1 2026**. Do not re-litigate
without reading all four findings.

**1. The 6-month step is wrong; the real interval is 4.3 months.** Median 10k→20k gap is **132 days**
over 44,528 vehicles with both records (p25 87d, p75 184d). A `10k + 6mo` rule is systematically
**+45 days late**. Tuned offsets, same-quarter hit rate: 4.5mo **46.0%**, 5.0mo 44.6%, 6.0mo 40.9%.

**2. A pure time-from-10k rule LOSES to the current rule** — because production already blends two
signals, one of which (burn rate) adapts to how hard each car is driven:

| | median abs err | within ±45d | same quarter |
|---|---|---|---|
| current `earliest(sched, burn)` | **42d** | **52.4%** | 45.4% |
| `10k + 4.5mo` alone | 48d | 46.0% | 46.0% |
| `10k + 6.0mo` alone | 57d | 42.7% | 40.9% |

**3. Adding it as a THIRD leg improves same-quarter accuracy but churns ~45% of the cohort.**
`earliest(sched, burn, 10k+4.5mo)`: same-quarter **45.4% → 48.6%**, |err| 42d → 44d. But on the
labelled quarters:

| | current | proposed | overlap |
|---|---|---|---|
| Q1 2026 | 2,239 rows / 73.9% pos | 2,020 / 71.6% | 661 dropped, 442 added, **Jaccard 0.589** |
| Q2 2026 | 1,945 rows / 67.5% pos | 1,865 / 65.9% | 616 dropped, 536 added, **Jaccard 0.536** |

Volume by quarter: Q1 −219, Q2 −80, Q3 **+235**, Q4 −153 — the leg mostly *moves* vehicles between
quarters rather than adding them, and **shrinks Q1/Q2**.

**4. REJECTED for now, on three grounds:** (a) it makes the lists smaller, which is the opposite of
the volume goal; (b) at ~45% churn the new AUC is not comparable to the published 0.9290/0.9093, so
a drop could not be attributed to model vs population; (c) **the +3.2pp accuracy gain is measured on
completers only** — vehicles that actually turned up — so it shows timing accuracy for known
completers, NOT that the rule better identifies who will complete. That is survivorship bias and it
was not controlled for.
**Revisit at the next full rebuild** (when newer data forces regeneration and the churn cost is
already being paid), and generalise as `(M-10) completion + 4.5 months` rather than hardcoding 20k.

### EDA's `Next Service Date` is NOT a usable due-date leg — omit it (2026-10-08)
Considered as a fourth candidate leg alongside `schedule`/`burn-rate` (and the rejected 10k-anchored
leg above) for union-based cohort construction. Rejected on data-availability grounds, no build
needed: `Next Service Date` is a **recurring, advisor-entered field** — the service advisor sets it
at *every* visit, roughly 6 months ahead of *that* visit, not tied to any particular future milestone.
EDA is the current-state master sheet, so it carries only the **latest** value per VIN; there is no
append-only log of `service visit → quoted next-service date` pairs. Once a later visit overwrites
the field, there is no way to tell which service code the surviving value was originally quoted
against — using it would silently mix "next 20k" and "next 30k" (etc.) quotes together with no way to
separate them. Omit this attribute from due-date leg candidates unless a visit-level log of
`(Service_Code, Next Service Date)` pairs is sourced separately.

### Third due-date leg — `due_date.last_completed_plus_months()` — MEASURED, clean recall win (2026-10-08)
Found by tracing one vehicle by hand: `schedule` said due 2025-07-08 (6 months
too early — it hadn't even had its 10k yet), `burn_rate` said 2026-11-24 (11 months too late — an
early low-mileage stretch dragged its pooled rate down), but it actually completed 2026-01-02. A
third rule — **last completed milestone below the target + a flat 6 months, no rate** (same "hi"
anchor `burn_rate_date()` already uses for its lower endpoint) — lands on 2026-01-22, 20 days off.
Unlike `schedule` (anchors on the vehicle's very first service; drift compounds every milestone
since) or `burn_rate` (pools a rate over the vehicle's *entire* history to "hi"), this only has to be
right about the next interval, and resets the clock at the vehicle's most recent actual visit.

Scoping check before building it: of the 444 (20k) / 460 (30k) real Q1-2026 completers the existing
two-leg `union(schedule, burn_rate)` rule missed entirely, this leg alone would additionally have
placed within a ±45-day window **36% of the 20k misses, 61% of the 30k misses** — stronger on 30k,
consistent with 30k's schedule-keeping features not being broken the way 20k's are (see the
train/test inversion gotcha above).

Built into `due_date.py` as `last_completed_plus_months()`; `earliest_date()` was generalized from a
fixed 2-argument `min(schedule, mileage)` to `*legs` (variadic, backward compatible — every existing
call site still passes exactly 2 positional args) so a third leg doesn't need a signature change at
every caller. Added as a third leg to the union rule (`build_unionstale_20k.py
--include-last-milestone-leg`) and scored with the **unmodified** promoted `production_models/20k`
— no retrain — measured on strict-quarter ground truth, 20k Q1 and Q2 2026 (each cell Q1 / Q2):

| | listed | overlap | recall | precision | AUC |
|---|---|---|---|---|---|
| production (2-leg min, baseline) | 2,231 / 1,923 | 915 / 834 | 47.0% / 44.2% | 41.0% / 43.4% | 0.642 / 0.668 |
| 2-leg union+stale, pad0 | 1,852 / 1,624 | 1,073 / 943 | 55.1% / 50.0% | 57.9% / 58.1% | 0.867 / 0.882 |
| **3-leg union+stale, pad0** | 2,273 / 2,078 | 1,310 / 1,233 | **67.3% / 65.4%** | 57.6% / 59.3% | 0.854 / 0.866 |
| 2-leg union+stale, pad45 | 3,190 / 2,827 | 1,502 / 1,378 | 77.2% / 73.1% | 47.1% / 48.7% | 0.869 / 0.855 |
| **3-leg union+stale, pad45** | 3,588 / 3,341 | 1,662 / 1,632 | **85.4% / 86.6%** | 46.3% / 48.8% | 0.864 / 0.857 |

**Recall rises 8–15 points at essentially no precision or AUC cost** (both move <1.6pp / <0.016
either way — inside noise). That is unlike the schedule/burn-rate union itself, which buys recall by
spending precision (see the cohort-rule table above). Cost efficiency: **1.6–2.5 extra rows listed
per extra real completer caught** — cheaper than the plain union leg (9.6, pad0 arm) or the
padding-only lever (4.7) — the most efficient lever measured so far on 20k.

Scored with the unmodified promoted model deliberately: this is a cohort-construction change, not a
model change, consistent with the retrain findings elsewhere in this file (retraining was not shown
to beat just rescoring a better-built cohort with the existing production weights).

**Not yet adopted into the training matrix or the production pipeline.**
`pmstrainfeatureeng_refactored.py` and `prepare_test_set.py` still hardcode the two-leg
`earliest(schedule, burn_rate)` rule everywhere except this throwaway harness.

**Extended to 40k and 50k (2026-10-08) — same pattern holds, with one real caveat.** Full
build→score→reconcile (not just the scoping check 30k got), scored with the unmodified promoted
`production_models/{m}k`, strict-quarter ground truth, Q1 and Q2 2026 (each cell Q1 / Q2):

| 40k | listed | overlap | recall | precision | AUC |
|---|---|---|---|---|---|
| production (baseline) | 2,172 / 2,096 | 776 / 858 | 46.6% / 46.3% | 35.7% / 40.9% | 0.760 / 0.739 |
| 2-leg union+stale, pad0 | 2,225 / 2,193 | 830 / 920 | 49.8% / 49.6% | 37.3% / 42.0% | 0.866 / 0.865 |
| **3-leg union+stale, pad0** | 3,036 / 3,213 | 1,110 / 1,290 | **66.6% / 69.5%** | 36.6% / 40.1% | 0.864 / 0.880 |
| 2-leg union+stale, pad45 | 4,059 / 3,997 | 1,196 / 1,339 | 71.7% / 72.2% | 29.5% / 33.5% | 0.869 / 0.869 |
| **3-leg union+stale, pad45** | 5,190 / 5,951 | 1,484 / 1,659 | **89.0% / 89.4%** | 28.6% / 27.9% | 0.866 / 0.882 |

| 50k | listed | overlap | recall | precision | AUC |
|---|---|---|---|---|---|
| production (baseline) | 1,801 / 1,970 | 511 / 652 | 39.4% / 42.5% | 28.4% / 33.1% | 0.823 / 0.737 |
| 2-leg union+stale, pad0 | 1,996 / 2,113 | 556 / 701 | 42.9% / 45.7% | 27.9% / 33.2% | 0.935 / 0.919 |
| **3-leg union+stale, pad0** | 2,918 / 3,320 | 821 / 979 | **63.3% / 63.8%** | 28.1% / 29.5% | 0.921 / 0.926 |
| 2-leg union+stale, pad45 | 3,881 / 4,017 | 875 / 1,077 | 67.5% / 70.2% | 22.5% / 26.8% | 0.923 / 0.924 |
| **3-leg union+stale, pad45** | 5,262 / 6,739 | 1,102 / 1,327 | **85.0% / 86.5%** | 20.9% / 19.7% | 0.919 / 0.928 |

Same headline as 20k: the third leg adds 17–23 points of recall on top of the 2-leg union, at
essentially flat AUC (never moves more than ±0.016 vs its pad-matched 2-leg counterpart). **Unlike
20k, precision is not free here** — it holds flat at pad0 but drops noticeably at pad45,
worst case 50k pad45 Q2 (26.8% → 19.7%, a 7.1pp loss). Cost efficiency (extra rows listed per extra
real completer caught) also widens and gets worse at the padded, later-quarter end:

| | pad0 Q1 | pad0 Q2 | pad45 Q1 | pad45 Q2 |
|---|---|---|---|---|
| 40k | 2.90 | 2.76 | 3.93 | 6.11 |
| 50k | 3.48 | 4.34 | 6.08 | **10.89** |

(20k's equivalent range was a flat 1.6–2.5 across all four — see above.) **Reading so far: the
pad0 variant (no window padding) is the efficient, low-cost way to add this leg at every milestone
measured; pad45 compounds with it unfavourably at 40k/50k, especially in Q2, where it should be
re-examined per-milestone before adopting rather than assumed to transfer from 20k.** 30k has only
had the lighter scoping check (not this full build→score→reconcile), still open.

### All five due-date methods, isolated, across all four milestones tested (2026-10-08)
> **Look-ahead warning (2026-10-14):** this and every Q1 due-date table after it used history past
> Q1; precision at 30k+ is overstated. See `### Q3 2026 forward test… D′ + prev-step fix`.

The tables above only ever reported the *combined* arms (2-leg union, 3-leg union). This breaks each
method out individually — `schedule` alone, `burn_rate` alone, `last_completed_plus_months` alone —
at both pad0 and ±45d, for 20k/30k/40k/50k, Q1 2026. **Scope note: this is the due-date math only** —
gate-filtered population (`Last Service - PMS != '-'`), no `Vehicle Service Status` filter and no
stale-completion filter layered on top, so "listed" here runs higher than the delivery-ready numbers
in the sections above (those add both filters). Deliberately isolated this way to compare the three
leg formulas on equal footing, not to represent a deliverable list.

**20k** (actual Q1 2026 completers: 1,946)

| method | padding | listed | overlap | overlap % of actual (recall) | overlap % of listed (precision) |
|---|---|---|---|---|---|
| Schedule only | none | 2,530 | 560 | 28.8% | 22.1% |
| Burn-rate only | none | 1,519 | 671 | 34.5% | 44.2% |
| Last-service+6mo only | none | 2,271 | 791 | 40.6% | 34.8% |
| Union (schedule+burn) | none | 3,749 | 1,073 | 55.1% | 28.6% |
| Union (all 3 legs) | none | 4,779 | 1,310 | **67.3%** | 27.4% |
| Schedule only | ±45d | 5,180 | 951 | 48.9% | 18.4% |
| Burn-rate only | ±45d | 3,002 | 1,013 | 52.1% | 33.7% |
| Last-service+6mo only | ±45d | 4,548 | 1,271 | 65.3% | 27.9% |
| Union (schedule+burn) | ±45d | 6,997 | 1,502 | 77.2% | 21.5% |
| Union (all 3 legs) | ±45d | 7,800 | 1,662 | **85.4%** | 21.3% |

**30k** (actual Q1 2026 completers: 1,869)

| method | padding | listed | overlap | overlap % of actual | overlap % of listed |
|---|---|---|---|---|---|
| Schedule only | none | 1,649 | 256 | 13.7% | 15.5% |
| Burn-rate only | none | 1,631 | 813 | 43.5% | 49.8% |
| Last-service+6mo only | none | 2,191 | 624 | 33.4% | 28.5% |
| Union (schedule+burn) | none | 3,146 | 1,019 | 54.5% | 32.4% |
| Union (all 3 legs) | none | 4,679 | 1,304 | **69.8%** | 27.9% |
| Schedule only | ±45d | 3,490 | 444 | 23.8% | 12.7% |
| Burn-rate only | ±45d | 3,184 | 1,167 | 62.4% | 36.7% |
| Last-service+6mo only | ±45d | 4,393 | 1,300 | 69.6% | 29.6% |
| Union (schedule+burn) | ±45d | 6,130 | 1,409 | 75.4% | 23.0% |
| Union (all 3 legs) | ±45d | 7,975 | 1,690 | **90.4%** | 21.2% |

**40k** (actual Q1 2026 completers: 1,667)

| method | padding | listed | overlap | overlap % of actual | overlap % of listed |
|---|---|---|---|---|---|
| Schedule only | none | 1,907 | 234 | 14.0% | 12.3% |
| Burn-rate only | none | 1,494 | 644 | 38.6% | 43.1% |
| Last-service+6mo only | none | 2,008 | 551 | 33.1% | 27.4% |
| Union (schedule+burn) | none | 3,273 | 830 | 49.8% | 25.4% |
| Union (all 3 legs) | none | 4,695 | 1,110 | **66.6%** | 23.6% |
| Schedule only | ±45d | 3,568 | 357 | 21.4% | 10.0% |
| Burn-rate only | ±45d | 3,034 | 976 | 58.5% | 32.2% |
| Last-service+6mo only | ±45d | 4,060 | 1,107 | 66.4% | 27.3% |
| Union (schedule+burn) | ±45d | 6,147 | 1,196 | 71.7% | 19.5% |
| Union (all 3 legs) | ±45d | 8,095 | 1,484 | **89.0%** | 18.3% |

**50k** (actual Q1 2026 completers: 1,296)

| method | padding | listed | overlap | overlap % of actual | overlap % of listed |
|---|---|---|---|---|---|
| Schedule only | none | 1,740 | 122 | 9.4% | 7.0% |
| Burn-rate only | none | 1,253 | 457 | 35.3% | 36.5% |
| Last-service+6mo only | none | 1,821 | 442 | 34.1% | 24.3% |
| Union (schedule+burn) | none | 2,915 | 556 | 42.9% | 19.1% |
| Union (all 3 legs) | none | 4,337 | 821 | **63.3%** | 18.9% |
| Schedule only | ±45d | 3,479 | 256 | 19.8% | 7.4% |
| Burn-rate only | ±45d | 2,576 | 695 | 53.6% | 27.0% |
| Last-service+6mo only | ±45d | 3,712 | 783 | 60.4% | 21.1% |
| Union (schedule+burn) | ±45d | 5,737 | 875 | 67.5% | 15.3% |
| Union (all 3 legs) | ±45d | 7,816 | 1,102 | **85.0%** | 14.1% |

**Finding: the plain `schedule` leg degrades steadily as the milestone climbs, and it is the
weakest of the three legs at every milestone measured.** Its own-recall at pad0: 28.8% (20k) →
13.7% (30k) → 14.0% (40k) → 9.4% (50k). `burn_rate` and `last_completed_plus_months` show no such
decay — both stay in the 33–44% band at every milestone, pad0. This is consistent with `schedule`'s
known design flaw (anchors on the vehicle's very first service, so drift compounds over every
milestone since): the more milestones between first service and the target, the worse it gets.
`burn_rate` and `last_completed_plus_months` both reset from a recent anchor instead, so neither
accumulates that compounding error. Practical reading: `schedule` contributes the least of the three
legs to the union's recall, and contributes proportionally less at higher milestones — a candidate
for de-prioritising or dropping first if the union is ever simplified back down to two legs for cost
or complexity reasons at 40k/50k specifically.

Source data: `q1_due_rule_experiment/all_methods_all_milestones_Q1_2026.csv`.

### Fourth due-date leg — `due_date.avg_interval_date()` — MEASURED, the strongest leg yet on 30k/40k/50k (2026-10-09)
Same "reset the clock at the last completed milestone" family as the third leg, but instead of a
flat 6 months, projects forward by the vehicle's **own average historical block length in months**
— no mileage, no rate. Breakpoints: `FirstSrvDate -> date(Service_Num==1) -> date(Service_Num==10)
-> date(Service_Num==20) -> ...` up to the vehicle's highest completed milestone below the target
(same "hi" as `burn_rate_date()` / `last_completed_plus_months()`). Each consecutive pair is one
segment; a segment missing either endpoint is dropped from **both** the sum and the count, so the
average is over whichever segments the vehicle actually has — never padded or backfilled. For 20k
that's 2 possible segments (`->1`, `1->10`); for 30k, 3 (`->1`, `1->10`, `10->20`); generalizes the
same way for 40k/50k. `due = date_hi + mean(available segment lengths)`.

Scoping check (pure due-date math, no model, no build/retrain — same methodology as the "all five
methods" table above), Q1 2026, all four milestones:

**Standalone, pad0:**

| milestone | listed | overlap | recall | precision |
|---|---|---|---|---|
| 20k | 1,536 | 560 | 28.8% | 36.5% |
| 30k | 2,098 | 990 | **53.0%** | 47.2% |
| 40k | 2,148 | 885 | **53.1%** | 41.2% |
| 50k | 1,868 | 689 | **53.2%** | 36.9% |

On 20k it ties `schedule`'s recall exactly (560 overlap each) but on **different vehicles** (only
107 VINs in common) with far better precision (36.5% vs 22.1%). On 30k/40k/50k it beats every other
single leg measured so far, standalone — strikingly consistent at ~53% recall across three different
milestones.

**Folded into the union as a 4th leg, pad0 vs pad45:**

| milestone | 3-leg recall/prec | 4-leg recall/prec (pad0) | 3-leg recall/prec | 4-leg recall/prec (pad45) |
|---|---|---|---|---|
| 20k | 67.3% / 27.4% | 75.6% / 26.8% (+8.3pp rec, −0.6pp prec) | 85.4% / 21.3% | 89.1% / 20.4% (+3.7pp rec, −0.9pp prec) |
| 30k | 69.8% / 27.9% | **85.1% / 28.5%** (+15.3pp rec, **+0.6pp prec**) | 90.4% / 21.2% | 95.0% / 19.6% (+4.6pp rec, −1.6pp prec) |
| 40k | 66.6% / 23.6% | **80.4% / 24.0%** (+13.8pp rec, **+0.4pp prec**) | 89.0% / 18.3% | 93.6% / 16.7% (+4.6pp rec, −1.6pp prec) |
| 50k | 63.3% / 18.9% | **76.6% / 19.4%** (+13.3pp rec, **+0.5pp prec**) | 85.0% / 14.1% | 91.8% / 13.0% (+6.8pp rec, −1.1pp prec) |

**Headline: on 30k/40k/50k at pad0, this is the first leg in the whole investigation that buys
double-digit recall for free or better precision.** Every other addition measured so far (union
itself, padding, the third leg) bought recall by spending precision somewhere; this one doesn't, at
pad0. At pad45 the gain shrinks (expected — padding already recovers most of what this leg would
add) and does cost a little precision there, but recall still rises on all four milestones.

Not yet adopted into `build_unionstale_20k.py`'s CLI, the training matrix, or the production
pipeline — scoping check only, same stage the third leg was at before its full build→score→reconcile
run. Source: `q1_due_rule_experiment/method4_avg_interval_all_milestones_Q1_2026.csv`.

**Which lever costs the most precision, measured directly (2026-10-09):** broke the cumulative drop
from the production `min(schedule, burn)` baseline down into its individual steps, pad0, averaged
across all four milestones:

| lever | avg precision change |
|---|---|
| `min` -> 2-leg **union** (schedule OR burn, instead of whichever is sooner) | **−11.3pp** |
| ±45d padding, on top of any union stage | −3.8 to −9.4pp (avg ≈ −7.3pp) |
| 2-leg -> 3-leg union (+last-service+6mo) | −1.9pp |
| 3-leg -> 4-leg union (+avg-interval) | **+0.2pp** (often improves it) |

**Switching `min` to `union` costs more precision than padding does, and the two newer legs are
nearly free by comparison.** `min` is an AND-like choice of whichever single leg looks earliest;
`union` is an OR, so it inherits every false positive from *every* leg instead of just the sharpest
one — that's a bigger one-time cost than padding each leg's own window. Padding then stacks the same
kind of noise on top, roughly the same amount regardless of how many legs are already unioned
(unlike leg-additions, whose marginal cost shrinks or vanishes as more legs join). Source:
`q1_due_rule_experiment/precision_cost_by_lever_Q1_2026.csv`.

**`min` of all 4 legs — TESTED AND REJECTED (2026-10-09).** Tried replacing the 2-leg `min` with a
4-leg `min(schedule, burn, last-service+6mo, avg-interval)` to see if more legs would help the
*production-style* rule the way they help the union. They don't:

| milestone | min-of-2 (production) | min-of-4 | delta |
|---|---|---|---|
| 20k | 47.0% rec / 41.0% prec | **33.9% / 31.8%** | worse on BOTH |
| 30k | 50.2% / 45.6% | 50.9% / 40.9% | +0.7pp rec, −4.7pp prec |
| 40k | 46.6% / 35.7% | 50.3% / 35.0% | +3.7pp rec, ~flat prec |
| 50k | 39.4% / 28.4% | 47.5% / 29.4% | +8.1pp rec, ~flat prec |

Nowhere close to the 4-leg union's 75.6% / 85.1% / 80.4% / 76.6% recall, and it actively makes 20k
worse on both axes. **Cause: `min` picks whichever single leg projects earliest, so adding more legs
doesn't add more chances to be listed (that's what `union` does) — it adds more chances for a new
leg to win the "earliest" slot and drag the vehicle into an earlier, wrong quarter.** On 20k the new
legs apparently win that race often and pull vehicles earlier than they actually return, which is
why both recall and precision fall there; the mild gains on 30k/40k/50k are a side effect of which
leg happens to win, not real added coverage. **Ruled out — `union` remains the right combinator for
these legs, `min` is not worth extending past 2.** Closed, no further `min`-variant testing planned
for this leg set.

### Who the base still misses, and two due-date fixes tested (2026-10-12)
All measured on real Q1 2026 completers, strict quarter, no padding, rules A = `min(schedule,
burn)`, B = `union(schedule, burn)`, C = union of all four legs. Population: every completer in
service history — all of them turned out to be inside the EDA gate, so counts match the tables above
(20k 1,946 / 30k 1,869 / 40k 1,667 / 50k 1,296).

**Missed-completer profile — artifact "Missed Q1 2026 Completers"**
(`https://claude.ai/artifact/3p78gEoiqNKDjWMdnrjyfQ`, private; data from
`q1_due_rule_experiment/missed_profile_by_rule.py` → `.json` / `.csv`). Milestone and rule switches;
three mutually exclusive cards (past visits = distinct service dates before the milestone visit):
first-time (0), only one past visit, only non-PMS (2+ visits, none PMS); then a section for the rest
(2+ visits incl. a PMS) with PMS-visit counts, mileage bands (step −2k/+5k km, first service 0–6k,
checked on the milestone visit and every past PMS), "did every 10k step", "by the book" (every step,
all in band), 5 examples with full history and a due-date table (all four legs per vehicle).

| missed, A / B / C | first-time | one past visit | non-PMS only | PMS history | of which by the book (C) |
|---|---|---|---|---|---|
| 20k 1,031 / 873 / 474 | 24 | 194 / 194 / 156 | 2 | 811 / 653 / 292 | 192 |
| 30k 930 / 850 / 279 | 15 | 31 / 31 / 23 | 1 | 883 / 803 / 240 | 121 |
| 40k 891 / 837 / 327 | 14 | 25 / 25 / 24 | 3 | 849 / 795 / 286 | 118 |
| 50k 785 / 740 / 303 | 5 | 7 / 7 / 6 | 0 | 773 / 728 / 292 | 127 |

The three small cards are a minority; most misses are vehicles with normal PMS history.

**Most PMS-history misses are pace changes, not method failures**
(`q1_due_rule_experiment/missed_pms_rhythm.py` / `.csv`). Realized gap (last PMS → milestone visit)
vs the vehicle's own average past PMS gap; within ±25% = kept rhythm (a genuine miss). Rule C:

| | missed w/ PMS history | came later (>1.25×) | came sooner (<0.75×) | one past PMS only | **kept rhythm** | of those, last PMS + own avg gap lands in Q1 |
|---|---|---|---|---|---|---|
| 20k | 292 | 83 | 86 | 86 | **37** | 21 |
| 30k | 240 | 85 | 89 | 19 | **47** | 15 |
| 40k | 286 | 165 | 58 | 11 | **52** | 20 |
| 50k | 292 | 78 | 156 | 7 | **51** | 24 |

Typical pace change is large (later: 1.7–1.8× their usual gap; sooner: 0.2–0.5×). The ±25% line is a
judgement call. Traced example (one car, 20k done on 2026-01-02): service
1 → 10k took 33 days, so burn-rate (265.7 km/day) projected 2025-12-12; schedule (invoice 2025-09-26
+ 12 mo) 2026-09-26; last+6mo 2026-05-05; avg-interval 2025-11-25 (segments 7 and 33 days). It then
took 58 days 10k → 20k — a pace change no leg could see. Not a bug.

**Avg-interval without the invoice → first-service segment — NOT worth it in the union.**
`avg_interval_date(..., include_first_srv_segment=False)` (default True, unchanged). Alone it is much
better (recall/precision 20k 28.8/36.5 → 43.6/50.3; 50k 53.2/36.9 → 58.1/43.3), but swapping it into
C *loses* completers (20k 1,472 → 1,397) because it overlaps the other legs; keeping both variants adds
only +22 / +20 / +27 / +22. Source: `q1_due_rule_experiment/avg_interval_no_invoice_Q1_2026.csv`.
(`burn_rate_date` has no invoice segment — it starts at service 1.)

**Burn-rate fallback when there is no first-service record — DEFERRED, act on later.**
`burn_rate_date()` needs a `Service_Num == 1` row as `lo` (`due_date.py`, `lo_rows`), else NaT — the
non-PMS rows in between are already ignored. Found via one car with no code-1 record whose 50k
burn-rate date came out blank. Common: 20.2% / 26.4% / 30.1% / 34.2% of vehicles with a PMS
below 20k / 30k / 40k / 50k have no first-service record. `burn_rate_date(..., lo_fallback=True)`
(default False, unchanged) anchors `lo` on the earliest PMS visit instead. Caught, current → fallback:

| | A (production) | B | C |
|---|---|---|---|
| 20k | 915 → 915 | 1,073 → 1,073 | 1,472 → 1,472 |
| 30k | 939 → **1,055**, prec 45.6 → 47.9% | 1,019 → 1,149 | 1,590 → 1,599 (+79 listed) |
| 40k | 776 → **883**, prec 35.7 → 39.5% | 830 → 944 | 1,340 → 1,351 (+102 listed) |
| 50k | 511 → **588**, prec 28.4 → 31.7% | 556 → 642 | 993 → 1,009 (+128 listed) |

Clear win (more recall *and* precision) under A and B; under C it is only a safety net — about 8 extra
listed names per extra real completer, precision −0.2pp. 20k unaffected (without service 1 the
earliest PMS is the 10k itself, no span). **Decision (user, 2026-10-12): not adopted yet because of
the precision cost under C; revisit later.** If production stays on rule A, switching it on in
`prepare_test_set.py` is the obvious move. Sources: `q1_due_rule_experiment/burn_rate_lo_fallback.py`,
`burn_rate_lo_fallback_gap_counts.csv`, `burn_rate_lo_fallback_Q1_2026.csv`.

### Dropping legs from the 4-leg union — MEASURED, NOT ADOPTED (2026-10-14)
Same population and method as above (real Q1 2026 completers, strict quarter, no padding; A/B/C
rows reproduce the earlier tables). Two slimmer unions tested against C:
- **D = union(burn, avg-interval)** — drops schedule and last-service+6mo.
- **D′ = union(burn, last-service+6mo, avg-interval)** — drops schedule only.

| | listed | caught | recall | precision | kept of C's catches |
|---|---|---|---|---|---|
| 20k A / C | 2,231 / 5,495 | 915 / 1,472 | 47.0 / 75.6% | 41.0 / 26.8% | 62% / 100% |
| 20k D / D′ | 2,595 / 4,156 | 961 / 1,411 | 49.4 / 72.5% | 37.0 / 34.0% | 65% / 96% |
| 30k A / C | 2,061 / 5,573 | 939 / 1,590 | 50.2 / 85.1% | 45.6 / 28.5% | 59% / 100% |
| 30k D / D′ | 2,968 / 4,402 | 1,282 / 1,549 | 68.6 / 82.9% | 43.2 / 35.2% | 81% / 97% |
| 40k A / C | 2,172 / 5,572 | 776 / 1,340 | 46.6 / 80.4% | 35.7 / 24.0% | 58% / 100% |
| 40k D / D′ | 2,893 / 4,122 | 1,058 / 1,295 | 63.5 / 77.7% | 36.6 / 31.4% | 79% / 97% |
| 50k A / C | 1,801 / 5,129 | 511 / 993 | 39.4 / 76.6% | 28.4 / 19.4% | 52% / 100% |
| 50k D / D′ | 2,579 / 3,631 | 814 / 960 | 62.8 / 74.1% | 31.6 / 26.4% | 82% / 97% |

- **D keeps ~3 in 4 of C's catches (4,115 of 5,395) on half the list, precision +10–15pp vs C.**
  Against A it wins on both axes at 40k/50k, trades −2.4pp precision for +343 at 30k, and barely
  helps 20k (+46; avg-interval only has 2 segments there). Per completer lost vs C it drops 4.7 /
  7.5 / 8.5 / 13.2 wrong names (20k→50k).
- **The loss is mostly last-service+6mo, not schedule.** C catches D loses, by which dropped leg
  had them: schedule only 61 / 41 / 45 / 33, last+6mo only 196 / 200 / 184 / 131, both 254 / 67 /
  53 / 15. Hence D′ keeps 96–97% of C's catches on a 25–35% shorter list, precision +5–7pp —
  consistent with schedule being the weakest leg (see "All five due-date methods" above).
- **Decision (user, 2026-10-14): recorded only — do NOT switch to D or D′.** Rule C stays the
  reference. Sources: `q1_due_rule_experiment/union_burn_avg.py`, `union_burn_avg_Q1_2026.csv`,
  `union_burn_avg_lost_vs_C_Q1_2026.csv`.
  **SUPERSEDED the same day** — these numbers carry the look-ahead below and have no prev-step fix.
  Re-measured forward, D′ + prev-step fix is the best rule; see next section.

### Q3 2026 forward test, the look-ahead in the Q1 tables, and the winning rule: D′ + prev-step fix (2026-10-14)
**WARNING — every Q1 due-date table above (3-leg, 4-leg, "all five methods", D/D′, missed-completer
profile, the artifacts and the summary/email numbers) was built with `cutoff = serv.max()` =
2026-06-29, i.e. using visits made AFTER Q1.** Recall is barely affected (the legs read only
`Service_Num < m`, and a completer's later rows are ≥ m), but **precision at 30k+ is overstated**: a
car's later visit moves its "last PMS" anchor forward and silently drops it from wrong lists.
4-leg union, Q1 as recorded → Q1 rebuilt with history cut at 2025-12-31 (raw, no stale filter):

| | 20k | 30k | 40k | 50k |
|---|---|---|---|---|
| recall | 75.6 → 75.6% | 85.1 → 85.3% | 80.4 → 80.7% | 76.6 → 82.4% |
| precision | 26.8 → 24.2% | 28.5 → **19.9%** | 24.0 → **13.6%** | 19.4 → **9.5%** |
| listed | 5,495 → 6,091 | 5,573 → 8,021 | 5,572 → 9,904 | 5,129 → 11,220 |

Any forward test must cut history at the day before the quarter (as `roi_check.py` does).

**Cause — the last-service legs ignore WHICH milestone the car is at.** `last_completed_plus_months()`
and `avg_interval_date()` anchor on the highest PMS below the target, whatever it is. A car whose
10k was on 2026-03-15 gets last+6mo = 2026-09-15 on the 20k, 30k, 40k **and** 50k Q3 lists. On Q3 50k,
2,620 of 4,583 last+6mo names had last done a 10k or 20k.

**Prev-step fix:** the two last-service legs only count when the vehicle's highest completed
milestone is exactly `m − 10` (service 1 counts as below 10k). Schedule and burn-rate are unchanged
(they already carry the target). Lost real visits are few: Q3 removed 1,725 / 3,834 / 5,763 names at
30k / 40k / 50k for 36 / 54 / 64 real visits.

**Q3 2026 — the first truly forward test** (`q3_union_rules.py`, `forward_cutoff_check.py`): dates from
base history only (ends 2026-06-29), outcomes from `Q3/Service_History_Q3-2026.csv` (never EDA — no
per-visit date). `resolve_service_num()` reads `Service_Code` first, so the Excel-damaged `Nov-20`
spelling is safe (all 1,746 20k rows → 20). Rule A + stale filter reproduces the delivered Q3 lists
exactly (20k 1,309 / 800 · 30k 1,651 / 832 · 40k 1,658 / 706 · 50k 1,907 / 702) — method verified.
Out-of-gate completers no rule can reach: 159 / 38 / 41 / 21.

**ROI ranking — `roi_check.py`, both quarters forward-built, stale-filtered** (what a list really
holds). Cells: listed · recall · precision.

| Q1 2026 forward | A (today) | 4-leg + prev-step | **D′ + prev-step** (burn, last+6mo, avg) |
|---|---|---|---|
| 20k | 1,542 · 45.5% · 57.5% | 2,653 · 71.9% · 52.8% | **2,256 · 68.8% · 59.4%** |
| 30k | 1,715 · 49.4% · 53.9% | 3,010 · 82.7% · 51.4% | **2,514 · 80.4% · 59.7%** |
| 40k | 1,932 · 46.3% · 39.9% | 3,385 · 78.5% · 38.6% | **2,602 · 75.8% · 48.5%** |
| 50k | 1,683 · 42.9% · 33.0% | 2,892 · 77.0% · 34.5% | **2,005 · 74.5% · 48.1%** |
| 70k | 1,295 · 37.6% · 21.5% | 2,426 · 77.4% · 23.6% | **1,659 · 75.4% · 33.6%** |
| 80k | 1,222 · 33.9% · 15.8% | 2,184 · 69.8% · 18.2% | **1,414 · 68.6% · 27.7%** |

| Q3 2026 forward | A (today) | 4-leg + prev-step | **D′ + prev-step** |
|---|---|---|---|
| 20k | 1,309 · 46.6% · 61.1% | 2,350 · 69.8% · 51.0% | **2,092 · 68.2% · 56.0%** |
| 30k | 1,651 · 45.8% · 50.4% | 2,942 · 83.6% · 51.6% | **2,505 · 81.6% · 59.1%** |
| 40k | 1,658 · 42.0% · 42.6% | 3,168 · 79.0% · 41.9% | **2,481 · 77.2% · 52.3%** |
| 50k | 1,907 · 47.4% · 36.8% | 3,272 · 81.1% · 36.7% | **2,353 · 79.1% · 49.9%** |
| 70k | 1,613 · 44.0% · 25.3% | 2,909 · 78.7% · 25.1% | **1,912 · 77.3% · 37.5%** |
| 80k | 1,335 · 36.4% · 19.9% | 2,406 · 75.0% · 22.7% | **1,624 · 74.1% · 33.3%** |

- **D′ + prev-step beats A on BOTH recall and precision at every milestone 20k–80k in real Q1**, and at
  30k–80k in Q3 (20k Q3: recall +21.6pp, precision −5.1pp). Lists 1.1–1.6× A. Best F1 everywhere.
- Dropping the schedule leg costs 1–3pp recall for +6–14pp precision once the fix is in (Q1 50k:
  schedule adds 887 names for 33 visits). Consistent with schedule decaying with milestone.
- **Holds at high mileage**: 70k/80k are where today's rule is weakest (precision 16–25%) and the
  gain is largest (+35–40pp recall, +12–14pp precision). The plain 4-leg union is useless there
  (precision 3–6%).
- Marginal hit rate (extra real visits per extra name vs A) 63–127% in Q1; >100% means it also
  swaps wrong names out, not just adds.

**Mileage-bracket rule for skipping vehicles — TESTED AND REJECTED** (`mileage_bracket_skippers.py`,
`mileage_bracket_variants.py`). Rate = latest odometer ÷ days since FirstSrvDate, projected to the
quarter midpoint, bracket 20 = 15–25k, 30 = 25–35k …; "skipper" = bracket 2+ steps above last PMS,
listed only on its bracket. Fails twice: (1) the bloat comes from RECENT low-mileage cars (normal
next step), which it does not touch — 59,825 of 86,481 cars count as skippers, mostly long-gone ones;
(2) the bracket is one step too high for skippers who come back (they come at higher km than the
label: 40k visitors were mostly at the 45–55k midpoint). Variants: brackets shifted down one — list
re-bloats (Q3 50k 9,218 names, 13.5%); prev-step fix + bracket rescue — 277–652 extra names for
11–19 real visits per milestone. Neither beats the plain prev-step fix.

**Decision (user, 2026-10-14): adopt D′ + prev-step fix for production.** Not yet implemented — the
builders still hardcode `earliest(schedule, burn)`.

**Why D′ + prev-step precision falls with mileage, and a plausible fix (2026-10-15).** All 9
milestones, real Q1 and Q3 forward (`precision_drop_diagnosis.py` → `precision_drop_cars_2026.csv`,
one row per listed car + per-part CSVs; `precision_filters.py` → `precision_filters_2026.csv`).
Precision 59% (20k/30k) → 22% (100k) both quarters, recall flat 69–83%.
- **Cause 1 — the burn leg has no prev-step gate.** It still lists cars 3+ steps behind (last PMS
  60k on the 100k list) because mileage keeps projecting. Burn-only precision at 60k–100k: Q3
  13.9 / 8.5 / 6.2 / 4.7 / 4.4%; cars 3+ steps behind come in <2% of the time (270–590 per list).
- **Cause 2 — the wrong names have left the dealer, not mistimed.** Q1 wrong names that came for
  the service later: 70% at 20k → 11% at 100k; silent all of Jan–Jun: 25% → 79%. Cars with no visit
  of any kind in the year before the cutoff: 0–2% precision at every milestone.
- **Not the cause:** natural return rate (did m−10 in 2024 → did m by 2026-06-29) only falls 92% →
  75%, far too little to explain 59% → 22%.
- **Plausible fix — candidate, NOT adopted:** drop a car when (a) only the burn leg lists it and it is
  3+ steps behind (`m − last PMS > 20`), or (b) it has no visit of any kind in the 365 days before the
  cutoff. Both known at list time. Worst recall loss **1.0pp** (Q3 100k) — inside the 2pp budget —
  for precision (Q1 / Q3):

  | | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
  |---|---|---|---|---|---|---|---|---|---|
  | Q1 | 59.4→59.6 | 59.7→60.7 | 48.5→51.0 | 48.1→53.6 | 42.7→54.5 | 33.6→45.7 | 27.7→43.8 | 25.5→45.1 | 22.3→41.5 |
  | Q3 | 56.0→56.4 | 59.1→59.8 | 52.3→55.6 | 49.9→54.6 | 49.0→57.5 | 37.5→48.2 | 33.3→48.6 | 29.0→48.3 | 21.7→40.3 |

  Rejected: no visit in 180 days (up to −3.2pp recall); EDA `Vehicle Service Status == Active` —
  cannot be validated on Q1 (snapshot dated 2026-06-29, after Q1: Lapsed/InActive read exactly 0%
  precision there, a leak) and loses 2.4pp at Q3 100k.
- Q3 70k actuals = **928 distinct vehicles** (947 rows; 945 with Description `61-70`; 19 vehicles
  have two same-day 70k rows — duplicate lines / WIPs). All figures here count distinct VINs.
Sources (all `q1_due_rule_experiment/`): `q3_union_rules.py` → `q3_union_rules_2026.csv`;
`forward_cutoff_check.py` → `forward_cutoff_check_2026.csv`; `roi_check.py [70,80]` →
`roi_check_2026.csv`, `roi_check_2026_70k_80k.csv`; `mileage_bracket_skippers.py`,
`mileage_bracket_variants.py` → `*_2026.csv`.

### Widening prev-step to cars TWO steps behind (gap 2) — MEASURED AND REJECTED for 70k (2026-10-26)
Question: let the two last-service legs also list cars whose last PMS is `m − 20` (e.g. did 50k, no
60k, on the 70k list). Same method as `roi_check.py` (forward-built Q1 / Q3 2026, stale-filtered,
strict quarter, no model); the D′ rows reproduce it exactly. Two variants for the gap-2 cars:
**unscaled** (legs as they are: last + 6 mo, own average gap × 1) and **scaled** (last + 12 mo,
`avg_interval_date(scale_by_steps=True)` = own average gap × 2).

| 70k, listed · recall · precision | Q1 forward | Q3 forward | extra real visits per extra name |
|---|---|---|---|
| D′ + prev-step (gap 1 only) | 1,659 · 75.4% · 33.6% | 1,912 · 77.3% · 37.5% | — |
| + gap 2, unscaled | 2,622 · 79.7% · 22.5% | 3,216 · 80.9% · 23.4% | 3.3% / 2.6% |
| + gap 2, scaled (both legs) | 1,957 · 78.2% · 29.5% | 2,254 · 79.2% · 32.6% | 7.0% / 5.3% |
| + gap 2, scaled avg-gap leg only | 1,870 · 77.8% · 30.7% | 2,166 · 79.2% · 33.9% | 8.5% / 7.1% |
| + gap 2, scaled last + 12 mo only | 1,776 · 76.0% · 31.6% | 2,035 · 77.3% · 35.2% | 4.3% / 0.0% |

- **The unscaled legs date the NEXT service, not the target.** For a car whose last PMS is 50k,
  last + 6 mo is its 60k date. Of the cars it added, 585 / 963 (Q1) and 815 / 1,304 (Q3) came in for
  their **60k** that quarter, and only 32 / 34 for their 70k.
- **Few real visitors are gap 2 at all**: 69 of 739 (Q1) / 73 of 928 (Q3) 70k visitors, 24 / 28 of
  them already listed by the burn leg. Ceiling ≈ +45 visitors (~+5pp recall) for any variant.
- Every variant costs 4–11pp precision for +1.9–4.3pp recall. The scaled last + 12 mo leg adds
  nothing the avg-gap leg doesn't (Q3: 0 extra visits).
- **Decision (user, 2026-10-26): recorded only — keep prev-step at gap 1.** Only 70k measured;
  20k–60k untested (skipping a service may be more common at low milestones). The script was a
  throwaway in the session scratchpad (not in the repo); rebuild from `roi_check.py` by adding the
  `hi == m − 20` mask.

### D′ v2 retrain — 20k/30k rebuilt end to end, beats production on real Q3 2026 (2026-10-16)
**Experiment, not promoted.** `production_models/`, `salman/`, `pmstrainfeatureeng_refactored.py`
and `prepare_test_set.py` are untouched. Everything lives in `dprime_v2/`; models in
`models/{20,30}k_dprime_v2_{nn,tree}_s{42,7,123}/`. Plan of record:
`C:\Users\panga\.claude\plans\sharded-wandering-dahl.md`.

**Why it was needed (found while scoping it).**
- The production training lists date every car ONCE from the full history to 2026-06-29
  (`build_candidates_cohort()`, `cutoff = serv.max()`, :424-431) — a car's training quarter is picked
  with visits made after that quarter. Same look-ahead as the Q1 tables above.
- 45–48% of training positives (20k 19,357 / 30k 17,267, rows 2021–2025) were **early completers** —
  already did the service a median 53–57 days before the quarter; 58–60% of a car's repeat rows were
  this. They never reach a real list (stale filter) and teach "visited very recently → positive".
- Production features: label-dependent code paths (LastNonPMSMileage / Last PMS Mileage; VHC joined on
  EDA `Service_Num` → any VHC value non-zero for 278 negatives, 0 positives; PMS_Delay `legacy`);
  extract-time EDA values (PMSRevenue + 4 fixed clones, Tyre/Battery/Brake revenue, wearables,
  Warranty, Contract, LowMileageFreqUsers); uncut VHC / Digital / RFM; duplicates (months_to_10k,
  interval pairs 100% equal at 20k, Multiplier ≈ Years_Since_First_PMS, Freq ≈ Count); 8 constants.
- **retrain.py best-epoch bug — FIXED 2026-10-16:** `best_model_state = model.state_dict()` kept a
  reference, so the saved "best" model was the LAST epoch. Now `copy.deepcopy(...)` (+ `import copy`).
  Affects future runs only; every promoted model was trained with the bug.

**Decisions (user, 2026-10-15/16).**
- D′ + prev-step lists **quarter by quarter**, history cut the day before each quarter; one row per
  car per quarter (repeats kept).
- **Early completers dropped from training** (overrides the candidates-mode "keep as positives"
  design for this model only). TargetFlag = came in during the quarter or ≤ 15 days after.
- Training rows **2021Q1–2025Q4** (VHC/appointments only exist from late 2020); dating still reads
  history from 2016. Tests: 2026Q1, 2026Q2, **2026Q3 held out** (outcomes from the Q3 file).
- Data to 2026-06-29. **RFM never a training feature** (status snapshot at list date; reserved for
  prioritising future forward lists). **Digital sessions dropped** — the file has no per-record date
  (one total per vehicle). EDA = `data/EDA_Q2-2026.csv`, static facts only (no historical EDA
  snapshots exist).
- Models: NN (retrain.py recipe, bug fixed) and HistGradientBoosting, 3 seeds each.

**Data notes (Step 0, `dprime_v2/check_inputs.py`).** Inputs in `data/v2/`:
`Service_Appointments_Q3-2026.csv` (bookings 2020 → 2026-09-30; `Due Date IN` has 22,527 `-`, 1
garbage, 23 dates past 2027 → blank), `VHC_Q3-2026.csv` (2020-12 → 2026-09; 99.9% of keys join),
`Digital_Sessions_Q2-2026.csv` (undated — dropped), `RFM_Q3-2026.csv` (unused), `EDA_Q2-2026.csv`
(copy). The `Service_History_Q3-2026.csv` dropped there is **Jul–Sep only** — training history is
still `data/Service_History_Q2-2026.csv`; `check_inputs.py` now flags a history that starts after 2017.
The old "Q2-2026" RFM / Digital / VHC CSVs in `data/` are byte-size identical to the 2025 files
(probably relabelled old data). `MILESTONE_ROLLOUT_RUNBOOK.md` referenced in this file **does not
exist on disk**; `test/CANDIDATE_RUNBOOK.md` does.

**Pipeline (`dprime_v2/`, run from repo root with `PYTHONPATH=.`).**
1. `build_lists.py 20 30` → `lists_{m}k.csv`. Per quarter: legs from `due_date.py` on the cut history,
   prev-step gate, listed = any leg in the quarter; label as `build_candidates_cohort` (:435-458);
   bookkeeping kept (legs, gap, EarlyCompleter, CameInQuarter). 20k 64,406 rows / 30k 64,516 over 23
   quarters. **Verified**: not-early 2026Q1 / Q3 rows reproduce `roi_check.py` exactly
   (20k 2,256 / 1,339 and 2,092 / 1,171; 30k 2,514 / 1,502 and 2,505 / 1,481).
2. `features_v2.py` (`load_sources()`, `block_features()`) — ONE code path, nothing reads an outcome:
   - service history ≤ cutoff, PMS rows ≥ m excluded: has_x, n_pms, PMS gap mean, rhythm at quarter
     end, PMS revenue, non-PMS count / recency / revenue, visits total / last 365 d, days since any
     visit, age, invoice → first visit, odometer, km/day (life + recent), km to target at mid-quarter,
     branch count / main-branch share / main-branch one-hot (fixed top 8), bodyshop visits, survey
     mean / detractors, keys ≤ cutoff, tyre/battery/brake revenue;
   - list context: which legs listed it, n_legs, each leg's days from mid-quarter, centre distance, gap;
   - VHC: a record counts only when BOTH `Service Date` and `VHC Completed` ≤ cutoff (usually 2 days
     apart); red / amber, deferred / invoiced / deleted counts and revenue, deferred-red, recency;
   - appointments: booked ≤ cutoff; show / no-show ONLY for appointments due ≤ cutoff; upcoming
     appointment due inside the quarter (caveat: file holds the final due date, so a reschedule after
     the cutoff would show its new date). Extract-relative flags (Future Booking, Soon To No Show) unused;
   - EDA static: CC, cylinders, weight, height, wheel base, franchise, used, AMA, gender, model
     (top 15), nationality (top 8), has_invoice.
3. `build_matrix.py 20 30` → `refactored_test_dir/final_processed_{m}k_dprime_v2.csv` (train 20k 34,159
   rows 60.8% positive; 30k 35,464 / 59.6%) and `test_sets/test_{m}k_Q{1,2,3}_2026_dprime_v2.csv`.
   Pruning on training rows only (constants, then |corr| > 0.98): 104 → **98** (20k), 105 → **101**
   (30k); lists in `pruned_{m}k.json`. ~3–4 s per quarter.
4. `test_no_label_leak.py` — PASS: no outcome name in the feature code; features byte-identical with
   outcomes scrambled; train-vs-Q3 single-feature screen 0 flags (sign flip or gap ≥ 0.10) on both.
5. `train_models.py 20 30` — retrain.py's recipe copied (sorted by due date, IsolationForest 0.005,
   chronological 80/20, median imputer, RobustScaler clamp, BCE pos_weight + smoothing 0.1 + 0.5 Brier,
   AdamW 3e-4 / 1e-2, ReduceLROnPlateau, batch 128, 100 epochs, patience 10, deep-copied best) and a
   HistGradientBoosting (lr 0.05, 31 leaves, l2 1.0, early stopping). Separate script on purpose: v2
   needs none of retrain.py's derivation / intersection steps. Scores in `scores_20k_30k.csv`.
6. `evaluate_q3.py` — same Q3 D′ list for every model; seed-averaged probabilities; threshold picked
   on one random half and scored on the other.

**Results on held-out real Q3 2026** (20k: 2,092 cars, 1,171 came in · 30k: 2,505 / 1,481):

| | production NN | new NN | new tree |
|---|---|---|---|
| 20k AUC | 0.575 | 0.792 | **0.794** |
| 20k best F1, holdout (threshold) | 0.720 (0.12) | 0.778 (0.37–0.43) | **0.779** (0.39–0.43) |
| 20k precision / recall at that threshold | 58.1 / 94.7% | 69.3 / 88.9% | 68.0 / 91.2% |
| 20k precision calling the same 1,011 as production at 0.5 | 58.2% | 78.2% | **78.5%** |
| 30k AUC | 0.581 | 0.800 | **0.812** |
| 30k best F1, holdout (threshold) | 0.748 (0.16) | 0.808 (0.41–0.46) | **0.813** (0.46–0.50) |
| 30k precision / recall at that threshold | 61.0 / 97.4% | 73.5 / 89.9% | 73.6 / 90.8% |
| 30k precision calling the same 1,331 | 62.5% | 79.6% | **80.6%** |

Flag everyone: F1 0.718 (20k) / 0.743 (30k). Precision by probability decile, 30k tree: 11, 16, 41,
52, 66, 72, 76, 82, 86, **91%** — monotonic; production 36, 56, 68, 65, 50, 56, 57, 58, 74, 71%.

**No inflation:** AUC is level across quarters — 20k NN Q1 / Q2 / Q3 0.785 / 0.791 / 0.791, tree 0.782
/ 0.793 / 0.792; 30k NN 0.813 / 0.812 / 0.799, tree 0.829 / 0.823 / 0.809 (production read ~0.93 on
Q1/Q2 vs 0.62 on Q3). Seed spread on Q3 < 0.006 (G7 < 0.04). Validation AUC NN 0.81 / tree 0.82–0.83.
Thresholds land at 0.37–0.50, so the probabilities are usable as probabilities.

**Caveats / open.** The individual changes (dating, early-completer drop, features, bug fix) were
made together — their separate contributions are not measured. 20k/30k/70k/80k built (70k/80k below);
40k/50k/60k/90k/100k pending.
The production model's Q3 scores here use its own feature build (cutoff 2026-06-30) — a fair
comparison on the same cars. No promotion, threshold or gate battery yet (G1/G2 are hard-wired to the
Q2-2026 files; G6 here = `test_no_label_leak.py`). Sources: `dprime_v2/*.log`,
`train_summary_20k_30k.csv` (was `train_summary.csv`), `eval_q3.csv`, `eval_q3_deciles.csv`,
`leak_screen_{m}k.csv`.

**Feature importance (2026-10-16, `feature_importance.py`, permutation on held-out Q3, seed-averaged).**
Family shuffled together, AUC drop tree / NN: list context (which leg listed it, leg dates vs
mid-quarter) **0.070 / 0.197** (20k), **0.166 / 0.239** (30k), **0.162 / 0.090** (70k), **0.195 / 0.067**
(80k) — by far the largest everywhere; then PMS history & rhythm (0.02–0.07; at 70k/80k `has_60` /
`has_70` = "did the step before" carries the NN), mileage, appointments (`days_since_last_booking`,
stronger at high mileage). Branch / bodyshop / survey ≈ 0 or negative, VHC ≈ 0, EDA static facts
≈ 0 at 70k/80k. 68–77 features each move Q3 AUC ≤ 0.0005 alone — dead-weight candidates (untested).
Outputs `feature_importance_{m}k.csv`, `feature_importance_groups*.csv`.

### D′ v2 extended to 70k/80k and checked for stability (2026-10-16) — the strategy holds
Same pipeline, same decisions (`build_lists.py` / `build_matrix.py` / `train_models.py 70 80`; production
scored on the same Q3 lists by `q3_dprime_model_threshold.py 70 80`; layer-by-layer check
`stability_report.py 20 30 70 80`). Lists reproduce `roi_check.py` exactly (70k Q1 1,659 / 557, Q3
1,912 / 717; 80k Q1 1,414 / 391, Q3 1,624 / 540). Training 29,978 / 26,585 rows (33% positive),
108 features each, leak check PASS / 0 flags.

| Q3 2026 | production NN | new NN | new tree |
|---|---|---|---|
| 70k AUC | 0.739 | 0.843 | **0.856** |
| 70k F1 (holdout halves) | 0.651 | 0.721 | **0.726** |
| 70k precision calling the same 1,141 | 52.6% | 57.8% | **59.2%** |
| 80k AUC | 0.781 | 0.872 | **0.880** |
| 80k F1 (holdout halves) | 0.642 | **0.734** | 0.726 |
| 80k precision calling the same 669 | 56.7% | 64.9% | **65.5%** |

At high mileage production is a harder baseline (F1 0.651 / 0.642 vs 0.545 / 0.499 for calling
everyone) — the new models still win clearly.
- **Sampling holds** across all 23 quarters 2021Q1–2026Q3 (strict quarter, gated completers): list
  recall mean 20k 68.7% / 30k 76.7% / 70k 74.6% / 80k 68.9% (std 3.8–4.8, rising slightly toward
  2026); list precision 54.9 / 54.0 / 30.9 / 29.8% (std 2.3–4.0). Early completers 22% at 70k/80k.
- **Models hold**: 2025 validation and 2026 Q1/Q2/Q3 AUC within ~0.02 of each other (70k tree
  0.859 / 0.844 / 0.856; 80k tree 0.897 / 0.902 / 0.880). Seed spread < 0.04 everywhere (tree ≤ 0.009;
  70k NN widest 0.022). The tree's training-quarter AUC (0.91–0.96) is in-sample — never quote it.
- **Calibration**: tree mean probability tracks the real came-in rate (70k/80k Q3 0.39 / 0.36 vs 0.38 /
  0.33); the NN **over-predicts at 70k/80k** (0.50 / 0.47) — class weighting; ranking unaffected.
- **Threshold drift at 70k**: in-quarter best-F1 threshold 0.26–0.48 (tree) / 0.48–0.63 (NN) across
  2026 Q1–Q3 (flat F1 curve). Stable at 20k (~0.42), 30k (~0.47), 80k NN (0.50–0.56).
- **The precision filter is learned**: cars the burn-gap + 365-day filter would drop sit in the new
  models' bottom 30% — 100% at 20k/30k, 99–100% at 70k, ~90% at 80k (production 14–83%).

**Best model per service, honest counts (`best_model_confusion.py`)** — model and threshold picked on
2026 Q1+Q2, applied once to Q3 (tree won all four on Q1+Q2):

| | threshold | flagged | TP | FP | precision | recall in list | recall of ALL Q3 visitors |
|---|---|---|---|---|---|---|---|
| 20k tree | 0.44 | 1,523 | 1,056 | 467 | 69.3% | 90.2% | 61.5% of 1,717 |
| 30k tree | 0.49 | 1,817 | 1,342 | 475 | 73.9% | 90.6% | 73.9% of 1,815 |
| 70k tree | 0.48 | 843 | 560 | 283 | 66.4% | 78.1% | 60.3% of 928 |
| 70k NN | 0.50 | 1,093 | 652 | 441 | 59.7% | 90.9% | 70.3% |
| 80k tree | 0.40 | 765 | 479 | 286 | 62.6% | 88.7% | 65.7% of 729 |

Production at 0.5: 588 / 832 / 600 / 379 found at 58.2 / 62.5 / 52.6 / 56.7%. At 70k the Q1+Q2
threshold is too strict for Q3 (drift) — the NN beats production on both axes there.

### Precision–recall equilibrium: targets, behaviour features, ±45-day lists (2026-10-16)
**Target (user, 2026-10-16).** End goal is an equilibrium of precision and recall of **ALL** actual
visitors (TP / every car that came in for the service in the quarter, incl. cars the list never held):
**20k** may miss 20–30% of actuals (nuanced behaviour, tolerated) — recall of all 70–80%, precision
70–80%; **all other services** both north of 80%. Every number below: threshold chosen on 2026 Q1+Q2,
applied once to Q3; tree + NN averaged unless stated.

**1. Balance on the exact-quarter lists (`pr_balance.py`).** In-list balance (precision ≈ recall of the
list): 20k 75.1 / 78.2, **30k 79.0 / 78.6**, 70k 70.8 / 68.1, 80k 68.9 / 71.3. 80/80 in-list needs
recall 80% at a false-positive rate of 0.2·b/(1−b) — 25% / 29% / 12% / 10% — where the models reach
70 / 77 / 55 / 55%: 30k is close, 70k/80k would need a much stronger ranking. **Recall of ALL is capped
by the list**: an exact-quarter D′ list holds only 68 / 82 / 77 / 74% of all Q3 visitors.

**2. Behaviour features (v3) — TESTED, NO GAIN, not adopted.** `features_v2.behaviour_features()`
(`--set v3` on build_matrix / test_no_label_leak / train_models; outputs `*_dprime_v3*`,
`models/{m}k_dprime_v3_*`, `dprime_v2/v3/`): prior listings for the same milestone (membership only —
a car still due did not come in any earlier listing), listed last quarter, punctuality vs the 6-month
schedule and vs the milestone km, gap regularity / slowing down, rhythm at cutoff, season, the
customer's other vehicles (EDA Customer ID, visits ≤ cutoff). Leak check PASS. Q3 AUC change
−0.008…+0.006 (inside seed noise), balance points ±1–3pp (`compare_v2_v3.py`). v2 already carries the
timing signal; the residual errors are pace changes. "Listed last quarter, didn't come" points the right
way at 20k (AUC 0.61) but adds nothing on top. Lists now start 2019Q1 (training still 2021Q1+; the
2021+ rows are unchanged — v2/v3 Q3 car sets identical).

**3. List padding ceiling (`list_padding_ceiling.py`).** Any D′ + prev-step leg within ±N days of the
quarter, outcome still strict. Recall of ALL, Q3 / Q1: exact 68/69 · 82/80 · 77/75 · 74/69%; **±45 →
81/80 · 90/92 · 89/88 · 85/86%** (20k / 30k / 70k / 80k); ±60 adds only 1–3 more. List precision falls
(Q3, exact → ±45): 56→52, 59→48, 38→27, 33→23%.

**4. ±45-day retrain (`build_lists.py --pad 45`, `build_matrix.py --pad 45`, `train_models.py --set
v2_pad45`, `eval_pad45.py`).** Padded training AND test lists; **label = strict quarter** (a padded list
must not train on the +15-day grace it is not judged on). Training 43–50k rows (positive 49.6 / 43.2 /
21.4 / 20.7%), leak PASS / 0 flags. Q3, recall of ALL / precision:

| | target | ±45, closest point | ±45 balance (max of the lower) | exact-quarter list |
|---|---|---|---|---|
| 20k | 70–80 / 70–80 | **69.2 / 70.6** (1,189 found, 1,685 called) | 72.3 / 68.4 | 70% recall not reachable |
| 30k | 80 / 80 | **78.8 / 72.3** (1,431 found, 1,978 called) | 74.7 / 75.2 | 81.2 / 61.8 |
| 70k | 80 / 80 | 81.4 / 54.9 (755 found, 1,375 called) | 66.2 / 68.5 | 80% not reachable |
| 80k | 80 / 80 | 78.6 / 50.0 (573 found, 1,145 called) | 67.4 / 63.3 | 80% not reachable |

**±45 helps at every service** (more precision at the same recall of all — 30k at ~80% recall 62 → 72%)
and makes 80% recall reachable at 20k/70k/80k. **20k meets its target band (~70/70); 30k is close; 70k
and 80k reach 80% recall only at 50–55% precision.** The residual problem is timing — in a ±45 list only
21–27% of 70k/80k cars come in the strict quarter. Padded-list AUCs (0.81 / 0.85 / 0.90 / 0.91) are NOT
comparable with exact-list AUCs (more easy negatives). The "balance" rule is max(min(precision,
recall)); a min|P−R| rule picked a degenerate zero-flag threshold at 80k — don't use it.

**5. Carry-over of no-shows to the next quarter — MEASURED, idea parked for a LATER STAGE (user,
2026-10-16).** Proposal: a customer listed for a quarter who does not come moves to the next quarter with
a slightly raised probability. **Not a substitute for a wider success window** (that changes when a call
counts as a success; carry-over credits the same visit to the next list). Of a quarter's no-shows,
the share coming next quarter is 62–70% (20k), 59–61% (30k), 21–23% (70k), 19% (80k). Most of those are
**already on the next list**: exact 79–83% (20k) / 81–87% (30k) / 74–76% (70k) / 67–75% (80k); ±45
92–97% at every milestone (`lists_{m}k.csv` vs `lists_{m}k_pad45.csv`, 2026Q1→Q2 and Q2→Q3). Carrying the rest adds
Q2→Q3 89 / 76 / 70 / 45 visits at 33 / 23 / 9 / 6% precision — below each list's rate (56 / 59 / 38 /
33%). The "raise the probability" half is already learned (v3 test above).

**6. Time-to-visit model — REJECTED for now (user, 2026-10-16).** The business needs an expected date
and the probability the customer comes in the quarter that date falls in; a model predicting the visit
date itself does not fit that deliverable. Revisit only as later analysis.

**Still open for 30k–80k:** monthly refreshed lists (re-score with each month's visits — timing info
weeks old instead of up to 3 months); a wider success window — measured 2026-10-17 (see `### Wider
success window` below), adoption is a business decision. Timeline page of
14–16 Oct work: `https://claude.ai/artifact/W4g64FvQXYXraLCZ6kK5CU` (private).
Script/output conventions: `train_models.py` / `evaluate_q3.py` / `feature_importance.py` /
`q3_dprime_model_threshold.py` take milestones and tag outputs by them (`scores_{tag}.csv`,
`train_summary_{tag}.csv`, `eval_q3_{tag}.csv`); `--set v3` / `--set v2_pad45` outputs live in
`dprime_v2/v3/` and `dprime_v2/v2_pad45/` so the v2 globs never mix sets.

### Wider success window — MEASURED, adoption is a business decision (2026-10-17)
**Not the same as the ±45-day padding.** Padding changes WHO is on the list (legs within ±45 days of the
quarter) and keeps the label on the strict quarter. The success window changes WHAT counts as a success:
a visit from quarter start up to **quarter end + N days**. (Visits before quarter start never count —
those cars are early completers, already removed at list time.) `dprime_v2/success_window.py 20 30 70 80`
→ `success_window_20k_30k_70k_80k.csv` / `success_window.log`.

Method: N ∈ {0, 15, 30, 45} × list {exact quarter (v2), ±45 days (v2_pad45)}; labels rebuilt from
`lists_{m}k[_pad45].csv` `Actual{m}kDate`; HistGradientBoosting retrained per N (train_models.py tree
recipe, **seed 42 only** — tree seed spread was ≤ 0.009), features unchanged; threshold picked on **2026 Q1**,
applied once to **2026 Q2**. Q3 cannot be used: data ends 2026-09-29, so visits after Q3 are unknown
(Q2 + 45 days ends 2026-08-14, inside the Q3 file). Recall of all = flagged cars that came in the window /
every car whose first-ever m visit falls in [quarter start, quarter end + N], all VINs.

Q2 2026, recall of all / precision (balance point = max of min(precision, recall of all); 20k at the
recall ≥ 70% point):

| | ±45 list, N=0 | N=15 | N=30 | **N=45** | ±45 list ceiling N=0 → 45 | exact list, N=0 → 45 (balance) |
|---|---|---|---|---|---|---|
| 20k (recall ≥ 70%) | 69.0 / 74.5 | 70.1 / 76.2 | 69.8 / 76.4 | **70.1 / 77.3** | 81.8 → 72.7% | 64.1 / 66.8 → 58.8 / 74.4; ceiling 66.4 → 58.8% |
| 30k | 76.5 / 76.4 | 79.2 / 76.2 | 80.9 / 75.6 | **80.0 / 77.9** | 91.2 → 84.8% | 74.7 / 73.9 → 71.1 / 77.1 |
| 70k | 65.6 / 64.9 | 67.3 / 64.3 | 69.3 / 64.3 | **70.4 / 66.7** | 89.7 → 84.3% | 63.5 / 62.9 → 65.6 / 61.4 |
| 80k | 66.1 / 69.7 | 69.1 / 67.9 | 71.0 / 67.5 | **71.2 / 67.7** | 87.2 → 80.1% | 65.1 / 66.8 → 62.0 / 60.8 |

- **Best setting = ±45 list + 45-day window**: 20k inside its 70–80 band; 30k ~2 points short of 80/80;
  70k/80k still ~70/67. 80% recall of all at 70k/80k only comes with 53–57% precision.
- **The window only helps together with the ±45 list.** On exact-quarter lists the ceiling falls at every
  service (20k 66 → 59, 30k 80 → 71, 70k 78 → 66, 80k 73 → 63%) because the extra late visitors are mostly
  cars the exact list never held; balance-point recall of all drops at 20k/30k/80k (70k +2.1 at −1.5
  precision).
- The list rate rises with N (±45: 20k 56 → 71%, 30k 50 → 67%, 70k 26 → 36%, 80k 24 → 31%) while Q2 AUC
  falls slightly (20k 0.818 → 0.784, 30k 0.868 → 0.830, 70k 0.898 → 0.879, 80k 0.926 → 0.915).
- **Caveats before adopting:** (1) with ±45 lists and a +45 window, one visit can be credited to two
  quarters' lists — credit each visit once, to the earliest list holding the car, before summing quarters;
  (2) the probability then means "comes by quarter end + N days", which the business must agree to;
  (3) one test quarter (Q2), one seed — confirm with the tree/NN average and 3 seeds before quoting.
- Feature importance under N=45: see the importance block in the next section.

### Final Q3 2026 result — the best model so far, and the method behind it (2026-10-20)
**Experiment, not promoted** — `production_models/`, `salman/` and both production builders are untouched.
`dprime_v2/final_q3_report.py 20 30 70 80` → `final_q3_report_20k_30k_70k_80k.csv` / `.log` (read-only,
from saved scores). Only 20k / 30k / 70k / 80k are built; 40k/50k/60k/90k/100k are pending.

**Method, end to end — nothing is tuned on Q3:**
1. **List (who is on it):** D′ + prev-step — legs `burn_rate_date`, `last_completed_plus_months` and
   `avg_interval_date`, the last two only when the highest PMS below m is exactly m−10 — dated quarter by
   quarter from history cut the day before the quarter. A car is listed when any leg falls within
   **±45 days of the quarter** (`build_lists.py --pad 45`). Early completers (did m before the quarter —
   known at list time) are removed.
2. **Features:** `features_v2.py`, one code path for every row, cut at the cutoff, no outcome columns;
   pruned on training rows only (constants, then |corr| > 0.98) → 99–109 features. Leak check PASS.
3. **Label:** came in for service m in the **strict quarter** (`CameInQuarter`), no grace days.
4. **Training:** rows 2021Q1–2025Q4. HistGradientBoosting and NN (retrain.py recipe, best epoch
   deep-copied), 3 seeds each. **Score = average of the tree and the NN, each seed-averaged.**
5. **Threshold:** the balance point — max of min(precision, recall of all visitors) — chosen on
   2026 Q1 + Q2 pooled, applied **once** to Q3.
6. **Outcome:** `Q3/Service_History_Q3-2026.csv`, 1 Jul – 30 Sep 2026. **Recall of all** = TP / every
   distinct VIN with an m visit in Q3, including cars no list held. Training ends 2025Q4, so Q1/Q2/Q3 2026
   are all unseen; Q3 is the held-out quarter.

**Best model on Q3 2026 (±45 list + tree/NN average, balance threshold):**

| | thr | listed | came in from list | all Q3 visitors | flagged | TP | FP | FN | TN | accuracy | precision | recall in list | **recall of all** | F1 in list | F1 vs all | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20k | 0.41 | 2,641 | 1,383 | 1,717 | 1,814 | 1,241 | 573 | 142 | 685 | 72.9% | 68.4% | 89.7% | **72.3%** | 77.6% | 70.3% | 0.815 |
| 30k | 0.495 | 3,379 | 1,634 | 1,815 | 1,803 | 1,355 | 448 | 279 | 1,297 | 78.5% | 75.2% | 82.9% | **74.7%** | 78.8% | 74.9% | 0.851 |
| 70k | 0.555 | 3,091 | 827 | 928 | 897 | 614 | 283 | 213 | 1,981 | 84.0% | 68.5% | 74.2% | **66.2%** | 71.2% | 67.3% | 0.902 |
| 80k | 0.54 | 2,723 | 619 | 729 | 776 | 491 | 285 | 128 | 1,819 | 84.8% | 63.3% | 79.3% | **67.4%** | 70.4% | 65.2% | 0.912 |

**Against the alternatives on the same quarter** (precision / recall of all / F1 vs all / AUC):

| | today: delivered list + production NN at 0.5 | exact-quarter D′ list + new tree (max-F1 thr on Q1+Q2) | **±45 list + tree/NN average** |
|---|---|---|---|
| 20k | 64.3 / 22.5 / 33.3 / 0.619 (1,309 listed, 600 flagged) | 69.3 / 61.5 / 65.2 / 0.794 | **68.4 / 72.3 / 70.3 / 0.815** |
| 30k | 60.4 / 24.2 / 34.6 / 0.673 (1,651 / 728) | 73.9 / 73.9 / 73.9 / 0.812 | **75.2 / 74.7 / 74.9 / 0.851** |
| 70k | 54.2 / 37.0 / 43.9 / 0.852 (1,613 / 633) | 66.4 / 60.3 / 63.2 / 0.856 | **68.5 / 66.2 / 67.3 / 0.902** |
| 80k | 50.7 / 26.3 / 34.7 / 0.855 (1,335 / 379) | 62.6 / 65.7 / 64.1 / 0.880 | **63.3 / 67.4 / 65.2 / 0.912** |

The list itself (share of all Q3 visitors on it): today 46.6 / 45.8 / 44.0 / 36.4% → ±45 list 80.5 / 90.0 /
89.1 / 84.9% — the recall ceiling for any model on that list.

How to read it:
- **Quote recall of ALL visitors**, not recall in list (it ignores cars the list never held).
- **Accuracy is not comparable across lists** — it rises with the number of easy "won't come" rows; today's
  70k/80k lists read 78–81% accuracy while finding only 1 visitor in 3–4.
- ±45-list AUC is not comparable with exact-list AUC (more easy negatives).
- The exact-quarter tree has a slightly higher F1 in list (0.784 / 0.814 / 0.718 / 0.734) but finds fewer
  of all visitors — the ±45 set-up is the one that serves the recall-of-all goal.
- Verified: today's rows match `analysis/reconcile_quarter.py` (Q3 AUC 0.619 / 0.673 / 0.852 / 0.855), the exact
  rows match `best_model_confusion.py`, the ±45 rows match `eval_pad45.py`.

**Feature importance of the ±45 tree (`success_window_importance.py`, 2026-10-19).** Permutation AUC
drop on 2026 Q2, ±45-list tree, seed 42, 5 shuffles; families as `feature_importance.py`. N=0 = strict
quarter label (the tree half of the model above); N=45 = retrained with a visit up to 45 days after the
quarter counted as success. Q2 AUC N=0 / N=45: 0.818 / 0.784, 0.868 / 0.830, 0.898 / 0.879, 0.926 / 0.915.

| family, AUC drop N=0 → N=45 | 20k | 30k | 70k | 80k |
|---|---|---|---|---|
| list context (which leg, leg dates vs mid-quarter, gap) | 0.076 → 0.069 | **0.236 → 0.181** | **0.219 → 0.143** | **0.225 → 0.167** |
| PMS history & rhythm | 0.041 → 0.038 | 0.038 → 0.034 | 0.048 → 0.056 | 0.016 → 0.018 |
| appointments | **0.056 → 0.040** | 0.008 → 0.012 | 0.018 → 0.018 | 0.014 → 0.015 |
| vehicle facts (EDA) | 0.033 → 0.038 | 0.003 → 0.006 | 0.002 → 0.005 | 0.003 → 0.003 |
| visit recency & activity | 0.024 → 0.016 | 0.003 → 0.008 | 0.007 → 0.008 | 0.024 → 0.026 |
| mileage | 0.015 → 0.012 | 0.019 → 0.022 | 0.010 → 0.013 | 0.005 → 0.005 |
| non-PMS visits, VHC, branch / bodyshop / survey | ≤ 0.009 | ≤ 0.003 | ≤ 0.006 | ≤ 0.006 |

Top single features (N=45): 20k `rhythm_qend`, `days_since_last_booking`, `avg_days_from_qmid`,
`n_legs`, `weight`; 30k `n_legs`, `last6_days_from_qmid`, `rhythm_qend`, `pms_gap_last_days`;
70k `days_since_last_pms`, `gap`, `days_since_last_booking`, `pms_gap_last_days`; 80k
`last6_days_from_qmid`, `gap`, `days_since_any_visit`, `days_since_last_booking`.
- Same ranking as the exact-quarter importance above: due-date timing first, rhythm second; VHC, branch,
  bodyshop and survey contribute nothing.
- With the wider success window exact timing matters less (list context −25…−35% at 30k/70k/80k) and
  "method agreement" (`n_legs`, 20k/30k) and "still active" (`gap`, `days_since_any_visit`, 70k/80k) gain.
- 72 / 81 / 80 / 86 of 99 / 101 / 108 / 109 features move Q2 AUC ≤ 0.0005 alone — dead-weight
  candidates (untested). One seed, one quarter; read families, not single columns.
Outputs `dprime_v2/v2_pad45/success_window_importance_{m}k.csv`, `…_groups_20k_30k_70k_80k.csv`, `.log`.

### Related, same investigation: what does NOT explain 20k's low volume
- **Missing 10k records are already handled.** 22,429 eligible vehicles have no 10k record; the
  schedule leg covers them (141 land in Q3, 125 in Q4). A missing 10k only kills the *burn-rate*
  leg and `earliest()` falls back automatically. No rule change needed.
- **Relaxing `Last Service - PMS == '-'` is not worth it.** That filter drops 43,949 of 130,759 EDA
  rows (34%); 8,665 of them have both a real Invoice date and service-history presence, and 551
  would land in Q3/Q4. But **97.7% of real 20k completers had a prior 1k or 10k on record — only
  2.3% (89 vehicles) came with no prior PMS at all.** Adding never-PMS vehicles chases a ~2% pattern.
- **The genuine cause is data truncation, not logic.** 20k is due ~12 months after first service, so
  its Q3/Q4 pipeline needs vehicles first serviced in late-2025/2026 — exactly what the
  2026-06-29 extract truncates. Projected due dates collapse forward (2,247 Q1 → 1,797 Q3 → 1,092 Q4
  → 429 2027Q1) where 30k's decline gently. Eligible pools are near-identical (20k 11,519 vs 30k
  11,023), so it is **not** a pool shortage. Only a newer extract fixes this.
- **The overdue list is the lever that worked** — see below.

### Overdue lists: `salman/overdue_{m}k_Q{3,4}_2026.csv` (2026-09-10, 20k only so far)
Vehicles that completed the preceding milestone, came due, never showed, and are still
`Active`/`Lapsed`. Built by running `prepare_test_set.py` over a trailing 12-month window
(`--window-start 2025-07-01 --window-end 2026-06-30 --train-cutoff 2026-06-30`) and keeping
`TargetFlag == 0`. Funnel on 20k: 1,513 came due and never showed → 976 had actually done their 10k
→ **898** still Active/Lapsed (700 Active, 198 Lapsed; 841 seen in the last 12 months).
Split Q3 (≤6mo overdue, 656 rows) / Q4 (6-12mo, 242 rows), sorted by probability, with three extra
columns (`days_overdue`, `service_status`, `last_service_date`). **456 flagged — +35% on top of the
1,309-row Q3 forecast list, with no rebuild and no invalidated metrics.**
**Caveat: these probabilities RANK, they do not calibrate.** Features are computed at 2026-06-30,
up to a year after the vehicle's due date, which is out-of-distribution versus training (where
features sit just before the due window). Use the ordering for call priority; do not compare these
probabilities against the Q3/Q4 forecast numbers. The list itself is a record of fact and does not
depend on the model.
Note `Vehicle Service Status` is **RFM-derived business logic, not visit recency** — do not
substitute a "seen recently" proxy for it.

### Burn-rate due date is CUMULATIVE and pooled (2026-09-09)
`due_date.burn_rate_date()` anchors on **service 1 → the highest completed milestone below the
target** (so 1→10 for 20k, 1→20 for 30k, 1→30 for 40k …), using total km / total days — a
days-weighted pooled rate over every completed interval, not the last block alone. Superseded two
earlier versions the same day: the original fixed `1→10` for every milestone, then a single
`(M-20)→(M-10)` interval. 20k is unchanged by all of this (1→10 either way).
A vehicle with no completed milestone below the target is **excluded (NaT)**, never backfilled from
a more distant pair. Note the function only projects forward from history — it does **not** check
whether the vehicle already completed the target; that is the separate
`stale_completion_filter.py` post-filter's job.

`refactored_test_dir/mi_scores_analysis.md` — feature-selection analysis.
`AGENTS.md` is NOT just a pointer — it is 83 lines with its own project framing, and it still
describes the `ever` convention as the labelling rule. Production is candidates mode; treat
`AGENTS.md`'s labelling section as superseded by this file. Treat the target-definition claims in `old_vs_refactored_comparison.md`
and `later/refactoring_summary.md` as wrong (see Project).

Not previously listed here, all current:
- `MILESTONE_ROLLOUT_RUNBOOK.md` — **the operational source of truth** for promoting a milestone:
  9 steps, exact commands, 3 human checkpoints. Keep it in sync with any pipeline change.
- `test/CANDIDATE_RUNBOOK.md` — the earlier candidate-mode runbook (20k–100k), including a
  PowerShell batch runner for 50k–100k. Superseded operationally by the rollout runbook above;
  still the fullest write-up of *why* candidate mode replaced the earlier modes.
- `milestones_20k_90k_100k_stats.md` — full Checkpoint A stats, gate battery and 3-seed tables for
  20k/90k/100k.

**Deleted 2026-10-07** (do not go looking for them, and do not recreate them from memory):
`S3_MIGRATION_GUIDE.md` and `FABRIC_MIGRATION_GUIDE.md` (both aspirational 8-stage migration plans;
none of the files they described were ever written), `FABLE5_INVESTIGATION_BRIEF.md` (hand-off audit
brief, superseded by this file and the rollout runbook) and `REFACTOR_LOSSES.md` (line-referenced
`prepare_test_set.py` vs `monoliths/predservicemil_4.py` audit — it was the source of the zero-fill percentages
and the two-`selected_features.json` finding, both of which are now recorded in the Gotchas above).
