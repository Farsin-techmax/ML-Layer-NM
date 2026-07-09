# PMS Feature Engineering & Prediction Data README

This document summarizes the two Python scripts in this folder, explains their functions, how they connect, the data flow (feature pipeline), expected inputs, outputs, and notes for running / next steps.

Files
- **pmstrainfeatureEng 6.py** — Primary feature engineering pipeline intended to prepare a training dataset for PMS (Periodic Maintenance Service) prediction. Main entrypoint: `main()` which calls `prepare_pms_datasets()` then `process_service_data()` to produce the cleaned, imputed feature table (returned as `pms_imputed`).
- **predservicemil 4.py** — Prediction dataset pipeline. Iterates across service milestones (10k/20k/30k/... up to target), derives the same feature set used by training for each milestone, concatenates them into a prediction dataset, and builds a Non-PMS dataset. Produces CSVs for validation/prediction.

High-level purpose
- Convert raw service history, EDA / master records, appointment logs, VHC (value handling/parts) and digital sessions into a feature matrix suitable for model training and prediction.
- Produce both: (a) training master (labelled via `TargetFlag`) and (b) final prediction datasets for vehicles expected to be due for a target PMS.

Common inputs (examples seen in code)
- EDA / master file: `EDA - Q3 2025.csv` or `EDA Datasheet Till 2025.csv`
- Service history: `Service History Q1 - 2026.csv` / `Service History Feb 2026.csv`
- RFM segments: `RFM Segments Q2 - 2025.csv` / `RFM till 2025.csv`
- Appointment files: `Appoinment - Showed Up.csv`, `No Show VINs.csv`, `Appoinments - Q3 - 2025.csv` / `Appoinments2025.csv`
- VHC files: `VHC Q3 - 2025.csv`, `VHC PMS till 2025.csv`
- Service code description: `Service Code Desc.csv`
- Digital sessions: `Digital sessions Q3 - 2025.csv` / `DigitalData2025.csv`

Main outputs (examples)
- `validatecode/finalmerged{last_service_code}kQ3.csv` — final merged training features for a given target (pmstrain script)
- `chckpredMAin.csv`, `chckpredMAinNonPms.csv`, `chckpred1.csv` — concatenated prediction / validation datasets (predservice script)
- Several intermediate CSVs are written to `validatecode/` throughout processing for debugging/inspection (cluster files, monthly intervals, etc.).

Dependencies
- Python packages used (non-exhaustive): `pandas`, `numpy`, `scikit-learn`, `scipy`, `gower`, `duckdb`, `kneed`, `python-dateutil`.
- Suggested install: `pip install pandas numpy scikit-learn scipy gower duckdb kneed python-dateutil`

High-level data flow (training script: `pmstrainfeatureEng 6.py`)
1. `prepare_pms_datasets(...)`
   - Reads master EDA and service history, computes a target-window sample (TargetFlag) by moving a start date until the desired positive rate is reached for the selected PMS milestone.
   - Output: a filtered `masterdata` DataFrame (one row per VIN / target row) with `TargetFlag` indicating turn-up vs missed.
2. `process_service_data(mastertrain, servhistory, ...)`
   - Reads service history and other source files (RFM, appointments, VHC, digital sessions).
   - Calls many feature derivation helpers (listed below), collects per-VIN feature tables, and merges them into one wide table (`filtered_dfnew`).
   - Performs final transformations, one-hot encoding, iterative imputation and writes `finalmerged{last_service_code}kQ3.csv` (returned DataFrame is `pms_imputed`).
3. Output: cleaned + imputed training DataFrame ready for feature selection and model training. Column `TargetFlag` is the label.

High-level data flow (prediction script: `predservicemil 4.py`)
1. Read service history and EDA.
2. Build a loop over milestone services (e.g., 10,20,30 for a 40k target): for each milestone `svc`:
   - identify eligible VINs (those having that milestone within a date window), compute the same features as training (PMS/NPMS, mileage intervals, branch features, appointment features, complaint features, VHC aggregates), impute and produce `pms_imputed` for that milestone.
   - append per-milestone outputs to a master prediction dataframe `maindf`.
3. Run `NonPMS(...)` to build a dataset for vehicles with no prior PMS (new vehicles) using EDA rules.
4. Combine `maindf` and NonPMS outputs into `finalbase`, enrich with digital & VHC, compute Non-PMS events, and export final CSV(s) for prediction/validation.

Key function families and roles (both files)
- Utilities & parsing
  - `extract_k1`, `extract_kk`, `extract_k`, `derive_servcode`: small parsers converting service description strings into numeric service numbers (e.g., "50k" -> 50 or "<=10" -> 10).
  - `ensure_list`, `normalize_to_list`: normalize list-like values stored as text.
  - `months_to_days`, `calc_next_due`, `calculatenxt_service_interval`: helper math functions for date/month interval computation and converting predicted intervals to next-due dates.

- PMS / NPMS feature derivation
  - `derive_pms_features1(serv1, last_service_code, reference_date)`: computes months since first/last PMS and `SinglePMS` flag per VIN.
  - `derive_npms_features(serv1, reference_date)`: months since first/last NON-PMS per VIN.
  - `derive_pms_mileage_features(serv1, dfpmsdate, last_service_code, svc)` (predservice) / (pmstrain variant): compute average mileage intervals per VIN and apply adjustments for single-record VINs.
  - `derive_npms_mileage_features(serv1)`: average mileage interval for non-PMS events.
  - `derive_pms_service_intervals(serv1, last_service_code)`: compute monthly intervals normalized by mileage and provide fallback to pure-month intervals when needed.
  - `adjust_service_intervals(...)` / `adjust_service_intervalspred(...)`: apply multipliers by distance-to-target to predict service intervals for a future target milestone (avoid leakage by filtering Service_Num < target).
  - `derive_npms_features2(serv1)`: aggregates NPMS counts, NPMS revenue and frequency metrics.

- Appointment / show-up features
  - `map_appointments_to_services(appointments_df, service_df, svc?)`: map each appointment to whether a service occurred within a small window (booked → showed-up mapping).
  - `derive_appointment_show_features(...)`, `compute_late_appointment_metrics(...)`, `compute_last_appointment_status_with_constant_service_code(...)`, `compute_nonpms_appointment_metrics(...)`, `compute_no_show_appointments_test(...)`: routines to compute appointment-level aggregates (show / no-show, late, median delay, last appointment status) used as features.
  - `build_final_pms_appointment_summary(...)`, `adjust_for_target_pms(...)`: helpers to summarize appointment behavior around target PMS and adjust counts for prior-show behavior.

- Branch & VHC features
  - `branch_visit_features(serv, last_service_code, top_n)`: pivot top branch visit counts per VIN (keeps top-N branches and an `otherbranch_services` column).
  - `branch_diversity_features(serv, last_service_code)`: count unique branches serviced.
  - `calculate_bodyshop_count(...)`: count Bodyshop visits per VIN.
  - `vhcpreparation(df, last_service_code, maindf)`: prepare VHC parts/revenue/criticality KPIs and derive top part flags, revenue buckets, conversion & risk metrics.
  - `map_vhc_history(main_df, history_df, svc)`: merge VHC + survey features into the main dataset per `VIN` + `Service_Num`.
  - `backfill_vhc_leakage_safe(...)`, `backfill_vhc_for_target_rows(...)`: backfill VHC/survey features from previous service where target rows are missing (careful about leakage — these functions try to avoid leaking future info).

- Revenue & service counters
  - `compute_service_features(serv, filterdate)`: last PMS/NonPMS revenue, days since last non-PMS and PMS revenue aggregates per VIN.
  - `calculate_revenue_spend(df, last_service)`: sum revenue across service columns (10K.. up to target) into `PMSRevenue`.

- NPMS events & non-PMS specialized logic
  - `get_non_pms_events(serv, servcode_desc, mastertrain, last_service_code)`: derive OHE of last non-PMS event type and revenue for top event types, flag last non-pms event.
  - `get_last_nonpms_mileage(...)` / `get_last_nonpms_before_targetpms(...)`: extract last non-PMS mileage relative to last PMS for fallback features.

- Clustering & categorical compression (pmstrain)
  - `hierarchical_gower_clustering(dfmain, mergedf, feat, ...)`: compute Gower distance on aggregated stats, run hierarchical clustering, assign clusters back to `dfmain`, and produce one-hot cluster columns. Used to compress high-cardinality categorical fields like `Model`, `Variant`, `Nationality`.
  - `map_cluster` (predservice) maps vehicle-level nationality → cluster id using the cluster files produced by the training run.

- Final merging / imputation / export
  - `process_service_data(...)` (pmstrain): the orchestrator that calls the above functions and merges their outputs into a single wide table, performs cleaning, one-hot, `IterativeImputer` (with `Lasso`) and returns `pms_imputed`.
  - `main()` (pmstrain): example driver that sets file names, calls `prepare_pms_datasets(...)` and `process_service_data(...)` and (optionally) writes outputs.
  - Predservice top-level script (not wrapped in `main()`): iterates `svc` in milestone list, runs the same type of feature derivation per milestone, concatenates per-svc `pms_imputed` frames into `maindf`, builds NonPMS set via `NonPMS()` and writes final prediction files.

How the functions serve the training/prediction pipeline
- The scripts are organized so that each domain of features is computed independently and returned as a per-VIN DataFrame (e.g., PMS date features, mileage intervals, branch visits, appointment stats, complaint summaries, VHC aggregates).
- The orchestrator (`process_service_data` or per-milestone loop in `predservicemil`) merges these feature tables on VIN and applies deterministic cleaning (fillna, clipping), then encodes categorical fields (one-hot), and finally runs the `IterativeImputer` to produce a numeric matrix suitable for modelling.
- `TargetFlag` (present in master records returned by `prepare_pms_datasets`) acts as the label for training. The `feature_sel()` helper computes mutual information and can be used to pick a feature subset for model training.

Notes / Gotchas / Recommendations
- Hardcoded filenames: both scripts use many hardcoded CSV filenames and write intermediate files under `validatecode/`. If you intend to run them, ensure the working directory contains these files or adapt the code to accept CLI args.
- Duplication: both files contain large overlapping blocks (PMS feature functions are duplicated). Consider refactoring to a shared module (e.g., `features.py`) and importing it from both scripts.
- Top-level execution: `predservicemil 4.py` executes the pipeline at import time (script-body code at bottom). If you want to call it programmatically, wrap the bottom section in a `main()` and add `if __name__ == '__main__': main()`.
- Performance: many groupby/merge operations and the `gower` matrix (O(n^2)) can be slow on large VIN counts. Run on a machine with sufficient memory and consider sampling or optimizing clustering.
- Leakage: the code attempts to avoid leakage by filtering merges with `Service_Num < target` or `Service_Date <= filter_date`. Keep these rules if you refactor.

Quick run examples
- Run training pipeline (pmstrain example):
```bash
cd d:\Techmax\nurture-mate\ML_layer\code
python "pmstrainfeatureEng 6.py"
```
- Run prediction pipeline (predservice example):
```bash
cd d:\Techmax\nurture-mate\ML_layer\code
python "predservicemil 4.py"
```

Next steps you might want me to do
- Convert duplicated helpers into a single `features.py` and import from both scripts.
- Wrap the bottom-of-file procedural code in `predservemil 4.py` into a `main()` and add CLI args using `argparse`.
- Add a lightweight unit test that checks a small synthetic service history and ensures outputs match expectations.

If you want, I can: (a) refactor duplicated functions into one module, (b) wrap `predservicemil 4.py` execution into a `main()` with configurable arguments, or (c) run the scripts with sample files if you provide them.
