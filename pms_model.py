"""
Shared model + inference pieces for the current PMS pipeline.

Used by retrain.py (training), score_milestone.py (scoring) and eval_test_metrics.py (held-out
metrics). Before this module existed, BinaryClassifier was defined 5 times, add_milestone_flags
3 times and the inference feature-derivation block 3 times across those files. They agreed only
because they were hand-copied; any fix to one silently left the others behind.

The legacy trainers (train_milestone.py, legacy/parked/training_run.py) keep their own copies on
purpose: they use a different network (ServicePredictionNN, dropout 0.5) and StandardScaler.
"""
import os
import json

import pandas as pd
import torch
import torch.nn as nn
import joblib


# 80k is exempt from has_x: its Q2 test file records Service_Num = 0 for all 651 positives (Q1
# correctly records 80), so has_x becomes 0 everywhere for them and the model calls them all
# no-shows (1.7% recall, TP=11/651). Leaving has_x off for 80k keeps both quarters usable.
HAS_X_EXCLUDE = {80}

# Physically implausible for a passenger/light-commercial vehicle even across the full 2016-2025
# candidates span -- values above this in 'LastNonPMSMileage' / 'Last Service Mileage' are data
# entry errors, not real odometer readings. Measured 2026-08-19 on the 20k candidates matrix: the
# distribution is continuous from 0 up through ~975,505 (no gap -- high-mileage fleet/commercial
# vehicles are real), then jumps to 1,001,127 and climbs to 9,953,052 (73 rows over 1,000,000, none
# of them caught by IsolationForest's contamination=0.005 outlier removal -- all 73 fed straight
# into training). A hard domain cap, not a percentile, so it does not shift on every data refresh.
MAX_PLAUSIBLE_MILEAGE_KM = 500_000

# RFM segment one-hots are a cohort-vintage marker, not a behaviour signal. Segments are
# recency-based against a single fixed snapshot, so training positives (completed the milestone
# years ago) land in Lost/Hibernating/At Risk -- ~67% of them -- while test positives (completed
# it in 2026) land in Potential Loyalist/Promising. Those training segments are 0.0% of test
# positives, so the model learns a rule that cannot fire at test time. Q1 and Q2 positives sit on
# opposite sides of the Potential Loyalist/Promising recency boundary, which is what made 60k
# score 92.02 on Q1 and 73.65 on Q2. Dropping them: Q1 87.50 / Q2 88.00 -- 11.5pt gap -> 0.5pt.
DROP_RFM = {60}

# Columns that leak the label or encode it indirectly. Service_Num is exact:
# Service_Num == milestone <=> TargetFlag == 1, zero exceptions.
LEAK_COLS = ['Service_Num', 'Vehicle_Key_Actual_Service', 'Vehicle Age',
             'Current Age', 'Vehicle Lifetime in Months']

# Built at inference time by derive_inference_features(), never produced by
# process_service_data(). Do not try to select a test set down to these.
DERIVED_FEATURES = ['months_to_10k', 'Last Service Mileage', 'Max_PMS_Revenue',
                    'Last_PMS_Revenue', 'Min_PMS_Revenue', 'StdDev_PMS_Revenue']

# Which of the four candidate PMS_Delay definitions reaches the model. features.compute_pms_delay()
# emits all four into every matrix; exactly one survives here, always renamed to `PMS_Delay` so the
# feature list is identical across variants and an A/B compares values, not schema.
#
#   legacy    EDAexp - EDAact               production default. Carries an IMPLICIT -1 on positives,
#                                           because EDAact counts the target service for turn-ups
#                                           and only for turn-ups (measured: EDAact == sum(has_x)
#                                           + TargetFlag to three decimals on both matrices).
#   doc       legacy - TargetFlag           the doc's explicit turn-up -1, which given the above is
#                                           really -2 on positives. Label-conditioned by
#                                           construction and unavailable at inference -- expect it
#                                           to score well in training and not transfer.
#   derivedA  EDAexp - sum(has_x)           actual term from cutoff-bound history; no label term.
#   derivedB  2*YSFP - sum(has_x)           both terms from cutoff-bound history. The only variant
#                                           that also breaks the tie to `Vehicle Age` (LEAK_COLS),
#                                           since EDAexp == 2 * Vehicle Age (99.90% within 1).
PMS_DELAY_VARIANT = os.environ.get('PMS_DELAY_VARIANT', 'legacy')
PMS_DELAY_VARIANT_COL = {'legacy': 'PMS_Delay', 'doc': 'PMS_Delay_Doc',
                         'derivedA': 'PMS_Delay_DerivedA', 'derivedB': 'PMS_Delay_DerivedB'}


def select_pms_delay_variant(df, variant=None):
    """Collapse the four PMS_Delay* candidates to a single column named `PMS_Delay`.

    Call immediately after loading a matrix, in every train/score/eval path, so the choice cannot
    diverge between training and inference. The survivor is always `PMS_Delay`, so
    selected_features_{m}k.json is byte-identical across variants -- an A/B then differs only in
    the values behind that one name.

    Falls back to whatever `PMS_Delay` the frame already has when the requested variant is absent
    (older matrices predate the multi-variant output), and says so.
    """
    variant = variant or PMS_DELAY_VARIANT
    if variant not in PMS_DELAY_VARIANT_COL:
        raise ValueError(f"PMS_DELAY_VARIANT={variant!r} unknown; "
                         f"expected one of {sorted(PMS_DELAY_VARIANT_COL)}")

    src = PMS_DELAY_VARIANT_COL[variant]
    if src in df.columns:
        df['PMS_Delay'] = df[src]
        print(f"PMS_Delay variant: {variant} (from {src})")
    else:
        print(f"WARNING select_pms_delay_variant: {src!r} absent -- keeping the existing "
              f"PMS_Delay. Regenerate the matrix to A/B variant {variant!r}.")

    extra = [c for c in PMS_DELAY_VARIANT_COL.values()
             if c != 'PMS_Delay' and c in df.columns]
    return df.drop(columns=extra, errors='ignore')


def resolve_path(path):
    """Data dirs (traindataq1_q2/, testdataq1_q2/) currently sit under tests_alan/ rather than at
    the repo root the scripts assume. Try the root path first, then the tests_alan/ copy."""
    if os.path.exists(path):
        return path
    alt = os.path.join("tests_alan", path)
    if os.path.exists(alt):
        return alt
    return path


def model_dir(milestone):
    """Trained artifacts for one milestone: models/{m}k/ -- best_model.pt, imputer.joblib,
    scaler.joblib, threshold.json, metrics_*.json.

    Was models/models_alan/{m}k/ until 2026-08-12. Everything now lives under models/ directly.
    """
    return os.path.join("models", f"{milestone}k")


def features_path(milestone):
    """THE feature list for one milestone: models/selected_features_{m}k.json.

    One file per model, at the models/ root. Written by retrain.py after the test-schema
    intersection, read by every scoring path -- so what the model was trained on and what it is fed
    at inference cannot diverge.

    Replaces two same-named, incompatible files that used to coexist:
      models/{m}k/selected_features.json         -> feature_sel() MI shortlist, ~33 cols
      models/models_alan/{m}k/selected_features.json -> real model inputs, ~113-141 cols
    Loading the wrong one zero-filled 67-78% of the model's inputs and still ran clean.
    """
    return os.path.join("models", f"selected_features_{milestone}k.json")


def id_cols(milestone):
    """Identifier columns plus LEAK_COLS -- dropped from both training and inference matrices.

    `CohortQuarter` is bookkeeping written by run_history_mode() (which quarterly cohort a row came
    from). It is a string, it exists only in history-mode matrices, and it names the row's due date
    -- i.e. it is an identifier, not a feature.

    `EarlyCompleter`, `DueSource`, `QuarterEnd` and `Actual{m}kDate` are the same kind of thing,
    written by run_candidates_mode()/build_candidates_cohort() (--label-mode candidates):
    EarlyCompleter flags a positive that completed the milestone before its own due quarter opened
    (so its outcome was already settled -- see the labelling plan's Stage D slicing), DueSource
    records which of the two due-date models won ('schedule' or 'burn-rate'), QuarterEnd is the due
    quarter's end date, and Actual{m}kDate is the milestone's real completion date. All four are
    bookkeeping used to build/audit the label, never a model input -- Actual{m}kDate in particular
    is TargetFlag's source, so leaving it in would be as direct a leak as Service_Num.
    """
    return ['VIN', 'Vehicle Key', 'Customer ID', 'Last Service Date - PMS', 'CohortQuarter',
            'EarlyCompleter', 'DueSource', 'QuarterEnd', f'Actual{milestone}kDate',
            f'Next{milestone}K_Due'] + LEAK_COLS


class BinaryClassifier(nn.Module):
    """64 -> 32 -> 1 with BatchNorm and dropout 0.2. Outputs logits, not probabilities."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def add_milestone_flags(df, milestone, use_has_x=True):
    """has_x = 1 if the vehicle completed prior milestone x. FALLBACK ONLY -- never overwrites.

    Added for x in [10, 20, ..., milestone-10] only -- the target milestone is excluded so no flag
    directly encodes TargetFlag.

    A flag already in the frame is LEFT ALONE. pmstrainfeatureeng_refactored.py:343-348 builds these
    from `serv1`, which is already cut at `Service_Date <= filterdate`, so the matrices and test sets
    carry exactly the intended thing: "by the cutoff, had this vehicle done its 50k?". The version
    computed here -- (Service_Num >= x), where Service_Num comes from EDA's post-outcome
    `Last Service - PMS` -- is a snapshot taken after the outcome, and until 2026-08-16 it silently
    replaced the good one. Measured on 60k, AUC as built -> after the overwrite:

        has_10  0.6197 -> 0.5000     has_40  0.7108 -> 0.8313
        has_20  0.5119 -> 0.6128     has_50  0.7879 -> 0.9223
        has_30  0.5617 -> 0.6907

    has_50's positive-class rate went 0.732 -> 1.000 while negatives held at 0.155: the overwrite
    forced the flag on for every training positive. It injected a leak AND destroyed real signal
    (has_10 collapsed to constant 1). The fallback is kept because inference frames built outside
    process_service_data() genuinely lack the columns; Service_Num must still be present then.
    """
    if use_has_x and milestone not in HAS_X_EXCLUDE and 'Service_Num' in df.columns:
        sn = pd.to_numeric(df['Service_Num'], errors='coerce').fillna(0)
        built = []
        for x in range(10, milestone, 10):   # excludes milestone itself
            assert x != milestone, "target-milestone flag must never be created"
            if f'has_{x}' in df.columns:
                continue                     # cutoff-bound version already there -- keep it
            df[f'has_{x}'] = (sn >= x).astype(int)
            built.append(f'has_{x}')
        if built:
            print(f"WARNING add_milestone_flags: {len(built)} flag(s) missing from the frame, "
                  f"derived from post-outcome Service_Num instead: {built}")
    return df


def derive_inference_features(X):
    """Add the DERIVED_FEATURES columns the models expect but the feature-eng output lacks.

    Every fallback here is now ANNOUNCED. It used to be silent, and that cost a real result:
    'Last Service Mileage' is built from 'Last PMS Mileage' and 'LastNonPMSMileage', and there was
    a branch for the first alone but none for the second alone. Dropping 'Last PMS Mileage' as an
    EDA current-state column therefore sent every row down the else branch to a CONSTANT 0 -- for
    all 41,983 rows of the 20k candidates matrix. The test sets still had the source column, so the
    same feature varied at inference: the model could learn nothing from it in training and was fed
    noise at scoring time. Measured single-feature AUC: 0.5000 on train (i.e. constant), 0.158 on
    test. Nothing failed, nothing warned; it took a feature-by-feature audit to find.

    A fallback firing is not automatically wrong -- a test cohort legitimately lacks categories a
    training frame has -- but a fallback that fires for EVERY row means the column is constant, and
    a constant feature is at best dead weight. retrain.py now checks for that separately.
    """
    fired = []

    if 'Avg_Service_Interval_PMS' in X.columns:
        X['months_to_10k'] = X['Avg_Service_Interval_PMS'].fillna(6.0)
    else:
        X['months_to_10k'] = 6.0
        fired.append("months_to_10k <- constant 6.0 ('Avg_Service_Interval_PMS' absent)")

    has_pms = 'Last PMS Mileage' in X.columns
    has_npms = 'LastNonPMSMileage' in X.columns
    if has_pms and has_npms:
        X['Last Service Mileage'] = X[['Last PMS Mileage', 'LastNonPMSMileage']].max(axis=1)
    elif has_pms:
        X['Last Service Mileage'] = X['Last PMS Mileage']
        fired.append("Last Service Mileage <- 'Last PMS Mileage' only "
                     "('LastNonPMSMileage' absent)")
    elif has_npms:
        # THE MISSING BRANCH. Without it this case fell through to the constant 0 below.
        X['Last Service Mileage'] = X['LastNonPMSMileage']
        fired.append("Last Service Mileage <- 'LastNonPMSMileage' only "
                     "('Last PMS Mileage' absent -- expected when the EDA current-state "
                     "columns are dropped)")
    else:
        X['Last Service Mileage'] = 0
        fired.append("Last Service Mileage <- CONSTANT 0 (both source columns absent) "
                     "-- this feature is dead, drop it via PMS_DROP")

    # Winsorize regardless of which branch built it -- the source columns carry the same corrupted
    # tail (see MAX_PLAUSIBLE_MILEAGE_KM). This runs at both training and inference/scoring time
    # (derive_inference_features() is the shared path), so a genuinely bad reading on a real vehicle
    # at serving time is bounded the same way a training-set outlier is, instead of producing an
    # unbounded scaled input the network was never trained to handle.
    lsm = pd.to_numeric(X['Last Service Mileage'], errors='coerce')
    _n_clipped = int((lsm > MAX_PLAUSIBLE_MILEAGE_KM).sum())
    if _n_clipped:
        fired.append(f"Last Service Mileage: clipped {_n_clipped} row(s) above "
                     f"{MAX_PLAUSIBLE_MILEAGE_KM:,} km (implausible odometer reading)")
    X['Last Service Mileage'] = lsm.clip(upper=MAX_PLAUSIBLE_MILEAGE_KM)

    if 'PMSRevenue' in X.columns:
        pms_rev = X['PMSRevenue']
    else:
        pms_rev = pd.Series(0, index=X.index)
        fired.append("Max/Last/Min/StdDev_PMS_Revenue <- CONSTANT 0 ('PMSRevenue' absent) "
                     "-- all four are dead, drop them via PMS_DROP")
    # Note: these four are all deterministic transforms of one column, so they carry no information
    # the model does not already have from PMSRevenue itself. Min/StdDev in particular are invented
    # (0.5x and 0.2x), not measured. Kept because the trained feature lists expect them.
    X['Max_PMS_Revenue'] = pms_rev
    X['Last_PMS_Revenue'] = pms_rev
    X['Min_PMS_Revenue'] = pms_rev * 0.5
    X['StdDev_PMS_Revenue'] = pms_rev * 0.2

    for msg in fired:
        print(f"WARNING derive_inference_features: {msg}")
    return X


def load_artifacts(milestone, device=None, mdir=None, features_file=None):
    """Load a trained milestone model. Returns (model, imputer, scaler, selected_features).

    The feature list comes from features_path(milestone) -- one canonical file per model.
    `mdir` overrides only where the weights/scaler/imputer are read from (experiment dirs).

    `features_file` overrides the feature list, and a PMS_RUN_TAG run needs it: retrain.py writes
    models/selected_features_{m}k{tag}.json, so loading a tagged mdir against the untagged list
    raises a state_dict size mismatch when the two disagree on feature count -- or, worse, loads
    silently when they happen to match and feeds the model a differently-ordered matrix.
    """
    device = device or torch.device("cpu")
    mdir = mdir or model_dir(milestone)
    selected_features = json.load(open(features_file or features_path(milestone)))
    imputer = joblib.load(os.path.join(mdir, 'imputer.joblib'))
    scaler = joblib.load(os.path.join(mdir, 'scaler.joblib'))
    model = BinaryClassifier(input_dim=len(selected_features)).to(device)
    model.load_state_dict(torch.load(os.path.join(mdir, 'best_model.pt'), map_location=device))
    model.eval()
    return model, imputer, scaler, selected_features


def family_of(col):
    """The per-category encoding family a column belongs to.

    `DeferredPart__Air Filter` -> `DeferredPart`, `Franchise_0` -> `Franchise`. Columns with no
    separator are their own family, so they never match anything else by accident.
    """
    if '__' in col:
        return col.split('__')[0]
    return col.rsplit('_', 1)[0] if '_' in col else col


def split_missing(missing, df_columns):
    """Split zero-filled columns into (expected_absent, real_gaps).

    Not every zero-fill is a problem. Per-category columns are only emitted for categories present
    in the frame, so a small cohort legitimately lacks the rare ones -- and 0 (absent, or a count of
    zero) is the CORRECT value there. That is completely different from a feature the pipeline never
    computed, where 0 is meaningless and any metric built on it is worthless.

    A missing column is "expected absent" when the frame already carries at least one sibling from
    the same family: that proves the pipeline does produce the family, so the gap is the category,
    not the code. If the whole family is missing, the pipeline never produced it -- a real gap.

    Note this deliberately does NOT test for binary 0/1 values. Several `DeferredPart__*` /
    `InvoicedPart__*` / `LostPart__*` columns are per-category COUNTS rather than flags, and a count
    of 0 for a category that never occurred is just as correct as an absent dummy.
    """
    present = {family_of(c) for c in df_columns}
    expected = [c for c in missing if family_of(c) in present]
    gaps = [c for c in missing if family_of(c) not in present]
    return expected, gaps


def build_inference_matrix(df, milestone, selected_features, use_has_x=True):
    """Turn a raw cohort/test frame into the exact matrix the model consumes.

    Returns (X, missing) where missing lists the selected features that were absent and had to be
    zero-filled. Pass `missing` through split_missing() before reporting it: real gaps are the
    dominant cause of bad metrics, absent one-hot categories are harmless and would otherwise bury
    them.
    """
    df = add_milestone_flags(df, milestone, use_has_x=use_has_x)
    # Collapse the four PMS_Delay* candidates to one before anything else reads them, so scoring
    # uses the same definition the model was trained on. Covers score_milestone.py and
    # eval_test_metrics.py, which both reach the matrix through here.
    df = select_pms_delay_variant(df)
    X = df.drop(columns=[c for c in id_cols(milestone) if c in df.columns], errors='ignore')
    X = X.drop(columns=['TargetFlag'], errors='ignore')

    X = derive_inference_features(X)

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(0)

    missing = [c for c in selected_features if c not in X.columns]
    for c in missing:
        X[c] = 0
    return X[selected_features], missing


def predict_proba(model, imputer, scaler, X, device=None):
    """Impute -> scale -> forward -> sigmoid. Always returns a 1-D numpy array."""
    import numpy as np
    device = device or torch.device("cpu")
    Xs = scaler.transform(imputer.transform(X))
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.FloatTensor(Xs).to(device)).squeeze()).cpu().numpy()
    return np.atleast_1d(probs)
