"""
eval_q1_expected_cohort.py -- Full expected-cohort evaluation for Q1 2026, 20k milestone.

Builds the expected cohort using create_candidates_test_cohort() from prepare_test_set.py
(no edits to that file), labels each vehicle as:
    1 = actually turned up for 20k (by end of Q1 + grace)
    0 = expected but did not show up

Then runs feature engineering (with milestone history masked to prevent leakage), scores
with the candidate model, and reports full classification metrics: AUC, precision, recall,
F1, confusion matrix, and recall-at-threshold sweep.

Usage:
    python eval_q1_expected_cohort.py
    python eval_q1_expected_cohort.py --model-dir models/20k_cand_refix_wins_2023
    python eval_q1_expected_cohort.py --threshold 0.6
    python eval_q1_expected_cohort.py --grace-days 30

No existing files are modified by this script.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.abspath(os.path.dirname(__file__)) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from date_utils import parse_dates
from features import resolve_service_num
from pms_model import (
    build_inference_matrix,
    features_path,
    load_artifacts,
    model_dir,
    predict_proba,
    split_missing,
)
from pmstrainfeatureeng_refactored import process_service_data, extract_k1

# Import the cohort builder from prepare_test_set without modifying it
from prepare_test_set import create_candidates_test_cohort

DEVICE = torch.device("cpu")

MILESTONE = 20

# Q1 2026 window
Q1_START = "2026-01-01"
Q1_END   = "2026-03-31"
# Feature cutoff: last day before the quarter, same rule as prepare_test_set.py
FEATURE_CUTOFF = "2025-12-31"

DATA = {
    "eda":      os.path.join("data", "EDA_Q2-2026.csv"),
    "serv":     os.path.join("data", "Service_History_Q2-2026.csv"),
    "rfm":      os.path.join("data", "RFM_Q2-2026.csv"),
    "appoint":  os.path.join("data", "Appointments_Q2-2026.csv"),
    "digi":     os.path.join("data", "Digital_Sessions_Q2-2026.csv"),
    "vhc":      os.path.join("data", "VHC_Q2-2026.csv"),
    "servcode": os.path.join("data", "Service_Code_Desc.csv"),
}

DEFAULT_MODEL_DIR = os.path.join("models", "20k_cand_refix_wins_2023")
DEFAULT_FEATURES  = os.path.join("models", "selected_features_20k_cand_refix_wins_2023.json")


def load_masked_history(serv_path: str, milestone: int) -> pd.DataFrame:
    """Load service history and drop all rows where Service_Num == milestone.

    Prevents the feature pipeline from seeing the target visit while keeping
    all earlier and later service rows intact for feature derivation.
    Returned as a parsed DataFrame so process_service_data() does not re-read
    the 124 MB CSV.
    """
    print(f"Loading + masking history (dropping Service_Num == {milestone}) ...")
    serv = pd.read_csv(serv_path, low_memory=False, encoding="ISO-8859-1")
    n_before = len(serv)
    serv["Service_Date"] = parse_dates(serv["Service_Date"])
    serv["Service_Num"]  = resolve_service_num(serv)
    serv = serv[serv["Service_Num"] != milestone].copy()
    print(f"  Dropped {n_before - len(serv):,} rows; {len(serv):,} remain")
    return serv


def build_eda_cohort(cohort_labels: pd.DataFrame, milestone: int) -> pd.DataFrame:
    """Attach pipeline-required columns to the labelled cohort frame.

    create_candidates_test_cohort() returns the EDA frame plus Expected{m}kDate,
    TargetFlag, EarlyCompleter, etc.  process_service_data() additionally needs
    Service_Num (from 'Last Service - PMS') and FirstSrvDate to already exist.
    Both are set here without touching the original cohort builder.
    """
    df = cohort_labels.copy()

    # FirstSrvDate may already exist from the cohort builder; ensure it does.
    if "FirstSrvDate" not in df.columns:
        df["Invoice date"]       = parse_dates(df["Invoice date"])
        df["First Service Date"] = parse_dates(df["First Service Date"])
        df["FirstSrvDate"]       = df["Invoice date"].fillna(df["First Service Date"])

    # Service_Num from EDA 'Last Service - PMS'
    if "Last Service - PMS" in df.columns:
        df["Service_Num"] = df["Last Service - PMS"].apply(extract_k1)
    else:
        df["Service_Num"] = 0

    return df


def run_feature_engineering(
    cohort: pd.DataFrame,
    serv_masked: pd.DataFrame,
    milestone: int,
    feature_cutoff: str,
) -> pd.DataFrame:
    """Run process_service_data() on the masked cohort.

    is_test=True skips the mastertrain date filter that would otherwise drop
    vehicles whose Expected{m}kDate is after feature_cutoff (i.e. the whole
    expected cohort), matching how prepare_test_set.py invokes it.
    """
    print(f"\nRunning feature engineering (cutoff={feature_cutoff}, n={len(cohort)}) ...")
    return process_service_data(
        mastersheet=cohort,
        servhistory=serv_masked,
        rfm=DATA["rfm"],
        servcode=DATA["servcode"],
        appointdf=DATA["appoint"],
        digidf=DATA["digi"],
        vhc=DATA["vhc"],
        filter_date=feature_cutoff,
        last_service_code=milestone,
        is_test=True,
    )


def score_cohort(
    features_df: pd.DataFrame,
    milestone: int,
    mdir: str,
    features_file: str,
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load artifacts, build inference matrix, predict.

    Returns (out_df, y, probs, pred).
    """
    for fname in ["best_model.pt", "imputer.joblib", "scaler.joblib"]:
        fpath = os.path.join(mdir, fname)
        if not os.path.exists(fpath):
            sys.exit(f"Missing {fpath} -- run: python retrain.py {milestone}")
    if not os.path.exists(features_file):
        sys.exit(f"Missing {features_file} -- run: python retrain.py {milestone}")

    model, imputer, scaler, selected_features = load_artifacts(
        milestone, DEVICE, mdir=mdir, features_file=features_file
    )

    y = (
        features_df["TargetFlag"].astype(int).values
        if "TargetFlag" in features_df.columns
        else None
    )
    vin = (
        features_df["VIN"]
        if "VIN" in features_df.columns
        else pd.Series(features_df.index, name="VIN")
    )

    X, missing = build_inference_matrix(features_df, milestone, selected_features)
    print(f"\n{milestone}k | rows={len(X)} | features={len(selected_features)} | threshold={threshold:.4f}")

    if missing:
        expected_m, gaps = split_missing(missing, features_df.columns)
        if expected_m:
            print(f"  note: {len(expected_m)} absent one-hot categories zero-filled (correct)")
        if gaps:
            print(
                f"  WARNING: {len(gaps)} features genuinely missing: "
                f"{gaps[:15]}{'...' if len(gaps) > 15 else ''}"
            )

    probs = predict_proba(model, imputer, scaler, X, DEVICE)
    pred  = (probs > threshold).astype(int)

    out = pd.DataFrame({"VIN": vin.values, "prob_turnup": probs, "pred": pred})
    if y is not None:
        out["TargetFlag"] = y
    out = out.sort_values("prob_turnup", ascending=False).reset_index(drop=True)

    return out, y, probs, pred


def print_metrics(y: np.ndarray, probs: np.ndarray, pred: np.ndarray, threshold: float) -> None:
    """Full evaluation report: distribution, classification metrics, recall sweep."""
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    sep = "-" * 60

    print(f"\n{sep}")
    print(f"FULL COHORT EVALUATION   20k  Q1 2026")
    print(sep)
    print(f"Total vehicles   : {len(y):,}  ({n_pos:,} positives / {n_neg:,} negatives)")
    print(f"Base rate        : {100.0 * n_pos / max(len(y), 1):.1f}%")

    print(f"\nProbability distribution (all vehicles):")
    print(f"  mean   {probs.mean():.4f}")
    print(f"  median {np.median(probs):.4f}")
    print(f"  p25    {np.percentile(probs, 25):.4f}")
    print(f"  p75    {np.percentile(probs, 75):.4f}")

    print(f"\nProbability by class:")
    print(f"  Positives  mean={probs[y == 1].mean():.4f}  median={np.median(probs[y == 1]):.4f}")
    print(f"  Negatives  mean={probs[y == 0].mean():.4f}  median={np.median(probs[y == 0]):.4f}")

    print(f"\nClassification metrics at threshold {threshold:.4f}:")
    print(f"  Accuracy   {100 * accuracy_score(y, pred):.2f}%")
    print(f"  Precision  {100 * precision_score(y, pred, zero_division=0):.2f}%")
    print(f"  Recall     {100 * recall_score(y, pred, zero_division=0):.2f}%")
    print(f"  F1         {100 * f1_score(y, pred, zero_division=0):.2f}%")
    print(f"  AUC-ROC    {roc_auc_score(y, probs):.4f}")
    print(f"  Brier      {brier_score_loss(y, probs):.4f}")

    cm = confusion_matrix(y, pred)
    print(f"\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':12} {'pred=0':>8} {'pred=1':>8}")
    print(f"  {'actual=0':12} {cm[0, 0]:>8,} {cm[0, 1]:>8,}")
    print(f"  {'actual=1':12} {cm[1, 0]:>8,} {cm[1, 1]:>8,}")

    print(f"\nPrecision / Recall at thresholds:")
    print(f"  {'Threshold':>10}  {'Captured':>9}  {'Recall %':>9}  {'Precision %':>12}  {'Pred list':>9}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
        p_pred = (probs >= thr).astype(int)
        cap    = int((p_pred[y == 1] == 1).sum())
        rec    = 100.0 * cap / max(n_pos, 1)
        prec   = 100.0 * precision_score(y, p_pred, zero_division=0)
        n_pred = int(p_pred.sum())
        marker = " <-- current" if abs(thr - threshold) < 1e-4 else ""
        print(f"  {thr:>10.2f}  {cap:>9,}  {rec:>8.1f}%  {prec:>11.1f}%  {n_pred:>9,}{marker}")
    print(sep)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Full expected-cohort evaluation: Q1 2026, 20k, candidate model"
    )
    p.add_argument("--model-dir",  default=DEFAULT_MODEL_DIR)
    p.add_argument("--features",   default=DEFAULT_FEATURES)
    p.add_argument("--threshold",  type=float, default=None,
                   help="Score threshold (default: threshold.json or 0.5)")
    p.add_argument("--grace-days", type=int, default=0,
                   help="Grace window in days after Q1 end for labelling positives (default: 0)")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: predictions/20k/eval_q1_expected_full.csv)")
    args = p.parse_args()

    for name, path in DATA.items():
        if not os.path.exists(path):
            sys.exit(f"Missing data file [{name}]: {path}")

    # Resolve threshold
    threshold = args.threshold
    if threshold is None:
        tp = os.path.join(args.model_dir, "threshold.json")
        threshold = json.load(open(tp))["threshold"] if os.path.exists(tp) else 0.5
    print(f"Model dir : {args.model_dir}")
    print(f"Features  : {args.features}")
    print(f"Threshold : {threshold:.4f}")
    print(f"Grace days: {args.grace_days}")

    # Step 1: Build the expected cohort with labels via the unmodified cohort builder
    print(f"\nStep 1: Building expected cohort (Q1 {Q1_START} to {Q1_END}) ...")
    cohort_labels = create_candidates_test_cohort(
        eda_path=DATA["eda"],
        service_history_path=DATA["serv"],
        milestone=MILESTONE,
        start_date=Q1_START,
        end_date=Q1_END,
        grace_days=args.grace_days,
    )

    # Step 2: Load masked service history
    print(f"\nStep 2: Loading masked service history ...")
    serv_masked = load_masked_history(DATA["serv"], MILESTONE)

    # Step 3: Attach pipeline columns to the labelled cohort
    print(f"\nStep 3: Preparing cohort for feature engineering ...")
    cohort = build_eda_cohort(cohort_labels, MILESTONE)

    # Step 3b: Drop early completers.
    # EarlyCompleter == 1 means the vehicle finished 20k BEFORE the quarter window opened
    # (before 2026-01-01), so its outcome was already known at the feature cutoff (2025-12-31).
    # Scoring them inflates recall and AUC: the model sees rich history and scores them high,
    # but we already knew they had come in before the quarter even started.
    # Per prepare_test_set.py:76-82 and the AGENTS.md rule, they are dropped from test evaluation.
    if "EarlyCompleter" in cohort.columns:
        n_early = int((cohort["EarlyCompleter"] == 1).sum())
        cohort = cohort[cohort["EarlyCompleter"] != 1].copy()
        print(f"  Dropped {n_early:,} early completers (finished before {Q1_START})")
    else:
        print("  WARNING: EarlyCompleter column not present -- no early-completer exclusion applied")

    print(f"  Cohort size after exclusion: {len(cohort):,}  "
          f"(positives={cohort['TargetFlag'].sum():,}  "
          f"negatives={(cohort['TargetFlag'] == 0).sum():,})")


    # Step 4: Feature engineering
    print(f"\nStep 4: Feature engineering ...")
    features_df = run_feature_engineering(cohort, serv_masked, MILESTONE, FEATURE_CUTOFF)

    # TargetFlag may have been dropped or joined differently inside process_service_data;
    # re-attach from the cohort frame by VIN to be safe.
    if "TargetFlag" not in features_df.columns and "VIN" in features_df.columns:
        label_map = cohort.set_index("VIN")["TargetFlag"].to_dict()
        features_df["TargetFlag"] = features_df["VIN"].map(label_map).fillna(0).astype(int)
        print(f"  TargetFlag re-attached from cohort; "
              f"positives={features_df['TargetFlag'].sum():,}")

    # Step 5: Score
    print(f"\nStep 5: Scoring ...")
    out, y, probs, pred = score_cohort(
        features_df, MILESTONE, args.model_dir, args.features, threshold
    )

    # Step 6: Save
    out_path = args.out or os.path.join("predictions", "20k", "eval_q1_expected_full.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out):,} rows -> {out_path}")

    # Step 7: Full metrics
    if y is None or len(set(y)) < 2:
        print("No binary labels available -- skipping metrics.")
        return

    print_metrics(y, probs, pred, threshold)


if __name__ == "__main__":
    main()
