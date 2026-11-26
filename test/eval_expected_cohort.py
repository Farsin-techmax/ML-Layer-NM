"""
eval_expected_cohort.py -- Full expected-cohort evaluation for any milestone and quarter.

Generalizes eval_q1_expected_cohort.py to accept milestone and quarter as CLI args.
No existing pipeline files are modified.

Usage:
    python eval_expected_cohort.py 30 --quarter Q1 --model-dir models/30k_cand_wins_2023 --features models/selected_features_30k_cand_wins_2023.json
    python eval_expected_cohort.py 60 --quarter Q1 --model-dir models/60k --features models/selected_features_60k.json
    python eval_expected_cohort.py 40 --quarter Q2 --model-dir models/40k_cand_wins_2023 --features models/selected_features_40k_cand_wins_2023.json --grace-days 30

Quarter windows (hardcoded to 2026 -- update QUARTER_DATES if your quarter differs):
    Q1: 2026-01-01 to 2026-03-31  (feature cutoff: 2025-12-31)
    Q2: 2026-04-01 to 2026-06-30  (feature cutoff: 2026-03-31)

Output:
    predictions/{m}k/eval_{m}k_{quarter}_expected_full.csv
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
from prepare_test_set import create_candidates_test_cohort

DEVICE = torch.device("cpu")

QUARTER_DATES = {
    "Q1": ("2026-01-01", "2026-03-31", "2025-12-31"),
    "Q2": ("2026-04-01", "2026-06-30", "2026-03-31"),
}

DATA = {
    "eda":      os.path.join("data", "EDA_Q2-2026.csv"),
    "serv":     os.path.join("data", "Service_History_Q2-2026.csv"),
    "rfm":      os.path.join("data", "RFM_Q2-2026.csv"),
    "appoint":  os.path.join("data", "Appointments_Q2-2026.csv"),
    "digi":     os.path.join("data", "Digital_Sessions_Q2-2026.csv"),
    "vhc":      os.path.join("data", "VHC_Q2-2026.csv"),
    "servcode": os.path.join("data", "Service_Code_Desc.csv"),
}


def load_masked_history(serv_path: str, milestone: int) -> pd.DataFrame:
    """Load service history and drop rows where Service_Num == milestone."""
    print(f"Loading + masking history (dropping Service_Num == {milestone}) ...")
    serv = pd.read_csv(serv_path, low_memory=False, encoding="ISO-8859-1")
    n_before = len(serv)
    serv["Service_Date"] = parse_dates(serv["Service_Date"])
    serv["Service_Num"]  = resolve_service_num(serv)
    serv = serv[serv["Service_Num"] != milestone].copy()
    print(f"  Dropped {n_before - len(serv):,} rows; {len(serv):,} remain")
    return serv


def build_eda_cohort(cohort_labels: pd.DataFrame, milestone: int) -> pd.DataFrame:
    """Attach pipeline-required columns to the labelled cohort frame."""
    df = cohort_labels.copy()
    if "FirstSrvDate" not in df.columns:
        df["Invoice date"]       = parse_dates(df["Invoice date"])
        df["First Service Date"] = parse_dates(df["First Service Date"])
        df["FirstSrvDate"]       = df["Invoice date"].fillna(df["First Service Date"])
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
    """Run process_service_data() on the masked cohort with is_test=True."""
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
) -> tuple:
    """Load artifacts, build matrix, predict. Returns (out_df, y, probs, pred)."""
    for fname in ["best_model.pt", "imputer.joblib", "scaler.joblib"]:
        fpath = os.path.join(mdir, fname)
        if not os.path.exists(fpath):
            sys.exit(f"Missing {fpath} -- run: python retrain.py {milestone}")
    if not os.path.exists(features_file):
        sys.exit(f"Missing {features_file} -- run: python retrain.py {milestone}")

    model, imputer, scaler, selected_features = load_artifacts(
        milestone, DEVICE, mdir=mdir, features_file=features_file
    )

    y   = features_df["TargetFlag"].astype(int).values if "TargetFlag" in features_df.columns else None
    vin = features_df["VIN"] if "VIN" in features_df.columns else pd.Series(features_df.index, name="VIN")

    X, missing = build_inference_matrix(features_df, milestone, selected_features)
    print(f"\n{milestone}k | rows={len(X)} | features={len(selected_features)} | threshold={threshold:.4f}")

    if missing:
        exp_m, gaps = split_missing(missing, features_df.columns)
        if exp_m:
            print(f"  note: {len(exp_m)} absent one-hot categories zero-filled (correct)")
        if gaps:
            print(f"  WARNING: {len(gaps)} features genuinely missing: {gaps[:15]}{'...' if len(gaps) > 15 else ''}")

    probs = predict_proba(model, imputer, scaler, X, DEVICE)
    pred  = (probs > threshold).astype(int)

    out = features_df.copy()
    out["prob_turnup"] = probs
    out["pred"] = pred
    if "TargetFlag" not in out.columns and y is not None:
        out["TargetFlag"] = y

    # Position key identification and prediction columns first, followed by all features
    lead_cols = [c for c in ["VIN", "Vehicle Key", "Customer ID", "TargetFlag", "prob_turnup", "pred"] if c in out.columns]
    other_cols = [c for c in out.columns if c not in lead_cols]
    out = out[lead_cols + other_cols].sort_values("prob_turnup", ascending=False).reset_index(drop=True)
    return out, y, probs, pred


def print_metrics(
    y: np.ndarray,
    probs: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    milestone: int,
    quarter: str,
) -> None:
    """Full evaluation report."""
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    sep = "-" * 62

    print(f"\n{sep}")
    print(f"FULL COHORT EVALUATION   {milestone}k  {quarter} 2026")
    print(sep)
    print(f"Total vehicles   : {len(y):,}  ({n_pos:,} positives / {n_neg:,} negatives)")
    print(f"Base rate        : {100.0 * n_pos / max(len(y), 1):.1f}%")

    print(f"\nProbability by class:")
    print(f"  Positives  mean={probs[y == 1].mean():.4f}  median={np.median(probs[y == 1]):.4f}")
    print(f"  Negatives  mean={probs[y == 0].mean():.4f}  median={np.median(probs[y == 0]):.4f}")

    print(f"\nMetrics at threshold {threshold:.4f}:")
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

    print(f"\nThreshold sweep (Recall | Precision | F1 | Pred list):")
    print(f"  {'Thr':>5}  {'Recall':>7}  {'Precis':>7}  {'F1':>6}  {'List':>6}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
        p_pred = (probs >= thr).astype(int)
        cap    = int((p_pred[y == 1] == 1).sum())
        rec    = 100.0 * cap / max(n_pos, 1)
        prec   = 100.0 * precision_score(y, p_pred, zero_division=0)
        f1v    = 100.0 * f1_score(y, p_pred, zero_division=0)
        n_pred = int(p_pred.sum())
        mark   = " <--" if abs(thr - threshold) < 1e-4 else ""
        print(f"  {thr:>5.2f}  {rec:>6.1f}%  {prec:>6.1f}%  {f1v:>5.1f}%  {n_pred:>6,}{mark}")
    print(sep)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Full expected-cohort evaluation for any milestone and quarter"
    )
    p.add_argument("milestone", type=int,
                   help="Milestone integer, e.g. 30 for 30k")
    p.add_argument("--quarter", choices=["Q1", "Q2"], default="Q1",
                   help="Quarter to evaluate (default: Q1)")
    p.add_argument("--model-dir", default=None,
                   help="Model directory. Default: models/{m}k")
    p.add_argument("--features", default=None,
                   help="Feature list JSON. Default: models/selected_features_{m}k.json")
    p.add_argument("--threshold", type=float, default=None,
                   help="Score threshold. Default: threshold.json in model-dir, else 0.5")
    p.add_argument("--grace-days", type=int, default=0,
                   help="Days of grace after quarter end for labelling positives (default: 0)")
    p.add_argument("--out", default=None,
                   help="Output CSV path. Default: predictions/{m}k/eval_{m}k_{quarter}_expected_full.csv")
    p.add_argument("--no-early-drop", action="store_true",
                   help="Keep early completers in the evaluation (off by default)")
    args = p.parse_args()

    m = args.milestone
    quarter = args.quarter

    if quarter not in QUARTER_DATES:
        sys.exit(f"Unknown quarter {quarter!r}. Choose Q1 or Q2.")
    start_date, end_date, feature_cutoff = QUARTER_DATES[quarter]

    mdir = args.model_dir or model_dir(m)
    feat_file = args.features or features_path(m)

    for name, path in DATA.items():
        if not os.path.exists(path):
            sys.exit(f"Missing data file [{name}]: {path}")

    threshold = args.threshold
    if threshold is None:
        tp = os.path.join(mdir, "threshold.json")
        threshold = json.load(open(tp))["threshold"] if os.path.exists(tp) else 0.5

    print(f"Milestone : {m}k")
    print(f"Quarter   : {quarter}  ({start_date} to {end_date})")
    print(f"Cutoff    : {feature_cutoff}")
    print(f"Model dir : {mdir}")
    print(f"Features  : {feat_file}")
    print(f"Threshold : {threshold:.4f}")
    print(f"Grace days: {args.grace_days}")
    print(f"Drop early: {not args.no_early_drop}")

    # Step 1: Build labelled expected cohort
    print(f"\nStep 1: Building expected cohort ...")
    cohort_labels = create_candidates_test_cohort(
        eda_path=DATA["eda"],
        service_history_path=DATA["serv"],
        milestone=m,
        start_date=start_date,
        end_date=end_date,
        grace_days=args.grace_days,
    )

    # Step 2: Mask history
    print(f"\nStep 2: Masking service history ...")
    serv_masked = load_masked_history(DATA["serv"], m)

    # Step 3: Attach pipeline columns
    print(f"\nStep 3: Preparing cohort ...")
    cohort = build_eda_cohort(cohort_labels, m)

    # Step 3b: Drop early completers (default on, disable with --no-early-drop)
    if not args.no_early_drop:
        if "EarlyCompleter" in cohort.columns:
            n_early = int((cohort["EarlyCompleter"] == 1).sum())
            cohort = cohort[cohort["EarlyCompleter"] != 1].copy()
            print(f"  Dropped {n_early:,} early completers (finished before {start_date})")
        else:
            print("  WARNING: EarlyCompleter column not present -- skipping early drop")

    print(f"  Cohort: {len(cohort):,} vehicles  "
          f"(pos={cohort['TargetFlag'].sum():,}  neg={(cohort['TargetFlag'] == 0).sum():,})")

    # Step 4: Feature engineering
    print(f"\nStep 4: Feature engineering ...")
    features_df = run_feature_engineering(cohort, serv_masked, m, feature_cutoff)

    # Re-attach labels if process_service_data dropped them
    if "TargetFlag" not in features_df.columns and "VIN" in features_df.columns:
        label_map = cohort.set_index("VIN")["TargetFlag"].to_dict()
        features_df["TargetFlag"] = features_df["VIN"].map(label_map).fillna(0).astype(int)
        print(f"  TargetFlag re-attached; pos={features_df['TargetFlag'].sum():,}")

    # Step 5: Score
    print(f"\nStep 5: Scoring ...")
    out, y, probs, pred = score_cohort(features_df, m, mdir, feat_file, threshold)

    # Step 6: Save
    out_path = args.out or os.path.join(
        "predictions", f"{m}k", f"eval_{m}k_{quarter}_expected_full.csv"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out):,} rows -> {out_path}")

    if y is None or len(set(y)) < 2:
        print("No binary labels -- skipping metrics.")
        return

    print_metrics(y, probs, pred, threshold, m, quarter)


if __name__ == "__main__":
    main()
