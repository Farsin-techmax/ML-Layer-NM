"""
hindcast_actual_turnups.py — Retrospective recall evaluation on confirmed turn-ups.

Goal: Take real vehicles that turned up for a milestone in Q1 or Q2 2026, mask the
milestone's own service visit (Service_Num == milestone) from history, run feature
engineering and score them with the saved model.  Because every row is a known positive
(TargetFlag = 1), AUC is undefined — we report recall-at-threshold and the full
probability distribution instead.

Usage:
    python hindcast_actual_turnups.py 20 --quarter Q1
    python hindcast_actual_turnups.py 20 --quarter Q2
    python hindcast_actual_turnups.py 20 --quarter Q1 --model-dir models/20k_cand_wins_2023
    python hindcast_actual_turnups.py 20 --quarter both

Output:
    predictions/{m}k/hindcast_{m}k_{quarter}.csv   (VIN, prob_turnup, pred)
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

DEVICE = torch.device("cpu")

# Quarter -> (window_start, window_end, feature_cutoff)
# feature_cutoff is set to the last day BEFORE the quarter so no within-quarter
# service data leaks into the features — identical logic to prepare_test_set.py.
QUARTER_DATES = {
    "Q1": ("2026-01-01", "2026-03-31", "2025-12-31"),
    "Q2": ("2026-04-01", "2026-06-30", "2026-03-31"),
}

# Paths — must stay in step with prepare_test_set.py
DATA = {
    "eda":      os.path.join("data", "EDA_Q2-2026.csv"),
    "serv":     os.path.join("data", "Service_History_Q2-2026.csv"),
    "rfm":      os.path.join("data", "RFM_Q2-2026.csv"),
    "appoint":  os.path.join("data", "Appointments_Q2-2026.csv"),
    "digi":     os.path.join("data", "Digital_Sessions_Q2-2026.csv"),
    "vhc":      os.path.join("data", "VHC_Q2-2026.csv"),
    "servcode": os.path.join("data", "Service_Code_Desc.csv"),
}


def _quarter_bounds(quarter: str):
    """Return (window_start, window_end, feature_cutoff) for a quarter key."""
    if quarter not in QUARTER_DATES:
        sys.exit(f"Unknown quarter {quarter!r}. Choose from: {list(QUARTER_DATES)}")
    return QUARTER_DATES[quarter]


def find_actual_turnup_vins(
    serv_path: str, milestone: int, window_start: str, window_end: str
) -> pd.Index:
    """Return VINs that completed `milestone` inside [window_start, window_end].

    Reads the full service history once, filters by Service_Num == milestone and
    Service_Date in the quarter.  These are the confirmed positives.
    """
    print(f"Reading service history: {serv_path}")
    serv = pd.read_csv(serv_path, low_memory=False, encoding="ISO-8859-1")
    serv["Service_Date"] = parse_dates(serv["Service_Date"])
    serv["Service_Num"] = resolve_service_num(serv)

    start_dt = pd.to_datetime(window_start)
    end_dt   = pd.to_datetime(window_end)

    turnups = serv[
        (serv["Service_Num"] == milestone)
        & (serv["Service_Date"] >= start_dt)
        & (serv["Service_Date"] <= end_dt)
    ]
    vins = pd.Index(turnups["Vin_No"].unique())
    print(f"Found {len(vins):,} VINs that completed {milestone}k in {window_start} to {window_end}")
    return vins


def load_masked_service_history(serv_path: str, milestone: int) -> pd.DataFrame:
    """Read full service history and remove all rows for the target milestone.

    Masking Service_Num == milestone ensures the feature pipeline cannot see the
    target visit.  It leaves all earlier and later service rows intact so features
    like PMS_Count_Prior, Avg_Service_Interval_PMS, etc. are computed correctly.

    Returns a fully parsed DataFrame ready to be passed as `servhistory` directly
    to process_service_data() (which accepts a pre-loaded frame).
    """
    print(f"Loading + masking service history (dropping Service_Num == {milestone}) ...")
    serv = pd.read_csv(serv_path, low_memory=False, encoding="ISO-8859-1")
    n_before = len(serv)

    serv["Service_Date"] = parse_dates(serv["Service_Date"])
    serv["Service_Num"]  = resolve_service_num(serv)

    # The mask: drop only the exact milestone service visit
    serv = serv[serv["Service_Num"] != milestone].copy()
    n_dropped = n_before - len(serv)
    print(f"  Masked {n_dropped:,} rows (Service_Num == {milestone}); {len(serv):,} rows remain")
    return serv


def build_cohort_from_eda(
    eda_path: str,
    turnup_vins: pd.Index,
    milestone: int,
    window_start: str,
    window_end: str,
) -> pd.DataFrame:
    """Subset EDA mastersheet to confirmed turn-up VINs and attach required columns.

    Mirrors the setup in prepare_test_set.py:
    - Parses Invoice/First Service dates
    - Computes Expected{m}kDate via the schedule model (FirstSrvDate + expected months)
    - Sets TargetFlag = 1 for all rows (all are known positives)
    - Sets Service_Num from 'Last Service - PMS' (needed by process_service_data)
    """
    from features import expected_milestone_months

    print(f"Loading EDA mastersheet: {eda_path}")
    df = pd.read_csv(eda_path, low_memory=False, encoding="ISO-8859-1")

    # Filter to turn-up VINs only
    df = df[df["VIN"].isin(turnup_vins)].copy()
    print(f"EDA rows matched to turn-up VINs: {len(df):,}")

    if len(df) == 0:
        sys.exit("No EDA rows matched the turn-up VINs. Check the VIN column name in EDA.")

    # Date parsing (day-first format in these files)
    df["Invoice date"]       = parse_dates(df["Invoice date"])
    df["First Service Date"] = parse_dates(df["First Service Date"])
    df["FirstSrvDate"]       = df["Invoice date"].fillna(df["First Service Date"])

    # Expected milestone date (schedule model, same as prepare_test_set.py default)
    months = expected_milestone_months(milestone, "schedule")
    df[f"Expected{milestone}kDate"] = df["FirstSrvDate"] + pd.DateOffset(months=months)

    # Drop EDA rows where the vehicle has no sale date
    missing_date = df["FirstSrvDate"].isna().sum()
    if missing_date:
        print(f"  Dropping {missing_date} rows with no FirstSrvDate")
        df = df.dropna(subset=["FirstSrvDate"])

    # All are confirmed positives
    df["TargetFlag"] = 1

    # Service_Num from EDA 'Last Service - PMS' — needed by process_service_data
    if "Last Service - PMS" in df.columns:
        df["Service_Num"] = df["Last Service - PMS"].apply(extract_k1)
    else:
        df["Service_Num"] = 0

    return df


def run_features(
    cohort: pd.DataFrame,
    serv_masked: pd.DataFrame,
    milestone: int,
    feature_cutoff: str,
) -> pd.DataFrame:
    """Run feature engineering on the masked cohort.

    Passes the pre-loaded masked service history directly (no re-read).
    feature_cutoff further trims history to before the quarter — belt-and-suspenders
    on top of the Service_Num mask.
    """
    print(f"\nRunning feature engineering (cutoff: {feature_cutoff}) ...")
    features_df = process_service_data(
        mastersheet=cohort,
        servhistory=serv_masked,   # pre-loaded DataFrame, already masked
        rfm=DATA["rfm"],
        servcode=DATA["servcode"],
        appointdf=DATA["appoint"],
        digidf=DATA["digi"],
        vhc=DATA["vhc"],
        filter_date=feature_cutoff,
        last_service_code=milestone,
        is_test=True,
    )
    return features_df


def score_and_report(
    features_df: pd.DataFrame,
    milestone: int,
    quarter: str,
    mdir: Optional[str],
    threshold: Optional[float],
    features_file: Optional[str],
    out_path: str,
) -> None:
    """Load model artifacts, score the cohort, and report recall statistics."""
    mdir = mdir or model_dir(milestone)
    for f in ["best_model.pt", "imputer.joblib", "scaler.joblib"]:
        if not os.path.exists(os.path.join(mdir, f)):
            sys.exit(
                f"Missing {os.path.join(mdir, f)} -- run: python retrain.py {milestone}"
            )

    feat_path = features_file or features_path(milestone)
    if not os.path.exists(feat_path):
        sys.exit(f"Missing {feat_path} -- run: python retrain.py {milestone}")

    # Resolve threshold from threshold.json if not overridden
    if threshold is None:
        tp = os.path.join(mdir, "threshold.json")
        threshold = json.load(open(tp))["threshold"] if os.path.exists(tp) else 0.5
    print(f"\nModel dir : {mdir}")
    print(f"Threshold : {threshold:.4f}")

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
    print(f"\n{milestone}k | rows={len(X)} | features={len(selected_features)}")

    if missing:
        expected, gaps = split_missing(missing, features_df.columns)
        if expected:
            print(
                f"  note: {len(expected)} absent one-hot categories zero-filled "
                f"(expected, 0 is correct)"
            )
        if gaps:
            print(
                f"  WARNING: {len(gaps)} features genuinely missing: "
                f"{gaps[:15]}{'...' if len(gaps) > 15 else ''}"
            )

    probs = predict_proba(model, imputer, scaler, X, DEVICE)
    pred  = (probs > threshold).astype(int)

    # --- Save scored output ---
    out = pd.DataFrame({"VIN": vin.values, "prob_turnup": probs, "pred": pred})
    if y is not None:
        out["TargetFlag"] = y
    out = out.sort_values("prob_turnup", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} scored rows -> {out_path}")

    # --- Recall-focused report (all-positive cohort, AUC undefined) ---
    n_total = len(probs)
    sep = "-" * 55
    print(f"\n{sep}")
    print(f"HINDCAST RECALL REPORT   {milestone}k  {quarter}")
    print(sep)
    print(f"Total confirmed positives (actual turn-ups): {n_total:,}")
    print(f"Probability distribution:")
    print(f"  mean   {probs.mean():.4f}")
    print(f"  median {np.median(probs):.4f}")
    print(f"  p25    {np.percentile(probs, 25):.4f}")
    print(f"  p75    {np.percentile(probs, 75):.4f}")
    print(f"  min    {probs.min():.4f}")
    print(f"  max    {probs.max():.4f}")

    print(f"\nRecall at thresholds (fraction of confirmed positives captured):")
    print(f"  {'Threshold':>10}  {'Captured':>8}  {'Recall %':>9}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        captured = int((probs >= thr).sum())
        recall_pct = 100.0 * captured / max(n_total, 1)
        marker = " <-- model threshold" if abs(thr - threshold) < 1e-4 else ""
        print(f"  {thr:>10.2f}  {captured:>8,}  {recall_pct:>8.1f}%{marker}")

    # Print model threshold explicitly if not in the fixed list above
    if not any(abs(threshold - t) < 1e-4 for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        captured = int((probs >= threshold).sum())
        recall_pct = 100.0 * captured / max(n_total, 1)
        print(
            f"\n  Model threshold ({threshold:.4f}): captured {captured:,}  "
            f"recall {recall_pct:.1f}%"
        )
    print(sep)


def run_quarter(
    milestone: int,
    quarter: str,
    mdir: Optional[str],
    threshold: Optional[float],
    features_file: Optional[str],
) -> None:
    """Full pipeline for one quarter."""
    window_start, window_end, feature_cutoff = _quarter_bounds(quarter)
    sep2 = "=" * 60
    print(f"\n{sep2}")
    print(f"  Hindcast: {milestone}k  {quarter}  ({window_start} to {window_end})")
    print(f"  Feature cutoff: {feature_cutoff}")
    print(sep2)

    # 1. Find confirmed turn-up VINs from service history
    turnup_vins = find_actual_turnup_vins(DATA["serv"], milestone, window_start, window_end)
    if len(turnup_vins) == 0:
        print(f"No turn-ups found for {milestone}k in {quarter} — skipping.")
        return

    # 2. Load masked service history (milestone rows removed)
    serv_masked = load_masked_service_history(DATA["serv"], milestone)

    # 3. Build EDA cohort from confirmed VINs
    cohort = build_cohort_from_eda(
        DATA["eda"], turnup_vins, milestone, window_start, window_end
    )

    # 4. Run feature engineering
    features_df = run_features(cohort, serv_masked, milestone, feature_cutoff)

    # 5. Score and report
    out_path = os.path.join(
        "predictions", f"{milestone}k", f"hindcast_{milestone}k_{quarter}.csv"
    )
    score_and_report(
        features_df, milestone, quarter, mdir, threshold, features_file, out_path
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Hindcast recall evaluation on confirmed Q1/Q2 turn-ups"
    )
    p.add_argument("milestone", type=int, help="Milestone integer, e.g. 20 for 20k")
    p.add_argument(
        "--quarter",
        choices=["Q1", "Q2", "both"],
        default="Q1",
        help="Which quarter(s) to evaluate (default: Q1)",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="Override model dir (default: models/{m}k)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Score threshold (default: model-dir threshold.json, else 0.5)",
    )
    p.add_argument(
        "--features",
        default=None,
        help="Feature-list JSON override (needed for tagged model dirs)",
    )
    args = p.parse_args()

    # Validate data files exist before any heavy work
    for name, path in DATA.items():
        if not os.path.exists(path):
            sys.exit(f"Missing data file [{name}]: {path}")

    quarters = ["Q1", "Q2"] if args.quarter == "both" else [args.quarter]
    for q in quarters:
        run_quarter(
            milestone=args.milestone,
            quarter=q,
            mdir=args.model_dir,
            threshold=args.threshold,
            features_file=args.features,
        )


if __name__ == "__main__":
    main()
