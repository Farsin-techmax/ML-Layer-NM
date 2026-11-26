"""build_unionstale_20k.py -- exploratory retrain check: does TRAINING on a union-rule cohort,
with early completers dropped (the "stale filter" concept applied to training labels instead of a
post-hoc delivery filter), beat the production 20k model?

Confirmed with the user before writing this:
  - "stale filter" for training literally means dropping EarlyCompleter rows (a vehicle whose real
    completion predates the due-quarter it got bucketed into) -- NOT the usual deliberate
    candidates-mode policy of keeping them as positives. This run intentionally reverses that policy
    to test it, it does not change the policy project-wide.
  - "+-45 days" has no single window to pad here (training tiles all 40 quarters, 2016Q1-2025Q4), so
    it is implemented as a sliding band at every quarter boundary: a vehicle whose due date (on
    either leg) falls within 45 days of a quarter's start or end also becomes a candidate for that
    neighboring quarter, not just its own.
  - Quick single-seed check only -- not rollout-grade (no 3-seed comparison, no gate battery).

Two artifacts per call: a TRAIN matrix (`--mode train`, all 40 quarters) or a matched Q1/Q2 TEST
cohort (`--mode test`, one quarter, stale-filtered the same way prepare_test_set.py's candidates
path does). Both share due-leg computation and the union/pad assignment so train and test cannot
disagree about which VIN belongs to which quarter.

Run:
  python build_unionstale_20k.py --mode train --window-pad 0  --out refactored_test_dir/final_processed_20k_candidates_unionstale.csv
  python build_unionstale_20k.py --mode train --window-pad 45 --out refactored_test_dir/final_processed_20k_candidates_unionstale_pad45.csv
  python build_unionstale_20k.py --mode test --year 2026 --quarter Q1 --window-pad 0 --train-cutoff 2025-12-31 --out test_sets/test_20k_Q1_2026_cand_unionstale.csv
"""
import argparse
import json as _json
import os
import sys
import time

import numpy as np
import pandas as pd

from pmstrainfeatureeng_refactored import (
    process_service_data, extract_k1, _apply_first_srv_date, quarter_bounds,
)
from features import resolve_service_num, expected_milestone_months
from date_utils import parse_dates
from due_date import burn_rate_date, earliest_date, last_completed_plus_months

MILESTONE = 20
EDA_PATH = "data/EDA_Q2-2026.csv"
SERV_PATH = "data/Service_History_Q2-2026.csv"
RFM_PATH = "data/RFM_Q2-2026.csv"
APPT_PATH = "data/Appointments_Q2-2026.csv"
DIGI_PATH = "data/Digital_Sessions_Q2-2026.csv"
VHC_PATH = "data/VHC_Q2-2026.csv"
SERVCODE_PATH = "data/Service_Code_Desc.csv"


def load_base():
    df = pd.read_csv(EDA_PATH, low_memory=False, encoding="ISO-8859-1")
    df = df.query("`Last Service - PMS` != '-'").copy()
    df["Last Service Mileage"] = pd.to_numeric(df["Last Service Mileage"], errors="coerce")
    df["Last PMS Mileage"] = pd.to_numeric(df["Last PMS Mileage"], errors="coerce")
    for col in ["Invoice date", "First Service Date", "Last Service Date - PMS",
                "Last Service Date", "Next Service Date"]:
        df[col] = parse_dates(df[col])
    df["Service_Num"] = df["Last Service - PMS"].apply(extract_k1)

    serv = pd.read_csv(SERV_PATH, low_memory=False, encoding="ISO-8859-1")
    serv["Service_Date"] = parse_dates(serv["Service_Date"])
    serv["Mileage"] = pd.to_numeric(serv["Mileage"], errors="coerce")
    serv = serv.dropna(subset=["Service_Date"])
    serv["Service_Num"] = resolve_service_num(serv)

    df = _apply_first_srv_date(df, serv)
    return df, serv


def compute_due_legs(df, serv, milestone, date_model="schedule", include_last_milestone_leg=False):
    vins = df["VIN"].values
    first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
    months = expected_milestone_months(milestone, date_model)
    due_sched = first_srv + pd.DateOffset(months=months)
    cutoff = serv["Service_Date"].max()
    due_burn = burn_rate_date(vins, serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]],
                               milestone, cutoff)
    due_last = None
    if include_last_milestone_leg:
        due_last = last_completed_plus_months(
            vins, serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]], milestone, months=6)
    return due_sched, due_burn, due_last


def quarters_in_range(from_y, from_q, to_y, to_q):
    out, y, q = [], from_y, from_q
    while (y, q) <= (to_y, to_q):
        out.append((y, q))
        q += 1
        if q == 5:
            q, y = 1, y + 1
    return out


def assign_candidates(due, quarters, pad_days, leg_name):
    """Long-format (VIN, CohortQuarter, Leg) for every quarter a due-date qualifies for, brute-force
    over all quarters -- simplest correct way to let padding create overlap with a neighboring
    quarter without hand-deriving which boundary a date is close to."""
    due_valid = due.dropna()
    vin_chunks, q_chunks = [], []
    pad = pd.Timedelta(days=pad_days)
    for (y, q) in quarters:
        q_start, q_end = quarter_bounds(y, q)
        mask = (due_valid >= q_start - pad) & (due_valid <= q_end + pad)
        matched = due_valid.index[mask]
        if len(matched):
            vin_chunks.append(matched.to_numpy())
            q_chunks.append(np.full(len(matched), f"{y}Q{q}"))
    if not vin_chunks:
        return pd.DataFrame(columns=["VIN", "CohortQuarter", "Leg"])
    return pd.DataFrame({
        "VIN": np.concatenate(vin_chunks),
        "CohortQuarter": np.concatenate(q_chunks),
        "Leg": leg_name,
    })


def label_candidates(candidates_long, serv, milestone, grace_days, near_grace_days=0):
    """EarlyCompleter marks a completion more than `near_grace_days` before the quarter opens --
    i.e. genuinely stale. A completion WITHIN near_grace_days before the quarter start (e.g. a
    vehicle that finished 20 days early) is NOT flagged EarlyCompleter and survives as an ordinary
    positive (TargetFlag was already 1 for it under the existing forward formula below) -- the
    confirmed middle ground between "drop every early completer" (near_grace_days=0, the previous
    retrain) and "keep every early completer regardless of how early" (production's convention).
    """
    ms_col = f"Actual{milestone}kDate"
    at_ms = serv[serv["Service_Num"] == milestone]
    date_ms = at_ms.sort_values("Service_Date").groupby("Vin_No")["Service_Date"].first()

    quarters = sorted(candidates_long["CohortQuarter"].unique())
    bounds = {cq: quarter_bounds(int(cq[:4]), int(cq[5])) for cq in quarters}

    out = candidates_long.copy()
    out[ms_col] = out["VIN"].map(date_ms)
    out["QuarterStart"] = out["CohortQuarter"].map(lambda cq: bounds[cq][0])
    out["QuarterEnd"] = out["CohortQuarter"].map(lambda cq: bounds[cq][1])

    grace = pd.Timedelta(days=grace_days)
    near_grace = pd.Timedelta(days=near_grace_days)
    ms = out[ms_col]
    out["TargetFlag"] = (ms.notna() & (ms <= out["QuarterEnd"] + grace)).astype(int)
    out["EarlyCompleter"] = (ms.notna() & (ms < out["QuarterStart"] - near_grace)).astype(int)
    return out


def build_long_cohort(df, serv, milestone, quarters, due_rule, pad_days, grace_days,
                       near_grace_days=0, include_last_milestone_leg=False):
    due_sched, due_burn, due_last = compute_due_legs(
        df, serv, milestone, include_last_milestone_leg=include_last_milestone_leg)
    if due_rule == "earliest":
        legs = [due_sched, due_burn] + ([due_last] if include_last_milestone_leg else [])
        due = earliest_date(*legs)
        long_c = assign_candidates(due, quarters, pad_days, "earliest")
    elif due_rule == "union":
        parts = [assign_candidates(due_sched, quarters, pad_days, "schedule"),
                 assign_candidates(due_burn, quarters, pad_days, "burn-rate")]
        if include_last_milestone_leg:
            parts.append(assign_candidates(due_last, quarters, pad_days, "last-milestone"))
        long_c = pd.concat(parts, ignore_index=True)
    else:
        raise ValueError(due_rule)

    labelled = label_candidates(long_c, serv, milestone, grace_days, near_grace_days)

    n_before = len(labelled)
    n_early = int(labelled["EarlyCompleter"].sum())
    labelled = labelled[labelled["EarlyCompleter"] == 0].copy()
    print(f"Dropped {n_early}/{n_before} EarlyCompleter rows ({100*n_early/max(n_before,1):.1f}%) "
          f"-- more than {near_grace_days}d before the quarter opened; completions within "
          f"{near_grace_days}d of the boundary were kept as ordinary positives")

    labelled = labelled.drop_duplicates(subset=["VIN", "CohortQuarter"])

    # process_service_data() requires an Expected{m}kDate column to exist (it parses it
    # unconditionally at line ~750); under is_test=True it never filters on it, so this is pure
    # bookkeeping to satisfy that requirement -- not used to decide membership, which union/pad
    # already settled above. Representative value = earliest(schedule, burn[, last-milestone]), same
    # convention the single-window test-side experiment used for this column.
    bk_legs = [due_sched, due_burn] + ([due_last] if include_last_milestone_leg else [])
    due = earliest_date(*bk_legs)
    exp_col = f"Expected{milestone}kDate"
    labelled[exp_col] = labelled["VIN"].map(due)

    return labelled[["VIN", "CohortQuarter", "QuarterEnd", f"Actual{milestone}kDate", "TargetFlag",
                      exp_col]]


def process_per_quarter(df, serv, labelled, milestone, currentstate_cols):
    ms_col = f"Actual{milestone}kDate"
    bookkeeping_cols = ["CohortQuarter", "QuarterEnd", ms_col]

    full = df.merge(labelled, on="VIN", how="inner")
    quarters = sorted(full["CohortQuarter"].unique())
    print(f"\n{len(quarters)} quarters present, {quarters[0]} .. {quarters[-1]}\n")

    parts, t0 = [], time.time()
    for cq in quarters:
        sub = full[full["CohortQuarter"] == cq]
        if len(sub) < 5:
            print(f"    {cq} skipped -- only {len(sub)} rows")
            continue
        y, q = int(cq[:4]), int(cq[5])
        q_start, _ = quarter_bounds(y, q)
        bk = sub[["VIN"] + bookkeeping_cols].set_index("VIN")
        feats = process_service_data(
            mastersheet=sub.drop(columns=bookkeeping_cols, errors="ignore"),
            servhistory=serv, rfm=RFM_PATH, servcode=SERVCODE_PATH,
            appointdf=APPT_PATH, digidf=DIGI_PATH, vhc=VHC_PATH,
            filter_date=q_start.strftime("%Y-%m-%d"),
            last_service_code=milestone, is_test=True,
        )
        feats = feats.drop(columns=bookkeeping_cols, errors="ignore").merge(
            bk, left_on="VIN", right_index=True, how="left")
        if currentstate_cols:
            feats = feats.drop(columns=[c for c in currentstate_cols if c in feats.columns],
                               errors="ignore")
        parts.append(feats)
        print(f"    {cq} -> {feats.shape[0]:,} rows x {feats.shape[1]} cols "
              f"({time.time()-t0:.0f}s elapsed)")

    if not parts:
        sys.exit("no cohorts produced")

    union_cols = list(dict.fromkeys(c for p in parts for c in p.columns))
    for i, p in enumerate(parts):
        parts[i] = p.reindex(columns=union_cols, fill_value=0)

    out = pd.concat(parts, axis=0, ignore_index=True, sort=False)
    dupes = out["VIN"].duplicated().sum()
    if dupes:
        print(f"WARNING: {dupes} duplicate VINs across cohorts -- keeping the earliest")
        out = out.drop_duplicates(subset="VIN", keep="first")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", type=int, default=20)
    ap.add_argument("--mode", choices=["train", "test"], required=True)
    ap.add_argument("--due-rule", choices=["earliest", "union"], default="union")
    ap.add_argument("--window-pad", type=int, default=0)
    ap.add_argument("--grace-days", type=int, default=15)
    ap.add_argument("--near-grace-days", type=int, default=0,
                     help="Keep an EarlyCompleter as an ordinary positive if it completed within "
                          "this many days BEFORE the quarter opened, instead of dropping it. "
                          "0 (default) reproduces the previous retrain's drop-all-early behaviour.")
    ap.add_argument("--include-last-milestone-leg", action="store_true",
                     help="Add due_date.last_completed_plus_months() (last completed milestone "
                          "below the target + 6 months, flat) as a third due-date leg. Default off "
                          "-- reproduces the earlier two-leg (schedule, burn-rate) results exactly.")
    ap.add_argument("--quarters-from", default="2016Q1")
    ap.add_argument("--quarters-to", default="2025Q4")
    ap.add_argument("--year", type=int, default=2026, help="test mode only")
    ap.add_argument("--quarter", default="Q1", help="test mode only, e.g. Q1 or Q2")
    ap.add_argument("--train-cutoff", default="2025-12-31", help="test mode only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df, serv = load_base()
    print(f"EDA rows {len(df):,} | service history rows {len(serv):,}")

    if args.mode == "train":
        fy, fq = int(args.quarters_from[:4]), int(args.quarters_from[5])
        ty, tq = int(args.quarters_to[:4]), int(args.quarters_to[5])
        quarters = quarters_in_range(fy, fq, ty, tq)
        feature_filter_date = None  # per-quarter cutoff used instead, see process_per_quarter
    else:
        q_num = int(args.quarter[1])
        quarters = [(args.year, q_num)]

    labelled = build_long_cohort(df, serv, args.milestone, quarters, args.due_rule, args.window_pad,
                                  args.grace_days, args.near_grace_days,
                                  args.include_last_milestone_leg)

    currentstate_path = "refactored_test_dir/eda_currentstate_features.json"
    currentstate_cols = _json.load(open(currentstate_path)) if os.path.exists(currentstate_path) else []

    if args.mode == "train":
        out = process_per_quarter(df, serv, labelled, args.milestone, currentstate_cols)
    else:
        # Test mode: ONE quarter, feature cutoff = train-cutoff (matches prepare_test_set.py's
        # candidates path -- NOT the quarter start, since this is a held-out eval window, not a
        # forward live quarter), and stale-filter applied the same way
        # stale_completion_filter.py does for forward lists: drop a VIN whose real completion
        # predates the window open (equivalent to EarlyCompleter here, already dropped above by
        # build_long_cohort, so this mode's cohort is ALREADY stale-filtered by construction).
        ms_col = f"Actual{args.milestone}kDate"
        bookkeeping_cols = ["CohortQuarter", "QuarterEnd", ms_col]
        full = df.merge(labelled, on="VIN", how="inner")
        bk = full[["VIN"] + bookkeeping_cols].set_index("VIN")
        full["Service_Num"] = full["Last Service - PMS"].apply(extract_k1)
        out = process_service_data(
            mastersheet=full.drop(columns=bookkeeping_cols, errors="ignore"),
            servhistory=serv, rfm=RFM_PATH, servcode=SERVCODE_PATH,
            appointdf=APPT_PATH, digidf=DIGI_PATH, vhc=VHC_PATH,
            filter_date=args.train_cutoff, last_service_code=args.milestone, is_test=True,
        )
        out = out.drop(columns=bookkeeping_cols, errors="ignore").merge(
            bk, left_on="VIN", right_index=True, how="left")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} rows x {out.shape[1]} cols -> {args.out}")


if __name__ == "__main__":
    main()
