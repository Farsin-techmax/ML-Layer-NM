"""july_candidates_overlap.py -- how well does the candidate list for a month match reality?

No model, no probabilities. Just the two sets and their intersection:

  CANDIDATES  vehicles whose due date for milestone m lands inside the window, built exactly the
              way create_candidates_test_cohort() builds cohort MEMBERSHIP (earliest of the
              6-months-per-10k schedule and the service-1 -> service-10 burn-rate projection,
              invoice-date guarded), with the projection cut off at --cutoff so nothing inside the
              window feeds the prediction of the window.
  ACTUAL      vehicles that actually completed milestone m inside the window, read from a service
              log that was NOT part of the base history (data/Service_History_Q2-2026.csv stops at
              2026-06-29; the July log is a separate extract).

Reported per milestone:
  recall     overlap / actual      -- of everyone who really turned up, how many did we list?
  precision  overlap / candidates  -- of everyone we listed, how many really turned up?

Two cohort variants are printed because "already did this milestone" is a real business filter:
  raw        every vehicle whose due date lands in the window
  open       raw minus vehicles that already completed m before the window opened

Run from the repo root: python july_candidates_overlap.py
"""
import argparse
import os

import numpy as np
import pandas as pd

from date_utils import parse_dates
from features import resolve_service_num
from due_date import burn_rate_date, earliest_date
from features import expected_milestone_months

MILESTONES = [20, 30, 40, 50, 60, 70, 80, 90, 100]


def load_base_history(path, cutoff):
    """Base service history, restricted to Service_Date <= cutoff.

    dayfirst=True is required: the raw values are '29-06-2026 00:00'. Parsed without it, pandas
    reads that as 2026-12-06 and the file appears to run six months longer than it does.
    """
    serv = pd.read_csv(path, low_memory=False, encoding="ISO-8859-1")
    serv["Service_Date"] = parse_dates(serv["Service_Date"])
    serv["Mileage"] = pd.to_numeric(serv["Mileage"], errors="coerce")
    serv["Service_Num"] = resolve_service_num(serv)
    serv = serv.dropna(subset=["Service_Date"])
    print(f"base history: {len(serv):,} dated rows, {serv['Service_Date'].min().date()} .. "
          f"{serv['Service_Date'].max().date()}")
    cut = serv[serv["Service_Date"] <= pd.Timestamp(cutoff)]
    print(f"cut at {cutoff}: {len(cut):,} rows kept")
    return cut


def load_actuals(path):
    """The held-out month's log. Same schema as the base history, read on its own terms."""
    act = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else \
        pd.read_csv(path, low_memory=False, encoding="ISO-8859-1")
    act["Service_Date"] = parse_dates(act["Service_Date"])
    act["Service_Num"] = resolve_service_num(act)
    act = act.dropna(subset=["Service_Date"])
    print(f"actuals: {len(act):,} rows, {act['Service_Date'].min().date()} .. "
          f"{act['Service_Date'].max().date()}, {act['Vin_No'].nunique():,} VINs")
    return act


def load_eda(path, relax_guard=False):
    df = pd.read_csv(path, low_memory=False, encoding="ISO-8859-1")
    df = df.query("`Last Service - PMS` != '-'").copy()
    df["Invoice date"] = parse_dates(df["Invoice date"])
    df["First Service Date"] = parse_dates(df["First Service Date"])
    df["FirstSrvDate"] = df["Invoice date"].fillna(df["First Service Date"])
    n_before = len(df)
    if relax_guard:
        # A/B arm: anchor on First Service Date when the sale date is missing, instead of
        # dropping the vehicle. Caveat: the schedule due date then measures from the first
        # VISIT, not the sale, so it lands late for these vehicles by the sale->first-service
        # lag. Reachability up, timing precision of the added vehicles unknown -- measure both.
        df = df[df["FirstSrvDate"].notna()].copy()
        print(f"EDA: {n_before:,} -> {len(df):,} vehicles, RELAXED guard "
              f"({(df['Invoice date'].isna()).sum():,} anchored on First Service Date)")
    else:
        df = df[df["Invoice date"].notna()].copy()
        print(f"EDA: {n_before:,} -> {len(df):,} vehicles after the invoice-date guard "
              f"({n_before - len(df):,} dropped, no genuine sale date)")
    return df


def candidates_for(df, serv_cut, milestone, start, end, cutoff):
    """Cohort membership only -- the due-date half of create_candidates_test_cohort()."""
    vins = df["VIN"].values
    first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
    months = expected_milestone_months(milestone, "schedule")
    due_sched = first_srv + pd.DateOffset(months=months)
    due_burn = burn_rate_date(vins, serv_cut[["Vin_No", "Service_Date", "Mileage", "Service_Num"]],
                              milestone, cutoff)
    due = earliest_date(due_sched, due_burn)
    source = np.where(due_burn.notna().to_numpy() & (due_burn.to_numpy() <= due_sched.to_numpy()),
                      "burn-rate", "schedule")

    out = pd.DataFrame({
        "VIN": vins,
        "DueDate": due.values,
        "DueSource": source,
        "Status": df["Vehicle Service Status"].values,
    })
    in_window = (out["DueDate"] >= pd.Timestamp(start)) & (out["DueDate"] <= pd.Timestamp(end))
    eligible = out["Status"].isin(["InActive", "Lapsed", "Active"])
    return out[in_window & eligible].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-start", default="2026-07-01")
    ap.add_argument("--window-end", default="2026-07-31")
    ap.add_argument("--cutoff", default="2026-06-30",
                    help="Nothing on or after this date may inform the due-date projection.")
    ap.add_argument("--actuals", default=os.path.join("data", "Service_History_Jul_2026.xlsx"))
    ap.add_argument("--eda", default=os.path.join("data", "EDA_Q2-2026.csv"))
    ap.add_argument("--history", default=os.path.join("data", "Service_History_Q2-2026.csv"))
    ap.add_argument("--milestones", default=",".join(str(m) for m in MILESTONES))
    ap.add_argument("--out-dir", default="candidates_jul2026")
    ap.add_argument("--relax-invoice-guard", action="store_true",
                    help="A/B arm: keep vehicles without a sale date, anchoring their schedule "
                         "due date on First Service Date instead of dropping them")
    args = ap.parse_args()

    milestones = [int(m) for m in args.milestones.split(",") if m.strip()]
    start, end = args.window_start, args.window_end
    print(f"--- Candidate vs actual overlap, {start} .. {end} (projection cutoff {args.cutoff}) ---")

    eda = load_eda(args.eda, relax_guard=args.relax_invoice_guard)
    serv_cut = load_base_history(args.history, args.cutoff)
    actuals = load_actuals(args.actuals)

    win_lo, win_hi = pd.Timestamp(start), pd.Timestamp(end)
    act_win = actuals[(actuals["Service_Date"] >= win_lo) & (actuals["Service_Date"] <= win_hi)]
    eda_vins = set(eda["VIN"])
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for m in milestones:
        cand = candidates_for(eda, serv_cut, m, start, end, args.cutoff)
        actual_vins = set(act_win.loc[act_win["Service_Num"] == m, "Vin_No"].unique())

        # "open" cohort: drop anyone who already did this milestone before the window opened.
        done_before = set(serv_cut.loc[serv_cut["Service_Num"] == m, "Vin_No"].unique())
        cand_open = cand[~cand["VIN"].isin(done_before)]

        cand_vins = set(cand["VIN"])
        open_vins = set(cand_open["VIN"])
        hit_raw = cand_vins & actual_vins
        hit_open = open_vins & actual_vins

        # Why the misses: an actual turn-up we could never have listed because the vehicle is not
        # in the EDA universe at all (or lost the invoice-date guard) is a different failure from
        # one we had but dated to the wrong month.
        reachable = actual_vins & eda_vins

        cand["ActualJuly"] = cand["VIN"].isin(actual_vins).astype(int)
        cand["AlreadyDone"] = cand["VIN"].isin(done_before).astype(int)
        cand.to_csv(os.path.join(args.out_dir, f"candidates_{m}k.csv"), index=False)

        rows.append({
            "milestone": m,
            "candidates": len(cand_vins),
            "cand_open": len(open_vins),
            "actual": len(actual_vins),
            "actual_in_eda": len(reachable),
            "overlap": len(hit_raw),
            "overlap_open": len(hit_open),
            "recall_%": 100 * len(hit_raw) / max(len(actual_vins), 1),
            "recall_open_%": 100 * len(hit_open) / max(len(actual_vins), 1),
            "precision_%": 100 * len(hit_raw) / max(len(cand_vins), 1),
            "precision_open_%": 100 * len(hit_open) / max(len(open_vins), 1),
        })
        print(f"{m:>4}k  cand {len(cand_vins):>6} (open {len(open_vins):>6})  "
              f"actual {len(actual_vins):>5}  overlap {len(hit_raw):>5} "
              f"(open {len(hit_open):>5})  recall {rows[-1]['recall_%']:5.1f}%  "
              f"precision {rows[-1]['precision_%']:5.1f}%")

    summary = pd.DataFrame(rows)
    out_path = os.path.join(args.out_dir, "overlap_summary.csv")
    summary.to_csv(out_path, index=False)
    print("\n" + summary.to_string(index=False,
                                   float_format=lambda v: f"{v:.1f}"))
    tot_c, tot_a = summary["candidates"].sum(), summary["actual"].sum()
    tot_o = summary["overlap"].sum()
    print(f"\nAll milestones: {tot_c:,} candidates, {tot_a:,} actual, {tot_o:,} overlap "
          f"-> recall {100*tot_o/max(tot_a,1):.1f}%, precision {100*tot_o/max(tot_c,1):.1f}%")
    print(f"Per-milestone candidate lists + summary written to {args.out_dir}/")


if __name__ == "__main__":
    main()
