"""july_overlap_diagnose.py -- why the July candidate list misses what it misses.

Two questions the headline overlap number cannot answer:

  1. REACHABILITY. Before any date is computed, the cohort builder drops vehicles twice: EDA rows
     with `Last Service - PMS == '-'`, then rows with no parseable `Invoice date` (the invoice
     guard). A vehicle removed there can never be a candidate no matter how good the due date is,
     so it caps recall. This prints the funnel for the vehicles that actually turned up in July.

  2. TIMING. For the reachable ones, is the due date biased (systematically early/late) or just
     noisy? Prints the distribution of (predicted due date - actual service date) in months, and
     the recall that a wider window either side of July would have bought.

Run from the repo root: python july_overlap_diagnose.py
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
PADS = [0, 15, 30, 60, 90, 180]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-start", default="2026-07-01")
    ap.add_argument("--window-end", default="2026-07-31")
    ap.add_argument("--cutoff", default="2026-06-30")
    ap.add_argument("--actuals", default=os.path.join("data", "Service_History_Jul_2026.xlsx"))
    ap.add_argument("--eda", default=os.path.join("data", "EDA_Q2-2026.csv"))
    ap.add_argument("--history", default=os.path.join("data", "Service_History_Q2-2026.csv"))
    ap.add_argument("--out-dir", default="candidates_jul2026")
    args = ap.parse_args()

    raw = pd.read_csv(args.eda, low_memory=False, encoding="ISO-8859-1")
    all_eda_vins = set(raw["VIN"])
    step1 = raw.query("`Last Service - PMS` != '-'").copy()
    step1_vins = set(step1["VIN"])

    step1["Invoice date"] = parse_dates(step1["Invoice date"])
    step1["First Service Date"] = parse_dates(step1["First Service Date"])
    step1["FirstSrvDate"] = step1["Invoice date"].fillna(step1["First Service Date"])
    eda = step1[step1["Invoice date"].notna()].copy()
    guarded_vins = set(eda["VIN"])
    # EDA carries duplicate VIN rows; the cohort builder never dedupes (set() hides it downstream),
    # but a per-VIN due-date lookup needs a unique index.
    n_dup = len(eda) - eda["VIN"].nunique()
    if n_dup:
        print(f"EDA duplicate VIN rows: {n_dup:,} (keeping first per VIN)")
        eda = eda.drop_duplicates(subset="VIN", keep="first").copy()

    serv = pd.read_csv(args.history, low_memory=False, encoding="ISO-8859-1")
    serv["Service_Date"] = parse_dates(serv["Service_Date"])
    serv["Mileage"] = pd.to_numeric(serv["Mileage"], errors="coerce")
    serv["Service_Num"] = resolve_service_num(serv)
    serv_cut = serv.dropna(subset=["Service_Date"])
    serv_cut = serv_cut[serv_cut["Service_Date"] <= pd.Timestamp(args.cutoff)]

    act = pd.read_excel(args.actuals)
    act["Service_Date"] = parse_dates(act["Service_Date"])
    act["Service_Num"] = resolve_service_num(act)
    win_lo, win_hi = pd.Timestamp(args.window_start), pd.Timestamp(args.window_end)
    act = act[(act["Service_Date"] >= win_lo) & (act["Service_Date"] <= win_hi)]

    # Status filter is the third gate, applied per-cohort in the builder.
    status_ok = set(eda.loc[eda["Vehicle Service Status"].isin(["InActive", "Lapsed", "Active"]),
                            "VIN"])

    # Precompute every vehicle's due date once per milestone.
    vins = eda["VIN"].values
    first_srv = eda.set_index("VIN")["FirstSrvDate"].reindex(vins)
    serv_slim = serv_cut[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]

    funnel_rows, lag_rows, pad_rows = [], [], []
    for m in MILESTONES:
        a = act[act["Service_Num"] == m]
        actual_vins = set(a["Vin_No"].unique())
        n_act = len(actual_vins)

        in_eda = actual_vins & all_eda_vins
        in_step1 = actual_vins & step1_vins
        in_guard = actual_vins & guarded_vins
        in_status = actual_vins & status_ok

        funnel_rows.append({
            "milestone": m,
            "actual": n_act,
            "in_EDA": len(in_eda),
            "has_LastPMS": len(in_step1),
            "has_invoice": len(in_guard),
            "status_ok": len(in_status),
            "reachable_%": 100 * len(in_status) / max(n_act, 1),
        })

        months = expected_milestone_months(m, "schedule")
        due_sched = first_srv + pd.DateOffset(months=months)
        due_burn = burn_rate_date(vins, serv_slim, m, args.cutoff)
        due = earliest_date(due_sched, due_burn)
        due.index = vins

        # Actual date per VIN for the reachable turn-ups, vs their predicted due date.
        act_date = a.sort_values("Service_Date").groupby("Vin_No")["Service_Date"].first()
        reach = sorted(in_status)
        pred = due.reindex(reach)
        real = act_date.reindex(reach)
        lag_days = (pred - real).dt.days.astype("float64")
        lag_m = lag_days / 30.44

        lag_rows.append({
            "milestone": m,
            "reachable": len(reach),
            "no_due_date": int(pred.isna().sum()),
            "p10_mo": np.nanpercentile(lag_m, 10) if len(reach) else np.nan,
            "median_mo": np.nanmedian(lag_m) if len(reach) else np.nan,
            "p90_mo": np.nanpercentile(lag_m, 90) if len(reach) else np.nan,
            "burn_used_%": 100 * float((due_burn.notna().to_numpy() &
                                        (due_burn.to_numpy() <= due_sched.to_numpy())).mean()),
        })

        row = {"milestone": m, "actual": n_act}
        for pad in PADS:
            lo, hi = win_lo - pd.Timedelta(days=pad), win_hi + pd.Timedelta(days=pad)
            hit = int(((pred >= lo) & (pred <= hi)).sum())
            n_cand = int(((due >= lo) & (due <= hi) &
                          pd.Series(vins, index=vins).isin(status_ok)).sum())
            row[f"recall_+-{pad}d"] = 100 * hit / max(n_act, 1)
            row[f"cands_+-{pad}d"] = n_cand
        pad_rows.append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    fun = pd.DataFrame(funnel_rows)
    lag = pd.DataFrame(lag_rows)
    pad = pd.DataFrame(pad_rows)

    fmt = lambda v: f"{v:.1f}"
    print("\n=== 1. Reachability funnel (July turn-ups that the builder could ever list) ===")
    print(fun.to_string(index=False, float_format=fmt))
    print("\n=== 2. Due-date error for reachable turn-ups (predicted - actual, months; "
          "negative = we said too early) ===")
    print(lag.to_string(index=False, float_format=fmt))
    print("\n=== 3. Recall vs window width (candidate count in brackets) ===")
    cols = ["milestone", "actual"] + [f"recall_+-{p}d" for p in PADS]
    print(pad[cols].to_string(index=False, float_format=fmt))
    print()
    print(pad[["milestone"] + [f"cands_+-{p}d" for p in PADS]].to_string(index=False))

    fun.to_csv(os.path.join(args.out_dir, "diag_funnel.csv"), index=False)
    lag.to_csv(os.path.join(args.out_dir, "diag_duedate_error.csv"), index=False)
    pad.to_csv(os.path.join(args.out_dir, "diag_window_width.csv"), index=False)
    print(f"\nWritten to {args.out_dir}/diag_*.csv")


if __name__ == "__main__":
    main()
