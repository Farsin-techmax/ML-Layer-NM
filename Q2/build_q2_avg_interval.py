"""Q2 2026 expected candidate base -- average-service-interval leg ONLY (no union of legs).

Decided with the user 2026-10-21:
  - history cut at 2026-03-31, the day before Q2: the list you would have had on 1 April.
  - due date = due_date.avg_interval_date(..., include_first_srv_segment=False, scale_by_steps=True):
    the vehicle's own average gap between PMS visits from service 1 onward (1->10, 10->20, ...),
    multiplied by how many 10k-steps away the target milestone is, added to the date of its highest
    PMS step below the milestone. Invoice -> service 1 is NOT a segment.
  - `scale_by_steps=True` REPLACES the single-shot version adopted earlier the same day. Without it,
    a vehicle's due date was IDENTICAL for every milestone above its current step (hi=10 gave the
    same date for the 20k through the 100k list) -- 95.0% of the 8,341 listed VINs were duplicated
    this way, with copy-pasted, not staggered, dates. Scaling by distance fixed that: cross-milestone
    duplication fell to 5.8% of listed VINs, and every one of those remaining duplicates now carries
    a genuinely different due date per milestone -- i.e. it correctly keeps the small population of
    fast-pace vehicles that really can come due for two consecutive milestones in one quarter. See
    due_date.avg_interval_date()'s docstring for the full measurement.
  - NO prev-step gate: a vehicle can still be listed for a milestone several steps above its current
    one -- scale_by_steps makes that projection honest (correctly far out in time) rather than wrong
    (copy-pasted from the next step), it does not restrict membership to only the immediate next step.
  - strict quarter: listed when the due date falls in 2026-04-01 .. 2026-06-30.
  - EDA gate `Last Service - PMS != '-'`; cars that already did the milestone before 2026-04-01
    are dropped (stale filter).
  - outcome (evaluation only, never used to build the list): first-ever Service_Num == m visit
    inside Q2, read from service history (which runs to 2026-06-29).

Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe Q2/build_q2_avg_interval.py
    PYTHONPATH=. ./venv/Scripts/python.exe Q2/build_q2_avg_interval.py --eda data/EDA_Q2-2026.csv --out-dir <scratch>

Writes <out-dir>/lists/avg_interval_{m}k_Q2_2026.csv (one row per listed VIN) and
<out-dir>/summary_avg_interval_Q2_2026.csv.
"""
import argparse
import os
import sys

import pandas as pd

from date_utils import parse_dates
from due_date import avg_interval_date
from features import resolve_service_num
from pmstrainfeatureeng_refactored import _apply_first_srv_date, quarter_bounds

MILESTONES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
SERV_PATH = "data/Service_History_Q2-2026.csv"
CUTOFF = pd.Timestamp("2026-03-31")
Q_START, Q_END = quarter_bounds(2026, 2)


def load(eda_path):
    df = pd.read_csv(eda_path, low_memory=False, encoding="ISO-8859-1")
    df = df.query("`Last Service - PMS` != '-'").copy()
    for col in ["Invoice date", "First Service Date", "Last Service Date"]:
        df[col] = parse_dates(df[col])

    serv = pd.read_csv(SERV_PATH, low_memory=False, encoding="ISO-8859-1")
    serv["Service_Date"] = parse_dates(serv["Service_Date"])
    serv["Mileage"] = pd.to_numeric(serv["Mileage"], errors="coerce")
    serv = serv.dropna(subset=["Service_Date"])
    serv["Service_Num"] = resolve_service_num(serv)

    sc = serv.loc[serv["Service_Date"] <= CUTOFF, ["Vin_No", "Service_Date", "Mileage", "Service_Num"]]
    # Anchor from the cut history too, so nothing dated after 31 March is read to build the list.
    df = _apply_first_srv_date(df, sc)
    return df, serv, sc


def segment_detail(vins, sc, m):
    """Last PMS step below m, its date, how many usable segments (service 1 onward) it has, and the
    resulting per-vehicle AVERAGE service interval in months -- the exact same breakpoints and the
    exact same number due_date.avg_interval_date() averages over (include_first_srv_segment=False),
    reported here for transparency since that function itself only returns the final due date, not
    the per-step average that feeds it."""
    s = sc.dropna(subset=["Service_Date"]).sort_values("Service_Date", kind="mergesort")
    below = s[(s["Service_Num"] > 0) & (s["Service_Num"] < m)]
    hi_step = below.groupby("Vin_No")["Service_Num"].max().reindex(vins)
    hi_rows = below[below["Service_Num"] == below.groupby("Vin_No")["Service_Num"].transform("max")]
    hi_date = hi_rows.groupby("Vin_No")["Service_Date"].first().reindex(vins)

    bps = [s[s["Service_Num"] == sn].groupby("Vin_No")["Service_Date"].first().reindex(vins)
           for sn in [1] + list(range(10, m, 10))]
    DAYS_PER_MONTH = 30.4368  # matches due_date.avg_interval_date() exactly
    n_seg = pd.Series(0, index=vins)
    month_sum = pd.Series(0.0, index=vins)
    for a, b in zip(bps[:-1], bps[1:]):
        days = (b - a).dt.days.astype(float)
        ok = a.notna() & b.notna() & (days > 0)
        n_seg += ok.astype(int)
        month_sum[ok] = month_sum[ok] + days[ok] / DAYS_PER_MONTH

    avg_months = pd.Series(float("nan"), index=vins)
    has_seg = n_seg > 0
    avg_months[has_seg] = month_sum[has_seg] / n_seg[has_seg]
    return hi_step, hi_date, n_seg, avg_months


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda", default="data/EDA_Q1-2026.csv")
    ap.add_argument("--out-dir", default="Q2")
    args = ap.parse_args()
    if not os.path.exists(args.eda):
        sys.exit(f"EDA file not found: {args.eda}")

    df, serv, sc = load(args.eda)
    vins = df["VIN"].values
    first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
    key = df.set_index("VIN")["Vehicle Key"].reindex(vins)
    print(f"EDA {args.eda}: {len(vins):,} VINs in gate | history cut {CUTOFF.date()}: {len(sc):,} rows")

    os.makedirs(os.path.join(args.out_dir, "lists"), exist_ok=True)
    rows, listed_by_m = [], {}
    for m in MILESTONES:
        due = avg_interval_date(vins, first_srv, sc, m, include_first_srv_segment=False,
                                scale_by_steps=True)
        hi_step, hi_date, n_seg, avg_months = segment_detail(vins, sc, m)

        first_m = serv[serv["Service_Num"] == m].groupby("Vin_No")["Service_Date"].min()
        done_before = set(first_m.index[first_m < Q_START])
        visitors_all = set(first_m.index[(first_m >= Q_START) & (first_m <= Q_END)])

        in_q = due.notna() & (due >= Q_START) & (due <= Q_END)
        stale = pd.Series(vins, index=vins).isin(done_before)
        keep = in_q & ~stale

        steps_away = (m - hi_step[keep]) / 10.0
        out = pd.DataFrame({
            "VIN": vins[keep.values],
            "Vehicle_Key": key[keep].values,
            "last_pms_step": hi_step[keep].astype(int).values,
            "last_pms_date": hi_date[keep].dt.strftime("%Y-%m-%d").values,
            "n_segments": n_seg[keep].values,
            "steps_away": steps_away.values,
            "AvgServiceIntervalMonths": avg_months[keep].round(2).values,
            "avg_interval_days": (due[keep] - hi_date[keep]).dt.days.values,
            "NextServiceDate": due[keep].dt.strftime("%Y-%m-%d").values,
        })
        out[f"Actual{m}kDate"] = out["VIN"].map(first_m).dt.strftime("%Y-%m-%d")
        out["CameInQ2"] = out["VIN"].isin(visitors_all).astype(int)
        out = out.sort_values("NextServiceDate").reset_index(drop=True)
        out.to_csv(os.path.join(args.out_dir, "lists", f"avg_interval_{m}k_Q2_2026.csv"), index=False)
        listed_by_m[m] = set(out["VIN"])

        came = int(out["CameInQ2"].sum())
        rows.append({
            "milestone": f"{m}k",
            "has_last_pms_below_m": int(hi_step.notna().sum()),
            "no_usable_segment": int((hi_step.notna() & (n_seg == 0)).sum()),
            "due_in_Q2": int(in_q.sum()),
            "stale_dropped": int((in_q & stale).sum()),
            "listed": len(out),
            "came_in_from_list": came,
            "precision_pct": round(100 * came / len(out), 1) if len(out) else None,
            "all_Q2_visitors": len(visitors_all),
            "recall_of_all_pct": round(100 * came / len(visitors_all), 1) if visitors_all else None,
            "median_avg_interval_days": int(out["avg_interval_days"].median()) if len(out) else None,
        })
        print(f"{m}k: listed {len(out):,}, came in {came:,}")

    summary = pd.DataFrame(rows)
    counts = pd.Series([v for s in listed_by_m.values() for v in s]).value_counts()
    summary.to_csv(os.path.join(args.out_dir, "summary_avg_interval_Q2_2026.csv"), index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nVINs on 2+ milestone lists: {int((counts > 1).sum()):,} of {len(counts):,} distinct listed VINs")


if __name__ == "__main__":
    main()
