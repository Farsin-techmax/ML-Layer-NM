"""Scoping check for the 4th due-date leg, due_date.avg_interval_date() -- same pattern as the
earlier last_completed_plus_months scoping check: pure due-date membership math, no model, no
build/retrain. Checks the new leg ALONE and folded into the union, against real Q1-2026 actual
completers, for 20k and 30k. Pad0 and +-45d. Reuses build_unionstale_20k.load_base() so the
population and due-date math match every other table in this investigation. Run from repo root.
"""
import pandas as pd

from build_unionstale_20k import load_base
from due_date import burn_rate_date, last_completed_plus_months, avg_interval_date
from features import expected_milestone_months
from pmstrainfeatureeng_refactored import quarter_bounds

PAD_DAYS = [0, 45]
MILESTONES = [20, 30, 40, 50]


def listed_set(due, q_start, q_end, pad_days):
    pad = pd.Timedelta(days=pad_days)
    m = due.notna() & (due >= q_start - pad) & (due <= q_end + pad)
    return set(due.index[m])


def row(method, pad_label, listed, overlap, actual_n):
    return {
        "method": method, "padding": pad_label, "listed": len(listed), "overlap": len(overlap),
        "overlap_pct_of_actual": round(100 * len(overlap) / actual_n, 1) if actual_n else None,
        "overlap_pct_of_listed": round(100 * len(overlap) / len(listed), 1) if listed else None,
    }


df, serv = load_base()
vins = df["VIN"].values
first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
cutoff = serv["Service_Date"].max()
serv_cols = serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]

all_rows = []
for milestone in MILESTONES:
    q_start, q_end = quarter_bounds(2026, 1)

    due_sched = first_srv + pd.DateOffset(months=expected_milestone_months(milestone, "schedule"))
    due_burn = burn_rate_date(vins, serv_cols, milestone, cutoff)
    due_last = last_completed_plus_months(vins, serv_cols, milestone, months=6)
    due_avg = avg_interval_date(vins, first_srv, serv_cols, milestone)

    at_ms = serv[serv["Service_Num"] == milestone]
    date_ms = at_ms.sort_values("Service_Date").groupby("Vin_No")["Service_Date"].first()
    actual = date_ms[(date_ms >= q_start) & (date_ms <= q_end)]
    actual = set(actual.index[actual.index.isin(df["VIN"])])
    actual_n = len(actual)
    print(f"\n=== {milestone}k Q1 2026 -- actual completers (in gate): {actual_n} ===")

    for pad in PAD_DAYS:
        pad_label = "none" if pad == 0 else f"+-{pad}d"

        l_sched = listed_set(due_sched, q_start, q_end, pad)
        l_burn = listed_set(due_burn, q_start, q_end, pad)
        l_last = listed_set(due_last, q_start, q_end, pad)
        l_avg = listed_set(due_avg, q_start, q_end, pad)

        methods = {
            "Schedule only": l_sched,
            "Burn-rate only": l_burn,
            "Last-service+6mo only": l_last,
            "Average-interval only (NEW)": l_avg,
            "Union (schedule+burn)": l_sched | l_burn,
            "Union (3-leg, +last-service)": l_sched | l_burn | l_last,
            "Union (4-leg, +avg-interval)": l_sched | l_burn | l_last | l_avg,
        }
        for name, listed in methods.items():
            overlap = listed & actual
            r = row(name, pad_label, listed, overlap, actual_n)
            r["milestone"] = f"{milestone}k"
            all_rows.append(r)
            star = " <-- NEW" if "NEW" in name else ""
            print(f"  [{pad_label:>5}] {name:<32} listed={len(listed):>5} overlap={len(overlap):>5} "
                  f"recall={r['overlap_pct_of_actual']:>5}% prec={r['overlap_pct_of_listed']:>5}%{star}")

out = pd.DataFrame(all_rows)[["milestone", "padding", "method", "listed", "overlap",
                               "overlap_pct_of_actual", "overlap_pct_of_listed"]]
out_path = "q1_due_rule_experiment/method4_avg_interval_all_milestones_Q1_2026.csv"
out.to_csv(out_path, index=False)
print(f"\nwrote {len(out)} rows -> {out_path}")
