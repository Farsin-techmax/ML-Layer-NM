"""How many vehicles lack a first-service (Service_Num==1) record, and does anchoring the burn rate
on their earliest PMS visit instead (burn_rate_date(lo_fallback=True)) catch more real Q1-2026
completers? Rules A/B/C as in missed_profile_by_rule.py, strict Q1, no padding. Run from repo root.
"""
import pandas as pd

from build_unionstale_20k import load_base
from due_date import burn_rate_date, last_completed_plus_months, avg_interval_date, earliest_date
from features import expected_milestone_months
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONES = [20, 30, 40, 50]
Q_START, Q_END = quarter_bounds(2026, 1)

df, serv = load_base()
vins = df["VIN"].values
vin_set = set(vins)
first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
cutoff = serv["Service_Date"].max()
serv_cols = serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]
has_svc1 = set(serv.loc[serv["Service_Num"] == 1, "Vin_No"]) & vin_set


def in_window(due):
    return set(due.index[due.notna() & (due >= Q_START) & (due <= Q_END)])


gap_rows, score_rows = [], []
for m in MILESTONES:
    has_hi = set(serv.loc[(serv["Service_Num"] > 0) & (serv["Service_Num"] < m), "Vin_No"]) & vin_set
    at_ms = serv[serv["Service_Num"] == m].sort_values("Service_Date")
    first_ms = at_ms.groupby("Vin_No")["Service_Date"].first()
    actual = set(first_ms[(first_ms >= Q_START) & (first_ms <= Q_END)].index)

    due_sched = first_srv + pd.DateOffset(months=expected_milestone_months(m, "schedule"))
    burn = burn_rate_date(vins, serv_cols, m, cutoff)
    burn_fb = burn_rate_date(vins, serv_cols, m, cutoff, lo_fallback=True)
    due_last = last_completed_plus_months(vins, serv_cols, m, months=6)
    due_avg = avg_interval_date(vins, first_srv, serv_cols, m)

    no_svc1 = has_hi - has_svc1
    newly_dated = set(burn_fb.dropna().index) - set(burn.dropna().index)
    gap_rows.append({
        "milestone": f"{m}k",
        "vehicles_with_a_PMS_below_target": len(has_hi),
        "no_first_service_record": len(no_svc1),
        "no_first_service_pct": round(100 * len(no_svc1) / len(has_hi), 1),
        "burn_rate_dated_before": int(burn.notna().sum()),
        "newly_dated_by_fallback": len(newly_dated),
        "actual_completers_no_first_service": len(no_svc1 & actual),
    })

    for label, b in [("current", burn), ("fallback", burn_fb)]:
        arms = {
            "burn-rate alone": in_window(b),
            "A: min(schedule, burn)": in_window(earliest_date(due_sched, b)),
            "B: union(schedule, burn)": in_window(due_sched) | in_window(b),
            "C: 4-method union": in_window(due_sched) | in_window(b) | in_window(due_last)
                                 | in_window(due_avg),
        }
        for arm, listed in arms.items():
            ov = listed & actual
            score_rows.append({"milestone": f"{m}k", "arm": arm, "burn_rate": label,
                               "listed": len(listed), "caught": len(ov),
                               "recall_pct": round(100 * len(ov) / len(actual), 1),
                               "precision_pct": round(100 * len(ov) / len(listed), 1)})

pd.set_option("display.width", 200)
g = pd.DataFrame(gap_rows)
print(g.to_string(index=False))
s = pd.DataFrame(score_rows)
w = s.pivot_table(index=["milestone", "arm"], columns="burn_rate",
                  values=["listed", "caught", "recall_pct", "precision_pct"], aggfunc="first")
print(w.to_string())
g.to_csv("q1_due_rule_experiment/burn_rate_lo_fallback_gap_counts.csv", index=False)
s.to_csv("q1_due_rule_experiment/burn_rate_lo_fallback_Q1_2026.csv", index=False)
