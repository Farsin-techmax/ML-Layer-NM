"""Are the Q1 union tables optimistic? And does anchoring the last-service legs on the PREVIOUS step
fix the Q3 precision drop? Pure due-date math, no model, strict quarter, no padding, 20k-50k.
Run from repo root:  PYTHONPATH=. ./venv/Scripts/python.exe q1_due_rule_experiment/forward_cutoff_check.py

Three set-ups, same rules (A min(schedule, burn) | B union | 3-leg | 4-leg):
  Q1 as recorded   dates built from the FULL base history (to 2026-06-29, i.e. including visits after
                   Q1) -- how every Q1 table in CLAUDE.md was made
  Q1 forward       dates built from history cut at 2025-12-31, as a real Q1 list would be
  Q3 forward       dates from the base history (ends 2026-06-29), outcomes from the Q3 file
Actuals: Q1 = first-ever Service_Num == m dated in Q1, inside the EDA gate (as the Q1 tables);
Q3 = any Service_Num == m record in the Q3 file (as analysis/reconcile_quarter.py / q3_union_rules.py).

Variant "prev-step": last-service+6mo and avg-interval only count when the vehicle's highest completed
milestone is exactly m-10 (a car whose last PMS was its 10k is due for 20k, not for 40k or 50k).
"Stale-filtered" precision drops vehicles that already did m before the window opened.
"""
import pandas as pd

from build_unionstale_20k import load_base
from date_utils import parse_dates
from due_date import (burn_rate_date, last_completed_plus_months, avg_interval_date,
                      earliest_date)
from features import expected_milestone_months, resolve_service_num
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONES = [20, 30, 40, 50]

df, serv = load_base()
vins = df["VIN"].values
gate = set(vins)
fs_all = df.set_index("VIN")["FirstSrvDate"].reindex(vins)

q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)

SETUPS = {
    "Q1 as recorded": dict(q=(2026, 1), cutoff=serv["Service_Date"].max()),
    "Q1 forward": dict(q=(2026, 1), cutoff=pd.Timestamp("2025-12-31")),
    "Q3 forward": dict(q=(2026, 3), cutoff=serv["Service_Date"].max()),
}


def in_q(due, qs, qe):
    return set(due.index[due.notna() & (due >= qs) & (due <= qe)])


rows = []
for setup, cfg in SETUPS.items():
    qs, qe = quarter_bounds(*cfg["q"])
    cutoff = cfg["cutoff"]
    sc = serv.loc[serv["Service_Date"] <= cutoff, ["Vin_No", "Service_Date", "Mileage", "Service_Num"]]
    fs = fs_all.where(fs_all <= cutoff)
    for m in MILESTONES:
        sched = fs + pd.DateOffset(months=expected_milestone_months(m, "schedule"))
        burn = burn_rate_date(vins, sc, m, cutoff)
        last = last_completed_plus_months(vins, sc, m, months=6)
        avg = avg_interval_date(vins, fs, sc, m)
        below = sc[(sc["Service_Num"] > 0) & (sc["Service_Num"] < m)]
        hi_num = below.groupby("Vin_No")["Service_Num"].max().reindex(vins)
        prev = (hi_num == m - 10).values
        s, b = in_q(sched, qs, qe), in_q(burn, qs, qe)
        l, a = in_q(last, qs, qe), in_q(avg, qs, qe)
        lp, ap = in_q(last.where(prev), qs, qe), in_q(avg.where(prev), qs, qe)

        if cfg["q"] == (2026, 3):
            actual = set(q3.loc[(q3["Service_Num"] == m) & (q3["Service_Date"] >= qs)
                                & (q3["Service_Date"] <= qe), "Vin_No"])
        else:
            d = serv[serv["Service_Num"] == m].groupby("Vin_No")["Service_Date"].min()
            actual = set(d.index[(d >= qs) & (d <= qe)]) & gate
        done_before = set(serv.loc[(serv["Service_Num"] == m) & (serv["Service_Date"] < qs), "Vin_No"])

        rules = {
            "A min(schedule, burn)": in_q(earliest_date(sched, burn), qs, qe),
            "B union(schedule, burn)": s | b,
            "3-leg union": s | b | l,
            "4-leg union": s | b | l | a,
            "3-leg, prev-step anchor": s | b | lp,
            "4-leg, prev-step anchor": s | b | lp | ap,
        }
        for name, listed in rules.items():
            hit = listed & actual
            fresh = listed - done_before
            rows.append({"setup": setup, "milestone": f"{m}k", "rule": name, "actual": len(actual),
                         "listed": len(listed), "overlap": len(hit),
                         "recall_pct": round(100 * len(hit) / len(actual), 1),
                         "precision_pct": round(100 * len(hit) / len(listed), 1),
                         "listed_stale_filtered": len(fresh),
                         "overlap_stale_filtered": len(fresh & actual),
                         "recall_stale_filtered_pct": round(100 * len(fresh & actual) / len(actual), 1),
                         "precision_stale_filtered_pct":
                             round(100 * len(fresh & actual) / len(fresh), 1)})
        print(f"{setup} {m}k done")

out = pd.DataFrame(rows)
out.to_csv("q1_due_rule_experiment/forward_cutoff_check_2026.csv", index=False)
for m in MILESTONES:
    t = out[out["milestone"] == f"{m}k"].pivot(index="rule", columns="setup",
                                                 values=["listed", "recall_pct", "precision_pct"])
    print(f"\n=== {m}k ===")
    print(t.to_string())
print("\nwrote q1_due_rule_experiment/forward_cutoff_check_2026.csv")
