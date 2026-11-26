"""Which list-building rule gives the best return? Compares every candidate rule against today's
method A on REAL outcomes, forward-built (no look-ahead). Pure due-date math, no model, strict
quarter, 20k-50k by default (pass e.g. 70,80 for others), stale-filtered lists (what a forward list really holds). Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe q1_due_rule_experiment/roi_check.py

Set-ups: Q1 forward (history cut 2025-12-31; actuals = first-ever Service_Num == m in Q1, in gate)
and Q3 forward (base history to 2026-06-29; actuals from Q3/Service_History_Q3-2026.csv).
"prev" = last-service+6mo / avg-interval only count when the last PMS is the step right below m.

Return = extra real visits caught vs A per extra name listed vs A ("marginal hit rate"), plus overall
precision / recall / F1.
"""
import sys

import pandas as pd

from build_unionstale_20k import load_base
from date_utils import parse_dates
from due_date import (burn_rate_date, last_completed_plus_months, avg_interval_date,
                      earliest_date)
from features import expected_milestone_months, resolve_service_num
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONES = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [20, 30, 40, 50]
SUFFIX = "" if len(sys.argv) == 1 else "_" + "_".join(f"{m}k" for m in MILESTONES)

df, serv = load_base()
vins = df["VIN"].values
gate = set(vins)
fs_all = df.set_index("VIN")["FirstSrvDate"].reindex(vins)

q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)

SETUPS = {
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
        prev = (below.groupby("Vin_No")["Service_Num"].max().reindex(vins) == m - 10).values
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
            "A min(schedule, burn) (today)": in_q(earliest_date(sched, burn), qs, qe),
            "B union(schedule, burn)": s | b,
            "3-leg union": s | b | l,
            "4-leg union": s | b | l | a,
            "D union(burn, avg)": b | a,
            "D' union(burn, last, avg)": b | l | a,
            "3-leg, prev-step fix": s | b | lp,
            "4-leg, prev-step fix": s | b | lp | ap,
            "D prev-step (burn, avg)": b | ap,
            "D' prev-step (burn, last, avg)": b | lp | ap,
        }
        base = rules["A min(schedule, burn) (today)"] - done_before
        base_hit = len(base & actual)
        for name, listed in rules.items():
            fresh = listed - done_before
            hit = len(fresh & actual)
            rec, prec = hit / len(actual), hit / len(fresh)
            d_list, d_hit = len(fresh) - len(base), hit - base_hit
            rows.append({"setup": setup, "milestone": f"{m}k", "rule": name, "actual": len(actual),
                         "listed": len(fresh), "caught": hit,
                         "recall_pct": round(100 * rec, 1), "precision_pct": round(100 * prec, 1),
                         "f1": round(2 * rec * prec / (rec + prec), 3),
                         "extra_caught_vs_A": d_hit, "extra_listed_vs_A": d_list,
                         "marginal_hit_rate_pct": round(100 * d_hit / d_list, 1) if d_list > 0 else None})
        print(f"{setup} {m}k done")

out = pd.DataFrame(rows)
out.to_csv(f"q1_due_rule_experiment/roi_check_2026{SUFFIX}.csv", index=False)
for setup in SETUPS:
    print(f"\n===== {setup} (stale-filtered) =====")
    print(out[out["setup"] == setup].drop(columns="setup").to_string(index=False))
print(f"\nwrote q1_due_rule_experiment/roi_check_2026{SUFFIX}.csv")
