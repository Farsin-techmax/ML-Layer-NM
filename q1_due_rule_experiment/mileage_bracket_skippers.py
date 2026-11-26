"""EXPERIMENT ONLY -- mileage-bracket assignment for vehicles that skipped services.
Pure due-date math, no model, strict quarter, 20k-50k. Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe q1_due_rule_experiment/mileage_bracket_skippers.py

Problem it targets: the last-service legs anchor on the vehicle's highest PMS so far, so a car whose
last PMS was its 10k lands on the 20k, 30k, 40k AND 50k lists at once.

Rule (user design, 2026-10-14):
  1. rate = latest odometer reading (any visit, PMS or non-PMS) / days since FirstSrvDate (>= 30 days)
  2. project that reading forward to quarter start and quarter end; take the MIDPOINT mileage
  3. bracket = nearest 10k: 20 = 15-25k, 30 = 25-35k, 40 = 35-45k, 50 = 45-55k, ...
  4. "skipper" = bracket is 2+ steps above the last PMS (last PMS 10k, now ~40k km -> missed 20k and
     30k). Service 1 counts as step 0. Skippers are listed ONLY on their bracket's milestone and are
     removed from the union everywhere else.
  5. everyone else (and any car with no usable rate) keeps the plain 4-leg union.

Set-ups: Q3 forward (dates from base history to 2026-06-29, outcomes from Q3 file) and Q1 forward
(history cut at 2025-12-31, outcomes = first-ever Service_Num == m in Q1, in gate).
"""
import numpy as np
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
serv["Mileage"] = serv["Mileage"].where(serv["Mileage"] <= 500_000)

q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)

SETUPS = {
    "Q3 forward": dict(q=(2026, 3), cutoff=serv["Service_Date"].max()),
    "Q1 forward": dict(q=(2026, 1), cutoff=pd.Timestamp("2025-12-31")),
}


def in_q(due, qs, qe):
    return set(due.index[due.notna() & (due >= qs) & (due <= qe)])


rows, skip_rows = [], []
for setup, cfg in SETUPS.items():
    qs, qe = quarter_bounds(*cfg["q"])
    q_mid = qs + (qe - qs) / 2
    cutoff = cfg["cutoff"]
    sc = serv.loc[serv["Service_Date"] <= cutoff, ["Vin_No", "Service_Date", "Mileage", "Service_Num"]]
    fs = fs_all.where(fs_all <= cutoff)

    # ---- vehicle-level bracket ----
    odo = sc[sc["Mileage"] > 0].sort_values("Service_Date").groupby("Vin_No").last()
    km = odo["Mileage"].reindex(vins)
    km_date = odo["Service_Date"].reindex(vins)
    age = (km_date - fs).dt.days
    rate = km / age.where(age >= 30)
    km_start = km + rate * (qs - km_date).dt.days
    km_end = km + rate * (qe - km_date).dt.days
    km_mid = (km_start + km_end) / 2
    bracket = (np.floor((km_mid / 1000 + 5) / 10) * 10).where(rate.notna())
    pms = sc[sc["Service_Num"] > 0]
    last_step = pms.groupby("Vin_No")["Service_Num"].max().reindex(vins).replace(1, 0)
    skipper = (bracket >= last_step.fillna(0) + 20) & bracket.notna()
    skip_set = set(skipper.index[skipper])
    print(f"\n{setup}: {len(skip_set):,} of {len(vins):,} gated vehicles are skippers "
          f"(bracket 2+ steps above last PMS); {int(rate.isna().sum()):,} have no usable rate")

    for m in MILESTONES:
        sched = fs + pd.DateOffset(months=expected_milestone_months(m, "schedule"))
        burn = burn_rate_date(vins, sc, m, cutoff)
        last = last_completed_plus_months(vins, sc, m, months=6)
        avg = avg_interval_date(vins, fs, sc, m)
        u4 = in_q(sched, qs, qe) | in_q(burn, qs, qe) | in_q(last, qs, qe) | in_q(avg, qs, qe)
        below = sc[(sc["Service_Num"] > 0) & (sc["Service_Num"] < m)]
        prev = (below.groupby("Vin_No")["Service_Num"].max().reindex(vins) == m - 10).values
        u4_prev = (in_q(sched, qs, qe) | in_q(burn, qs, qe) | in_q(last.where(prev), qs, qe)
                   | in_q(avg.where(prev), qs, qe))
        bracket_m = set(skipper.index[skipper & (bracket == m)])
        new = (u4 - skip_set) | bracket_m

        if cfg["q"] == (2026, 3):
            actual = set(q3.loc[(q3["Service_Num"] == m) & (q3["Service_Date"] >= qs)
                                & (q3["Service_Date"] <= qe), "Vin_No"])
        else:
            d = serv[serv["Service_Num"] == m].groupby("Vin_No")["Service_Date"].min()
            actual = set(d.index[(d >= qs) & (d <= qe)]) & gate
        done_before = set(serv.loc[(serv["Service_Num"] == m) & (serv["Service_Date"] < qs), "Vin_No"])

        rules = {
            "A min(schedule, burn) (today)": in_q(earliest_date(sched, burn), qs, qe),
            "4-leg union": u4,
            "4-leg, prev-step fix": u4_prev,
            "4-leg + mileage bracket for skippers (NEW)": new,
        }
        for name, listed in rules.items():
            fresh = listed - done_before
            rows.append({"setup": setup, "milestone": f"{m}k", "rule": name, "actual": len(actual),
                         "listed": len(listed), "overlap": len(listed & actual),
                         "recall_pct": round(100 * len(listed & actual) / len(actual), 1),
                         "precision_pct": round(100 * len(listed & actual) / len(listed), 1),
                         "listed_stale_filtered": len(fresh),
                         "overlap_stale_filtered": len(fresh & actual),
                         "recall_stale_filtered_pct": round(100 * len(fresh & actual) / len(actual), 1),
                         "precision_stale_filtered_pct": round(100 * len(fresh & actual) / len(fresh), 1)})

        # skippers who really came in for m: where did the bracket put them?
        sk_act = actual & skip_set
        put = bracket.reindex(list(sk_act)).value_counts().sort_index()
        skip_rows.append({"setup": setup, "milestone": f"{m}k",
                          "skippers_removed_from_union": len(u4 & skip_set),
                          "of_which_came_in": len(u4 & skip_set & actual),
                          "skippers_put_in_this_bracket": len(bracket_m),
                          "of_which_came_in_bracket": len(bracket_m & actual),
                          "real_visitors_who_are_skippers": len(sk_act),
                          "their_bracket": {int(k): int(v) for k, v in put.items()}})

out = pd.DataFrame(rows)
sk = pd.DataFrame(skip_rows)
out.to_csv("q1_due_rule_experiment/mileage_bracket_skippers_2026.csv", index=False)
sk.to_csv("q1_due_rule_experiment/mileage_bracket_skippers_detail_2026.csv", index=False)
for setup in SETUPS:
    print(f"\n===== {setup} (stale-filtered = what a forward list would really hold) =====")
    t = out[out["setup"] == setup]
    print(t[["milestone", "rule", "actual", "listed_stale_filtered", "overlap_stale_filtered",
             "recall_stale_filtered_pct", "precision_stale_filtered_pct"]].to_string(index=False))
    print(sk[sk["setup"] == setup].drop(columns="setup").to_string(index=False))
print("\nwrote q1_due_rule_experiment/mileage_bracket_skippers_2026.csv / _detail_2026.csv")
