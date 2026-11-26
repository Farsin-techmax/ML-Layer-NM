"""EXPERIMENT ONLY -- variants on the mileage-bracket rescue idea for vehicles that skipped
services. Pure due-date math, no model, strict quarter, 20k-50k. Copied from
mileage_bracket_skippers.py (template) -- only the RULES differ, everything else (load_base(),
set-ups, actuals, stale filter, rate/midpoint computation) is identical. Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe q1_due_rule_experiment/mileage_bracket_variants.py

The template's rule failed two ways:
  1. the bracket is usually one step too high -- skippers who really came in for 40k were mostly
     projected at the 45-55k km midpoint (the NEAREST bracket's next band up).
  2. the cars bloating the lists are recent low-mileage cars whose last PMS is far below the
     target (e.g. a 10k done in March 2026 lands on the 20k, 30k, 40k AND 50k lists via the
     last-service legs), and those were not classed as skippers (skipper needs bracket >=
     last_step + 20, i.e. 2+ steps ahead -- a car 1 step ahead, prev==True, was never touched).

Two bracket functions from the midpoint mileage km_mid:
  nearest  = floor((km_mid/1000 + 5)/10)*10   -- the original: 40 = 35-45k km
  shifted  = nearest - 10                     -- 40 = 45-55k km (one step down, per finding 1)

Six rules per milestone:
  1. A min(schedule, burn) (today)
  2. 4-leg union
  3. 4-leg, prev-step fix
  4. V1 shifted bracket for skippers        -- template's rule, shifted bracket for BOTH the
                                                skipper test and the assignment
  5. V2 prev-step fix + bracket rescue (nearest) -- union(schedule, burn, last6[prev], avg[prev])
                                                      plus a rescue: cars with last_step < m-10 (or
                                                      no PMS at all) whose NEAREST bracket == m
  6. V2 prev-step fix + bracket rescue (shifted) -- same as 5, SHIFTED bracket for the rescue

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


rows = []
for setup, cfg in SETUPS.items():
    qs, qe = quarter_bounds(*cfg["q"])
    cutoff = cfg["cutoff"]
    sc = serv.loc[serv["Service_Date"] <= cutoff, ["Vin_No", "Service_Date", "Mileage", "Service_Num"]]
    fs = fs_all.where(fs_all <= cutoff)

    # ---- vehicle-level bracket (identical to the template) ----
    odo = sc[sc["Mileage"] > 0].sort_values("Service_Date").groupby("Vin_No").last()
    km = odo["Mileage"].reindex(vins)
    km_date = odo["Service_Date"].reindex(vins)
    age = (km_date - fs).dt.days
    rate = km / age.where(age >= 30)
    km_start = km + rate * (qs - km_date).dt.days
    km_end = km + rate * (qe - km_date).dt.days
    km_mid = (km_start + km_end) / 2
    bracket_nearest = (np.floor((km_mid / 1000 + 5) / 10) * 10).where(rate.notna())
    bracket_shifted = (bracket_nearest - 10).where(bracket_nearest.notna())

    pms = sc[sc["Service_Num"] > 0]
    last_step = pms.groupby("Vin_No")["Service_Num"].max().reindex(vins).replace(1, 0)

    # skipper tests, one per bracket variant (bracket 2+ steps above last PMS; NaN last_step ->
    # treated as step 0 for the skipper test itself, matching the template)
    skipper_nearest = (bracket_nearest >= last_step.fillna(0) + 20) & bracket_nearest.notna()
    skip_set_nearest = set(skipper_nearest.index[skipper_nearest])
    skipper_shifted = (bracket_shifted >= last_step.fillna(0) + 20) & bracket_shifted.notna()
    skip_set_shifted = set(skipper_shifted.index[skipper_shifted])

    # rescue eligibility (rules 5/6): vehicle has NOT done the immediately-preceding milestone --
    # last_step < m-10 -- OR has no PMS record at all (last_step is NaN). Raw (unfilled) last_step
    # on purpose: NaN < anything is False, so the isna() clause carries the no-PMS case separately.
    print(f"\n{setup}: {len(skip_set_nearest):,} skippers (nearest bracket), "
          f"{len(skip_set_shifted):,} (shifted bracket) of {len(vins):,} gated vehicles; "
          f"{int(rate.isna().sum()):,} have no usable rate")

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

        # rule 4: V1, shifted bracket used for BOTH the skipper test and the assignment
        bracket_m_shifted_skip = set(skipper_shifted.index[skipper_shifted & (bracket_shifted == m)])
        v1 = (u4 - skip_set_shifted) | bracket_m_shifted_skip

        # rules 5/6: prev-step-fixed union + a rescue for cars that skipped the immediately
        # preceding step (or have no PMS at all), assigned by their bracket for THIS milestone
        not_prev_or_no_pms = last_step.isna() | (last_step < (m - 10))
        rescue_nearest = set(bracket_nearest.index[not_prev_or_no_pms & (bracket_nearest == m)
                                                    & bracket_nearest.notna()])
        rescue_shifted = set(bracket_shifted.index[not_prev_or_no_pms & (bracket_shifted == m)
                                                    & bracket_shifted.notna()])
        v2_nearest = u4_prev | rescue_nearest
        v2_shifted = u4_prev | rescue_shifted
        rescued_listed_nearest = rescue_nearest - u4_prev
        rescued_listed_shifted = rescue_shifted - u4_prev

        if cfg["q"] == (2026, 3):
            actual = set(q3.loc[(q3["Service_Num"] == m) & (q3["Service_Date"] >= qs)
                                & (q3["Service_Date"] <= qe), "Vin_No"])
        else:
            d = serv[serv["Service_Num"] == m].groupby("Vin_No")["Service_Date"].min()
            actual = set(d.index[(d >= qs) & (d <= qe)]) & gate
        done_before = set(serv.loc[(serv["Service_Num"] == m) & (serv["Service_Date"] < qs), "Vin_No"])

        rules = {
            "A min(schedule, burn) (today)": (in_q(earliest_date(sched, burn), qs, qe), None, None),
            "4-leg union": (u4, None, None),
            "4-leg, prev-step fix": (u4_prev, None, None),
            "V1 shifted bracket for skippers": (v1, None, None),
            "V2 prev-step fix + bracket rescue (nearest)": (v2_nearest, rescued_listed_nearest, actual),
            "V2 prev-step fix + bracket rescue (shifted)": (v2_shifted, rescued_listed_shifted, actual),
        }
        for name, (listed, rescued_listed, _actual) in rules.items():
            fresh = listed - done_before
            row = {"setup": setup, "milestone": f"{m}k", "rule": name, "actual": len(actual),
                   "listed": len(listed), "overlap": len(listed & actual),
                   "recall_pct": round(100 * len(listed & actual) / len(actual), 1),
                   "precision_pct": round(100 * len(listed & actual) / len(listed), 1),
                   "listed_stale_filtered": len(fresh),
                   "overlap_stale_filtered": len(fresh & actual),
                   "recall_stale_filtered_pct": round(100 * len(fresh & actual) / len(actual), 1),
                   "precision_stale_filtered_pct": round(100 * len(fresh & actual) / len(fresh), 1)}
            if rescued_listed is not None:
                row["rescued_listed"] = len(rescued_listed)
                row["rescued_came_in"] = len(rescued_listed & actual)
            rows.append(row)

out = pd.DataFrame(rows)
out.to_csv("q1_due_rule_experiment/mileage_bracket_variants_2026.csv", index=False)
for setup in SETUPS:
    print(f"\n===== {setup} (stale-filtered = what a forward list would really hold) =====")
    t = out[out["setup"] == setup]
    print(t[["milestone", "rule", "actual", "listed_stale_filtered", "overlap_stale_filtered",
             "recall_stale_filtered_pct", "precision_stale_filtered_pct"]].to_string(index=False))
    r56 = t[t["rule"].str.startswith("V2") & t["rescued_listed"].notna()]
    if not r56.empty:
        print(r56[["milestone", "rule", "rescued_listed", "rescued_came_in"]].to_string(index=False))
print("\nwrote q1_due_rule_experiment/mileage_bracket_variants_2026.csv")
