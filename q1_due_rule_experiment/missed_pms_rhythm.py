"""Missed Q1-2026 completers with PMS history (2+ past visits, >=1 PMS): did they keep their own
service rhythm? Compares the realized gap (last past PMS -> milestone visit) with the vehicle's own
average gap between past PMS visits. Rules B and C as in missed_profile_by_rule.py. Run from root.

  kept_rhythm   realized / own average within 0.75..1.25  -> history pointed the right way
  slowed        ratio > 1.25 (came later than its own rhythm)
  sped_up       ratio < 0.75 (came sooner than its own rhythm)
  one_pms_only  only one past PMS visit, so no past PMS gap to compare against
Also: for kept_rhythm vehicles, whether last PMS date + own average PMS gap lands in Q1 2026.
"""
import pandas as pd

from build_unionstale_20k import load_base
from due_date import burn_rate_date, last_completed_plus_months, avg_interval_date
from features import expected_milestone_months
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONES = [20, 30, 40, 50]
Q_START, Q_END = quarter_bounds(2026, 1)
LO, HI = 0.75, 1.25

df, serv = load_base()
vins = df["VIN"].values
first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
cutoff = serv["Service_Date"].max()
serv_cols = serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]
by_vin = {v: g for v, g in serv[["Vin_No", "Service_Date", "Service_Num"]].groupby("Vin_No")}


def in_window(due):
    return set(due.index[due.notna() & (due >= Q_START) & (due <= Q_END)])


rows = []
summary = []
for m in MILESTONES:
    due_sched = first_srv + pd.DateOffset(months=expected_milestone_months(m, "schedule"))
    due_burn = burn_rate_date(vins, serv_cols, m, cutoff)
    due_last = last_completed_plus_months(vins, serv_cols, m, months=6)
    due_avg = avg_interval_date(vins, first_srv, serv_cols, m)
    listed = {"B": in_window(due_sched) | in_window(due_burn)}
    listed["C"] = listed["B"] | in_window(due_last) | in_window(due_avg)

    at_ms = serv[serv["Service_Num"] == m].sort_values("Service_Date")
    first_ms = at_ms.groupby("Vin_No")["Service_Date"].first()
    actual = first_ms[(first_ms >= Q_START) & (first_ms <= Q_END)]

    for rule in ["B", "C"]:
        counts = {"kept_rhythm": 0, "slowed": 0, "sped_up": 0, "one_pms_only": 0,
                  "kept_rhythm_own_avg_in_q1": 0}
        total = 0
        for vin, ms_date in actual.items():
            if vin in listed[rule]:
                continue
            past = by_vin[vin]
            past = past[past["Service_Date"] < ms_date]
            visits = past.groupby("Service_Date")["Service_Num"].max()
            if len(visits) < 2 or (visits == 0).all():
                continue  # not in the PMS-history group
            total += 1
            pms_dates = visits[visits > 0].index.sort_values()
            realized = (ms_date - pms_dates[-1]).days
            if len(pms_dates) < 2:
                cls, own_avg, ratio, own_in_q1 = "one_pms_only", None, None, None
            else:
                own_avg = pd.Series(pms_dates).diff().dt.days.dropna().mean()
                ratio = realized / own_avg if own_avg > 0 else None
                if ratio is None:
                    cls = "one_pms_only"
                elif ratio > HI:
                    cls = "slowed"
                elif ratio < LO:
                    cls = "sped_up"
                else:
                    cls = "kept_rhythm"
                proj = pms_dates[-1] + pd.Timedelta(days=round(own_avg))
                own_in_q1 = bool(Q_START <= proj <= Q_END)
            counts[cls] += 1
            if cls == "kept_rhythm" and own_in_q1:
                counts["kept_rhythm_own_avg_in_q1"] += 1
            rows.append({"milestone": f"{m}k", "rule": rule, "VIN": vin,
                         "milestone_visit_date": ms_date.date(), "last_pms_date": pms_dates[-1].date(),
                         "realized_gap_days": realized,
                         "own_avg_pms_gap_days": None if own_avg is None else round(own_avg, 1),
                         "ratio": None if ratio is None else round(ratio, 2), "class": cls,
                         "last_pms_plus_own_avg_in_q1": own_in_q1})
        summary.append({"milestone": f"{m}k", "rule": rule, "pms_history_missed": total, **counts})

s = pd.DataFrame(summary)
print(s.to_string(index=False))
s.to_csv("q1_due_rule_experiment/missed_pms_rhythm_summary.csv", index=False)
pd.DataFrame(rows).to_csv("q1_due_rule_experiment/missed_pms_rhythm.csv", index=False)
d = pd.DataFrame(rows)
print("\nmedian ratio by class (rule C):")
print(d[d["rule"] == "C"].groupby(["milestone", "class"])["ratio"].median().unstack())
