"""Two-leg union of burn_rate + avg_interval (drops schedule and last-service+6mo from the 4-leg
union). How many real Q1-2026 completers it keeps vs the 4-leg union, and what it does to list
size and precision. Pure due-date math, no model, strict quarter, no padding -- same population
and dates as method4_avg_interval.py, so rows A/B/C match the tables already in CLAUDE.md.
Run from repo root.
"""
import pandas as pd

from build_unionstale_20k import load_base
from due_date import (burn_rate_date, last_completed_plus_months, avg_interval_date,
                      earliest_date)
from features import expected_milestone_months
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONES = [20, 30, 40, 50]
Q_START, Q_END = quarter_bounds(2026, 1)


def in_q(due):
    return set(due.index[due.notna() & (due >= Q_START) & (due <= Q_END)])


df, serv = load_base()
vins = df["VIN"].values
first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
cutoff = serv["Service_Date"].max()
sc = serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]

rows, lost_rows = [], []
for m in MILESTONES:
    sched = first_srv + pd.DateOffset(months=expected_milestone_months(m, "schedule"))
    burn = burn_rate_date(vins, sc, m, cutoff)
    last = last_completed_plus_months(vins, sc, m, months=6)
    avg = avg_interval_date(vins, first_srv, sc, m)
    s, b, l, a = in_q(sched), in_q(burn), in_q(last), in_q(avg)

    date_ms = serv[serv["Service_Num"] == m].sort_values("Service_Date").groupby("Vin_No")["Service_Date"].first()
    actual = date_ms[(date_ms >= Q_START) & (date_ms <= Q_END)]
    actual = set(actual.index[actual.index.isin(df["VIN"])])

    rules = {
        "A min(schedule, burn)": in_q(earliest_date(sched, burn)),
        "B union(schedule, burn)": s | b,
        "C union of all four": s | b | l | a,
        "D union(burn, avg)": b | a,
        "D' union(burn, last, avg)": b | l | a,
    }
    c_hit = rules["C union of all four"] & actual
    for name, listed in rules.items():
        hit = listed & actual
        rows.append({"milestone": f"{m}k", "rule": name, "actual": len(actual),
                     "listed": len(listed), "caught": len(hit),
                     "recall_pct": round(100 * len(hit) / len(actual), 1),
                     "precision_pct": round(100 * len(hit) / len(listed), 1),
                     "kept_of_C_caught": len(hit & c_hit),
                     "kept_of_C_caught_pct": round(100 * len(hit & c_hit) / len(c_hit), 1)})

    # completers C catches but D does not: which dropped leg was carrying them
    lost = c_hit - (b | a)
    lost_rows.append({"milestone": f"{m}k", "lost": len(lost),
                      "schedule_only": len(lost & (s - l)),
                      "last6_only": len(lost & (l - s)),
                      "both_schedule_and_last6": len(lost & s & l)})
    # non-completers C lists but D does not (wrong names dropped)
    c_list = rules["C union of all four"]
    lost_rows[-1]["wrong_names_dropped"] = len((c_list - (b | a)) - actual)

out = pd.DataFrame(rows)
lost = pd.DataFrame(lost_rows)
print(out.to_string(index=False))
print()
print(lost.to_string(index=False))
out.to_csv("q1_due_rule_experiment/union_burn_avg_Q1_2026.csv", index=False)
lost.to_csv("q1_due_rule_experiment/union_burn_avg_lost_vs_C_Q1_2026.csv", index=False)
print("wrote q1_due_rule_experiment/union_burn_avg_Q1_2026.csv / _lost_vs_C_Q1_2026.csv")
