"""Q3 2026 -- the first truly forward test of the due-date rules. Pure due-date math, no model.

Dates are built ONLY from the base history (data/Service_History_Q2-2026.csv, ends 2026-06-29), the
way a Q3 list is built before the quarter starts. Outcomes come from the separate Q3 file
(Q3/Service_History_Q3-2026.csv, 2026-07-01..2026-09-29) -- never from EDA, which has no reliable
per-visit date. Strict quarter 2026-07-01..2026-09-30, no padding, 20k-50k. Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe q1_due_rule_experiment/q3_union_rules.py

Rules: A min(schedule, burn) (production) | B union(schedule, burn) | 3-leg B + last-service+6mo |
4-leg 3-leg + avg-interval; each single leg alone too; plus the delivered Q3 list
(salman/expected_cohort_{m}k_Q3_2026.csv, every row) as a reference.

Actual completers = VINs with a Service_Num == m record in the Q3 file (same definition as
analysis/reconcile_quarter.py, so the delivered-list row reproduces its overlap). Service_Num comes from
features.resolve_service_num(): Service_Code first, Description only as fallback, which already maps
the Excel-damaged 'Nov-20' (and '11-20' / '11_20') to 20.

Recall is over ALL actual completers (out-of-gate ones count as missed -- no rule could list them).
"stale-filtered" = the same list minus vehicles whose base history already shows Service_Num == m
before 2026-07-01, i.e. what stale_completion_filter.py would remove from a forward list.
"""
import pandas as pd

from build_unionstale_20k import load_base
from date_utils import parse_dates
from due_date import (burn_rate_date, last_completed_plus_months, avg_interval_date,
                      earliest_date)
from features import expected_milestone_months, resolve_service_num
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONES = [20, 30, 40, 50]
Q_START, Q_END = quarter_bounds(2026, 3)
Q3_FILE = "Q3/Service_History_Q3-2026.csv"
# measured facts of the Q3 file (reconcile_quarter.EXPECTED_SERVICE_NUM_COUNTS) -- stop if they drift
EXPECTED = {20: 1746, 30: 1846, 40: 1713, 50: 1499}

q3 = pd.read_csv(Q3_FILE, low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)
print(f"{Q3_FILE}: {len(q3):,} rows, {q3['Service_Date'].min().date()} .. "
      f"{q3['Service_Date'].max().date()}, {int(q3['Service_Date'].isna().sum())} unparsed dates")
for m, n in EXPECTED.items():
    got = int((q3["Service_Num"] == m).sum())
    spell = q3.loc[q3["Service_Num"] == m, "Description"].astype(str).value_counts().to_dict()
    print(f"  {m}k rows {got} (expected {n}) -- Description spellings {spell}")
    assert got == n, f"{m}k row count drifted: {got} != {n}"

df, serv = load_base()
vins = df["VIN"].values
gate = set(vins)
first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
cutoff = serv["Service_Date"].max()
sc = serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]
print(f"base history ends {cutoff.date()} -- every date below is built from it alone")


def in_q(due):
    return set(due.index[due.notna() & (due >= Q_START) & (due <= Q_END)])


rows = []
for m in MILESTONES:
    sched = first_srv + pd.DateOffset(months=expected_milestone_months(m, "schedule"))
    burn = burn_rate_date(vins, sc, m, cutoff)
    last = last_completed_plus_months(vins, sc, m, months=6)
    avg = avg_interval_date(vins, first_srv, sc, m)
    s, b, l, a = in_q(sched), in_q(burn), in_q(last), in_q(avg)

    actual = set(q3.loc[(q3["Service_Num"] == m) & (q3["Service_Date"] >= Q_START)
                        & (q3["Service_Date"] <= Q_END), "Vin_No"])
    done_before = set(serv.loc[(serv["Service_Num"] == m) & (serv["Service_Date"] < Q_START),
                               "Vin_No"])
    out_gate = actual - gate
    repeat = actual & done_before
    delivered = set(pd.read_csv(f"salman/expected_cohort_{m}k_Q3_2026.csv")["VIN"])
    print(f"\n{m}k: {len(actual):,} came in during Q3 -- {len(out_gate)} outside the EDA gate, "
          f"{len(repeat)} already had a {m}k record before 1 Jul")

    rules = {
        "Schedule only": s,
        "Burn-rate only": b,
        "Last service + 6 mo only": l,
        "Avg interval only": a,
        "A min(schedule, burn)": in_q(earliest_date(sched, burn)),
        "B union(schedule, burn)": s | b,
        "3-leg union": s | b | l,
        "4-leg union": s | b | l | a,
        "Delivered Q3 list (reference)": delivered,
    }
    for name, listed in rules.items():
        hit = listed & actual
        fresh = listed - done_before
        fresh_hit = fresh & actual
        rows.append({
            "milestone": f"{m}k", "rule": name, "actual": len(actual),
            "listed": len(listed), "overlap": len(hit),
            "recall_pct": round(100 * len(hit) / len(actual), 1),
            "precision_pct": round(100 * len(hit) / len(listed), 1) if listed else None,
            "listed_stale_filtered": len(fresh), "overlap_stale_filtered": len(fresh_hit),
            "recall_stale_filtered_pct": round(100 * len(fresh_hit) / len(actual), 1),
            "precision_stale_filtered_pct":
                round(100 * len(fresh_hit) / len(fresh), 1) if fresh else None,
        })
        r = rows[-1]
        print(f"  {name:<30} listed {r['listed']:>6,} overlap {r['overlap']:>5,} "
              f"recall {r['recall_pct']:>5}% prec {r['precision_pct']:>5}% | stale-filtered: "
              f"listed {r['listed_stale_filtered']:>6,} recall {r['recall_stale_filtered_pct']:>5}% "
              f"prec {r['precision_stale_filtered_pct']:>5}%")

out = pd.DataFrame(rows)
out.to_csv("q1_due_rule_experiment/q3_union_rules_2026.csv", index=False)
print("\nwrote q1_due_rule_experiment/q3_union_rules_2026.csv")
