"""Near misses among the most predictable customers: real Q1-2026 completers that the base did NOT
list, although they (a) did every 10k PMS step below the target and (b) kept their own service
rhythm (gap from last PMS to this milestone within 0.75-1.25x their own average past PMS gap).
Per milestone and rule (A = min(schedule, burn), B = union(schedule, burn), C = union of all four),
the 5 whose closest projected due date (among the legs that rule uses) missed Q1 by the fewest days.
Strict Q1 window, no padding, same population as missed_profile_by_rule.py. Run from repo root.
"""
import json

import pandas as pd

from build_unionstale_20k import load_base
from due_date import (burn_rate_date, last_completed_plus_months, avg_interval_date,
                      earliest_date)
from features import expected_milestone_months
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONES = [20, 30, 40, 50]
N = 5
Q_START, Q_END = quarter_bounds(2026, 1)
LO, HI = 0.75, 1.25

df, serv = load_base()
vins = df["VIN"].values
first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
cutoff = serv["Service_Date"].max()
serv_cols = serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]
hist_cols = ["Vin_No", "Vehicle_Key", "Service_Date", "Service_Code", "Description", "Mileage",
             "Service_Num"]
by_vin = {v: g.sort_values("Service_Date") for v, g in serv[hist_cols].groupby("Vin_No")}


def d(x):
    return None if pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d")


def days_outside(x):
    """Signed days outside Q1: negative = before Q1 opened, positive = after Q1 closed, 0 = inside."""
    if pd.isna(x):
        return None
    if x < Q_START:
        return -(Q_START - x).days
    if x > Q_END:
        return (x - Q_END).days
    return 0


def in_window(due):
    return set(due.index[due.notna() & (due >= Q_START) & (due <= Q_END)])


out = {"milestones": {}}
csv_rows = []
for m in MILESTONES:
    legs = {
        "sched": first_srv + pd.DateOffset(months=expected_milestone_months(m, "schedule")),
        "burn": burn_rate_date(vins, serv_cols, m, cutoff),
        "last": last_completed_plus_months(vins, serv_cols, m, months=6),
        "avg": avg_interval_date(vins, first_srv, serv_cols, m),
    }
    legs["min"] = earliest_date(legs["sched"], legs["burn"])
    # burn-rate dates carry a time of day; Q_END is 31 Mar 00:00, so compare whole days only
    legs = {k: v.dt.normalize() for k, v in legs.items()}
    rule_legs = {"A": ["min"], "B": ["sched", "burn"], "C": ["sched", "burn", "last", "avg"]}
    listed = {r: set().union(*[in_window(legs[k]) for k in ks]) for r, ks in rule_legs.items()}

    at_ms = serv[serv["Service_Num"] == m].sort_values("Service_Date")
    first_ms = at_ms.groupby("Vin_No").first()
    actual = first_ms[(first_ms["Service_Date"] >= Q_START) & (first_ms["Service_Date"] <= Q_END)]

    # predictable customers: every 10k step done + kept rhythm
    keepers = {}
    for vin, r in actual.iterrows():
        past = by_vin[vin]
        past = past[past["Service_Date"] < r["Service_Date"]]
        pms = (past[past["Service_Num"] > 0].sort_values(["Service_Date", "Service_Num"])
               .groupby("Service_Date").last())
        if not set(range(10, m, 10)).issubset(set(pms["Service_Num"].astype(int))):
            continue
        if len(pms) < 2:
            continue
        own_avg = pd.Series(pms.index).diff().dt.days.dropna().mean()
        realized = (r["Service_Date"] - pms.index[-1]).days
        ratio = realized / own_avg if own_avg > 0 else None
        if ratio is None or not (LO <= ratio <= HI):
            continue
        keepers[vin] = {"row": r, "past": past, "own_avg": own_avg, "realized": realized,
                        "ratio": ratio, "last_pms": pms.index[-1]}

    ms_out = {"actual": len(actual), "keepers": len(keepers), "rules": {}}
    for rule, ks in rule_legs.items():
        cand = []
        for vin, k in keepers.items():
            if vin in listed[rule]:
                continue
            dists = {leg: days_outside(legs[leg].get(vin)) for leg in ks}
            known = {leg: v for leg, v in dists.items() if v is not None}
            if not known:
                continue
            closest = min(known, key=lambda leg: abs(known[leg]))
            cand.append((abs(known[closest]), vin, closest, known[closest]))
        cand.sort()
        rows = []
        for absd, vin, leg, signed in cand[:N]:
            k = keepers[vin]
            rows.append({
                "key": k["row"]["Vehicle_Key"],
                "actual": d(k["row"]["Service_Date"]),
                "actual_mileage": None if pd.isna(k["row"]["Mileage"]) else int(k["row"]["Mileage"]),
                "due": {leg: d(legs[leg].get(vin)) for leg in ["sched", "burn", "last", "avg", "min"]},
                "off": {leg: days_outside(legs[leg].get(vin)) for leg in ["sched", "burn", "last", "avg", "min"]},
                "closest_leg": leg, "closest_off": signed,
                "own_avg_gap": round(k["own_avg"]), "realized_gap": k["realized"],
                "last_pms": d(k["last_pms"]),
                "history": [{"date": d(h["Service_Date"]), "code": str(h["Service_Code"]),
                             "desc": str(h["Description"]),
                             "mileage": None if pd.isna(h["Mileage"]) else int(h["Mileage"]),
                             "pms": bool(h["Service_Num"] > 0)} for _, h in k["past"].iterrows()],
            })
            csv_rows.append({"milestone": f"{m}k", "rule": rule, "VIN": vin, "Vehicle_Key": rows[-1]["key"],
                             "actual_date": rows[-1]["actual"], "closest_leg": leg,
                             "days_outside_q1": signed, "own_avg_gap_days": round(k["own_avg"]),
                             "realized_gap_days": k["realized"]})
        ms_out["rules"][rule] = {"missed_keepers": len(cand), "examples": rows}
        print(f"{m}k {rule}: keepers {len(keepers)}, missed keepers {len(cand)}, "
              f"top-5 days off: {[c[3] for c in cand[:N]]} via {[c[2] for c in cand[:N]]}")
    out["milestones"][str(m)] = ms_out

json.dump(out, open("q1_due_rule_experiment/near_miss_rhythm_keepers.json", "w"), indent=1)
pd.DataFrame(csv_rows).to_csv("q1_due_rule_experiment/near_miss_rhythm_keepers.csv", index=False)
print("wrote q1_due_rule_experiment/near_miss_rhythm_keepers.json / .csv")
