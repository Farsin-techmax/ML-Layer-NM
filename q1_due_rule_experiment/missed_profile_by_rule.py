"""Profile of real Q1-2026 milestone completers that each due-date rule did NOT list, for
20k/30k/40k/50k. Feeds the missed-completers artifact. Run from repo root.

Rules (all strict Q1 2026 window, no padding):
  A  min(schedule, burn_rate)                         -- production rule
  B  union(schedule, burn_rate)
  C  union(schedule, burn_rate, last_completed_plus_months, avg_interval_date)

Actual completers = every VIN in service history whose first Service_Num == milestone record is
dated 2026-01-01..2026-03-31 (no EDA gate). Rules only list EDA-gated vehicles (load_base()).

Past visits = distinct service dates strictly before the vehicle's milestone visit. Mutually
exclusive groups for missed vehicles:
  first_time   0 past visits
  single       exactly 1 past visit (PMS or non-PMS)
  nonpms_only  2+ past visits, none of them PMS (Service_Num == 0 on every row)
  other        2+ past visits, at least one PMS
"""
import json

import pandas as pd

from build_unionstale_20k import load_base
from due_date import (burn_rate_date, last_completed_plus_months, avg_interval_date,
                      earliest_date)
from features import expected_milestone_months
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONES = [20, 30, 40, 50]
N_EXAMPLES = 5
Q_START, Q_END = quarter_bounds(2026, 1)

df, serv = load_base()
vins = df["VIN"].values
first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
cutoff = serv["Service_Date"].max()
serv_cols = serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]

visit_rows = serv[["Vin_No", "Vehicle_Key", "Service_Date", "Service_Code", "Description",
                   "Mileage", "Service_Num"]].copy()
by_vin = {v: g.sort_values("Service_Date") for v, g in visit_rows.groupby("Vin_No")}


def in_window(due):
    return set(due.index[due.notna() & (due >= Q_START) & (due <= Q_END)])


def classify(vin, ms_date):
    past = by_vin[vin]
    past = past[past["Service_Date"] < ms_date]
    visits = past.groupby("Service_Date")["Service_Num"].max()
    n = len(visits)
    if n == 0:
        group = "first_time"
    elif n == 1:
        group = "single"
    elif (visits == 0).all():
        group = "nonpms_only"
    else:
        group = "other"
    return group, n, past


def fmt_mileage(x):
    return None if pd.isna(x) else int(x)


def band_status(service_num, km):
    """Proper mileage band for a PMS step: (step - 2k) to (step + 5k), floor 0. 20k -> 18k-25k."""
    if pd.isna(km) or km <= 0:
        return "unknown"
    lo, hi = max(0, service_num - 2) * 1000, (service_num + 5) * 1000
    if km < lo:
        return "below"
    if km > hi:
        return "above"
    return "in"


def pms_profile(past, ms_row, m):
    """For the 'other' group: past PMS visits (one per date, the highest PMS step that day)."""
    pms_rows = (past[past["Service_Num"] > 0].sort_values(["Service_Date", "Service_Num"])
                .groupby("Service_Date").last())
    n_pms = len(pms_rows)
    steps = set(pms_rows["Service_Num"].astype(int))
    complete = set(range(10, m, 10)).issubset(steps)
    bands = [band_status(int(r["Service_Num"]), r["Mileage"]) for _, r in pms_rows.iterrows()]
    known = [b for b in bands if b != "unknown"]
    if not known:
        past_band = "unknown"
    elif all(b == "in" for b in known):
        past_band = "all"
    elif any(b == "in" for b in known):
        past_band = "some"
    else:
        past_band = "none"
    ms_band = band_status(m, ms_row["Mileage"])
    return {"n_pms": n_pms, "complete": complete, "past_band": past_band, "ms_band": ms_band,
            "by_the_book": complete and past_band == "all" and ms_band == "in"}


def fmt_date(x):
    return None if pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d")


def example(p, m, vin=None, dues=None):
    due = None
    if dues is not None:
        due = {k: fmt_date(s.get(vin)) for k, s in dues.items()}
    return {
        "due": due,
        "key": p["ms_row"]["Vehicle_Key"],
        "visit_date": p["ms_row"]["Service_Date"].strftime("%Y-%m-%d"),
        "visit_mileage": fmt_mileage(p["ms_row"]["Mileage"]),
        "visit_code": str(p["ms_row"]["Service_Code"]),
        "visit_band": band_status(m, p["ms_row"]["Mileage"]),
        "history": [{
            "date": h["Service_Date"].strftime("%Y-%m-%d"),
            "code": str(h["Service_Code"]),
            "desc": str(h["Description"]),
            "mileage": fmt_mileage(h["Mileage"]),
            "pms": bool(h["Service_Num"] > 0),
            "band": band_status(int(h["Service_Num"]), h["Mileage"]) if h["Service_Num"] > 0 else None,
        } for _, h in p["past"].iterrows()],
    }


out = {"milestones": {}}
detail_rows = []
for m in MILESTONES:
    due_sched = first_srv + pd.DateOffset(months=expected_milestone_months(m, "schedule"))
    due_burn = burn_rate_date(vins, serv_cols, m, cutoff)
    due_last = last_completed_plus_months(vins, serv_cols, m, months=6)
    due_avg = avg_interval_date(vins, first_srv, serv_cols, m)

    listed = {
        "A": in_window(earliest_date(due_sched, due_burn)),
        "B": in_window(due_sched) | in_window(due_burn),
    }
    listed["C"] = listed["B"] | in_window(due_last) | in_window(due_avg)
    dues = {"sched": due_sched, "burn": due_burn, "last": due_last, "avg": due_avg,
            "min": earliest_date(due_sched, due_burn)}

    at_ms = serv[serv["Service_Num"] == m].sort_values("Service_Date")
    first_ms = at_ms.groupby("Vin_No").first()
    actual = first_ms[(first_ms["Service_Date"] >= Q_START) & (first_ms["Service_Date"] <= Q_END)]

    prof = {}
    for vin, r in actual.iterrows():
        group, n, past = classify(vin, r["Service_Date"])
        prof[vin] = {"group": group, "n_past": n, "past": past, "ms_row": r}
        if group == "other":
            prof[vin]["pms"] = pms_profile(past, r, m)

    ms_out = {"actual": len(actual), "rules": {}}
    for rule in ["A", "B", "C"]:
        missed = sorted((v for v in actual.index if v not in listed[rule]),
                        key=lambda v: (actual.loc[v, "Service_Date"], v))
        counts = {"first_time": 0, "single": 0, "nonpms_only": 0, "other": 0}
        examples = {"first_time": [], "single": [], "nonpms_only": []}
        ph = {"total": 0, "pms_count": {"1": 0, "2": 0, "3": 0, "4+": 0},
              "ms_band": {"in": 0, "below": 0, "above": 0, "unknown": 0},
              "past_band": {"all": 0, "some": 0, "none": 0, "unknown": 0},
              "complete": 0, "by_the_book": 0, "examples": []}
        complete_vins = []
        for vin in missed:
            p = prof[vin]
            counts[p["group"]] += 1
            row = {"milestone": f"{m}k", "rule": rule, "VIN": vin,
                   "Vehicle_Key": p["ms_row"]["Vehicle_Key"],
                   "milestone_visit_date": p["ms_row"]["Service_Date"].date(),
                   "milestone_visit_mileage": fmt_mileage(p["ms_row"]["Mileage"]),
                   "past_visits": p["n_past"], "group": p["group"]}
            if p["group"] in examples and len(examples[p["group"]]) < N_EXAMPLES:
                examples[p["group"]].append(example(p, m))
            if p["group"] == "other":
                q = p["pms"]
                ph["total"] += 1
                ph["pms_count"]["4+" if q["n_pms"] >= 4 else str(q["n_pms"])] += 1
                ph["ms_band"][q["ms_band"]] += 1
                ph["past_band"][q["past_band"]] += 1
                ph["complete"] += q["complete"]
                ph["by_the_book"] += q["by_the_book"]
                if q["complete"]:
                    complete_vins.append(vin)
                row.update({"past_pms_visits": q["n_pms"], "all_10k_steps_done": q["complete"],
                            "milestone_visit_band": q["ms_band"], "past_pms_band": q["past_band"],
                            "by_the_book": q["by_the_book"]})
            detail_rows.append(row)
        # examples: by-the-book first, then other complete-history vehicles
        complete_vins.sort(key=lambda v: (not prof[v]["pms"]["by_the_book"],
                                          actual.loc[v, "Service_Date"], v))
        ph["examples"] = [example(prof[v], m, v, dues) for v in complete_vins[:N_EXAMPLES]]
        ms_out["rules"][rule] = {"caught": len(actual) - len(missed), "missed": len(missed),
                                 "counts": counts, "examples": examples, "pms_hist": ph}
        print(f"{m}k rule {rule}: actual {len(actual)}, caught {len(actual)-len(missed)}, "
              f"missed {len(missed)} -> {counts}")
        print(f"    PMS-history group {ph['total']}: pms_count {ph['pms_count']}, "
              f"ms_band {ph['ms_band']}, past_band {ph['past_band']}, "
              f"complete {ph['complete']}, by_the_book {ph['by_the_book']}")
    out["milestones"][str(m)] = ms_out

json.dump(out, open("q1_due_rule_experiment/missed_profile_by_rule.json", "w"), indent=1)
pd.DataFrame(detail_rows).to_csv("q1_due_rule_experiment/missed_profile_by_rule.csv", index=False)
print("wrote q1_due_rule_experiment/missed_profile_by_rule.json / .csv")
