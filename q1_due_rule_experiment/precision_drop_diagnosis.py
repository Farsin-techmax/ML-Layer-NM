"""Why does D' + prev-step precision fall as mileage rises? Parts 1-3 of the diagnosis (Part 4, the
what-if filters, reads the per-car file written here). Pure due-date math, no model, 20k-100k,
strict quarter, stale-filtered, forward-built exactly as roi_check.py. Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe q1_due_rule_experiment/precision_drop_diagnosis.py

D' + prev-step = burn | last+6mo[prev] | avg[prev]; prev = highest PMS below m is exactly m-10.
Part 1  which leg(s) listed each car, and burn-listed cars by gap (m - last PMS step)
Part 2  what each wrong name actually did (Q1 has follow-up to 2026-06-29; Q3 only Jul-Sep)
Part 3  traits at the cutoff (recency, EDA status, warranty, age, km/day) + natural return rate
Outputs: precision_drop_cars_2026.csv (one row per listed car) and part1/part2/part3 CSVs.
"""
import numpy as np
import pandas as pd

from build_unionstale_20k import load_base
from date_utils import parse_dates
from due_date import burn_rate_date, last_completed_plus_months, avg_interval_date
from features import resolve_service_num
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
OUT = "q1_due_rule_experiment/precision_drop"

df, serv = load_base()
vins = df["VIN"].values
gate = set(vins)
eda = df.drop_duplicates("VIN").set_index("VIN")
fs_all = eda["FirstSrvDate"].reindex(vins)
status = eda["Vehicle Service Status"].astype(str) if "Vehicle Service Status" in eda else None
warranty = eda["Warranty Status"].astype(str) if "Warranty Status" in eda else None
serv["Mileage"] = serv["Mileage"].where(serv["Mileage"] <= 500_000)
data_end = serv["Service_Date"].max()

q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)

SETUPS = {
    "Q1 forward": dict(q=(2026, 1), cutoff=pd.Timestamp("2025-12-31"), outcomes=serv,
                       follow_end=data_end),
    "Q3 forward": dict(q=(2026, 3), cutoff=data_end, outcomes=q3, follow_end=None),
}


def in_q(due, qs, qe):
    return set(due.index[due.notna() & (due >= qs) & (due <= qe)])


def bucket(x, edges, labels):
    return pd.cut(pd.Series(x), edges, labels=labels).astype(object).fillna("unknown").values


cars = []
for setup, cfg in SETUPS.items():
    qs, qe = quarter_bounds(*cfg["q"])
    cutoff, outc = cfg["cutoff"], cfg["outcomes"]
    sc = serv.loc[serv["Service_Date"] <= cutoff]
    sc4 = sc[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]
    fs = fs_all.where(fs_all <= cutoff)

    # vehicle-level traits at the cutoff
    last_visit = sc.groupby("Vin_No")["Service_Date"].max().reindex(vins)
    days_since = (cutoff - last_visit).dt.days
    odo = sc[sc["Mileage"] > 0].sort_values("Service_Date").groupby("Vin_No").last()
    age_at_odo = (odo["Service_Date"].reindex(vins) - fs).dt.days
    kmday = (odo["Mileage"].reindex(vins) / age_at_odo.where(age_at_odo >= 30))
    age_years = (cutoff - fs).dt.days / 365.25

    win = outc[(outc["Service_Date"] >= qs) & (outc["Service_Date"] <= qe)]
    later = (serv[(serv["Service_Date"] > qe) & (serv["Service_Date"] <= cfg["follow_end"])]
             if cfg["follow_end"] is not None else serv.iloc[0:0])

    for m in MILESTONES:
        burn = burn_rate_date(vins, sc4, m, cutoff)
        last = last_completed_plus_months(vins, sc4, m, months=6)
        avg = avg_interval_date(vins, fs, sc4, m)
        below = sc[(sc["Service_Num"] > 0) & (sc["Service_Num"] < m)]
        hi = below.groupby("Vin_No")["Service_Num"].max().reindex(vins)
        prev = (hi == m - 10).values
        b, lp, ap = in_q(burn, qs, qe), in_q(last.where(prev), qs, qe), in_q(avg.where(prev), qs, qe)
        done_before = set(serv.loc[(serv["Service_Num"] == m) & (serv["Service_Date"] < qs), "Vin_No"])
        listed = (b | lp | ap) - done_before
        if cfg["q"] == (2026, 3):
            actual = set(win.loc[win["Service_Num"] == m, "Vin_No"])
        else:
            d = serv[serv["Service_Num"] == m].groupby("Vin_No")["Service_Date"].min()
            actual = set(d.index[(d >= qs) & (d <= qe)]) & gate

        L = sorted(listed)
        w = win[win["Vin_No"].isin(L)]
        pms_other = w[(w["Service_Num"] > 0) & (w["Service_Num"] != m)].groupby("Vin_No")["Service_Num"].max()
        any_win = set(w["Vin_No"])
        m_later = set(later.loc[(later["Service_Num"] == m) & later["Vin_No"].isin(L), "Vin_No"])
        any_later = set(later.loc[later["Vin_No"].isin(L), "Vin_No"])

        hi_l = hi.reindex(L)
        frame = pd.DataFrame({
            "setup": setup, "milestone": m, "VIN": L,
            "tp": [v in actual for v in L],
            "by_burn": [v in b for v in L], "by_last6": [v in lp for v in L],
            "by_avg": [v in ap for v in L],
            "last_step": hi_l.fillna(-1).replace(1, 0).values,
            "days_since_visit": days_since.reindex(L).values,
            "status": status.reindex(L).values if status is not None else "n/a",
            "warranty": warranty.reindex(L).values if warranty is not None else "n/a",
            "age_years": age_years.reindex(L).values,
            "km_per_day": kmday.reindex(L).values,
        })

        def outcome(v, tp):
            if tp:
                return "came in (TP)"
            if v in m_later:
                return "came for this service LATER (after quarter)"
            if v in pms_other.index:
                return ("came for an EARLIER service" if pms_other[v] < m
                        else "came for a LATER-step service")
            if v in any_win:
                return "non-PMS visit only"
            if v in any_later:
                return "no visit in quarter, back later"
            return "silent (no visit at all)"
        frame["outcome"] = [outcome(v, t) for v, t in zip(L, frame["tp"])]
        cars.append(frame)
        print(f"{setup} {m}k: listed {len(L):,}, TP {int(frame['tp'].sum()):,}, "
              f"precision {100 * frame['tp'].mean():.1f}%")

cars = pd.concat(cars, ignore_index=True)
cars["gap"] = np.where(cars["last_step"] >= 0, cars["milestone"] - cars["last_step"], np.nan)
cars.to_csv(f"{OUT}_cars_2026.csv", index=False)


def prec_table(col, name):
    t = cars.groupby(["setup", col, "milestone"])["tp"].agg(["size", "mean"]).reset_index()
    t["precision_pct"] = (100 * t["mean"]).round(1)
    t = t.rename(columns={"size": "listed"}).drop(columns="mean")
    t.to_csv(f"{OUT}_{name}_2026.csv", index=False)
    for setup in SETUPS:
        p = t[t["setup"] == setup].pivot(index=col, columns="milestone",
                                         values=["listed", "precision_pct"])
        print(f"\n--- {name} | {setup} ---")
        print(p.to_string())


# Part 1 -- legs
cars["legs"] = np.select(
    [cars.by_burn & ~cars.by_last6 & ~cars.by_avg, ~cars.by_burn & cars.by_last6 & ~cars.by_avg,
     ~cars.by_burn & ~cars.by_last6 & cars.by_avg],
    ["burn only", "last6 only", "avg only"], "2+ legs")
prec_table("legs", "part1_legs")
cars["gap_bucket"] = np.where(cars["by_burn"], cars["gap"].map(
    lambda g: "no PMS" if pd.isna(g) else ("10 (prev step)" if g == 10 else ("20" if g == 20 else "30+"))),
    "not burn-listed")
prec_table("gap_bucket", "part1_burn_gap")

# Part 2 -- what the wrong names did
fp = cars[~cars["tp"]]
p2 = fp.groupby(["setup", "milestone", "outcome"]).size().unstack(fill_value=0)
p2_pct = (100 * p2.div(p2.sum(axis=1), axis=0)).round(1)
p2.to_csv(f"{OUT}_part2_fp_outcomes_2026.csv")
print("\n--- Part 2: what the wrong names did (share of wrong names, %) ---")
print(p2_pct.to_string())

# Part 3 -- traits
cars["recency"] = bucket(cars["days_since_visit"], [-1, 90, 180, 365, 730, 1e9],
                         ["<=90d", "91-180d", "181-365d", "1-2y", ">2y"])
cars["age"] = bucket(cars["age_years"], [-1, 2, 3, 4, 5, 6, 100], ["<2y", "2-3y", "3-4y", "4-5y", "5-6y", "6y+"])
cars["kmday_b"] = bucket(cars["km_per_day"], [-1, 30, 50, 80, 1e9], ["<30", "30-50", "50-80", "80+"])
for col in ["recency", "status", "warranty", "age", "kmday_b"]:
    prec_table(col, f"part3_{col}")

# natural return rate: did m-10 in 2024 -> did m by data end
rr = []
for m in MILESTONES:
    prev_d = serv[serv["Service_Num"] == m - 10].groupby("Vin_No")["Service_Date"].min()
    cohort = set(prev_d.index[(prev_d >= "2024-01-01") & (prev_d <= "2024-12-31")]) & gate
    did = set(serv.loc[serv["Service_Num"] == m, "Vin_No"])
    rr.append({"milestone": m, "did_prev_step_in_2024": len(cohort),
               "came_back_for_m_by_2026-06-29_pct": round(100 * len(cohort & did) / len(cohort), 1)})
rr = pd.DataFrame(rr)
rr.to_csv(f"{OUT}_part3_return_rate_2026.csv", index=False)
print("\n--- natural return rate ---")
print(rr.to_string(index=False))
