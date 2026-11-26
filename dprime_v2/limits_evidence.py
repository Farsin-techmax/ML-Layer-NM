"""Evidence for "why 90/90 is out of reach": is customer behaviour getting less predictable over time, and is
there a visible disruption in 2025-26? Read-only. Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/limits_evidence.py

1. Own-rhythm predictability by year: for every PMS visit 20k-100k (first visit per service per car) whose car
   has >= 2 earlier PMS gaps, predict  previous PMS date + the car's own average earlier gap  and measure the
   error. The best a timing rule can do from a customer's own history.
2. Return within 12 months by cohort quarter: of cars that did service m-10, the share doing m within 365 days.
3. Monthly workshop volume (distinct car-days, all visits and PMS only), 2023-01 .. 2026-09.
4. Appointment no-show rate by due quarter.
Writes dprime_v2/limits_evidence_{rhythm,return,volume,noshow}.csv.
"""
import numpy as np
import pandas as pd

from date_utils import parse_dates
from features import resolve_service_num

cols = ["Vin_No", "Service_Date", "Service_Code", "Description"]
h = pd.concat([pd.read_csv(f, low_memory=False, encoding="ISO-8859-1", usecols=cols)
               for f in ["data/Service_History_Q2-2026.csv", "Q3/Service_History_Q3-2026.csv"]])
h["Service_Date"] = parse_dates(h["Service_Date"])
h["Service_Num"] = resolve_service_num(h)
h = h.dropna(subset=["Service_Date"])
END = pd.Timestamp("2026-09-29")

# 1. own-rhythm predictability
p = (h[h["Service_Num"] > 0].groupby(["Vin_No", "Service_Num"])["Service_Date"].min().reset_index()
     .sort_values(["Vin_No", "Service_Date"]))
p["prev"] = p.groupby("Vin_No")["Service_Date"].shift()
p["gap"] = (p["Service_Date"] - p["prev"]).dt.days
g = p.groupby("Vin_No")["gap"]
p["n_prior_gaps"] = g.cumcount() - 1                       # gaps strictly before this visit's own gap
p["own_avg"] = g.transform(lambda s: s.shift().expanding().mean())
t = p[(p["Service_Num"] >= 20) & (p["n_prior_gaps"] >= 2) & (p["gap"] > 0)].copy()
t["err"] = t["gap"] - t["own_avg"]
t["ratio"] = t["gap"] / t["own_avg"]
t["pred"] = t["prev"] + pd.to_timedelta(t["own_avg"].round(), unit="D")
t["year"] = t["Service_Date"].dt.year
t = t[t["year"].between(2021, 2026)]
r1 = t.groupby("year").agg(visits=("err", "size"), median_gap_days=("gap", "median"),
                           median_abs_err_days=("err", lambda e: e.abs().median()),
                           within_45d_pct=("err", lambda e: 100 * (e.abs() <= 45).mean()),
                           within_25pct_of_own_rhythm_pct=("ratio", lambda r: 100 * r.between(0.75, 1.25).mean()),
                           came_later_than_1_5x_pct=("ratio", lambda r: 100 * (r > 1.5).mean()),
                           came_sooner_than_0_5x_pct=("ratio", lambda r: 100 * (r < 0.5).mean()))
t["same_q"] = t["pred"].dt.to_period("Q") == t["Service_Date"].dt.to_period("Q")
r1["same_quarter_pct"] = t.groupby("year")["same_q"].mean() * 100
r1 = r1.round(1)
r1.to_csv("dprime_v2/limits_evidence_rhythm.csv")
print("1. Predict the next PMS from the car's own average gap (cars with >= 2 earlier gaps), by visit year")
print(r1.to_string(), "\n")

# 2. return within 12 months
first = h[h["Service_Num"] > 0].groupby(["Vin_No", "Service_Num"])["Service_Date"].min()
rows = []
for m in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
    a = first.xs(m - 10, level="Service_Num") if m > 10 else None
    b = first.xs(m, level="Service_Num").reindex(a.index)
    d = pd.DataFrame({"a": a, "ret": (b - a).dt.days.between(0, 365)})
    d = d[(d["a"] >= "2021-01-01") & (d["a"] <= END - pd.Timedelta(days=365))]
    d["q"] = d["a"].dt.to_period("Q").astype(str)
    s = d.groupby("q")["ret"].agg(["size", "mean"])
    for q, (n, v) in s.iterrows():
        rows.append({"milestone": f"{m}k", "cohort_quarter": q, "cars": n, "return_12m_pct": round(100 * v, 1)})
r2 = pd.DataFrame(rows)
r2.to_csv("dprime_v2/limits_evidence_return.csv", index=False)
r2["year"] = r2["cohort_quarter"].str[:4]
yr = r2.groupby(["milestone", "year"]).apply(lambda x: np.average(x["return_12m_pct"], weights=x["cars"])).unstack().round(1)
print("2. Did service m-10, came back for m within 12 months (%), by year of the m-10 visit "
      "(2025 = Jan-Sep cohorts, full 12 months observed)")
print(yr.to_string(), "\n")

# 3. monthly volume
v = h[(h["Service_Date"] >= "2023-01-01") & (h["Service_Date"] <= END)].drop_duplicates(["Vin_No", "Service_Date", "Service_Num"])
v["month"] = v["Service_Date"].dt.to_period("M")
allv = v.drop_duplicates(["Vin_No", "Service_Date"]).groupby("month").size().rename("car_days_all")
pmsv = v[v["Service_Num"] > 0].drop_duplicates(["Vin_No", "Service_Date"]).groupby("month").size().rename("car_days_pms")
r3 = pd.concat([allv, pmsv], axis=1)
r3.to_csv("dprime_v2/limits_evidence_volume.csv")
piv = r3["car_days_all"].reset_index()
piv["y"], piv["mo"] = piv["month"].dt.year, piv["month"].dt.month
print("3. Workshop visits per month (distinct car-days, all visit types)")
print(piv.pivot(index="mo", columns="y", values="car_days_all").to_string(), "\n")
pp = r3["car_days_pms"].reset_index()
pp["y"], pp["mo"] = pp["month"].dt.year, pp["month"].dt.month
print("   PMS visits per month")
print(pp.pivot(index="mo", columns="y", values="car_days_pms").to_string(), "\n")

# 4. appointment no-shows
a = pd.read_csv("data/v2/Service_Appointments_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
a["due"] = parse_dates(a["Due Date IN"].replace("-", np.nan))
a["noshow"] = pd.to_numeric(a["No Show"], errors="coerce").fillna(0)
a["showed"] = pd.to_numeric(a["[Vehicle Checkedin (Show Flag)]"], errors="coerce").fillna(0)
a = a[(a["due"] >= "2021-01-01") & (a["due"] <= END)]
a["q"] = a["due"].dt.to_period("Q").astype(str)
r4 = a.groupby("q").agg(appointments=("noshow", "size"), noshow_pct=("noshow", "mean"), showed_pct=("showed", "mean"))
r4[["noshow_pct", "showed_pct"]] = (100 * r4[["noshow_pct", "showed_pct"]]).round(1)
r4.to_csv("dprime_v2/limits_evidence_noshow.csv")
print("4. Appointments by due quarter")
print(r4.to_string())
