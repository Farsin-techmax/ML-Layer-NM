"""Add service-history columns to the nine Q2 deliverables (user requests 2026-10-21).

  LastPMSServiceDate     = latest Service_Date with Service_Num > 0  (a PMS visit)
  LastNonPMSServiceDate  = latest Service_Date with Service_Num == 0 (any other visit; blank if none)
  PMSDoneBeforeMilestone = number of DISTINCT PMS services (Service_Num > 0) completed BELOW the
                           milestone the row is listed for -- a repeated line for the same service
                           counts once; the first free service ("1") counts. Max = m/10 (30k: 1,10,20 -> 3).
  ExpectedMinusActualDays = NextServiceDate - ActualServiceDate in days, signed (NOT absolute):
                           positive = came in before the expected date, negative = after it,
                           blank = no actual visit on record.

All read from data/Service_History_Q2-2026.csv, by VIN (all of a vehicle's keys), cut at 2026-03-31 --
the day the Q2 list was built from -- so a Q2 visit never shows up here (that is ActualServiceDate).
Service_Num via features.resolve_service_num(), the only safe parser (0 = non-PMS, load-bearing).

Re-run after Q2/score_q2_avg_interval.py and Q2/retrain/apply_to_deliverables.py; it replaces these
columns if they already exist. Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe Q2/add_last_service_dates.py
"""
import os

import pandas as pd

from date_utils import parse_dates
from features import resolve_service_num

MILESTONES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
CUTOFF = pd.Timestamp("2026-03-31")
COLS = ["LastPMSServiceDate", "LastNonPMSServiceDate", "PMSDoneBeforeMilestone", "ExpectedMinusActualDays"]

serv = pd.read_csv("data/Service_History_Q2-2026.csv", low_memory=False, encoding="ISO-8859-1")
serv["Service_Date"] = parse_dates(serv["Service_Date"])
serv = serv.dropna(subset=["Service_Date"])
serv["Service_Num"] = resolve_service_num(serv)
serv = serv[serv["Service_Date"] <= CUTOFF]
last_pms = serv[serv["Service_Num"] > 0].groupby("Vin_No")["Service_Date"].max().dt.strftime("%Y-%m-%d")
last_non = serv[serv["Service_Num"] == 0].groupby("Vin_No")["Service_Date"].max().dt.strftime("%Y-%m-%d")
pms_steps = serv.loc[serv["Service_Num"] > 0, ["Vin_No", "Service_Num"]].drop_duplicates()

locked = []
for m in MILESTONES:
    path = f"Q2/expected_cohort_{m}k_Q2_2026.csv"
    d = pd.read_csv(path).drop(columns=COLS, errors="ignore")
    d["LastPMSServiceDate"] = d["VIN"].map(last_pms)
    d["LastNonPMSServiceDate"] = d["VIN"].map(last_non)
    below = pms_steps[pms_steps["Service_Num"] < m].groupby("Vin_No").size()
    d["PMSDoneBeforeMilestone"] = d["VIN"].map(below).fillna(0).astype(int)
    diff = (pd.to_datetime(d["NextServiceDate"]) - pd.to_datetime(d["ActualServiceDate"])).dt.days
    d["ExpectedMinusActualDays"] = diff.astype("Int64")
    tmp = path + ".tmp"
    d.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
    except PermissionError:
        locked.append(m)
        print(f"{m}k: LOCKED (open elsewhere) -- new version left at {tmp}")
        continue
    print(f"{m}k: {len(d):,} rows | last PMS date found {d['LastPMSServiceDate'].notna().sum():,} | "
          f"last non-PMS date found {d['LastNonPMSServiceDate'].notna().sum():,} | "
          f"PMS done below {m}k (of max {m // 10}): "
          f"{d['PMSDoneBeforeMilestone'].value_counts().sort_index().to_dict()}")
    x = d["ExpectedMinusActualDays"].dropna()
    print(f"      expected - actual (n={len(x):,}): median {x.median():.0f} d, "
          f"came early {(x > 0).sum():,}, same day {(x == 0).sum():,}, came late {(x < 0).sum():,}, "
          f"range {x.min()}..{x.max()}")
print(f"\nlocked: {locked}" if locked else "\nall nine updated")
