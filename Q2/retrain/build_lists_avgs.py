"""Step 1 of the average-interval retrain (decided with the user 2026-10-21, 30k first, replicate if it
works): quarter-by-quarter lists built with the SAME rule as the Q2 2026 deliverable, so training and
scoring populations match.

For every quarter 2021Q1 .. 2026Q3 and milestone m:
  history  = service history dated BEFORE the quarter starts (data/Service_History_Q2-2026.csv, ends
             2026-06-29). Nothing the quarter could not have known is read.
  rule     = due_date.avg_interval_date(include_first_srv_segment=False, scale_by_steps=True) --
             average-interval leg only, scaled by steps away, NO prev-step gate. Listed when the due
             date falls inside the quarter. One row per car per quarter (repeats across quarters kept).
  EDA      = data/EDA_Q1-2026.csv, the same snapshot the Q2 deliverable used. EDA is a recurring
             current-state sheet, so it cannot be sliced back in time: it supplies only the gate
             (`Last Service - PMS != '-'`, redundant here -- a car can only get a due date if it has a
             completed PMS in history before the cutoff) and the Vehicle_Key / VIN mapping.
  labels   = Actual = first-ever Service_Num == m date (base history + Q3/Service_History_Q3-2026.csv
             for 2026Q3). CameInQuarter = Actual inside the strict quarter (the training label).
             EarlyCompleter = Actual before the quarter starts (dropped in build_matrix_avgs.py, the
             training-side equivalent of the stale filter on forward lists).

Columns match what dprime_v2/features_v2.block_features() reads (burn/last6 legs left empty: this rule
has one leg), so the leak-free v2 feature code is reused unchanged.

Run from repo root:  PYTHONPATH=. ./venv/Scripts/python.exe Q2/retrain/build_lists_avgs.py 30
Writes Q2/retrain/lists_{m}k_avgs.csv.
"""
import sys

import pandas as pd

from date_utils import parse_dates
from due_date import avg_interval_date
from features import resolve_service_num
from pmstrainfeatureeng_refactored import quarter_bounds, _apply_first_srv_date

MILESTONES = [int(x) for x in sys.argv[1:] if x.isdigit()] or [30]
FIRST_Q, LAST_Q = (2021, 1), (2026, 3)
GRACE = pd.Timedelta(days=15)
EDA_PATH = "data/EDA_Q1-2026.csv"

serv = pd.read_csv("data/Service_History_Q2-2026.csv", low_memory=False, encoding="ISO-8859-1")
serv["Service_Date"] = parse_dates(serv["Service_Date"])
serv["Mileage"] = pd.to_numeric(serv["Mileage"], errors="coerce")
serv = serv.dropna(subset=["Service_Date"])
serv["Service_Num"] = resolve_service_num(serv)
q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)

eda = pd.read_csv(EDA_PATH, low_memory=False, encoding="ISO-8859-1")
eda = eda.query("`Last Service - PMS` != '-'").copy()
for c in ["Invoice date", "First Service Date", "Last Service Date"]:
    eda[c] = parse_dates(eda[c])
eda = _apply_first_srv_date(eda, serv)
vins = eda["VIN"].values
fs_all = eda.set_index("VIN")["FirstSrvDate"].reindex(vins)
cols4 = ["Vin_No", "Service_Date", "Mileage", "Service_Num"]

quarters, (y, q) = [], FIRST_Q
while (y, q) <= LAST_Q:
    quarters.append((y, q))
    y, q = (y + 1, 1) if q == 4 else (y, q + 1)

for m in MILESTONES:
    actual_all = pd.concat([serv, q3])
    actual_all = actual_all[actual_all["Service_Num"] == m].groupby("Vin_No")["Service_Date"].min()
    parts = []
    for (y, q) in quarters:
        qs, qe = quarter_bounds(y, q)
        cutoff = qs - pd.Timedelta(days=1)
        sc = serv.loc[serv["Service_Date"] <= cutoff, cols4]
        fs = fs_all.where(fs_all <= cutoff)
        avg = avg_interval_date(vins, fs, sc, m, include_first_srv_segment=False, scale_by_steps=True)
        avg = pd.Series(avg.values, index=vins)
        inq = avg.notna() & (avg >= qs) & (avg <= qe)
        f = pd.DataFrame(index=pd.Index(vins[inq.values], name="VIN"))
        f["burn_date"], f["last6_date"], f["avg_date"] = pd.NaT, pd.NaT, avg[inq].values
        f["by_burn"], f["by_last6"], f["by_avg"], f["n_legs"] = False, False, True, 1
        f[f"Expected{m}kDate"] = f["avg_date"]
        below = sc[(sc["Service_Num"] > 0) & (sc["Service_Num"] < m)]
        step = below.groupby("Vin_No")["Service_Num"].max().reindex(f.index)
        f["last_step"] = step.replace(1, 0).values
        f["gap"] = m - f["last_step"]
        f["CohortQuarter"], f["q_start"], f["q_end"], f["cutoff"] = f"{y}Q{q}", qs, qe, cutoff
        act = actual_all.reindex(f.index)
        f[f"Actual{m}kDate"] = act.values
        f["EarlyCompleter"] = (act.notna() & (act < qs)).astype(int).values
        f["TargetFlag"] = (act.notna() & (act <= qe + GRACE)).astype(int).values
        f["CameInQuarter"] = (act.notna() & (act >= qs) & (act <= qe)).astype(int).values
        parts.append(f.reset_index())
        ne = f[f["EarlyCompleter"] == 0]
        print(f"{m}k {y}Q{q}: listed {len(f):,} | early {f['EarlyCompleter'].mean():.1%} | "
              f"after stale drop {len(ne):,}, came in quarter {ne['CameInQuarter'].mean():.1%}")
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(f"Q2/retrain/lists_{m}k_avgs.csv", index=False)
    print(f"\n{m}k: {len(out):,} rows over {len(quarters)} quarters -> Q2/retrain/lists_{m}k_avgs.csv\n")
