"""How high can 'recall of ALL visitors' go if the D' + prev-step list is widened? List-level only, no model.
Run from repo root:  PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/list_padding_ceiling.py 20 30 70 80

Same forward set-ups as roi_check.py (Q1 2026: history cut 2025-12-31; Q3 2026: base history, outcomes
from the Q3 file), stale-filtered, outcome = the STRICT quarter. A car is listed when any D' + prev-step
leg date falls inside [quarter start - pad, quarter end + pad], pad = 0 / 30 / 45 / 60 days.
Recall of all = listed cars that came in / every car that came in for the service in the quarter
(Q3: all distinct VINs in the Q3 file; Q1: first-ever visit in Q1, inside the EDA gate).
Writes dprime_v2/list_padding_ceiling_{tag}.csv.
"""
import sys

import pandas as pd

from build_unionstale_20k import load_base
from date_utils import parse_dates
from due_date import burn_rate_date, last_completed_plus_months, avg_interval_date
from features import resolve_service_num
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONES = [int(x) for x in sys.argv[1:]] or [20, 30, 70, 80]
TAG = "_".join(f"{m}k" for m in MILESTONES)
PADS = [0, 30, 45, 60]

df, serv = load_base()
vins = df["VIN"].values
gate = set(vins)
fs_all = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)
SETUPS = {"Q1 2026": dict(q=(2026, 1), cutoff=pd.Timestamp("2025-12-31")),
          "Q3 2026": dict(q=(2026, 3), cutoff=serv["Service_Date"].max())}

rows = []
for setup, cfg in SETUPS.items():
    qs, qe = quarter_bounds(*cfg["q"])
    cutoff = cfg["cutoff"]
    sc = serv.loc[serv["Service_Date"] <= cutoff, ["Vin_No", "Service_Date", "Mileage", "Service_Num"]]
    fs = fs_all.where(fs_all <= cutoff)
    for m in MILESTONES:
        burn = burn_rate_date(vins, sc, m, cutoff)
        last = last_completed_plus_months(vins, sc, m, months=6)
        avg = avg_interval_date(vins, fs, sc, m)
        below = sc[(sc["Service_Num"] > 0) & (sc["Service_Num"] < m)]
        prev = (below.groupby("Vin_No")["Service_Num"].max().reindex(vins) == m - 10).values
        legs = pd.DataFrame({"b": burn.reindex(vins).values, "l": last.where(prev).reindex(vins).values,
                             "a": avg.where(prev).reindex(vins).values}, index=vins).apply(pd.to_datetime)
        if cfg["q"] == (2026, 3):
            actual = set(q3.loc[(q3["Service_Num"] == m) & (q3["Service_Date"] >= qs)
                                & (q3["Service_Date"] <= qe), "Vin_No"])
        else:
            d = serv[serv["Service_Num"] == m].groupby("Vin_No")["Service_Date"].min()
            actual = set(d.index[(d >= qs) & (d <= qe)]) & gate
        done_before = set(serv.loc[(serv["Service_Num"] == m) & (serv["Service_Date"] < qs), "Vin_No"])
        for pad in PADS:
            lo, hi = qs - pd.Timedelta(days=pad), qe + pd.Timedelta(days=pad)
            listed = set(legs.index[legs.apply(lambda c: c.notna() & (c >= lo) & (c <= hi)).any(axis=1)]) - done_before
            hit = listed & actual
            rows.append({"quarter": setup, "milestone": f"{m}k", "pad_days": pad, "listed": len(listed),
                         "came_in": len(hit), "all_visitors": len(actual),
                         "recall_of_all_pct": round(100 * len(hit) / len(actual), 1),
                         "list_precision_pct": round(100 * len(hit) / len(listed), 1)})
        print(f"{setup} {m}k done")
res = pd.DataFrame(rows)
res.to_csv(f"dprime_v2/list_padding_ceiling_{TAG}.csv", index=False)
print(res.pivot_table(index=["milestone", "pad_days"], columns="quarter",
                      values=["listed", "recall_of_all_pct", "list_precision_pct"]).to_string())
