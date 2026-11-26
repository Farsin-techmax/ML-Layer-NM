"""Step-by-step trace of the four due dates for one vehicle, exactly as due_date.py computes them.
Mode 1 (no args): list 50k Q1-2026 completers with a full history (service 1, 10k..40k) where the
min(schedule, burn) date misses Q1 but at least one leg lands in Q1. Mode 2 (key): print the trace.
Run from repo root: python q1_due_rule_experiment/trace_due_dates.py [VEHICLE_KEY]
"""
import sys

import pandas as pd

from build_unionstale_20k import load_base
from due_date import (burn_rate_date, last_completed_plus_months, avg_interval_date,
                      earliest_date)
from features import expected_milestone_months
from pmstrainfeatureeng_refactored import quarter_bounds

M = 50
Q_START, Q_END = quarter_bounds(2026, 1)

df, serv = load_base()
vins = df["VIN"].values
first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
cutoff = serv["Service_Date"].max()
sc = serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]
legs = {
    "schedule": first_srv + pd.DateOffset(months=expected_milestone_months(M, "schedule")),
    "burn": burn_rate_date(vins, sc, M, cutoff),
    "last6": last_completed_plus_months(vins, sc, M, months=6),
    "avg": avg_interval_date(vins, first_srv, sc, M),
}
legs["min"] = earliest_date(legs["schedule"], legs["burn"])


def inq(d):
    return pd.notna(d) and Q_START <= d <= Q_END


at_ms = serv[serv["Service_Num"] == M].sort_values("Service_Date").groupby("Vin_No").first()
actual = at_ms[(at_ms["Service_Date"] >= Q_START) & (at_ms["Service_Date"] <= Q_END)]

if len(sys.argv) == 1:
    rows = []
    for v, r in actual.iterrows():
        h = serv[(serv["Vin_No"] == v) & (serv["Service_Date"] < r["Service_Date"])]
        steps = set(h.loc[h["Service_Num"] > 0, "Service_Num"].astype(int))
        if not {1, 10, 20, 30, 40}.issubset(steps) or h["Mileage"].isna().any():
            continue
        d = {k: legs[k].get(v) for k in legs}
        if any(pd.isna(x) for x in d.values()) or inq(d["min"]):
            continue
        in_q = [k for k in ["schedule", "burn", "last6", "avg"] if inq(d[k])]
        if not in_q:
            continue
        if serv.loc[serv["Vin_No"] == v, "Vehicle_Key"].nunique() > 1:
            continue
        rows.append({"key": r["Vehicle_Key"], "VIN": v, "actual": r["Service_Date"].date(),
                     **{k: d[k].date() for k in d}, "in_q1": "+".join(in_q),
                     "gap_to_actual": min(abs((d[k] - r["Service_Date"]).days) for k in in_q)})
    out = pd.DataFrame(rows).sort_values("gap_to_actual")
    print(len(out), "candidates")
    print(out.head(15).to_string(index=False))
    sys.exit()

key = sys.argv[1]
v = serv.loc[serv["Vehicle_Key"] == key, "Vin_No"].iloc[0]
e = df[df["VIN"] == v].iloc[0]
h = serv[serv["Vin_No"] == v].sort_values("Service_Date")
print("VIN", v, "| keys", serv.loc[serv["Vin_No"] == v, "Vehicle_Key"].unique().tolist())
print("EDA Invoice date", e["Invoice date"], "| EDA First Service Date", e["First Service Date"])
print("earliest service-history date", h["Service_Date"].min())
print("FirstSrvDate (used)", first_srv.get(v))
print(h[["Service_Date", "Service_Code", "Description", "Service_Num", "Mileage"]].to_string(index=False))
for k in legs:
    print(k, legs[k].get(v))
print("actual 50k", actual.loc[v, "Service_Date"] if v in actual.index else None)
