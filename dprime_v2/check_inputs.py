"""Step 0 of the D' v2 retrain: check the master tables dropped into data/v2/ BEFORE anything is built.
Read-only. Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/check_inputs.py

Each .csv / .xlsx in data/v2/ is recognised by its columns:
  eda          VIN + 'Last Service - PMS'
  history      Vin_No + Service_Date + Service_Code
  vhc          Vehicle Key + Service Date + RTS Code
  appointments Vehicle Magic + WIP Booking Date
  digital      Vehicle Key + session counts -- MUST carry a per-record date to be usable (the current
               data/Digital_Sessions_Q2-2026.csv has none, which is why it can't be cut per quarter)
For each: rows, every date column parsed with date_utils.parse_dates (min / max / unparsed), the join
key's coverage against service history, exact duplicate rows. History also gets Service_Num counts
per milestone via features.resolve_service_num (Excel 'Nov-20' safe), same-day duplicate milestone
rows, and the check that it reaches 2026-06-29. Prints PASS / CHECK lines; fixes nothing.
"""
import glob
import os

import pandas as pd

from date_utils import parse_dates
from features import resolve_service_num

FOLDER = "data/v2"
HISTORY_MUST_REACH = pd.Timestamp("2026-06-29")
CURRENT = {"eda": "data/EDA_Q2-2026.csv", "history": "data/Service_History_Q2-2026.csv",
           "vhc": "data/VHC_Q2-2026.csv", "appointments": "data/Appointments_Q2-2026.csv",
           "digital": "data/Digital_Sessions_Q2-2026.csv"}


def read(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False, encoding="ISO-8859-1")


def role(cols):
    c = set(cols)
    if {"VIN", "Last Service - PMS"} <= c:
        return "eda"
    if {"Vin_No", "Service_Date", "Service_Code"} <= c:
        return "history"
    if {"Vehicle Key", "Service Date", "RTS Code"} <= c:
        return "vhc"
    if {"Vehicle Magic", "WIP Booking Date"} <= c:
        return "appointments"
    if "Vehicle Key" in c and any("Sessions" in x for x in c):
        return "digital"
    return None


def date_cols(df):
    return [c for c in df.columns if "date" in c.lower() or c.lower() in ("due date in",)]


files = sorted(glob.glob(os.path.join(FOLDER, "*.csv")) + glob.glob(os.path.join(FOLDER, "*.xlsx")))
if not files:
    raise SystemExit(f"No .csv/.xlsx files in {FOLDER}/ yet -- drop the master tables there and re-run.")

tables = {}
for f in files:
    df = read(f)
    r = role(df.columns)
    print(f"\n=== {f}: {len(df):,} rows x {df.shape[1]} cols -> {r or 'UNRECOGNISED'}")
    if r is None:
        print("    columns:", list(df.columns)[:25])
        continue
    if r in tables:
        print(f"    CHECK: a second '{r}' file -- only the first one is used")
        continue
    tables[r] = df
    if os.path.exists(CURRENT[r]):
        cur_rows = sum(1 for _ in open(CURRENT[r], encoding="ISO-8859-1")) - 1
        print(f"    current {CURRENT[r]}: {cur_rows:,} rows")
    dcs = date_cols(df)
    if not dcs:
        print("    CHECK: no date column at all -- this table cannot be cut per quarter")
    for c in dcs:
        d = parse_dates(df[c])
        bad = int(d.isna().sum() - df[c].isna().sum())
        print(f"    {c}: {d.min()} .. {d.max()} | unparsed {bad:,}")
    dup = int(df.duplicated().sum())
    print(f"    exact duplicate rows: {dup:,}")

h = tables.get("history")
if h is not None:
    h = h.copy()
    h["Service_Date"] = parse_dates(h["Service_Date"])
    h["Service_Num"] = resolve_service_num(h)
    start, end = h["Service_Date"].min(), h["Service_Date"].max()
    print(f"\n--- history: {start.date()} .. {end.date()} -> "
          f"{'PASS' if end >= HISTORY_MUST_REACH else 'CHECK (must reach 2026-06-29)'}")
    if start > pd.Timestamp("2017-01-01"):
        print("    CHECK: this is NOT a full history (starts after 2017) -- training needs history from "
              "2016; a single-quarter file can only serve as outcomes")
    counts = h[h["Service_Num"] > 0].groupby("Service_Num")["Vin_No"].agg(["size", "nunique"])
    print("    PMS rows / distinct VINs per milestone (first 12):")
    print(counts.head(12).to_string())
    same_day = h[h["Service_Num"] > 0].duplicated(["Vin_No", "Service_Num", "Service_Date"]).sum()
    print(f"    same-day duplicate milestone rows: {int(same_day):,} (counted once in every metric)")
    hv, hk, hm = set(h["Vin_No"]), set(h["Vehicle_Key"].astype(str)), set(h.get("Vehicle Magic", pd.Series(dtype=str)).astype(str))
    if "eda" in tables:
        ev = set(tables["eda"]["VIN"])
        print(f"    EDA VINs found in history: {len(ev & hv):,} of {len(ev):,}")
    for r, key, pool in [("vhc", "Vehicle Key", hk), ("digital", "Vehicle Key", hk),
                         ("appointments", "Vehicle Magic", hm)]:
        if r in tables:
            k = set(tables[r][key].astype(str))
            print(f"    {r}: {len(k & pool):,} of {len(k):,} distinct {key} values join to service history")
print("\nDone. Anything marked CHECK must be resolved before Step 2.")
