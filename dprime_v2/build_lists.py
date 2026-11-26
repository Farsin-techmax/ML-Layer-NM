"""Step 2 of the D' v2 retrain: quarter-by-quarter D' + prev-step lists with labels, no features yet.
Run from repo root:  PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/build_lists.py 20 30

For every quarter 2019Q1 .. 2026Q3 and each milestone m (2019-2020 rows are never trained on --
build_matrix.py starts training at 2021Q1 -- they only feed the v3 "listed in earlier quarters" features):
  history  = service history BEFORE the quarter starts (data/Service_History_Q2-2026.csv, which ends
             2026-06-29) -- dates are never built from anything the quarter could not have known
  legs     = burn_rate_date | last_completed_plus_months[prev] | avg_interval_date[prev]
             (prev = highest completed PMS below m is exactly m-10) -- D' + prev-step, as roi_check.py
  listed   = any leg date inside the quarter; one row per car per quarter (repeats across quarters
             are kept on purpose)
  label    = today's candidates rule (build_candidates_cohort): Actual = first Service_Num == m date,
             TargetFlag = Actual <= quarter end + 15 days, EarlyCompleter = Actual < quarter start
             (early completers stay POSITIVE; the column lets metrics drop them).
             2026Q3 outcomes come from Q3/Service_History_Q3-2026.csv (held-out test quarter).
Gate and anchor exactly as the rest of the investigation: EDA `Last Service - PMS != '-'`,
_apply_first_srv_date (relaxed invoice guard).
Writes dprime_v2/lists_{m}k.csv. `--pad N` widens the list window to [quarter start - N, quarter end + N]
(labels unchanged, still the strict quarter) and writes dprime_v2/lists_{m}k_padN.csv instead.
"""
import sys

import pandas as pd

from date_utils import parse_dates
from due_date import burn_rate_date, last_completed_plus_months, avg_interval_date
from features import resolve_service_num
from pmstrainfeatureeng_refactored import quarter_bounds, _apply_first_srv_date

ARGV = sys.argv[1:]
PAD = int(ARGV[ARGV.index("--pad") + 1]) if "--pad" in ARGV else 0   # list window = quarter +- PAD days
MILESTONES = [int(x) for x in ARGV if x.isdigit() and (not PAD or x != str(PAD))] or [20, 30]
PAD_TD = pd.Timedelta(days=PAD)
SUFFIX = f"_pad{PAD}" if PAD else ""
FIRST_Q, LAST_Q = (2019, 1), (2026, 3)
GRACE = pd.Timedelta(days=15)

serv = pd.read_csv("data/Service_History_Q2-2026.csv", low_memory=False, encoding="ISO-8859-1")
serv["Service_Date"] = parse_dates(serv["Service_Date"])
serv["Mileage"] = pd.to_numeric(serv["Mileage"], errors="coerce")
serv = serv.dropna(subset=["Service_Date"])
serv["Service_Num"] = resolve_service_num(serv)
q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)

eda = pd.read_csv("data/EDA_Q2-2026.csv", low_memory=False, encoding="ISO-8859-1")
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
        burn = burn_rate_date(vins, sc, m, cutoff)
        last = last_completed_plus_months(vins, sc, m, months=6)
        avg = avg_interval_date(vins, fs, sc, m)
        below = sc[(sc["Service_Num"] > 0) & (sc["Service_Num"] < m)]
        last_step = below.groupby("Vin_No")["Service_Num"].max().reindex(vins)
        prev = (last_step == m - 10).values
        legs = pd.DataFrame({"burn_date": burn.reindex(vins).values,
                             "last6_date": last.where(prev).reindex(vins).values,
                             "avg_date": avg.where(prev).reindex(vins).values}, index=vins)
        legs = legs.apply(lambda c: pd.to_datetime(c))
        inq = legs.apply(lambda c: c.notna() & (c >= qs - PAD_TD) & (c <= qe + PAD_TD))
        listed = inq.any(axis=1)
        f = legs[listed].copy()
        f["by_burn"], f["by_last6"], f["by_avg"] = (inq.loc[listed, c].values for c in legs.columns)
        f["n_legs"] = f[["by_burn", "by_last6", "by_avg"]].sum(axis=1)
        f[f"Expected{m}kDate"] = legs[listed].where(inq[listed]).min(axis=1)
        step = last_step.reindex(f.index)
        f["last_step"] = step.replace(1, 0).values
        f["gap"] = m - f["last_step"]
        f["CohortQuarter"], f["q_start"], f["q_end"], f["cutoff"] = f"{y}Q{q}", qs, qe, cutoff
        act = actual_all.reindex(f.index)
        f[f"Actual{m}kDate"] = act.values
        f["EarlyCompleter"] = (act.notna() & (act < qs)).astype(int).values
        f["TargetFlag"] = (act.notna() & (act <= qe + GRACE)).astype(int).values
        f["CameInQuarter"] = (act.notna() & (act >= qs) & (act <= qe)).astype(int).values
        parts.append(f.rename_axis("VIN").reset_index())
        print(f"{m}k {y}Q{q}: listed {len(f):,} | positive {f['TargetFlag'].mean():.1%} | "
              f"early {f['EarlyCompleter'].mean():.1%} | came in quarter {f['CameInQuarter'].sum():,}")
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(f"dprime_v2/lists_{m}k{SUFFIX}.csv", index=False)
    rep = out.groupby("VIN").size()
    print(f"\n{m}k: {len(out):,} rows over {len(quarters)} quarters, {out['VIN'].nunique():,} cars "
          f"({int((rep > 1).sum()):,} appear in 2+ quarters) -> dprime_v2/lists_{m}k{SUFFIX}.csv\n")
