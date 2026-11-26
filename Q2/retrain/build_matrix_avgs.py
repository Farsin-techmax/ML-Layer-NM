"""Step 2 of the average-interval retrain: lists_{m}k_avgs.csv + dprime_v2/features_v2 -> training
matrix and 2026 Q1/Q2/Q3 test files.

- Features: dprime_v2.features_v2.block_features(), unchanged -- one code path for every row, all
  service-history / VHC / appointment features cut at the row's own cutoff, EDA static facts only
  (EDA is a current-state sheet and cannot be sliced back in time). The only difference from
  features_v2.load_sources() is the EDA file: data/EDA_Q1-2026.csv, the snapshot the Q2 list used.
- Early completers DROPPED (user decision 2026-10-21), same as the stale filter on forward lists.
- Label: TargetFlag := CameInQuarter, the STRICT quarter -- what the Q2 deliverable's
  ActualTargetFlag measures. (D' v2 used a +15-day grace; not here.)
- Training = 2021Q1..2025Q4; tests = 2026Q1, 2026Q2 (the deliverable quarter), 2026Q3 (forward check).
- Pruning on TRAINING rows only: drop constants, then for |corr| > 0.98 pairs drop the later column.

Run from repo root:  PYTHONPATH=. ./venv/Scripts/python.exe Q2/retrain/build_matrix_avgs.py 30
Writes Q2/retrain/train_{m}k_avgs.csv, Q2/retrain/test_{m}k_Q{1,2,3}_2026_avgs.csv,
Q2/retrain/pruned_{m}k_avgs.json.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from date_utils import parse_dates
from dprime_v2.features_v2 import Sources, block_features
from features import resolve_service_num
from pmstrainfeatureeng_refactored import _apply_first_srv_date

MILESTONES = [int(x) for x in sys.argv[1:] if x.isdigit()] or [30]
TEST_Q = {"2026Q1": "Q1", "2026Q2": "Q2", "2026Q3": "Q3"}
TRAIN_FROM = "2021Q1"
EDA_PATH = "data/EDA_Q1-2026.csv"


def load_sources():
    """features_v2.load_sources() with the EDA file swapped to the Q1 snapshot; otherwise identical."""
    serv = pd.read_csv("data/Service_History_Q2-2026.csv", low_memory=False, encoding="ISO-8859-1")
    serv["Service_Date"] = parse_dates(serv["Service_Date"])
    serv["Mileage"] = pd.to_numeric(serv["Mileage"], errors="coerce").where(lambda x: x <= 500_000)
    serv["Revenue"] = pd.to_numeric(serv["Revenue"], errors="coerce").fillna(0)
    serv = serv.dropna(subset=["Service_Date"])
    serv["Service_Num"] = resolve_service_num(serv)
    eda = pd.read_csv(EDA_PATH, low_memory=False, encoding="ISO-8859-1")
    eda = eda.query("`Last Service - PMS` != '-'").copy()
    for c in ["Invoice date", "First Service Date", "Last Service Date"]:
        eda[c] = parse_dates(eda[c])
    eda = _apply_first_srv_date(eda, serv)
    return Sources(eda, serv)


if __name__ == "__main__":
    src = load_sources()
    print("sources loaded")
    for m in MILESTONES:
        lst = pd.read_csv(f"Q2/retrain/lists_{m}k_avgs.csv",
                          parse_dates=["q_start", "q_end", "cutoff", "burn_date", "last6_date", "avg_date",
                                       f"Expected{m}kDate"])
        lst = lst[(lst["EarlyCompleter"] == 0) & (lst["CohortQuarter"] >= TRAIN_FROM)].copy()
        lst["TargetFlag"] = lst["CameInQuarter"]
        parts = []
        for cq, blk in lst.groupby("CohortQuarter"):
            t0 = time.time()
            r = blk.iloc[0]
            f = block_features(src, blk, m, r["cutoff"], r["q_start"], r["q_end"]).reindex(blk["VIN"].values)
            head = pd.DataFrame({"VIN": blk["VIN"].values, "TargetFlag": blk["TargetFlag"].values,
                                 "CameInQuarter": blk["CameInQuarter"].values, "CohortQuarter": cq,
                                 f"Next{m}K_Due": blk[f"Expected{m}kDate"].values})
            parts.append(pd.concat([head, f.reset_index(drop=True)], axis=1))
            print(f"{m}k {cq}: {len(blk):,} rows, {f.shape[1]} features, came in {blk['TargetFlag'].mean():.1%}, "
                  f"{time.time() - t0:.1f}s")
        full = pd.concat(parts, ignore_index=True)
        head_cols = ["VIN", "TargetFlag", "CameInQuarter", "CohortQuarter", f"Next{m}K_Due"]
        feats = [c for c in full.columns if c not in head_cols]
        train = full[~full["CohortQuarter"].isin(TEST_Q)]

        const = [c for c in feats if train[c].nunique(dropna=False) <= 1]
        keep = [c for c in feats if c not in const]
        corr = train[keep].astype(float).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, bool), k=1))
        dup = {}
        for c in upper.columns:
            hit = upper.index[upper[c] > 0.98].tolist()
            if hit:
                dup[c] = hit[0]
        keep = [c for c in keep if c not in dup]
        json.dump({"constant": const, "near_duplicate_of": dup, "kept": keep},
                  open(f"Q2/retrain/pruned_{m}k_avgs.json", "w"), indent=1)
        print(f"\n{m}k: {len(feats)} features built -> {len(const)} constant, {len(dup)} near-duplicate "
              f"dropped -> {len(keep)} kept")
        print(f"   constant: {const}\n   near-duplicate (dropped -> kept twin): {dup}")

        cols = head_cols + keep
        train[cols].to_csv(f"Q2/retrain/train_{m}k_avgs.csv", index=False)
        for cq, q in TEST_Q.items():
            t = full[full["CohortQuarter"] == cq]
            t[cols].to_csv(f"Q2/retrain/test_{m}k_{q}_2026_avgs.csv", index=False)
            print(f"   {q} 2026 test: {len(t):,} rows, came in {int(t['CameInQuarter'].sum()):,}")
        print(f"   train: {len(train):,} rows, came in {train['TargetFlag'].mean():.1%}\n")
