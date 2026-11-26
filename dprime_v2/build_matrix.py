"""Step 3b: lists_{m}k.csv + features_v2 -> training matrix and 2026 Q1/Q2/Q3 test files.
Run from repo root:  PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/build_matrix.py 20 30 [--set v3] [--smoke]

- Early completers are DROPPED everywhere (decision 2026-10-16): rows look like a real forward list
  after the stale filter. TargetFlag = came in during the quarter or within 15 days after it.
- Training = 2021Q1..2025Q4 (lists also hold 2019-2020, used only for the v3 prior-listing counts);
  tests = 2026Q1, 2026Q2, 2026Q3 (held out; outcomes from the Q3 file).
- Pruning decided on TRAINING rows only, then applied to every file: drop constant columns, then for
  every pair with |corr| > 0.98 drop the later column. Dropped names are printed and saved.
- Next{m}K_Due = earliest in-quarter leg date (retrain.py sorts and cuts on it).
--set v2 (default): the v2 features. --set v3: v2 + the behaviour block (features_v2.behaviour_features),
  written under _dprime_v3 names so v2 stays reproducible.
--pad N reads lists_{m}k_padN.csv (build_lists.py --pad N), sets TargetFlag = came in during the STRICT
  quarter, and writes under _dprime_{set}_padN names.
--smoke builds 2025Q4 + 2026Q3 only, to check speed and output before the full run.
Writes refactored_test_dir/final_processed_{m}k_dprime_{set}.csv,
test_sets/test_{m}k_Q{1,2,3}_2026_dprime_{set}.csv, dprime_v2/pruned_{m}k[_v3].json.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from dprime_v2.features_v2 import load_sources, block_features, prior_listing_table

argv = sys.argv[1:]
SET = argv[argv.index("--set") + 1] if "--set" in argv else "v2"
PAD = int(argv[argv.index("--pad") + 1]) if "--pad" in argv else 0
args = [a for a in argv if not a.startswith("--") and a != SET and (not PAD or a != str(PAD))]
NAME = SET + (f"_pad{PAD}" if PAD else "")   # output set name, e.g. v2_pad45
SMOKE = "--smoke" in argv
MILESTONES = [int(x) for x in args] or [20, 30]
TEST_Q = {"2026Q1": "Q1", "2026Q2": "Q2", "2026Q3": "Q3"}
TRAIN_FROM = "2021Q1"
BEHAVIOUR = SET == "v3"

src = load_sources()
print(f"sources loaded (feature set {SET})")

for m in MILESTONES:
    lst_all = pd.read_csv(f"dprime_v2/lists_{m}k" + (f"_pad{PAD}" if PAD else "") + ".csv",
                          parse_dates=["q_start", "q_end", "cutoff", "burn_date", "last6_date", "avg_date",
                                       f"Expected{m}kDate"])
    prior = prior_listing_table(lst_all) if BEHAVIOUR else None
    lst = lst_all[(lst_all["EarlyCompleter"] == 0) & (lst_all["CohortQuarter"] >= TRAIN_FROM)].copy()
    if PAD:   # a padded list must be trained on the strict quarter it is judged on (no +15-day grace)
        lst["TargetFlag"] = lst["CameInQuarter"]
    if SMOKE:
        lst = lst[lst["CohortQuarter"].isin(["2025Q4", "2026Q3"])]
    parts = []
    for cq, blk in lst.groupby("CohortQuarter"):
        t0 = time.time()
        r = blk.iloc[0]
        f = block_features(src, blk, m, r["cutoff"], r["q_start"], r["q_end"], behaviour=BEHAVIOUR, prior=prior)
        f = f.reindex(blk["VIN"].values)
        head = pd.DataFrame({"VIN": blk["VIN"].values, "TargetFlag": blk["TargetFlag"].values,
                             "CameInQuarter": blk["CameInQuarter"].values, "CohortQuarter": cq,
                             f"Next{m}K_Due": blk[f"Expected{m}kDate"].values})
        parts.append(pd.concat([head, f.reset_index(drop=True)], axis=1))
        print(f"{m}k {cq}: {len(blk):,} rows, {f.shape[1]} features, positive "
              f"{blk['TargetFlag'].mean():.1%}, {time.time() - t0:.1f}s")
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
              open(f"dprime_v2/pruned_{m}k{'' if NAME == 'v2' else '_' + NAME}.json", "w"), indent=1)
    print(f"\n{m}k: {len(feats)} features built -> {len(const)} constant, {len(dup)} near-duplicate dropped "
          f"-> {len(keep)} kept")
    print(f"   constant: {const}\n   near-duplicate (dropped -> kept twin): {dup}")

    cols = head_cols + keep
    suffix = f"_dprime_{NAME}" + ("_smoke" if SMOKE else "")
    train[cols].to_csv(f"refactored_test_dir/final_processed_{m}k{suffix}.csv", index=False)
    for cq, q in TEST_Q.items():
        t = full[full["CohortQuarter"] == cq]
        if len(t):
            t[cols].to_csv(f"test_sets/test_{m}k_{q}_2026{suffix}.csv", index=False)
            print(f"   {q} 2026 test: {len(t):,} rows, came in quarter {int(t['CameInQuarter'].sum()):,}")
    print(f"   train: {len(train):,} rows, positive {train['TargetFlag'].mean():.1%}\n")
