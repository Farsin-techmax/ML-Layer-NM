"""
Leak screen: single-feature AUC train-vs-test (gate G6) + grouped permutation importance.

Two diagnostics, both read-only:

1. Single-feature AUC on train vs on test, for every feature the model uses. A genuine feature
   does not get stronger on held-out data -- test AUC substantially above train AUC is the leak
   signature (Service Frequency: 0.830 train -> 0.945 test; pre-fix PMS_Delay: 0.985 test).
   Gate rule (G6): test - train >= 0.20 with test >= 0.65 -> STOP; >= 0.10 -> flag.
   AUC is reported RAW (not folded above 0.5): < 0.5 means the relationship is INVERTED on that
   split, the signature of the ever-mode sampling artifact.

2. Grouped permutation: permute a whole feature family together and measure test-AUC delta.
   Single-column permutation under-credits redundant features here (PMS_Delay measured +0.0005
   alone despite 0.90 single-feature test AUC, because has_x/PMS_Count_Prior covered for it).
   A family whose joint shuffle RAISES test AUC is actively harmful (the 20k ever-mode
   schedule-keeping family raised it by +0.19).

Usage:
    python leak_screen.py 20 --train refactored_test_dir/final_processed_20k_candidates.csv ^
        --test test_sets/test_20k_Q1_2026_cand.csv ^
        --model-dir models/20k_cand_fix --features models/selected_features_20k_cand_fix.json ^
        --out models/20k_cand_fix/leak_screen_Q1.csv
Omit --model-dir/--features to run only the single-feature screen (no model needed).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from pms_model import (add_milestone_flags, build_inference_matrix, derive_inference_features,
                       load_artifacts, predict_proba, select_pms_delay_variant)

DEVICE = torch.device("cpu")


def prep(df, milestone):
    """Same derivation the train/score paths apply, so the screen sees what the model sees."""
    df = derive_inference_features(df)
    df = add_milestone_flags(df, milestone)
    df = select_pms_delay_variant(df)
    return df


def single_feature_auc(df, features, y):
    out = {}
    for f in features:
        if f not in df.columns:
            out[f] = np.nan
            continue
        x = pd.to_numeric(df[f], errors="coerce").fillna(0)
        if x.nunique() < 2:
            out[f] = np.nan  # constant -- no AUC
            continue
        out[f] = roc_auc_score(y, x)
    return out


def feature_groups(features):
    """The families that must be permuted together (redundancy hides per-column importance)."""
    sched = [f for f in features
             if f.startswith("has_") or f in ("PMS_Count_Prior", "PMS_Freq_PerYear", "PMS_Delay",
                                              "Years_Since_First_PMS", "Avg_Service_Interval_PMS")]
    vhc = [f for f in features if "VHC" in f or "Revenue" in f]
    status = [f for f in features if f.startswith("PMS Status_")]
    groups = {"schedule_keeping": sched, "vhc_revenue": vhc}
    if status:
        groups["pms_status_residue"] = status
    return {k: v for k, v in groups.items() if v}


def group_permutation(model, imputer, scaler, X, y, features, n_repeats=5, seed=0):
    base = roc_auc_score(y, predict_proba(model, imputer, scaler, X, DEVICE))
    rng = np.random.default_rng(seed)
    rows = []
    for name, cols in feature_groups(features).items():
        idx = [features.index(c) for c in cols if c in features]
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp.iloc[:, idx] = Xp.iloc[perm, idx].to_numpy()
            deltas.append(base - roc_auc_score(y, predict_proba(model, imputer, scaler, Xp, DEVICE)))
        rows.append({"group": name, "n_cols": len(idx),
                     "importance_mean": float(np.mean(deltas)),
                     "importance_std": float(np.std(deltas))})
    return base, pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Leak screen: single-feature AUC + group permutation")
    p.add_argument("milestone", type=int)
    p.add_argument("--train", required=True, help="Training matrix CSV")
    p.add_argument("--test", required=True, help="Test cohort CSV")
    p.add_argument("--features", default=None, help="selected_features json (for a tagged run)")
    p.add_argument("--model-dir", default=None, help="If set, also run group permutation")
    p.add_argument("--out", default=None, help="Write the per-feature table here as CSV")
    args = p.parse_args()

    m = args.milestone
    if args.features:
        features = json.load(open(args.features))
    elif args.model_dir:
        sys.exit("--model-dir without --features would screen the wrong list for a tagged run")
    else:
        features = None

    tr = prep(pd.read_csv(args.train, low_memory=False), m)
    te = prep(pd.read_csv(args.test, low_memory=False), m)
    y_tr = tr["TargetFlag"].astype(int).values
    y_te = te["TargetFlag"].astype(int).values
    if features is None:
        # No model context: screen every numeric column common to both frames.
        skip = {"TargetFlag", "VIN"}
        features = [c for c in tr.columns if c in te.columns and c not in skip]

    auc_tr = single_feature_auc(tr, features, y_tr)
    auc_te = single_feature_auc(te, features, y_te)
    tbl = pd.DataFrame({"feature": features,
                        "auc_train": [auc_tr[f] for f in features],
                        "auc_test": [auc_te[f] for f in features]})
    tbl["gap"] = tbl["auc_test"] - tbl["auc_train"]
    tbl["flag"] = ""
    tbl.loc[(tbl["gap"] >= 0.10) & (tbl["auc_test"] >= 0.65), "flag"] = "FLAG_gap>=0.10"
    tbl.loc[(tbl["gap"] >= 0.20) & (tbl["auc_test"] >= 0.65), "flag"] = "STOP_gap>=0.20"
    inverted = (tbl["auc_train"] < 0.45) & (tbl["auc_test"] > 0.55)
    tbl.loc[inverted, "flag"] = (tbl.loc[inverted, "flag"] + "+INVERTED").str.lstrip("+")
    tbl = tbl.sort_values("gap", ascending=False)

    stop = tbl[tbl["flag"].str.startswith("STOP")]
    flags = tbl[tbl["flag"].str.startswith("FLAG")]
    print(f"\n=== G6 single-feature screen: {len(tbl)} features ===")
    print(f"STOP (gap>=0.20, test>=0.65): {len(stop)}")
    if len(stop):
        print(stop.head(20).to_string(index=False))
    print(f"FLAG (gap>=0.10, test>=0.65): {len(flags)}")
    if len(flags):
        print(flags.head(20).to_string(index=False))
    print(f"INVERTED (train<0.45, test>0.55): {int(inverted.sum())}")
    if inverted.sum():
        print(tbl[tbl['flag'].str.contains('INVERTED')].head(20).to_string(index=False))

    if args.out:
        tbl.to_csv(args.out, index=False)
        print(f"Per-feature table -> {args.out}")

    if args.model_dir:
        model, imputer, scaler, sel = load_artifacts(m, DEVICE, mdir=args.model_dir,
                                                     features_file=args.features)
        X, _ = build_inference_matrix(pd.read_csv(args.test, low_memory=False), m, sel)
        base, gp = group_permutation(model, imputer, scaler, X, y_te, sel)
        print(f"\n=== Group permutation (test AUC base {base:.4f}) ===")
        print("importance = base - permuted AUC; NEGATIVE means shuffling HELPED (harmful family)")
        print(gp.to_string(index=False))
        if args.out:
            gp_out = args.out.replace(".csv", "_groups.csv")
            gp.to_csv(gp_out, index=False)
            print(f"Group table -> {gp_out}")

    print(f"\nG6 verdict: {'STOP' if len(stop) else ('FLAG' if len(flags) else 'PASS')}")
    sys.exit(2 if len(stop) else 0)


if __name__ == "__main__":
    main()
