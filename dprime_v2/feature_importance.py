"""Feature importance for the D' v2 models, measured on the HELD-OUT Q3 2026 list (honest data the
models never saw). Run from repo root:  PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/feature_importance.py

Permutation importance = how much Q3 AUC drops when one feature's values are shuffled (5 shuffles,
averaged), for the tree and the NN, each averaged over the 3 seeds. Because correlated features cover
for each other when shuffled one at a time (CLAUDE.md: "read groups, not columns"), a second table
shuffles whole FAMILIES together.
Writes dprime_v2/feature_importance_{m}k.csv and dprime_v2/feature_importance_groups_{tag}.csv
(milestones from the command line, default 20 30; the first 20k/30k run wrote feature_importance_groups.csv).
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from pms_model import BinaryClassifier

SEEDS, REPEATS = [42, 7, 123], 5
MILESTONES = [int(x) for x in sys.argv[1:]] or [20, 30]
TAG = "_".join(f"{m}k" for m in MILESTONES)
GROUPS = {
    "list context (which method listed it, leg dates)": lambda c: c.startswith(("by_", "n_legs", "centre_days", "gap"))
    or c.endswith("_days_from_qmid"),
    "PMS history & rhythm": lambda c: c.startswith(("has_", "n_pms", "pms_", "rhythm", "days_since_last_pms")),
    "mileage": lambda c: c.startswith(("last_km", "km_")),
    "visit recency & activity": lambda c: c in ("visits_total", "visits_365d", "days_since_any_visit", "age_days",
                                                "invoice_to_first_visit_days", "n_keys"),
    "non-PMS visits & revenue": lambda c: c.startswith(("n_nonpms", "days_since_last_nonpms", "nonpms_", "rev_",
                                                        "tyre_", "battery_", "brake_")),
    "branch / bodyshop / survey": lambda c: c.startswith(("n_branches", "share_main", "main_branch", "bodyshop",
                                                          "survey", "n_detractor")),
    "VHC": lambda c: c.startswith(("vhc_", "has_vhc", "days_since_vhc")),
    "appointments": lambda c: c.startswith(("appt_", "has_appt", "days_since_last_booking", "upcoming_")),
    "vehicle & customer facts (EDA)": lambda c: True,
}


def family(c):
    for g, f in GROUPS.items():
        if f(c):
            return g


def nn_predictor(d):
    feats = json.load(open(f"{d}/selected_features.json"))
    imp, sc = joblib.load(f"{d}/imputer.joblib"), joblib.load(f"{d}/scaler.joblib")
    model = BinaryClassifier(input_dim=len(feats))
    model.load_state_dict(torch.load(f"{d}/best_model.pt"))
    model.eval()

    def p(X):
        with torch.no_grad():
            return torch.sigmoid(model(torch.FloatTensor(sc.transform(imp.transform(X)))).squeeze(-1)).numpy()
    return p


rng = np.random.default_rng(0)
group_rows = []
for m in MILESTONES:
    t = pd.read_csv(f"test_sets/test_{m}k_Q3_2026_dprime_v2.csv")
    feats = json.load(open(f"models/{m}k_dprime_v2_tree_s42/selected_features.json"))
    X, y = t[feats].astype(float).reset_index(drop=True), t["CameInQuarter"].values
    preds = {}
    for kind in ["tree", "nn"]:
        fns = []
        for s in SEEDS:
            d = f"models/{m}k_dprime_v2_{kind}_s{s}"
            fns.append((lambda Z, c=joblib.load(f"{d}/model.joblib"): c.predict_proba(Z)[:, 1]) if kind == "tree"
                       else nn_predictor(d))
        preds[kind] = lambda Z, fns=fns: np.mean([f(Z) for f in fns], axis=0)

    def drop(kind, cols):
        base = roc_auc_score(y, preds[kind](X))
        out = []
        for _ in range(REPEATS):
            Z = X.copy()
            idx = rng.permutation(len(Z))
            Z[cols] = Z[cols].values[idx]
            out.append(base - roc_auc_score(y, preds[kind](Z)))
        return float(np.mean(out))

    rows = []
    for c in feats:
        rows.append({"feature": c, "family": family(c), "tree_auc_drop": drop("tree", [c]),
                     "nn_auc_drop": drop("nn", [c])})
    r = pd.DataFrame(rows)
    r["mean_auc_drop"] = r[["tree_auc_drop", "nn_auc_drop"]].mean(axis=1)
    r = r.sort_values("mean_auc_drop", ascending=False).reset_index(drop=True)
    r.index += 1
    r.round(4).to_csv(f"dprime_v2/feature_importance_{m}k.csv", index_label="rank")
    for g in GROUPS:
        cols = [c for c in feats if family(c) == g]
        if cols:
            group_rows.append({"milestone": f"{m}k", "family": g, "n_features": len(cols),
                               "tree_auc_drop": drop("tree", cols), "nn_auc_drop": drop("nn", cols)})
    print(f"\n=== {m}k: Q3 AUC tree {roc_auc_score(y, preds['tree'](X)):.4f} / NN {roc_auc_score(y, preds['nn'](X)):.4f}"
          f" -- top 25 single features (AUC drop when shuffled) ===")
    print(r.head(25).round(4).to_string())
    neg = r[r["mean_auc_drop"] < 0]
    print(f"   {len(r[r['mean_auc_drop'] <= 0.0005])} of {len(r)} features change Q3 AUC by <= 0.0005 alone; "
          f"{len(neg)} slightly negative")
g = pd.DataFrame(group_rows).round(4)
g.to_csv(f"dprime_v2/feature_importance_groups_{TAG}.csv", index=False)
print("\n=== whole families shuffled together ===")
print(g.sort_values(["milestone", "tree_auc_drop"], ascending=[True, False]).to_string(index=False))
