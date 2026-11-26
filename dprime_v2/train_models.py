"""Step 5: train the NN (retrain.py's recipe, best-epoch fix included) and a gradient-boosted tree on
the D' v2 matrices, 3 seeds each, then score 2026 Q1 / Q2 / Q3. Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/train_models.py 20 30

Kept separate from retrain.py on purpose: the v2 matrix needs none of retrain.py's derivation steps
(add_milestone_flags / derive_inference_features / PMS_Delay variants / test-schema intersection),
and a separate script leaves the production training path untouched.

NN recipe = retrain.py: rows sorted by Next{m}K_Due, IsolationForest(0.005) outlier drop,
chronological 80/20 split, median imputer, RobustScaler (tiny scales clamped to 1), pms_model
BinaryClassifier, BCE(pos_weight) on 0.1-smoothed labels + 0.5 * Brier, AdamW lr 3e-4 wd 1e-2,
ReduceLROnPlateau, batch 128, 100 epochs, patience 10, BEST epoch kept (deep copy).
Tree = HistGradientBoostingClassifier on the same train split (no scaling needed, NaN native).
Label = TargetFlag (came in during the quarter or within 15 days). Test metrics use the STRICT
quarter outcome (CameInQuarter), as every table in the investigation does.
Models -> models/{m}k_dprime_v2_{nn|tree}_s{seed}/; scores -> dprime_v2/scores_{tag}.csv,
summary -> dprime_v2/train_summary_{tag}.csv (tag = milestones run, e.g. 20k_30k).
--set v3 trains on the v3 matrices (v2 + behaviour) -> models/{m}k_dprime_v3_*, outputs in dprime_v2/v3/.
"""
import copy
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from date_utils import parse_dates
from pms_model import BinaryClassifier

ARGV = sys.argv[1:]
SET = ARGV[ARGV.index("--set") + 1] if "--set" in ARGV else "v2"   # v2 | v3 (v2 + behaviour)
MILESTONES = [int(x) for x in ARGV if x.isdigit()] or [20, 30]
OUT = "dprime_v2/" if SET == "v2" else f"dprime_v2/{SET}/"   # v3 outputs kept apart from v2 globs
TAG = "_".join(f"{m}k" for m in MILESTONES)   # output names, so runs for other milestones never overwrite
SEEDS = [42, 7, 123]
HEAD = ["VIN", "TargetFlag", "CameInQuarter", "CohortQuarter"]


def train_nn(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = RobustScaler().fit(imp.transform(Xtr))
    sc.scale_ = np.where(sc.scale_ < 1e-6, 1.0, sc.scale_)
    tr = TensorDataset(torch.FloatTensor(sc.transform(imp.transform(Xtr))), torch.FloatTensor(ytr))
    va = TensorDataset(torch.FloatTensor(sc.transform(imp.transform(Xva))), torch.FloatTensor(yva))
    tl, vl = DataLoader(tr, batch_size=128, shuffle=True), DataLoader(va, batch_size=128)
    model = BinaryClassifier(input_dim=Xtr.shape[1])
    pw = torch.tensor([(ytr == 0).sum() / ((ytr == 1).sum() + 1e-6)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    def loss_fn(logits, y):
        return crit(logits, y * 0.9 + 0.05) + 0.5 * torch.mean((torch.sigmoid(logits) - y) ** 2)

    best, best_state, wait = np.inf, None, 0
    for _ in range(100):
        model.train()
        for xb, yb in tl:
            opt.zero_grad()
            loss_fn(model(xb).squeeze(-1), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = np.mean([loss_fn(model(xb).squeeze(-1), yb).item() for xb, yb in vl])
        sched.step(vloss)
        if vloss < best:
            best, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= 10:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(model(torch.FloatTensor(sc.transform(imp.transform(X)))).squeeze(-1)).numpy()
    return model, imp, sc, predict


os.makedirs(OUT, exist_ok=True)
all_scores, summary = [], []
for m in MILESTONES:
    tr = pd.read_csv(f"refactored_test_dir/final_processed_{m}k_dprime_{SET}.csv")
    due = f"Next{m}K_Due"
    tr = tr.assign(_d=parse_dates(tr[due])).sort_values("_d").drop(columns="_d").reset_index(drop=True)
    feats = [c for c in tr.columns if c not in HEAD + [due]]
    tests = {q: pd.read_csv(f"test_sets/test_{m}k_{q}_2026_dprime_{SET}.csv") for q in ["Q1", "Q2", "Q3"]}
    X, y = tr[feats].astype(float), tr["TargetFlag"].values.astype(float)
    keep = IsolationForest(contamination=0.005, random_state=42).fit_predict(X.fillna(X.median())) == 1
    X, y = X[keep].reset_index(drop=True), y[keep]
    cut = int(len(X) * 0.8)
    Xtr, ytr, Xva, yva = X.iloc[:cut], y[:cut], X.iloc[cut:], y[cut:]
    print(f"\n=== {m}k: {len(feats)} features | train {len(Xtr):,} / val {len(Xva):,} "
          f"(dropped {int((~keep).sum())} outliers) | positive {ytr.mean():.1%} ===")

    for seed in SEEDS:
        for kind in ["nn", "tree"]:
            out_dir = f"models/{m}k_dprime_{SET}_{kind}_s{seed}"
            os.makedirs(out_dir, exist_ok=True)
            if kind == "nn":
                model, imp, sc, predict = train_nn(Xtr, ytr, Xva, yva, seed)
                torch.save(model.state_dict(), f"{out_dir}/best_model.pt")
                joblib.dump(imp, f"{out_dir}/imputer.joblib")
                joblib.dump(sc, f"{out_dir}/scaler.joblib")
            else:
                clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                                     l2_regularization=1.0, early_stopping=True,
                                                     validation_fraction=0.1, n_iter_no_change=30,
                                                     random_state=seed).fit(Xtr, ytr)
                joblib.dump(clf, f"{out_dir}/model.joblib")
                predict = lambda Z, c=clf: c.predict_proba(Z)[:, 1]  # noqa: E731
            json.dump(feats, open(f"{out_dir}/selected_features.json", "w"))
            val_auc = roc_auc_score(yva, predict(Xva))
            line = {"milestone": f"{m}k", "model": kind, "seed": seed, "val_auc": round(val_auc, 4)}
            for q, t in tests.items():
                p = predict(t[feats].astype(float))
                line[f"auc_{q}"] = round(roc_auc_score(t["CameInQuarter"], p), 4)
                all_scores.append(pd.DataFrame({"milestone": f"{m}k", "model": kind, "seed": seed,
                                                "quarter": q, "VIN": t["VIN"], "prob": p,
                                                "came_in": t["CameInQuarter"]}))
            summary.append(line)
            print(line)

pd.concat(all_scores).to_csv(f"{OUT}scores_{TAG}.csv", index=False)
s = pd.DataFrame(summary)
s.to_csv(f"{OUT}train_summary_{TAG}.csv", index=False)
print("\n", s.groupby(["milestone", "model"])[["val_auc", "auc_Q1", "auc_Q2", "auc_Q3"]]
      .agg(["mean", "min", "max"]).round(4).to_string())
