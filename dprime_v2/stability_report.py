"""Does the D' v2 strategy hold, quarter after quarter and milestone after milestone?
Run from repo root (after build_lists / build_matrix / train_models for the milestones):
    PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/stability_report.py 20 30 70 80

Checks each layer separately:
  A. the SAMPLING (D' + prev-step lists, lists_{m}k.csv) per quarter 2021Q1..2026Q3: list size,
     early-completer share, list precision (came in / not-early listed) and list recall (came in /
     every first-ever m visit in that quarter by a gated car). Outcome is the strict quarter.
  B. the TRAINING DATA built on it (final_processed_{m}k_dprime_v2.csv): rows and positive rate.
  C. the MODELS per quarter: seed-averaged NN and tree scored on every training-matrix quarter and on
     2026 Q1/Q2/Q3; AUC by quarter with each quarter's role (train / val / test -- the chronological
     80/20 split is reproduced exactly as train_models.py does it), mean predicted probability vs the
     real came-in rate (calibration), best-F1 threshold per test quarter.
  D. SEED spread per test quarter (train_summary_*.csv) and the FEATURE checks (leak screen flags,
     feature counts after pruning).
Writes dprime_v2/stability_{sampling,auc_by_quarter,thresholds}_{tag}.csv.
"""
import glob
import json
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from date_utils import parse_dates
from features import resolve_service_num
from pms_model import BinaryClassifier
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONES = [int(x) for x in sys.argv[1:]] or [20, 30, 70, 80]
TAG = "_".join(f"{m}k" for m in MILESTONES)
SEEDS = [42, 7, 123]
HEAD = ["VIN", "TargetFlag", "CameInQuarter", "CohortQuarter"]
THR = np.round(np.arange(0.01, 1.0, 0.01), 2)

cols = ["Vin_No", "Service_Date", "Service_Code", "Description"]
serv = pd.read_csv("data/Service_History_Q2-2026.csv", low_memory=False, encoding="ISO-8859-1", usecols=cols)
q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1", usecols=cols)
hist = pd.concat([serv, q3], ignore_index=True)
hist["Service_Date"] = parse_dates(hist["Service_Date"])
hist["Service_Num"] = resolve_service_num(hist)
eda = pd.read_csv("data/EDA_Q2-2026.csv", low_memory=False, encoding="ISO-8859-1", usecols=["VIN", "Last Service - PMS"])
gate = set(eda.loc[eda["Last Service - PMS"] != "-", "VIN"])


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


def best_f1_thr(y, p):
    best = (0, 0.5)
    for t in THR:
        fl = p >= t
        tp = (fl & (y == 1)).sum()
        if fl.sum() and tp:
            pr, rc = tp / fl.sum(), tp / y.sum()
            best = max(best, (2 * pr * rc / (pr + rc), t))
    return best[1], best[0]


samp, aucq, thr_rows = [], [], []
for m in MILESTONES:
    # ---- A. sampling ----
    first_m = hist[hist["Service_Num"] == m].groupby("Vin_No")["Service_Date"].min()
    first_m = first_m[first_m.index.isin(gate)]
    lst = pd.read_csv(f"dprime_v2/lists_{m}k.csv", parse_dates=["q_start", "q_end"])
    for cq, b in lst.groupby("CohortQuarter"):
        qs, qe = b["q_start"].iloc[0], b["q_end"].iloc[0]
        ne = b[b["EarlyCompleter"] == 0]
        total = int(((first_m >= qs) & (first_m <= qe)).sum())
        came = int(ne["CameInQuarter"].sum())
        samp.append({"milestone": f"{m}k", "quarter": cq, "listed_all": len(b),
                     "early_share_pct": round(100 * b["EarlyCompleter"].mean(), 1),
                     "listed_not_early": len(ne), "came_in": came, "completers_in_gate": total,
                     "list_precision_pct": round(100 * came / len(ne), 1),
                     "list_recall_pct": round(100 * came / total, 1) if total else np.nan})

    # ---- C. models per quarter ----
    tr = pd.read_csv(f"refactored_test_dir/final_processed_{m}k_dprime_v2.csv")
    due = f"Next{m}K_Due"
    tr = tr.assign(_d=parse_dates(tr[due])).sort_values("_d").reset_index(drop=True)
    feats = [c for c in tr.columns if c not in HEAD + [due, "_d"]]
    X = tr[feats].astype(float)
    keep = IsolationForest(contamination=0.005, random_state=42).fit_predict(X.fillna(X.median())) == 1
    pos = np.full(len(tr), -1)
    pos[keep] = np.arange(keep.sum())
    cut = int(keep.sum() * 0.8)
    tr["role"] = np.where(~keep, "outlier", np.where(pos < cut, "train", "val"))
    tests = {f"2026{q}": pd.read_csv(f"test_sets/test_{m}k_{q}_2026_dprime_v2.csv") for q in ["Q1", "Q2", "Q3"]}
    frames = [tr.drop(columns="_d")] + [t.assign(role="test") for t in tests.values()]
    allr = pd.concat(frames, ignore_index=True)
    XA = allr[feats].astype(float)
    for kind in ["nn", "tree"]:
        ps = []
        for s in SEEDS:
            d = f"models/{m}k_dprime_v2_{kind}_s{s}"
            ps.append(joblib.load(f"{d}/model.joblib").predict_proba(XA)[:, 1] if kind == "tree"
                      else nn_predictor(d)(XA))
        allr[f"p_{kind}"] = np.mean(ps, axis=0)
    for cq, b in allr[allr["role"] != "outlier"].groupby("CohortQuarter"):
        roles = b["role"].value_counts()
        role = roles.index[0] if len(roles) == 1 else "/".join(sorted(roles.index))
        y = b["CameInQuarter"].values
        row = {"milestone": f"{m}k", "quarter": cq, "role": role, "rows": len(b),
               "came_in_rate": round(y.mean(), 3)}
        for kind in ["nn", "tree"]:
            p = b[f"p_{kind}"].values
            row[f"auc_{kind}"] = round(roc_auc_score(y, p), 4) if 0 < y.sum() < len(y) else np.nan
            row[f"mean_prob_{kind}"] = round(p.mean(), 3)
        aucq.append(row)
        if role == "test":
            for kind in ["nn", "tree"]:
                t, f = best_f1_thr(y, b[f"p_{kind}"].values)
                thr_rows.append({"milestone": f"{m}k", "quarter": cq, "model": kind,
                                 "best_f1_threshold": t, "best_f1": round(f, 3)})
    print(f"{m}k done")

S = pd.DataFrame(samp)
A = pd.DataFrame(aucq)
T = pd.DataFrame(thr_rows)
S.to_csv(f"dprime_v2/stability_sampling_{TAG}.csv", index=False)
A.to_csv(f"dprime_v2/stability_auc_by_quarter_{TAG}.csv", index=False)
T.to_csv(f"dprime_v2/stability_thresholds_{TAG}.csv", index=False)

print("\n=== A. sampling: list precision / recall per quarter (strict quarter, gated completers) ===")
print(S.pivot(index="quarter", columns="milestone", values=["list_precision_pct", "list_recall_pct"]).to_string())
print("\nsummary over 2021Q1..2026Q3:")
print(S.groupby("milestone")[["listed_not_early", "early_share_pct", "list_precision_pct", "list_recall_pct"]]
      .agg(["mean", "min", "max", "std"]).round(1).to_string())

print("\n=== C. model AUC by quarter (seed-averaged; role = train / val / test) ===")
for m in MILESTONES:
    a = A[A["milestone"] == f"{m}k"]
    print(f"\n{m}k")
    print(a[["quarter", "role", "rows", "came_in_rate", "auc_nn", "auc_tree", "mean_prob_nn", "mean_prob_tree"]]
          .to_string(index=False))
print("\nAUC summary by role:")
print(A.groupby(["milestone", "role"])[["auc_nn", "auc_tree"]].agg(["mean", "min", "max"]).round(3).to_string())
print("\n=== best-F1 threshold per test quarter (in-quarter, to see whether it drifts) ===")
print(T.pivot_table(index=["milestone", "model"], columns="quarter", values=["best_f1_threshold", "best_f1"]).to_string())

print("\n=== D. seed spread on each test quarter (max - min AUC over 3 seeds; G7 rule < 0.04) ===")
ts = pd.concat([pd.read_csv(f) for f in glob.glob("dprime_v2/train_summary_*.csv")])
ts = ts[ts["milestone"].isin([f"{m}k" for m in MILESTONES])]
spread = ts.groupby(["milestone", "model"])[["val_auc", "auc_Q1", "auc_Q2", "auc_Q3"]].agg(lambda x: x.max() - x.min())
print(spread.round(4).to_string())
for m in MILESTONES:
    pr = json.load(open(f"dprime_v2/pruned_{m}k.json"))
    ls = pd.read_csv(f"dprime_v2/leak_screen_{m}k.csv")
    print(f"{m}k features: kept {len(pr['kept'])}, constant {len(pr['constant'])}, near-duplicate "
          f"{len(pr['near_duplicate_of'])}; leak screen flags: {int((ls['sign_flip'] | (ls['gap'] >= 0.10)).sum())}")
