"""Did the v3 behaviour features help? v2 vs v3 on held-out Q3 2026, same cars, thresholds from Q1+Q2.
Run from repo root:  ./venv/Scripts/python.exe dprime_v2/compare_v2_v3.py 20 30 70 80

For tree, NN and their average (seed-averaged): Q3 AUC; the BALANCE point (threshold where
precision ~= recall on 2026 Q1+Q2, applied to Q3); the best-F1 threshold (also from Q1+Q2, applied to
Q3). Recall is reported in the list and against ALL Q3 visitors (distinct VINs in the Q3 file).
Also checks the v2 and v3 Q3 lists hold exactly the same cars.
Writes dprime_v2/v3/compare_v2_v3_{tag}.csv.
"""
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from date_utils import parse_dates
from features import resolve_service_num

MILESTONES = [int(x) for x in sys.argv[1:]] or [20, 30, 70, 80]
TAG = "_".join(f"{m}k" for m in MILESTONES)
THR = np.round(np.arange(0.01, 1.0, 0.005), 3)

q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)


def load(pattern):
    sc = pd.concat([pd.read_csv(f) for f in glob.glob(pattern)])
    sc = sc.groupby(["milestone", "model", "quarter", "VIN"], as_index=False).agg(
        prob=("prob", "mean"), came_in=("came_in", "first"))
    w = sc.pivot_table(index=["milestone", "quarter", "VIN", "came_in"], columns="model", values="prob").reset_index()
    w["ensemble"] = (w["tree"] + w["nn"]) / 2
    return w


def stats(y, p, t):
    fl = p >= t
    tp = int((fl & (y == 1)).sum())
    fp = int((fl & (y == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / y.sum()
    return P, R, tp, (2 * P * R / (P + R) if P + R else 0.0)


def pick(y, p, how):
    best, best_t = None, 0.5
    for t in THR:
        P, R, _, f = stats(y, p, t)
        score = -abs(P - R) if how == "balance" else f
        if P > 0 and (best is None or score > best):
            best, best_t = score, t
    return best_t


W = {"v2": load("dprime_v2/scores_*.csv"), "v3": load("dprime_v2/v3/scores_*.csv")}
rows = []
for m in MILESTONES:
    mk = f"{m}k"
    all_actual = q3.loc[q3["Service_Num"] == m, "Vin_No"].nunique()
    sets = {v: set(W[v].loc[(W[v]["milestone"] == mk) & (W[v]["quarter"] == "Q3"), "VIN"]) for v in W}
    assert sets["v2"] == sets["v3"], f"{mk}: v2 and v3 Q3 lists differ"
    for v, w in W.items():
        w12 = w[(w["milestone"] == mk) & w["quarter"].isin(["Q1", "Q2"])]
        w3 = w[(w["milestone"] == mk) & (w["quarter"] == "Q3")]
        y3 = w3["came_in"].values
        for kind in ["tree", "nn", "ensemble"]:
            r = {"milestone": mk, "set": v, "model": kind, "q3_auc": round(roc_auc_score(y3, w3[kind]), 4)}
            for how in ["balance", "best_f1"]:
                t = pick(w12["came_in"].values, w12[kind].values, how)
                P, R, tp, f = stats(y3, w3[kind].values, t)
                r.update({f"{how}_thr": t, f"{how}_precision": round(P, 3), f"{how}_recall_in_list": round(R, 3),
                          f"{how}_recall_of_all": round(tp / all_actual, 3), f"{how}_f1": round(f, 3)})
            r["list_ceiling_of_all"] = round(y3.sum() / all_actual, 3)
            rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(f"dprime_v2/v3/compare_v2_v3_{TAG}.csv", index=False)
show = ["milestone", "set", "model", "q3_auc", "balance_thr", "balance_precision", "balance_recall_in_list",
        "balance_recall_of_all", "best_f1_precision", "best_f1_recall_in_list", "best_f1_recall_of_all",
        "best_f1_f1", "list_ceiling_of_all"]
print(res[show].to_string(index=False))
