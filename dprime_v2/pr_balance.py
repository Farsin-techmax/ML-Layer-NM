"""How close can the D' v2 models get to precision = recall = 80%? Read-only, from saved scores.
Run from repo root:  ./venv/Scripts/python.exe dprime_v2/pr_balance.py 20 30 70 80

For tree, NN and their average (ensemble), seed-averaged:
  1. balance point: the threshold where precision ~= recall (in-list), picked on 2026 Q1+Q2, applied
     to Q3 -> Q3 precision / recall / recall of ALL Q3 actuals. Also the in-sample Q3 balance point.
  2. what 80/80 needs: with base rate b, precision 80% at recall 80% means a false-positive rate of
     0.2*b/(1-b). The model's actual Q3 recall at that false-positive rate shows the gap.
  3. the list ceiling: recall of all Q3 actuals if everyone on the list is called.
Writes dprime_v2/pr_balance_{tag}.csv.
"""
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from date_utils import parse_dates
from features import resolve_service_num

MILESTONES = [int(x) for x in sys.argv[1:]] or [20, 30, 70, 80]
TAG = "_".join(f"{m}k" for m in MILESTONES)
THR = np.round(np.arange(0.01, 1.0, 0.005), 3)

q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)
sc = pd.concat([pd.read_csv(f) for f in glob.glob("dprime_v2/scores_*.csv")])
sc = sc.groupby(["milestone", "model", "quarter", "VIN"], as_index=False).agg(prob=("prob", "mean"),
                                                                                came_in=("came_in", "first"))
wide = sc.pivot_table(index=["milestone", "quarter", "VIN", "came_in"], columns="model", values="prob").reset_index()
wide["ensemble"] = (wide["tree"] + wide["nn"]) / 2


def pr(y, p, t):
    fl = p >= t
    tp = (fl & (y == 1)).sum()
    return (tp / fl.sum() if fl.sum() else 0.0), tp / y.sum(), tp


def balance_thr(y, p):
    gaps = []
    for t in THR:
        P, R, _ = pr(y, p, t)
        gaps.append(abs(P - R) if P > 0 else 9)
    return THR[int(np.argmin(gaps))]


rows = []
for m in MILESTONES:
    mk = f"{m}k"
    all_actual = q3.loc[q3["Service_Num"] == m, "Vin_No"].nunique()
    w12 = wide[(wide["milestone"] == mk) & wide["quarter"].isin(["Q1", "Q2"])]
    w3 = wide[(wide["milestone"] == mk) & (wide["quarter"] == "Q3")]
    y3 = w3["came_in"].values
    b = y3.mean()
    fpr_needed = 0.2 * b / (1 - b)
    for kind in ["tree", "nn", "ensemble"]:
        t = balance_thr(w12["came_in"].values, w12[kind].values)
        P, R, tp = pr(y3, w3[kind].values, t)
        t3 = balance_thr(y3, w3[kind].values)
        P3, R3, _ = pr(y3, w3[kind].values, t3)
        fpr, tpr, _ = roc_curve(y3, w3[kind].values)
        tpr_at = float(np.interp(fpr_needed, fpr, tpr))
        rows.append({"milestone": mk, "model": kind, "q3_base_rate": round(b, 3),
                     "balance_thr_from_Q1Q2": t, "q3_precision": round(P, 3), "q3_recall_in_list": round(R, 3),
                     "q3_recall_of_all_actuals": round(tp / all_actual, 3),
                     "q3_in_sample_balance": round((P3 + R3) / 2, 3),
                     "fpr_needed_for_80_80": round(fpr_needed, 3),
                     "recall_at_that_fpr_now": round(tpr_at, 3),
                     "list_ceiling_recall_of_all": round(y3.sum() / all_actual, 3)})
res = pd.DataFrame(rows)
res.to_csv(f"dprime_v2/pr_balance_{TAG}.csv", index=False)
print(res.to_string(index=False))
