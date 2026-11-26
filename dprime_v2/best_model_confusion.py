"""Confusion counts for the best D' v2 model per milestone on held-out Q3 2026.
Run from repo root:  ./venv/Scripts/python.exe dprime_v2/best_model_confusion.py 20 30 70 80

Honest procedure (nothing is tuned on Q3):
  1. For the NN and the tree (each averaged over 3 seeds), pick the threshold that maximises F1 on
     2026 Q1 + Q2 combined (both quarters are outside training; training ends 2025Q4).
  2. "Best model" = the one with the higher Q1+Q2 F1 at its threshold.
  3. Apply that model + threshold to Q3 2026 once and count TP / FP / FN / TN on the D' list.
Recall is given two ways: within the list (TP / listed cars that came in) and against EVERY car that
came in for the service in Q3 (Q3/Service_History_Q3-2026.csv, distinct VINs, incl. cars the list
never caught and cars outside the EDA gate).
Reference rows: contact everyone on the list; production NN at its shipped 0.5.
Writes dprime_v2/best_model_confusion_{tag}.csv.
"""
import glob
import sys

import numpy as np
import pandas as pd

from date_utils import parse_dates
from features import resolve_service_num

MILESTONES = [int(x) for x in sys.argv[1:]] or [20, 30, 70, 80]
TAG = "_".join(f"{m}k" for m in MILESTONES)
THR = np.round(np.arange(0.01, 1.0, 0.01), 2)

q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)
sc = pd.concat([pd.read_csv(f) for f in glob.glob("dprime_v2/scores_*.csv")])
sc = sc.groupby(["milestone", "model", "quarter", "VIN"], as_index=False).agg(prob=("prob", "mean"),
                                                                                came_in=("came_in", "first"))


def counts(y, flag):
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    return tp, fp, fn, tn


def f1_at(y, p, t):
    tp, fp, fn, _ = counts(y, p >= t)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


rows = []
for m in MILESTONES:
    mk = f"{m}k"
    all_actual = q3.loc[q3["Service_Num"] == m, "Vin_No"].nunique()
    pick = {}
    for kind in ["tree", "nn"]:
        d = sc[(sc["milestone"] == mk) & (sc["model"] == kind) & (sc["quarter"].isin(["Q1", "Q2"]))]
        y, p = d["came_in"].values, d["prob"].values
        f = [f1_at(y, p, t) for t in THR]
        pick[kind] = (THR[int(np.argmax(f))], max(f))
    best = max(pick, key=lambda k: pick[k][1])

    def add(name, y, flag, thr, chosen=""):
        tp, fp, fn, tn = counts(y, flag)
        rows.append({"milestone": mk, "model": name, "chosen_on_Q1Q2": chosen, "threshold": thr,
                     "listed": len(y), "flagged_TP+FP": tp + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "precision_TP/(TP+FP)": round(tp / (tp + fp), 3) if tp + fp else np.nan,
                     "recall_in_list_TP/(TP+FN)": round(tp / (tp + fn), 3),
                     "all_Q3_actuals": all_actual,
                     "recall_of_all_actuals": round(tp / all_actual, 3),
                     "f1": round(2 * tp / (2 * tp + fp + fn), 3)})

    for kind in ["tree", "nn"]:
        d = sc[(sc["milestone"] == mk) & (sc["model"] == kind) & (sc["quarter"] == "Q3")]
        t, f = pick[kind]
        add(f"new {kind}", d["came_in"].values, d["prob"].values >= t, t,
            f"BEST (Q1+Q2 F1 {f:.3f})" if kind == best else f"Q1+Q2 F1 {f:.3f}")
    prod = pd.read_csv(f"q1_due_rule_experiment/q3_dprime_scored_{mk}.csv")
    add("production NN", prod["came_in"].values, prod["prob"].values > 0.5, 0.5, "shipped")
    add("contact everyone on the list", prod["came_in"].values, np.ones(len(prod), bool), 0.0, "-")

res = pd.DataFrame(rows)
res.to_csv(f"dprime_v2/best_model_confusion_{TAG}.csv", index=False)
print(res.to_string(index=False))
