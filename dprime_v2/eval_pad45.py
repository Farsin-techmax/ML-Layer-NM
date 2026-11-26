"""Exact-quarter list (v2) vs the +-45-day list (v2_pad45): how much of ALL visitors can we recall while
staying precise? Read-only, from saved scores. Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/eval_pad45.py 20 30 70 80

Outcome = came in during the STRICT quarter. Recall of ALL = flagged cars that came in / every distinct
car with a visit for that service in the quarter (Q3: the Q3 file; Q1/Q2: base history) -- cars the list
never held count as misses. Precision = TP / (TP + FP) among flagged cars.
Every threshold is chosen on 2026 Q1 + Q2 (pooled) and applied ONCE to Q3:
  balance     the equilibrium: maximise min(precision, recall of all)
  recall>=T   highest precision while recall of all >= T on Q1+Q2   (T = 0.80; 20k also 0.70 / 0.75)
  precision>=T  highest recall of all while precision >= T          (T = 0.80; 20k also 0.70)
Models: tree, NN and their average, each seed-averaged. Writes dprime_v2/v2_pad45/eval_pad45_{tag}.csv.
"""
import glob
import sys

import numpy as np
import pandas as pd

from date_utils import parse_dates
from features import resolve_service_num
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONES = [int(x) for x in sys.argv[1:]] or [20, 30, 70, 80]
TAG = "_".join(f"{m}k" for m in MILESTONES)
THR = np.round(np.arange(0.005, 1.0, 0.005), 3)
QUARTERS = {"Q1": (2026, 1), "Q2": (2026, 2), "Q3": (2026, 3)}

cols = ["Vin_No", "Service_Date", "Service_Code", "Description"]
base = pd.read_csv("data/Service_History_Q2-2026.csv", low_memory=False, encoding="ISO-8859-1", usecols=cols)
q3f = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1", usecols=cols)
for d in (base, q3f):
    d["Service_Date"] = parse_dates(d["Service_Date"])
    d["Service_Num"] = resolve_service_num(d)


def all_visitors(m, q):
    qs, qe = quarter_bounds(*QUARTERS[q])
    src = q3f if q == "Q3" else base
    w = src[(src["Service_Num"] == m) & (src["Service_Date"] >= qs) & (src["Service_Date"] <= qe)]
    return w["Vin_No"].nunique()


def load(pattern):
    sc = pd.concat([pd.read_csv(f) for f in glob.glob(pattern)])
    sc = sc.groupby(["milestone", "model", "quarter", "VIN"], as_index=False).agg(
        prob=("prob", "mean"), came_in=("came_in", "first"))
    w = sc.pivot_table(index=["milestone", "quarter", "VIN", "came_in"], columns="model", values="prob").reset_index()
    w["ensemble"] = (w["tree"] + w["nn"]) / 2
    return w


def curve(y, p, denom):
    """precision and recall-of-all at every threshold."""
    out = []
    for t in THR:
        fl = p >= t
        tp = int((fl & (y == 1)).sum())
        n = int(fl.sum())
        out.append((t, tp / n if n else 0.0, tp / denom, tp, n))
    return out


def choose(cv, rule, target=None):
    if rule == "balance":   # equilibrium: the threshold where the LOWER of the two is highest
        ok = [c for c in cv if c[4] > 0]
        return max(ok, key=lambda c: min(c[1], c[2]))[0]
    if rule == "recall":
        ok = [c for c in cv if c[2] >= target and c[4] > 0]
        return max(ok, key=lambda c: (c[1], -c[0]))[0] if ok else None
    if rule == "precision":
        ok = [c for c in cv if c[1] >= target]
        return max(ok, key=lambda c: (c[2], c[0]))[0] if ok else None


W = {"exact quarter (v2)": load("dprime_v2/scores_*.csv"), "+-45 days (v2_pad45)": load("dprime_v2/v2_pad45/scores_*.csv")}
rows = []
for m in MILESTONES:
    mk = f"{m}k"
    den = {q: all_visitors(m, q) for q in QUARTERS}
    rules = [("balance", "balance", None), ("recall>=80%", "recall", 0.80), ("precision>=80%", "precision", 0.80)]
    if m == 20:
        rules += [("recall>=70%", "recall", 0.70), ("recall>=75%", "recall", 0.75), ("precision>=70%", "precision", 0.70)]
    for setname, w in W.items():
        w12 = w[(w["milestone"] == mk) & w["quarter"].isin(["Q1", "Q2"])]
        w3 = w[(w["milestone"] == mk) & (w["quarter"] == "Q3")]
        y3 = w3["came_in"].values
        for kind in ["tree", "nn", "ensemble"]:
            cv12 = curve(w12["came_in"].values, w12[kind].values, den["Q1"] + den["Q2"])
            for label, rule, target in rules:
                t = choose(cv12, rule, target)
                r = {"milestone": mk, "list": setname, "model": kind, "rule": label,
                     "listed_q3": len(y3), "list_ceiling_q3": round(y3.sum() / den["Q3"], 3), "all_visitors_q3": den["Q3"]}
                if t is None:
                    r.update({"threshold": None, "note": "target not reachable on Q1+Q2"})
                else:
                    fl = w3[kind].values >= t
                    tp, fp = int((fl & (y3 == 1)).sum()), int((fl & (y3 == 0)).sum())
                    r.update({"threshold": t, "flagged": tp + fp, "TP": tp, "FP": fp,
                              "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                              "recall_of_all": round(tp / den["Q3"], 3),
                              "recall_in_list": round(tp / y3.sum(), 3)})
                rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(f"dprime_v2/v2_pad45/eval_pad45_{TAG}.csv", index=False)
show = ["milestone", "list", "model", "rule", "threshold", "flagged", "TP", "FP", "precision", "recall_of_all",
        "list_ceiling_q3"]
for m in MILESTONES:
    print(f"\n=== {m}k (all Q3 visitors {res.loc[res['milestone'] == f'{m}k', 'all_visitors_q3'].iloc[0]:,}) ===")
    print(res[res["milestone"] == f"{m}k"][show].to_string(index=False))
