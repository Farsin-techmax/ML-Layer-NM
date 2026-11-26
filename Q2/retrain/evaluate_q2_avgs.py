"""Step 5 of the average-interval retrain: head-to-head on the Q2 2026 list (the same 1,755 30k VINs the
deliverable holds, verified identical), strict-quarter outcome.

  production = predicted_prob in Q2/expected_cohort_{m}k_Q2_2026.csv (production_models/{m}k), flag at 0.5
  new tree / new NN = seed-averaged (3 seeds); ensemble = mean of the two
Thresholds for the new models: 0.5, and the max-F1 threshold picked on 2026 Q1 (a different quarter --
nothing tuned on Q2), applied once to Q2. Recall of ALL = TP / every first-ever m visit in Q2 (any VIN).
Also: how many of production's false negatives the new model recovers, and the LowMileageFreqUsers check.

Run from repo root:  PYTHONPATH=. ./venv/Scripts/python.exe Q2/retrain/evaluate_q2_avgs.py 30
Writes Q2/retrain/eval_q2_{m}k_avgs.csv.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

M = ([int(x) for x in sys.argv[1:] if x.isdigit()] or [30])[0]
ALL_Q2_VISITORS = int(pd.read_csv("Q2/summary_avg_interval_Q2_2026.csv").set_index("milestone")
                      .loc[f"{M}k", "all_Q2_visitors"])

s = pd.read_csv(f"Q2/retrain/scores_{M}k_avgs.csv")
avg = s.groupby(["quarter", "model", "VIN"]).agg(prob=("prob", "mean"), came=("came_in", "first")).reset_index()
wide = avg.pivot_table(index=["quarter", "VIN"], columns="model", values="prob").reset_index()
wide = wide.merge(avg[["quarter", "VIN", "came"]].drop_duplicates(), on=["quarter", "VIN"])
wide["ensemble"] = (wide["nn"] + wide["tree"]) / 2

prod = pd.read_csv(f"Q2/expected_cohort_{M}k_Q2_2026.csv")[["VIN", "predicted_prob", "predicted_flag",
                                                           "ActualTargetFlag"]]
q2 = wide[wide["quarter"] == "Q2"].merge(prod, on="VIN", how="inner")
assert len(q2) == len(prod), "Q2 retrain set and deliverable differ"
assert (q2["came"] == q2["ActualTargetFlag"]).all(), "outcome mismatch"
q1 = wide[wide["quarter"] == "Q1"]


def f1_best_threshold(y, p):
    best, thr = -1, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        f = p >= t
        tp = (f & (y == 1)).sum()
        prec, rec = tp / max(f.sum(), 1), tp / max((y == 1).sum(), 1)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        if f1 > best:
            best, thr = f1, round(float(t), 2)
    return thr


def row(name, y, p, flag, thr):
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    rec_all = tp / ALL_Q2_VISITORS
    return {"model": name, "threshold": thr, "AUC": round(roc_auc_score(y, p), 4), "flagged": tp + fp,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision_pct": round(100 * prec, 1),
            "recall_in_list_pct": round(100 * rec, 1), "recall_of_all_pct": round(100 * rec_all, 1),
            "F1_in_list": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0}


y = q2["came"].values
rows = [row("production NN", y, q2["predicted_prob"].values, q2["predicted_flag"].values == 1, 0.5)]
for name in ["tree", "nn", "ensemble"]:
    p = q2[name].values
    rows.append(row(f"new {name}", y, p, p >= 0.5, 0.5))
    t = f1_best_threshold(q1["came"].values, q1[name].values)
    rows.append(row(f"new {name}", y, p, p >= t, t))
# same call volume as production: flag the top-N by the ensemble, N = production's flagged count
n = int(q2["predicted_flag"].sum())
top = np.zeros(len(q2), bool)
top[np.argsort(-q2["ensemble"].values)[:n]] = True
rows.append(row("new ensemble, same volume as prod", y, q2["ensemble"].values, top, f"top {n}"))
out = pd.DataFrame(rows)
out.to_csv(f"Q2/retrain/eval_q2_{M}k_avgs.csv", index=False)
print(f"{M}k Q2 2026 -- {len(q2):,} listed, {int(y.sum()):,} came in, {ALL_Q2_VISITORS:,} visitors in total\n")
print(out.to_string(index=False))

# production's false negatives -- how many does the new ensemble (Q1-picked threshold) catch?
t_ens = f1_best_threshold(q1["came"].values, q1["ensemble"].values)
fn_prod = (q2["predicted_flag"] == 0) & (q2["came"] == 1)
print(f"\nProduction false negatives: {int(fn_prod.sum())}; new ensemble (thr {t_ens}) flags "
      f"{int((fn_prod & (q2['ensemble'] >= t_ens)).sum())} of them")

feats = pd.read_csv(f"Q2/features/features_{M}k_Q2_2026.csv", usecols=["VIN", "LowMileageFreqUsers"],
                    low_memory=False)
g = q2.merge(feats, on="VIN").groupby("LowMileageFreqUsers").agg(
    n=("VIN", "size"), actual_came_in=("came", "mean"), production_prob=("predicted_prob", "mean"),
    new_ensemble_prob=("ensemble", "mean")).round(3)
print("\nLowMileageFreqUsers group (production's leaked feature) -- actual vs predicted:")
print(g.to_string())

print("\nAUC by quarter (seed-averaged):")
print(wide.groupby("quarter").apply(lambda d: pd.Series(
    {k: round(roc_auc_score(d["came"], d[k]), 4) for k in ["tree", "nn", "ensemble"]})).to_string())
