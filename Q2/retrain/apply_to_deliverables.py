"""Step 6 of the average-interval retrain (user decision 2026-10-21: replace all nine): swap the
production model's predicted_prob / predicted_flag in Q2/expected_cohort_{m}k_Q2_2026.csv for the
retrained ensemble's.

  predicted_prob = mean(seed-averaged tree, seed-averaged NN) from Q2/retrain/scores_{m}k_avgs.csv
  predicted_flag = predicted_prob >= 0.5
Every other column is kept as-is (same rows). Rows are re-sorted by the new probability, highest first.
The production-scored versions are copied to Q2/pre_retrain/ first (never overwritten on a rerun), and
Q2/eval_model_Q2_2026.csv is rebuilt for the new probabilities (old copy also kept in Q2/pre_retrain/).

NOTE: re-running Q2/score_q2_avg_interval.py rewrites the deliverables with PRODUCTION scores again --
run this script after it, then Q2/add_last_service_dates.py (it adds the last PMS / non-PMS dates).

Run from repo root:  PYTHONPATH=. ./venv/Scripts/python.exe Q2/retrain/apply_to_deliverables.py
"""
import os
import shutil

import pandas as pd
from sklearn.metrics import roc_auc_score

MILESTONES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
BACKUP = "Q2/pre_retrain"
THRESHOLD = 0.5

os.makedirs(BACKUP, exist_ok=True)
summary = pd.read_csv("Q2/summary_avg_interval_Q2_2026.csv").set_index("milestone")
rows, locked = [], []
for m in MILESTONES:
    path = f"Q2/expected_cohort_{m}k_Q2_2026.csv"
    if not os.path.exists(os.path.join(BACKUP, os.path.basename(path))):
        shutil.copy2(path, BACKUP)
    coh = pd.read_csv(path)

    s = pd.read_csv(f"Q2/retrain/scores_{m}k_avgs.csv")
    s = s[s["quarter"] == "Q2"].groupby(["model", "VIN"])["prob"].mean().unstack("model")
    ens = (s["nn"] + s["tree"]) / 2
    missing = set(coh["VIN"]) - set(ens.index)
    assert not missing, f"{m}k: {len(missing)} deliverable VINs have no new score"

    coh["predicted_prob"] = coh["VIN"].map(ens).round(6)
    coh["predicted_flag"] = (coh["predicted_prob"] >= THRESHOLD).astype(int)
    coh = coh.sort_values("predicted_prob", ascending=False).reset_index(drop=True)
    tmp = path + ".tmp"
    coh.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
    except PermissionError:
        locked.append(m)
        print(f"{m}k: LOCKED (open elsewhere) -- new version left at {tmp}")
        continue

    y, p, f = coh["ActualTargetFlag"], coh["predicted_prob"], coh["predicted_flag"]
    tp, fp = int(((f == 1) & (y == 1)).sum()), int(((f == 1) & (y == 0)).sum())
    fn, tn = int(((f == 0) & (y == 1)).sum()), int(((f == 0) & (y == 0)).sum())
    allv = int(summary.loc[f"{m}k", "all_Q2_visitors"])
    rows.append({"milestone": f"{m}k", "listed": len(coh), "came_in_from_list": int(y.sum()),
                 "all_Q2_visitors": allv, "flagged": tp + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                 "precision_pct": round(100 * tp / max(tp + fp, 1), 1),
                 "recall_in_list_pct": round(100 * tp / max(tp + fn, 1), 1),
                 "recall_of_all_pct": round(100 * tp / allv, 1),
                 "AUC": round(roc_auc_score(y, p), 4), "mean_prob": round(p.mean(), 3)})
    print(f"{m}k: replaced {len(coh):,} rows, flagged {tp + fp:,}")

if not os.path.exists(os.path.join(BACKUP, "eval_model_Q2_2026.csv")):
    shutil.copy2("Q2/eval_model_Q2_2026.csv", BACKUP)
ev = pd.DataFrame(rows)
if not locked:
    ev.to_csv("Q2/eval_model_Q2_2026.csv", index=False)
print("\n" + ev.to_string(index=False))
print(f"\nlocked: {locked}" if locked else "\nall nine replaced; production-scored originals in Q2/pre_retrain/")
