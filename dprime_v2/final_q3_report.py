"""Final Q3 2026 report: accuracy / precision / recall / F1 of the best models so far, next to today's delivered
lists, plus the expected-base (list) result itself. Read-only, from saved scores. Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/final_q3_report.py 20 30 70 80

Three set-ups per milestone, every threshold fixed BEFORE Q3 (nothing tuned on Q3):
  today      delivered list salman/expected_cohort_{m}k_Q3_2026.csv (min(schedule, burn) rule, stale filter),
             production NN, shipped predicted_flag (threshold 0.5)
  exact      D' + prev-step list (exact quarter), best D' v2 model and threshold from best_model_confusion.py
             (max F1 on 2026 Q1+Q2; tree won all four)
  pad45      D' + prev-step list widened to +-45 days, tree + NN averaged (3 seeds each), threshold = the
             equilibrium point on 2026 Q1+Q2 (max of min(precision, recall of all)) from eval_pad45.py
Outcome = came in for the service during the strict quarter 1 Jul - 30 Sep 2026 (Q3/Service_History_Q3-2026.csv).
In-list metrics (accuracy, recall in list, F1) are over the cars on that list; "recall of all" divides by EVERY
car that came in for the service in Q3 (distinct VINs), including cars no list held.
Writes dprime_v2/final_q3_report_{tag}.csv.
"""
import glob
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

from date_utils import parse_dates
from features import resolve_service_num

MILESTONES = [int(x) for x in sys.argv[1:]] or [20, 30, 70, 80]
TAG = "_".join(f"{m}k" for m in MILESTONES)

q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)
q3 = q3[(q3["Service_Date"] >= "2026-07-01") & (q3["Service_Date"] <= "2026-09-30")]


def seed_avg(path):
    s = pd.read_csv(path)
    s = s[s["quarter"] == "Q3"].groupby(["milestone", "model", "VIN"], as_index=False).agg(
        prob=("prob", "mean"), came_in=("came_in", "first"))
    return s.pivot_table(index=["milestone", "VIN", "came_in"], columns="model", values="prob").reset_index()


def load_all(pattern, dedupe_cols, reader=pd.read_csv):
    paths = sorted(glob.glob(pattern))
    df = pd.concat([reader(p) for p in paths], ignore_index=True)
    return df.drop_duplicates(subset=dedupe_cols)


exact = load_all("dprime_v2/scores_*.csv", ["milestone", "VIN"], seed_avg)
pad = load_all("dprime_v2/v2_pad45/scores_*.csv", ["milestone", "VIN"], seed_avg)
pad["ensemble"] = (pad["tree"] + pad["nn"]) / 2
best = load_all("dprime_v2/best_model_confusion_*.csv", ["milestone", "model"])
best = best[best["chosen_on_Q1Q2"].str.startswith("BEST")].set_index("milestone")
ev = load_all("dprime_v2/v2_pad45/eval_pad45_*.csv", ["milestone", "list", "model", "rule"])
ev = ev[(ev["model"] == "ensemble") & ev["list"].str.startswith("+-45") & (ev["rule"] == "balance")].set_index("milestone")


def metrics(name, mk, y, p, thr, n_all):
    flag = p >= thr
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    prec, rec, rec_all = tp / (tp + fp), tp / (tp + fn), tp / n_all
    return {"milestone": mk, "setup": name, "threshold": thr,
            "listed": len(y), "came_in_from_list": int(y.sum()), "all_q3_visitors": n_all,
            "list_coverage_of_visitors": round(y.sum() / n_all, 3), "list_hit_rate": round(y.mean(), 3),
            "flagged": tp + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": round((tp + tn) / len(y), 3), "precision": round(prec, 3),
            "recall_in_list": round(rec, 3), "recall_of_all": round(rec_all, 3),
            "f1_in_list": round(2 * prec * rec / (prec + rec), 3),
            "f1_vs_all_visitors": round(2 * prec * rec_all / (prec + rec_all), 3),
            "auc": round(roc_auc_score(y, p), 4)}


rows = []
for m in MILESTONES:
    mk = f"{m}k"
    visitors = set(q3.loc[q3["Service_Num"] == m, "Vin_No"])
    d = pd.read_csv(f"salman/expected_cohort_{mk}_Q3_2026.csv").drop_duplicates("VIN")
    rows.append(metrics("today: delivered list + production NN", mk, d["VIN"].isin(visitors).astype(int).values,
                        d["predicted_prob"].values, 0.5, len(visitors)))
    if not ((d["predicted_prob"] >= 0.5).astype(int) == d["predicted_flag"]).all():
        print(f"  note: {mk} delivered predicted_flag is not prob >= 0.5 on every row")
    e = exact[exact["milestone"] == mk]
    kind = "tree" if "tree" in best.loc[mk, "model"] else "nn"
    rows.append(metrics(f"exact-quarter D' list + new {kind}", mk, e["came_in"].values, e[kind].values,
                        float(best.loc[mk, "threshold"]), len(visitors)))
    q = pad[pad["milestone"] == mk]
    rows.append(metrics("+-45-day D' list + tree/NN average", mk, q["came_in"].values, q["ensemble"].values,
                        float(ev.loc[mk, "threshold"]), len(visitors)))
res = pd.DataFrame(rows)
res.to_csv(f"dprime_v2/final_q3_report_{TAG}.csv", index=False)
pd.set_option("display.width", 250)
print(res[["milestone", "setup", "threshold", "listed", "came_in_from_list", "all_q3_visitors", "flagged", "TP", "FP",
           "FN", "TN", "accuracy", "precision", "recall_in_list", "recall_of_all", "f1_in_list", "f1_vs_all_visitors",
           "auc"]].to_string(index=False))
