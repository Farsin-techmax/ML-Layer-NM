"""Standard results table for one evaluated quarter: +-45-day D' + prev-step list, scored by the saved models,
one row per milestone plus an All row. Read-only, from saved scores. Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/results_table.py --quarter Q3
    PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/results_table.py --quarter Q3 --model tree 20 30

Arguments (all optional):
  milestones      e.g. 20 30 70  (default: all nine)
  --quarter Qn    the 2026 quarter to report (default Q3); must exist in the scores files
  --model M       ensemble (tree + NN average, the current best set-up) | tree | nn   (default ensemble)
  --outcomes F    service-history file holding the quarter's visits
                  (default: Q3/Service_History_Q3-2026.csv for Q3, data/Service_History_Q2-2026.csv for Q1/Q2)

Inputs: scores dprime_v2/v2_pad45/scores_*.csv (train_models.py --set v2_pad45), each model seed-averaged;
cut-offs dprime_v2/v2_pad45/eval_pad45_*.csv (eval_pad45.py, 'balance' rule chosen on 2026 Q1+Q2);
list gaps dprime_v2/lists_{m}k_pad45.csv (for AUC on next-step cars).
Outcome = came in for the service during the strict quarter. visitors = every distinct car with a visit for that
service in the quarter, including cars the list never held. correct = TP + TN. accuracy = correct / listed.
Writes dprime_v2/results_table_{quarter}_{model}.csv and prints a Markdown table.
"""
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from date_utils import parse_dates
from features import resolve_service_num
from pmstrainfeatureeng_refactored import quarter_bounds

ARGV = sys.argv[1:]


def opt(name, default):
    return ARGV[ARGV.index(name) + 1] if name in ARGV else default


QUARTER = opt("--quarter", "Q3")
MODEL = opt("--model", "ensemble")
OUTCOMES = opt("--outcomes", "Q3/Service_History_Q3-2026.csv" if QUARTER == "Q3" else "data/Service_History_Q2-2026.csv")
values = {QUARTER, MODEL, OUTCOMES}
MILESTONES = [int(x) for x in ARGV if x.isdigit() and x not in values] or [20, 30, 40, 50, 60, 70, 80, 90, 100]
assert MODEL in ("ensemble", "tree", "nn"), f"--model must be ensemble, tree or nn, got {MODEL}"

# scores: seed-averaged per model, then the tree/NN average
sc = pd.concat([pd.read_csv(f) for f in glob.glob("dprime_v2/v2_pad45/scores_*.csv")])
sc = sc[sc["quarter"] == QUARTER].drop_duplicates(["milestone", "model", "seed", "VIN"])
assert len(sc), f"no scores for {QUARTER} in dprime_v2/v2_pad45/scores_*.csv"
sc = sc.groupby(["milestone", "model", "VIN"], as_index=False).agg(prob=("prob", "mean"), y=("came_in", "first"))
W = sc.pivot_table(index=["milestone", "VIN", "y"], columns="model", values="prob").reset_index()
W["ensemble"] = (W["tree"] + W["nn"]) / 2

# cut-offs chosen on 2026 Q1+Q2 by eval_pad45.py
ev = pd.concat([pd.read_csv(f) for f in glob.glob("dprime_v2/v2_pad45/eval_pad45_*.csv")])
ev = ev[(ev["model"] == MODEL) & ev["list"].str.startswith("+-45") & (ev["rule"] == "balance")]
cut = ev.drop_duplicates("milestone").set_index("milestone")["threshold"]

# visitors in the quarter
hist = pd.read_csv(OUTCOMES, low_memory=False, encoding="ISO-8859-1",
                   usecols=["Vin_No", "Service_Date", "Service_Code", "Description"])
hist["Service_Date"] = parse_dates(hist["Service_Date"])
hist["Service_Num"] = resolve_service_num(hist)
qs, qe = quarter_bounds(2026, int(QUARTER[1]))
hist = hist[(hist["Service_Date"] >= qs) & (hist["Service_Date"] <= qe)]

rows = []
for m in MILESTONES:
    mk = f"{m}k"
    d = W[W["milestone"] == mk]
    if d.empty or mk not in cut.index:
        print(f"skipping {mk}: no scores or no cut-off")
        continue
    gap = pd.read_csv(f"dprime_v2/lists_{mk}_pad45.csv", usecols=["VIN", "CohortQuarter", "EarlyCompleter", "gap"])
    gap = gap[(gap["CohortQuarter"] == f"2026{QUARTER}") & (gap["EarlyCompleter"] == 0)].set_index("VIN")["gap"]
    y, p, t = d["y"].values, d[MODEL].values, float(cut[mk])
    nxt = d["VIN"].map(gap).values == 10
    visitors = hist.loc[hist["Service_Num"] == m, "Vin_No"].nunique()
    f = p >= t
    tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
    fn, tn = int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum())
    prec, rec = tp / (tp + fp), tp / visitors
    rows.append({"milestone": mk, "cut_off": t, "listed": len(d), "visitors": visitors, "list_holds": y.sum() / visitors,
                 "TP": tp, "FP": fp, "FN": fn, "TN": tn, "correct": tp + tn, "accuracy": (tp + tn) / len(d),
                 "precision": prec, "recall_of_all": rec, "f1": 2 * prec * rec / (prec + rec),
                 "auc_whole_list": roc_auc_score(y, p),
                 "auc_next_step": roc_auc_score(y[nxt], p[nxt]) if 0 < y[nxt].sum() < nxt.sum() else np.nan})
R = pd.DataFrame(rows)
s = R[["listed", "visitors", "TP", "FP", "FN", "TN", "correct"]].sum()
came = (R["list_holds"] * R["visitors"]).sum()
R.loc[len(R)] = {"milestone": "All", "listed": s["listed"], "visitors": s["visitors"], "list_holds": came / s["visitors"],
                 "TP": s["TP"], "FP": s["FP"], "FN": s["FN"], "TN": s["TN"], "correct": s["correct"],
                 "accuracy": s["correct"] / s["listed"], "precision": s["TP"] / (s["TP"] + s["FP"]),
                 "recall_of_all": s["TP"] / s["visitors"]}
out = f"dprime_v2/results_table_{QUARTER}_{MODEL}.csv"
R.to_csv(out, index=False)


def fmt(v, kind):
    if pd.isna(v):
        return ""
    return {"pct": f"{100 * v:.1f}%", "int": f"{int(round(v)):,}", "auc": f"{v:.3f}"}[kind]


cols = [("cut_off", "cut-off", "pct"), ("listed", "listed", "int"), ("visitors", "visitors", "int"),
        ("list_holds", "list holds", "pct"), ("TP", "TP", "int"), ("FP", "FP", "int"), ("FN", "FN", "int"),
        ("TN", "TN", "int"), ("correct", "correct (TP + TN)", "int"), ("accuracy", "accuracy", "pct"),
        ("precision", "precision", "pct"), ("recall_of_all", "recall of all", "pct"), ("f1", "F1", "auc"),
        ("auc_whole_list", "AUC whole list", "auc"), ("auc_next_step", "AUC next-step cars", "auc")]
print(f"\n{QUARTER} 2026, model = {MODEL}, outcomes = {OUTCOMES}\n")
print("| | " + " | ".join(c[1] for c in cols) + " |")
print("|---" * (len(cols) + 1) + "|")
for _, r in R.iterrows():
    print(f"| {r['milestone']} | " + " | ".join(fmt(r[c], k) for c, _, k in cols) + " |")
print(f"\nwrote {out}")
