"""Wider success window: a visit up to N days after the quarter counts as success. Free experiment
(user, 2026-10-16). Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/success_window.py 20 30 70 80

For N in 0 / 15 / 30 / 45 and both lists (exact quarter = v2, +-45 days = v2_pad45):
  label     came in [quarter start, quarter end + N]  (first-ever visit for the service; early completers
            were already removed from every list)
  model     HistGradientBoosting retrained on that label (train_models.py's tree recipe, seed 42 -- the
            tree's seed spread was <= 0.009), features unchanged (v2 matrices, relabelled from the lists)
  test      threshold picked on 2026 Q1, applied once to 2026 Q2. Q3 cannot be used: the data ends
            2026-09-29, so visits after Q3 are unknown. Q2 + 45 days ends 2026-08-14 (Q3 file covers it).
  recall    of ALL visitors = flagged cars that came in the window / every car whose first-ever visit for
            the service falls in the window (all VINs, base history + Q3 file)
Operating points: balance = max(min(precision, recall of all)); recall >= target (highest precision);
precision >= target (highest recall). Writes dprime_v2/success_window_{tag}.csv.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score

from date_utils import parse_dates
from features import resolve_service_num

MILESTONES = [int(x) for x in sys.argv[1:]] or [20, 30, 70, 80]
TAG = "_".join(f"{m}k" for m in MILESTONES)
WINDOWS = [0, 15, 30, 45]
LISTS = {"exact": ("v2", ""), "pad45": ("v2_pad45", "_pad45")}
HEAD = ["VIN", "TargetFlag", "CameInQuarter", "CohortQuarter"]
THR = np.round(np.arange(0.005, 1.0, 0.005), 3)

cols = ["Vin_No", "Service_Date", "Service_Code", "Description"]
hist = pd.concat([pd.read_csv(f, low_memory=False, encoding="ISO-8859-1", usecols=cols)
                  for f in ["data/Service_History_Q2-2026.csv", "Q3/Service_History_Q3-2026.csv"]])
hist["Service_Date"] = parse_dates(hist["Service_Date"])
hist["Service_Num"] = resolve_service_num(hist)


def curve(y, p, denom):
    out = []
    for t in THR:
        fl = p >= t
        tp, n = int((fl & (y == 1)).sum()), int(fl.sum())
        out.append((t, tp / n if n else 0.0, tp / denom, tp, n))
    return out


def choose(cv, rule, target=None):
    if rule == "balance":
        return max([c for c in cv if c[4] > 0], key=lambda c: min(c[1], c[2]))[0]
    if rule == "recall":
        ok = [c for c in cv if c[2] >= target and c[4] > 0]
        return max(ok, key=lambda c: (c[1], -c[0]))[0] if ok else None
    ok = [c for c in cv if c[1] >= target]
    return max(ok, key=lambda c: (c[2], c[0]))[0] if ok else None


rows = []
for m in MILESTONES:
    first = hist[hist["Service_Num"] == m].groupby("Vin_No")["Service_Date"].min()
    for lname, (setname, lsuf) in LISTS.items():
        lst = pd.read_csv(f"dprime_v2/lists_{m}k{lsuf}.csv", usecols=["VIN", "CohortQuarter", "q_start", "q_end",
                                                                      f"Actual{m}kDate"],
                          parse_dates=["q_start", "q_end", f"Actual{m}kDate"])
        lst = lst.drop_duplicates(["VIN", "CohortQuarter"])
        tr = pd.read_csv(f"refactored_test_dir/final_processed_{m}k_dprime_{setname}.csv")
        due = f"Next{m}K_Due"
        tr = tr.assign(_d=parse_dates(tr[due])).sort_values("_d").drop(columns="_d").reset_index(drop=True)
        feats = [c for c in tr.columns if c not in HEAD + [due]]
        te = {q: pd.read_csv(f"test_sets/test_{m}k_{q}_2026_dprime_{setname}.csv") for q in ["Q1", "Q2"]}
        frames = {"train": tr, **te}
        frames = {k: v.merge(lst, on=["VIN", "CohortQuarter"], how="left") for k, v in frames.items()}
        Xall = frames["train"][feats].astype(float)
        keep = IsolationForest(contamination=0.005, random_state=42).fit_predict(Xall.fillna(Xall.median())) == 1
        for N in WINDOWS:
            lab = {}
            for k, f in frames.items():
                a = f[f"Actual{m}kDate"]
                lab[k] = (a.notna() & (a >= f["q_start"]) & (a <= f["q_end"] + pd.Timedelta(days=N))).astype(int).values
            X, y = Xall[keep].reset_index(drop=True), lab["train"][keep]
            cut = int(len(X) * 0.8)
            clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                                 l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                                 n_iter_no_change=30, random_state=42).fit(X.iloc[:cut], y[:cut])
            den = {}
            for q in ["Q1", "Q2"]:
                qs, qe = frames[q]["q_start"].iloc[0], frames[q]["q_end"].iloc[0]
                den[q] = int(((first >= qs) & (first <= qe + pd.Timedelta(days=N))).sum())
            p1 = clf.predict_proba(frames["Q1"][feats].astype(float))[:, 1]
            p2 = clf.predict_proba(frames["Q2"][feats].astype(float))[:, 1]
            y1, y2 = lab["Q1"], lab["Q2"]
            cv1 = curve(y1, p1, den["Q1"])
            base = {"milestone": f"{m}k", "list": lname, "window_days": N, "q2_listed": len(y2),
                    "q2_came_in_window": int(y2.sum()), "q2_all_visitors": den["Q2"],
                    "q2_ceiling": round(y2.sum() / den["Q2"], 3), "q2_list_rate": round(y2.mean(), 3),
                    "q2_auc": round(roc_auc_score(y2, p2), 4), "train_pos_rate": round(y[:cut].mean(), 3)}
            rules = [("balance", "balance", None), ("recall>=80%", "recall", .80), ("precision>=80%", "precision", .80)]
            if m == 20:
                rules += [("recall>=70%", "recall", .70), ("precision>=70%", "precision", .70)]
            for label, rule, target in rules:
                t = choose(cv1, rule, target)
                r = dict(base, rule=label, threshold=t)
                if t is not None:
                    fl = p2 >= t
                    tp, fp = int((fl & (y2 == 1)).sum()), int((fl & (y2 == 0)).sum())
                    r.update(flagged=tp + fp, TP=tp, FP=fp, precision=round(tp / (tp + fp), 3) if tp + fp else None,
                             recall_of_all=round(tp / den["Q2"], 3))
                rows.append(r)
            print(f"{m}k {lname} N={N}: Q2 AUC {base['q2_auc']}, list rate {base['q2_list_rate']}, "
                  f"ceiling {base['q2_ceiling']}", flush=True)
res = pd.DataFrame(rows)
res.to_csv(f"dprime_v2/success_window_{TAG}.csv", index=False)
show = ["milestone", "list", "window_days", "rule", "threshold", "flagged", "TP", "precision", "recall_of_all",
        "q2_ceiling", "q2_auc"]
print(res[show].to_string(index=False))
