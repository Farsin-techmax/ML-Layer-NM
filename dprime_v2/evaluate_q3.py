"""Step 6: production NN vs new NN vs new tree on the SAME held-out Q3 2026 D' + prev-step list.
Run from repo root:  ./venv/Scripts/python.exe dprime_v2/evaluate_q3.py 20 30 70 80

Outcome = came in during Q3 2026 (distinct VIN, Q3/Service_History_Q3-2026.csv).
New models: probabilities averaged over the 3 seeds (seed spread is in train_summary_{tag}.csv),
read from every dprime_v2/scores_*.csv. Production NN: q1_due_rule_experiment/q3_dprime_scored_{m}k.csv
(same cars, production features, written by q3_dprime_model_threshold.py).
Metrics: AUC; best-F1 threshold picked on one random half and scored on the other (both ways,
averaged) -- the honest number; precision among the top N, N = production's flag count at 0.5;
F1 of flagging everyone; precision by probability decile.
Writes dprime_v2/eval_q3_{tag}.csv and dprime_v2/eval_q3_deciles_{tag}.csv
(the first 20k/30k run wrote eval_q3.csv / eval_q3_deciles.csv -- same numbers).
"""
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MILESTONES = [f"{int(x)}k" for x in sys.argv[1:]] or ["20k", "30k"]
TAG = "_".join(MILESTONES)
THR = np.round(np.arange(0.01, 1.0, 0.01), 2)


def f1(y, p, t):
    fl = p >= t
    tp = (fl & (y == 1)).sum()
    pr, rc = (tp / fl.sum() if fl.sum() else 0), tp / y.sum()
    return (2 * pr * rc / (pr + rc) if pr + rc else 0), pr, rc


def honest(y, p):
    half = np.random.default_rng(42).permutation(len(y)) % 2 == 0
    out, ts = [], []
    for a, b in [(half, ~half), (~half, half)]:
        t = THR[int(np.argmax([f1(y[a], p[a], t)[0] for t in THR]))]
        ts.append(t)
        out.append(f1(y[b], p[b], t))
    return ts, np.mean([o[0] for o in out]), np.mean([o[1] for o in out]), np.mean([o[2] for o in out])


sc = pd.concat([pd.read_csv(f) for f in glob.glob("dprime_v2/scores_*.csv")])
rows, dec = [], []
for m in MILESTONES:
    prod = pd.read_csv(f"q1_due_rule_experiment/q3_dprime_scored_{m}.csv").rename(columns={"prob": "p_prod"})
    new = (sc[(sc["milestone"] == m) & (sc["quarter"] == "Q3")]
           .pivot_table(index="VIN", columns="model", values="prob", aggfunc="mean")
           .rename(columns={"nn": "p_nn", "tree": "p_tree"}))
    d = prod.merge(new, left_on="VIN", right_index=True, how="inner")
    assert len(d) == len(prod), f"{m}: {len(prod) - len(d)} cars missing from the new scores"
    y = d["came_in"].values
    n_prod = int((d["p_prod"] >= 0.5).sum())
    for name, col in [("production NN", "p_prod"), ("new NN (D' v2)", "p_nn"), ("new tree (D' v2)", "p_tree")]:
        p = d[col].values
        ts, hf1, hpr, hrc = honest(y, p)
        rows.append({"milestone": m, "model": name, "cars": len(y), "came_in": int(y.sum()),
                     "auc": round(roc_auc_score(y, p), 4),
                     "threshold_picked": "/".join(f"{t:.2f}" for t in ts),
                     "f1_holdout": round(hf1, 3), "precision_holdout": round(hpr, 3),
                     "recall_holdout": round(hrc, 3), "prod_volume": n_prod,
                     "precision_at_prod_volume": round(y[np.argsort(-p)[:n_prod]].mean(), 3) if n_prod else np.nan,
                     "f1_flag_everyone": round(f1(y, p, 0.0)[0], 3)})
        q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=range(1, 11))
        dec.append(pd.Series(y).groupby(q.values).mean().round(3).rename(f"{m} {name}"))
res = pd.DataFrame(rows)
res.to_csv(f"dprime_v2/eval_q3_{TAG}.csv", index=False)
decs = pd.concat(dec, axis=1).T
decs.to_csv(f"dprime_v2/eval_q3_deciles_{TAG}.csv")
print(res.to_string(index=False))
print("\nprecision by probability decile (1 = lowest scores):\n", decs.to_string())
