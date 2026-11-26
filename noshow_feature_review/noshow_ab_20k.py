"""Follow-up A/B for the 20k no-show review. Reads frames_20k.pkl written by noshow_features_20k.py.
Run from repo root:  PYTHONPATH=. ./venv/Scripts/python.exe noshow_feature_review/noshow_ab_20k.py

Question: do the forward (q3) no-shows get pushed down if we
  (a) drop the features that are strong in training but dead on a forward list -- computed through
      label-dependent code paths or from uncut / extract-time data (LEAK below), and/or
  (b) add the own-rhythm features from noshow_features_20k.py?
Same learner for every arm (HistGradientBoosting, 3 seeds, trained on the training rows) so only the
feature set changes. Paired bootstrap on q3 (2,000 resamples) for the arm-vs-base differences.
Also: the own-rhythm ratio used as a plain post-score rule on the production list (no retrain).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

OUT = "noshow_feature_review"
SEEDS = [0, 1, 2]
F = pd.read_pickle(f"{OUT}/frames_20k.pkl")
X, NEW, PROB, Y = F["X"], F["NEW"], F["PROB"], F["Y"]

LEAK = ["PMS_Delay", "Last Service Mileage", "LastNonPMSMileage",
        "Avg_Service_Interval_PMS", "Avg_Service_Interval_PMSold", "Avg_Service_Interval_PMS1old",
        "Avg_Service_Interval_PMSper10k", "months_to_10k", "months_base_to_10"]
RHYTHM = ["rhythm_ratio_at_qend", "rhythm_ratio_at_cutoff", "own_gap_days", "avg_days_from_qmid",
          "burn_days_from_qmid", "days_since_last_pms"]
BASE = list(X["q3"].columns)
NO_LEAK = [c for c in BASE if c not in LEAK]
ARMS = {
    "base_160": (BASE, []),
    "minus_leak9": (NO_LEAK, []),
    "base_plus_rhythm": (BASE, RHYTHM),
    "minus_leak9_plus_rhythm": (NO_LEAK, RHYTHM),
}


def frame(k, cols, extra):
    f = X[k][cols].reset_index(drop=True)
    return pd.concat([f, NEW[k][extra].reset_index(drop=True)], axis=1) if extra else f


def prec_at(y, p, n):
    return y[np.argsort(-p)[:n]].mean()


preds, rows = {}, []
for arm, (cols, extra) in ARMS.items():
    ps = {k: [] for k in ["q1", "q2", "q3"]}
    for s in SEEDS:
        clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=True,
                                             validation_fraction=0.1, random_state=s)
        clf.fit(frame("train", cols, extra), Y["train"])
        for k in ps:
            ps[k].append(clf.predict_proba(frame(k, cols, extra))[:, 1])
    for k in ps:
        p = np.mean(ps[k], axis=0)
        preds[(arm, k)] = p
        n = int((PROB[k] >= 0.5).sum())
        rows.append({"arm": arm, "pop": k, "auc": roc_auc_score(Y[k], p),
                     "precision_at_prod_volume": prec_at(Y[k], p, n),
                     "noshows_in_bottom_20pct": 1 - Y[k][np.argsort(p)[: len(p) // 5]].mean()})
    print(f"  {arm} done")
for k in ["q1", "q2", "q3"]:
    n = int((PROB[k] >= 0.5).sum())
    rows.append({"arm": "production_NN", "pop": k, "auc": roc_auc_score(Y[k], PROB[k]),
                 "precision_at_prod_volume": Y[k][PROB[k] >= 0.5].mean(),
                 "noshows_in_bottom_20pct": 1 - Y[k][np.argsort(PROB[k])[: len(PROB[k]) // 5]].mean()})
res = pd.DataFrame(rows).round(4)
res.to_csv(f"{OUT}/ab_leak_rhythm_20k.csv", index=False)
print(res.pivot(index="arm", columns="pop").to_string())

# paired bootstrap on q3: arm minus base_160
rng = np.random.default_rng(42)
y = Y["q3"]
n_flag = int((PROB["q3"] >= 0.5).sum())
idx = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
boot = []
for arm in [a for a in ARMS if a != "base_160"] + ["production_NN"]:
    pa = PROB["q3"] if arm == "production_NN" else preds[(arm, "q3")]
    pb = preds[("base_160", "q3")]
    d_auc, d_prec = [], []
    for i in idx:
        if len(set(y[i])) < 2:
            continue
        d_auc.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
        d_prec.append(prec_at(y[i], pa[i], n_flag) - prec_at(y[i], pb[i], n_flag))
    boot.append({"arm_vs_base160": arm,
                 "d_auc": np.mean(d_auc), "d_auc_lo95": np.percentile(d_auc, 2.5),
                 "d_auc_hi95": np.percentile(d_auc, 97.5),
                 "d_prec": np.mean(d_prec), "d_prec_lo95": np.percentile(d_prec, 2.5),
                 "d_prec_hi95": np.percentile(d_prec, 97.5)})
boot = pd.DataFrame(boot).round(4)
boot.to_csv(f"{OUT}/ab_leak_rhythm_20k_bootstrap_q3.csv", index=False)
print("\nq3 paired bootstrap vs base_160 (HGB):")
print(boot.to_string(index=False))

# own-rhythm ratio as a plain rule on the production list, no retrain
BINS = [-np.inf, 0.75, 1.0, 1.25, np.inf]
LABELS = ["<0.75", "0.75-1.0", "1.0-1.25", ">=1.25"]
rule = []
for k in ["q1", "q2", "q3"]:
    r = NEW[k]["rhythm_ratio_at_qend"].values
    b = pd.cut(pd.Series(r), BINS, labels=LABELS).cat.add_categories("no rhythm").fillna("no rhythm")
    for flagged_only in [False, True]:
        m = (PROB[k] >= 0.5) if flagged_only else np.ones(len(r), bool)
        t = pd.DataFrame({"bucket": b[m].values, "y": Y[k][m]}).groupby("bucket")["y"].agg(
            ["size", "mean"])
        for bucket, s in t.iterrows():
            rule.append({"pop": k, "rows": "flagged" if flagged_only else "all listed",
                         "rhythm_ratio_at_qend": bucket, "n": int(s["size"]),
                         "came_in_pct": round(100 * s["mean"], 1)})
rule = pd.DataFrame(rule)
rule.to_csv(f"{OUT}/rhythm_rule_20k.csv", index=False)
print("\nrhythm ratio at quarter end (quarter end minus last PMS, over own usual gap):")
print(rule.pivot_table(index=["rows", "rhythm_ratio_at_qend"], columns="pop",
                       values=["n", "came_in_pct"]).to_string())
