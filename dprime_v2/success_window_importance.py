"""Which features carry the success-window models (the last run)? Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/success_window_importance.py 20 30 70 80

Rebuilds exactly the success_window.py tree (+-45-day list, seed 42) for N = 0 and N = 45, then measures
permutation importance on its TEST quarter, 2026 Q2: how much Q2 AUC drops when one feature (or a whole
family) is shuffled, 5 shuffles averaged. Families as feature_importance.py ("read groups, not columns").
Writes dprime_v2/v2_pad45/success_window_importance_{m}k.csv and success_window_importance_groups_{tag}.csv.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score

from date_utils import parse_dates

MILESTONES = [int(x) for x in sys.argv[1:]] or [20, 30, 70, 80]
TAG = "_".join(f"{m}k" for m in MILESTONES)
WINDOWS, REPEATS = [0, 45], 5
HEAD = ["VIN", "TargetFlag", "CameInQuarter", "CohortQuarter"]
GROUPS = {
    "list context (which method listed it, leg dates)": lambda c: c.startswith(("by_", "n_legs", "centre_days", "gap"))
    or c.endswith("_days_from_qmid"),
    "PMS history & rhythm": lambda c: c.startswith(("has_", "n_pms", "pms_", "rhythm", "days_since_last_pms")),
    "mileage": lambda c: c.startswith(("last_km", "km_")),
    "visit recency & activity": lambda c: c in ("visits_total", "visits_365d", "days_since_any_visit", "age_days",
                                                "invoice_to_first_visit_days", "n_keys"),
    "non-PMS visits & revenue": lambda c: c.startswith(("n_nonpms", "days_since_last_nonpms", "nonpms_", "rev_",
                                                        "tyre_", "battery_", "brake_")),
    "branch / bodyshop / survey": lambda c: c.startswith(("n_branches", "share_main", "main_branch", "bodyshop",
                                                          "survey", "n_detractor")),
    "VHC": lambda c: c.startswith(("vhc_", "has_vhc", "days_since_vhc")),
    "appointments": lambda c: c.startswith(("appt_", "has_appt", "days_since_last_booking", "upcoming_")),
    "vehicle & customer facts (EDA)": lambda c: True,
}


def family(c):
    for g, f in GROUPS.items():
        if f(c):
            return g


rng = np.random.default_rng(0)
group_rows = []
for m in MILESTONES:
    lst = pd.read_csv(f"dprime_v2/lists_{m}k_pad45.csv", usecols=["VIN", "CohortQuarter", "q_start", "q_end",
                                                                  f"Actual{m}kDate"],
                      parse_dates=["q_start", "q_end", f"Actual{m}kDate"]).drop_duplicates(["VIN", "CohortQuarter"])
    tr = pd.read_csv(f"refactored_test_dir/final_processed_{m}k_dprime_v2_pad45.csv")
    due = f"Next{m}K_Due"
    tr = tr.assign(_d=parse_dates(tr[due])).sort_values("_d").drop(columns="_d").reset_index(drop=True)
    feats = [c for c in tr.columns if c not in HEAD + [due]]
    te = pd.read_csv(f"test_sets/test_{m}k_Q2_2026_dprime_v2_pad45.csv")
    tr, te = (f.merge(lst, on=["VIN", "CohortQuarter"], how="left") for f in (tr, te))
    Xall = tr[feats].astype(float)
    keep = IsolationForest(contamination=0.005, random_state=42).fit_predict(Xall.fillna(Xall.median())) == 1
    X2 = te[feats].astype(float).reset_index(drop=True)

    def label(f, N):
        a = f[f"Actual{m}kDate"]
        return (a.notna() & (a >= f["q_start"]) & (a <= f["q_end"] + pd.Timedelta(days=N))).astype(int).values

    per_n = {}
    for N in WINDOWS:
        X, y = Xall[keep].reset_index(drop=True), label(tr, N)[keep]
        cut = int(len(X) * 0.8)
        clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                             n_iter_no_change=30, random_state=42).fit(X.iloc[:cut], y[:cut])
        y2 = label(te, N)
        base = roc_auc_score(y2, clf.predict_proba(X2)[:, 1])

        def drop(cols):
            out = []
            for _ in range(REPEATS):
                Z = X2.copy()
                Z[cols] = Z[cols].values[rng.permutation(len(Z))]
                out.append(base - roc_auc_score(y2, clf.predict_proba(Z)[:, 1]))
            return float(np.mean(out))

        per_n[N] = {c: drop([c]) for c in feats}
        for g in GROUPS:
            cols = [c for c in feats if family(c) == g]
            if cols:
                group_rows.append({"milestone": f"{m}k", "window_days": N, "family": g, "n_features": len(cols),
                                   "q2_auc": round(base, 4), "auc_drop": drop(cols)})
        print(f"{m}k N={N}: Q2 AUC {base:.4f}", flush=True)
    r = pd.DataFrame({"feature": feats, "family": [family(c) for c in feats],
                      **{f"auc_drop_N{N}": [per_n[N][c] for c in feats] for N in WINDOWS}})
    r = r.sort_values("auc_drop_N45", ascending=False).reset_index(drop=True)
    r.index += 1
    r.round(4).to_csv(f"dprime_v2/v2_pad45/success_window_importance_{m}k.csv", index_label="rank")
    print(f"\n=== {m}k: top 15 single features, +-45 list, +45-day window (Q2 AUC drop when shuffled) ===")
    print(r.head(15).round(4).to_string())
    print(f"   {int((r['auc_drop_N45'] <= 0.0005).sum())} of {len(r)} features move Q2 AUC by <= 0.0005 alone\n")
g = pd.DataFrame(group_rows).round(4)
g.to_csv(f"dprime_v2/v2_pad45/success_window_importance_groups_{TAG}.csv", index=False)
print("=== whole families shuffled together (Q2 AUC drop) ===")
print(g.pivot_table(index="family", columns=["milestone", "window_days"], values="auc_drop").round(3).to_string())
