"""Step 3 leak guard for the average-interval retrain -- same three checks as
dprime_v2/test_no_label_leak.py, pointed at the avgs lists/matrices:

A (code)   : dprime_v2/features_v2.py never references an outcome column.
B (flip)   : build 2025Q4's features twice -- as-is and with every outcome column scrambled; identical.
C (screen) : single-feature AUC on training rows vs held-out 2026 Q2 and Q3 (strict-quarter outcome).
             Flags a SIGN FLIP (both >0.03 from 0.5, opposite sides) or a gap >= 0.10. Read flags with
             the coverage-drift caveat in CLAUDE.md (G6 blind spot).
Exit 0 = A and B pass. Writes Q2/retrain/leak_screen_{m}k_avgs.csv.
Run from repo root:  PYTHONPATH=. ./venv/Scripts/python.exe Q2/retrain/test_no_label_leak_avgs.py 30
"""
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MILESTONES = [int(x) for x in sys.argv[1:] if x.isdigit()] or [30]
OUTCOME = ["TargetFlag", "CameInQuarter", "EarlyCompleter", "Actual"]

body = open("dprime_v2/features_v2.py", encoding="utf-8").read().split('"""', 2)[2]
hits = [w for w in OUTCOME if re.search(rf"\b{w}", body)]
print(f"A code: outcome names in features_v2.py body: {hits or 'none'} -> {'PASS' if not hits else 'FAIL'}")
ok = not hits

sys.path.insert(0, "Q2/retrain")
from build_matrix_avgs import load_sources  # noqa: E402
from dprime_v2.features_v2 import block_features  # noqa: E402
src = load_sources()

for m in MILESTONES:
    lst = pd.read_csv(f"Q2/retrain/lists_{m}k_avgs.csv",
                      parse_dates=["q_start", "q_end", "cutoff", "burn_date", "last6_date", "avg_date"])
    blk = lst[(lst["CohortQuarter"] == "2025Q4") & (lst["EarlyCompleter"] == 0)].copy()
    r = blk.iloc[0]
    a = block_features(src, blk, m, r["cutoff"], r["q_start"], r["q_end"])
    scr = blk.copy()
    rng = np.random.default_rng(0)
    for c in [c for c in scr.columns if any(c.startswith(o) for o in OUTCOME)]:
        scr[c] = rng.permutation(scr[c].values)
    b = block_features(src, scr, m, r["cutoff"], r["q_start"], r["q_end"])
    same = a.sort_index().equals(b.sort_index())
    print(f"B flip {m}k 2025Q4: features identical with outcomes scrambled -> {'PASS' if same else 'FAIL'}")
    ok &= same

    tr = pd.read_csv(f"Q2/retrain/train_{m}k_avgs.csv")
    tests = {q: pd.read_csv(f"Q2/retrain/test_{m}k_{q}_2026_avgs.csv") for q in ["Q2", "Q3"]}

    def auc(d, c):
        x = pd.to_numeric(d[c], errors="coerce")
        k = x.notna()
        return roc_auc_score(d["CameInQuarter"][k], x[k]) if k.sum() > 100 and x[k].nunique() > 1 else np.nan

    rows = []
    for c in tr.columns[5:]:
        at = auc(tr, c)
        row = {"feature": c, "auc_train": round(at, 3)}
        flag = False
        for q, t in tests.items():
            aq = auc(t, c)
            row[f"auc_{q}"] = round(aq, 3)
            if pd.notna(at) and pd.notna(aq):
                flip = (at - .5) * (aq - .5) < 0 and abs(at - .5) > .03 and abs(aq - .5) > .03
                gap = abs(aq - .5) - abs(at - .5)
                row[f"gap_{q}"] = round(gap, 3)
                flag |= flip or gap >= 0.10
        row["flagged"] = flag
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(f"Q2/retrain/leak_screen_{m}k_avgs.csv", index=False)
    print(f"C screen {m}k: {len(s)} features, {int(s['flagged'].sum())} flagged")
    if s["flagged"].any():
        print(s[s["flagged"]].to_string(index=False))
sys.exit(0 if ok else 1)
