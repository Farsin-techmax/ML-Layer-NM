"""Step 4 leak guard for features_v2. Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe dprime_v2/test_no_label_leak.py 20 30

Check A (code): features_v2.py never references an outcome column.
Check B (flip): build one quarter's features twice -- once as-is, once with every outcome column
  (TargetFlag, CameInQuarter, EarlyCompleter, Actual{m}kDate) scrambled -- the two must be identical.
Check C (screen): single-feature AUC on the training rows vs the held-out Q3 2026 rows (outcome =
  came in during the quarter). Flags a SIGN FLIP (both sides >0.03 away from 0.5, opposite sides) or
  a gap >= 0.10 -- the G6 logic. Read flags with the coverage-drift caveat in CLAUDE.md.
Exit 0 = A and B pass. Writes dprime_v2/leak_screen_{m}k.csv (v3: leak_screen_{m}k_v3.csv; run with --set v3).
"""
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ARGV = sys.argv[1:]
SET = ARGV[ARGV.index("--set") + 1] if "--set" in ARGV else "v2"   # v3 also checks the behaviour block
MILESTONES = [int(x) for x in ARGV if x.isdigit()] or [20, 30]
TAGSET = "" if SET == "v2" else f"_{SET}"
OUTCOME = ["TargetFlag", "CameInQuarter", "EarlyCompleter", "Actual"]

code = open("dprime_v2/features_v2.py", encoding="utf-8").read()
body = code.split('"""', 2)[2]  # skip the module docstring, which names them on purpose
hits = [w for w in OUTCOME if re.search(rf"\b{w}", body)]
print(f"A code: outcome names referenced in features_v2.py body: {hits or 'none'} -> "
      f"{'PASS' if not hits else 'FAIL'}")
ok = not hits

from dprime_v2.features_v2 import load_sources, block_features, prior_listing_table  # noqa: E402
src = load_sources()

for m in MILESTONES:
    lst = pd.read_csv(f"dprime_v2/lists_{m}k.csv",
                      parse_dates=["q_start", "q_end", "cutoff", "burn_date", "last6_date", "avg_date"])
    blk = lst[(lst["CohortQuarter"] == "2025Q4") & (lst["EarlyCompleter"] == 0)].copy()
    r = blk.iloc[0]
    prior = prior_listing_table(lst) if SET == "v3" else None
    a = block_features(src, blk, m, r["cutoff"], r["q_start"], r["q_end"], behaviour=SET == "v3", prior=prior)
    rng = np.random.default_rng(0)
    scr = blk.copy()
    for c in [c for c in scr.columns if any(c.startswith(o) for o in OUTCOME)]:
        scr[c] = rng.permutation(scr[c].values)
    b = block_features(src, scr, m, r["cutoff"], r["q_start"], r["q_end"], behaviour=SET == "v3", prior=prior)
    same = a.sort_index().equals(b.sort_index())
    print(f"B flip {m}k 2025Q4: features identical with outcomes scrambled -> {'PASS' if same else 'FAIL'}")
    ok &= same

    tr = pd.read_csv(f"refactored_test_dir/final_processed_{m}k_dprime_{SET}.csv")
    q3 = pd.read_csv(f"test_sets/test_{m}k_Q3_2026_dprime_{SET}.csv")
    rows = []
    for c in tr.columns[5:]:
        def auc(d):
            x = pd.to_numeric(d[c], errors="coerce")
            k = x.notna()
            return roc_auc_score(d["CameInQuarter"][k], x[k]) if k.sum() > 100 and x[k].nunique() > 1 else np.nan
        at, a3 = auc(tr), auc(q3)
        flip = (pd.notna(at) and pd.notna(a3) and (at - .5) * (a3 - .5) < 0
                and abs(at - .5) > .03 and abs(a3 - .5) > .03)
        rows.append({"feature": c, "auc_train": round(at, 3), "auc_q3": round(a3, 3),
                     "gap": round(abs(a3 - .5) - abs(at - .5), 3) if pd.notna(at) and pd.notna(a3) else np.nan,
                     "sign_flip": flip})
    s = pd.DataFrame(rows)
    s.to_csv(f"dprime_v2/leak_screen_{m}k{TAGSET}.csv", index=False)
    flagged = s[s["sign_flip"] | (s["gap"] >= 0.10)]
    print(f"C screen {m}k: {len(s)} features, {len(flagged)} flagged")
    if len(flagged):
        print(flagged.to_string(index=False))
sys.exit(0 if ok else 1)
