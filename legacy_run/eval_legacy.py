"""
The evaluation /legacy has never had (see the plan, Step 4). Scores both trained arms
(legacy_asis, legacy_noleak) -- produced by train_legacy.py -- on the same Q1/Q2 2026 test cohorts
the current 20k production model is measured on, so the two are comparable.

Reused from pms_model.py, not reimplemented: predict_proba(), split_missing(), family_of().

NOT reused: build_inference_matrix(). That function's X = df.drop(columns=id_cols(milestone)) step
unconditionally drops LEAK_COLS (Service_Num, Vehicle_Key_Actual_Service, Vehicle Age, Current Age,
Vehicle Lifetime in Months) from the frame BEFORE the selected-features reindex -- for current
production models this is a no-op (retrain.py's DROP_COLS already excludes those names from
selected_features, confirmed at retrain.py:142/254, so they were never in the list build_inference_
matrix reindexes onto). But legacy_asis's selected feature list DOES include those 5 names (that is
the entire point of the arm), so calling build_inference_matrix on it would zero-fill exactly the
columns under test, silently forcing legacy_asis to a near-random score regardless of whether the
leak actually transfers to these test sets. That would misreport the plan's central question. So
this script does the column alignment itself (reindex test cols onto the trained feature list,
coerce numeric, fillna 0) -- the same alignment build_inference_matrix does MINUS the LEAK_COLS
drop -- and still reuses split_missing()/family_of() for the zero-fill reporting and predict_proba()
for scoring. See the final report for this flagged as a deliberate deviation from the plan's literal
"reuse build_inference_matrix()" instruction, with the reasoning above.

predict_proba() requires a fitted imputer; legacy never fits one (its own commented-out
`X_train = linimput(X_train)`). A no-op SimpleImputer(strategy='constant', fill_value=0) satisfies
the signature -- by the time X reaches it every value is already numeric and NaN-free (coerced here
before predict_proba is called), so the impute step is a pass-through.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from pms_model import BinaryClassifier, split_missing, family_of, predict_proba  # noqa: E402

MILESTONE = 20
ARMS = ['legacy_asis', 'legacy_noleak']
QUARTERS = [('Q1', 'test_sets/test_20k_Q1_2026.csv'), ('Q2', 'test_sets/test_20k_Q2_2026.csv')]

THRESH_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)  # 0.05 .. 0.95 step 0.01, per the plan


def metrics_at(y, probs, thr):
    preds = (probs >= thr).astype(int)
    acc = accuracy_score(y, preds)
    # zero_division guards: at extreme thresholds every pred can be one class
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return dict(threshold=thr, accuracy=acc, precision=prec, recall=rec, f1=f1,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def best_f1_threshold(y, probs):
    best = None
    for t in THRESH_GRID:
        m = metrics_at(y, probs, t)
        if best is None or m['f1'] > best['f1']:
            best = m
    return best


results = []
zero_fill_report = []

for arm in ARMS:
    arm_dir = f'legacy_run/{arm}_{MILESTONE}k'
    cols_path = os.path.join(arm_dir, 'feature_columns.json')
    if not os.path.exists(cols_path):
        print(f"SKIP {arm}: {cols_path} not found -- train it first with train_legacy.py")
        continue

    cols = json.load(open(cols_path))
    scaler = joblib.load(os.path.join(arm_dir, 'scaler.joblib'))
    model = BinaryClassifier(input_dim=len(cols))
    model.load_state_dict(torch.load(os.path.join(arm_dir, 'best_model.pt'), map_location='cpu'))
    model.eval()

    # No-op imputer: X is already numeric/NaN-free by the time predict_proba runs (see docstring).
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    imputer.fit(np.zeros((2, len(cols))))

    print(f"\n{'='*70}\nARM: {arm}  ({len(cols)} trained features)\n{'='*70}")

    for qlabel, path in QUARTERS:
        if not os.path.exists(path):
            print(f"SKIP {arm}/{qlabel}: {path} not found")
            continue
        test_df = pd.read_csv(path, low_memory=False)
        y = test_df['TargetFlag'].values

        missing = [c for c in cols if c not in test_df.columns]
        expected, gaps = split_missing(missing, test_df.columns)
        zero_fill_report.append(dict(arm=arm, quarter=qlabel, n_features=len(cols),
                                     n_missing=len(missing), n_dummyfill=len(expected),
                                     n_zerofill=len(gaps), zerofill_cols=gaps))

        print(f"\n--- {arm} / {qlabel} ({path}, {len(test_df)} rows) ---")
        print(f"  Missing from test set: {len(missing)}/{len(cols)} "
              f"(DummyFill absent-category: {len(expected)}, ZeroFill real-gap: {len(gaps)})")
        if gaps:
            print(f"  ZeroFill (REAL GAPS, must be near zero or comparison is void): "
                  f"{gaps[:20]}{' ...' if len(gaps) > 20 else ''}")

        # Align: reindex test columns onto the trained feature list (this IS the "intersection"
        # alignment build_inference_matrix would do, minus its LEAK_COLS strip -- see docstring).
        X = test_df.reindex(columns=cols, fill_value=0)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        probs = predict_proba(model, imputer, scaler, X.values)
        auc = roc_auc_score(y, probs)
        m50 = metrics_at(y, probs, 0.5)
        mbest = best_f1_threshold(y, probs)

        print(f"  AUC: {auc:.4f}")
        print(f"  @0.5   acc={m50['accuracy']:.4f} prec={m50['precision']:.4f} "
              f"rec={m50['recall']:.4f} f1={m50['f1']:.4f} "
              f"TN={m50['tn']} FP={m50['fp']} FN={m50['fn']} TP={m50['tp']}")
        print(f"  @{mbest['threshold']:.2f}(F1-tuned) acc={mbest['accuracy']:.4f} "
              f"prec={mbest['precision']:.4f} rec={mbest['recall']:.4f} f1={mbest['f1']:.4f} "
              f"TN={mbest['tn']} FP={mbest['fp']} FN={mbest['fn']} TP={mbest['tp']}")

        results.append(dict(arm=arm, quarter=qlabel, n_test_rows=len(test_df), auc=auc,
                            **{f'{k}_50': v for k, v in m50.items()},
                            **{f'{k}_best': v for k, v in mbest.items()}))

# --- Summary table ---
print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
res_df = pd.DataFrame(results)
if not res_df.empty:
    print(res_df.to_string(index=False))
    res_df.to_csv('legacy_run/eval_results.csv', index=False)
    print("\nWritten: legacy_run/eval_results.csv")

zf_df = pd.DataFrame(zero_fill_report)
if not zf_df.empty:
    print("\nZero-fill report (DummyFill = absent one-hot category, harmless; "
          "ZeroFill = real pipeline gap, invalidates the comparison if large):")
    print(zf_df[['arm', 'quarter', 'n_features', 'n_missing', 'n_dummyfill', 'n_zerofill']]
          .to_string(index=False))
    zf_df.to_csv('legacy_run/eval_zerofill_report.csv', index=False)
    print("Written: legacy_run/eval_zerofill_report.csv")
