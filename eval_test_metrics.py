"""
Evaluate the retrained per-milestone models on the Q1 and Q2 test sets and report the
full benchmark-style metrics, including the confusion-matrix breakdown.

Feature derivation, the has_x milestone flags and the model definition are shared with retrain.py
and score_milestone.py via pms_model.py, and the threshold here is 0.5, so the numbers match what
training reported (score_milestone.py uses the tuned threshold.json instead and will differ).

Usage:
    python eval_test_metrics.py            # all milestones, Q1 + Q2
    python eval_test_metrics.py 30         # single milestone
"""
import os
import sys
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

from pms_model import (build_inference_matrix, features_path, load_artifacts, model_dir,
                       predict_proba, resolve_path, split_missing)

MILESTONES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
DEVICE = torch.device("cpu")


def evaluate(milestone, quarter_label, test_path):
    mdir = model_dir(milestone)
    if not (os.path.exists(test_path) and os.path.exists(os.path.join(mdir, 'best_model.pt'))
            and os.path.exists(features_path(milestone))):
        return None

    model, imputer, scaler, selected_features = load_artifacts(milestone, DEVICE)

    test_df = pd.read_csv(test_path, low_memory=False)
    y = test_df['TargetFlag'].astype(int).values

    X, missing = build_inference_matrix(test_df, milestone, selected_features)
    # This used to be discarded (`X, _ = ...`), so a heavily zero-filled evaluation looked exactly
    # like a clean one. Absent one-hot categories are harmless; genuine gaps invalidate the row.
    _expected, gaps = split_missing(missing, test_df.columns) if missing else ([], [])
    if gaps:
        print(f"  WARNING {milestone}k {quarter_label}: {len(gaps)}/{len(selected_features)} "
              f"features GENUINELY missing from {test_path} and zero-filled: "
              f"{gaps[:10]}{' ...' if len(gaps) > 10 else ''}")
    probs = predict_proba(model, imputer, scaler, X, DEVICE)
    pred = (probs > 0.5).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    return dict(
        Milestone=f"{milestone}K", Quarter=quarter_label, Records=len(y),
        Accuracy=round(100 * accuracy_score(y, pred), 2),
        Precision=round(100 * precision_score(y, pred, zero_division=0), 2),
        Recall=round(100 * recall_score(y, pred, zero_division=0), 2),
        AUC=round(roc_auc_score(y, probs), 4) if len(set(y)) > 1 else float('nan'),
        F1=round(100 * f1_score(y, pred, zero_division=0), 2),
        TP=tp, TN=tn, FP=fp, FN=fn,
        Pred_Pos=tp + fp, Actual_Pos=tp + fn,
        ZeroFill=len(gaps),          # genuine schema gaps -- non-zero invalidates the row
        DummyFill=len(_expected),    # absent one-hot categories -- harmless, 0 is correct
    )


def main():
    targets = [int(sys.argv[1])] if len(sys.argv) > 1 else MILESTONES
    rows = []
    for m in targets:
        # resolve_path also looks under tests_alan/, which is where testdataq1_q2/ actually lives.
        # Without it every path missed and this script silently printed "No results."
        for q, path in [("Q1", resolve_path(f"testdataq1_q2/{m}kPMSTestDataQ1.csv")),
                        ("Q2", resolve_path(f"testdataq1_q2/{m}kPMSTestDataQ2_2026v1.csv"))]:
            r = evaluate(m, q, path)
            if r:
                rows.append(r)
    if not rows:
        print("No results.")
        return
    df = pd.DataFrame(rows)
    # ZeroFill = model inputs the test pipeline never produced; a non-zero value invalidates the
    # row's metrics, so read it before the accuracy. DummyFill = one-hot categories that simply did
    # not occur in the cohort; 0 is the correct value there and the count is informational.
    cols = ['Milestone', 'Quarter', 'Records', 'Accuracy', 'Precision', 'Recall', 'AUC', 'F1',
            'TP', 'TN', 'FP', 'FN', 'Pred_Pos', 'Actual_Pos', 'ZeroFill', 'DummyFill']
    df = df[cols]
    for q in ['Q1', 'Q2']:
        sub = df[df.Quarter == q]
        if len(sub):
            print(f"\n================ {q} TEST ================")
            print(sub.drop(columns=['Quarter']).to_string(index=False))
    df.to_csv("retrained_test_metrics_final.csv", index=False)
    print("\nSaved -> retrained_test_metrics_final.csv")


if __name__ == "__main__":
    main()
