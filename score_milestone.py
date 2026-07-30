"""
Score a prepared cohort with a retrained per-milestone model and save per-VIN probabilities.

Feature derivation mirrors retrain.py's evaluate_test() / eval_test_metrics.py exactly (same
has_x flags, same derived columns, same selected_features ordering), so probabilities match what
those scripts report. Metrics are only printed when the input carries a real TargetFlag.

Usage:
    python score_milestone.py 20 --test testdataq1_q2/20kPMSTestDataQ3_2026.csv
    python score_milestone.py 20 --test <path> --out predictions/20k/scored_Q3_2026.csv --threshold 0.87
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             accuracy_score, brier_score_loss, confusion_matrix)

DEVICE = torch.device("cpu")


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def add_milestone_flags(df, milestone):
    """Same as retrain.py: has_x = 1 if Service_Num >= x, for x in [10..milestone-10]."""
    if 'Service_Num' in df.columns:
        sn = pd.to_numeric(df['Service_Num'], errors='coerce').fillna(0)
        for x in range(10, milestone, 10):
            df[f'has_{x}'] = (sn >= x).astype(int)
    return df


def build_X(df, milestone, selected_features):
    """Reproduce retrain.py's inference-side feature derivation."""
    leak_cols = ['Service_Num', 'Vehicle_Key_Actual_Service', 'Vehicle Age',
                 'Current Age', 'Vehicle Lifetime in Months']
    id_cols = ['VIN', 'Vehicle Key', 'Customer ID', 'Last Service Date - PMS',
               f'Next{milestone}K_Due'] + leak_cols

    df = add_milestone_flags(df, milestone)
    X = df.drop(columns=[c for c in id_cols if c in df.columns], errors='ignore')
    X = X.drop(columns=['TargetFlag'], errors='ignore')

    X['months_to_10k'] = (X['Avg_Service_Interval_PMS'].fillna(6.0)
                          if 'Avg_Service_Interval_PMS' in X.columns else 6.0)
    if 'Last PMS Mileage' in X.columns and 'LastNonPMSMileage' in X.columns:
        X['Last Service Mileage'] = X[['Last PMS Mileage', 'LastNonPMSMileage']].max(axis=1)
    elif 'Last PMS Mileage' in X.columns:
        X['Last Service Mileage'] = X['Last PMS Mileage']
    else:
        X['Last Service Mileage'] = 0
    pms_rev = X.get('PMSRevenue', pd.Series(0, index=X.index))
    X['Max_PMS_Revenue'] = pms_rev
    X['Last_PMS_Revenue'] = pms_rev
    X['Min_PMS_Revenue'] = pms_rev * 0.5
    X['StdDev_PMS_Revenue'] = pms_rev * 0.2

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(0)

    missing = [c for c in selected_features if c not in X.columns]
    for c in missing:
        X[c] = 0
    return X[selected_features], missing


def main():
    p = argparse.ArgumentParser(description="Score a cohort and save per-VIN probabilities")
    p.add_argument("milestone", type=int)
    p.add_argument("--test", required=True, help="Cohort CSV (full process_service_data output)")
    p.add_argument("--model-dir", default=None, help="Default: models/models_alan/{m}k")
    p.add_argument("--out", default=None, help="Default: predictions/{m}k/scored_<testname>.csv")
    p.add_argument("--threshold", type=float, default=None,
                   help="Default: model-dir threshold.json, else 0.5")
    args = p.parse_args()

    m = args.milestone
    mdir = args.model_dir or f"models/models_alan/{m}k"
    for f in ['best_model.pt', 'selected_features.json', 'imputer.joblib', 'scaler.joblib']:
        if not os.path.exists(os.path.join(mdir, f)):
            sys.exit(f"Missing {os.path.join(mdir, f)} -- run: python retrain.py {m}")
    if not os.path.exists(args.test):
        sys.exit(f"Missing test file {args.test}")

    threshold = args.threshold
    if threshold is None:
        tp = os.path.join(mdir, 'threshold.json')
        threshold = json.load(open(tp))['threshold'] if os.path.exists(tp) else 0.5

    selected_features = json.load(open(os.path.join(mdir, 'selected_features.json')))
    imputer = joblib.load(os.path.join(mdir, 'imputer.joblib'))
    scaler = joblib.load(os.path.join(mdir, 'scaler.joblib'))
    model = BinaryClassifier(input_dim=len(selected_features)).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(mdir, 'best_model.pt'), map_location=DEVICE))
    model.eval()

    df = pd.read_csv(args.test, low_memory=False)
    vin = df['VIN'] if 'VIN' in df.columns else pd.Series(df.index, name='VIN')
    y = df['TargetFlag'].astype(int).values if 'TargetFlag' in df.columns else None

    X, missing = build_X(df, m, selected_features)
    print(f"{m}k | rows={len(X)} | features={len(selected_features)} | threshold={threshold:.4f}")
    if missing:
        # Zero-filled columns are the dominant cause of bad metrics -- surface, don't hide.
        print(f"WARNING: {len(missing)}/{len(selected_features)} features absent from the test "
              f"file and zero-filled: {missing[:15]}{' ...' if len(missing) > 15 else ''}")

    Xs = scaler.transform(imputer.transform(X))
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.FloatTensor(Xs).to(DEVICE)).squeeze()).cpu().numpy()
    probs = np.atleast_1d(probs)
    pred = (probs > threshold).astype(int)

    out = pd.DataFrame({'VIN': vin.values, 'prob_turnup': probs, 'pred': pred})
    if f'Next{m}K_Due' in df.columns:
        out[f'Next{m}K_Due'] = df[f'Next{m}K_Due'].values
    if y is not None:
        out['TargetFlag'] = y
    out = out.sort_values('prob_turnup', ascending=False).reset_index(drop=True)

    out_path = args.out or os.path.join(
        "predictions", f"{m}k",
        f"scored_{os.path.splitext(os.path.basename(args.test))[0]}.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} scored rows -> {out_path}")
    print(f"Predicted turn-up: {pred.sum()} ({100 * pred.mean():.1f}%) | "
          f"mean prob {probs.mean():.4f}")

    if y is None:
        print("No TargetFlag in input -- scoring only, no metrics.")
        return
    if len(set(y)) < 2:
        print(f"TargetFlag is constant ({y[0]}) -- metrics undefined. "
              "Expected for a window with no outcomes yet.")
        return

    print(f"\nAccuracy  {100 * accuracy_score(y, pred):.2f}")
    print(f"Precision {100 * precision_score(y, pred, zero_division=0):.2f}")
    print(f"Recall    {100 * recall_score(y, pred, zero_division=0):.2f}")
    print(f"F1        {100 * f1_score(y, pred, zero_division=0):.2f}")
    print(f"AUC       {roc_auc_score(y, probs):.4f}")
    print(f"Brier     {brier_score_loss(y, probs):.4f}")
    print(f"Confusion (rows=actual 0/1, cols=pred 0/1):\n{confusion_matrix(y, pred)}")


if __name__ == "__main__":
    main()
