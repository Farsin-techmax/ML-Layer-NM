"""
Evaluate the retrained per-milestone models on the Q1 and Q2 test sets and report the
full benchmark-style metrics, including the confusion-matrix breakdown.

Mirrors retrain.py's evaluate_test() exactly (same feature derivation, same has_x milestone
flags, same 0.5 threshold) so the numbers match what training reported.

Usage:
    python eval_test_metrics.py            # all milestones, Q1 + Q2
    python eval_test_metrics.py 30         # single milestone
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

MILESTONES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
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


def evaluate(milestone, quarter_label, test_path):
    mdir = f"models/models_alan/{milestone}k"
    if not (os.path.exists(test_path) and os.path.exists(os.path.join(mdir, 'best_model.pt'))):
        return None

    selected_features = json.load(open(os.path.join(mdir, 'selected_features.json')))
    imputer = joblib.load(os.path.join(mdir, 'imputer.joblib'))
    scaler = joblib.load(os.path.join(mdir, 'scaler.joblib'))
    model = BinaryClassifier(input_dim=len(selected_features)).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(mdir, 'best_model.pt'), map_location=DEVICE))
    model.eval()

    leak_cols = ['Service_Num', 'Vehicle_Key_Actual_Service', 'Vehicle Age',
                 'Current Age', 'Vehicle Lifetime in Months']
    id_cols = ['VIN', 'Vehicle Key', 'Customer ID', 'Last Service Date - PMS',
               f'Next{milestone}K_Due'] + leak_cols

    test_df = pd.read_csv(test_path, low_memory=False)
    test_df = add_milestone_flags(test_df, milestone)
    test_df = test_df.drop(columns=[c for c in id_cols if c in test_df.columns], errors='ignore')

    y = test_df['TargetFlag'].astype(int).values
    X = test_df.drop(columns=['TargetFlag'], errors='ignore')

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
    for c in selected_features:
        if c not in X.columns:
            X[c] = 0
    X = X[selected_features]

    Xs = scaler.transform(imputer.transform(X))
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.FloatTensor(Xs).to(DEVICE)).squeeze()).cpu().numpy()
    probs = np.atleast_1d(probs)
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
    )


def main():
    targets = [int(sys.argv[1])] if len(sys.argv) > 1 else MILESTONES
    rows = []
    for m in targets:
        for q, path in [("Q1", f"testdataq1_q2/{m}kPMSTestDataQ1.csv"),
                        ("Q2", f"testdataq1_q2/{m}kPMSTestDataQ2_2026v1.csv")]:
            r = evaluate(m, q, path)
            if r:
                rows.append(r)
    if not rows:
        print("No results.")
        return
    df = pd.DataFrame(rows)
    cols = ['Milestone', 'Quarter', 'Records', 'Accuracy', 'Precision', 'Recall', 'AUC', 'F1',
            'TP', 'TN', 'FP', 'FN', 'Pred_Pos', 'Actual_Pos']
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
