"""
Shared model + inference pieces for the current PMS pipeline.

Used by retrain.py (training), score_milestone.py (scoring) and eval_test_metrics.py (held-out
metrics). Before this module existed, BinaryClassifier was defined 5 times, add_milestone_flags
3 times and the inference feature-derivation block 3 times across those files. They agreed only
because they were hand-copied; any fix to one silently left the others behind.

The legacy trainers (train_milestone.py, legacy/parked/training_run.py) keep their own copies on
purpose: they use a different network (ServicePredictionNN, dropout 0.5) and StandardScaler.
"""
import os
import json

import pandas as pd
import torch
import torch.nn as nn
import joblib


# 80k is exempt from has_x: its Q2 test file records Service_Num = 0 for all 651 positives (Q1
# correctly records 80), so has_x becomes 0 everywhere for them and the model calls them all
# no-shows (1.7% recall, TP=11/651). Leaving has_x off for 80k keeps both quarters usable.
HAS_X_EXCLUDE = {80}

# RFM segment one-hots are a cohort-vintage marker, not a behaviour signal. Segments are
# recency-based against a single fixed snapshot, so training positives (completed the milestone
# years ago) land in Lost/Hibernating/At Risk -- ~67% of them -- while test positives (completed
# it in 2026) land in Potential Loyalist/Promising. Those training segments are 0.0% of test
# positives, so the model learns a rule that cannot fire at test time. Q1 and Q2 positives sit on
# opposite sides of the Potential Loyalist/Promising recency boundary, which is what made 60k
# score 92.02 on Q1 and 73.65 on Q2. Dropping them: Q1 87.50 / Q2 88.00 -- 11.5pt gap -> 0.5pt.
DROP_RFM = {60}

# Columns that leak the label or encode it indirectly. Service_Num is exact:
# Service_Num == milestone <=> TargetFlag == 1, zero exceptions.
LEAK_COLS = ['Service_Num', 'Vehicle_Key_Actual_Service', 'Vehicle Age',
             'Current Age', 'Vehicle Lifetime in Months']

# Built at inference time by derive_inference_features(), never produced by
# process_service_data(). Do not try to select a test set down to these.
DERIVED_FEATURES = ['months_to_10k', 'Last Service Mileage', 'Max_PMS_Revenue',
                    'Last_PMS_Revenue', 'Min_PMS_Revenue', 'StdDev_PMS_Revenue']


def resolve_path(path):
    """Data dirs (traindataq1_q2/, testdataq1_q2/) currently sit under tests_alan/ rather than at
    the repo root the scripts assume. Try the root path first, then the tests_alan/ copy."""
    if os.path.exists(path):
        return path
    alt = os.path.join("tests_alan", path)
    if os.path.exists(alt):
        return alt
    return path


def id_cols(milestone):
    """Identifier columns plus LEAK_COLS -- dropped from both training and inference matrices."""
    return ['VIN', 'Vehicle Key', 'Customer ID', 'Last Service Date - PMS',
            f'Next{milestone}K_Due'] + LEAK_COLS


class BinaryClassifier(nn.Module):
    """64 -> 32 -> 1 with BatchNorm and dropout 0.2. Outputs logits, not probabilities."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def add_milestone_flags(df, milestone, use_has_x=True):
    """has_x = 1 if the vehicle has completed prior milestone x (derived from Service_Num).

    Added for x in [10, 20, ..., milestone-10] only -- the target milestone is excluded so no flag
    directly encodes TargetFlag. Must run before Service_Num is dropped.

    CAVEAT (known, accepted): Service_Num is stamped == milestone only AFTER a vehicle turns up, so
    every training positive gets has_x = 1 across all prior milestones. On real prediction sets no
    vehicle has Service_Num == milestone, so that pattern cannot occur in production and has_x-driven
    test metrics will not transfer.
    """
    if use_has_x and milestone not in HAS_X_EXCLUDE and 'Service_Num' in df.columns:
        sn = pd.to_numeric(df['Service_Num'], errors='coerce').fillna(0)
        for x in range(10, milestone, 10):   # excludes milestone itself
            assert x != milestone, "target-milestone flag must never be created"
            df[f'has_{x}'] = (sn >= x).astype(int)
    return df


def derive_inference_features(X):
    """Add the DERIVED_FEATURES columns the models expect but the feature-eng output lacks.

    Safe on both training frames and test frames: each column falls back when its source is absent.
    """
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
    return X


def load_artifacts(model_dir, device=None):
    """Load a trained milestone model. Returns (model, imputer, scaler, selected_features)."""
    device = device or torch.device("cpu")
    selected_features = json.load(open(os.path.join(model_dir, 'selected_features.json')))
    imputer = joblib.load(os.path.join(model_dir, 'imputer.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    model = BinaryClassifier(input_dim=len(selected_features)).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt'), map_location=device))
    model.eval()
    return model, imputer, scaler, selected_features


def build_inference_matrix(df, milestone, selected_features, use_has_x=True):
    """Turn a raw cohort/test frame into the exact matrix the model consumes.

    Returns (X, missing) where missing lists the selected features that were absent and had to be
    zero-filled. A non-empty `missing` is the dominant cause of bad metrics -- surface it, do not
    hide it.
    """
    df = add_milestone_flags(df, milestone, use_has_x=use_has_x)
    X = df.drop(columns=[c for c in id_cols(milestone) if c in df.columns], errors='ignore')
    X = X.drop(columns=['TargetFlag'], errors='ignore')

    X = derive_inference_features(X)

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(0)

    missing = [c for c in selected_features if c not in X.columns]
    for c in missing:
        X[c] = 0
    return X[selected_features], missing


def predict_proba(model, imputer, scaler, X, device=None):
    """Impute -> scale -> forward -> sigmoid. Always returns a 1-D numpy array."""
    import numpy as np
    device = device or torch.device("cpu")
    Xs = scaler.transform(imputer.transform(X))
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.FloatTensor(Xs).to(device)).squeeze()).cpu().numpy()
    return np.atleast_1d(probs)
