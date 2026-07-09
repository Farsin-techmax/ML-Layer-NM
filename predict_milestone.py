"""
Run inference for a single PMS milestone.

Loads the prediction dataset produced by predservicemil_4_refactored.py,
applies the trained model, and saves probability scores.

Outputs:
    predictions/{milestone}k/scored_predictions.csv

Usage:
    python predict_milestone.py 70        # predict for 70k
    python predict_milestone.py 40        # predict for 40k
"""

import json
import os
import sys
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import argparse

# ── Config ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('milestone', type=int, default=70, nargs='?')
parser.add_argument('--threshold', type=float, default=None, help='Override probability threshold for positive class')
parser.add_argument('--target-rate', type=float, default=None, help='Force a specific percentage of data to be predicted positive (e.g. 0.64)')
args = parser.parse_args()

MILESTONE = args.milestone
OVERRIDE_THRESHOLD = args.threshold
MODEL_DIR = f'models/{MILESTONE}k'
PRED_DIR = f'predictions/{MILESTONE}k'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
META_COLS = 5  # VIN, Last Service Date, Vehicle Key, Customer ID, NextDue

PRED_CSV = f'{PRED_DIR}/prediction_dataset.csv'

print(f'=== Predicting {MILESTONE}k ===')
print(f'Model dir: {MODEL_DIR}')
print(f'Pred CSV: {PRED_CSV}')


# ── Validate files exist ────────────────────────────────────────────
for fpath in [f'{MODEL_DIR}/model.pt', f'{MODEL_DIR}/scaler.pkl',
              f'{MODEL_DIR}/feature_list.json', PRED_CSV]:
    if not os.path.exists(fpath):
        print(f'[ERROR] Missing: {fpath}')
        sys.exit(1)


# ── Load artifacts ──────────────────────────────────────────────────
with open(f'{MODEL_DIR}/feature_list.json') as f:
    feature_names = json.load(f)
print(f'Features expected: {len(feature_names)}')

with open(f'{MODEL_DIR}/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# ── Model definition (must match train_milestone.py) ────────────────
class BinaryClassifier(nn.Module):
    """Simple feedforward binary classifier with BatchNorm."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


model = BinaryClassifier(input_dim=len(feature_names))
model.load_state_dict(torch.load(f'{MODEL_DIR}/model.pt', map_location=DEVICE))
model.to(DEVICE)
model.eval()
print('Model loaded.')


# ── Load prediction data ────────────────────────────────────────────
predfuture = pd.read_csv(PRED_CSV, low_memory=False)
print(f'Prediction dataset: {len(predfuture):,} rows x {len(predfuture.columns)} cols')

# Nissan only
if 'Franchise_NISSAN' in predfuture.columns:
    predfuture = predfuture.query("Franchise_NISSAN == 1").copy()
    print(f'After Nissan filter: {len(predfuture):,} rows')

if len(predfuture) == 0:
    print('[WARN] No rows after filtering. Nothing to predict.')
    sys.exit(0)

# Fix leakage artifacts for vehicles that already turned up in the data
if 'TargetFlag' in predfuture.columns:
    mask = predfuture['TargetFlag'] == 1
    if 'PMS_Delay' in predfuture.columns:
        predfuture.loc[mask, 'PMS_Delay'] -= 1
    if 'Vehicle_Key_Actual_Service' in predfuture.columns:
        predfuture.loc[mask, 'Vehicle_Key_Actual_Service'] -= 1


# ── Align features ─────────────────────────────────────────────────
# Add any features in training that are missing in prediction (as 0)
missing_feats = list(set(feature_names) - set(predfuture.columns))
if missing_feats:
    print(f'Missing features (zero-filled): {missing_feats}')
    predfuture[missing_feats] = 0

X_sample = predfuture[feature_names]
X_scaled = scaler.transform(X_sample)


# ── Inference ───────────────────────────────────────────────────────
def predict_proba(mdl, X, temperature=1.0):
    """Get calibrated probabilities from model logits."""
    mdl.eval()
    with torch.no_grad():
        logits = mdl(X.to(DEVICE))
        probs = torch.sigmoid(logits / temperature)
    return probs.cpu()


X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
probs = predict_proba(model, X_tensor)
prob_array = probs.numpy().flatten()

if args.target_rate is not None:
    THRESHOLD = np.percentile(prob_array, 100 * (1 - args.target_rate))
    print(f'Using target-rate override ({args.target_rate*100}%): computed threshold = {THRESHOLD:.5f}')
elif OVERRIDE_THRESHOLD is not None:
    THRESHOLD = OVERRIDE_THRESHOLD
    print(f'Using override threshold: {THRESHOLD:.5f}')
else:
    # Load optimal threshold from training
    thresh_file = f'{MODEL_DIR}/threshold.json'
    if os.path.exists(thresh_file):
        with open(thresh_file, 'r') as f:
            thresh_data = json.load(f)
        THRESHOLD = thresh_data.get('threshold', 0.5)
        print(f'Loaded optimal F1 threshold from training: {THRESHOLD:.5f}')
    else:
        THRESHOLD = 0.5
        print(f'Warning: {thresh_file} not found. Using default threshold 0.5')

preds = (probs > THRESHOLD).float()

predfuture['Predicted_Prob'] = prob_array.round(5)
predfuture['Predicted_Class'] = preds.numpy().flatten().astype(int)
predfuture['Milestone'] = f'{MILESTONE}k'

# ── Analyze Probabilities ───────────────────────────────────────────
print('\nProbability Deciles:')
for q in range(10, 100, 10):
    print(f'  {q}th percentile: {np.percentile(prob_array, q):.4f}')
print(f'  Mean Prob: {np.mean(prob_array):.4f}')


# ── Save ────────────────────────────────────────────────────────────
out_path = f'{PRED_DIR}/scored_predictions.csv'
predfuture.to_csv(out_path, index=False)
print(f'\nScored predictions saved: {out_path}')
print(f'Total vehicles: {len(predfuture):,}')
print(f'Predicted turn-up (threshold={THRESHOLD:.4f}): {int(preds.sum().item()):,} ({preds.mean().item()*100:.1f}%)')
print(f'=== {MILESTONE}k prediction complete ===')
