"""
Train a PMS turn-up classifier for a single milestone.

Reads the feature-engineered CSV produced by pmstrainfeatureEng_6.py,
trains a PyTorch binary classifier, and saves:
    models/{milestone}k/model.pt        - trained model weights
    models/{milestone}k/scaler.pkl      - fitted RobustScaler
    models/{milestone}k/feature_list.json - ordered feature names used

Usage:
    python train_milestone.py 70          # train 70k model
    python train_milestone.py 40          # train 40k model
"""

import json
import os
import sys
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# ── Config ──────────────────────────────────────────────────────────
MILESTONE = int(sys.argv[1]) if len(sys.argv) > 1 else 70
FILTER_DATE_CUTOFF = '2026-07-01'  # rows with NextDue < this go to training
EPOCHS = 100
BATCH_SIZE = 128
PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
META_COLS = 5  # first 5 columns are VIN/Vehicle Key/Customer ID/dates — not features

MODEL_DIR = f'models/{MILESTONE}k'
os.makedirs(MODEL_DIR, exist_ok=True)

TRAINING_CSV = f'{MODEL_DIR}/training_features.csv'

print(f'=== Training {MILESTONE}k Model ===')
print(f'Device: {DEVICE}')
print(f'Training CSV: {TRAINING_CSV}')


# ── Load Data ───────────────────────────────────────────────────────
df = pd.read_csv(TRAINING_CSV, low_memory=False)
print(f'Loaded {len(df):,} rows x {len(df.columns)} cols')

# Filter: only rows where the next due date is before the cutoff
due_col = f'Next{MILESTONE}K_Due'
if due_col in df.columns:
    df[due_col] = pd.to_datetime(df[due_col], errors='coerce')
    train_df = df[df[due_col] < FILTER_DATE_CUTOFF]
    print(f'After date filter ({due_col} < {FILTER_DATE_CUTOFF}): {len(train_df):,} rows')
else:
    print(f'[WARN] Column {due_col} not found, using all rows')
    train_df = df

# Nissan only
if 'Franchise_NISSAN' in train_df.columns:
    train_df = train_df.query("Franchise_NISSAN == 1")
    print(f'After Nissan filter: {len(train_df):,} rows')

X_train = train_df.iloc[:, :-1]  # everything except TargetFlag
y_train = train_df['TargetFlag']

# Save feature names (columns after META_COLS, which is what gets scaled)
feature_names = list(X_train.columns[META_COLS:])
with open(f'{MODEL_DIR}/feature_list.json', 'w') as f:
    json.dump(feature_names, f, indent=2)
print(f'Features: {len(feature_names)}')


# ── Outlier Removal ─────────────────────────────────────────────────
iso = IsolationForest(contamination=0.005, random_state=42)
outliers = iso.fit_predict(X_train.iloc[:, META_COLS:])
X_clean = X_train[outliers == 1]
y_clean = y_train[outliers == 1]
print(f'After outlier removal: {len(X_clean):,} rows ({(outliers == -1).sum()} removed)')


# ── Scale ───────────────────────────────────────────────────────────
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_clean.iloc[:, META_COLS:])
with open(f'{MODEL_DIR}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f'Scaler saved to {MODEL_DIR}/scaler.pkl')


# ── Tensors ─────────────────────────────────────────────────────────
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
y_tensor = torch.tensor(y_clean.to_numpy().reshape(-1, 1), dtype=torch.float32).to(DEVICE)

pos_count = int(y_tensor.sum().item())
neg_count = len(y_tensor) - pos_count
# Use pos_weight=1.0 to preserve the natural training class prior instead of balancing to 50/50
pos_weight_value = 1.0
pos_weight = torch.tensor([pos_weight_value]).to(DEVICE)
print(f'Class balance — Pos: {pos_count}, Neg: {neg_count}, pos_weight: {pos_weight_value:.4f}')


# ── Model ───────────────────────────────────────────────────────────
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


model = BinaryClassifier(input_dim=X_scaled.shape[1]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# ── Calibration helpers ─────────────────────────────────────────────
def smooth_labels(y, smoothing=0.1):
    """Apply label smoothing."""
    return y * (1 - smoothing) + 0.5 * smoothing


def brier_loss(logits, targets):
    """Brier score for probability calibration."""
    probs = torch.sigmoid(logits)
    return torch.mean((probs - targets) ** 2)


# ── Training Loop ───────────────────────────────────────────────────
best_loss = np.inf
counter = 0
best_model_state = None

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)

        batch_y_smooth = smooth_labels(batch_y, smoothing=0.1)
        bce = criterion(outputs, batch_y_smooth)
        brier = brier_loss(outputs, batch_y_smooth)
        logit_penalty = 0.01 * torch.mean(outputs ** 2)
        loss = 0.6 * bce + 0.3 * brier + 0.1 * logit_penalty

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f'Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}')

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            sample = model(X_tensor[:2000])
            print(f'  Logits range: [{sample.min().item():.2f}, {sample.max().item():.2f}]')

    if avg_loss < best_loss - 1e-4:
        best_loss = avg_loss
        best_model_state = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            print('Early stopping triggered.')
            break

# ── Save Best Model ─────────────────────────────────────────────────
model.load_state_dict(best_model_state)
torch.save(best_model_state, f'{MODEL_DIR}/model.pt')
print(f'\nModel saved to {MODEL_DIR}/model.pt')
print(f'Best loss: {best_loss:.4f}')

# ── Find Optimal Threshold ──────────────────────────────────────────
from sklearn.metrics import f1_score
model.eval()
with torch.no_grad():
    train_logits = model(X_tensor)
    train_probs = torch.sigmoid(train_logits).cpu().numpy().flatten()
    
y_true = y_tensor.cpu().numpy().flatten()
best_f1 = 0.0
best_thresh = 0.5

for thresh in np.arange(0.05, 0.95, 0.01):
    preds = (train_probs > thresh).astype(int)
    f1 = f1_score(y_true, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = float(thresh)

print(f'Optimal Training Threshold (F1): {best_thresh:.4f} (F1: {best_f1:.4f})')
with open(f'{MODEL_DIR}/threshold.json', 'w') as f:
    json.dump({'threshold': best_thresh}, f)

print(f'=== {MILESTONE}k training complete ===')
