"""
Driver for legacy/training_run.py's 235-line fragment (see the plan at
C:\\Users\\panga\\.claude\\plans\\in-simple-terms-how-floofy-avalanche.md, Step 3). That fragment has
no imports and no main() -- line 1 references a `df` that is never defined anywhere in the file. This
script supplies exactly what's missing (imports, the CSV read, the two-arm column selection, artifact
saving) and otherwise keeps every modeling line VERBATIM from the fragment:
  - X_train = train_df.iloc[:, :-1]              (drop TargetFlag only)
  - IsolationForest(contamination=0.005, random_state=42) on X.iloc[:, 5:]
  - RobustScaler fit on the cleaned X.iloc[:, 5:]
  - BinaryClassifier 64->32->1, BatchNorm, Dropout(0.2)      (imported from pms_model -- identical
    architecture, confirmed by reading both class definitions side by side; not reimplemented here)
  - BCEWithLogitsLoss(pos_weight) + label smoothing 0.1, combined 0.6*bce + 0.3*brier + 0.1*logit_penalty
  - AdamW lr=3e-4, weight_decay=1e-2; ReduceLROnPlateau(factor=0.5, patience=5)
  - Early stop on TRAINING loss (avg_loss), patience=10 -- NO validation split, matching the fragment
    exactly (its test_df / X_test / y_test lines are commented out in the original).

Two arms, differing ONLY in which columns are fed to the model:
  legacy_asis   -- iloc[:, 5:] verbatim, Service_Num included (the leak, per the plan's Context)
  legacy_noleak -- same, minus Service_Num, Vehicle_Key_Actual_Service, Vehicle Age, Current Age,
                   Vehicle Lifetime in Months, Service Frequency

Usage:
    PMS_SEED=42 ./venv/Scripts/python.exe legacy_run/train_legacy.py 20 legacy_asis 2025-12-31
    PMS_SEED=42 ./venv/Scripts/python.exe legacy_run/train_legacy.py 20 legacy_noleak 2025-12-31
"""
import sys
import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from pms_model import BinaryClassifier  # noqa: E402 -- identical arch to the fragment's own class

# --- Seed: unseeded runs on an identical matrix differed by 6.7 accuracy points (measured on the
# current pipeline, 2026-08-12) -- carrying that lesson over here rather than re-learning it.
SEED = int(os.environ.get('PMS_SEED', 42))
np.random.seed(SEED)
torch.manual_seed(SEED)

MILESTONE = int(sys.argv[1])            # 20
ARM = sys.argv[2]                       # 'legacy_asis' | 'legacy_noleak'
CUTOFF = sys.argv[3] if len(sys.argv) > 3 else '2025-12-31'

LEAK_6 = ['Service_Num', 'Vehicle_Key_Actual_Service', 'Vehicle Age', 'Current Age',
          'Vehicle Lifetime in Months', 'Service Frequency']

MATRIX_PATH = f'validatecode/finalmerged{MILESTONE}kQ3.csv'
print(f"Loading {MATRIX_PATH} ...")
df = pd.read_csv(MATRIX_PATH, low_memory=False)

due_col = f'Next{MILESTONE}K_Due'
df[due_col] = pd.to_datetime(df[due_col], format='mixed', dayfirst=True, errors='coerce')
cutoff_ts = pd.Timestamp(CUTOFF) + pd.Timedelta(days=1)  # '<' cutoff+1day == '<=' cutoff
# Replaces training_run.py line 1's hardcoded: train_df = df[df['Next70K_Due'] < '2025-07-01']
train_df = df[df[due_col] < cutoff_ts].reset_index(drop=True)
print(f"train_df rows after {due_col} < {CUTOFF}: {len(train_df)} (of {len(df)} total)")

# =========================
# training_run.py verbatim from here (lines 4-236 of legacy/training_run.py), except:
#   - X_train (renamed X_train_full below) drops the LEAK_6 columns first for the noleak arm
#   - artifact saving replaces the ad hoc validatecode/scaler.joblib dump
# =========================
X_train_full = train_df.iloc[:, :-1]   # drop TargetFlag only, matches iloc[:,:-1]
if ARM == 'legacy_noleak':
    drop_now = [c for c in LEAK_6 if c in X_train_full.columns]
    print(f"legacy_noleak: dropping {drop_now}")
    X_train_full = X_train_full.drop(columns=drop_now)
elif ARM != 'legacy_asis':
    sys.exit(f"Unknown arm {ARM!r}; expected 'legacy_asis' or 'legacy_noleak'")

y_train = train_df['TargetFlag']

feature_cols = list(X_train_full.iloc[:, 5:].columns)
print(f"Arm={ARM}: {len(feature_cols)} feature columns fed to the model (after the 5 ID columns)")

iso = IsolationForest(contamination=0.005, random_state=42)
outliers = iso.fit_predict(X_train_full.iloc[:, 5:])

# Filter out outliers
X_clean = X_train_full[outliers == 1]
y_clean = y_train[outliers == 1]
print(f"After IsolationForest: {len(X_clean)} rows kept of {len(X_train_full)} "
      f"({(outliers == -1).sum()} outliers dropped)")

# --- Glue, not a recipe change: the fragment's own `X_train= linimput(X_train)` line is commented
# out, i.e. it assumes no NaNs reach this point. process_service_data()'s IterativeImputer only
# covers columns present BEFORE the later merges (bodyshop counts, gower clusters, digital
# sessions), so a few tail columns can still carry NaN after those left-joins. Filling them with 0
# here is the direct equivalent of predict_proba's no-op imputer on the eval side (see eval_legacy.py)
# -- it lets sklearn's fit_transform run at all without touching which columns are fed or how they're
# scaled.
feat_frame = X_clean.iloc[:, 5:].apply(pd.to_numeric, errors='coerce')
n_nan = int(feat_frame.isna().sum().sum())
if n_nan:
    print(f"NOTE: {n_nan} residual NaN cells in the feature block (post-merge columns the "
          f"feature-eng imputer never covered) -- filling with 0 so IsolationForest/RobustScaler "
          f"can run. Not part of the legacy recipe; necessary glue only.")
    feat_frame = feat_frame.fillna(0)

scaler = RobustScaler()
X_train = scaler.fit_transform(feat_frame)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_clean.to_numpy().reshape(-1, 1), dtype=torch.float32).to(device)

# =========================
# PREP MODEL
# =========================
model = BinaryClassifier(input_dim=X_train.shape[1]).to(device)

# =========================
# CLASS IMBALANCE HANDLING
# =========================
y_np = y_train_tensor.cpu().numpy()
pos_count = np.sum(y_np == 1)
neg_count = np.sum(y_np == 0)
pos_weight_value = neg_count / (pos_count + 1e-6)
pos_weight = torch.tensor([pos_weight_value]).to(device)
print(f"Positive count: {pos_count}, Negative count: {neg_count}")
print(f"Using pos_weight: {pos_weight_value:.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# =========================
# OPTIMIZER + SCHEDULER
# =========================
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# =========================
# EARLY STOPPING
# =========================
best_loss = np.inf
patience = 10
counter = 0

# =========================
# DATALOADER
# =========================
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

# =========================
# TRAINING LOOP
# =========================
epochs = 100


def smooth_labels(y, smoothing=0.1):
    return y * (1 - smoothing) + 0.5 * smoothing


def brier_loss(logits, targets):
    probs = torch.sigmoid(logits)
    return torch.mean((probs - targets) ** 2)


best_model_state = None
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

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

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            sample_logits = model(X_train_tensor[:2000].to(device))
            if (epoch + 1) % 10 == 0:
                print(f"Logits range: [{sample_logits.min().item():.2f}, {sample_logits.max().item():.2f}]")

    # EARLY STOPPING on TRAINING loss (matches the fragment -- no validation split exists)
    if avg_loss < best_loss - 1e-4:
        best_loss = avg_loss
        best_model_state = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(best_model_state)
print(f"Final training loss: {best_loss:.4f}")

# =========================
# SAVE ARTIFACTS (not part of the fragment -- it never saved anything reusable)
# =========================
out_dir = f'legacy_run/{ARM}_{MILESTONE}k'
os.makedirs(out_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))
with open(os.path.join(out_dir, 'feature_columns.json'), 'w') as f:
    json.dump(feature_cols, f)
with open(os.path.join(out_dir, 'train_meta.json'), 'w') as f:
    json.dump({
        'milestone': MILESTONE, 'arm': ARM, 'cutoff': CUTOFF, 'seed': SEED,
        'n_features': len(feature_cols), 'n_train_rows': int(len(X_train_full)),
        'n_after_isoforest': int(len(X_clean)), 'pos_count': int(pos_count),
        'neg_count': int(neg_count), 'final_train_loss': float(best_loss),
    }, f, indent=2)
print(f"Saved arm={ARM} milestone={MILESTONE}k -> {out_dir}/ "
      f"({len(feature_cols)} features, {len(X_clean)} training rows)")
