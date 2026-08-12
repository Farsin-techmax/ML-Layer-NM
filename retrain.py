import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss, confusion_matrix, accuracy_score
import joblib
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Model, feature derivation, has_x flags, leak/ID column lists and the HAS_X_EXCLUDE / DROP_RFM
# policies all live in pms_model.py, shared with score_milestone.py and eval_test_metrics.py.
from pms_model import (BinaryClassifier, DERIVED_FEATURES, DROP_RFM, add_milestone_flags,
                       build_inference_matrix, derive_inference_features, id_cols, resolve_path)

USE_HAS_X = True


# --- Config ---
MILESTONE = int(sys.argv[1])
TRAIN_Q1 = resolve_path(f"traindataq1_q2/finalmerged{MILESTONE}kQ1.csv")
if MILESTONE in [20, 30]:
    TRAIN_Q2 = resolve_path(f"traindataq1_q2/finalmerged{MILESTONE}kQ2026v1.csv")
else:
    TRAIN_Q2 = resolve_path(f"traindataq1_q2/finalmerged{MILESTONE}kQ2_2026v1.csv")

# Both test files must carry the FULL process_service_data output (~170-240 cols). The training
# columns are intersected with every test file that exists (see _test_feature_cols below), so a
# narrow test file silently shrinks the model: pointing TEST_Q1 at prepare_test_set.py's 35-column
# output collapsed 50k from ~120 features to 32 and cost ~24 accuracy points on Q1.
TEST_Q1 = resolve_path(f"testdataq1_q2/{MILESTONE}kPMSTestDataQ1.csv")
TEST_Q2 = resolve_path(f"testdataq1_q2/{MILESTONE}kPMSTestDataQ2_2026v1.csv")
MODEL_DIR = f"models/models_alan/{MILESTONE}k"
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 10
LR = 3e-4
WEIGHT_DECAY = 1e-2

# === Step 1: Load and combine Q1+Q2 training data ===
logger.info("Loading new training data...")
df_q1 = pd.read_csv(TRAIN_Q1)
df_q2 = pd.read_csv(TRAIN_Q2)
logger.info(f"Q1 shape: {df_q1.shape}, Q2 shape: {df_q2.shape}")
logger.info(f"Q1 Target: {df_q1['TargetFlag'].value_counts().to_dict()}")
logger.info(f"Q2 Target: {df_q2['TargetFlag'].value_counts().to_dict()}")

# Combine Q1 and Q2
train_df = pd.concat([df_q1, df_q2], axis=0, ignore_index=True)
logger.info(f"Combined train shape: {train_df.shape}")
logger.info(f"Combined Target: {train_df['TargetFlag'].value_counts().to_dict()}")

# Temporal cutoff: train only on data up to Q4 2025 (test sets are Q1/Q2 2026).
_due = pd.to_datetime(train_df[f'Next{MILESTONE}K_Due'], errors='coerce', dayfirst=True, format='mixed')
train_df = train_df[_due <= pd.Timestamp('2025-12-31')].reset_index(drop=True)
logger.info(f"After <=Q4-2025 cutoff: {train_df.shape}, Target: {train_df['TargetFlag'].value_counts().to_dict()}")

# === Step 2: Derive missing features ===
# Same function the test/scoring paths use, so train and inference cannot drift apart.
logger.info("Deriving missing features...")
train_df = derive_inference_features(train_df)
logger.info(f"Derived features added: {DERIVED_FEATURES}")

# === Step 3: Prepare features ===
train_df = add_milestone_flags(train_df, MILESTONE, use_has_x=USE_HAS_X)  # before Service_Num goes
DROP_COLS = id_cols(MILESTONE)   # IDs + LEAK_COLS, from pms_model
_dropped = [c for c in DROP_COLS if c in train_df.columns]
train_df = train_df.drop(columns=_dropped, errors='ignore')
logger.info(f"Dropped leaky/ID columns: {_dropped}")

target_col = 'TargetFlag'
X = train_df.drop(columns=[target_col])
y = train_df[target_col].values

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)


def _test_feature_cols(path):
    """Columns the test set will actually have after evaluate_test's derivation."""
    d = pd.read_csv(path, nrows=5, low_memory=False)
    d = add_milestone_flags(d, MILESTONE, use_has_x=USE_HAS_X)
    d = d.drop(columns=[c for c in DROP_COLS if c in d.columns], errors='ignore')
    d = d.drop(columns=['TargetFlag'], errors='ignore')
    return set(d.columns) | set(DERIVED_FEATURES)


# Keep only features that also exist in the test sets, so nothing is zero-filled at inference.
_common = set(X.columns)
for _p in [TEST_Q1, TEST_Q2]:
    if os.path.exists(_p):
        _common &= _test_feature_cols(_p)
_n_before = X.shape[1]
X = X[[c for c in X.columns if c in _common]]
logger.info(f"Test-schema intersection: kept {X.shape[1]} / {_n_before} features "
            f"(dropped {_n_before - X.shape[1]} train-only cols)")

# Gower cluster IDs are not stable across feature-eng runs (cluster "4" in the training run is a
# different group than cluster "4" in the test run), so these one-hots switch on weights that were
# never trained and destroy probability separation at inference. Drop them.
_clust = [c for c in X.columns if '_Cluster_' in c]
if _clust:
    X = X.drop(columns=_clust)
    logger.info(f"Dropped {len(_clust)} unstable cluster one-hots -> {X.shape[1]} features")

# RFM segments encode cohort vintage rather than behaviour (see DROP_RFM note at top).
_rfm = [c for c in X.columns if c.startswith('RFM_segments_')] if MILESTONE in DROP_RFM else []
if _rfm:
    X = X.drop(columns=_rfm)
    logger.info(f"Dropped {len(_rfm)} RFM segment one-hots -> {X.shape[1]} features")

logger.info(f"Feature matrix shape: {X.shape}")
logger.info(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")

# === Step 4: IsolationForest outlier removal ===
logger.info("Running IsolationForest outlier removal...")
iso = IsolationForest(contamination=0.005, random_state=42)
outliers = iso.fit_predict(X)
mask = outliers == 1
X_clean = X[mask].reset_index(drop=True)
y_clean = y[mask]
logger.info(f"After outlier removal: {X_clean.shape[0]} rows (removed {(~mask).sum()})")

# === Step 5: Chronological train/val split (80/20) ===
split_idx = int(len(X_clean) * 0.8)
X_train = X_clean.iloc[:split_idx]
y_train = y_clean[:split_idx]
X_val = X_clean.iloc[split_idx:]
y_val = y_clean[split_idx:]
logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
selected_features = list(X_train.columns)

# === Step 6: Impute + Scale ===
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

with open(os.path.join(MODEL_DIR, "selected_features.json"), 'w') as f:
    json.dump(selected_features, f)
logger.info(f"Saved {len(selected_features)} selected features")

# === Step 7: PyTorch datasets ===
train_ds = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
val_ds = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# === Step 8: Model setup ===  (BinaryClassifier comes from pms_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BinaryClassifier(input_dim=X_train_scaled.shape[1]).to(device)

num_pos = (y_train == 1).sum()
num_neg = (y_train == 0).sum()
pos_weight_value = num_neg / (num_pos + 1e-6)
pos_weight = torch.tensor([pos_weight_value]).to(device)
logger.info(f"Positive: {num_pos}, Negative: {num_neg}, pos_weight: {pos_weight_value:.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

def smooth_labels(y, smoothing=0.1):
    return y * (1 - smoothing) + 0.5 * smoothing

def brier_loss(logits, targets):
    probs = torch.sigmoid(logits)
    return torch.mean((probs - targets) ** 2)

best_loss = np.inf
best_model_state = None
counter = 0

logger.info("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_X).squeeze()
        y_smoothed = smooth_labels(batch_y)
        loss_bce = criterion(logits, y_smoothed)
        loss_brier = brier_loss(logits, batch_y)
        loss = loss_bce + 0.5 * loss_brier
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_X).squeeze()
            y_smoothed = smooth_labels(batch_y)
            loss_bce = criterion(logits, y_smoothed)
            loss_brier = brier_loss(logits, batch_y)
            loss = loss_bce + 0.5 * loss_brier
            val_loss += loss.item()
            
    epoch_loss /= len(train_loader)
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    
    if (epoch + 1) % 5 == 0:
        logger.info(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_state = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

model.load_state_dict(best_model_state)
torch.save(best_model_state, os.path.join(MODEL_DIR, "best_model.pt"))
logger.info("Best model saved.")

def evaluate_test(test_path, label):
    logger.info(f"\n--- Evaluating on {label} ---")
    test_df = pd.read_csv(test_path)
    y_test = test_df['TargetFlag'].values

    X_test, missing = build_inference_matrix(test_df, MILESTONE, selected_features,
                                             use_has_x=USE_HAS_X)
    if missing:
        # Zero-filled columns are the dominant cause of bad metrics -- surface, don't hide.
        logger.warning(f"{len(missing)}/{len(selected_features)} features absent from {test_path} "
                       f"and zero-filled: {missing[:15]}{' ...' if len(missing) > 15 else ''}")
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    model.eval()
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    with torch.no_grad():
        logits = model(X_test_t).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    
    try:
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        brier_s = brier_score_loss(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        
        logger.info(f"Accuracy:  {acc:.4f}")
        logger.info(f"ROC-AUC:   {auc:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info(f"Precision (Returner): {prec:.4f}")
        logger.info(f"Recall (Returner):    {rec:.4f}")
        logger.info(f"Brier:     {brier_s:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        metrics_dict = {'Accuracy': float(acc), 'ROC-AUC': float(auc), 'F1-Score': float(f1), 'Precision': float(prec), 'Recall': float(rec), 'Brier': float(brier_s)}
        with open(os.path.join(MODEL_DIR, f'metrics_{label.replace(" ", "_")}.json'), 'w') as mf:
            json.dump(metrics_dict, mf)
    except Exception as e:
        logger.error(f"Error: {e}")

evaluate_test(TEST_Q1, f"{MILESTONE}k Q1 Test")
evaluate_test(TEST_Q2, f"{MILESTONE}k Q2 Test")
logger.info("\nDone! Model and artifacts saved to " + MODEL_DIR)
