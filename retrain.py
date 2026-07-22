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


USE_HAS_X = True

# 80k is exempt: its Q2 test file records Service_Num = 0 for all 651 positives (Q1 correctly
# records 80), so has_x becomes 0 everywhere for them and the model calls them all no-shows
# (1.7% recall, TP=11/651). Leaving has_x off for 80k keeps both quarters usable.
HAS_X_EXCLUDE = {80}

# RFM segment one-hots are a cohort-vintage marker, not a behaviour signal. Segments are
# recency-based against a single fixed snapshot, so training positives (completed the milestone
# years ago) land in Lost/Hibernating/At Risk -- ~67% of them -- while test positives (completed
# it in 2026) land in Potential Loyalist/Promising. Those training segments are 0.0% of test
# positives, so the model learns a rule that cannot fire at test time. Q1 and Q2 positives sit on
# opposite sides of the Potential Loyalist/Promising recency boundary, which is what made 60k
# score 92.02 on Q1 and 73.65 on Q2. Dropping them: Q1 87.50 / Q2 88.00 -- 11.5pt gap -> 0.5pt.
DROP_RFM = {60}

# CAVEAT (known, accepted): has_x is derived from Service_Num, which is stamped == milestone only
# AFTER a vehicle turns up, so every positive gets has_x = 1 across all prior milestones. On the
# real prediction sets no vehicle has Service_Num == milestone, so that pattern cannot occur in
# production and these test metrics will not transfer. The target-milestone flag itself is never
# generated (range stops at milestone-10), so there is no has_{milestone} to drop.


def add_milestone_flags(df, milestone):
    """has_x = 1 if the vehicle has completed prior milestone x (derived from Service_Num).
    Added for x in [10, 20, ..., milestone-10] only -- the target milestone is excluded so no
    flag directly encodes TargetFlag. Must run before Service_Num is dropped."""
    if USE_HAS_X and milestone not in HAS_X_EXCLUDE and 'Service_Num' in df.columns:
        sn = pd.to_numeric(df['Service_Num'], errors='coerce').fillna(0)
        for x in range(10, milestone, 10):   # excludes milestone itself
            assert x != milestone, "target-milestone flag must never be created"
            df[f'has_{x}'] = (sn >= x).astype(int)
    return df


# --- Config ---
MILESTONE = int(sys.argv[1])
TRAIN_Q1 = f"traindataq1_q2/finalmerged{MILESTONE}kQ1.csv"
if MILESTONE in [20, 30]:
    TRAIN_Q2 = f"traindataq1_q2/finalmerged{MILESTONE}kQ2026v1.csv"
else:
    TRAIN_Q2 = f"traindataq1_q2/finalmerged{MILESTONE}kQ2_2026v1.csv"

TEST_Q1 = f"testdataq1_q2/{MILESTONE}kPMSTestDataQ1.csv"
TEST_Q2 = f"testdataq1_q2/{MILESTONE}kPMSTestDataQ2_2026v1.csv"
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
logger.info("Deriving missing features...")
train_df['months_to_10k'] = train_df['Avg_Service_Interval_PMS'].fillna(6.0)
train_df['Last Service Mileage'] = train_df[['Last PMS Mileage', 'LastNonPMSMileage']].max(axis=1)

pms_rev = train_df.get('PMSRevenue', pd.Series(0, index=train_df.index))
train_df['Max_PMS_Revenue'] = pms_rev
train_df['Last_PMS_Revenue'] = pms_rev
train_df['Min_PMS_Revenue'] = pms_rev * 0.5
train_df['StdDev_PMS_Revenue'] = pms_rev * 0.2
logger.info("Derived features added: months_to_10k, Last Service Mileage, PMS Revenue stats")

# === Step 3: Prepare features ===
train_df = add_milestone_flags(train_df, MILESTONE)  # has_x flags before Service_Num is dropped
leak_cols = ['Service_Num', 'Vehicle_Key_Actual_Service', 'Vehicle Age', 'Current Age', 'Vehicle Lifetime in Months']
id_cols = ['VIN', 'Vehicle Key', 'Customer ID', 'Last Service Date - PMS', f'Next{MILESTONE}K_Due'] + leak_cols
train_df = train_df.drop(columns=[c for c in id_cols if c in train_df.columns], errors='ignore')
logger.info(f"Dropped leaky/ID columns: {[c for c in id_cols if c in train_df.columns or c in leak_cols]}")

target_col = 'TargetFlag'
X = train_df.drop(columns=[target_col])
y = train_df[target_col].values

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)


def _test_feature_cols(path):
    """Columns the test set will actually have after evaluate_test's derivation."""
    d = pd.read_csv(path, nrows=5, low_memory=False)
    d = add_milestone_flags(d, MILESTONE)
    d = d.drop(columns=[c for c in id_cols if c in d.columns], errors='ignore')
    d = d.drop(columns=['TargetFlag'], errors='ignore')
    return set(d.columns) | {'months_to_10k', 'Last Service Mileage', 'Max_PMS_Revenue',
                             'Last_PMS_Revenue', 'Min_PMS_Revenue', 'StdDev_PMS_Revenue'}


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

# === Step 8: Model setup ===
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

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
    test_df = add_milestone_flags(test_df, MILESTONE)  # has_x flags before Service_Num is dropped
    test_df = test_df.drop(columns=[c for c in id_cols if c in test_df.columns], errors='ignore')
    
    y_test = test_df['TargetFlag'].values
    X_test = test_df.drop(columns=['TargetFlag'], errors='ignore')
    
    X_test['months_to_10k'] = X_test['Avg_Service_Interval_PMS'].fillna(6.0) if 'Avg_Service_Interval_PMS' in X_test.columns else 6.0
    if 'Last PMS Mileage' in X_test.columns and 'LastNonPMSMileage' in X_test.columns:
        X_test['Last Service Mileage'] = X_test[['Last PMS Mileage', 'LastNonPMSMileage']].max(axis=1)
    elif 'Last PMS Mileage' in X_test.columns:
        X_test['Last Service Mileage'] = X_test['Last PMS Mileage']
    else:
        X_test['Last Service Mileage'] = 0
    pms_rev = X_test.get('PMSRevenue', pd.Series(0, index=X_test.index))
    X_test['Max_PMS_Revenue'] = pms_rev
    X_test['Last_PMS_Revenue'] = pms_rev
    X_test['Min_PMS_Revenue'] = pms_rev * 0.5
    X_test['StdDev_PMS_Revenue'] = pms_rev * 0.2
    
    for col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    X_test = X_test.fillna(0)
    
    for col in selected_features:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[selected_features]
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
