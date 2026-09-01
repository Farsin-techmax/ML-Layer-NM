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

from date_utils import parse_dates

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Model, feature derivation, has_x flags, leak/ID column lists and the HAS_X_EXCLUDE / DROP_RFM
# policies all live in pms_model.py, shared with score_milestone.py and eval_test_metrics.py.
from pms_model import (BinaryClassifier, DERIVED_FEATURES, DROP_RFM, add_milestone_flags,
                       build_inference_matrix, derive_inference_features, family_of, features_path,
                       id_cols, model_dir, resolve_path, select_pms_delay_variant, split_missing)

# Ablation switches. Defaults reproduce production behaviour exactly -- set them in the shell to
# A/B a configuration without editing code:
#   $env:PMS_USE_HAS_X=0                     drop the has_x milestone flags
#   $env:PMS_DROP="PMS_Delay,Service Frequency"   drop named columns before training
#   $env:PMS_RUN_TAG=_nohasx                 write to models/{m}k_nohasx/ instead of models/{m}k/
#
# NOTE on PMS_USE_HAS_X: it only stops add_milestone_flags() from RE-creating the flags.
# process_service_data() already emits has_10..has_{m-10} into the feature matrix, so the columns
# survive regardless. To actually remove them, name them in PMS_DROP.
USE_HAS_X = os.environ.get('PMS_USE_HAS_X', '1') != '0'
EXTRA_DROP = [c.strip() for c in os.environ.get('PMS_DROP', '').split(',') if c.strip()]
RUN_TAG = os.environ.get('PMS_RUN_TAG', '')

# Without this, torch weight init and DataLoader shuffling are unseeded and two runs on an
# IDENTICAL feature matrix differed by 6.7 accuracy points (97.06 vs 90.33 on 20k, 2026-08-12).
# Any A/B smaller than that is unreadable without it.
SEED = int(os.environ.get('PMS_SEED', '42'))
np.random.seed(SEED)
torch.manual_seed(SEED)


# --- Config ---
MILESTONE = int(sys.argv[1])

# ONE training matrix per milestone, produced by the same process_service_data() that builds the
# test sets (python pmstrainfeatureeng_refactored.py 2026 Q1 <milestone> --train-cutoff 2025-12-31),
# so train and test cannot disagree on column names or encodings.
#
# This replaces concatenating traindataq1_q2/finalmerged{m}kQ1.csv + ...Q2_2026v1.csv. Those were
# two pulls of the same history (2016-2025) that overlapped on 89-96% of their VINs with identical
# labels: most vehicles were counted twice, and because the 80/20 split cut by row position, the
# "validation" slice was largely Q2 rows whose VINs already sat in the training slice.
#
# $env:PMS_LABEL_MODE=window picks up the matrix built with --label-mode window instead, where
# positives and negatives both have to be DUE in the same window. The test sets are unchanged --
# create_test_cohort() already builds a due-in-window cohort, so only training had the mismatch.
#
# $env:PMS_LABEL_MODE=history picks up the matrix built with --label-mode history: labels taken from
# raw service history rather than EDA's `Last Service - PMS`, one cohort per quarter, each with its
# features cut at its own band start. That mode REQUIRES the matching test sets --
# $env:PMS_GRACE_DAYS=45 selects test_{m}k_Q{1,2}_2026_h45.csv, built by
# `prepare_test_set.py --grace-days 45`. Training on a grace-band label and scoring against an
# 'ever' label would compare two different questions.
# $env:PMS_LABEL_MODE=candidates picks up the matrix built with --label-mode candidates: due date =
# whichever comes first of the schedule or a service-1->service-10 burn-rate projection, invoice-
# date guard, 2016Q1-2025Q4 by default, early completers counted positive (flagged EarlyCompleter)
# instead of excluded. Matching test sets are the _cand files from
# `prepare_test_set.py --label-mode candidates --grace-days 15`.
LABEL_MODE = os.environ.get('PMS_LABEL_MODE', 'ever')
if LABEL_MODE not in ('ever', 'window', 'all', 'history', 'candidates'):
    sys.exit(f"PMS_LABEL_MODE must be 'ever', 'window', 'all', 'history' or 'candidates', "
             f"got {LABEL_MODE!r}")
LABEL_SUFFIX = '' if LABEL_MODE == 'ever' else f'_{LABEL_MODE}'
# $env:PMS_DATE_MODEL=calibrated mirrors --date-model on both builders. It changes WHICH vehicles
# are due in the window, so it selects a different training matrix AND a different pair of test
# sets -- they must move together or the model is scored on a population it was not trained for.
DATE_MODEL = os.environ.get('PMS_DATE_MODEL', 'schedule')
if DATE_MODEL not in ('schedule', 'calibrated'):
    sys.exit(f"PMS_DATE_MODEL must be 'schedule' or 'calibrated', got {DATE_MODEL!r}")
DATE_SUFFIX = '' if DATE_MODEL == 'schedule' else '_cal'
LABEL_SUFFIX += DATE_SUFFIX
# Matches --label-tag on the feature-eng side, so two window builds with different --label-start
# (e.g. a wide 2018 window vs a narrow 2025 one) can be trained without overwriting each other.
if os.environ.get('PMS_LABEL_TAG', ''):
    LABEL_SUFFIX += f"_{os.environ['PMS_LABEL_TAG']}"
TRAIN = resolve_path(f"refactored_test_dir/final_processed_{MILESTONE}k{LABEL_SUFFIX}.csv")

# Test sets, built by prepare_test_set.py -- the same process_service_data() as TRAIN above, so the
# schemas agree by construction:
#   python prepare_test_set.py --milestone <m> --window-start 2026-01-01 --window-end 2026-03-31 \
#          --train-cutoff 2025-12-31
#
# These must carry the FULL process_service_data output. The training columns are intersected with
# every test file that exists (see _test_feature_cols below), so a narrow test file silently shrinks
# the model: an earlier 35-column test file collapsed 50k from ~120 features to 32 and cost ~24
# accuracy points on Q1.
#
# The old tests_alan/testdataq1_q2/ files are NOT used any more: they come from the legacy code
# path, which one-hot encodes RFM_segments / LastNPMS_Event / Number of Cylinders while the current
# path does not, so intersecting against them throws away good features for no reason.
#
# $env:PMS_GRACE_DAYS=45 switches to the grace-band cohorts (_h45), which is what --label-mode
# history trains against. Guarded below: history mode without it is a silent train/test rule
# mismatch, and that is exactly the class of bug this pipeline keeps producing.
GRACE_DAYS = os.environ.get('PMS_GRACE_DAYS', '')
if LABEL_MODE == 'candidates':
    # A different due-date model AND a different label rule than 'history' -- its own suffix, not
    # the '_h{N}' grace-band files (those match --label-mode history, a different training run).
    TEST_SUFFIX = DATE_SUFFIX + '_cand'
else:
    TEST_SUFFIX = DATE_SUFFIX + (f'_h{GRACE_DAYS}' if GRACE_DAYS else '')
if LABEL_MODE == 'history' and not GRACE_DAYS:
    sys.exit("PMS_LABEL_MODE=history needs PMS_GRACE_DAYS set to the same value the matrix was "
             "built with (e.g. $env:PMS_GRACE_DAYS=45), so the test sets use the same label rule.")
TEST_Q1 = resolve_path(f"test_sets/test_{MILESTONE}k_Q1_2026{TEST_SUFFIX}.csv")
TEST_Q2 = resolve_path(f"test_sets/test_{MILESTONE}k_Q2_2026{TEST_SUFFIX}.csv")

# $env:PMS_TRAIN_FROM=2021 (candidates mode ablation): keep only rows whose CohortQuarter year is
# >= this value. A 2016 customer may simply not resemble a 2026 one; this is the cheap way to find
# out, since CohortQuarter is already on every row -- no feature rebuild needed, just a filter
# before the chronological split below. Default '' keeps all years (the full 2016-2025 span).
TRAIN_FROM = os.environ.get('PMS_TRAIN_FROM', '')
# $env:PMS_TRAIN_TO=2024: mirror of PMS_TRAIN_FROM, keeps cohorts whose year is <= this value.
# Exists for window selection: train the full/2021+/2023+ arms only through 2024 so all three can
# be compared on an identical held-back 2025 cohort slice, instead of on the two 2026 test quarters
# (which must never drive a tuning decision -- there are only two of them).
TRAIN_TO = os.environ.get('PMS_TRAIN_TO', '')

# models/{m}k/ for the weights/scaler/imputer, models/selected_features_{m}k.json for the feature
# list -- one file per model, at the models/ root. Was models/models_alan/{m}k/ until 2026-08-12.
MODEL_DIR = model_dir(MILESTONE) + RUN_TAG
FEATURES_PATH = features_path(MILESTONE).replace('.json', f'{RUN_TAG}.json')
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 10
LR = 3e-4
WEIGHT_DECAY = 1e-2

# === Step 1: Load training data ===
if not os.path.exists(TRAIN):
    sys.exit(f"Missing {TRAIN} -- build it first with:\n"
             f"  python pmstrainfeatureeng_refactored.py 2026 Q1 {MILESTONE} --train-cutoff 2025-12-31")
logger.info(f"Loading training data: {TRAIN}")
train_df = pd.read_csv(TRAIN, low_memory=False)
logger.info(f"Train shape: {train_df.shape}, Target: {train_df['TargetFlag'].value_counts().to_dict()}")

if TRAIN_FROM or TRAIN_TO:
    if 'CohortQuarter' not in train_df.columns:
        sys.exit(f"PMS_TRAIN_FROM={TRAIN_FROM!r}/PMS_TRAIN_TO={TRAIN_TO!r} need a CohortQuarter "
                 f"column -- only --label-mode history/candidates matrices have one")
    _yr = train_df['CohortQuarter'].astype(str).str[:4].astype(int)
    _before = len(train_df)
    if TRAIN_FROM:
        train_df = train_df[_yr >= int(TRAIN_FROM)]
        _yr = _yr[_yr >= int(TRAIN_FROM)]
    if TRAIN_TO:
        train_df = train_df[_yr <= int(TRAIN_TO)]
    train_df = train_df.reset_index(drop=True)
    logger.info(f"PMS_TRAIN_FROM={TRAIN_FROM or '-'} PMS_TRAIN_TO={TRAIN_TO or '-'}: "
                f"{_before} -> {len(train_df)} rows")

# Temporal cutoff: train only on data up to Q4 2025 (test sets are Q1/Q2 2026), then sort by due
# date so the 80/20 split below really is chronological -- it holds out the most recent vehicles
# rather than whatever happened to sit last in the file. Rows with an unparseable due date fail the
# comparison and are dropped, as before.
# parse_dates, not a bare dayfirst=True parse: the matrix stores this column as ISO with a time
# component, and on pandas 3.0.3 dayfirst=True corrupts exactly those values (2026-08-09 00:00:00
# -> 2026-09-08). Measured on the 20k candidates matrix 2026-09-01: 50.6% of values parsed to a
# different date, scrambling the chronological sort (0 rows flipped across the cutoff).
_due = parse_dates(train_df[f'Next{MILESTONE}K_Due'])
_keep = _due <= pd.Timestamp('2025-12-31')
train_df = train_df[_keep].copy()
train_df['_due_sort'] = _due[_keep]
train_df = (train_df.sort_values('_due_sort', kind='mergesort')
                    .drop(columns='_due_sort')
                    .reset_index(drop=True))
logger.info(f"After <=Q4-2025 cutoff + chronological sort: {train_df.shape}, "
            f"Target: {train_df['TargetFlag'].value_counts().to_dict()}")

# === Step 2: Derive missing features ===
# Same function the test/scoring paths use, so train and inference cannot drift apart.
logger.info("Deriving missing features...")
train_df = derive_inference_features(train_df)
logger.info(f"Derived features added: {DERIVED_FEATURES}")

# === Step 3: Prepare features ===
train_df = add_milestone_flags(train_df, MILESTONE, use_has_x=USE_HAS_X)  # before Service_Num goes
# Pick ONE of the four PMS_Delay* candidates ($env:PMS_DELAY_VARIANT). The survivor is always named
# PMS_Delay, so selected_features_{m}k.json is identical across variants and an A/B differs only in
# the values behind that name. Must also run in _test_feature_cols(), or the reconcile below sees
# the extra candidate columns in the test set and keeps them as real features.
train_df = select_pms_delay_variant(train_df)
DROP_COLS = id_cols(MILESTONE) + EXTRA_DROP   # IDs + LEAK_COLS from pms_model, + any ablation drops
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
    d = select_pms_delay_variant(d)
    d = d.drop(columns=[c for c in DROP_COLS if c in d.columns], errors='ignore')
    d = d.drop(columns=['TargetFlag'], errors='ignore')
    return set(d.columns) | set(DERIVED_FEATURES)


# Reconcile the training columns with the test sets.
#
# A train-only column is one of two very different things, and the old blanket intersection treated
# both as the second:
#   1. a ONE-HOT DUMMY whose category simply did not occur in the (much smaller) test cohort.
#      pd.get_dummies only emits columns for categories present in the frame, so a 951-row test set
#      cannot produce a dummy for a rare part. "Category absent" means the value is 0 -- the
#      CORRECT value -- so the feature is perfectly usable and should be kept.
#   2. a feature the test pipeline never computes at all. A real schema gap; still dropped, because
#      zero-filling it would feed the model a value that means nothing.
#
# Measured on 20k (2026-08-12): ALL 41 previously-dropped columns were case 1 --
# 15 DeferredPart__, 10 InvoicedPart__, 10 LostPart__, 2 PMS Status_, plus 4 dealer/appointment
# columns. Not one continuous feature. The guard was discarding usable behavioural features to
# defend against a problem that did not exist here.
#
# Telling them apart by FAMILY, via family_of() in pms_model (shared with the reporting side so the
# two cannot disagree): if the test set already has a sibling from the same family, the pipeline
# clearly produces that family and the missing member is just a category that did not occur.
# Do NOT test for binary 0/1 values -- several DeferredPart__/InvoicedPart__/LostPart__ columns are
# per-category COUNTS, and a count of 0 for a category that never occurred is equally correct.
_common = set(X.columns)
for _p in [TEST_Q1, TEST_Q2]:
    if os.path.exists(_p):
        _common &= _test_feature_cols(_p)

# PMS Status_* describes the LAST PMS visit. For a training positive that visit IS the target
# milestone, and in the test set the field comes from EDA's post-outcome snapshot -- so it restates
# the label. Its family is present in the test set, but it is leak, not signal: keep it out.
LEAKY_FAMILY_PREFIXES = ('PMS Status_',)

_present_families = {family_of(c) for c in _common}
_train_only = [c for c in X.columns if c not in _common]
_recovered, _gap = [], []
for _c in _train_only:
    if family_of(_c) in _present_families and not _c.startswith(LEAKY_FAMILY_PREFIXES):
        _recovered.append(_c)
    else:
        _gap.append(_c)

_n_before = X.shape[1]
X = X[[c for c in X.columns if c in _common or c in _recovered]]
# The prefix check above only sees TRAIN-ONLY columns, so a leaky-family column present in both
# train and test sailed through it (this is how 'PMS Status_PMS only' reached the promoted 20k
# model). Apply the same rule to the final set, wherever the column came from.
_leaky = [c for c in X.columns if c.startswith(LEAKY_FAMILY_PREFIXES)]
if _leaky:
    X = X.drop(columns=_leaky)
    logger.info(f"Dropped {len(_leaky)} leaky-family columns present in train AND test: {_leaky}")
logger.info(f"Test-schema reconcile: kept {X.shape[1]} / {_n_before} features")
if _recovered:
    _fams = sorted({_c.split('__')[0] if '__' in _c else _c.rsplit('_', 1)[0]
                    for _c in _recovered})
    logger.info(f"  recovered {len(_recovered)} absent-category dummies (0 is correct for these); "
                f"families: {_fams}")
if _gap:
    logger.info(f"  dropped {len(_gap)} genuine train-only cols: {_gap[:15]}"
                f"{' ...' if len(_gap) > 15 else ''}")

# Constant features carry no information the model can learn from, and when the constant is an
# accident they are actively harmful: `Last Service Mileage` was constant 0 across all 41,983 rows
# of the 20k candidates matrix, because derive_inference_features() had no branch for
# 'LastNonPMSMileage' present without 'Last PMS Mileage' and fell through to a hardcoded 0. The
# test sets still carried the source column, so the same feature VARIED at inference -- the model
# had a weight trained on a constant and was then fed real values through it. Single-feature AUC
# read 0.5000 on train and 0.158 on test. Nothing warned; it took a column-by-column audit to find.
#
# The derive_inference_features() branch is fixed, but the class of bug is not specific to it --
# any upstream change that removes a source column can silently flatten a derived one. This check
# is the general net. It reports rather than drops: a legitimately rare dummy can be constant in
# training and still be worth keeping, so the call is left to a human reading the log.
_const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
if _const:
    logger.warning(f"{len(_const)} CONSTANT feature(s) in training -- the model cannot learn from "
                   f"these, and if they vary at inference they inject pure noise: {_const}")
    logger.warning("  If a constant is unintended, drop it: "
                   f"$env:PMS_DROP=\"{','.join(_const)}\"")

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
# Genuinely chronological now: train_df was sorted by Next{m}K_Due in Step 1, and the matrix holds
# one row per VIN, so the validation slice is the most recent 20% of vehicles and shares no VIN with
# the training slice.
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
scaler.fit(X_train_imp)
# RobustScaler's own zero-IQR guard only catches an EXACT 0.0 scale; it does not catch a
# floating-point-noise scale like 5.68e-14, which is just as fatal -- dividing a real value (e.g. a
# revenue column with a legitimate few-thousand max, mostly-zero otherwise) by that produces a
# ~1e17-scale feature that swamps every other input and stops the network learning entirely (flat
# loss from epoch 1). Measured on the 20k candidates matrix (--label-mode candidates): 'Tyre
# Revenue' has Q1=Q3=0 over the full 41,983 rows, but the chronological 80% TRAIN split alone lands
# its Q3 on residual FP noise from an earlier near-zero subtraction (5.684341886080803e-14) instead
# of a clean 0 -- so this was invisible on the full matrix and only appeared after the split. Any
# scale this small is noise, never a real unit; clamp it the same way sklearn already clamps an
# exact zero.
_tiny_scale = scaler.scale_ < 1e-6
if _tiny_scale.any():
    logger.warning(f"Clamping {int(_tiny_scale.sum())} near-zero RobustScaler scale_ value(s) to "
                   f"1.0 (floating-point noise, not a real unit): "
                   f"{[c for c, t in zip(X_train.columns, _tiny_scale) if t]}")
    scaler.scale_ = np.where(_tiny_scale, 1.0, scaler.scale_)
X_train_scaled = scaler.transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

with open(FEATURES_PATH, 'w') as f:
    json.dump(selected_features, f)
logger.info(f"Saved {len(selected_features)} selected features -> {FEATURES_PATH}")

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
    if not os.path.exists(test_path):
        # Skip rather than crash: the model and its artifacts are already saved by this point, and
        # not every window has a built test set yet.
        logger.warning(f"No test file at {test_path} -- skipping {label}")
        return
    test_df = pd.read_csv(test_path)
    y_test = test_df['TargetFlag'].values

    X_test, missing = build_inference_matrix(test_df, MILESTONE, selected_features,
                                             use_has_x=USE_HAS_X)
    if missing:
        # Absent one-hot categories are harmless (0 is the right value); genuine gaps are the
        # dominant cause of bad metrics. Report them separately so the second is not buried.
        _exp, _gaps = split_missing(missing, test_df.columns)
        if _exp:
            logger.info(f"{len(_exp)}/{len(selected_features)} absent one-hot categories "
                        f"zero-filled in {os.path.basename(test_path)} (expected, 0 is correct)")
        if _gaps:
            logger.warning(f"{len(_gaps)}/{len(selected_features)} features GENUINELY missing from "
                           f"{test_path} and zero-filled: {_gaps[:15]}"
                           f"{' ...' if len(_gaps) > 15 else ''}")
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
