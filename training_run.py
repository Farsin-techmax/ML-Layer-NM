train_df = df[df['Next70K_Due'] < '2026-07-01']

# test_df = df[(df['Next80K_Due'] >= '2025-07-01') & (df['Next80K_Due'] <= '2025-09-30')]
 
X_train = train_df.iloc[:,:-1]

# X_train= linimput(X_train)
 
y_train = train_df['TargetFlag']

# X_test = test_df.iloc[:,:-1]

# X_test = linimput(X_test)

# y_test = test_df['TargetFlag']

a = X_train.columns
 
from sklearn.ensemble import IsolationForest
 
iso = IsolationForest(contamination=0.005, random_state=42)

outliers = iso.fit_predict(X_train.iloc[:,5:])
 
# Filter out outliers

X_clean = X_train[outliers == 1]

y_clean = y_train[outliers == 1]

# turnuprat.append(y_clean.value_counts()[1]/len(y_clean))

scaler = RobustScaler()

X_train = scaler.fit_transform(X_clean.iloc[:,5:])

# X_test = scaler.transform(X_test.iloc[:,5:])
 
# fnltrain = train_df[outliers == 1]

# is_sc = torch.tensor(fnltrain['SC Status'].values, dtype=torch.float32)         # 1 if SC active, else 0

# is_lowfreq = torch.tensor(fnltrain['LowMileageFreqUsers'].values, dtype=torch.float32)    # 1 if LowMileageFreq, else 0
 
# # Tiered weights

# w_tensor  = torch.ones(len(is_sc))  # default = 1
 
# # Tier 1: SC + LowFreq -> highest weight

# w_tensor[(is_sc == 1) & (is_lowfreq == 1)] = 5.0  
 
# # Tier 2: SC only

# w_tensor[(is_sc == 1) & (is_lowfreq == 0)] = 3.0  
 
# # Tier 3: LowFreq only

# w_tensor[(is_sc == 0) & (is_lowfreq == 1)] = 2.0
 
 
# Convert to tensors

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

y_train_tensor = torch.tensor(y_clean.to_numpy().reshape(-1, 1), dtype=torch.float32)

# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# y_test_tensor = torch.tensor(y_test.to_numpy().reshape(-1, 1), dtype=torch.float32)
 
# GPU support

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')
 
X_train_tensor = X_train_tensor.to(device)

y_train_tensor = y_train_tensor.to(device)

# X_test_tensor = X_test_tensor.to(device)

# y_test_tensor = y_test_tensor.to(device)
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# =========================

# MODEL (Reduced + BatchNorm)

# =========================

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
 
 
# =========================

# CUSTOM WEIGHTED BCE (OPTIONAL)

# =========================

def custom_bce_with_weights(outputs, targets, sample_weights, pos_weight):

    bce = torch.nn.functional.binary_cross_entropy_with_logits(

        outputs, targets,

        reduction='none',

        pos_weight=pos_weight

    )

    return (bce * sample_weights).mean()
 
 
# =========================

# PREP MODEL

# =========================

model = BinaryClassifier(input_dim=X_train.shape[1]).to(device)
 
 
# =========================

# CLASS IMBALANCE HANDLING (FIX 1)

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

# optimizer = optim.Adam(model.parameters(), lr=3e-4)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode='min',

    factor=0.5,

    patience=5

)
 
 
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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,drop_last=True)
 
 
# =========================

# TRAINING LOOP

# =========================

epochs = 100
 
# --- Calibration helpers ---

def smooth_labels(y, smoothing=0.1):

    return y * (1 - smoothing) + 0.5 * smoothing
 
def brier_loss(logits, targets):

    probs = torch.sigmoid(logits)

    return torch.mean((probs - targets) ** 2)
 
for epoch in range(epochs):

    model.train()

    epoch_loss = 0.0
 
    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)

        batch_y = batch_y.to(device)
 
        optimizer.zero_grad()
 
        outputs = model(batch_X)
 
        # =========================

        # 🔥 CALIBRATION CHANGES

        # =========================
 
        # 1. Label smoothing

        batch_y_smooth = smooth_labels(batch_y, smoothing=0.1)
 
        # 2. BCE Loss (with smoothing)

        bce = criterion(outputs, batch_y_smooth)
 
        # 3. Brier loss (probability calibration)

        brier = brier_loss(outputs, batch_y_smooth)
 
        # 4. Logit regularization (controls extreme confidence)

        logit_penalty = 0.01 * torch.mean(outputs ** 2)
 
        # 5. Final combined loss

        loss = 0.6 * bce + 0.3 * brier + 0.1 * logit_penalty
 
        # =========================
 
        loss.backward()

        optimizer.step()
 
        epoch_loss += loss.item()
 
    avg_loss = epoch_loss / len(train_loader)

    scheduler.step(avg_loss)
 
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
 
    # =========================

    # LOGIT DEBUGGING (FIX 5)

    # =========================

    if (epoch + 1) % 5 == 0:

        model.eval()

        with torch.no_grad():

            sample_logits = model(X_train_tensor[:2000].to(device))

            print(f"Logits range: [{sample_logits.min().item():.2f}, {sample_logits.max().item():.2f}]")
 
    # =========================

    # EARLY STOPPING

    # =========================

    if avg_loss < best_loss - 1e-4:

        best_loss = avg_loss

        best_model_state = model.state_dict()

        counter = 0

    else:

        counter += 1

        if counter >= patience:

            print("⏹️ Early stopping triggered.")

            break
 
 
# =========================

# LOAD BEST MODEL

# =========================

model.load_state_dict(best_model_state)

 