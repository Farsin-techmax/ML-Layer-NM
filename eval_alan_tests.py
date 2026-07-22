import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import json
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss, confusion_matrix
import joblib

class ServicePredictionNN(nn.Module):
    def __init__(self, input_dim):
        super(ServicePredictionNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def evaluate(milestone, q, test_path):
    print(f"\n--- Evaluating {milestone}k {q} ---")
    model_dir = f"models/{milestone}k/training"
    
    # 1. Load selected features
    feat_path = f"{model_dir}/selected_features.json"
    if not os.path.exists(feat_path):
        feat_path = f"{model_dir}/../selected_features.json"
        
    with open(feat_path, 'r') as f:
        selected_features = json.load(f)
    
    # 2. Load test data
    test_df = pd.read_csv(test_path)
    
    # Left join preserved all rows, but they might have NaNs for missing features.
    # We should fill missing columns with 0
    for col in selected_features:
        if col not in test_df.columns:
            test_df[col] = 0
            
    # Conditional dropping of PMS_Delay and Vehicle_Key_Actual_Service if milestone is 100k
    if milestone >= 100:
        cols_to_drop = ['PMS_Delay', 'Vehicle_Key_Actual_Service']
        selected_features = [f for f in selected_features if f not in cols_to_drop]
            
    X_test = test_df[selected_features].copy()
    y_test = test_df['TargetFlag'].values
    
    # 3. Impute and Scale
    imputer = joblib.load(f"{model_dir}/imputer.pkl")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    
    X_test_scaled = scaler.transform(imputer.transform(X_test))
    
    # 4. Load Model
    device = torch.device("cpu")
    model = ServicePredictionNN(input_dim=len(selected_features)).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/best_nn_model.pt", map_location=device))
    model.eval()
    
    # 5. Predict
    X_tensor = torch.FloatTensor(X_test_scaled).to(device)
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
        
    preds = (probs > 0.5).astype(int)
    
    # 6. Metrics
    try:
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        brier = brier_score_loss(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        
        print(f"ROC-AUC:   {auc:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"Brier:     {brier:.4f}")
        print(f"Confusion Matrix:\n{cm}")
    except ValueError as e:
        print(f"Error calculating metrics: {e}")

configs = [
    {"q": "Q1", "milestone": 80, "path": r"c:\Techmax\cwf\code\tests_alan\Q1\80k\test_features.csv"},
    {"q": "Q1", "milestone": 90, "path": r"c:\Techmax\cwf\code\tests_alan\Q1\90k\test_features.csv"},
    {"q": "Q1", "milestone": 100, "path": r"c:\Techmax\cwf\code\tests_alan\Q1\100k\test_features.csv"},
    {"q": "Q2", "milestone": 80, "path": r"c:\Techmax\cwf\code\tests_alan\Q2\80k\test_features.csv"},
    {"q": "Q2", "milestone": 90, "path": r"c:\Techmax\cwf\code\tests_alan\Q2\90k\test_features.csv"},
    {"q": "Q2", "milestone": 100, "path": r"c:\Techmax\cwf\code\tests_alan\Q2\100k\test_features.csv"}
]

for cfg in configs:
    evaluate(cfg['milestone'], cfg['q'], cfg['path'])
