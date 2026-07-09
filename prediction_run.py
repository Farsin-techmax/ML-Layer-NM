predfuture = pd.read_csv('70kPMSTestDataQ3.csv')

predfuture.loc[predfuture['TargetFlag'] == 1, 'PMS_Delay'] -= 1

predfuture.loc[predfuture['TargetFlag'] == 1, 'Vehicle_Key_Actual_Service'] -= 1

predfuture = predfuture.query("Franchise_NISSAN == 1")

actuals = predfuture['TargetFlag']

result = list(set(indfeats) - set(list(predfuture.iloc[:,5:].columns))) #indfeats - Features used in training

print(result)

predfuture[result] = 0

X_sample = predfuture[indfeats] #indfeats - Features used in training

X_sample_scaled = scaler.transform(X_sample)

X_sample_scaled
 
def predict_proba(model, X, temperature=1.0):

    model.eval()

    with torch.no_grad():

        logits = model(X.to(device))

        probs = torch.sigmoid(logits / temperature)  # smoother probabilities

    return probs.cpu()
 
X_sample_tensor = torch.tensor(X_sample_scaled, dtype=torch.float32).to('cpu') #-6 -3

model.eval()

# model.eval()

logits_test = model(X_sample_tensor).detach().cpu().numpy()

# with torch.no_grad():

#     logits = model(X_sample_tensor)

#     # logits = model(X_sample_tensor)

#     probs = torch.sigmoid(logits)             # Get probabilities (if binary classification)

#     # cal_probs = platt.predict_proba(logits_test)[:, 1]

#     # preds = (cal_probs >= 0.5).astype(float).reshape(-1, 1)

#     preds = (probs >= 0.6).float()              # Convert to class (0 or 1)

probs = predict_proba(model, X_sample_tensor)

preds = (probs > 0.5).float() 

predfuture['Predicted_Prob'] = probs.numpy().flatten()

predfuture['Predicted_Class'] = preds.numpy().flatten().astype(int)

predfuture['Predicted_Prob'] = round(predfuture['Predicted_Prob'],5)

predfuture
 
import matplotlib.pyplot as plt
 
# If probs is shape (N,1), flatten it

probs_flat = probs.flatten()
 
# =========================

# HISTOGRAM + KDE STYLE CURVE

# =========================

plt.figure(figsize=(10,6))
 
# Histogram

plt.hist(probs_flat, bins=50, density=True)
 
# KDE-like smooth curve (manual Gaussian smoothing)

from scipy.stats import gaussian_kde

kde = gaussian_kde(probs_flat)

x_vals = np.linspace(0, 1, 500)

y_vals = kde(x_vals)
 
plt.plot(x_vals, y_vals)
 
plt.title("Probability Distribution (Histogram + KDE)")

plt.xlabel("Predicted Probability")

plt.ylabel("Density")
 
plt.grid(alpha=0.3)

plt.show()
 