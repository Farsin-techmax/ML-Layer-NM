import pandas as pd
import numpy as np
import time

print("Creating dummy data...", flush=True)
np.random.seed(42)
N = 766000
vins = np.random.randint(0, 8900, N)
service_nums = np.random.randint(0, 10, N)
mileages = np.random.randint(0, 100000, N)

serv1 = pd.DataFrame({
    'Vin_No': vins,
    'Service_Num': service_nums,
    'Mileage': mileages
})

dfpmsdate = pd.DataFrame({
    'Vin_No': np.arange(8900),
    'SinglePMS': np.random.randint(0, 2, 8900)
})
last_service_code = 70
svc = None

print("Running derive_pms_mileage_features logic...", flush=True)
start = time.time()

dfpmsmil = serv1[['Vin_No', 'Service_Num', 'Mileage']].copy()
print("1. Filtering", flush=True)
upper = svc if svc is not None else last_service_code
if svc is not None:
    dfpmsmil = dfpmsmil.query('Service_Num > 0 and Service_Num <= @upper')
else:
    dfpmsmil = dfpmsmil.query('Service_Num > 0 and Service_Num < @last_service_code')

print("2. Groupby Diff", flush=True)
dfpmsmil = dfpmsmil.copy()
dfpmsmil['Mileage_Diff'] = dfpmsmil.groupby('Vin_No')['Mileage'].diff()
dfpmsmil['Mileage_Diff'] = dfpmsmil['Mileage_Diff'].fillna(dfpmsmil['Mileage'])
print("3. Groupby Mean", flush=True)
avg_mileage_interval = (
    dfpmsmil.groupby('Vin_No')['Mileage_Diff']
    .mean()
    .reset_index(name='Avg_Mileage_Interval_PMS')
)

print("4. Merges", flush=True)
avg_mileage_interval = avg_mileage_interval.merge(
    dfpmsdate[['Vin_No', 'SinglePMS']], on='Vin_No', how='left'
)

df_max_service = dfpmsmil.groupby("Vin_No", as_index=False)["Service_Num"].max()
avg_mileage_interval = avg_mileage_interval.merge(df_max_service, on='Vin_No', how='left')

print("5. Replacement Means", flush=True)
replacement_means = (
    avg_mileage_interval.query("SinglePMS == 0")
    .groupby('Service_Num')['Avg_Mileage_Interval_PMS']
    .mean()
)

mapped_means = avg_mileage_interval['Service_Num'].map(replacement_means)
mapped_means = mapped_means.fillna(avg_mileage_interval['Avg_Mileage_Interval_PMS'])

avg_mileage_interval['Avg_Mileage_Interval_PMS'] = np.where(
    avg_mileage_interval['SinglePMS'] == 1,
    mapped_means,
    avg_mileage_interval['Avg_Mileage_Interval_PMS']
)

print(f"Done in {time.time() - start:.2f} seconds!")
