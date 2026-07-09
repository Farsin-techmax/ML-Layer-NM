import numpy as np
import pandas as pd
#adding a function to flag milestones the vehicle has completed

#loading the service history data
service_history_clean = None
try:
  service_history_clean = pd.read_parquet('./data/service_history_clean.parquet', engine='fastparquet')
except FileNotFoundError:
  print("File not found")
except pd.errors.EmptyDataError:
  print("No data in file")
except pd.errors.ParserError:
  print("Error parsing file")

# Exit if file could not be loaded
if service_history_clean is None:
  exit()

# Ensure that the 'vehicle_key' and 'description' columns are string type
service_history_clean['vehicle_key'] = service_history_clean['vehicle_key'].astype(str)
service_history_clean['description'] = service_history_clean['description'].astype(str)

# Define the milestones
milestones = {
    '1':      ('has_1',   0,     5000),
    '<=10':   ('has_10',  5000,  15000),
    '11-20':  ('has_20',  10000, 25000),
    '21-30':  ('has_30',  20000, 35000),
    '31-40':  ('has_40',  30000, 45000),
    '41-50':  ('has_50',  40000, 65000),
    '51-60':  ('has_60',  50000, 75000),
    '61-70':  ('has_70',  60000, 85000),
    '71-80':  ('has_80',  70000, 95000),
    '81-90':  ('has_90',  80000, 105000),
    '91-100': ('has_100', 90000, 115000),
    '101-110': ('has_110', 105000, 125000),
    '111-120': ('has_120', 115000, 135000),
    '121-130': ('has_130', 120000, 145000),
}

# Create a flag DataFrame
flag_df = pd.DataFrame(index=service_history_clean['vehicle_key'].unique(), columns=[flag for desc, (flag, low, high) in milestones.items()])

# Loop over the milestones
for desc, (flag, low, high) in milestones.items():
  # Create a mask for the current milestone
  mask = (service_history_clean['description'] == desc) & (service_history_clean['mileage'].between(low, high))

  # Group by 'vehicle_key' and sum the mask
  flag_df[flag] = service_history_clean.loc[mask].groupby('vehicle_key')['description'].count().fillna(0).astype(int)

# Replace non-zero values with 1
flag_df = (flag_df > 0).astype(int)

# Reset the index
flag_df = flag_df.reset_index()
print(flag_df.head())

#saving the flag DataFrame to a parquet file
try:
  flag_df.to_parquet('./data/flags_df.parquet', engine='fastparquet', index=False)
except Exception as e:
  print(f"Error saving flag DataFrame: {e}")

