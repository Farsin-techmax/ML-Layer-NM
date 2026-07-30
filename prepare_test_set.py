import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
from pmstrainfeatureeng_refactored import process_service_data, extract_k1, extract_kk

def create_test_cohort(eda_path, service_history_path, milestone, start_date, end_date, cutoff_date, use_mileage_projection=True):
    """
    Identifies vehicles due for the milestone in the target window,
    and assigns ground truth based on actual service history in that window.
    """
    print(f"Loading EDA data from {eda_path}")
    df = pd.read_csv(eda_path, low_memory=False, encoding="ISO-8859-1")
    df = df.query("`Last Service - PMS` != '-'").copy()
    
    # 1. Compute Expected Date for the milestone
    df["FirstSrvDate"] = df["Invoice date"].fillna(df["First Service Date"])
    df["FirstSrvDate"] = pd.to_datetime(df["FirstSrvDate"], format="mixed", errors="coerce")
    
    months_to_add = int(milestone / 10) * 6
    df["Time_Based_Date"] = df["FirstSrvDate"] + pd.offsets.DateOffset(months=months_to_add)
    
    print(f"Loading actual service history from {service_history_path}...")
    serv = pd.read_csv(service_history_path, low_memory=False, encoding='ISO-8859-1')
    serv['Service_Date'] = pd.to_datetime(serv['Service_Date'], format='mixed', errors='coerce')
    serv['Mileage'] = pd.to_numeric(serv['Mileage'], errors='coerce')
    serv["Service_Num"] = serv["Description"].apply(extract_kk)
    cutoff_dt = pd.to_datetime(cutoff_date)
    
    if use_mileage_projection:
        print("Calculating dynamic mileage projection based on history prior to cutoff...")
        serv_hist = serv[serv['Service_Date'] <= cutoff_dt].dropna(subset=['Mileage'])
        
        # Get max mileage and date per VIN
        latest_serv = serv_hist.sort_values('Service_Date').groupby('Vin_No').tail(1)[['Vin_No', 'Service_Date', 'Mileage']]
    
        latest_serv = latest_serv.rename(columns={'Service_Date': 'Last_Date', 'Mileage': 'Last_Mileage'})
        
        cohort_proj = df[['VIN', 'FirstSrvDate', 'Time_Based_Date']].merge(latest_serv, left_on='VIN', right_on='Vin_No', how='left')
        
        cohort_proj['Days_Since_Start'] = (cohort_proj['Last_Date'] - cohort_proj['FirstSrvDate']).dt.days
        cohort_proj['Daily_Mileage'] = np.where(cohort_proj['Days_Since_Start'] > 30, cohort_proj['Last_Mileage'] / cohort_proj['Days_Since_Start'], np.nan)
        
        target_mileage = milestone * 1000
        cohort_proj['Miles_Remaining'] = target_mileage - cohort_proj['Last_Mileage']
        cohort_proj['Days_Remaining'] = cohort_proj['Miles_Remaining'] / cohort_proj['Daily_Mileage']

        # limit the number of days remaining between 0 and 10 years
        cohort_proj['Days_Remaining'] = cohort_proj['Days_Remaining'].where(cohort_proj['Days_Remaining'].between(0, 3650), np.nan)
        
        days_rem = cohort_proj['Days_Remaining'].replace([np.inf, -np.inf], np.nan).fillna(0)
        cohort_proj['Projected_Date'] = cohort_proj['Last_Date'] + pd.to_timedelta(days_rem, unit='D')
        
        invalid_mask = cohort_proj['Projected_Date'].isnull() | (cohort_proj['Days_Remaining'] < 0) | (cohort_proj['Daily_Mileage'] <= 0)
        cohort_proj.loc[invalid_mask, 'Projected_Date'] = cohort_proj.loc[invalid_mask, 'Time_Based_Date']
        
        df[f"Expected{milestone}kDate"] = cohort_proj['Projected_Date'].values
    else:
        df[f"Expected{milestone}kDate"] = df["Time_Based_Date"]
    
    # 2. Filter for vehicles DUE in the target window
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    cohort = df[
        (df[f"Expected{milestone}kDate"] >= start_dt) & 
        (df[f"Expected{milestone}kDate"] <= end_dt) &
        (df["Vehicle Service Status"].isin(["InActive", "Lapsed", "Active"]))
    ].copy()
    
    # 3. Exclude vehicles that ALREADY completed this milestone (or higher) before the cutoff date
    # DO NOT use EDA's 'Last Service - PMS' because it leaks future data! 
    # Use raw service history up to cutoff_dt.
    print("Checking historical records for early completions...")
    past_services = serv[
        (serv["Service_Date"] <= cutoff_dt) & 
        (serv["Service_Num"] >= milestone) & 
        (serv["Vin_No"].isin(cohort["VIN"]))
    ]
    already_done_vins = set(past_services["Vin_No"].unique())
    cohort = cohort[~cohort["VIN"].isin(already_done_vins)].copy()
    
    print(f"Found {len(cohort)} vehicles due for {milestone}k between {start_date} and {end_date}")
    
    # 4. Label with Ground Truth (Did they EVER show up for this milestone?)
    print(f"Assigning ground truth...")
    actual_services = serv[
        (serv["Service_Num"] == milestone)
    ]
    
    vins_that_showed_up = set(actual_services["Vin_No"].unique())
    
    cohort["TargetFlag"] = cohort["VIN"].apply(lambda x: 1 if x in vins_that_showed_up else 0)
    
    turnup_count = cohort["TargetFlag"].sum()
    print(f"Ground Truth assigned: {turnup_count} showed up ({turnup_count/len(cohort)*100:.1f}%), {len(cohort)-turnup_count} did not.")
    
    return cohort

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PMS Test Set")
    parser.add_argument("--milestone", type=int, default=20)
    parser.add_argument("--window-start", type=str, default="2026-01-01", help="Start of prediction window")
    parser.add_argument("--window-end", type=str, default="2026-03-31", help="End of prediction window")
    parser.add_argument("--train-cutoff", type=str, default="2025-12-31", help="Date to cut off feature generation")
    parser.add_argument("--enable-mileage-projection", action="store_true", help="Use dynamic mileage projection for expected date")
    args = parser.parse_args()
    
    # Files
    eda_path = os.path.join("data", "EDA - Q3 2025.csv")
    service_history_path = os.path.join("data", "Service History Q1 - 2026.csv")
    rfm_path = "data/RFM_Q2-2026.csv"
    appointdf_path = "data/Appointments_Q2-2026.csv"
    digidf_path = "data/Digital_Sessions_Q2-2026.csv"
    vhc_path = "data/VHC_Q2-2026.csv"
    servcode_path = "data/Service Code Desc.csv"
    
    print(f"--- Building Test Set for {args.milestone}k ---")
    print(f"Prediction Window: {args.window_start} to {args.window_end}")
    print(f"Feature Cutoff Date: {args.train_cutoff}")
    print(f"Dynamic Mileage Projection: {'Enabled' if args.enable_mileage_projection else 'Disabled'}")
    
    # 1. Build Cohort and Label Ground Truth
    cohort = create_test_cohort(
        eda_path=eda_path,
        service_history_path=service_history_path,
        milestone=args.milestone,
        start_date=args.window_start,
        end_date=args.window_end,
        cutoff_date=args.train_cutoff,
        use_mileage_projection=args.enable_mileage_projection
    )
    
    # 2. Process Service Data to generate features
    # CRITICAL: We pass filter_date=train_cutoff so it only uses history BEFORE the prediction window
    print(f"\nProcessing features using data up to {args.train_cutoff}...")
    
    # Ensure Service_Num is present as process_service_data expects it
    cohort["Service_Num"] = cohort["Last Service - PMS"].apply(extract_k1)
    
    test_features = process_service_data(
        mastersheet=cohort,
        servhistory=service_history_path,
        rfm=rfm_path,
        servcode=servcode_path,
        appointdf=appointdf_path,
        digidf=digidf_path,
        vhc=vhc_path,
        filter_date=args.train_cutoff,
        last_service_code=args.milestone,
        is_test=True
    )
    
    # 3. Filter Columns to match Training Set exactly
    feature_list_path = rf"models\{args.milestone}k\selected_features.json"
    if not os.path.exists(feature_list_path):
        print(f"Error: Could not find trained features at {feature_list_path}. Run training first.")
        sys.exit(1)
        
    with open(feature_list_path, 'r') as f:
        selected_features = json.load(f)
        
    final_cols = ["VIN", "TargetFlag"] + selected_features
    
    # Handle missing columns (in case a one-hot encoded category didn't appear in the test set)
    for col in final_cols:
        if col not in test_features.columns:
            print(f"Warning: Column {col} missing in test set. Filling with 0.")
            test_features[col] = 0
            
    final_test_set = test_features[final_cols]
    
    os.makedirs("test_sets", exist_ok=True)
    test_path = rf"test_sets\test_{args.milestone}k_Q1_2026.csv"
    final_test_set.to_csv(test_path, index=False)
    
    print(f"\nSuccess! Test set saved to {test_path}")
    print(f"Final test shape: {final_test_set.shape}")
