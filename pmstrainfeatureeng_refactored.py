import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from features import extract_k1, extract_kk

def get_quarter_dates(selected_quarter, year):
    """
    Returns start_date (Jan 1 of previous year) and quarter_end (last day of the previous quarter).
    """
    if selected_quarter not in ["Q1", "Q2", "Q3", "Q4"]:
        raise ValueError("Invalid quarter selected. Choose from Q1, Q2, Q3, Q4.")

    quarter_months = {"Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)}
    start_month, end_month = quarter_months[selected_quarter]
    
    quarter_start = pd.Timestamp(year=year, month=start_month, day=1)
    previous_quarter_end = quarter_start - pd.Timedelta(days=1)
    
    start_date = pd.Timestamp(year=year - 1, month=1, day=1)
    return start_date, previous_quarter_end

def prepare_pms_datasets(
    df_path,
    service_history_path,
    service_types,
    selected_quarter,
    year,
    target_range=(0.35, 0.45),
    output_prefix="TestTrain"
):
    """
    Refactored, highly-optimized dataset preparation logic.
    Strips away while-loops and correctly captures True Skipped vehicles as negatives.
    Enforces class balance mathematically.
    """
    start_date_dt, quarter_end_dt = get_quarter_dates(selected_quarter, year)
    print(f"Processing for Quarter End: {quarter_end_dt.date()}")
    
    # 1. Vectorized Initialization
    df = pd.read_csv(df_path, low_memory=False, encoding="ISO-8859-1")
    df = df.query("`Last Service - PMS` != '-'").copy()
    df["Last Service Mileage"] = pd.to_numeric(df["Last Service Mileage"], errors="coerce")
    df["Last PMS Mileage"] = pd.to_numeric(df["Last PMS Mileage"], errors="coerce")
    
    date_cols = ["Invoice date", "First Service Date", "Last Service Date - PMS", "Last Service Date", "Next Service Date"]
    for col in date_cols:
        df[col] = df[col].replace("-", pd.NA)
        df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=True, errors="coerce")
        
    df["Service_Num"] = df["Last Service - PMS"].apply(extract_k1)
    df["FirstSrvDate"] = df["Invoice date"].fillna(df["First Service Date"])
    
    # Load Service History (used to verify true skipped vehicles)
    serv = pd.read_csv(service_history_path, low_memory=False, encoding='ISO-8859-1')
    serv["Service_Num"] = serv["Description"].apply(extract_kk)
    
    target_ratio = (target_range[0] + target_range[1]) / 2.0  # e.g., 0.40
    
    for svc in service_types:
        print(f"\nProcessing {svc} ...")
        svc_num = extract_k1(svc)
        
        # 2. Turn-ups (The Positives - TargetFlag = 1)
        turnup = df.query(f"`Last Service - PMS` == '{svc}'").copy()
        turnup["TargetFlag"] = 1
        
        # We need a pool of eligible Negatives to balance against.
        # Compute Expected Date for ALL active/lapsed vehicles
        eligible = df[df["Vehicle Service Status"].isin(["InActive", "Lapsed", "Active"])].copy()
        months_to_add = int(svc_num / 10) * 6
        
        # Fast, vectorized date math
        eligible[f"Expected{svc}Date"] = eligible["FirstSrvDate"] + pd.offsets.DateOffset(months=months_to_add)
        
        # 3. Fixing the Skipped Logic (The True Negatives)
        # Vehicles whose latest service is greater than the target milestone...
        past_milestone = eligible.query(f"Service_Num > {svc_num}")
        # But who NEVER actually did the target milestone in their history!
        serv_target_vins = serv[serv["Service_Num"] == svc_num]["Vin_No"].unique()
        skipped = past_milestone[~past_milestone["VIN"].isin(serv_target_vins)].copy()
        
        # 4. Pending Logic (The Overdue Negatives)
        # Vehicles whose latest service is strictly less than target milestone...
        pending = eligible.query(f"Service_Num < {svc_num}").copy()
        # ...but they were expected to have done it by the end of the quarter
        pending = pending[(pending[f"Expected{svc}Date"] >= start_date_dt) & (pending[f"Expected{svc}Date"] <= quarter_end_dt)]
        # Filter out high mileage pending vehicles just like original code
        milthreshold = svc_num * 1000
        pending = pending.query("`Last Service Mileage` < @milthreshold")
        
        # Combine Skipped + Pending into a single pool of negatives
        all_negatives = pd.concat([skipped, pending], axis=0)
        all_negatives["TargetFlag"] = 0
        
        # 5. Mathematical Balancing
        num_positives = len(turnup)
        # Calculate exactly how many negatives we need to hit the target ratio
        if num_positives > 0:
            required_negatives = int(num_positives / target_ratio) - num_positives
        else:
            required_negatives = 0
            
        print(f"Total Positives: {num_positives}")
        print(f"Total Eligible Negatives: {len(all_negatives)} (Skipped: {len(skipped)}, Pending: {len(pending)})")
        print(f"Target Ratio: {target_ratio*100:.1f}% -> Required Negatives: {required_negatives}")
        
        if len(all_negatives) > required_negatives:
            # Sort by ExpectedDate descending so we keep the most recent ones
            all_negatives = all_negatives.sort_values(by=f"Expected{svc}Date", ascending=False)
            sampled_negatives = all_negatives.head(required_negatives)
        else:
            print("Warning: Not enough eligible negatives to reach target ratio. Taking all available.")
            sampled_negatives = all_negatives
            
        finaltr = pd.concat([turnup, sampled_negatives], axis=0)
        finaltr[f"Expected{svc}Date"] = finaltr[f"Expected{svc}Date"].fillna(finaltr["Last Service Date - PMS"])
        finaltr = finaltr.drop(columns=["FirstSrvDate"], errors="ignore")
        
        final_ratio = len(turnup) / len(finaltr) if len(finaltr) > 0 else 0
        print(f"Final TargetFlag ratio: {round(final_ratio*100, 2)}% | Rows: {len(finaltr)}")
        
        output_file = f"{output_prefix}_{svc}.csv"
        # finaltr.to_csv(output_file, index=False)
        print(f"{svc} dataset saved: {output_file}")
        
    return finaltr

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python pmstrainfeatureeng_refactored.py <milestone_number>")
        print("Example: python pmstrainfeatureeng_refactored.py 20")
        sys.exit(1)
        
    milestone_num = sys.argv[1]
    svc_target = f"{milestone_num}k"
    
    eda_path = os.path.join("data", "EDA - Q3 2025.csv")
    service_history_path = os.path.join("data", "Service History Q1 - 2026.csv")
    
    if not os.path.exists(eda_path):
        print(f"Error: Could not find {eda_path}")
        sys.exit(1)
        
    if not os.path.exists(service_history_path):
        print(f"Error: Could not find {service_history_path}")
        sys.exit(1)
        
    print(f"--- Testing Refactored Data Preparation for {svc_target} ---")
    
    final_df = prepare_pms_datasets(
        df_path=eda_path,
        service_history_path=service_history_path,
        service_types=[svc_target],
        selected_quarter="Q2",
        year=2026,
        target_range=(0.35, 0.45),
        output_prefix="TestRefactored"
    )
    
    print("\n--- Summary ---")
    print(f"Returned DataFrame shape: {final_df.shape}")
    print(final_df['TargetFlag'].value_counts())
    os.makedirs("refactored_test_dir", exist_ok=True)
    final_df.to_parquet(os.path.join("refactored_test_dir", "refactored_mastersheet.parquet"), index=False)
    print(f"Saved final_Df with  {final_df['TargetFlag'].value_counts().to_dict()} and shape  {final_df.shape}")