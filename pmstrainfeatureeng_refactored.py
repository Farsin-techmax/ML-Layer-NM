import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from features import extract_k1, extract_kk



from kneed import KneeLocator
# import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from collections import defaultdict
from sklearn.linear_model import Lasso
from datetime import datetime, timedelta
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform
from dateutil.relativedelta import relativedelta
from sklearn.exceptions import ConvergenceWarning
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_selection import mutual_info_classif
import sys
from features import *  # Shared feature engineering functions

# Ignore DeprecationWarning only
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ignore FutureWarning only
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# ---------------- Helper Functions ----------------

try:
    warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
except Exception:
    # Fallback for pandas versions without SettingWithCopyWarning
    try:
        pd.options.mode.chained_assignment = None
    except Exception:
        pass
warnings.filterwarnings("ignore", category=hierarchy.ClusterWarning)



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Line %(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('pms_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)




#function starts here




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
    output_prefix="TestTrain",
):
    """
    Original target labelling and sampling:
    - Positives: ALL vehicles whose EDA master sheet shows they completed the target PMS
    - Negatives: Vehicles pending/overdue for the target PMS within a sliding date window
    - NotTurnUp: Vehicles that skipped the target PMS (did a higher service without the target)
    - While loop adjusts the date window until positive ratio is within target_range
    """
    start_date_dt, quarter_end_dt = get_quarter_dates(selected_quarter, year)
    
    print(f"Processing for Quarter End: {quarter_end_dt.date()}")
    
    # Load main dataframe
    df = pd.read_csv(df_path, low_memory=False, encoding="ISO-8859-1")
    df = df.query("`Last Service - PMS` != '-'").copy()
    df["Last Service Mileage"] = pd.to_numeric(df["Last Service Mileage"], errors="coerce")
    df["Last PMS Mileage"] = pd.to_numeric(df["Last PMS Mileage"], errors="coerce")
    
    date_cols = ["Invoice date", "First Service Date", "Last Service Date - PMS", "Last Service Date", "Next Service Date"]
    for col in date_cols:
        df[col] = df[col].replace("-", pd.NA)
        df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=True, errors="coerce")
    
    print(f"Main DataFrame Shape: {df.shape[0]}")
    df["Service_Num"] = df["Last Service - PMS"].apply(extract_k1)
    df["FirstSrvDate"] = df["Invoice date"].fillna(df["First Service Date"])
    
    # Load Service History
    serv = pd.read_csv(service_history_path, low_memory=False, encoding='ISO-8859-1')
    serv["Service_Num"] = serv["Description"].apply(extract_kk)
    serv['Service_Date'] = pd.to_datetime(serv['Service_Date'], format='mixed', dayfirst=True, errors='coerce')
    
    start_date = pd.to_datetime("2018-01-01")
    
    for svc in service_types:
        print(f"\nProcessing {svc} ...")
        svc_num = extract_k1(svc)
        start_date_adj = start_date_dt
        max_iterations = (quarter_end_dt.year - start_date.year) * 12 + (quarter_end_dt.month - start_date.month)
        iteration = 0
        finaltr = pd.DataFrame()
        print(f'PMS number: {svc_num}')
        
        # ── Precompute base sets outside the loop ──
        # POSITIVES
        turnup = df.query(f"`Last Service - PMS` == '{svc}'").copy()
        turnup["TargetFlag"] = 1
        
        df_target = df[~df["VIN"].isin(turnup["VIN"].unique())]
        
        # SKIPPED VINs
        dfskipped = df_target.query(f"Service_Num > {svc_num - 10}")["VIN"].unique()
        serv_filtered = serv[serv["Vin_No"].isin(dfskipped)]
        history_grouped = serv_filtered.groupby("Vin_No")["Service_Num"].apply(list).reset_index(name="Completed_Services")
        filtered_df = history_grouped[~history_grouped["Completed_Services"].apply(lambda x: svc_num in x)]
        
        # NotTurnUp VINs
        NotTurnUpdf = df[df["VIN"].isin(filtered_df["Vin_No"].unique())].query(f"Service_Num == {svc_num + 10}")
        if not NotTurnUpdf.empty:
            NotTurnUpdf = NotTurnUpdf.copy()
            NotTurnUpdf["VIN_base"] = NotTurnUpdf["Vehicle Key"].str.rsplit("-", n=1).str[0]
            NotTurnUpdf["Owner_No"] = NotTurnUpdf["Vehicle Key"].str.extract(r"-(\d+)$")[0].fillna(0).astype(int)
            NotTurnUpdf = NotTurnUpdf.loc[
                NotTurnUpdf.groupby("VIN_base")["Owner_No"].idxmax()
            ].reset_index(drop=True)
            NotTurnUpdf = NotTurnUpdf.sort_values(by="Service_Num", ascending=False).drop_duplicates(subset="VIN", keep="first")
            NotTurnUpdf = NotTurnUpdf.drop(["VIN_base", "Owner_No"], axis=1)
            
        # PENDING VINs Base (Negatives)
        missed = df[~df["VIN"].isin(NotTurnUpdf["VIN"].unique()) if not NotTurnUpdf.empty else pd.Series([True]*len(df))]
        missed = missed[~missed["VIN"].isin(turnup["VIN"].unique())].copy()
        missed["FirstSrvDate"] = missed["Invoice date"].fillna(missed["First Service Date"])
        missed["Service_Num"] = missed["Last Service - PMS"].apply(extract_k1)
        months_to_add = int(svc_num / 10) * 6
        missed[f"Expected{svc}Date"] = missed["FirstSrvDate"] + pd.DateOffset(months=months_to_add)
        
        missed = missed[missed["Vehicle Service Status"].isin(["InActive", "Lapsed", "Active"])]
        pending_base = missed.query(f"Service_Num < {svc_num}").copy()
        dfafter = df.query(f"Service_Num >= {svc_num}")["VIN"].unique()
        pending_base = pending_base[~pending_base["VIN"].isin(dfafter)]
        pending_base = pending_base.drop(["FirstSrvDate"], axis=1)
        milthreshold = svc_num * 1000
        pending_base = pending_base.query("`Last Service Mileage` < @milthreshold")
        
        while iteration < max_iterations:
            print(f'{svc} - Iteration {iteration + 1}: Start Date = {start_date_adj.date()}')
            
            pending = pending_base[
                (pending_base[f"Expected{svc}Date"] >= start_date_adj) &
                (pending_base[f"Expected{svc}Date"] <= quarter_end_dt)
            ]
            
            # ── Combine Positives + Negatives ──
            if not pending.empty:
                pending = pending.copy()
                pending["TargetFlag"] = 0
            
            finaltr = pd.concat([turnup, pending], axis=0)
            finaltr[f"Expected{svc}Date"] = finaltr[f"Expected{svc}Date"].fillna(finaltr["Last Service Date - PMS"])
            
            # Calculate ratio
            if len(finaltr) > 0:
                ratio = finaltr["TargetFlag"].sum() / len(finaltr)
            else:
                ratio = 0
            print(f'Target Range: {target_range[0]} to {target_range[1]}')
            print(f'Iteration {iteration + 1}: TargetFlag ratio = {round(ratio * 100, 2)}% | Rows: {len(finaltr)}')
            
            # Stop if ratio in range
            if target_range[0] <= ratio <= target_range[1]:
                break
            elif ratio < target_range[0]:
                start_date_adj += relativedelta(months=1)  # move forward
            else:  # ratio > target_range[1]
                start_date_adj -= relativedelta(months=1)  # move backward
            
            iteration += 1
        
        finaltr = finaltr.drop(columns=["FirstSrvDate"], errors="ignore")
        print(f"Final start date for {svc}: {start_date_adj.date()}")
        print(f"Final TargetFlag ratio: {round(ratio * 100, 2)}% | Rows: {len(finaltr)}")
        
        output_file = f"{output_prefix}_{svc}.csv"
        print(f"{svc} dataset saved: {output_file}")
    
    return finaltr




"""
function 2 -refactored process_service_date
"""
def derive_milestone_intervals(serv1: pd.DataFrame, mastertrain: pd.DataFrame, last_service_code: int) -> pd.DataFrame:
    """
    Vectorized extraction of months to each milestone (months_to_10k, months_10_to_20, etc.)
    """
    # 1. Base date
    npms_serv = serv1[serv1['Service_Num'] < 10]
    first_npms = npms_serv.groupby('Vin_No')['Service_Date'].min().reset_index()
    first_npms = first_npms.rename(columns={'Service_Date': 'Date_first_npms'})
    
    master_dates = mastertrain[['VIN', 'Invoice date', 'First Service Date']].copy()
    master_dates['FirstSrvDate'] = pd.to_datetime(master_dates['Invoice date'].fillna(master_dates['First Service Date']), format='mixed', errors='coerce')
    
    temp_dates = pd.DataFrame({'Vin_No': mastertrain['VIN'].unique()})
    temp_dates = temp_dates.merge(first_npms, on='Vin_No', how='left')
    temp_dates = temp_dates.merge(master_dates[['VIN', 'FirstSrvDate']], left_on='Vin_No', right_on='VIN', how='left')
    temp_dates['Base_Date'] = temp_dates[['FirstSrvDate', 'Date_first_npms']].min(axis=1)
    
    # 2. Pivot milestones
    previous_milestones = [10 * i for i in range(1, (last_service_code // 10))]
    if not previous_milestones:
        return pd.DataFrame({'Vin_No': temp_dates['Vin_No']})
        
    milestone_dates = serv1[serv1['Service_Num'].isin(previous_milestones)].groupby(['Vin_No', 'Service_Num'])['Service_Date'].min().unstack()
    milestone_dates.columns = [f'Date_{col}k' for col in milestone_dates.columns]
    temp_dates = temp_dates.merge(milestone_dates, left_on='Vin_No', right_index=True, how='left')
    
    # 3. Intervals
    res = pd.DataFrame({'Vin_No': temp_dates['Vin_No']})
    if 'Date_10k' in temp_dates.columns:
        res['months_base_to_10'] = (temp_dates['Date_10k'] - temp_dates['Base_Date']).dt.days / 30.44
        
    for i in range(1, len(previous_milestones)):
        prev_mil = previous_milestones[i-1]
        curr_mil = previous_milestones[i]
        col_curr = f'Date_{curr_mil}k'
        col_prev = f'Date_{prev_mil}k'
        if col_curr in temp_dates.columns and col_prev in temp_dates.columns:
            res[f'months_{prev_mil}_to_{curr_mil}'] = (temp_dates[col_curr] - temp_dates[col_prev]).dt.days / 30.44
            
    return res

def process_service_data(mastersheet: str,servhistory: str, rfm: str, appointdf: str, digidf: str,vhc: str,servcode: str,filter_date: str,last_service_code: int, is_test: bool = False) -> pd.DataFrame:
    """
    Process service history data from CSV file.
    
    Args:
        mastersheet (str): Path to the master sheet CSV file
        servhistory (str): Path to the service history CSV file
        filter_date (str): Date string in YYYY-MM-DD format
        last_service_code (int): Code for the last service
        is_test (bool): If True, skips dropping mastertrain rows with expected dates > filter_date.
        
    Returns:
        pd.DataFrame: Processed and filtered service data
    """
    logger.info("Starting service data processing")
    
    try:
        # Convert filter date
        filterdate = pd.to_datetime(filter_date)
        logger.info(f'Filter date: {filterdate}')
        
        # Validate input files
        if not os.path.exists(servhistory):
            logger.error(f"Service history file not found: {servhistory}")
            raise FileNotFoundError(f"Service history file not found: {servhistory}")
            
        # Read and process master sheet
        try:
            logger.info("Reading master sheet data")
            col_name = f"Expected{last_service_code}kDate"
            if isinstance(mastersheet, pd.DataFrame):
                mastertrain = mastersheet.copy()
            elif mastersheet.endswith('.parquet'):
                mastertrain = pd.read_parquet(mastersheet, engine='fastparquet')
            else:
                mastertrain = pd.read_csv(mastersheet, low_memory=False)
            logger.debug(f"Master sheet initial shape: {mastertrain.shape}")
            
            logger.info("Converting dates in master sheet")
            mastertrain['Last Service Date - PMS'] = pd.to_datetime(mastertrain['Last Service Date - PMS'],format='mixed',dayfirst=True,errors='coerce')
            mastertrain[col_name] = pd.to_datetime(mastertrain[col_name],format='mixed',errors='coerce')
            
            if not is_test:
                mastertrain = mastertrain.query(f"{col_name} <= @filterdate and `Last Service Date - PMS` <= @filterdate")
                logger.info(f"Master sheet shape after date filtering: {mastertrain.shape}")
            else:
                logger.info(f"Skipping master sheet date filtering for TEST mode. Shape: {mastertrain.shape}")
            servcode_desc = pd.read_csv(servcode, skipinitialspace=True)
            servcode_desc = servcode_desc.rename(columns={'SO_CO_CODE': 'Service_Code'})
            # mastertrain.to_csv('validatecode/mastertrain_initial.csv', index=False)
        except Exception as e:
            logger.error(f"Error processing master sheet: {str(e)}")
            raise
        
        # Read and process service history
        try:
            logger.info("Reading service history data")
            serv1 = pd.read_csv(servhistory,low_memory=False) #Input Service History
            logger.debug(f"Service history initial shape: {serv1.shape}")
            
            logger.info("Processing service history data")
            serv1['Service_Date'] = pd.to_datetime(serv1['Service_Date'],format='mixed',dayfirst=True,errors='coerce')
            invalid_dates = serv1['Service_Date'].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} invalid dates in service history. Dropping them.")
                serv1 = serv1.dropna(subset=['Service_Date'])
            
            serv1 = serv1.query(f"Service_Date <= @filterdate")
            logger.info(f"Service history shape after date filtering: {serv1.shape}")
            
            logger.info("Converting numeric columns")
            serv1['Mileage'] = pd.to_numeric(serv1['Mileage'],errors='coerce')
            serv1['Revenue'] = pd.to_numeric(serv1['Revenue'],errors='coerce')
            
            logger.info("Processing service numbers")
            serv1['Service_Num'] = serv1['Description'].apply(extract_kk)
            serv1['Service_Num'] = np.where(serv1['Description'] == '<=10', 10, serv1['Service_Num'])
            
            # --- New Feature: Milestone Completion Flags ---
            previous_milestones = [10 * i for i in range(1, (last_service_code // 10))]
            for mil in previous_milestones:
                vins_with_mil = serv1[serv1['Service_Num'] == mil]['Vin_No'].unique()
                mastertrain[f'has_{mil}'] = mastertrain['VIN'].isin(vins_with_mil).astype(int)
            logger.info(f"Added milestone flags for: {previous_milestones}")
            
            # Filter and process service data
            logger.info("Filtering and processing service data")
            serv1 = serv1.query(f"Service_Num < {last_service_code}")
            serv = serv1[serv1['Vin_No'].isin(mastertrain['VIN'].unique())]
            servM = serv
            serv.to_csv('validatecode/serv_filteredpmsapp.csv', index=False)
            print("Completed filtering service data for PMS features")
            serv = serv.query("Description != 'Others'")
        
            # serv.to_csv('validatecode/serv_filteredpmschkmileage.csv', index=False)
            
            logger.info(f'Max service number in data: {serv["Service_Num"].max()}')
            logger.info(f'Max Service date in data: {serv["Service_Date"].max()}')
        except Exception as e:
            logger.error(f"Error processing service history: {str(e)}")
            raise
        
        # Aggregate service data
        logger.info("Aggregating service data")
        newserv20k = serv.groupby(['Vin_No']).agg(
            Service_Date=('Service_Date', 'max'),
            Mileage=('Mileage', 'max'),
            Total_Revenue=('Revenue', 'sum'),
        ).reset_index()
        
        logger.info(f'UNIQUE VINS in service data: {newserv20k["Vin_No"].nunique()}')

        # Feature Derivation
        logger.info("Starting feature derivation")
        
        # Feature derivation starts here
        try:
            logger.info("Merging PMS dates")
            pmsdf = derive_pms_features1(serv1, last_service_code, reference_date=pd.to_datetime(filterdate))
            logger.info(f"Merging PMS dates, Type: {type(pmsdf)}")

            logger.info("Merging NPMS dates")
            npmsdf = derive_npms_features(serv1, reference_date=pd.to_datetime(filterdate))
            logger.info(f"Merging NPMS dates, Type: {type(npmsdf)}")

            logger.info("Last Non PMS Mileage calculation")
            lastnonpmsmil = get_last_nonpms_mileage(serv1,mastertrain, vin_col="Vin_No", mileage_col="Mileage",
                            date_col="Service_Date")
            
            logger.info("Deriving NonPMS Mileage")
            res1 = get_last_nonpms_before_targetpms(serv1, "Vin_No","Mileage","Service_Date",mastertrain)
            res2 = get_last_nonpms_mileage(serv1, mastertrain,vin_col="Vin_No", mileage_col="Mileage",date_col="Service_Date")
            lastnonpmsmil = pd.concat([res1,res2],axis=0)

            logger.info("Calculating average mileage intervals for PMS")
            avg_mileage_interval = derive_pms_mileage_features(serv1, pmsdf, last_service_code)
            
            logger.info("Calculating average mileage intervals for NPMS")
            avg_mileage_interval_non_pms = derive_npms_mileage_features(serv1)
            
            logger.info("Deriving PMS service intervals")
            avg_monthly_interval = derive_pms_service_intervals(serv, last_service_code)
            # avg_monthly_interval.to_csv(f'validatecode/avgmonthlyintervalpmsM{last_service_code}kpreldb.csv', index=False)
            logger.info("Adjusting service intervals")
            avg_monthly_interval1 = adjust_service_intervals(avg_monthly_interval,last_service_code)
            
            logger.info("Calculating NPMS metrics")
            freq_npms, npms_counts, npmsrevenue, pms_counts = derive_npms_features2(serv1)
            Npmsevents,npms_counts = get_non_pms_events(servM, servcode_desc, mastertrain, last_service_code)

            logger.info("Calculating NPMS/PMS Revenue metrics")
            revenu = compute_service_features(servM,filterdate ,last_service_num=last_service_code)

        except Exception as e:
            logger.error(f"Error in feature derivation: {str(e)}")
            raise

        # Log descriptive statistics
        logger.info("Logging descriptive statistics")
        logger.info(f'NPMS counts:\n{npms_counts["nNPMS"].describe()}')
        logger.info(f'NPMS revenue:\n{npmsrevenue["npmsRevenue"].describe()}')
        logger.info(f'PMS counts:\n{pms_counts["nPMS"].describe()}')
        logger.info(f'Freq NPMS:\n{freq_npms["freq_NPMS"].describe()}')

        logger.info("Aggregating service data")
        newserv20k = serv.groupby(['Vin_No']).agg(
            Service_Date=('Service_Date','max'),
            Mileage=('Mileage', 'max'),
            Total_Revenue=('Revenue', 'sum'),
        ).reset_index()
        logger.info(f'UNIQUE VINS in service data: {newserv20k["Vin_No"].nunique()}')

        # Log descriptive statistics
        logger.info(f'NPMS counts:\n{npms_counts["nNPMS"].describe()}')
        logger.info(f'NPMS revenue:\n{npmsrevenue["npmsRevenue"].describe()}')
        logger.info(f'PMS counts:\n{pms_counts["nPMS"].describe()}')
        logger.info(f'Freq NPMS:\n{freq_npms["freq_NPMS"].describe()}')

        # Branch features
        logger.info("Calculating branch-related features")
        vin_branch_pivot,top_branches = branch_visit_features(serv, last_service_code)
        branch_grouped = branch_diversity_features(serv, last_service_code)

        # appoinshow = pd.read_csv(appointshow, low_memory=False) #Input appointment show data
        # appoindf = map_appointments_to_services(appoinshow, servM) 
        # finalappoint = build_final_pms_appointment_summary(appoindf, mastertrain[['VIN','Service_Num']])
        # fnlappnt = adjust_for_target_pms(finalappoint, appoindf, last_service_code)
        # appoinoshow = pd.read_csv(appointnoshow, low_memory=False) #Input appointment no-show data
        # fnlappoinoshow = compute_no_show_appointments(
        #     master_df= mastertrain[['VIN','Last Service Date - PMS','TargetFlag']],
        #     noshow_df=appoinoshow,
        #     filter_date=filterdate,   
        #     vin_col="VIN",
        #     targetflag_col="TargetFlag",
        #     last_pms_date_col="Last Service Date - PMS",
        #     wip_deleted_col="WIP Deleted"
        # )
        complaint_features = transform_complaint_features(mastertrain[['VIN','Service_Num']], servM)
        # fnlappnt.to_csv('validatecode/finalappointpms40kpred.csv', index=False)
        # fnlappoinoshow.to_csv('validatecode/finalnoshowpms40kpred.csv', index=False)
        # complaint_features.to_csv('validatecode/complaintfeatures40kpred.csv', index=False)
        avg_monthly_interval1 = avg_monthly_interval1.rename(columns = {
            "Avg_Service_Interval_PMS": "Avg_Service_Interval_PMSold",
            "Avg_Service_Interval_PMS1": "Avg_Service_Interval_PMS1old",
            "Avg_Service_Interval_PMSnew": "Avg_Service_Interval_PMSper10k",
            "Avg_Service_Interval_PMS1new": "Avg_Service_Interval_PMS"
                })
        # avg_monthly_interval1.to_csv(f'validatecode/avgmonthlyintervalpms{last_service_code}kpred.csv', index=False)
        # Merging all features
        logger.info("Merging all features")
        try:
            logger.info("Deriving milestone intervals feature")
            milestone_intervals_df = derive_milestone_intervals(serv1, mastertrain, last_service_code)

            newserv20ka = newserv20k.copy()
            # Unpack the tuple returned from derive_npms_features

            merge_operations = [
                (npms_counts, 'NPMS counts'),
                (Npmsevents, 'NPMS events'),
                (pmsdf, 'PMS dates'),
                (npmsdf, 'NPMS dates'),
                (avg_mileage_interval, 'Mileage intervals PMS'),
                (avg_mileage_interval_non_pms, 'Mileage intervals NPMS'),
                (npmsrevenue, 'NPMS revenue'),
                (revenu, 'PMS/NPMS revenue'),
                (vin_branch_pivot, 'Branch visits'),
                (avg_monthly_interval1, 'Monthly intervals'),
                (freq_npms, 'NPMS frequency'),
                (branch_grouped, 'Branch diversity'),
                (lastnonpmsmil, 'Last Non-PMS mileage'),
                # (fnlappnt, 'Service Appointment'),
                # (fnlappoinoshow, 'No-show Appointments'),
                (complaint_features, 'Complaint features'),
                (milestone_intervals_df, 'Milestone intervals')
            ]

            
            for df, description in merge_operations:
                print(f"Merging {description}: {df['Vin_No'].is_unique}")

                logger.info(f"Merging {description}")
                before_shape = newserv20ka.shape
                print(f'Merging {description}, Type: {type(df)}')
                newserv20ka = pd.merge(newserv20ka, df, on=['Vin_No'], how='left')
                after_shape = newserv20ka.shape
                logger.debug(f"Shape after merging {description}: {after_shape}")
                if before_shape[0] != after_shape[0]:
                    logger.warning(f"Row count changed after merging {description}")

            # Fill missing values
            logger.info("Handling missing values")
            fill_columns = ['nNPMS', 'npmsRevenue', 'freq_NPMS', 'unique_branch_serviced','otherbranch_services','SinglePMS',
                            'has_complaint_history','num_past_complaints','avg_complaint_resolution_days','max_complaint_resolution_days',
                            'recent_complaint_resolution_days'] + top_branches
            newserv20ka[fill_columns] = newserv20ka[fill_columns].fillna(0)
            newserv20ka[revenu.columns[1:]] = newserv20ka[revenu.columns[1:]].fillna(0)
            # Final processing
            logger.info("Performing final data transformations")
            newserv20ka = newserv20ka.drop(['Total_Revenue','PMS_Service'], axis=1,errors='ignore')
            newserv20ka = newserv20ka.rename(columns={'Vin_No': 'VIN'})
            
            # Merge with master train data
            logger.info("Merging with master train data")
            filtered_dfnew = pd.merge(mastertrain, newserv20ka, on=['VIN'], how='left')
            filtered_dfnew[fill_columns] = filtered_dfnew[fill_columns].fillna(0)
            filtered_dfnew['latest_complaint_category'] = filtered_dfnew['latest_complaint_category'].fillna('unknown') 
            # Save output
            # output_path = 'validatecode/finalmerged40k.csv'
            # logger.info(f"Saving final output to {output_path}")
            filtered_dfnew = filtered_dfnew[filtered_dfnew['VIN'].isin(mastertrain['VIN'].unique())]
            filtered_dfnew[filtered_dfnew["Service_Num_x"] == filtered_dfnew["Service_Num_y"]]
            filtered_dfnew = filtered_dfnew.drop_duplicates(subset=['Vehicle Key'])
            filtered_dfnew = filtered_dfnew.drop(['Service_Num_y','Branch_List'], axis=1)
            filtered_dfnew = filtered_dfnew.rename(columns={'Service_Num_x': 'Service_Num'})
            
            pmsvins = mastertrain.query("TargetFlag == 1")['VIN'].unique()
            service_histories = serv[serv['Vin_No'].isin(pmsvins)].to_dict(orient="records")
            result = find_previous_service_for_all_vins(service_histories, target_service_num=last_service_code)
            cleaned = {vin: rec for vin, rec in result.items() if rec is not None}
            unclean = {vin: rec for vin, rec in result.items() if rec is None}
            if len(cleaned) == 0:
                finlmlg = pd.DataFrame(columns=['Vin_No', 'Mileage'])
            else:
                finlmlg = pd.DataFrame.from_dict(cleaned, orient='index').reset_index(drop=True)
                finlmlg = finlmlg[['Vin_No', 'Mileage']]
            pmsfsttime = (pd.DataFrame.from_dict(unclean, orient="index", columns=["Mileage"])
            .reset_index()
            .rename(columns={"index": "Vin_No"}))
            pmsfsttime['Mileage'] = np.nan  # No previous PMS mileage exists — leave as NaN, do not impute with population mean
            lstmiltg = pd.concat([finlmlg, pmsfsttime], axis=0)
            lstmiltg = lstmiltg.rename(columns={'Mileage':'MileagePMS','Vin_No':'VIN'})
            # lstmiltg.to_csv('validatecode/lastpmstimeservicemileage40k.csv', index=False)
            filtered_dfnew.loc[filtered_dfnew["TargetFlag"] == 1, "Last PMS Mileage"] = pd.NA
            filtered_dfnew = filtered_dfnew.merge(lstmiltg,on='VIN',how='left')
            filtered_dfnew["Last PMS Mileage"] = filtered_dfnew["Last PMS Mileage"] \
                                      .fillna(filtered_dfnew["MileagePMS"])
            filtered_dfnew["LastNonPMSMileage"] = filtered_dfnew["LastNonPMSMileage"] \
                                     .fillna(filtered_dfnew["Last PMS Mileage"])
            # filtered_dfnew.to_csv(f'validatecode/checkdf{last_service_code}kpred.csv', index=False)
            # serv.to_csv(f'validatecode/servfiltered{last_service_code}kpred.csv', index=False)
            # servM.to_csv(f'validatecode/servMfiltered{last_service_code}kpred.csv', index=False)
            # filtered_dfnew = map_vhc_history(filtered_dfnew, serv) 
            # vhcfill = ['Survey Score','VHC Quoted','VHC Sold','VHC Lost Sale','VHC Lost Red Sale','VHC Deferred','VHC Amber Deferred','VHC Completed_Flag']
            # filtered_dfnew[vhcfill] = filtered_dfnew[vhcfill].fillna(0)
            # filtered_dfnew = backfill_vhc_for_target_rows(
            #         merged_df=filtered_dfnew,
            #         history_df=servM.query("Service_Num > 0"),
            #         target_service_num=last_service_code,
            #         vhc_cols=["VHC Quoted", "VHC Sold", "VHC Lost Sale","VHC Lost Red Sale", "VHC Deferred", "VHC Amber Deferred"],
            #         survey_cols=["Survey Score", "Survey Status", "VHC Completed_Flag"]
            #     )
            # filtered_dfnew = filtered_dfnew.drop(vhcfill, axis=1, errors='ignore')
            vhcdf = pd.read_csv(vhc,low_memory=False)
            vhcdata = vhcpreparation(vhcdf,last_service_code,filtered_dfnew)
            filtered_dfnew = filtered_dfnew.merge(vhcdata,on=['Vehicle Key','Service_Num'],how='left')
            filtered_dfnew = backfill_vhc_leakage_safe(
            filtered_dfnew,
            vhcdata,
            vhcdata.columns[2:],
            vin_col="Vehicle Key",
            service_col="Service_Num",
            targetflag_col="TargetFlag"
            )
            filtered_dfnew = filtered_dfnew.fillna(0)
            # filtered_dfnew.to_csv(f'validatecode/checkdfwithvhc{last_service_code}kpred.csv', index=False)
            
            appointdfN = pd.read_csv(appointdf, low_memory=False) #Input appointment full data
            appointdfN['Due Date IN'] = pd.to_datetime(appointdfN['Due Date IN'],format='mixed',dayfirst=True,errors='coerce')

            ltappoins = compute_late_appointment_metrics(appointdfN, servM,filter_date=filterdate)
            appoinstat = compute_last_appointment_status_with_constant_service_code(appointdfN,serv,last_service_code,filter_date=filterdate)
            fnlappnt = derive_appointment_show_features(appointdfN,servM,last_service_code,filter_date=filterdate)
            filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Key"].str.split("-", n=1).str[1]
            filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Magic"].str.split("-", n=1).str[0]
            filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Magic"].astype(int)
            
            filtered_dfnew = filtered_dfnew.merge(appoinstat, on='Vehicle Magic',how='left').merge(ltappoins,on='Vehicle Magic',how='left').merge(fnlappnt,on='Vehicle Magic',how='left')
            nservappbk = count_service_appointments_booked(
                        appointdfN,
                        filtered_dfnew,
                        last_service_code,
                        filterdate,
                        vehicle_col="Vehicle Magic",
                        booking_date_col="WIP Booking Date"
                    )
            filtered_dfnew = filtered_dfnew.merge(nservappbk,on='Vehicle Magic',how='left')
            filtered_dfnew['no_of_service_appointments_booked'] = filtered_dfnew['no_of_service_appointments_booked'].fillna(0)
            
            filtered_dfnew = filtered_dfnew.drop(['Vehicle Magic'],axis=1)

            filtered_dfnew[['last_appointment_delay_days', 'no_of_late_appointments','appointment_booked_showed_up_atleastonce','total_appointments_showed_up','no_of_appointments_booked_but_not_showed_up','Avg_appointment_delay_days']] = filtered_dfnew[['last_appointment_delay_days', 'no_of_late_appointments','appointment_booked_showed_up_atleastonce','total_appointments_showed_up','no_of_appointments_booked_but_not_showed_up','Avg_appointment_delay_days']].fillna(0)
            filtered_dfnew['last_appointment_status'] = filtered_dfnew['last_appointment_status'].fillna('No Appointment')
            
            logger.info(f"Processing completed. Final shape: {filtered_dfnew.shape}")
      
            # filtered_dfnew = pd.read_csv('validatecode/finalmerged40k.csv',low_memory=False)
            rfmdf = pd.read_csv(rfm,low_memory=False) #Input RFM segments file
            if 'Customer ID' not in rfmdf.columns and 'Contact Key' in rfmdf.columns:
                rfmdf = rfmdf.rename(columns={'Contact Key': 'Customer ID'})
            filtered_dfnew = filtered_dfnew.merge(rfmdf[['Customer ID','RFM_segments']],on=['Customer ID'],how='left')
            filtered_dfnew['Number of Cylinders'] = filtered_dfnew['Number of Cylinders'].astype('object')
            filtered_dfnew = filtered_dfnew.rename(columns={f'Expected{last_service_code}kDate':f'Next{last_service_code}K_Due'})
            rem = ['New / Used Category','Current Customer','First Service Date',
                'Last Service Date', 'Next Service Date','Vehicle Lifetime in Years','MileagePMS',
                'Last Service - PMS','Sale Invoice Year','Invoice date','Target Revenue','Potential Revenue',
                  'Final Revenue', 'Vehicle Age']
            filtered_dfnew= filtered_dfnew.drop(rem,axis=1,errors='ignore')
            
            # Compute PMS_Delay accurately
            filtered_dfnew['PMS_Delay'] = compute_pms_delay(filtered_dfnew, last_service_code)
            
            pms = filtered_dfnew
            logger.info("Detecting columns with '-' placeholders (pass 1)")
            obj_cols = pms.select_dtypes(include='object').columns
            hifunfeats = [c for c in obj_cols if (pms[c] == '-').any()]
            missfeats = [c for c in pms.columns if c not in hifunfeats and pms[c].isnull().any()]
            logger.info(f"Pass 1: {len(hifunfeats)} cols with '-', {len(missfeats)} cols with nulls")
            cond1 = pms['Brake Purchase Interval'] > 0
            cond2 = pms['Tyre Purchase Interval'] > 0
            cond3 = pms['Battery Purchase Interval'] > 0
            pms['wearablesBought'] = np.where(cond1 | cond2 | cond3,1,0)
            pms = calculate_revenue_spend(pms,last_service_code)
            # filtered_dfnew.to_csv(output_path, index=False)
            missfeatsdrop = ['10K','20K','30K','40K','50K','60K','70K','80K','90K','100K','110K',
                '10K R','20K R','30K R','40K R','50K R','60K R','70K R','80K R','90K R','100K R','110K R',
                '10K SC','20K SC','30K SC','40K SC','50K SC','60K SC','70K SC','80K SC','90K SC','100K SC',
                '110K SC', '120K', '120K R', '120K SC', '130K', '130K R', '130K SC', '140K', '140K R','140K SC',
                '150K', '150K R', '150K SC', '160K', '160K R', '160K SC', '170K', '170K R', '170K SC',
                '180K', '180K R', '180K SC', '190K', '190K R', '190K SC', '200K', '200K R', '200K SC',
                '200K+', '200K+   R', '200K+   SC','Brake Purchase Interval','Tyre Purchase Interval',
                'Battery Purchase Interval','70K R.1']
            pms = pms.drop(missfeatsdrop,axis=1,errors = 'ignore')

            logger.info("Detecting columns with '-' placeholders (pass 2)")
            obj_cols2 = pms.select_dtypes(include='object').columns
            hifunfeats1 = [c for c in obj_cols2 if (pms[c] == '-').any()]
            missfeats1 = [c for c in pms.columns if c not in hifunfeats1 and pms[c].isnull().any()]
            logger.info(f"Pass 2: {len(hifunfeats1)} cols with '-', {len(missfeats1)} cols with nulls")
            
            for i in hifunfeats1:
                pms[i] = pms[i].replace('-', pd.NA)

            # Convert '-'-cleaned columns to numeric ONLY if their non-null
            # values are actually numeric (preserves categoricals like
            # New/Used, Gender, Warranty Status that also had '-' placeholders)
            converted_count = 0
            for col in hifunfeats1:
                test = pd.to_numeric(pms[col], errors='coerce')
                non_null_orig = pms[col].notna().sum()
                if non_null_orig > 0 and test.notna().sum() / non_null_orig > 0.5:
                    pms[col] = test
                    converted_count += 1
                else:
                    logger.info(f"  Kept '{col}' as categorical (not numeric)")
            logger.info(f"Converted {converted_count}/{len(hifunfeats1)} columns to numeric after '-' cleanup")

            print(f'Dtype of Vehicle_Key_ExpectedServices: {pms["Vehicle_Key_ExpectedServices"].dtype}')
            catfeats = ['New / Used','Gender','Warranty Status', 'Number of Cylinders']
            for i in ['Nationality','Model','Variant','RFM_segments']:
                pms[i] = pms[i].fillna('Unknown')
            nationalitymap = pms[['Vehicle Key','Nationality']]
            vehmodelmap = pms[['Vehicle Key','Model']]
            vehvariantmap = pms[['Vehicle Key','Variant']]
            rfmpms= pms[['Vehicle Key','RFM_segments']]


            obint = ['Total Promoter','Total Passive','Total Detractor','Total Survey',
            'CC','Weight','Height','Wheel Base','Service Frequency']
            for i in obint:
                pms[i] = pd.to_numeric(pms[i], errors='coerce')
            pms = pms.drop('Service_Date',axis=1,errors = 'ignore')
            pms[['Months_Since_Last_NPMS','Avg_Mileage_Interval_NPMS']] = pms[['Months_Since_Last_NPMS','Avg_Mileage_Interval_NPMS']].fillna(0)
            colssrvd = pms.pop('Last Service Date - PMS')
            pms.insert(3, colssrvd.name, colssrvd)
            pms['Last Service Date - PMS'] = pd.to_datetime(pms['Last Service Date - PMS'],format='mixed',dayfirst=True,errors='coerce')
            pms[f'Next{last_service_code}K_Due'] = pd.to_datetime(pms[f'Next{last_service_code}K_Due'],format='mixed',errors='coerce')
            colsdue = pms.pop(f'Next{last_service_code}K_Due')
            pms.insert(4, colsdue.name, colsdue)
            # Drop high-cardinality columns before encoding (already saved for clustering)
            pms = pms.drop(['Nationality','Model','Variant','RFM_segments'],axis=1,errors='ignore')
            
            ID_COLS = {'VIN', 'Vehicle Key', 'Customer ID'}
            MAX_OHE_CARDINALITY = 50  # safety cap

            # 1) Try numeric conversion on every non-ID object column
            for col in pms.select_dtypes(include='object').columns:
                if col in ID_COLS:
                    continue
                converted = pd.to_numeric(pms[col], errors='coerce')
                if converted.notna().mean() > 0.5:  # >50% parseable → numeric
                    pms[col] = converted
                    logger.info(f"  Coerced '{col}' to numeric ({converted.notna().mean():.0%} parseable)")

            # 2) Identify true categoricals (non-ID, low-cardinality)
            obj_cols_remaining = [
                c for c in pms.select_dtypes(include='object').columns
                if c not in ID_COLS
            ]
            cols_to_encode = [c for c in obj_cols_remaining if pms[c].nunique() <= MAX_OHE_CARDINALITY]
            cols_too_high  = [c for c in obj_cols_remaining if pms[c].nunique() > MAX_OHE_CARDINALITY]

            for c in cols_to_encode:
                logger.info(f"  OHE: {c} ({pms[c].nunique()} unique values)")
            if cols_too_high:
                logger.warning(f"Dropping high-cardinality object columns (>{MAX_OHE_CARDINALITY} unique): {cols_too_high}")
                pms = pms.drop(columns=cols_too_high)

            # 3) Encode
            logger.info(f"One-hot encoding {len(cols_to_encode)} columns")
            pmsnew = pd.get_dummies(pms, columns=cols_to_encode, dtype=int)
            pmsnew = pmsnew.reset_index(drop=True)
            logger.info(f'One-hot encoding complete. Shape: {pmsnew.shape}')
            
            
            pmsnew["LowMileageFreqUsers"] = np.where(
             (pmsnew["Avg_Service_Interval_PMS"].between(0, 7)) & (pmsnew["Last Service Mileage"].between(0, (last_service_code-10)*1000)) & (pmsnew["TargetFlag"]==1),1,0)
            uy = ['Total Promoter','Total Passive','Total Detractor','Total Survey']
            for i in uy:
                pmsnew[i] = pmsnew[i].fillna(0)
            cols = pmsnew.pop('Last Service Date - PMS')
            pmsnew.insert(1, cols.name, cols)
            pmsnew = pmsnew.drop('70K R.1',axis=1,errors='ignore')
            # pmsnew.to_csv(f'validatecode/finalmerged{last_service_code}kv1.csv', index=False)
            # Fill all-NaN columns with 0 before imputing (IterativeImputer
            # silently drops them, causing a shape mismatch on reconstruction)
            numeric_block = pmsnew.iloc[:, 5:]
            all_nan_cols = numeric_block.columns[numeric_block.isnull().all()]
            if len(all_nan_cols) > 0:
                logger.warning(f"Filling {len(all_nan_cols)} all-NaN columns with 0 before imputation: {list(all_nan_cols)}")
                pmsnew[all_nan_cols] = 0
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            # Mileage columns must NOT be imputed — missing mileage means no history,
            # not an unknown value that can be predicted from categorical/OHE features.
            MILEAGE_COLS = [c for c in pmsnew.columns[5:] if c in ('Last PMS Mileage', 'LastNonPMSMileage')]
            # Only impute numeric columns
            numeric_candidates = pmsnew.iloc[:, 5:].select_dtypes(include=[np.number]).columns.tolist()
            cols_to_impute = [c for c in numeric_candidates if c not in MILEAGE_COLS and c != 'TargetFlag']
            numeric_cols_block = pmsnew[cols_to_impute]
            # Standardize before imputation: Lasso diverges when features span
            # very different scales (e.g., 0/1 OHE flags vs 100k+ revenue values).
            scaler = StandardScaler()
            scaled_block = scaler.fit_transform(numeric_cols_block.fillna(0))
            scaled_df = pd.DataFrame(scaled_block, columns=cols_to_impute)
            # Restore NaNs for the imputer to fill
            scaled_df[numeric_cols_block.isnull()] = np.nan
            imputer = IterativeImputer(
                random_state=0,
                estimator=Lasso(max_iter=10000),  # converges once features are scaled
                max_iter=10,                       # multiple rounds so values stabilise
                tol=1e-3,
            )
            imputed_scaled = imputer.fit_transform(scaled_df)
            # Invert standardisation to recover original feature units
            imputed_block = pd.DataFrame(
                scaler.inverse_transform(imputed_scaled), columns=cols_to_impute
            )
            # Re-attach mileage columns with their original (un-imputed) values
            mileage_block = pmsnew[MILEAGE_COLS].reset_index(drop=True)
            header_cols = pmsnew[['VIN', 'Last Service Date - PMS', 'Vehicle Key', 'Customer ID', f'Next{last_service_code}K_Due', 'TargetFlag']].reset_index(drop=True)
            pms_imputed = pd.concat([header_cols, imputed_block, mileage_block], axis=1)
            bs = serv1[serv1['Description']=='Others']
            bodyshop_counts = calculate_bodyshop_count(bs, branch_col='Service_Branch_Name', vin_col='Vin_No', keyword='Bodyshop')
            pms_imputed = pms_imputed.merge(bodyshop_counts,on='VIN',how='left')
            pms_imputed['Bodyshop_Services'] = pms_imputed['Bodyshop_Services'].fillna(0).astype(int)
            # pms_imputed.to_csv('validatecode/finalmerged40kv2.csv', index=False)
            ##Nationality
            mergedfnat = nationalitymap.merge(vehmodelmap,on='Vehicle Key',how='left')
            mergedfnat = mergedfnat.merge(rfmpms,on='Vehicle Key',how='left')
            pms_imputed, clustersnat = hierarchical_gower_clustering(pms_imputed,mergedfnat,
                                                            feat="Nationality", k_min=2, k_max=10)
            
            ##Model
            mergedfmod = vehmodelmap.merge(rfmpms,on='Vehicle Key',how='left')
            pms_imputed, clustersmod = hierarchical_gower_clustering(pms_imputed,mergedfmod,
                                                feat="Model", k_min=2, k_max=10)
            
            ##Variant
            mergedfvar = vehvariantmap.merge(rfmpms,on='Vehicle Key',how='left')
            pms_imputed, clustersvar = hierarchical_gower_clustering(pms_imputed,mergedfvar,
                                                feat="Variant", k_min=2, k_max=10)
            pms_imputed = pms_imputed.drop(['Mileage','Brake Points','Vehicle_Key_ExpectedServices',
                                            'Tyre Points','Battery Points'],axis=1,errors='ignore')
            
            clustersnat.to_csv(f'validatecode/Nationality_clusters_{last_service_code}.csv', index=False)
            clustersmod.to_csv(f'validatecode/Model_clusters_{last_service_code}.csv', index=False)
            clustersvar.to_csv(f'validatecode/Variant_clusters_{last_service_code}.csv', index=False)
            sessdf = pd.read_csv(digidf,low_memory=False)
            pms_imputed = pms_imputed.merge(sessdf,on=['Vehicle Key','Customer ID'],how='left')
            pms_imputed.iloc[:,-4:] = pms_imputed.iloc[:,-4:].fillna(0)
            pms_imputed['Vehicles Owned'] = pms_imputed['Vehicles Owned'].fillna(1)
            # Defragment before column reordering to avoid PerformanceWarning
            pms_imputed = pms_imputed.copy()
            # Move TargetFlag to last column and Last PMS Mileage to second-to-last
            cols = [c for c in pms_imputed.columns if c not in ("TargetFlag", "Last PMS Mileage")]
            cols = cols + ["Last PMS Mileage", "TargetFlag"]
            pms_imputed = pms_imputed[cols]
            pms_imputed = pms_imputed.rename(columns=lambda c: re.sub(r'\.0$', '', c))
            
            # --- New Logic: Intelligent Mileage Imputation ---
            previous_milestones = [10 * i for i in range(1, (last_service_code // 10))]
            
            def get_theoretical_mileage(row):
                for mil in sorted(previous_milestones, reverse=True):
                    if row.get(f'has_{mil}', 0) == 1:
                        return mil * 1000
                return 1000 # Proxy delivery mileage if NO previous milestones completed
                
            if len(previous_milestones) > 0:
                theoretical_mileage = pms_imputed.apply(get_theoretical_mileage, axis=1)
                pms_imputed['Last PMS Mileage'] = pms_imputed['Last PMS Mileage'].fillna(theoretical_mileage)
            
            # Only consider mileages >= 1000km to filter out early PDI/warranty registrations.
            # Vehicles with no mileage >= 1000 will naturally get NaN here.
            valid_mileages = pms_imputed[["Last PMS Mileage", "LastNonPMSMileage"]].where(
                pms_imputed[["Last PMS Mileage", "LastNonPMSMileage"]] >= 1000
            )
            pms_imputed["Last Service Mileage"] = valid_mileages.max(axis=1)
            
            # Final fallback: if Last Service Mileage is still NaN, use theoretical mileage
            if len(previous_milestones) > 0:
                pms_imputed["Last Service Mileage"] = pms_imputed["Last Service Mileage"].fillna(theoretical_mileage)
            else:
                pms_imputed["Last Service Mileage"] = pms_imputed["Last Service Mileage"].fillna(1000)
            os.makedirs(f'models/{last_service_code}k', exist_ok=True)
            pms_imputed.to_csv(f'models/{last_service_code}k/training_features.csv', index=False)
            pms_imputed.to_csv(f'validatecode/finalmerged{last_service_code}kQ3.csv', index=False)  # legacy compat
            # pms_imputed.to_csv('validatetrainnomseventrev.csv', index=False)
            return pms_imputed
        
        except Exception as e:
            logger.error(f"Error during feature merging: {str(e)}")
            raise
                
    except Exception as e:
        logger.error(f"Error in process_service_data: {str(e)}")
        raise


def feature_sel(df: pd.DataFrame, cumulative_threshold: float = 0.85):
    """
    Selects features using Mutual Information scores against TargetFlag.
    
    1. Hard-drops known identifier/leaky columns.
    2. Removes noisy OHE artifacts (Other, Unknown, etc.).
    3. Keeps features contributing to the top `cumulative_threshold` (default 85%)
       of total MI signal.
    
    Returns: list of selected feature column names.
    """
    # --- Hard-drop columns that should never enter the model ---
    hard_drop = [
        "Vehicle Key", 
    ]
    
    # Identifier columns that sit at the front of the DataFrame
    id_cols = ["Vehicle Key", "VIN", "Customer ID", "Franchise", "Model", "Variant", "Nationality"]
    
    # Target and special columns at the end
    exclude_cols = set(id_cols + ["TargetFlag", "Last PMS Mileage"] + hard_drop)
    
    # Select only feature columns by name (not position)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    
    # Drop noisy OHE artifact columns
    noise_pattern = r'Other|Service_Num|OTHERS|old|OTHER|Unknown|segments_-'
    X = X.loc[:, ~X.columns.str.contains(noise_pattern, case=False, na=False)]
    
    # Drop noisy OHE artifact patterns (matching old code's feature_sel regex)
    # Vehicle_Key_ExpectedServices is already dropped in process_service_data
    leaky_patterns = r'DeferredPart'
    X = X.loc[:, ~X.columns.str.contains(leaky_patterns, case=False, na=False)]
    
    # CRITICAL: Mutual information only works on numeric data. Drop any remaining date/string columns.
    X = X.select_dtypes(include=[np.number])
    
    y = df["TargetFlag"].astype(int)
    
    # Compute Mutual Information scores
    # sklearn MI algorithm cannot handle NaN values (e.g. from missing mileages).
    # Fill them with 0 just for the scoring step.
    X_filled = X.fillna(0)
    mi_scores = mutual_info_classif(X_filled, y, random_state=42)
    
    mi_df = pd.DataFrame({
        "Feature": X.columns,
        "MIScore": mi_scores
    }).sort_values(by="MIScore", ascending=False)
    
    # Keep features contributing to top 85% of cumulative MI
    mi_df["Cumulative_%"] = mi_df["MIScore"].cumsum() / mi_df["MIScore"].sum()
    mi_df_selected = mi_df.query("`Cumulative_%` <= @cumulative_threshold")
    
    logger.info(f"Feature selection: {len(mi_df_selected)} / {len(mi_df)} features kept "
                f"(top {cumulative_threshold*100:.0f}% MI)")
    
    return list(mi_df_selected["Feature"].values), mi_df


if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys
    
    parser = argparse.ArgumentParser(description="Run PMS Training Feature Engineering")
    parser.add_argument("year", type=int, help="Prediction Year (e.g., 2026)")
    parser.add_argument("quarter", type=str, help="Prediction Quarter (e.g., Q1, Q2, Q3, Q4)")
    parser.add_argument("milestone", type=str, help="Service milestone (e.g., 20, 20k, 30)")
    parser.add_argument("--train-cutoff", type=str, help="Optional: YYYY-MM-DD date to filter service history (e.g., 2025-12-31). Defaults to the end of the previous quarter.", default=None)
    args = parser.parse_args()
    
    milestone_val = int(args.milestone.lower().replace('k', ''))
    
    print(f"--- Running Training Data Preparation for {milestone_val}k ---")
    
    # Files
    eda_path = os.path.join("data", "EDA - Q3 2025.csv")
    service_history_path = os.path.join("data", "Service History Q1 - 2026.csv")
    rfm_path = "data/RFM_Q2-2026.csv"
    appointdf_path = "data/Appointments_Q2-2026.csv"
    digidf_path = "data/Digital_Sessions_Q2-2026.csv"
    vhc_path = "data/VHC_Q2-2026.csv"
    servcode_path = "data/Service Code Desc.csv"
    
    if not os.path.exists(eda_path):
        print(f"Error: Could not find {eda_path}")
        sys.exit(1)
    if not os.path.exists(service_history_path):
        print(f"Error: Could not find {service_history_path}")
        sys.exit(1)
    
    # 1. Prepare PMS Datasets (old-style target labelling)
    final_df = prepare_pms_datasets(
        df_path=eda_path,
        service_history_path=service_history_path,
        service_types=[f"{milestone_val}k"],
        selected_quarter=args.quarter,
        year=args.year,
        target_range=(0.35, 0.45),
        output_prefix="TestRefactored",
    )
    
    # Deduplicate: keep highest service milestone per VIN (matches legacy main logic)
    final_df = (
        final_df
        .sort_values(["VIN", "Service_Num"], ascending=[True, False])
        .drop_duplicates(subset="VIN", keep="first")
        .reset_index(drop=True)
    )
    print(f"After VIN deduplication: {final_df.shape[0]} rows")
    
    # Check if a custom train cutoff was provided; otherwise use quarter_end calculated inside prepare_pms_datasets
    if args.train_cutoff:
        filter_date = args.train_cutoff
    else:
        # Get the same quarter_end date that prepare_pms_datasets calculated internally
        _, quarter_end = get_quarter_dates(args.quarter, args.year)
        filter_date = quarter_end.strftime("%Y-%m-%d")
        
    print(f"Using filter_date (training cutoff): {filter_date}")
    
    print("starting processing of the datasets")
    final_df_after_process = process_service_data(
        mastersheet=final_df,  # Passing the DataFrame directly in memory
        servhistory=service_history_path,
        rfm=rfm_path,
        servcode=servcode_path,
        appointdf=appointdf_path,
        digidf=digidf_path,
        vhc=vhc_path,
        filter_date=filter_date,
        last_service_code=milestone_val,
    )
    
    # Save the huge raw processed file for debugging/checks
    raw_processed_path = rf"refactored_test_dir\final_processed_{milestone_val}k.csv"
    final_df_after_process.to_csv(raw_processed_path, index=False)
    
    # 3. Feature Selection
    print("\n--- Running Feature Selection ---")
    selected_features, mi_scores_df = feature_sel(final_df_after_process, cumulative_threshold=0.85)
    
    # We always keep 'TargetFlag' and 'VIN' for training
    final_cols = ["VIN", "TargetFlag"] + selected_features
    train_master = final_df_after_process[final_cols]
    
    print(f"Selected {len(selected_features)} features (from {final_df_after_process.shape[1]} total columns)")
    print(f"Final training shape: {train_master.shape}")
    
    # Setup directories
    model_dir = f"models/{milestone_val}k"
    training_dir = f"{model_dir}/training"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    
    # Save Final Training Dataset
    train_path = f"{training_dir}/training_features.csv"
    train_master.to_csv(train_path, index=False)
    print(f"Saved training set to {train_path}")
    
    # Save MI scores for inspection
    mi_scores_path = f"{training_dir}/mi_scores.csv"
    mi_scores_df.to_csv(mi_scores_path, index=False)
    print(f"Saved MI scores to {mi_scores_path}")
    
    # Save the selected features to a JSON file so the Test Set script can use them
    feature_list_path = f"{model_dir}/selected_features.json"
    with open(feature_list_path, 'w') as f:
        json.dump(selected_features, f)
    print(f"Saved selected feature list to {feature_list_path}")