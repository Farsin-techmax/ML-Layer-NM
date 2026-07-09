import re
import os
import sys
try:
    from duckdb import df
except Exception:
    # DuckDB is optional in this environment. If not installed, continue without it.
    df = None
try:
    import gower
except Exception:
    # Lightweight fallback for gower.gower_matrix when package is not installed.
    class _GowerFallback:
        @staticmethod
        def gower_matrix(df):
            import numpy as _np
            import pandas as _pd

            X = df.copy()
            num_cols = X.select_dtypes(include=[_np.number]).columns.tolist()
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            n = X.shape[0]
            D = _np.zeros((n, n))

            if len(num_cols) > 0:
                num = X[num_cols].astype(float)
                ranges = num.max() - num.min()
                ranges = ranges.replace(0, 1)
                num_scaled = (num - num.min()) / ranges
                num_arr = num_scaled.values
            else:
                num_arr = None

            for i in range(n):
                for j in range(i+1, n):
                    s = 0.0
                    cnt = 0
                    if num_arr is not None:
                        s += _np.nansum(_np.abs(num_arr[i] - num_arr[j]))
                        cnt += num_arr.shape[1]
                    if len(cat_cols) > 0:
                        a = X.iloc[i][cat_cols].astype(str).values
                        b = X.iloc[j][cat_cols].astype(str).values
                        s += _np.sum(a != b)
                        cnt += len(cat_cols)
                    D[i, j] = s / max(cnt, 1)
                    D[j, i] = D[i, j]
            return D

    gower = _GowerFallback()
import logging
import warnings
import traceback
import numpy as np
import pandas as pd
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



######################### Historical Data Sampling #########################

def get_quarter_dates(selected_quarter, year):
    
    """
    Returns start_date and quarter_end based on client-selected quarter and year.
    quarter_end = last day of the previous quarter.
    start_date = 1 Jan of the previous year.
    """
    if selected_quarter not in ["Q1", "Q2", "Q3", "Q4"]:
        raise ValueError("Invalid quarter selected. Choose from Q1, Q2, Q3, Q4.")

    quarter_months = {"Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)}
    start_month, end_month = quarter_months[selected_quarter]
    print(f"Selected Quarter: {selected_quarter} ({start_month}-{end_month}) of Year: {year}")
    quarter_start = pd.Timestamp(year=year, month=start_month, day=1)
    previous_quarter_end = quarter_start - pd.Timedelta(days=1)

    # start date = Jan 1 of previous year
    start_date = pd.Timestamp(year=year - 1, month=1, day=1)
    print(f"Quarter End Date: {previous_quarter_end}, Start Date: {start_date}")
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
    start_date_dt, quarter_end_dt = get_quarter_dates(selected_quarter, year)
    print(f"Processing for Quarter End: {quarter_end_dt.date()}")
    # Load main dataframe
    df = pd.read_csv(df_path, low_memory=False, encoding="ISO-8859-1")
    df = df.query("`Last Service - PMS` != '-'")
    df["Last Service Mileage"] = pd.to_numeric(df["Last Service Mileage"], errors="coerce")
    df["Last PMS Mileage"] = pd.to_numeric(df["Last PMS Mileage"], errors="coerce")

    for col in ["Invoice date", "First Service Date", "Last Service Date - PMS","Last Service Date","Next Service Date"]:
        df[col] = df[col].replace("-", pd.NA)
        df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=True, errors="coerce")
    print(f"Main DataFrame Shape: {df.shape[0]}")
    df["Service_Num"] = df["Last Service - PMS"].apply(extract_k1)
    df["FirstSrvDate"] = df["Invoice date"].fillna(df["First Service Date"])


    #deriving feature from service history
    
    # Load service history
    serv = pd.read_csv(service_history_path, low_memory=False,encoding='ISO-8859-1')
    serv["Service_Num"] = serv["Description"].apply(extract_kk)
    serv["Service_Date"] = pd.to_datetime(serv["Service_Date"], format="mixed", dayfirst=True, errors="coerce")
    start_date = pd.to_datetime("2018-01-01")
    for svc in service_types:
        print(f"\n Processing {svc} ...")
        svc_num = extract_k1(svc)
        # Adjustable start date
        start_date_adj = start_date_dt
        max_iterations =  (quarter_end_dt.year - start_date.year) * 12 + (quarter_end_dt.month - start_date.month)
        iteration = 0
        finaltr = pd.DataFrame()
        print(f'PMS number: {svc_num}')
        while iteration < max_iterations:
            # VINs already done target PMS
            #this line filter out all the vins that has done a 20k, which makes the 20k absent vehicles left in the 
            #frame df_target, all but 20k done
            df_target = df[~df["VIN"].isin(df.query(f"`Last Service - PMS` == '{svc}'")["VIN"].unique())]
            print(f'DFTarget:  {len(df_target)}')
            # Skipped VINs
            #to find the vins that has a higher service done while skipping 20k
            dfskipped = df_target.query(f"Service_Num > {svc_num-10}")["VIN"].unique()
            #checking which all services they have done expect 20k/skipping 20k
            serv_filtered = serv[serv["Vin_No"].isin(dfskipped)]
            #grouping them by the vin and the services they did
            history_grouped = serv_filtered.groupby("Vin_No")["Service_Num"].apply(list).reset_index(name="Completed_Services")
            filtered_df = history_grouped[~history_grouped["Completed_Services"].apply(lambda x: svc_num in x)]
            print(f'Filtered DF: {len(filtered_df)}')
            # NotTurnUp VINs
            NotTurnUpdf = df[df["VIN"].isin(filtered_df["Vin_No"].unique())].query(f"Service_Num == {svc_num + 10}")

            if not NotTurnUpdf.empty:
                NotTurnUpdf["VIN_base"] = NotTurnUpdf["Vehicle Key"].str.rsplit("-", n=1).str[0]
                NotTurnUpdf["Owner_No"] = NotTurnUpdf["Vehicle Key"].str.extract(r"-(\d+)$")[0].fillna(0).astype(int)
                NotTurnUpdf = NotTurnUpdf.loc[
                    NotTurnUpdf.groupby("VIN_base")["Owner_No"].idxmax()
                ].reset_index(drop=True)
                NotTurnUpdf = NotTurnUpdf.sort_values(by="Service_Num", ascending=False).drop_duplicates(subset="VIN", keep="first")
                NotTurnUpdf = NotTurnUpdf.drop(["VIN_base", "Owner_No"], axis=1)
            # Pending VINs

            print(f'{svc} - Iteration {iteration+1}: Start Date = {start_date_adj.date()}')
            #this line take the master dataframe and keeps all but vins that completed a 30 without a 20
            missed = df[~df["VIN"].isin(NotTurnUpdf["VIN"].unique())]
            #this line removes all the rows that indeed completed 20k
            missed = missed[~missed["VIN"].isin(df.query(f"`Last Service - PMS` == '{svc}'")["VIN"].unique())]
            missed["FirstSrvDate"] = missed["Invoice date"].fillna(missed["First Service Date"])
            missed["Service_Num"] = missed["Last Service - PMS"].apply(extract_k1)
            months_to_add = int(svc_num / 10) * 6
            missed[f"Expected{svc}Date"] = missed["FirstSrvDate"].apply(lambda d: d + relativedelta(months=months_to_add) if pd.notnull(d) else pd.NaT)
            missed = missed[missed["Vehicle Service Status"].isin(["InActive", "Lapsed","Active"])]
            pending = missed.query(f"Service_Num < {svc_num}")
            pending = pending[(pending[f"Expected{svc}Date"] >= start_date_adj) & (pending[f"Expected{svc}Date"] <= quarter_end_dt)]
            dfafter = df.query(f"Service_Num >= {svc_num}")["VIN"].unique()
            pending = pending[~pending["VIN"].isin(dfafter)]
            print(f'Pending count after dfafter: {len(pending)}')
            pending = pending.drop(["FirstSrvDate"], axis=1)
            milthreshold = svc_num * 1000
            pending = pending.query("`Last Service Mileage` < @milthreshold")

            # Turn-ups
            turnup = df.query(f"`Last Service - PMS` == '{svc}'").copy()
            turnup["TargetFlag"] = 1
            if not pending.empty:
                pending["TargetFlag"] = 0

            finaltr = pd.concat([turnup, pending], axis=0)
            finaltr[f"Expected{svc}Date"] = finaltr[f"Expected{svc}Date"].fillna(finaltr["Last Service Date - PMS"])

            # Calculate ratio
            if len(finaltr) > 0:
                ratio = finaltr["TargetFlag"].sum() / len(finaltr)
            else:
                ratio = 0
            print(f'Target Range: {target_range[0]} to {target_range[1]}')
            print(f'''Iteration {iteration+1}: TargetFlag ratio = {round(ratio*100, 2)}% | Rows: {len(finaltr)}''')
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
        print(f"Final TargetFlag ratio: {round(ratio*100, 2)}% | Rows: {len(finaltr)}")

        # Save CSV separately
        output_file = f"{output_prefix}_{svc}.csv"
        # finaltr.to_csv(output_file, index=False)
        print(f"{svc} dataset saved: {output_file}")
    return finaltr


############################# Historical Data Feature Engineering #############################

def extract_k(service_str):
    """
    Extract the service number from a service description string.
    
    Args:
        service_str (str): Service description string
        
    Returns:
        int: Extracted service number or 0 if extraction fails
    """
    try:
        return int(service_str.split('-')[-1])
    except:
        return 0

def derive_pms_features1(serv1: pd.DataFrame,last_service_code: int,reference_date: datetime = None) -> pd.DataFrame:
    """
    Build PMS-related features per VIN.
    
    Parameters
    ----------
    serv1 : pd.DataFrame
        Input dataframe with columns ['Vin_No', 'Service_Date', 'Service_Num'].
    reference_date : datetime, optional
        Reference date for calculating months difference. Defaults to today.
    
    Returns
    -------
    pd.DataFrame
        Features per VIN:
        - Months_Since_First_PMS
        - Months_Since_Last_PMS
        - SinglePMS (1 if only one PMS record, else 0)
    """
    
    if reference_date is None:
        reference_date = pd.Timestamp.today()

    # Step 1: Clean and filter
    dfpmsdate = serv1.sort_values(by=['Vin_No', 'Service_Date']).copy()

    dfpmsdate = dfpmsdate.query('Service_Num > 0 and Service_Num < @last_service_code')

    # Step 2: First and last PMS date per VIN
    first_service = dfpmsdate.groupby('Vin_No')['Service_Date'].min().reset_index(name='First_PMS_Date')
    last_service = dfpmsdate.groupby('Vin_No')['Service_Date'].max().reset_index(name='Last_PMS_Date')

    #to-reference
    dfpmsdate = dfpmsdate.merge(first_service, on='Vin_No', how='left')
    dfpmsdate = dfpmsdate.merge(last_service, on='Vin_No', how='left')

    # Step 4: Create features
    dfpmsdate['Months_Since_First_PMS'] = (reference_date - dfpmsdate['First_PMS_Date']).dt.days / 30.44
    dfpmsdate['Months_Since_Last_PMS'] = (reference_date - dfpmsdate['Last_PMS_Date']).dt.days / 30.44

    # Step 5: Aggregate at VIN level
    dfpmsdate = dfpmsdate.groupby(['Vin_No']).agg(
        Months_Since_First_PMS=('Months_Since_First_PMS', 'mean'),
        Months_Since_Last_PMS=('Months_Since_Last_PMS', 'mean')
    ).reset_index()

    # Step 6: Create SinglePMS flag
    dfpmsdate['SinglePMS'] = (dfpmsdate['Months_Since_First_PMS'] == dfpmsdate['Months_Since_Last_PMS']).astype(int)
    # dfpmsdate.to_csv('validatecode/dfpmsdate.csv', index=False)
    return dfpmsdate[['Vin_No','Months_Since_Last_PMS', 'SinglePMS']]

def derive_npms_features(serv1: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
    """
    Derive Non-PMS (Service_Num == 0) related features per VIN.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.today()

    # Step 1: Filter Non-PMS data
    dfnpmsdate = serv1.copy()
    dfnpmsdate = dfnpmsdate[dfnpmsdate['Service_Num'] == 0]

    # Step 2: Convert date column
    dfnpmsdate['Service_Date'] = pd.to_datetime(dfnpmsdate['Service_Date'], dayfirst=True, errors='coerce')
    dfnpmsdate = dfnpmsdate.sort_values(by=['Vin_No', 'Service_Date'])

    # Step 3: Compute min/max dates per VIN
    first = dfnpmsdate.groupby('Vin_No')['Service_Date'].min().reset_index(name='First_NPMS_Date')
    last = dfnpmsdate.groupby('Vin_No')['Service_Date'].max().reset_index(name='Last_NPMS_Date')

    # Step 4: Create features
    dfnpmsdate = dfnpmsdate.merge(first, on='Vin_No', how='left').merge(last, on='Vin_No', how='left')
    dfnpmsdate['Months_Since_First_NPMS'] = (reference_date - dfnpmsdate['First_NPMS_Date']).dt.days / 30.44
    dfnpmsdate['Months_Since_Last_NPMS'] = (reference_date - dfnpmsdate['Last_NPMS_Date']).dt.days / 30.44

    # Step 5: Aggregate at VIN level
    dfnpmsdate = dfnpmsdate.groupby(['Vin_No']).agg(
        Months_Since_First_NPMS=('Months_Since_First_NPMS', 'mean'),
        Months_Since_Last_NPMS=('Months_Since_Last_NPMS', 'mean')
    ).reset_index()

    return dfnpmsdate[['Vin_No', 'Months_Since_Last_NPMS']]

def derive_pms_mileage_features(serv1: pd.DataFrame, dfpmsdate: pd.DataFrame,last_service_code: int) -> pd.DataFrame:
    """
    Derive PMS mileage-based features per VIN.
    If SinglePMS == 1, replace its interval with the mean for the same Service_Num
    among VINs where SinglePMS == 0.
    """
    dfpmsmil = serv1[['Vin_No', 'Service_Num', 'Mileage']].copy()
    dfpmsmil = dfpmsmil.query('Service_Num > 0 and Service_Num < @last_service_code')
    dfpmsmil = dfpmsmil.copy()
    dfpmsmil['Mileage_Diff'] = dfpmsmil.groupby('Vin_No')['Mileage'].diff()
    dfpmsmil['Mileage_Diff'] = dfpmsmil['Mileage_Diff'].fillna(dfpmsmil['Mileage'])

    avg_mileage_interval = (
        dfpmsmil.groupby('Vin_No')['Mileage_Diff']
        .mean()
        .reset_index(name='Avg_Mileage_Interval_PMS')
    )

    avg_mileage_interval = avg_mileage_interval.merge(
        dfpmsdate[['Vin_No', 'SinglePMS']], on='Vin_No', how='left'
    )

    df_max_service = dfpmsmil.groupby("Vin_No", as_index=False)["Service_Num"].max()
    avg_mileage_interval = avg_mileage_interval.merge(df_max_service, on='Vin_No', how='left')

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

    avg_mileage_interval['skipped_blocks'] = ((last_service_code - 10 - avg_mileage_interval['Service_Num']) // 10).clip(lower=0)
    avg_mileage_interval['skipped_blocks'] = np.where(avg_mileage_interval['skipped_blocks'] == 1, 2, avg_mileage_interval['skipped_blocks'])
    avg_mileage_interval['predicted_interval'] = avg_mileage_interval['Avg_Mileage_Interval_PMS'] * avg_mileage_interval['skipped_blocks']
    avg_mileage_interval.loc[avg_mileage_interval['skipped_blocks'] == 0, 'predicted_interval'] = avg_mileage_interval.loc[avg_mileage_interval['skipped_blocks'] == 0, 'Avg_Mileage_Interval_PMS']
    avg_mileage_interval = avg_mileage_interval.rename(columns={'Avg_Mileage_Interval_PMS':'Avg_Mileage_Interval_PMSold','predicted_interval':'Avg_Mileage_Interval_PMS'})
    avg_mileage_interval = avg_mileage_interval.drop(columns=['SinglePMS','Service_Num','skipped_blocks'])
    return avg_mileage_interval

def derive_npms_mileage_features(serv1: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Non-PMS (Service_Num == 0) mileage-based features per VIN.

    Parameters
    ----------
    serv1 : pd.DataFrame
        Input dataframe with columns ['Vin_No', 'Service_Date', 'Mileage', 'Service_Num'].

    Returns
    -------
    pd.DataFrame
        Features per VIN:
        - Avg_Mileage_Interval_NPMS
    """

    dfnpmsmil = serv1[serv1['Service_Num'] == 0].copy()
    if dfnpmsmil.empty:
        return pd.DataFrame(columns=['Vin_No', 'Avg_Mileage_Interval_NPMS'])

    dfnpmsmil['Mileage_Diff'] = dfnpmsmil.groupby('Vin_No')['Mileage'].diff()
    dfnpmsmil['Mileage_Diff'] = dfnpmsmil['Mileage_Diff'].fillna(dfnpmsmil['Mileage'])
    return (
        dfnpmsmil.groupby('Vin_No')['Mileage_Diff']
        .mean()
        .reset_index(name='Avg_Mileage_Interval_NPMS')
    )

def derive_pms_service_intervals(serv1: pd.DataFrame,last_service_code: int) -> pd.DataFrame:
    """
    Derive PMS service interval features per VIN.
    Computes mileage-normalized service intervals, and if invalid,
    replaces them with pure month-based service intervals.
    
    Returns
    -------
    pd.DataFrame
        ['Vin_No', 'Avg_Service_Interval_PMS', 'Avg_Service_Interval_PMS1']
    """

    # Step 1: Preprocess
    dfpmsmil = serv1.copy()

    # Filter valid PMS services
    dfpmsmil = dfpmsmil.query("Service_Num > 0 and Service_Num < @last_service_code")

    # Step 2: Get latest service entry per VIN + Service_Num
    service_max_dates = (
        dfpmsmil
        .groupby(['Vin_No', 'Service_Num'], as_index=False)
        .agg({
            'Service_Date': 'max',
            'Mileage': 'max'
        })
    )
    df_max_service = dfpmsmil.groupby("Vin_No", as_index=False)["Service_Num"].max()
    service_max_dates = service_max_dates.sort_values(['Vin_No', 'Service_Date'])
    
    # Calculate differences across sorted DataFrame
    service_max_dates['Date_Diff_Days'] = service_max_dates.groupby('Vin_No')['Service_Date'].diff().dt.days
    service_max_dates['Mileage_Diff'] = service_max_dates.groupby('Vin_No')['Mileage'].diff()
    
    service_max_dates['Months_Diff'] = service_max_dates['Date_Diff_Days'] / 30.44
    service_max_dates['Intervals_per_10k'] = service_max_dates['Months_Diff'] / (service_max_dates['Mileage_Diff'] / 10000)
    
    avg_monthly_interval11 = (
        service_max_dates.groupby('Vin_No')['Months_Diff']
        .mean()
        .reset_index(name='Avg_Service_Interval_PMS1')
    )

    avg_monthly_interval = (
        service_max_dates.groupby('Vin_No')['Intervals_per_10k']
        .mean()
        .reset_index(name='Avg_Service_Interval_PMS')
    )

    # Step 5: Merge both
    result = avg_monthly_interval.merge(avg_monthly_interval11, on='Vin_No', how='left')
    result = result.merge(df_max_service, on='Vin_No', how='left')
    # Step 6: Replace invalid values in PMS with PMS1
    mask_invalid = (
        result['Avg_Service_Interval_PMS'].isna() |
        np.isinf(result['Avg_Service_Interval_PMS']) |
        (result['Avg_Service_Interval_PMS'] < 0.5)
    )
    result.loc[mask_invalid, 'Avg_Service_Interval_PMS'] = result.loc[mask_invalid, 'Avg_Service_Interval_PMS1']

    # Step 7: Keep only the required columns
    return result[['Vin_No', 'Avg_Service_Interval_PMS', 'Avg_Service_Interval_PMS1','Service_Num']]


def adjust_service_intervals(
    df: pd.DataFrame,
    prediction_km: int
) -> pd.DataFrame:

    assert prediction_km % 10 == 0 and prediction_km >= 20, \
        "prediction_km must be >=20 and a multiple of 10"

    cols = [
        "Avg_Service_Interval_PMS",
        "Avg_Service_Interval_PMS1"
    ]

    df = df.copy()

    # --------------------------------------------------
    # Step 1: Clean invalid values
    # --------------------------------------------------
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)

    # --------------------------------------------------
    # Step 2: Remove future services (NO LEAKAGE)
    # --------------------------------------------------
    df = df[df["Service_Num"] < prediction_km]

    # --------------------------------------------------
    # Step 3: Apply valid interval bounds
    # --------------------------------------------------
    for col in cols:
        df[col] = df[col].where(df[col].between(0.5, 60))

    # --------------------------------------------------
    # Step 4: Fill missing by Service_Num mean
    # --------------------------------------------------
    for col in cols:
        df[col] = (
            df.groupby("Service_Num")[col]
              .transform(lambda x: x.fillna(x.mean()))
        )

    # --------------------------------------------------
    # Step 5: Distance from prediction mileage
    # --------------------------------------------------
    distance = (prediction_km - df["Service_Num"]) // 10

    df["Multiplier"] = np.where(
        df["Service_Num"] < 10,
        1.0,  # First service or non-PMS
        np.select(
            [
                distance == 1,
                distance == 2,
                distance == 3,
                distance >= 4,
            ],
            [
                1.0,
                1.5,
                2.0,
                distance - 1.0,
            ],
            default=1.0
        )
    )

    # --------------------------------------------------
    # Step 7: Apply multipliers
    # --------------------------------------------------
    df["Avg_Service_Interval_PMSnew"] = (
        df["Avg_Service_Interval_PMS"] * df["Multiplier"]
    )
    df["Avg_Service_Interval_PMS1new"] = (
        df["Avg_Service_Interval_PMS1"] * df["Multiplier"]
    )

    return df[
        [
            "Vin_No",
            "Service_Num",
            "Avg_Service_Interval_PMS",
            "Avg_Service_Interval_PMS1",
            "Avg_Service_Interval_PMSnew",
            "Avg_Service_Interval_PMS1new",
        ]
    ]
# def adjust_service_intervals(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Optimized version of adjust_service_intervals.
#     Fully vectorized, much faster for large datasets.
#     """   
#     cols = ["Avg_Service_Interval_PMS", "Avg_Service_Interval_PMS1"]
    
#     # --- Step 1: Replace inf with NaN ---
#     df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    
#     # --- Step 2: Compute last service per VIN ---
#     last_service = df.groupby("Vin_No")["Service_Num"].transform("max")
    
#     # --- Step 3: Apply thresholds dynamically ---
#     # Create masks for each range
#     mask_10 = df["Service_Num"] == 10
#     mask_20 = df["Service_Num"] == 20
#     mask_30_plus = (df["Service_Num"] >= 30) & (df["Service_Num"] < last_service)
#     mask_others = ~(mask_10 | mask_20 | mask_30_plus)
    
#     # Lower and upper bounds
#     for col in cols:
#         df.loc[mask_20, col] = df.loc[mask_20, col].where(df.loc[mask_20, col].between(0.5, 60))
#         df.loc[mask_30_plus, col] = df.loc[mask_30_plus, col].where(df.loc[mask_30_plus, col].between(0.5, 60))
#         df.loc[mask_others, col] = df.loc[mask_others, col].where(df.loc[mask_others, col].between(0.5, 60))
    
#     # --- Step 4: Fill missing values with Vin_No + Service_Num group mean ---
#     for col in cols:
#         df[col] = df.groupby("Service_Num")[col].transform(lambda x: x.fillna(x.mean()))
#     # for col in cols:
#     #     df.loc[
#     #         (df["Service_Num"] == 10) , 
#     #         col
#     #     ] = 6 * (last_service.max() // 10 - 1)
    
#     # Compute last service per VIN
#     df["Last_Service"] = last_service.max()
    
#     # Minimum service to apply rules
#     min_service = 20
    
#     # Initialize multiplier column with 1.0
#     df["Multiplier"] = 1.0
    
#     # Loop-based multiplier logic
#     offset = 20
#     mult = 1.5

#     while True:
#         mask = df["Service_Num"] == df["Last_Service"] - offset
#         df.loc[mask & (df["Service_Num"] >= min_service), "Multiplier"] = mult
        
#         offset += 10
#         mult += 0.5
        
#         if (df["Last_Service"] - offset < min_service).all():
#             break

#     # NEW RULE: Service_Num == 10
#     df.loc[df["Service_Num"] == 10, "Multiplier"] = (last_service - 20) / 10

#     df['Avg_Service_Interval_PMSnew'] = df['Avg_Service_Interval_PMS'] * df['Multiplier']
#     df['Avg_Service_Interval_PMS1new'] = df['Avg_Service_Interval_PMS1'] * df['Multiplier']

#     return df[[
#         "Vin_No", "Service_Num","Multiplier",
#         "Avg_Service_Interval_PMS", "Avg_Service_Interval_PMS1",
#         "Avg_Service_Interval_PMSnew", "Avg_Service_Interval_PMS1new",
#         ]]

def derive_npms_features2(serv1: pd.DataFrame):
    """
    Derive NPMS-related features for vehicles.

    Parameters:
        serv1: pd.DataFrame
            Input service dataframe with columns:
            ['Vin_No', 'Service_Num', 'Service_Date', 'Mileage', 'Revenue']

    Returns:
        freq_npms: pd.DataFrame
            DataFrame containing ['Vin_No', 'freq_NPMS'].
    """

    # Separate PMS and non-PMS
    serv1['Revenue'] = abs(serv1['Revenue'])
    df_before_110kpms = serv1[serv1['Service_Num'] > 0]
    df_before_110knpms = serv1[serv1['Service_Num'] == 0]
    # Aggregate PMS data
    newserv110kpms = df_before_110kpms.groupby(['Vin_No','Service_Num','Service_Date']).agg(
        Mileage=('Mileage', 'mean'),
        Total_Revenue=('Revenue', 'sum')
    ).reset_index()

    # Preprocess NPMS
    df_before_110knpms['Service_Date'] = pd.to_datetime(
        df_before_110knpms['Service_Date'], format='mixed', dayfirst=True, errors='coerce'
    )
    df_before_110knpms['Mileage'] = pd.to_numeric(df_before_110knpms['Mileage'], errors='coerce')

    newserv110knpms = df_before_110knpms.groupby(['Vin_No','Service_Num','Service_Date']).agg(
        Mileage=('Mileage', 'mean'),
        npmsRevenue=('Revenue', 'sum')
    ).reset_index()

    # Count NPMS per VIN
    npms_counts = newserv110knpms.groupby('Vin_No').size().reset_index(name='nNPMS')
    
    # Count PMS per VIN
    pms_counts = newserv110kpms.groupby('Vin_No').size().reset_index(name='nPMS')

    # Sum NPMS revenue per VIN
    npmsrevenue = newserv110knpms.groupby('Vin_No').agg(
        npmsRevenue=('npmsRevenue', 'sum')
    ).reset_index()

    # First and last NPMS date per VIN
    first_last_npms = df_before_110knpms.groupby('Vin_No').agg(
        First_NPMS_Date=('Service_Date', 'min'),
        Last_NPMS_Date=('Service_Date', 'max')
    ).reset_index()

    # Merge counts
    freq_npms = first_last_npms.merge(npms_counts, on='Vin_No', how='left')

    # Calculate frequency in months
    freq_npms['freq_NPMS'] = (
        (freq_npms['Last_NPMS_Date'].dt.to_period("M") - freq_npms['First_NPMS_Date'].dt.to_period("M")).apply(lambda x: x.n)
    ) / freq_npms['nNPMS']

    # Return only relevant columns
    freq_npms = freq_npms[['Vin_No', 'freq_NPMS']]

    return freq_npms,npms_counts,npmsrevenue,pms_counts

def branch_visit_features(serv1: pd.DataFrame, last_service_code: int, top_n: int = 7,) -> pd.DataFrame:
    """
    Create branch visit features per VIN.

    Parameters:
        serv1: pd.DataFrame
            Input service dataframe with columns ['Vin_No', 'Service_Branch_Name', 'Service_Date']
        top_n: int
            Number of top branches to keep individually; rest are summed as 'otherbranch_services'

    Returns:
        vin_branch_pivot: pd.DataFrame
            DataFrame with VIN, top N branch visit counts, and 'otherbranch_services'.
    """
    
    # Sort by VIN and Service_Date
    dfbrchdate = serv1.sort_values(by=['Vin_No', 'Service_Date'])
    dfbrchdate = dfbrchdate.query(f"Service_Num < @last_service_code")
    # Count visits per VIN per branch
    vin_branch_counts = (
        dfbrchdate.groupby(['Vin_No', 'Service_Branch_Name'])
        .size()
        .reset_index(name='Visit_Count')
    )
    
    # Pivot to wide format
    vin_branch_pivot = vin_branch_counts.pivot(
        index='Vin_No', columns='Service_Branch_Name', values='Visit_Count'
    )
    
    # Replace NaNs with 0 (no visits)
    vin_branch_pivot = vin_branch_pivot.fillna(0).astype(int).reset_index()
    
    # ---- Find Top N Branches Globally ----
    branch_totals = vin_branch_pivot.drop(columns=['Vin_No']).sum().sort_values(ascending=False)
    top_branches = branch_totals.head(top_n).index.tolist()
    
    # Other branches = all except VIN + top N
    other_branches = [c for c in vin_branch_pivot.columns if c not in ['Vin_No'] + top_branches]
    
    # Create "otherbranch_services" as sum of all other branches
    vin_branch_pivot['otherbranch_services'] = vin_branch_pivot[other_branches].sum(axis=1)
    
    # Keep only Vin_No + top branches + otherbranch_services
    vin_branch_pivot = vin_branch_pivot[['Vin_No'] + top_branches + ['otherbranch_services']]
    
    return vin_branch_pivot,top_branches

def branch_diversity_features(serv1: pd.DataFrame, last_service_code: int) -> pd.DataFrame:
    """
    Compute branch diversity features per VIN.
    
    Parameters:
        serv1 : pd.DataFrame
            Input service dataframe with columns ['Vin_No', 'Service_Date', 'Service_Branch_Name'].
    
    Returns:
        pd.DataFrame
            DataFrame with VIN, list of unique serviced branches, 
            and count of unique branches per VIN.
    """
    
    # Sort by VIN and Service_Date
    df_sorted = serv1.sort_values(by=['Vin_No', 'Service_Date'])
    df_sorted = df_sorted.query(f"Service_Num < @last_service_code")
    # Group and collect unique branches
    branch_grouped = (
        df_sorted.groupby('Vin_No')['Service_Branch_Name']
        .apply(lambda x: list(set(x)))
        .reset_index(name='Branch_List')
    )
    
    # Count number of unique branches
    branch_grouped['unique_branch_serviced'] = branch_grouped['Branch_List'].apply(len)
    
    return branch_grouped

def hierarchical_gower_clustering(dfmain, mergedf,feat, k_min=2, k_max=10, method="ward"):
    """
    Perform hierarchical clustering with Gower distance.
    
    Parameters:
    - dfmain: pd.DataFrame, input data
    - feat: str, column name for grouping (like VIN or Model)
    - k_min, k_max: int, range of clusters to try
    - method: str, linkage method ('ward', 'average', 'complete', etc.)
    
    Returns:
    - dfmain with cluster assignments
    - cluster mapping (cluster → list of feature values)
    """
    dfmain = dfmain.merge(mergedf,on='Vehicle Key',how='left')
    # --- Step 1: Aggregate stats like in your KMeans code ---
    # Base aggregation columns (always present)
    agg_dict = {
        "Avg_Mileage_Interval_PMS": "mean",
        "PMSRevenue": "mean",
        "Current Age": "mean",
        "Avg_Service_Interval_PMS": "mean",
        "nNPMS": "mean",
        "Service Frequency": "mean",
    }
    # Optional columns — add only if they exist in dfmain
    _mode_fn = lambda x: x.mode()[0] if not x.mode().empty else np.nan
    _optional_mode_cols = ["New / Used_NEW", "RFM_segments"]
    if feat == 'Nationality':
        _optional_mode_cols.append("Model")
    for _oc in _optional_mode_cols:
        if _oc in dfmain.columns:
            agg_dict[_oc] = _mode_fn
        
    
    stats = dfmain.groupby(feat).agg(agg_dict)

    # Convert pandas custom types to standard numpy types to avoid gower package errors with StringDtypes
    for col in stats.columns:
        if pd.api.types.is_numeric_dtype(stats[col]):
            stats[col] = stats[col].astype(float)
        else:
            stats[col] = stats[col].astype(object)

    # --- Step 2: Compute Gower distance ---
    gower_dist = gower.gower_matrix(stats)

    # --- Step 3: Try different cluster numbers and compute silhouette scores ---
    silhouette_scores = []
    K = range(k_min, k_max+1)
    
    for k in K:
        linkage_matrix = linkage(gower_dist, method=method)
        labels = fcluster(linkage_matrix, k, criterion="maxclust")
        
        if k > 1:
            silhouette_scores.append(silhouette_score(gower_dist, labels, metric="precomputed"))
        else:
            silhouette_scores.append(None)

    # --- Step 4: Select best k (max silhouette) ---
    # best_k = K[np.argmax(silhouette_scores)]
    best_k = 4

    # --- Step 5: Final clustering ---
    linkage_matrix = linkage(gower_dist, method=method)
    labels = fcluster(linkage_matrix, best_k, criterion="maxclust")

    # --- Step 6: Assign clusters back to dfmain ---
    stats["Cluster"] = labels
    dfmain[f"{feat}_Cluster"] = dfmain[feat].map(stats["Cluster"].to_dict())

    # --- Step 7: Build cluster summary ---
    clusters = (
        dfmain.groupby(f"{feat}_Cluster")[feat]
        .apply(lambda x: list(set(x)))
        .reset_index()
    )
    # clusters.to_csv(f'validatecode/{feat}_clusters.csv', index=False)
    dfmain.drop(feat,axis=1,inplace=True)
    one_hot = pd.get_dummies(dfmain[f'{feat}_Cluster'], prefix=f'{feat}_Cluster',dtype=int)

    # Concatenate back to dfmain
    dfmain = pd.concat([dfmain, one_hot], axis=1)
    dfmain.drop([f'{feat}_Cluster','Model','Nationality','RFM_segments'],axis=1,inplace=True,errors='ignore')
    # --- Optional: Plot dendrogram ---
    # plt.figure(figsize=(12, 6))
    # dendrogram(linkage_matrix, labels=stats.index.tolist(), leaf_rotation=90)
    # plt.title("Hierarchical Clustering Dendrogram")
    # plt.xlabel(feat)
    # plt.ylabel("Distance")
    # plt.show()

    return dfmain, clusters

def calculate_revenue_spend(df, last_service):
    """Sum PMS revenue columns up to the target service milestone (vectorized)."""
    df = df.copy()

    # Convert service columns (10k, 20k, ...) to numeric
    service_cols = [col for col in df.columns if col.endswith("k")]
    df[service_cols] = df[service_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    cols_to_sum = [f"{i}K" for i in range(10, last_service - 10 + 1, 10) if f"{i}K" in df.columns]
    df["PMSRevenue"] = df[cols_to_sum].sum(axis=1) if cols_to_sum else 0
    return df

def one_hot(df):
    cat_cols = df.select_dtypes(include='O').keys().tolist()
    # Safely get columns to encode (skipping the first 3 if there are that many)
    cols_to_encode = cat_cols[3:] if len(cat_cols) > 3 else cat_cols
    
    if not cols_to_encode:
        return df
        
    df = pd.get_dummies(df, columns=cols_to_encode, dtype=int)
    return df

def calculate_bodyshop_count(df, branch_col='Service_Branch_Name', vin_col='Vin_No', keyword='Bodyshop'):
    """
    Calculate the number of Bodyshop visits per VIN.

    Parameters:
    - df: pd.DataFrame containing service data
    - branch_col: str, column name for branch names
    - vin_col: str, column name for VIN
    - keyword: str, keyword to search in branch names (default 'Bodyshop')

    Returns:
    - pd.DataFrame with VIN and Bodyshop_Services count
    """
    # Filter rows containing the keyword (case-insensitive)
    filtered_df = df[df[branch_col].str.contains(keyword, case=False, na=False)]

    # Group by VIN and count occurrences
    bodyshop_counts = (
        filtered_df.groupby(vin_col)
        .size()
        .rename('Bodyshop_Services')
        .reset_index()
    )

    # Rename VIN column to a standard name
    bodyshop_counts = bodyshop_counts.rename(columns={vin_col: 'VIN'})
    return bodyshop_counts

def get_last_nonpms_before_targetpms(df_nonpms, vin_col,
                               mileage_col,
                               date_col,
                              mastertrain):
    """
    df_nonpms = NON-PMS records only
    last_pms_df = dataframe with: Vin_No, LastServicePMSDate, LastPMSMileage
    """
    lastpmsvins = mastertrain.query("TargetFlag == 1")['VIN'].unique()
    last_pms_df = mastertrain.query("TargetFlag == 1")[['VIN','Last Service Date - PMS','Last PMS Mileage']]
    last_pms_df.rename(columns={'VIN':'Vin_No'},inplace=True)
    df_nonpms = df_nonpms.copy()
    df_nonpms[date_col] = pd.to_datetime(df_nonpms[date_col])
    df_nonpms = df_nonpms[df_nonpms['Vin_No'].isin(lastpmsvins)]
    # Merge NON-PMS rows with last PMS date per VIN
    merged = df_nonpms.merge(last_pms_df, on=vin_col, how="left")

    # Keep NON-PMS before the last PMS date
    valid = merged[merged[date_col] < merged["Last Service Date - PMS"]]

    # Pick the latest non-PMS mileage per VIN
    last_nonpms = (
        valid.sort_values([vin_col, date_col], ascending=[True, False])
             .groupby(vin_col)
             .first()[[mileage_col]]
             .rename(columns={mileage_col: "LastNonPMSMileage"})
    )

    # Merge back with PMS (for fallback)
    result = last_pms_df.merge(last_nonpms, on=vin_col, how="left")

    # Fallback: if no valid NON-PMS → use PMS mileage
    # result["LastNonPMSMileage"] = result["LastNonPMSMileage"] \
    #                                  .fillna(result["Last PMS Mileage"])

    return result[['Vin_No','LastNonPMSMileage']]

def get_non_pms_events(serv, servcode_desc, mastertrain, last_service_code):
    """
    Parameters:
        serv             : DataFrame - service records
        servcode_desc    : dict      - {Service_Code: Description}
        mastertrain      : DataFrame - master training data with Vehicle_Key
        last_service_code: int       - service number used to filter pre-40k records

    Returns:
        Npmsevents   : DataFrame - Vehicle_Key with Last_Service_Non_PMS_Flag,
                                   OHE of last Non-PMS event types (top 10),
                                   and Last NonPMS Revenue per top 5 event types
        non_pms_countn: DataFrame - Non-PMS visit count per Vin_No
    """
    # --- Prep ---
    servcode_desc = servcode_desc.dropna()
    servcode_desc = dict(zip(servcode_desc['Service_Code'], servcode_desc['SO_CO_DESCRIPN_001']))

    serv = serv.copy()
    serv['Service_Date'] = pd.to_datetime(
        serv['Service_Date'], format="mixed", dayfirst=True, errors="coerce"
    )
    serv['Revenue'] = pd.to_numeric(serv['Revenue'], errors='coerce').abs()

    # Filter to vehicles in mastertrain
    veh = list(mastertrain['Vehicle Key'].unique())

    exclude_codes = {
        'ADP', 'BR4', 'CON', 'CRE', 'EXC', 'ME', 'BES', 'MES', 'C01',
        'NS4', 'PNA', 'VAS', 'VAT', 'VCP', 'VHP'
    }
    serv = serv.sort_values(by=['Vin_No', 'Service_Date'], ascending=True)
    serv = serv.query("Service_Code not in @exclude_codes").reset_index(drop=True)
    serv = serv[serv['Vehicle_Key'].isin(veh)]

    # --- Filter records before first last_service_code service ---
    service_40k = (
        serv[serv['Service_Num'] == last_service_code]
        .sort_values('Service_Date')
        .groupby('Vin_No')['Service_Date']
        .first()
        .reset_index()
        .rename(columns={'Service_Date': f'First_{last_service_code}k_Date'})
    )
    serv = serv.merge(service_40k, on='Vin_No', how='left')
    filtered_df = serv[
        (serv[f'First_{last_service_code}k_Date'].isna()) |
        (serv['Service_Date'] < serv[f'First_{last_service_code}k_Date'])
    ].copy()
    filtered_df.drop(columns=[f'First_{last_service_code}k_Date'], inplace=True)

    # --- Identify Non-PMS events ---
    filtered_df['Is_Non_PMS'] = (
        (filtered_df['Service_Num'] == 0) &
        (~filtered_df['Service_Code'].astype(str).str.contains(r'\*', na=False))
    )

    # --- Last service record per vehicle ---
    filtered_df = filtered_df.sort_values(['Vehicle_Key', 'Service_Date'])
    lstnpmseventflag = filtered_df.groupby('Vehicle_Key').tail(1)

    # =========================================================
    # ── OHE : Top 10 last Non-PMS events ─────────────────────
    # =========================================================
    last_non_pms = lstnpmseventflag[lstnpmseventflag['Is_Non_PMS'].values].copy()
    last_non_pms['Last_NonPMS_Event'] = last_non_pms['Service_Code']

    top_10_events = (
        last_non_pms['Last_NonPMS_Event']
        .value_counts()
        .head(10)
        .index
    )
    last_non_pms['Event_Mapped'] = last_non_pms['Last_NonPMS_Event'].apply(
        lambda x: x if x in top_10_events else 'Oth'
    )
    last_non_pms['Event_Npms'] = last_non_pms['Event_Mapped'].map(servcode_desc)

    event_ohe = pd.get_dummies(last_non_pms['Event_Npms'], prefix='LastNPMS_Event', dtype=int)
    ohe_df = pd.concat([last_non_pms[['Vin_No']], event_ohe], axis=1)

    # =========================================================
    # ── Revenue : Top 5 last Non-PMS events ──────────────────
    # =========================================================
    top_5_events = (
        last_non_pms['Last_NonPMS_Event']
        .value_counts()
        .head(5)
        .index
    )

    # Map service code → description for top 5 (others → 'Others')
    last_non_pms['Event_Mapped_5'] = last_non_pms['Last_NonPMS_Event'].apply(
        lambda x: x if x in top_5_events else 'Oth'
    )
    last_non_pms['Event_Npms_5'] = (
        last_non_pms['Event_Mapped_5']
        .map(servcode_desc)
        .fillna(last_non_pms['Event_Mapped_5'])   # fallback to code if no description
    )

    # Revenue column name per vehicle
    last_non_pms['Rev_Col'] = 'LastNPMS_Rev_' + last_non_pms['Event_Npms_5']

    # Pivot: one revenue column per top-5 service type
    rev_pivot = (
        last_non_pms
        .assign(Revenue=last_non_pms['Revenue'].abs())        # already abs, safety guard
        .pivot_table(
            index='Vin_No',
            columns='Rev_Col',
            values='Revenue',
            aggfunc='last'                                    # last visit revenue
        )
        .reset_index()
        .fillna(0)
    )
    rev_pivot.columns.name = None

    # =========================================================
    # ── Flag column ───────────────────────────────────────────
    # =========================================================
    lstnpmseventflag = lstnpmseventflag.copy()
    lstnpmseventflag['Last_Service_Non_PMS_Flag'] = lstnpmseventflag['Is_Non_PMS'].astype(int)
    lstnpmseventflag = lstnpmseventflag[['Vin_No', 'Last_Service_Non_PMS_Flag']]

    # =========================================================
    # ── Final merge ───────────────────────────────────────────
    # =========================================================
    Npmsevents = (
        lstnpmseventflag
        .merge(ohe_df,    on='Vin_No', how='left')
        .merge(rev_pivot, on='Vin_No', how='left')
        .fillna(0)
    )

    # --- Non-PMS visit count ---
    non_pms_count = (
        filtered_df[filtered_df['Is_Non_PMS']]
        .groupby(['Vin_No', 'Service_Date'])
        .size()
        .reset_index(name='nNPMS')
    )
    non_pms_countn = non_pms_count.groupby('Vin_No').size().reset_index(name='nNPMS')

    return Npmsevents, non_pms_countn


def get_last_nonpms_mileage(df_nonpms,mastertrain, vin_col="Vin", mileage_col="Mileage",
                            date_col="Service Date"):
    """
    Returns the last NON-PMS mileage for each VIN.
    """
    lastpmsvins = mastertrain.query("TargetFlag == 0")['VIN'].unique()
    last_pms_df = mastertrain.query("TargetFlag == 0")[['VIN','Last PMS Mileage']]
    last_pms_df.rename(columns={'VIN':'Vin_No'},inplace=True)
    # Convert service date to datetime if needed
    df_nonpms[date_col] = pd.to_datetime(df_nonpms[date_col])
    df_nonpms = df_nonpms[df_nonpms['Vin_No'].isin(lastpmsvins)]
    # Sort by VIN and Service Date descending
    df_nonpms = df_nonpms.sort_values([vin_col, date_col], ascending=[True, False])
    df_nonpms = df_nonpms.query("Service_Num == 0")
    # Take the first (latest) record for each VIN
    last_nonpms = df_nonpms.groupby(vin_col).first().reset_index()
    last_nonpms.rename(columns={mileage_col:'LastNonPMSMileage'},inplace=True)
    last_nonpms = last_nonpms.merge(last_pms_df,on='Vin_No',how='left')
    last_nonpms["LastNonPMSMileage"] = last_nonpms["LastNonPMSMileage"] \
                                      .fillna(last_nonpms["Last PMS Mileage"])
    return last_nonpms[[vin_col, 'LastNonPMSMileage']]

def find_previous_service_mileage(service_histories, target, original_target=None):
    # Check for empty or null array
    if not service_histories or service_histories is None or len(service_histories) == 0:
        return None

    if original_target is None:
        original_target = target

    # Calculate buffer value once
    buffer = original_target * 1000 + 10000

    # Use iterative approach to find previous service
    current_target = target - 10

    while current_target >= 10:
        # Find all services with current target Service_Num
        matching_services = [
            service for service in service_histories 
            if service.get('Service_Num', 0) == current_target
        ]

        if matching_services:
            # If multiple matches, find the one with mileage closest to buffer
            best_service = min(
                matching_services,
                key=lambda s: abs((s.get('Mileage') or 0) - buffer)
            )

            return {
                'Vin_No':best_service.get('Vin_No'),
                'Mileage': best_service.get('Mileage'),
                'Service_Date': best_service.get('Service_Date'),
                'Service_Num': best_service.get('Service_Num')
            }

        # Move to next target (decrease by 10)
        current_target -= 10

    # No match found from target down to 10
    return None

def group_by_vin(service_records):
    vin_groups = defaultdict(list)
    for rec in service_records:
        vin = rec.get("Vin_No")
        vin_groups[vin].append(rec)
    return vin_groups

def find_previous_service_for_all_vins(records, target_service_num):
    vin_groups = group_by_vin(records)
    results = {}

    for vin, history in vin_groups.items():
        prev_service = find_previous_service_mileage(history, target_service_num)
        results[vin] = prev_service

    return results

def map_vhc_history(main_df, history_df):

    history_df = history_df.copy()
    history_df.rename(columns={'Vin_No':'VIN'}, inplace=True)

    # Replace "-" with NaN
    replace_cols = [
        "Survey Score", "Survey Status",
        "VHC Quoted", "VHC Sold", "VHC Lost Sale",
        "VHC Lost Red Sale", "VHC Deferred",
        "VHC Amber Deferred", "VHC Completed"
    ]
    history_df[replace_cols] = history_df[replace_cols].replace("-", np.nan)

    # Convert numeric VHC fields
    vhc_cols = [
        "VHC Quoted", "VHC Sold", "VHC Lost Sale",
        "VHC Lost Red Sale", "VHC Deferred", "VHC Amber Deferred"
    ]
    history_df[vhc_cols] = history_df[vhc_cols].apply(
        lambda c: pd.to_numeric(c, errors="coerce")
    )

    # Convert Survey Score
    history_df["Survey Score"] = pd.to_numeric(history_df["Survey Score"], errors="coerce")

    # Survey Status missing → Unknown
    history_df["Survey Status"] = history_df["Survey Status"].fillna("Unknown")

    # Convert VHC Completed → flag
    history_df["VHC Completed"] = pd.to_datetime(history_df["VHC Completed"], errors="coerce", dayfirst=True)
    history_df["VHC Completed"] = history_df["VHC Completed"].replace('-',np.nan)
    history_df["VHC Completed_Flag"] = history_df["VHC Completed"].notna().astype(int)

    # Standardize service date
    history_df["Service_Date"] = pd.to_datetime(history_df["Service_Date"], errors="coerce")

    # Sort before grouping
    history_df = history_df.sort_values(["VIN", "Service_Date"])

    # ---- DO NOT TAKE CUMULATIVE SUM ----
    # history_df[vhc_cols] = history_df.groupby("VIN")[vhc_cols].cumsum()

    # Keep only required cols
    map_cols = [
        "VIN", "Service_Num",
        "Survey Score", "Survey Status",
        "VHC Completed_Flag"
    ] + vhc_cols
    history_map = history_df[map_cols]

    # Aggregation logic: exact values per service
    numeric_cols = [
        "Survey Score", 
        "VHC Quoted", "VHC Sold", "VHC Lost Sale",
        "VHC Lost Red Sale", "VHC Deferred", "VHC Amber Deferred"
    ]
    categorical_cols = ["Survey Status", "VHC Completed_Flag"]

    agg_dict = {
        **{col: "last" for col in numeric_cols},  # exact value
        **{col: (lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
           for col in categorical_cols}
    }

    df_out = (
        history_map.groupby(["VIN", "Service_Num"])
            .agg(agg_dict)
            .reset_index()
    )
    # df_out.to_csv('validatecode/vhc_history_mapped.csv', index=False)
    # Merge into main data
    merged = main_df.merge(df_out, on=["VIN", "Service_Num"], how="left")
    # merged.to_csv('validatecode/final_merged_vhc.csv', index=False)
    return merged

def derive_servcode(value):
    try:
        num = int(value)  # convert to integer (handles leading zeros)
        return num if num % 10 == 0 else 0
    except ValueError:
        return 0  # non-numeric values

def map_appointments_to_services(appointments_df, service_df):
    # Ensure datetime formats
    appointments_df = appointments_df.copy()
    service_df = service_df.copy()
    appointments_df.rename(columns={'WIP_VEH_CHASSIS':'Vin_No'},inplace=True)
    appointments_df["WIP_DATECREATED"] = pd.to_datetime(appointments_df["WIP_DATECREATED"],format='mixed',dayfirst=True)
    appointments_df["Date_Due_In_+10 Days"] = pd.to_datetime(appointments_df["Date_Due_In_+10 Days"],format='mixed',dayfirst=True)
    service_df["Service_Date"] = pd.to_datetime(service_df["Service_Date"],format='mixed',dayfirst=True)
    appointments_df = appointments_df.query("WIP_DATECREATED <= '2025-06-30'")
    appointments_df = appointments_df[appointments_df['Vin_No'].isin(service_df['Vin_No'].unique())]
    results = []

    # Process each appointment row
    for _, appt in appointments_df.iterrows():
        vin = appt["Vin_No"]
        start_date = appt["WIP_DATECREATED"]
        end_date = appt["Date_Due_In_+10 Days"]

        # Filter service history for this VIN and date window
        matched_services = service_df[
            (service_df["Vin_No"] == vin) &
            (service_df["Service_Date"] >= start_date) &
            (service_df["Service_Date"] <= end_date)
        ]

        # Flag if any service appeared
        appointment_flag = 1 if len(matched_services) > 0 else 0

        # Optional: aggregate service_nums for reference
        service_nums = matched_services["Service_Num"].tolist()
        distinct_count = len(set(service_nums))
        
        results.append({
            "Vin_No": vin,
            "WIP_DATECREATED": start_date,
            "Date_Due_In_+10 Days": end_date,
            "appointment_booked_showed_up": appointment_flag,
            "matched_service_count": distinct_count,
            "matched_service_nums": service_nums
        })

    return pd.DataFrame(results)

def adjust_for_target_pms(final_df, appt_df, target_pms):
    """
    final_df = output from build_final_pms_appointment_summary()
    appt_df  = full appointment mapping dataframe (with matched_service_nums)
    target_pms = e.g. 70
    """

    prev_pms = target_pms - 10  # Example: 70 → check PMS 60

    final_df = final_df.copy()
    appt_df = appt_df.copy()

    # Step 1: Reduce total appointment count
    final_df["total_appointments_showed_up"] = (
        final_df["total_appointments_showed_up"] - final_df["appointment_booked_showed_up"]
    )

    # Step 2: Identify VINs showing up for previous PMS (target - 10)
    prev_pms_flag = (
        appt_df.groupby("Vin_No")["matched_service_nums"]
        .apply(lambda lists: int(any(prev_pms in lst for lst in lists)))
        .reset_index()
        .rename(columns={"matched_service_nums": "showed_prev_pms"})
    )

    # Merge this flag into final_df
    final_df = final_df.merge(prev_pms_flag, on="Vin_No", how="left")

    # Step 3: Apply rule ONLY for VINs whose PMS_Service == target
    mask = final_df["PMS_Service"] == target_pms

    final_df.loc[mask, "appointment_booked_showed_up"] = final_df.loc[mask, "showed_prev_pms"]

    # Drop helper column
    final_df = final_df.drop(columns=["showed_prev_pms"])

    return final_df

def derive_appointment_show_features(
    appt_df,
    service_df,
    serv_code,
    filter_date,
    vehicle_col="Vehicle Magic",
    status_col="WIP Status New",
    booking_date_col="Due Date IN"
):
    """
    Derives appointment show/no-show features per Vehicle Magic
    using data only up to filter_date.
    """

    df = appt_df.copy()
    df["Service_Num"] = df["WIP_SERVCODE"].apply(derive_servcode)
    
    service_df = service_df.copy()
    service_df["Vehicle Magic"] = service_df["Vehicle_Key"].str.split("-", n=1).str[1]
    service_df["Vehicle Magic"] = service_df["Vehicle Magic"].str.split("-", n=1).str[0]
    service_df["Vehicle Magic"] = service_df["Vehicle Magic"].astype(int)

    df = df[df['Vehicle Magic'].isin(service_df['Vehicle Magic'].unique())]
    # ----------------------------
    # Datetime conversion
    # ----------------------------
    df[booking_date_col] = pd.to_datetime(
        df[booking_date_col], errors="coerce", dayfirst=True,format='mixed'
    )
    filter_date = pd.to_datetime(filter_date)

    # ----------------------------
    # Apply filter date (NO leakage)
    # ----------------------------
    df = df[df[booking_date_col] <= filter_date]
    df = df.query("Service_Num < @serv_code")
    # ----------------------------
    # Normalize status values
    # ----------------------------
    df[status_col] = df[status_col].astype(str).str.strip().str.title()

    # Flags
    df["show_flag"] = (df[status_col] == "Show").astype(int)
    df["no_show_flag"] = (df[status_col] == "No Show").astype(int)

    # ----------------------------
    # Aggregate per Vehicle Magic
    # ----------------------------
    features = (
        df.groupby(vehicle_col)
        .agg(
            total_appointments_showed_up=("show_flag", "sum"),
            no_of_appointments_booked_but_not_showed_up=("no_show_flag", "sum"),
        )
        .reset_index()
    )

    # At least once flag
    features["appointment_booked_showed_up_atleastonce"] = (
        features["total_appointments_showed_up"] > 0
    ).astype(int)

    return features

def transform_complaint_features(main_df, service_df):
    """
    Build complaint-related features for each VIN based on LAST PMS SERVICE.
    
    Inputs:
        main_df   → contains Vin_No and Last_PMS_Service
        service_df → full service history including Complaint fields
        
    Outputs:
        DataFrame with:
            Vin_No
            latest_complaint_category
            has_complaint_history
            num_past_complaints
            avg_complaint_resolution_days
            max_complaint_resolution_days
            recent_complaint_resolution_days
    """

    # Copy to avoid modification
    main_df = main_df.copy()
    service_df = service_df.copy()
    main_df.rename(columns={'VIN':'Vin_No','Service_Num':'Last_PMS_Service'},inplace=True)
    # Standardize date and numeric fields
    if "Service_Date" in service_df.columns:
        service_df["Service_Date"] = pd.to_datetime(service_df["Service_Date"],
                                                    errors="coerce", dayfirst=True)

    # Check if complaint columns exist in the dataset; if not, add them as empty
    if "Complaint Closed in" not in service_df.columns:
        service_df["Complaint Closed in"] = np.nan
    if "Complaint Category" not in service_df.columns:
        service_df["Complaint Category"] = np.nan

    # Ensure numeric resolution days
    service_df["Complaint Closed in"] = pd.to_numeric(
        service_df["Complaint Closed in"], errors="coerce"
    )

    # Merge last PMS service with service history
    merged = service_df.merge(
        main_df[["Vin_No", "Last_PMS_Service"]],
        on="Vin_No",
        how="inner"
    )

    # Filter OUT services after the PMS milestone
    merged = merged[merged["Service_Num"] <= merged["Last_PMS_Service"]]

    # Sort for latest complaint extraction
    merged = merged.sort_values(["Vin_No", "Service_Date"])

    results = []

    for vin, grp in merged.groupby("Vin_No"):
        complaint_cats = grp["Complaint Category"].dropna()
        resolution_vals = grp["Complaint Closed in"].dropna()

        # Latest complaint category
        latest_category = complaint_cats.iloc[-1] if len(complaint_cats) > 0 else 'unknown'

        # Recent resolution time
        recent_resolution = resolution_vals.iloc[-1] if len(resolution_vals) > 0 else 0

        # Average resolution time
        avg_resolution = resolution_vals.mean() if len(resolution_vals) > 0 else 0

        # Max resolution time
        max_resolution = resolution_vals.max() if len(resolution_vals) > 0 else 0

        has_history = int(len(complaint_cats) > 0)

        # --------------------------
        # NEW RULE: If VIN has complaint history and the values are 0 → map to 1
        # --------------------------
        if has_history == 1:
            if avg_resolution == 0:
                avg_resolution = 1
            if max_resolution == 0:
                max_resolution = 1
            if recent_resolution == 0:
                recent_resolution = 1

        results.append({
            "Vin_No": vin,
            "latest_complaint_category": latest_category,
            "has_complaint_history": int(len(complaint_cats) > 0),
            "num_past_complaints": len(complaint_cats),
            "avg_complaint_resolution_days": avg_resolution,
            "max_complaint_resolution_days": max_resolution,
            "recent_complaint_resolution_days": recent_resolution
        })

    return pd.DataFrame(results)

def compute_late_appointment_metrics(appt_df, service_df,filter_date):
    """
    Computes:
      1) Number of late appointments per Vehicle Magic
      2) Median delay days for late appointments per Vehicle Magic
    """

    appt_df = appt_df.copy()
    service_df = service_df.copy()
    service_df = service_df.query("Service_Num > 1")
    service_df["Vehicle Magic"] = service_df["Vehicle_Key"].str.split("-", n=1).str[1]
    service_df["Vehicle Magic"] = service_df["Vehicle Magic"].str.split("-", n=1).str[0]
    service_df["Vehicle Magic"] = service_df["Vehicle Magic"].astype(int)
    # ----------------------------
    # Datetime conversions
    # ----------------------------
    appt_df["Due Date IN"] = pd.to_datetime(
        appt_df["Due Date IN"], errors="coerce", dayfirst=True, format='mixed'
    )
    service_df["Service_Date"] = pd.to_datetime(
        service_df["Service_Date"], errors="coerce", dayfirst=True, format = 'mixed'
    )
    appt_df = appt_df[appt_df['Vehicle Magic'].isin(service_df["Vehicle Magic"].unique())]

    appt_df["Service_Num"] = appt_df["WIP_SERVCODE"].apply(derive_servcode)
    appt_df = appt_df.query("`Due Date IN` <= @filter_date and  Service_Num > 1")

    # ----------------------------
    # Keep only appointments that showed up
    # ----------------------------
    showed_up = appt_df[appt_df["No Show"] == 0]

    # ----------------------------
    # Deduplicate service history
    # One row per Vehicle Magic + Service_Num
    # ----------------------------
    service_dedup = (
        service_df
        .groupby(["Vehicle Magic", "Service_Num"], as_index=False)
        .agg({"Service_Date": "min"})
    )

    # ----------------------------
    # Merge appointment with service history
    # ----------------------------
    merged = showed_up.merge(
        service_dedup,
        on=["Vehicle Magic", "Service_Num"],
        how="left"
    )

    # ----------------------------
    # Identify late appointments
    # ----------------------------
    merged["appoin_delay_days"] = (
        merged["Service_Date"] - merged["Due Date IN"]
    ).dt.days

    merged["is_late_appointment"] = (merged["appoin_delay_days"] > 0).astype(int)

    # ----------------------------
    # Aggregate metrics per Vehicle Magic
    # ----------------------------
    metrics = (
        merged[merged["is_late_appointment"] == 1]
        .groupby("Vehicle Magic")
        .agg(
            no_of_late_appointments=("is_late_appointment", "sum"),
            Avg_appointment_delay_days=("appoin_delay_days", "mean") 
        )
        .reset_index()
    )

    return metrics

def compute_last_appointment_status_with_constant_service_code(
    appt_df,
    service_df,
    last_service_code,
    filter_date
):
    """
    Computes last appointment status and delay days per Vehicle Magic
    based on the LAST COMPLETED SERVICE strictly BEFORE last_service_code.
    
    Parameters:
        last_service_code : int (e.g. 60)
    """

    appt_df = appt_df.copy()
    service_df = service_df.copy()
    service_df = service_df.query("Service_Num > 1")
    service_df["Vehicle Magic"] = service_df["Vehicle_Key"].str.split("-", n=1).str[1]
    service_df["Vehicle Magic"] = service_df["Vehicle Magic"].str.split("-", n=1).str[0]
    service_df["Vehicle Magic"] = service_df["Vehicle Magic"].astype(int)
    # ----------------------------
    # Datetime conversions
    # ----------------------------
    appt_df["Due Date IN"] = pd.to_datetime(
        appt_df["Due Date IN"], errors="coerce", dayfirst=True,format='mixed',
    )
    service_df["Service_Date"] = pd.to_datetime(
        service_df["Service_Date"], errors="coerce", dayfirst=True,format='mixed',
    )

    appt_df = appt_df[appt_df['Vehicle Magic'].isin(service_df["Vehicle Magic"].unique())]

    appt_df["Service_Num"] = appt_df["WIP_SERVCODE"].apply(derive_servcode)
    appt_df = appt_df.query("`Due Date IN` <= @filter_date and  Service_Num > 1")

    # ----------------------------
    # Deduplicate service history
    # One row per Vehicle Magic + Service_Num
    # ----------------------------
    service_dedup = (
        service_df
        .groupby(["Vehicle Magic", "Service_Num"], as_index=False)
        .agg({"Service_Date": "min"})
    )

    # ----------------------------
    # Derive EFFECTIVE last service (< last_service_code)
    # ----------------------------
    eligible_services = service_dedup[
        service_dedup["Service_Num"] < last_service_code
    ]

    effective_last_service = (
        eligible_services
        .sort_values(["Vehicle Magic", "Service_Num"])
        .groupby("Vehicle Magic")
        .tail(1)
        .rename(columns={"Service_Num": "Effective_Last_Service_Num"})
    )

    # ----------------------------
    # Match appointment for effective last service
    # ----------------------------
    appt_last = appt_df.merge(
        effective_last_service[["Vehicle Magic", "Effective_Last_Service_Num"]],
        left_on=["Vehicle Magic", "Service_Num"],
        right_on=["Vehicle Magic", "Effective_Last_Service_Num"],
        how="left"
    )

    # Keep only rows where effective service exists
    appt_last = appt_last[appt_last["Effective_Last_Service_Num"].notna()]

    # ----------------------------
    # Merge with service history
    # ----------------------------
    merged = appt_last.merge(
        service_dedup,
        on=["Vehicle Magic", "Service_Num"],
        how="left"
    )

    # ----------------------------
    # Derive appointment status
    # ----------------------------
    def derive_status(row):
        if row["No Show"] == 1 or pd.isna(row["Service_Date"]):
            return "NO_SHOW", np.nan

        delay = (row["Service_Date"] - row["Due Date IN"]).days

        if delay > 0:
            return "LATE_SHOW", delay
        else:
            return "IN_DUE_TIME", 0

    status_delay = merged.apply(derive_status, axis=1, result_type="expand")

    merged["last_appointment_status"] = status_delay[0]
    merged["last_appointment_delay_days"] = status_delay[1]
    merged = merged.drop_duplicates(subset="Vehicle Magic", keep="first")
    # ----------------------------
    # Final Output
    # ----------------------------
    return merged[[
        "Vehicle Magic",
        "last_appointment_status",
        "last_appointment_delay_days"
    ]]

def backfill_vhc_for_target_rows(
    merged_df,
    history_df,
    target_service_num,
    vhc_cols,
    survey_cols,
    vin_col="VIN",
    service_col="Service_Num",
    targetflag_col="TargetFlag"
):
    """
    Backfills VHC features for TargetFlag == 1 rows
    using the latest completed service BEFORE target_service_num.
    """

    merged_df = merged_df.copy()
    history_df = history_df.copy()
    history_df = history_df.rename(columns={'Vin_No':'VIN'})
    history_df[vhc_cols + ["Survey Score", "Survey Status"]] = history_df[vhc_cols + ["Survey Score", "Survey Status"]].replace("-",0)
    history_df['Survey Status'] = history_df['Survey Status'].replace(0,'Unknown')
    history_df["VHC Completed"] = history_df["VHC Completed"].replace('-',np.nan)
    history_df["VHC Completed_Flag"] = history_df["VHC Completed"].notna().astype(int)
    # Work only with services BEFORE target
    history_past = history_df[
        history_df[service_col] < target_service_num
    ]

    # Get last completed service per VIN before target
    last_past_service = (
        history_past
        .sort_values([vin_col, service_col])
        .groupby(vin_col)
        .tail(1)
    )

    # Columns to backfill
    fill_cols = vhc_cols + survey_cols

    # Index for fast lookup
    last_past_service = last_past_service.set_index(vin_col)

    # Mask: Target rows with missing VHC
    mask = (
        (merged_df[targetflag_col] == 1) &
        (merged_df[service_col] == target_service_num)
    )

    for col in fill_cols:
        merged_df.loc[mask, col] = (
            merged_df.loc[mask, vin_col]
            .map(last_past_service[col])
            .fillna(merged_df.loc[mask, col])
        )

    return merged_df

def count_service_appointments_booked(
    appt_df,
    service,
    serv_code,
    filter_date,
    vehicle_col="Vehicle Magic",
    booking_date_col="WIP Booking Date"
    ):
    df = appt_df.copy()
    df = df[df['Vehicle Magic'].isin(service['Vehicle Magic'].unique())]
    df[booking_date_col] = pd.to_datetime(df[booking_date_col], dayfirst = True,format='mixed',errors="coerce")
    filter_date = pd.to_datetime(filter_date)
    df["Service_Num"] = df["WIP_SERVCODE"].apply(derive_servcode)
    df = df[df[booking_date_col] <= filter_date]
    df = df.query("Service_Num < @serv_code")

    return (
        df.groupby(vehicle_col)
          .size()
          .reset_index(name="no_of_service_appointments_booked")
    )

def ensure_list(x):
    if isinstance(x, list):
        return x
    return [x]

def compute_revenue_buckets(vhs_list, revenue_list,vhs_to_category):
    buckets = {
        'Lost Revenue': 0.0,
        'Deferred Revenue': 0.0,
        'Invoiced Revenue': 0.0,
        'Other VHC Revenue':0.0
    }
    
    for status, rev in zip(vhs_list, revenue_list):
        category = vhs_to_category.get(status, 'Other VHC Revenue')
        buckets[category] += float(rev)
    
    return pd.Series(buckets)
    
def split_parts_by_status(parts, status):
    lost, deferred, invoiced = [], [], []

    for p, s in zip(parts, status):
        if s == 'Lost':
            lost.append(p)
        elif s == 'Deferred':
            deferred.append(p)
        elif s == 'Invoiced':
            invoiced.append(p)

    return pd.Series({
        'Lost_Parts': lost,
        'Deferred_Parts': deferred,
        'Invoiced_Parts': invoiced
    })
    
def clean_parts(x):
    # If DataFrame, take the first column
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]

    return (
        x
        .explode()
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: (s != '-') & (s != '') & (s != 'nan')]
    )

def red_revenue_split(crit_list, status_list, revenue_list):
    red_lost = 0.0
    red_deferred = 0.0
    red_invoiced = 0.0

    for c, s, r in zip(crit_list, status_list, revenue_list):
        if c == 'Red':
            if s == 'Lost':
                red_lost += float(r)
            elif s == 'Deferred':
                red_deferred += float(r)
            elif s == 'Invoiced':
                red_invoiced += float(r)

    total_red = red_lost + red_deferred + red_invoiced

    if total_red == 0:
        return pd.Series({
            'Red_Lost_Revenue_Pct': 0.0,
            'Red_Deferred_Revenue_Pct': 0.0,
            'Red_Invoiced_Revenue_Pct': 0.0
        })

    return pd.Series({
        'Red_Lost_Revenue_Pct': (red_lost / total_red)*100,
        'Red_Deferred_Revenue_Pct': (red_deferred / total_red)*100,
        'Red_Invoiced_Revenue_Pct': (red_invoiced / total_red)*100
    })

def vhcpreparation(df,last_service_code,maindf):
    df = df[df['Vehicle Key'].isin(maindf['Vehicle Key'].unique())]
    df = df.query("`Service Code`<= @last_service_code")
    df['VHC Revenue'] = df['VHC Revenue'].apply(lambda c: pd.to_numeric(c, errors="coerce"))
    df['VHS Status'] = df['VHS Status'].replace({'Deleted': 'Lost'})
    grouped_df = (
    df
    .groupby(['Vehicle Key', 'Service Code'], as_index=False)
    .agg({
        'Refined Description (Enhanced)': lambda x: list(x.dropna().values),
        'Criticality': lambda x: list(x.dropna().values),
        'VHS Status': lambda x: list(x.dropna().values),
        'VHC Revenue': lambda x: list(x.dropna().values),
        })
    )
    grouped_df['VHS Status'] = grouped_df['VHS Status'].apply(ensure_list)
    grouped_df['VHC Revenue'] = grouped_df['VHC Revenue'].apply(ensure_list)

    vhs_to_category = {
        'Lost': 'Lost Revenue',
        'Deleted': 'Lost Revenue',
    
        'Deferred': 'Deferred Revenue',
        'Open': 'Deferred Revenue',
        'Pending': 'Deferred Revenue',
        'In Progress': 'Deferred Revenue',
    
        'Invoiced': 'Invoiced Revenue',
        'Closed': 'Invoiced Revenue',
        '-': 'Other VHC Revenue'
    }
    revenue_cols = grouped_df.apply(
        lambda r: compute_revenue_buckets(
            r['VHS Status'],
            r['VHC Revenue'],
            vhs_to_category
        ),
        axis=1
    )
    revenue_cols['Total VHC Revenue'] = (
        revenue_cols['Lost Revenue'] +
        revenue_cols['Deferred Revenue'] +
        revenue_cols['Invoiced Revenue'] +
        revenue_cols['Other VHC Revenue']
    )
    grouped_df = pd.concat([grouped_df, revenue_cols], axis=1)
    grouped_df['VHC Revenue Convertion'] = ((grouped_df['Invoiced Revenue'] / grouped_df['Total VHC Revenue'])*100).fillna(0)
    grouped_df['Deferred Revenue Ratio'] = ((grouped_df['Deferred Revenue'] / grouped_df['Total VHC Revenue'])).fillna(0)
    grouped_df['VHC Revenue at Risk(%)'] = (((grouped_df['Deferred Revenue'] + grouped_df['Lost Revenue'])/ grouped_df['Total VHC Revenue'])*100).fillna(0)

    grouped_df['vhc_Lost'] = grouped_df['VHS Status'].apply(lambda x: int('Lost' in x))
    grouped_df['vhc_Deferred'] = grouped_df['VHS Status'].apply(lambda x: int('Deferred' in x))
    grouped_df['vhc_Invoiced'] = grouped_df['VHS Status'].apply(lambda x: int('Invoiced' in x))
    
    grouped_df['VHC_Criticality_Red'] = grouped_df['Criticality'].apply(lambda x: int('Red' in x))
    grouped_df['VHC_Criticality_Amber'] = grouped_df['Criticality'].apply(lambda x: int('Amber' in x))
    parts_split = grouped_df.apply(
        lambda r: split_parts_by_status(
            r['Refined Description (Enhanced)'],
            r['VHS Status']
        ),
        axis=1
    )
    grouped_df = pd.concat([grouped_df, parts_split], axis=1)
    TOP_K = 20
    top_lost_parts = [
        p for p, _ in Counter(
            clean_parts(grouped_df['Lost_Parts'])
        ).most_common(TOP_K)
    ]
    
    top_invoiced_parts = [
        p for p, _ in Counter(
            clean_parts(grouped_df['Invoiced_Parts'])
        ).most_common(TOP_K)
    ]
    
    top_deferred_parts = [
        p for p, _ in Counter(
            clean_parts(grouped_df['Deferred_Parts'])
        ).most_common(TOP_K)
    ]

    for p in top_lost_parts:
        grouped_df[f'LostPart__{p}'] = grouped_df['Lost_Parts'].apply(lambda x: int(p in x))

    for p in top_invoiced_parts:
        grouped_df[f'InvoicedPart__{p}'] = grouped_df['Invoiced_Parts'].apply(lambda x: int(p in x))
    
    for p in top_deferred_parts:
        grouped_df[f'DeferredPart__{p}'] = grouped_df['Deferred_Parts'].apply(lambda x: int(p in x))
    
    red_rev_features = grouped_df.apply(
        lambda r: red_revenue_split(
            r['Criticality'],
            r['VHS Status'],
            r['VHC Revenue']
        ),
        axis=1
    )
    grouped_df = pd.concat([grouped_df, red_rev_features], axis=1)
    grouped_dfN = grouped_df.drop(['Refined Description (Enhanced)','Criticality','VHS Status','VHC Revenue','Lost_Parts','Deferred_Parts','Invoiced_Parts'],axis=1,errors='ignore')
    grouped_dfN = grouped_dfN.rename(columns={'Service Code':'Service_Num'})
    grouped_dfN['VHC Completed_Flag'] = 1
    return grouped_dfN

def backfill_vhc_leakage_safe(
    merged_df,
    history_df,
    vhc_cols,
    vin_col="VIN",
    service_col="Service_Num",
    targetflag_col="TargetFlag"
):
    """
    Backfills VHC + Survey features for VINs whose
    LAST service is the target service.

    Prevents leakage by:
    - Identifying VINs whose max(Service_Num) == current row service
    - Backfilling only from strictly previous service
    """

    merged_df = merged_df.copy()
    history_df = history_df.copy()

    # -----------------------------
    # 1️⃣ Basic Normalization
    # -----------------------------

    history_df[vhc_cols] = (
        history_df[vhc_cols]
        .replace("-", 0)
    )
    # -----------------------------
    # 2️⃣ Identify last service per VIN
    # -----------------------------
    last_service_per_vin = (
        history_df
        .groupby(vin_col)[service_col]
        .max()
        .reset_index()
        .rename(columns={service_col: "Last_Service_Num"})
    )

    merged_df = merged_df.merge(
        last_service_per_vin,
        on=vin_col,
        how="left"
    )

    # -----------------------------
    # 3️⃣ Mask: VIN where current row IS last service
    # -----------------------------
    mask_target_last = (
        (merged_df[targetflag_col] == 1) &
        (merged_df[service_col] == merged_df["Last_Service_Num"])
    )

    # -----------------------------
    # 4️⃣ Get previous service (< current service)
    # -----------------------------
    history_prev = history_df.merge(
        last_service_per_vin,
        on=vin_col,
        how="left"
    )

    history_prev = history_prev[
        history_prev[service_col] < history_prev["Last_Service_Num"]
    ]

    # Get latest available previous service
    last_prev_service = (
        history_prev
        .sort_values([vin_col, service_col])
        .groupby(vin_col)
        .tail(1)
        .set_index(vin_col)
    )

    # -----------------------------
    # 5️⃣ Backfill columns
    # -----------------------------
    fill_cols = vhc_cols 

    for col in fill_cols:
        merged_df.loc[mask_target_last, col] = (
            merged_df.loc[mask_target_last, vin_col]
            .map(last_prev_service[col])
            .fillna(merged_df.loc[mask_target_last, col])
        )

    # Cleanup helper column
    merged_df = merged_df.drop(columns=["Last_Service_Num"])

    return merged_df

def compute_service_features(serv, filterdate,last_service_num=60):
    """
    Compute PMS and Non-PMS revenue features per Vin_No.

    Parameters
    ----------
    serv             : pd.DataFrame  — raw service history dataframe
    last_service_num : int           — Service_Num value to exclude (default: 60)

    Returns
    -------
    pd.DataFrame with one row per Vin_No and the derived features.
    """
    from datetime import datetime

    # ── Prep ──────────────────────────────────────────────────────────────────
    serv = serv.copy()
    serv['Service_Date'] = pd.to_datetime(serv['Service_Date'])
    serv['Revenue']      = pd.to_numeric(serv['Revenue'],      errors='coerce')
    serv['Service_Num']  = pd.to_numeric(serv['Service_Num'],  errors='coerce')
    serv['Revenue'] = abs(serv['Revenue'])
    serv = serv.query("Service_Num != @last_service_num")
    serv.sort_values(
        by=['Vin_No', 'Service_Date', 'Service_Num'],
        ascending=[True, True, True],
        inplace=True
    )

    reference_date = pd.to_datetime(filterdate) + pd.Timedelta(days=1) 

    # ── Non-PMS (Service_Num == 0) ────────────────────────────────────────────
    non_pms = serv[serv['Service_Num'] == 0].copy()
    non_pms_sorted = non_pms.sort_values('Service_Date')

    last_non_pms_rev = (
        non_pms_sorted
        .groupby('Vin_No', as_index=False)
        .last()[['Vin_No', 'Service_Date', 'Revenue']]
        .rename(columns={'Service_Date': 'Last_NonPMS_Date',
                         'Revenue':      'Last_NonPMS_Revenue'})
    )
    last_non_pms_rev['Days_Since_Last_NonPMS'] = (
        reference_date - last_non_pms_rev['Last_NonPMS_Date']
    ).dt.days

    # ── PMS (Service_Num != 0) ────────────────────────────────────────────────
    pms = serv[serv['Service_Num'] != 0].copy()
    pms_sorted = pms.sort_values('Service_Date')

    last_pms_rev = (
        pms_sorted
        .groupby('Vin_No', as_index=False)
        .last()[['Vin_No', 'Revenue']]
        .rename(columns={'Revenue': 'Last_PMS_Revenue'})
    )

    pms_agg = (
        pms_sorted
        .groupby('Vin_No')['Revenue']
        .agg(
            Max_PMS_Revenue='max',
            Min_PMS_Revenue='min',
            StdDev_PMS_Revenue='std'   # NaN if only 1 visit
        )
        .reset_index()
    )

    # ── Combine ───────────────────────────────────────────────────────────────
    all_vins = pd.DataFrame({'Vin_No': serv['Vin_No'].unique()})

    result = (
        all_vins
        .merge(last_non_pms_rev[['Vin_No',
                                  'Last_NonPMS_Revenue',
                                  'Last_NonPMS_Date',
                                  'Days_Since_Last_NonPMS']],
               on='Vin_No', how='left')
        .merge(last_pms_rev, on='Vin_No', how='left')
        .merge(pms_agg,      on='Vin_No', how='left')
    )
    result = result.drop('Last_NonPMS_Date', axis=1).fillna(0)
    return result


# ── Usage ─────────────────────────────────────────────────────────────────────


def process_service_data(mastersheet: str,servhistory: str, rfm: str, appointshow: str, appointnoshow: str, appointdf: str, digidf: str,vhc: str,servcode: str,filter_date: str,last_service_code: int) -> pd.DataFrame:
    """
    Process service history data from CSV file.
    
    Args:
        mastersheet (str): Path to the master sheet CSV file
        servhistory (str): Path to the service history CSV file
        filter_date (str): Date string in YYYY-MM-DD format
        last_service_code (int): Code for the last service
        
    Returns:
        pd.DataFrame: Processed and filtered service data
    """
    logger.info("Starting service data processing")
    
    try:
        # Convert filter date
        filterdate = pd.to_datetime(filter_date)
        logger.info(f'Filter date: {filterdate}')
        
        # Validate input files
        # if not os.path.exists(mastersheet):
        #     logger.error(f"Master sheet not found: {mastersheet}")
        #     raise FileNotFoundError(f"Master sheet not found: {mastersheet}")
        
        if not os.path.exists(servhistory):
            logger.error(f"Service history file not found: {servhistory}")
            raise FileNotFoundError(f"Service history file not found: {servhistory}")
            
        # Read and process master sheet
        
        try:
            
            logger.info("Reading master sheet data")
            col_name = f"Expected{last_service_code}kDate"
            # mastertrain = pd.read_csv(mastersheet, low_memory=False)
            mastertrain = mastersheet
            logger.debug(f"Master sheet initial shape: {mastertrain.shape}")
            
            
            logger.info("Converting dates in master sheet")
            mastertrain['Last Service Date - PMS'] = pd.to_datetime(mastertrain['Last Service Date - PMS'],format='mixed',dayfirst=True,errors='coerce')
            mastertrain[col_name] = pd.to_datetime(mastertrain[col_name],format='mixed',dayfirst=True)
            mastertrain = mastertrain.query(f"{col_name} <= @filterdate and `Last Service Date - PMS` <= @filterdate")
            logger.info(f"Master sheet shape after date filtering: {mastertrain.shape}")
            servcode_desc = pd.read_csv(servcode)
            servcode_desc = servcode_desc.rename(columns={'SO_CO_CODE': 'Service_Code'})
            # mastertrain.to_csv('validatecode/mastertrain_initial.csv', index=False)
        except Exception as e:
            logger.error(f"Error processing master sheet: {str(e)}")
            raise
        
        # Read and process service history
        try:
            logger.info("Reading service history data")
            serv1 = pd.read_csv(servhistory,low_memory=False,encoding='ISO-8859-1') #Input Service History
            logger.debug(f"Service history initial shape: {serv1.shape}")
            
            logger.info("Processing service history data")
            serv1['Service_Date'] = pd.to_datetime(serv1['Service_Date'],format='mixed',dayfirst=True,errors='coerce')
            invalid_dates = serv1['Service_Date'].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} invalid dates in service history")
            
            serv1 = serv1.query(f"Service_Date <= @filterdate")
            logger.info(f"Service history shape after date filtering: {serv1.shape}")
            
            logger.info("Converting numeric columns")
            serv1['Mileage'] = pd.to_numeric(serv1['Mileage'],errors='coerce')
            serv1['Revenue'] = pd.to_numeric(serv1['Revenue'],errors='coerce')
            
            logger.info("Processing service numbers")
            serv1['Service_Num'] = serv1['Description'].apply(extract_k)
            serv1['Service_Num'] = np.where(serv1['Description'] == '<=10', 10, serv1['Service_Num'])
            
            # Filter and process service data
            logger.info("Filtering and processing service data")
            serv1 = serv1.query(f"Service_Num <= {last_service_code}")
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
            logger.info("Deriving PMS features")
            dfpmsdate = derive_pms_features1(serv1, last_service_code)

            logger.info("Last Non PMS Mileage calculation")
            lastnonpmsmil = get_last_nonpms_mileage(serv1,mastertrain, vin_col="Vin_No", mileage_col="Mileage",
                            date_col="Service_Date")
            
            logger.info("Deriving NPMS features")
            dfnpmsdate = derive_npms_features(serv1)
            
            logger.info("Deriving NonPMS Mileage")
            res1 = get_last_nonpms_before_targetpms(serv1, "Vin_No","Mileage","Service_Date",mastertrain)
            res2 = get_last_nonpms_mileage(serv1, mastertrain,vin_col="Vin_No", mileage_col="Mileage",date_col="Service_Date")
            lastnonpmsmil = pd.concat([res1,res2],axis=0)

            logger.info("Calculating average mileage intervals for PMS")
            avg_mileage_interval = derive_pms_mileage_features(serv1, dfpmsdate, last_service_code)
            
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
            newserv20ka = newserv20k.copy()
            # Unpack the tuple returned from derive_npms_features

            merge_operations = [
                (npms_counts, 'NPMS counts'),
                (Npmsevents, 'NPMS events'),
                (dfpmsdate, 'PMS dates'),
                (dfnpmsdate, 'NPMS dates'),
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
                (complaint_features, 'Complaint features')
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
            finlmlg = pd.DataFrame.from_dict(cleaned, orient='index').reset_index(drop=True)
            finlmlg = finlmlg[['Vin_No','Mileage']]
            pmsfsttime = (pd.DataFrame.from_dict(unclean, orient="index", columns=["Mileage"])
            .reset_index()
            .rename(columns={"index": "Vin_No"}))
            pmsfsttime['Mileage'] = pmsfsttime['Mileage'].fillna(finlmlg['Mileage'].mean())
            lstmiltg = pd.concat([finlmlg,pmsfsttime],axis=0)
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
            vhcdf = pd.read_csv(vhc,encoding="ISO-8859-1",low_memory=False)
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
            # filtered_dfnew.query("no_of_service_appointments_booked == 0 and total_appointments_showed_up > 0").to_csv('validatecode/checknoofserviceappbooked.csv', index=False)
            # ltappoins.to_csv('validatecode/lateappointments60kpred.csv', index=False)
            # appoinstat.to_csv('validatecode/lastappointmentstatus60kpred.csv', index=False)
            filtered_dfnew = filtered_dfnew.drop(['Vehicle Magic'],axis=1)
            # filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Key"].str.split("-", n=1).str[1]
            # filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Magic"].str.split("-", n=1).str[0]
            # filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Magic"].astype(int)

            filtered_dfnew[['last_appointment_delay_days', 'no_of_late_appointments','appointment_booked_showed_up_atleastonce','total_appointments_showed_up','no_of_appointments_booked_but_not_showed_up','Avg_appointment_delay_days']] = filtered_dfnew[['last_appointment_delay_days', 'no_of_late_appointments','appointment_booked_showed_up_atleastonce','total_appointments_showed_up','no_of_appointments_booked_but_not_showed_up','Avg_appointment_delay_days']].fillna(0)
            filtered_dfnew['last_appointment_status'] = filtered_dfnew['last_appointment_status'].fillna('No Appointment')
            #last_appointment_status	last_appointment_delay_days
            # filtered_dfnew.loc[
            #         filtered_dfnew["Service_Num"] == 10,
            #         "Avg_Mileage_Interval_PMS"
            #     ] = (((last_service_code * 1000) - filtered_dfnew.loc[
            #         filtered_dfnew["Service_Num"] == 10, "Last PMS Mileage"
            #     ]) + (filtered_dfnew.loc[
            #         filtered_dfnew["Service_Num"] == 10, "Last PMS Mileage"
            #     ])) /2
            # filtered_dfnew.loc[filtered_dfnew['Service_Num'] == 10, 'Avg_Mileage_Interval_PMS'] = filtered_dfnew.loc[filtered_dfnew['Service_Num'] == 10, 'Avg_Mileage_Interval_PMSold']
            logger.info(f"Processing completed. Final shape: {filtered_dfnew.shape}")
      
            # filtered_dfnew = pd.read_csv('validatecode/finalmerged40k.csv',low_memory=False)
            rfmdf = pd.read_csv(rfm,low_memory=False,encoding='ISO-8859-1') #Input RFM segments file
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
            # nationalitymap.to_csv('validatecode/nationalitymap.csv', index=False)
            # vehmodelmap.to_csv('validatecode/vehmodelmap.csv', index=False)
            # vehvariantmap.to_csv('validatecode/vehvariantmap.csv', index=False)
            # rfmpms.to_csv('validatecode/rfmpms.csv', index=False)
            obint = ['Total Promoter','Total Passive','Total Detractor','Total Survey',
            'CC','Weight','Height','Wheel Base','Service Frequency']
            for i in obint:
                pms[i] = pd.to_numeric(pms[i], errors='coerce')
            pms = pms.drop('Service_Date',axis=1,errors = 'ignore')
            pms[['Months_Since_Last_NPMS','Avg_Mileage_Interval_NPMS']] = pms[['Months_Since_Last_NPMS','Avg_Mileage_Interval_NPMS']].fillna(0)
            colssrvd = pms.pop('Last Service Date - PMS')
            pms.insert(3, colssrvd.name, colssrvd)
            pms['Last Service Date - PMS'] = pd.to_datetime(pms['Last Service Date - PMS'],format='mixed',dayfirst=True,errors='coerce')
            pms[f'Next{last_service_code}K_Due'] = pd.to_datetime(pms[f'Next{last_service_code}K_Due'],format='mixed',dayfirst=True,errors='coerce')
            colsdue = pms.pop(f'Next{last_service_code}K_Due')
            pms.insert(4, colsdue.name, colsdue)
            # Drop high-cardinality columns before encoding (already saved for clustering)
            pms = pms.drop(['Nationality','Model','Variant','RFM_segments'],axis=1,errors='ignore')
            # pms.dtypes.reset_index().rename(
            #     columns={"index": "feature", 0: "dtype"}
            # ).to_csv('validatecode/pms_dtypes.csv', index=False)
            # pms.to_csv('validatecode/pms_before_onehot.csv', index=False)
            # pms.to_csv(f'validatecode/pms_before_onehot{last_service_code}k.csv', index=False)
            # numeric_cols = [
            #     "Survey Score",
            #     # "VHC Quoted",
            #     # "VHC Sold",
            #     # "VHC Lost Sale",
            #     # "VHC Lost Red Sale",
            #     # "VHC Deferred",
            #     # "VHC Amber Deferred"
            # ]

            # pms[numeric_cols] = pms[numeric_cols].apply(
            #     lambda col: pd.to_numeric(col, errors="coerce")
            # )
            # ── Robust one-hot encoding ─────────────────────────────
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
            imputer = IterativeImputer(random_state=0,estimator=Lasso(), max_iter=1)
            pms_imputed = imputer.fit_transform(pmsnew.iloc[:,5:])
            pms_imputed = pd.DataFrame(pms_imputed, columns=pmsnew.columns[5:])
            pms_imputed['VIN'] = pmsnew['VIN'].values
            col1 = pms_imputed.pop("VIN")
            pms_imputed.insert(0, col1.name, col1)
            pms_imputed['Last Service Date - PMS'] = pmsnew['Last Service Date - PMS'].values
            cols2 = pms_imputed.pop('Last Service Date - PMS')
            pms_imputed.insert(1, cols2.name, cols2)
            pms_imputed['Vehicle Key'] = pmsnew['Vehicle Key'].values
            col = pms_imputed.pop("Vehicle Key")
            pms_imputed.insert(2, col.name, col)
            pms_imputed['Customer ID'] = pmsnew['Customer ID'].values
            cols3 = pms_imputed.pop("Customer ID")
            pms_imputed.insert(3, cols3.name, cols3)
            pms_imputed[f'Next{last_service_code}K_Due'] = pmsnew[f'Next{last_service_code}K_Due'].values
            cols4 = pms_imputed.pop(f"Next{last_service_code}K_Due")
            pms_imputed.insert(4, cols4.name, cols4)
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
            coltarg = pms_imputed.pop("TargetFlag")      
            pms_imputed.insert(len(pms_imputed.columns), "TargetFlag", coltarg) 
            coltpmsmil = pms_imputed.pop("Last PMS Mileage")      
            pms_imputed.insert(len(pms_imputed.columns)-1, "Last PMS Mileage", coltpmsmil)
            # pms_imputed.rename(columns={'Avg_Service_Interval_PMS':'Avg_Service_Interval_PMS_per10k'},inplace=True)
            # pms_imputed.rename(columns={'Avg_Service_Interval_PMS1':'Avg_Service_Interval_PMS'},inplace=True)
            pms_imputed = pms_imputed.rename(columns=lambda c: re.sub(r'\.0$', '', c))
            pms_imputed["Last Service Mileage"] = pms_imputed[["Last PMS Mileage", "LastNonPMSMileage"]].max(axis=1)
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

def feature_sel(df: pd.DataFrame):
    X = df.iloc[:,7:-1]
    X = X.loc[:, ~X.columns.str.contains('Other|Service_Num|OTHERS|old|OTHER|Unknown|segments_-', case=False, na=False)]
    y = df['TargetFlag']

    # Compute MI scores
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # Convert to DataFrame for better readability (if X has column names)
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'MIScore': mi_scores
    }).sort_values(by='MIScore', ascending=False)

    # Display the MI scores
    # mi_df = mi_df.query('MIScore > 0.002')
    mi_df['Cumulative_%'] = mi_df['MIScore'].cumsum() / mi_df['MIScore'].sum()
    mi_df = mi_df.query('`Cumulative_%` <= 0.85')

    return list(mi_df['Feature'].values) #list(iv_df.index)

def main():
    """
    Main function to process service history data.
    """
    logger.info("Starting PMS data processing")
    
    
    try:
        # Configuration parameters
        last_service_code = int(sys.argv[1]) if len(sys.argv) > 1 else 70 #Input Target PMS Mileage(e.g., 40 for 40k)
        csv_path1 = f'TestTrain_{last_service_code}k.csv' 
        csv_path2 = 'data/Service History Q1 - 2026.csv'
        csvpath3 = 'data/RFM Segments Q2 - 2025.csv'
        csvpath4 = 'data/Appoinment - Showed Up.csv'
        csvpath6 = 'data/Appoinments - Q3 - 2025.csv'
        csvpath5 = 'data/No Show VINs.csv'
        csvpath7 = 'data/Digital sessions Q3 - 2025.csv'
        csvpath8 = 'data/VHC Q3 - 2025.csv'
        csvpath9 = 'data/Service Code Desc.csv'
        filter_date = '2026-06-30' #Input Date (need to derive based on current date/quarter)
        
        logger.info(f"Configuration - Filter date: {filter_date}, Last service code: {last_service_code}")
        
        masterdata = prepare_pms_datasets(
            df_path="data/EDA - Q3 2025.csv",
            service_history_path="data/Service History Q1 - 2026.csv",
            service_types=[f"{last_service_code}k"],
            selected_quarter="Q2",
            year=2026,
            target_range=(0.45, 0.55),
            output_prefix="TestTrain"
        )
        masterdata = (
            masterdata
            .sort_values(["VIN", "Service_Num"], ascending=[True, False])
            .drop_duplicates(subset="VIN", keep="first")
            .reset_index(drop=True)
        )
        # Process the service data
        # masterdata = '40kTestQ2new.csv'
        result_df = process_service_data(
            mastersheet=masterdata,
            servhistory=csv_path2,
            rfm = csvpath3,
            appointshow=csvpath4,
            appointnoshow=csvpath5,
            appointdf = csvpath6,
            digidf = csvpath7,
            vhc=csvpath8,
            servcode = csvpath9,
            filter_date=filter_date,
            last_service_code=last_service_code
        )

        # feats = result_df.columns[:5].tolist() + feature_sel(result_df)
        # fnlfts = feats + ['TargetFlag'] #OutputFeatures List --- Input to training module
        # TrainMaster = result_df[fnlfts]
        # TrainMaster.to_csv(f'validatecode/PMS{last_service_code}k_Features.csv', index=False) #Output Master Train Data
        # print(f"Selected features ({len(feats)}): {feats}")

        
        # Log results summary
        logger.info(f"Processing completed. Output shape: {result_df.shape}")
        logger.debug("First few rows of processed data:\n%s", result_df.head().to_string())
    
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.error("Error processing data: %s\nTraceback:\n%s", str(e), tb_str)
        return
    
    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()