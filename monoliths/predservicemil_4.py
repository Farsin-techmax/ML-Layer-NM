"""
LEGACY -- kept as reference, not part of the current pipeline.

Superseded by prepare_test_set.py (cohort + labels) and score_milestone.py (scoring). Still the
only script that reads the validatecode/ cluster files (Model_clusters_{m}.csv,
Variant_clusters_{m}.csv, Nationality_clusters_{m}.csv) around line 2307, so validatecode/ is an
input directory, not just old debug output.

WARNING: this module executes its whole pipeline at import time -- importing it runs it.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
import sys
import ast
import re
from collections import Counter
from sklearn.linear_model import Lasso
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
try:
    warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
except Exception:
    try:
        pd.options.mode.chained_assignment = None
    except Exception:
        pass


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

def extract_kk(service_str):
    """Extract numeric from Service History description"""
    try:
        return int(service_str.split('-')[-1])
    except:
        return 0

def extract_k1(service_str):
    """Convert '50k' -> 50"""
    try:
        return int(service_str.replace("k", "").strip())
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

    dfpmsdate = dfpmsdate.merge(first_service, on='Vin_No', how='left')
    dfpmsdate = dfpmsdate.merge(last_service, on='Vin_No', how='left')

    # Step 3: Months difference calculator
    def months_diff(start_date, end_date):
        rd = relativedelta(end_date, start_date)
        return rd.years * 12 + rd.months

    # Step 4: Create features
    dfpmsdate['Months_Since_First_PMS'] = dfpmsdate['First_PMS_Date'].apply(lambda d: months_diff(d, reference_date))
    dfpmsdate['Months_Since_Last_PMS'] = dfpmsdate['Last_PMS_Date'].apply(lambda d: months_diff(d, reference_date))

    # Step 5: Aggregate at VIN level
    dfpmsdate = dfpmsdate.groupby(['Vin_No']).agg(
        Months_Since_First_PMS=('Months_Since_First_PMS', 'mean'),
        Months_Since_Last_PMS=('Months_Since_Last_PMS', 'mean')
    ).reset_index()

    # Step 6: Flag for single PMS
    dfpmsdate['SinglePMS'] = (dfpmsdate['Months_Since_First_PMS'] == dfpmsdate['Months_Since_Last_PMS']).astype(int)

    return dfpmsdate[['Vin_No','Months_Since_Last_PMS', 'SinglePMS']]

def derive_npms_features(serv1: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
    """
    Derive Non-PMS (Service_Num == 0) related features per VIN.

    Parameters
    ----------
    serv1 : pd.DataFrame
        Input dataframe with columns ['Vin_No', 'Service_Date', 'Service_Num'].
    reference_date : datetime, optional
        Reference date for calculating months difference. Defaults to today.

    Returns
    -------
    pd.DataFrame
        Aggregated features per VIN:
        - Months_Since_First_NPMS
        - Months_Since_Last_NPMS
    """

    if reference_date is None:
        reference_date = pd.Timestamp.today()

    # Step 1: Filter Non-PMS services
    dfnpmsdate = serv1.sort_values(by=['Vin_No', 'Service_Date']).copy()
    dfnpmsdate = dfnpmsdate.query('Service_Num == 0')

    # Step 2: First and last Non-PMS date per VIN
    first_service = dfnpmsdate.groupby('Vin_No')['Service_Date'].min().reset_index(name='First_NPMS_Date')
    last_service = dfnpmsdate.groupby('Vin_No')['Service_Date'].max().reset_index(name='Last_NPMS_Date')

    dfnpmsdate = dfnpmsdate.merge(first_service, on='Vin_No', how='left')
    dfnpmsdate = dfnpmsdate.merge(last_service, on='Vin_No', how='left')

    # Step 3: Months difference calculator
    def months_diff(start_date, end_date):
        rd = relativedelta(end_date, start_date)
        return rd.years * 12 + rd.months

    # Step 4: Create features
    dfnpmsdate['Months_Since_First_NPMS'] = dfnpmsdate['First_NPMS_Date'].apply(lambda d: months_diff(d, reference_date))
    dfnpmsdate['Months_Since_Last_NPMS'] = dfnpmsdate['Last_NPMS_Date'].apply(lambda d: months_diff(d, reference_date))

    # Step 5: Aggregate at VIN level
    dfnpmsdate = dfnpmsdate.groupby(['Vin_No']).agg(
        Months_Since_First_NPMS=('Months_Since_First_NPMS', 'mean'),
        Months_Since_Last_NPMS=('Months_Since_Last_NPMS', 'mean')
    ).reset_index()

    return dfnpmsdate[['Vin_No', 'Months_Since_Last_NPMS']]

def derive_pms_mileage_features(serv1: pd.DataFrame, dfpmsdate: pd.DataFrame,last_service_code: int, svc:int) -> pd.DataFrame:
    """
    Derive PMS mileage-based features per VIN.
    If SinglePMS == 1, replace its interval with the mean for the same Service_Num
    among VINs where SinglePMS == 0.
    """

    # Step 1: Prepare PMS mileage data
    dfpmsmil = serv1.sort_values(by=['Vin_No', 'Service_Date', 'Mileage']).copy()
    dfpmsmil['Service_Date'] = pd.to_datetime(dfpmsmil['Service_Date'], format='mixed', dayfirst=True, errors='coerce')
    dfpmsmil['Mileage'] = pd.to_numeric(dfpmsmil['Mileage'], errors='coerce')
    dfpmsmil = dfpmsmil.query('Service_Num > 0 and Service_Num <= @svc')

    # Step 2: Define function for average mileage interval per VIN & Service_Num
    def avg_mile_interval(group):
        m = group.sort_values('Service_Date')['Mileage']
        intervals = m.diff().fillna(m)  # first interval = first mileage
        return intervals.mean()

    # Step 3: Apply per VIN + Service_Num
    avg_mileage_interval = (
        dfpmsmil
        .groupby(['Vin_No'])
        .apply(avg_mile_interval)
        .reset_index(name='Avg_Mileage_Interval_PMS')
    )

    # Step 4: Merge with SinglePMS flag (from dfpmsdate)
    avg_mileage_interval = avg_mileage_interval.merge(
        dfpmsdate[['Vin_No', 'SinglePMS']],
        on='Vin_No',
        how='left'
    )
    # Step 5: Replace SinglePMS == 1 values dynamically with Service_Num means
    df_max_service = dfpmsmil.groupby("Vin_No", as_index=False)["Service_Num"].max()
    avg_mileage_interval = avg_mileage_interval.merge(df_max_service, on='Vin_No', how='left')
    replacement_means = (
        avg_mileage_interval.query("SinglePMS == 0")
        .groupby('Service_Num')['Avg_Mileage_Interval_PMS']
        .mean()
    )
    def replace_if_single(row):
        if row['SinglePMS'] == 1:
            return replacement_means.get(row['Service_Num'], row['Avg_Mileage_Interval_PMS'])
        return row['Avg_Mileage_Interval_PMS']

    avg_mileage_interval['Avg_Mileage_Interval_PMS'] = avg_mileage_interval.apply(replace_if_single, axis=1)
    # avg_mileage_interval.loc[
    #         (avg_mileage_interval["Service_Num"] == 10) , 
    #         'Avg_Mileage_Interval_PMS'
    #     ] = (last_service_code-10)*1000
    
    if svc == 1:
        # avg_mileage_interval.to_csv(f'validatecode/avg_mileage_interval{svc}kpred.csv', index=False)
        return avg_mileage_interval.drop(columns=['SinglePMS','Service_Num'])   
    else:
        avg_mileage_interval['skipped_blocks'] = ((last_service_code - 10 - avg_mileage_interval['Service_Num']) // 10).clip(lower=0)
        avg_mileage_interval['skipped_blocks'] = np.where(avg_mileage_interval['skipped_blocks'] == 1, 2, avg_mileage_interval['skipped_blocks'])
        avg_mileage_interval['predicted_interval'] = avg_mileage_interval['Avg_Mileage_Interval_PMS'] * avg_mileage_interval['skipped_blocks']
        avg_mileage_interval.loc[avg_mileage_interval['skipped_blocks'] == 0, 'predicted_interval'] = avg_mileage_interval.loc[avg_mileage_interval['skipped_blocks'] == 0, 'Avg_Mileage_Interval_PMS']
        # avg_mileage_interval.loc[avg_mileage_interval['Service_Num'] == 10, 'predicted_interval'] = avg_mileage_interval.loc[avg_mileage_interval['Service_Num'] == 10, 'Avg_Mileage_Interval_PMS']
        avg_mileage_interval = avg_mileage_interval.rename(columns={'Avg_Mileage_Interval_PMS':'Avg_Mileage_Interval_PMSold','predicted_interval':'Avg_Mileage_Interval_PMS'})
        # avg_mileage_interval.to_csv(f'validatecode/avg_mileage_interval{svc}kpred.csv', index=False)
        avg_mileage_interval = avg_mileage_interval.drop(columns=['Service_Num','SinglePMS','skipped_blocks'],axis=1)
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

    # Step 1: Filter Non-PMS services
    dfnpmsmil = serv1.query('Service_Num == 0').copy()
    dfnpmsmil = dfnpmsmil.sort_values(by=['Vin_No', 'Service_Date'])
    # dfnpmsmil.to_csv('validatecode/dfnpmsmil.csv', index=False)
    # Step 2: Function to compute avg mileage interval per VIN
    def avg_mile_interval(group):
        m = group.sort_values('Service_Date')['Mileage']
        intervals = m.diff().fillna(m)  # first interval = first mileage
        return intervals.mean()
    # dfnpmsmil.to_csv('validatecode/dfnpmsmilissue.csv', index=False)
    # Step 3: Apply per VIN
    if dfnpmsmil.empty:
        avg_mileage_interval_non_pms = pd.DataFrame(
            columns=['Vin_No', 'Avg_Mileage_Interval_NPMS']
        )
    else:
        avg_mileage_interval_non_pms = (
            dfnpmsmil
            .groupby('Vin_No')
            .apply(avg_mile_interval)
            .reset_index(name='Avg_Mileage_Interval_NPMS')
        )
    # avg_mileage_interval_non_pms = (
    #     dfnpmsmil
    #     .groupby('Vin_No')
    #     .apply(avg_mile_interval)
    #     .reset_index(name='Avg_Mileage_Interval_NPMS')
    # )

    return avg_mileage_interval_non_pms

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
    # Step 3: Functions
    def avg_month_interval1(group):
        dates = group.sort_values('Service_Date')['Service_Date']
        intervals = dates.diff().dropna().dt.days / 30.44  # months
        return intervals.mean()

    def avg_month_interval(group):
        group = group.sort_values('Service_Date')
        delta_months = group['Service_Date'].diff().dropna().dt.days / 30.44
        delta_kms = group['Mileage'].diff().dropna()
        intervals_per_10k = delta_months / (delta_kms / 10000)
        return intervals_per_10k.mean()

    # Step 4: Calculate both versions
    avg_monthly_interval11 = (
        service_max_dates
        .groupby('Vin_No')
        .apply(avg_month_interval1)
        .reset_index(name='Avg_Service_Interval_PMS1')
    )

    avg_monthly_interval = (
        service_max_dates
        .groupby('Vin_No')
        .apply(avg_month_interval)
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

# def adjust_service_intervalspred(df: pd.DataFrame, last_service) -> pd.DataFrame:
#     """
#     Optimized version of adjust_service_intervals.
#     Adds rule:
#         If Service_Num == 10 → Multiplier = (last_service - 20) / 10
#     """   
#     cols = ["Avg_Service_Interval_PMS", "Avg_Service_Interval_PMS1"]
    
#     # Step 1: Replace inf with NaN
#     df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)

#     # Step 2: masks
#     mask_10 = df["Service_Num"] == 10
#     mask_20 = df["Service_Num"] == 20
#     mask_30_plus = (df["Service_Num"] >= 30) & (df["Service_Num"] < last_service)
#     mask_others = ~(mask_10 | mask_20 | mask_30_plus)

#     # Step 3: bounds
#     for col in cols:
#         df.loc[mask_20, col] = df.loc[mask_20, col].where(df.loc[mask_20, col].between(0.5, 60))
#         df.loc[mask_30_plus, col] = df.loc[mask_30_plus, col].where(df.loc[mask_30_plus, col].between(0.5, 60))
#         df.loc[mask_others, col] = df.loc[mask_others, col].where(df.loc[mask_others, col].between(0.5, 60))

#     # Step 4: Fill missing with Service_Num group mean
#     for col in cols:
#         df[col] = df.groupby("Service_Num")[col].transform(lambda x: x.fillna(x.mean()))

#     # Store last service
#     df["Last_Service"] = last_service

#     # Minimum service mileage
#     min_service = 10

#     # Initialize multiplier
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

#     # ⭐ NEW RULE: Service_Num == 10
#     df.loc[df["Service_Num"] == 10, "Multiplier"] = (last_service - 20) / 10

#     # Compute new intervals
#     df['Avg_Service_Interval_PMSnew'] = df['Avg_Service_Interval_PMS'] * df['Multiplier']
#     df['Avg_Service_Interval_PMS1new'] = df['Avg_Service_Interval_PMS1'] * df['Multiplier']

#     return df[[
#         "Vin_No", "Service_Num",
#         "Avg_Service_Interval_PMS", "Avg_Service_Interval_PMS1",
#         "Avg_Service_Interval_PMSnew", "Avg_Service_Interval_PMS1new",
#         "Multiplier"
#     ]]

def adjust_service_intervalspred(
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
            "Multiplier",
            "Avg_Service_Interval_PMS",
            "Avg_Service_Interval_PMS1",
            "Avg_Service_Interval_PMSnew",
            "Avg_Service_Interval_PMS1new",
        ]
    ]

def get_last_nonpms_mileagepred(df_nonpms,svc,mastertrain, vin_col="Vin", mileage_col="Mileage",
                            date_col="Service Date"):
    """
    Returns the last NON-PMS mileage for each VIN.
    """
    lastpmsvins = mastertrain['VIN'].unique()
    last_pms_df = mastertrain[['VIN','Last PMS Mileage']]
    last_pms_df.rename(columns={'VIN':'Vin_No'},inplace=True)
    # Convert service date to datetime if needed
    df_nonpms[date_col] = pd.to_datetime(df_nonpms[date_col])
    df_nonpms = df_nonpms.query("Service_Num <= @svc")
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
    serv1['Revenue'] = abs(serv1['Revenue'])  # Ensure revenue is positive
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

def compute_service_features(serv, filterdate):
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

    # ── Prep ──────────────────────────────────────────────────────────────────
    serv = serv.copy()
    serv['Service_Date'] = pd.to_datetime(serv['Service_Date'])
    serv['Revenue']      = pd.to_numeric(serv['Revenue'],      errors='coerce')
    serv['Service_Num']  = pd.to_numeric(serv['Service_Num'],  errors='coerce')
    serv['Revenue'] = abs(serv['Revenue'])
    # serv = serv.query("Service_Num != @last_service_num")
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
    dfbrchdate = dfbrchdate.query(f"Service_Num <= @last_service_code")
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
    df_sorted = df_sorted.query(f"Service_Num <= @last_service_code")
    # Group and collect unique branches
    branch_grouped = (
        df_sorted.groupby('Vin_No')['Service_Branch_Name']
        .apply(lambda x: list(set(x)))
        .reset_index(name='Branch_List')
    )
    
    # Count number of unique branches
    branch_grouped['unique_branch_serviced'] = branch_grouped['Branch_List'].apply(len)
    
    return branch_grouped

def calc_next_due(last_date, interval_months):
    if pd.isna(last_date) or pd.isna(interval_months):
        return pd.NaT
    
    whole_months = int(interval_months)
    fraction = interval_months - whole_months
    
    # Add whole months
    next_due = last_date + relativedelta(months=whole_months)
    
    # Add fractional remainder as days (based on calendar month length)
    days_in_month = pd.Period(next_due.strftime("%Y-%m")).days_in_month
    next_due += pd.Timedelta(days=fraction * days_in_month)

    # ✅ Return date in YYYY-MM-DD format
    return next_due.strftime("%Y-%m-%d")

def calculatenxt_service_interval(dfaa, target_mileage):
    """
    Compute service interval using:
    interval = avg_interval_months * (target_service_no - last_service_no)

    Additional rule:
    - If Service_Num == 10, predicted_interval_months = Avg_Service_Interval_PMS1

    Assumptions:
    - 10,000 km = 1 service unit
    - df contains:
        - 'Service_Num'
        - 'Avg_Service_Interval_PMS1'
    """

    # Convert to service numbers (each "10" corresponds to 10k)
    dfaa['last_service_no'] = dfaa['Service_Num'] / 10
    target_service_no = target_mileage / 10

    # Base calculation
    dfaa['predicted_interval_months'] = (
        dfaa['Avg_Service_Interval_PMS1'] * (target_service_no - dfaa['last_service_no'])
    )

    # Apply the special condition
    # dfaa.loc[dfaa['Service_Num'] == 10, 'predicted_interval_months'] = dfaa['Avg_Service_Interval_PMS1']

    return dfaa[['Vin_No','predicted_interval_months']]

def calculate_revenue_spend(df,last_service):
    df = df.copy()
    
    # Convert service columns (10k, 20k, ...) to numeric
    service_cols = [col for col in df.columns if col.endswith("k")]
    df[service_cols] = df[service_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    revenue_spend = []
    for _, row in df.iterrows():
               
        # Build the list of service columns up to last_service
        cols_to_sum = [f"{i}K" for i in range(10, last_service -10 + 1, 10) if f"{i}K" in df.columns]

        total = row[cols_to_sum].sum() if cols_to_sum else 0
        revenue_spend.append(total)

    df["PMSRevenue"] = revenue_spend
    return df

def one_hot(df):
    cat_cols = df.select_dtypes(include='O').keys().tolist()
    cat_data = df[cat_cols[3:]]
    for column in cat_data.columns[:]:
        tempdf = pd.get_dummies(df[column], prefix=column,dtype=int)
        df_new = pd.merge(left=df,right=tempdf,left_index=True,right_index=True)
        df = df_new.drop(columns=column)
#     else:
#         df = df
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

def normalize_to_list(val):
    """
    Convert val into a Python list of strings, handling:
    - actual lists
    - string representations like "['France', 'India']"
    - comma/semicolon-separated strings "France, India" or "France; India"
    - NaN / None -> empty list
    """
    if pd.isna(val):
        return []
    # already a list
    if isinstance(val, list):
        return val
    # string that looks like a Python list
    if isinstance(val, str):
        s = val.strip()
        # try python literal_eval for strings like "['A','B']"
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
            try:
                parsed = ast.literal_eval(s)
                # ensure it's a list
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed]
            except Exception:
                pass
        # try JSON-ish list
        if s.startswith('"[') and s.endswith(']"'):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed]
            except Exception:
                pass
        # comma- or semicolon-separated fallback
        if ',' in s:
            return [x.strip() for x in s.split(',') if x.strip()]
        if ';' in s:
            return [x.strip() for x in s.split(';') if x.strip()]
        # otherwise treat single string as single-item list
        return [s]
    # anything else: fallback to single-item str
    return [str(val)]

def map_cluster(df_vehicles, df_clusters,
                            vehicle_nat_col,
                            cluster_nat_col,
                            cluster_id_col):
    # Make copies so we don't mutate inputs
    df2 = df_clusters.copy()
    df1 = df_vehicles.copy()

    # Normalize cluster nationality column to lists
    df2[cluster_nat_col] = df2[cluster_nat_col].apply(normalize_to_list)

    # Explode
    df2_exploded = df2.explode(cluster_nat_col).reset_index(drop=True)

    # Optional: strip and standardize case for matching
    df2_exploded[cluster_nat_col] = df2_exploded[cluster_nat_col].astype(str).str.strip()
    df1[vehicle_nat_col] = df1[vehicle_nat_col].astype(str).str.strip()

    # Merge left: keep all vehicles, bring in cluster id
    merged = df1.merge(df2_exploded[[cluster_id_col, cluster_nat_col]],
                       left_on=vehicle_nat_col,
                       right_on=cluster_nat_col,
                       how='left')

    # If you want a single cluster value per vehicle and there are duplicates, take first
    merged = merged.drop(columns=[cluster_nat_col])
    return merged

def months_to_days(months, start_date=datetime.today()):
    end_date = start_date + relativedelta(months=months)
    return (end_date - start_date).days

def map_vhc_history(main_df, history_df,svc):

    history_df = history_df.copy()
    history_df = history_df.query("Service_Num <= @svc")
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

    # Merge into main data
    merged = main_df.merge(df_out, on=["VIN", "Service_Num"], how="left")

    return merged

def get_non_pms_events(serv, servcode_desc, mastertrain,svc):
    """
    Parameters:
        serv         : DataFrame - service records
        servcode_desc: dict      - {Service_Code: Description}
        mastertrain  : DataFrame - master training data with Vehicle_Key
    
    Returns:
        Npmsevents   : DataFrame - Vehicle_Key with Last_Service_Non_PMS_Flag 
                                   and OHE of last Non-PMS event types
    """

    # --- Prep ---
    servcode_desc = servcode_desc.dropna()
    servcode_desc = dict(zip(servcode_desc['Service_Code'], servcode_desc['SO_CO_DESCRIPN_001']))
    serv = serv.copy()
    serv['Service_Date'] = pd.to_datetime(
        serv['Service_Date'], format="mixed", dayfirst=True, errors="coerce"
    )

    # Filter to vehicles in mastertrain
    veh = list(mastertrain['Vehicle Key'].unique())

    # Sort and exclude irrelevant service codes
    exclude_codes = {
        'ADP', 'BR4', 'CON', 'CRE', 'EXC', 'ME', 'BES', 'MES', 'C01',
        'NS4', 'PNA', 'VAS', 'VAT', 'VCP', 'VHP'
    }
    serv = serv.sort_values(by=['Vin_No', 'Service_Date'], ascending=True)
    serv = serv.query("Service_Code not in @exclude_codes").reset_index(drop=True)
    serv = serv[serv['Vehicle_Key'].isin(veh)]

    # --- Filter records before first 40k service ---
    service_40k = (
        serv[serv['Service_Num'] == svc]
        .sort_values('Service_Date')
        .groupby('Vin_No')['Service_Date']
        .first()
        .reset_index()
        .rename(columns={'Service_Date': f'First_{svc}k_Date'})
    )
    serv = serv.merge(service_40k, on='Vin_No', how='left')

    filtered_df = serv[
        (serv[f'First_{svc}k_Date'].isna()) |
        (serv['Service_Date'] <= serv[f'First_{svc}k_Date'])
    ].copy()
    filtered_df.drop(columns=[f'First_{svc}k_Date'], inplace=True)

    # --- Identify Non-PMS events ---
    filtered_df['Is_Non_PMS'] = (
        (filtered_df['Service_Num'] == 0) &
        (~filtered_df['Service_Code'].astype(str).str.contains(r'\*', na=False))
    )

    # --- Last service record per vehicle ---
    filtered_df = filtered_df.sort_values(['Vehicle_Key', 'Service_Date'])
    lstnpmseventflag = filtered_df.groupby('Vehicle_Key').tail(1)

    # --- Last Non-PMS event mapping ---
    last_non_pms = lstnpmseventflag[lstnpmseventflag['Is_Non_PMS'].values].copy()
    last_non_pms['Last_NonPMS_Event'] = last_non_pms['Service_Code']

    top_10_events = (
        last_non_pms['Last_NonPMS_Event']
        .value_counts()
        .head(last_non_pms['Last_NonPMS_Event'].nunique())
        .index
    )
    last_non_pms['Event_Mapped'] = last_non_pms['Last_NonPMS_Event'].apply(
        lambda x: x if x in top_10_events else 'Others'
    )
    last_non_pms['Event_Npms'] = last_non_pms['Event_Mapped'].map(servcode_desc)

    # --- Flag column ---
    lstnpmseventflag = lstnpmseventflag.copy()
    lstnpmseventflag['Last_Service_Non_PMS_Flag'] = lstnpmseventflag['Is_Non_PMS'].astype(int)
    lstnpmseventflag = lstnpmseventflag[['Vin_No', 'Last_Service_Non_PMS_Flag']]

    # --- One-hot encode last Non-PMS event ---
    event_ohe = pd.get_dummies(last_non_pms['Event_Npms'], prefix='LastNPMS_Event', dtype=int)
    final_df = pd.concat([last_non_pms[['Vin_No']], event_ohe], axis=1)

    # --- Final merge ---
    Npmsevents = lstnpmseventflag.merge(final_df, on='Vin_No', how='left').fillna(0)

    non_pms_count = (
    filtered_df[filtered_df['Is_Non_PMS']]
    .groupby(['Vin_No','Service_Date'])
    .size()
    .reset_index(name='nNPMS')
    )
    non_pms_countn = non_pms_count.groupby(['Vin_No']).size().reset_index(name='nNPMS')
    return Npmsevents,non_pms_countn

def compute_nonpms_appointment_metrics(
    appt_df,
    service_df
):
    """
    Computes appointment metrics for NON-PMS vehicles only
    (vehicles with Service_Num == 0 only).
    """

    appt_df = appt_df.copy()
    service_df = service_df.copy()
    service_df["Vehicle Magic"] = service_df["Vehicle_Key"].str.split("-", n=1).str[1]
    service_df["Vehicle Magic"] = service_df["Vehicle Magic"].str.split("-", n=1).str[0]
    service_df["Vehicle Magic"] = service_df["Vehicle Magic"].astype(int)
    # ----------------------------
    # Datetime conversions
    # ----------------------------
    appt_df = appt_df[appt_df['Vehicle Magic'].isin(service_df['Vehicle Magic'].unique())]

    appt_df["Due Date IN"] = pd.to_datetime(
        appt_df["Due Date IN"], errors="coerce", dayfirst=True,format='mixed'
    )
    service_df["Service Date"] = pd.to_datetime(
        service_df["Service_Date"], errors="coerce", dayfirst=True,format='mixed'
    )
    appt_df["Service_Num"] = appt_df["WIP_SERVCODE"].apply(derive_servcode)
    appt_df = appt_df.rename(columns={'WIP_SERVCODE':'Service_Code'})
    # ----------------------------
    # Keep only Non-PMS records
    # ----------------------------
    appt_df = appt_df[appt_df["Service_Num"] == 0]
    service_df = service_df[service_df["Service_Num"] == 0]

    # ----------------------------
    # Deduplicate service history
    # (one record per appointment)
    # ----------------------------
    service_dedup = (
        service_df
        .groupby(["Vehicle Magic", "Service_Code"], as_index=False)
        .agg({"Service_Date": "min"})
    )

    events = []

    # ----------------------------
    # SHOWED-UP APPOINTMENTS
    # ----------------------------
    showed_up = appt_df[appt_df["No Show"] == 0]
    
    showed_merged = showed_up.merge(
        service_dedup,
        on=["Vehicle Magic", "Service_Code"],
        how="left"
    )

    showed_merged["Service_Date"] = pd.to_datetime(
        showed_merged["Service_Date"], errors="coerce", dayfirst=True,format='mixed')
    
    showed_merged["Due Date IN"] = pd.to_datetime(
        showed_merged["Due Date IN"], errors="coerce", dayfirst=True,format='mixed')

    showed_merged["delay_days"] = (
        showed_merged["Service_Date"] - showed_merged["Due Date IN"]
    ).dt.days

    for _, r in showed_merged.iterrows():
        if pd.isna(r["Service_Date"]):
            continue

        status = (
            "LATE_SHOW"
            if r["delay_days"] > 0
            else "IN_DUE_TIME"
        )

        events.append({
            "Vehicle Magic": r["Vehicle Magic"],
            "event_date": r["Service_Date"],
            "appointment_status": status,
            "delay_days": max(r["delay_days"], 0)
        })

    # ----------------------------
    # NO-SHOW APPOINTMENTS
    # ----------------------------
    no_show = appt_df[appt_df["No Show"] == 1]

    for _, r in no_show.iterrows():
        events.append({
            "Vehicle Magic": r["Vehicle Magic"],
            "event_date": r["Due Date IN"],
            "appointment_status": "NO_SHOW",
            "delay_days": np.nan
        })

    events_df = pd.DataFrame(events)

    # ----------------------------
    # Aggregate metrics per Vehicle Magic
    # ----------------------------
    results = []

    for vm, grp in events_df.groupby("Vehicle Magic"):
        late_delays = grp.loc[
            grp["appointment_status"] == "LATE_SHOW",
            "delay_days"
        ]

        last_event = grp.sort_values("event_date").iloc[-1]

        results.append({
            "Vehicle Magic": vm,
            "no_of_late_appointments": len(late_delays),
            "Avg_appointment_delay_days": (
                late_delays.mean() if len(late_delays) > 0 else np.nan
            ),
            "last_appointment_status": last_event["appointment_status"],
            "last_appointment_delay_days": last_event["delay_days"]
        })

    return pd.DataFrame(results)

def map_appointments_to_services(appointments_df, service_df, svc):
    # Ensure datetime formats
    appointments_df = appointments_df.copy()
    service_df = service_df.copy()
    service_df = service_df.query("Service_Num <= @svc")
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

def build_final_pms_appointment_summary(appt_df, service_df):
    # Step 1: get last PMS service per VIN
    last_pms = service_df.rename(columns={'VIN':'Vin_No'})

    # Merge appointments with last PMS value
    merged = appt_df.merge(last_pms, on="Vin_No", how="left")

    results = []

    for vin, group in merged.groupby("Vin_No"):
        pms_service = group["Service_Num"].iloc[0]

        # Flag if last PMS service appears in appointment windows
        showed_up_for_last_pms = int(
            group["matched_service_nums"]
            .apply(lambda lst: pms_service in lst)
            .any()
        )

        # Total appointments where showed up until last PMS service
        total_showed_up = int(group["appointment_booked_showed_up"].sum())

        results.append({
            "Vin_No": vin,
            "PMS_Service": pms_service,
            "appointment_booked_showed_up": showed_up_for_last_pms,
            "total_appointments_showed_up": total_showed_up
        })

    return pd.DataFrame(results)

def compute_no_show_appointments_test(
    master_df,
    noshow_df,
    filter_date,
    vin_col="Vin_No",
    wip_deleted_col="WIP Deleted"
):
    """
    Computes number of NO-SHOW appointments per VIN for TEST DATA.

    Logic:
    - NO targetflag logic.
    - NO last PMS service cutoff.
    - All VINs use the SAME global cutoff: filter_date.
    - Count appointments where WIP_Deleted_Date <= filter_date.
    """

    master_df = master_df.copy()
    noshow_df = noshow_df.copy()

    # Normalize VIN column
    if "VEHICLE CHASSIS" in noshow_df.columns:
        noshow_df.rename(columns={"VEHICLE CHASSIS": "VIN"}, inplace=True)

    # Keep only VINs that exist in master_df
    noshow_df = noshow_df[noshow_df["VIN"].isin(master_df[vin_col].unique())]

    # Convert to datetime
    noshow_df[wip_deleted_col] = pd.to_datetime(noshow_df[wip_deleted_col], format="mixed", dayfirst=True)
    filter_date = pd.to_datetime(filter_date)

    # Keep only deleted appointments (No-Shows)
    noshow_df = noshow_df[noshow_df[wip_deleted_col].notna()]

    # Only consider no-shows before or on cutoff date
    valid_noshows = noshow_df[noshow_df[wip_deleted_col] <= filter_date]

    # Count per VIN
    noshow_counts = (
        valid_noshows.groupby("VIN")
        .size()
        .reset_index(name="no_of_appointments_booked_but_not_showed_up")
    )

    # Merge result back into master_df
    final_df = master_df.merge(
        noshow_counts,
        left_on=vin_col,
        right_on="VIN",
        how="left"
    )

    final_df["no_of_appointments_booked_but_not_showed_up"] = (
        final_df["no_of_appointments_booked_but_not_showed_up"]
        .fillna(0)
        .astype(int)
    )
    final_df.rename(columns={"VIN": "Vin_No"}, inplace=True)

    # Always return standardized columns
    return final_df[['Vin_No', "no_of_appointments_booked_but_not_showed_up"]]

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

def compute_late_appointment_metrics(appt_df, service_df,filter_date,svc):
    """
    Computes:
      1) Number of late appointments per Vehicle Magic
      2) Median delay days for late appointments per Vehicle Magic
    """

    appt_df = appt_df.copy()
    service_df = service_df.copy()
    if svc != 0:
        service_df = service_df.query("Service_Num > 1 and Service_Num <= @svc")
    else:
        pass
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
    if svc != 0:
        appt_df = appt_df.query("`Due Date IN` <= @filter_date and  Service_Num > 1")
    else:
        appt_df = appt_df.query("`Due Date IN` <= @filter_date")

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

def derive_servcode(value):
    try:
        num = int(value)  # convert to integer (handles leading zeros)
        return num if num % 10 == 0 else 0
    except ValueError:
        return 0  # non-numeric values

def compute_last_appointment_status_with_constant_service_code(
    appt_df,
    service_df,
    svc,
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
    if svc != 0:
        service_df = service_df.query("Service_Num > 1 and Service_Num <= @svc")
    else:
        pass

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
    if svc != 0:
        appt_df = appt_df.query("`Due Date IN` <= @filter_date and  Service_Num > 1")
    else:
        appt_df = appt_df.query("`Due Date IN` <= @filter_date")

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

def compute_nonpms_appointment_metrics(
    appt_df,
    service_df
):
    """
    Computes appointment metrics for NON-PMS vehicles only
    (vehicles with Service_Num == 0 only).
    """

    appt_df = appt_df.copy()
    service_df = service_df.copy()
    service_df["Vehicle Magic"] = service_df["VehMagic WIP"].str.split("-", n=1).str[0]
    service_df["Vehicle Magic"] = service_df["Vehicle Magic"].astype(int)
    # ----------------------------
    # Datetime conversions
    # ----------------------------
    appt_df["Due Date IN"] = pd.to_datetime(
        appt_df["Due Date IN"], errors="coerce", dayfirst=True,format='mixed'
    )
    service_df["Service Date"] = pd.to_datetime(
        service_df["Service_Date"], errors="coerce", dayfirst=True,format='mixed'
    )
    appt_df["Service_Num"] = appt_df["WIP_SERVCODE"].apply(derive_servcode)
    # ----------------------------
    # Keep only Non-PMS records
    # ----------------------------
    appt_df = appt_df[appt_df["Service_Num"] == 0]
    service_df = service_df[service_df["Service_Num"] == 0]

    # ----------------------------
    # Deduplicate service history
    # (one record per appointment)
    # ----------------------------
    service_dedup = (
        service_df
        .groupby(["Vehicle Magic", "Service_Num"], as_index=False)
        .agg({"Service_Date": "min"})
    )

    events = []

    # ----------------------------
    # SHOWED-UP APPOINTMENTS
    # ----------------------------
    showed_up = appt_df[appt_df["No Show"] == 0]
    
    showed_merged = showed_up.merge(
        service_dedup,
        on=["Vehicle Magic", "Service_Num"],
        how="left"
    )

    showed_merged["Service_Date"] = pd.to_datetime(
        showed_merged["Service_Date"], errors="coerce", dayfirst=True,format='mixed')
    
    showed_merged["Due Date IN"] = pd.to_datetime(
        showed_merged["Due Date IN"], errors="coerce", dayfirst=True,format='mixed')

    showed_merged["delay_days"] = (
        showed_merged["Service_Date"] - showed_merged["Due Date IN"]
    ).dt.days

    for _, r in showed_merged.iterrows():
        if pd.isna(r["Service_Date"]):
            continue

        status = (
            "LATE_SHOW"
            if r["delay_days"] > 0
            else "IN_DUE_TIME"
        )

        events.append({
            "Vehicle Magic": r["Vehicle Magic"],
            "event_date": r["Service_Date"],
            "appointment_status": status,
            "delay_days": max(r["delay_days"], 0)
        })

    # ----------------------------
    # NO-SHOW APPOINTMENTS
    # ----------------------------
    no_show = appt_df[appt_df["No Show"] == 1]

    for _, r in no_show.iterrows():
        events.append({
            "Vehicle Magic": r["Vehicle Magic"],
            "event_date": r["Due Date IN"],
            "appointment_status": "NO_SHOW",
            "delay_days": np.nan
        })

    events_df = pd.DataFrame(events)

    # ----------------------------
    # Aggregate metrics per Vehicle Magic
    # ----------------------------
    results = []

    for vm, grp in events_df.groupby("Vehicle Magic"):
        late_delays = grp.loc[
            grp["appointment_status"] == "LATE_SHOW",
            "delay_days"
        ]

        last_event = grp.sort_values("event_date").iloc[-1]

        results.append({
            "Vehicle Magic": vm,
            "no_of_late_appointments": len(late_delays),
            "Avg_appointment_delay_days": (
                late_delays.median() if len(late_delays) > 0 else np.nan
            ),
            "last_appointment_status": last_event["appointment_status"],
            "last_appointment_delay_days": last_event["delay_days"]
        })

    return pd.DataFrame(results)

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

def NonPMS(rfmdf,filterdate,last_service_code,custom_date,natclus,varclus,modclus,appoinshow,appoinoshow,appointdfN,servcode_desc):
    eda = pd.read_csv('EDA Datasheet Till 2025.csv',low_memory=False,encoding='ISO-8859-1')
    print(f'''EDA shape before filtering PMS: {eda.shape}''')
    serv1 = pd.read_csv('Service History Feb 2026.csv',low_memory=False)
    serv1['Service_Date'] = pd.to_datetime(serv1['Service_Date'],format='mixed',dayfirst=True,errors='coerce')
    serv1 = serv1.query("Service_Date <= @filterdate")
    serv = serv1.query("Description != 'Others'")
    neweda = eda[
    (eda['New / Used Category'] == 'New Only') &
    (eda['New / Used'] == 'NEW') &
     (eda['Last Service - PMS'] == '-') &
    (~eda['Sale Invoice Year'].isna()) &
    (eda['Sale Invoice Year'] != '-')
    ]
    neweda = neweda[~neweda['VIN'].isin(serv['Vin_No'].unique())]
    print(f'''Neweda shape after filtering PMS: {neweda.shape}''')
    neweda = neweda.merge(rfmdf[['Customer ID','RFM_segments']],on=['Customer ID'],how='left')
    neweda['Invoice date'] = pd.to_datetime(neweda['Invoice date'],format='mixed',dayfirst=True)
    cutoff_date = custom_date - pd.DateOffset(months=(last_service_code/10)*6)
    print(f'Cutoff Date: {cutoff_date}')
    neweda = neweda[neweda['Invoice date'] >= cutoff_date]
    print(f'Neweda shape after filtering Invoice date: {neweda.shape}')
    print(f'Months to days function output: {months_to_days((last_service_code/10)*6)}')
    neweda[f'Next{last_service_code}K_Due'] = neweda['Invoice date'] +  pd.Timedelta(days=months_to_days((last_service_code/10)*6))
    # neweda.to_csv('NonPMS_initial.csv', index=False)
    fnlupd = neweda[(neweda[f'Next{last_service_code}K_Due'] > pd.Timestamp(filterdate)) & (neweda[f'Next{last_service_code}K_Due'] <= pd.Timestamp('2025-12-31'))]
    print(f'FilterDate: {filterdate}')
    print(f'Fnupd shape before filtering: {fnlupd.shape}')
    # fnlupd.to_csv('NonPMS_filtered.csv', index=False)
    fnlupd['Last Service Mileage'] = fnlupd['Last Service Mileage'].fillna(0) # -----Newly added
    lscode = last_service_code/10
    fnlupd = fnlupd.query('Vehicle_Key_ExpectedServices == @lscode')
    threshmil = (last_service_code * 1000) + 10000 ## --- Newly Added
    print(f'Threshmil: {threshmil}')
    fnlupd['Last Service Date - PMS'] = fnlupd['Invoice date']
    fnlupd = fnlupd.query("`Last Service Mileage` < @threshmil")
    fnlupd['Number of Cylinders'] = fnlupd['Number of Cylinders'].astype('object')
    rem = ['New / Used Category','Current Customer','First Service Date',
        'Last Service Date', 'Next Service Date','Vehicle Lifetime in Years',
        'Last Service - PMS','Sale Invoice Year','Invoice date','Target Revenue','Potential Revenue',
            'Final Revenue']
    fnlupd= fnlupd.drop(rem,axis=1,errors='ignore')
    pms = fnlupd
    missfeats = []
    hifunfeats = []
    for j in pms.columns:
        print(j)
        print('\n')
        if '-' in pms[j].values:
            hifunfeats.append(j)
        elif pms[j].isnull().sum() > 0:
            missfeats.append(j)
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
        'Battery Purchase Interval','70K R.1','LastServicePMS']
    pms = pms.drop(missfeatsdrop,axis=1,errors = 'ignore')

    hifunfeats1,missfeats1 = [],[]
    for j in pms.columns:
        print(j)
        print('\n')
        if '-' in pms[j].values:
            hifunfeats1.append(j)
        elif pms[j].isnull().sum() > 0:
            missfeats1.append(j)
        print('\n')
    
    for i in hifunfeats1:
        pms[i] = pms[i].replace('-',pd.NA)

    pms['Purchase Age'] = pd.to_numeric(pms['Purchase Age'], errors='coerce')
    pms['Current Age'] = pd.to_numeric(pms['Current Age'], errors='coerce')
    pms['Vehicle Age'] = pd.to_numeric(pms['Vehicle Age'], errors='coerce')

    pms.loc[(pms['Current Age'].isnull()) & (pms['Purchase Age'].notnull()), 'Current Age'] = pms['Purchase Age'] + pms['Vehicle Age']
    pms.loc[(pms['Purchase Age'].isnull()) & (pms['Current Age'].notnull()), 'Purchase Age'] = pms['Current Age'] - pms['Vehicle Age']
    print(f'Dtype of Vehicle_Key_ExpectedServices: {pms["Vehicle_Key_ExpectedServices"].dtype}')
    pms['Vehicle_Key_Actual_Service'] = pd.to_numeric(pms['Vehicle_Key_Actual_Service'], errors='coerce')
    print(f'Dtype of Vehicle_Key_Actual_Service: {pms["Vehicle_Key_Actual_Service"].dtype}')
    pms['Vehicle_Key_Actual_Service'] = pms['Vehicle_Key_Actual_Service'].fillna(0)
    pms['PMS_Delay'] = pms['Vehicle_Key_ExpectedServices'] - pms['Vehicle_Key_Actual_Service']
    pms['Service_Num'] = 0
    catfeats = ['New / Used','Gender','Warranty Status', 'Number of Cylinders']
    nationalitymap = pms[['Vehicle Key','Nationality']]
    vehmodelmap = pms[['Vehicle Key','Model']]
    vehvariantmap = pms[['Vehicle Key','Variant']]
    
    obint = ['Total Promoter','Total Passive','Total Detractor','Total Survey',
    'CC','Weight','Height','Wheel Base','Service Frequency']
    for i in obint:
        pms[i] = pd.to_numeric(pms[i], errors='coerce')
    serv2 = pd.read_csv('Service History Feb 2026.csv',low_memory=False)
    serv2 = serv2.query("Description == 'Others'")
    serv2['Service_Num'] = serv2['Description'].apply(extract_kk)
    serv2['Service_Date'] = pd.to_datetime(serv2['Service_Date'],format='mixed',dayfirst=True,errors='coerce')
    serv2 = serv2.query("Service_Date <= @filterdate")
    serv2['Mileage'] = pd.to_numeric(serv2['Mileage'], errors='coerce')
    serv2['Revenue'] = pd.to_numeric(serv2['Revenue'], errors='coerce')
    serv2 = serv2[serv2['Vin_No'].isin(pms['VIN'].unique())]
    serv2.to_csv('Filtered_NonPMS_Service_History.csv', index=False)
    newserv20k = serv2.groupby(['Vin_No']).agg(
        Service_Date=('Service_Date','max'),
        Mileage=('Mileage', 'max'),
        Total_Revenue=('Revenue', 'sum'),
    ).reset_index()
    newserv20k = newserv20k[newserv20k['Vin_No'].isin(pms['VIN'].unique())]
    dfnpmsdate = derive_npms_features(serv2)
    avg_mileage_interval_non_pms = derive_npms_mileage_features(serv2)
    vin_branch_pivot,top_branches = branch_visit_features(serv2, last_service_code)
    lastnonpmsmil = get_last_nonpms_mileagepred(serv2,0,pms, vin_col="Vin_No", mileage_col="Mileage",date_col="Service_Date")
    branch_grouped = branch_diversity_features(serv2, last_service_code)
    freq_npms, npms_counts, npmsrevenue, pms_counts = derive_npms_features2(serv2)
    bodyshop_counts = calculate_bodyshop_count(serv2, branch_col='Service_Branch_Name', vin_col='Vin_No', keyword='Bodyshop')
    bodyshop_counts = bodyshop_counts.rename(columns={'VIN':'Vin_No'})
    # appoindf = map_appointments_to_services(appoinshow, serv2,0)
    # fnlappnt = build_final_pms_appointment_summary(appoindf, pms[['VIN','Service_Num']])
    # fnlappntnoshow = compute_no_show_appointments_test(
    #     master_df=pms[['VIN']],
    #     noshow_df=appoinoshow,
    #     filter_date=filterdate,   
    #     vin_col="VIN",
    #     wip_deleted_col="WIP Deleted"
    # )
    revenu = compute_service_features(serv2,filterdate)
    complaint_features = transform_complaint_features(pms[['VIN','Service_Num']], serv2)
    newserv20ka = newserv20k
    newserv20ka  = pd.merge(newserv20ka,npms_counts,on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka,dfnpmsdate,on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka,revenu,on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka,avg_mileage_interval_non_pms,on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka,npmsrevenue,on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka,vin_branch_pivot,on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka,bodyshop_counts,on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka,freq_npms,on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka,branch_grouped,on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka,lastnonpmsmil,on=['Vin_No'], how='left')
    # newserv20ka = pd.merge(newserv20ka, fnlappnt, on=['Vin_No'], how='left')
    # newserv20ka = pd.merge(newserv20ka, fnlappntnoshow, on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka, complaint_features, on=['Vin_No'], how='left')
    newserv20ka[['nNPMS','npmsRevenue','Bodyshop_Services']] = newserv20ka[['nNPMS','npmsRevenue','Bodyshop_Services']].fillna(0)
    complfts = ['has_complaint_history','num_past_complaints','avg_complaint_resolution_days','max_complaint_resolution_days','recent_complaint_resolution_days'] 

    newserv20ka = newserv20ka.rename(columns={'Vin_No':'VIN'})
    pmsnew = pd.merge(pms,newserv20ka,on=['VIN'], how='left')
    pmsnew[['nNPMS','npmsRevenue','Bodyshop_Services','unique_branch_serviced','otherbranch_services','freq_NPMS']] = pmsnew[['nNPMS','npmsRevenue','Bodyshop_Services','unique_branch_serviced','otherbranch_services','freq_NPMS']].fillna(0)
    pmsnew[top_branches] = pmsnew[top_branches].fillna(0)
    pmsnew[revenu.columns[1:]] = pmsnew[revenu.columns[1:]].fillna(0)
    pmsnew[complfts] = pmsnew[complfts].fillna(0)
    pmsnew['latest_complaint_category'] = pmsnew['latest_complaint_category'].fillna('unknown')
    pmsnew = pmsnew.drop(['Service_Date','Mileage','Total_Revenue'],axis=1)
    pmsnew['Last PMS Mileage'] = 0
    pmsnew[['Total Promoter','Total Passive','Total Detractor','Total Survey']] = pmsnew[['Total Promoter','Total Passive','Total Detractor','Total Survey']].fillna(0)
    pmsnew['Warranty Status'] = pmsnew['Warranty Status'].fillna('No')
    # pmsnew['PMS_Delay'] = pms['Vehicle_Key_ExpectedServices']
    pmsnew['RFM_segments'] = pmsnew['RFM_segments'].fillna('Unknown')

    pmsnew[['Months_Since_Last_NPMS','Avg_Mileage_Interval_NPMS']] = pmsnew[['Months_Since_Last_NPMS','Avg_Mileage_Interval_NPMS']].fillna(0)
    colssrvd = pmsnew.pop('Last Service Date - PMS')
    pmsnew.insert(3, colssrvd.name, colssrvd)
    pmsnew['Last Service Date - PMS'] = pd.to_datetime(pmsnew['Last Service Date - PMS'],format='mixed',dayfirst=True,errors='coerce')
    pmsnew[f'Next{last_service_code}K_Due'] = pd.to_datetime(pmsnew[f'Next{last_service_code}K_Due'],format='mixed',dayfirst=True,errors='coerce')
    colsdue = pmsnew.pop(f'Next{last_service_code}K_Due')
    pmsnew.insert(4, colsdue.name, colsdue)
    pmsnew = pmsnew.drop(['Nationality','Model','Variant','Branch_List'],axis=1,errors='ignore')
    
    natres = map_cluster(nationalitymap, natclus,'Nationality','Nationality','Nationality_Cluster')
    modres = map_cluster(vehmodelmap, modclus,'Model','Model','Model_Cluster')
    varres = map_cluster(vehvariantmap, varclus,'Variant','Variant','Variant_Cluster')
    pmsnew = pmsnew.merge(natres[['Vehicle Key','Nationality_Cluster']],on='Vehicle Key',how='left')
    pmsnew = pmsnew.merge(modres[['Vehicle Key','Model_Cluster']],on='Vehicle Key',how='left')
    pmsnew = pmsnew.merge(varres[['Vehicle Key','Variant_Cluster']],on='Vehicle Key',how='left')
    pmsnew['Nationality_Cluster'] = pmsnew['Nationality_Cluster'].fillna('Unknown')
    pmsnew['Model_Cluster'] = pmsnew['Model_Cluster'].fillna('Unknown')       
    pmsnew['Variant_Cluster'] = pmsnew['Variant_Cluster'].fillna('Unknown')
    pmsnew['Nationality_Cluster'] = pmsnew['Nationality_Cluster'].astype(str)
    pmsnew['Model_Cluster'] = pmsnew['Model_Cluster'].astype(str)
    pmsnew['Variant_Cluster'] = pmsnew['Variant_Cluster'].astype(str) 
    pmsnew = map_vhc_history(pmsnew,serv2,0)

    appointdfN['Due Date IN'] = pd.to_datetime(appointdfN['Due Date IN'],format='mixed',dayfirst=True,errors='coerce')
    appoinstat = compute_nonpms_appointment_metrics(appointdfN,serv2)
    fnlappnt = derive_appointment_show_features(appointdfN,serv2,last_service_code,filter_date=filterdate)

    pmsnew["Vehicle Magic"] = pmsnew["Vehicle Key"].str.split("-", n=1).str[1]
    pmsnew["Vehicle Magic"] = pmsnew["Vehicle Magic"].str.split("-", n=1).str[0]
    pmsnew["Vehicle Magic"] = pmsnew["Vehicle Magic"].astype(int)
    nservappbk = count_service_appointments_booked(
            appointdfN,
            pmsnew,
            last_service_code,
            filterdate,
            vehicle_col="Vehicle Magic",
            booking_date_col="WIP Booking Date"
        )

    # appoinstat.to_csv('validatecode/lastappointmentstatus60kpred.csv', index=False)
    pmsnew = pmsnew.merge(appoinstat, on='Vehicle Magic',how='left').merge(fnlappnt,on='Vehicle Magic',how='left').merge(nservappbk,on='Vehicle Magic',how='left')
    # pmsnew.to_csv('validatecode/NonPMS_servdata_beforeimpute_60k.csv', index=False)
    pmsnew[['last_appointment_delay_days', 'no_of_late_appointments','appointment_booked_showed_up_atleastonce','total_appointments_showed_up','no_of_appointments_booked_but_not_showed_up','Avg_appointment_delay_days','no_of_service_appointments_booked']] = pmsnew[['last_appointment_delay_days', 'no_of_late_appointments','appointment_booked_showed_up_atleastonce','total_appointments_showed_up','no_of_appointments_booked_but_not_showed_up','Avg_appointment_delay_days','no_of_service_appointments_booked']].fillna(0)
    pmsnew['last_appointment_status'] = pmsnew['last_appointment_status'].fillna('No Appointment')
    pmsnew = pmsnew.drop('Vehicle Magic',axis=1,errors='ignore')


    pmsnew = one_hot(pmsnew)
    pmsnew = pmsnew.reset_index(drop=True)
    pmsnew = pmsnew.rename(columns=lambda c: re.sub(r'\.0$', '', c))
    pmsnew["LowMileageFreqUsers"] = 0
    # pmsnew.to_csv(f'validatecode/NonPMS_servdata_{last_service_code}k.csv', index=False)

    cols = pmsnew.pop('Last Service Date - PMS')
    pmsnew.insert(1, cols.name, cols)
    pmsnew = pmsnew.drop('70K R.1',axis=1,errors='ignore')
    vhcfill = ['Survey Score','VHC Quoted','VHC Sold','VHC Lost Sale','VHC Lost Red Sale','VHC Deferred','VHC Amber Deferred','VHC Completed_Flag']
    pmsnew[vhcfill] = pmsnew[vhcfill].fillna(0)



    # pmsnew.to_csv('validatecode/finalmerged40kv1.csv', index=False)
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
    pms_imputed = pms_imputed.drop(['Mileage','Brake Points','Vehicle_Key_ExpectedServices',
                                'Tyre Points','Battery Points'],axis=1,errors='ignore')
    pms_imputed[['Avg_Service_Interval_PMS', 'Avg_Service_Interval_PMS1old','Avg_Service_Interval_PMSper10k', 'Avg_Service_Interval_PMSold','Months_Since_Last_PMS','predicted_interval_months']] = 6*(last_service_code/10)
    pms_imputed['Avg_Mileage_Interval_PMS'] = last_service_code*1000
    # pms_imputed.to_csv(f'validatecode/NonPMS_servdata_{last_service_code}k.csv', index=False)

    return pms_imputed



filterdate = '2025-09-30' ##Input
custom_date = datetime.strptime(filterdate, '%Y-%m-%d') + timedelta(days=1)

service_interval = 10
initial = 8  # initial window for 30k, then +6 months per lower service

df =  pd.read_csv('Service History Feb 2026.csv',low_memory=False) #Input - Service History file
servhistory = df
df['Service_Date'] = pd.to_datetime(df['Service_Date'],format='mixed',dayfirst=True,errors='coerce')
df = df.query("Service_Date <= @filterdate")
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
df['Service_Num'] = df['Description'].apply(extract_kk)
df['Service_Num']  = np.where(df['Description'] == '<=10',10,df['Service_Num'])

finaldf = []
last_service_code = 60 # Input: Target service code (e.g., 40 for 40k)
service_history = list(range(service_interval, last_service_code, service_interval))  # [10k,20k,30k] for target=40k
eda = pd.read_csv('EDA Datasheet Till 2025.csv',low_memory=False,encoding='ISO-8859-1')
rfmdf = pd.read_csv('RFM till 2025.csv',low_memory=False,encoding='ISO-8859-1') #Input - RFM Segments file
modclus = pd.read_csv(f'validatecode/Model_clusters_{last_service_code}.csv') # Input - Model clusters file )(Output from Training)
varclus = pd.read_csv(f'validatecode/Variant_clusters_{last_service_code}.csv') # Input - Variant clusters file (Output from Training)
natclus = pd.read_csv(f'validatecode/Nationality_clusters_{last_service_code}.csv') # Input - Nationality clusters file (Output from Training)
appoinshow = pd.read_csv('Appoinment - Showed Up.csv', low_memory=False) # Input - Appointment Showed up file
appoinoshow = pd.read_csv('No Show VINs.csv', low_memory=False) # Input - No Show Appointments file
appointdfN = pd.read_csv('Appoinments2025.csv', low_memory=False) #Input appointment full data
digitalfts = pd.read_csv('DigitalData2025.csv', low_memory=False) # Input - Digital features file
servcode_desc = pd.read_csv('Service Code Desc.csv') # Input - Service code description file

for idx, svc in enumerate(service_history):
    months_back = initial + (len(service_history)-idx-1)*6
    cutoff_date = custom_date - pd.DateOffset(months=months_back)
    # df.to_csv(f'validatecode/Service_History_Upto_{svc}k.csv', index=False)
    vinsrem = df.query(f"Service_Num > {svc}")['Vin_No'].unique()
    pms30k = df.query(f"Service_Num == {svc}").copy()

    pms30k['Service_Date'] = pd.to_datetime(pms30k['Service_Date'],format='mixed',dayfirst=True)

    pms40k = df.query(f"Service_Num == {svc}")['Vin_No'].unique()
    pms30kupd = pms30k[pms30k['Vin_No'].isin(pms40k)]

    filtpms30k = pms30kupd[pms30kupd['Service_Date'] >= cutoff_date]
    filtpms30k['Mileage'] = pd.to_numeric(filtpms30k['Mileage'], errors='coerce')

    filtpms30kupd = filtpms30k

    filtpms30kupd['VIN_base'] = filtpms30kupd['Vehicle_Key'].str.rsplit('-', n=1).str[0]
    filtpms30kupd['Owner_No'] = (
        filtpms30kupd['Vehicle_Key']
            .str.extract(r'-(\d+)$')[0]
            .fillna(0)
            .astype(int)
    )
    filtpms30kupd = filtpms30kupd.loc[
        filtpms30kupd.groupby('VIN_base')['Owner_No'].idxmax()
    ].reset_index(drop=True)

    filtpms30kupdM = filtpms30kupd.groupby(['Vin_No']).agg(
        Mileage=('Mileage', 'max'),
        Service_Date=('Service_Date','max'),
        Total_Revenue=('Revenue', 'sum'),
    ).reset_index()

    finalserv40k = filtpms30kupdM[~filtpms30kupdM['Vin_No'].isin(vinsrem)]
    finalserv40k['Service_Num'] = svc

 #Input - EDA Datasheet file
    eda40k = eda
    eda40k['Last Service Date - PMS'] = pd.to_datetime(eda40k['Last Service Date - PMS'],format='mixed',dayfirst=True)
    eda40k = eda40k.rename(columns={'Last Service - PMS':'LastServicePMS'})
    eda40k['Service_Num'] = eda40k['LastServicePMS'].apply(extract_k1)
    eda40k = eda40k.query("Service_Num == @svc")
    edaM = eda40k[eda40k['VIN'].isin(list(finalserv40k['Vin_No'].unique()))]
    # edaM.to_csv(f'EDA_Service_{svc}k.csv', index=False)
    # results.append(edaM.shape)
   

    logger.info("Reading service history data")
    serv1 = servhistory
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
    serv1['Service_Num'] = serv1['Description'].apply(extract_kk)
    serv1['Service_Num'] = np.where(serv1['Description'] == '<=10', 10, serv1['Service_Num'])
    
    # Filter and process service data
    logger.info("Filtering and processing service data")
    serv1 = serv1.query(f"Service_Num <= {last_service_code}")
    serv = serv1[serv1['Vin_No'].isin(edaM['VIN'].unique())]
    servM = serv
    serv = serv.query("Description != 'Others'")
    
    logger.info(f'Max service number in data: {serv["Service_Num"].max()}')
    logger.info(f'Max Service date in data: {serv["Service_Date"].max()}')

    logger.info(f"Deriving PMS features for last service code {svc}k")
    dfpmsdate = derive_pms_features1(serv1, last_service_code)
    
    logger.info(f"Deriving NPMS features for last service code {svc}k")
    dfnpmsdate = derive_npms_features(serv1)
    
    logger.info(f"Deriving LastNonPMSMileage for last service code {svc}k")
    lastnonpmsmil = get_last_nonpms_mileagepred(serv1,svc,edaM, vin_col="Vin_No", mileage_col="Mileage",date_col="Service_Date")
    # lastnonpmsmil.to_csv(f'validatecode/lastnonpmsmil_{svc}k.csv', index=False) 

    logger.info(f"Calculating average mileage intervals for PMS - last service code {svc}k")
    avg_mileage_interval = derive_pms_mileage_features(serv1, dfpmsdate, last_service_code, svc)
    
    logger.info(f"Calculating average mileage intervals for NPMS - last service code {svc}k")
    avg_mileage_interval_non_pms = derive_npms_mileage_features(serv1)

    logger.info(f"Deriving PMS service intervals - last service code {svc}k")
    avg_monthly_interval = derive_pms_service_intervals(serv, last_service_code)
    # avg_monthly_interval.to_csv(f'validatecode/avg_monthly_interval_{svc}k.csv', index=False)
    logger.info(f"Adjusting service intervals - last service code {svc}k")
    avg_monthly_interval1 = adjust_service_intervalspred(avg_monthly_interval,last_service_code)
    avg_monthly_interval1.to_csv(f'validatecode/avg_monthly_interval1_{svc}k.csv', index=False)
    # avg_monthly_interval1.to_csv(f'validatecode/avg_monthly_interval1_{svc}k.csv', index=False)
    nextservinterv = calculatenxt_service_interval(avg_monthly_interval1, last_service_code)
  
    avg_monthly_interval1 = avg_monthly_interval1.rename(columns = {
    "Avg_Service_Interval_PMS": "Avg_Service_Interval_PMSold",
    "Avg_Service_Interval_PMS1": "Avg_Service_Interval_PMS1old",
    "Avg_Service_Interval_PMSnew": "Avg_Service_Interval_PMSper10k",
    "Avg_Service_Interval_PMS1new": "Avg_Service_Interval_PMS"
        })
    avg_monthly_interval1.drop(['Multiplier','last_service_no','predicted_interval_months'], axis=1, inplace=True)
    # avg_monthly_interval1.to_csv(f'validatecode/avg_monthly_interval1_{svc}k.csv', index=False)
    logger.info(f"Calculating NPMS metrics - last service code {svc}k")
    freq_npms, npms_counts, npmsrevenue, pms_counts = derive_npms_features2(serv1)
    # Npmsevents,npms_counts = get_non_pms_events(servM, servcode_desc, edaM, svc)
    logger.info("Calculating branch-related features")
    vin_branch_pivot,top_branches = branch_visit_features(serv, last_service_code)
    branch_grouped = branch_diversity_features(serv, last_service_code)

    logger.info("Calculating NPMS/PMS Revenue metrics")
    revenu = compute_service_features(servM,filterdate)

    # appoindf = map_appointments_to_services(appoinshow, servM,svc)
    # fnlappnt = build_final_pms_appointment_summary(appoindf, edaM[['VIN','Service_Num']])
    # fnlappntnoshow = compute_no_show_appointments_test(
    #     master_df=edaM[['VIN']],
    #     noshow_df=appoinoshow,
    #     filter_date=filterdate,   
    #     vin_col="VIN",
    #     wip_deleted_col="WIP Deleted"
    # )
    complaint_features = transform_complaint_features(edaM[['VIN','Service_Num']], servM)
    finalserv40k = finalserv40k[finalserv40k['Vin_No'].isin(list(edaM['VIN'].unique()))]
    
    finalserv40ka = finalserv40k
 
    merge_operations = [
        (npms_counts, 'NPMS counts'),
        (dfpmsdate, 'PMS dates'),
        (dfnpmsdate, 'NPMS dates'),
        (revenu, 'Service revenue'),
        (avg_mileage_interval, 'Mileage intervals PMS'),
        (avg_mileage_interval_non_pms, 'Mileage intervals NPMS'),
        (npmsrevenue, 'NPMS revenue'),
        (vin_branch_pivot, 'Branch visits'),
        (avg_monthly_interval1, 'Monthly intervals'),
        (freq_npms, 'NPMS frequency'),
        (branch_grouped, 'Branch diversity'),
        (nextservinterv, 'Next service interval'),
        (lastnonpmsmil, 'Last non-PMS mileage'),
        # (fnlappnt, 'Appointment show-up features'),
        # (fnlappntnoshow, 'No-show appointment features'),
        (complaint_features, 'Complaint features')
    ]

    for dfa, description in merge_operations:
        print(f'dfa columns before merging {description}: {dfa.columns.tolist()}')
        print(f"Merging {description}: {dfa['Vin_No'].is_unique}")

        logger.info(f"Merging {description}")
        before_shape = finalserv40ka.shape
        print(f'Merging {description}, Type: {type(dfa)}')
        finalserv40ka = pd.merge(finalserv40ka, dfa, on=['Vin_No'], how='left')
        after_shape = finalserv40ka.shape
        logger.debug(f"Shape after merging {description}: {after_shape}")
        if before_shape[0] != after_shape[0]:
            logger.warning(f"Row count changed after merging {description}")

    # Fill missing values
    logger.info("Handling missing values")
    fill_columns = ['nNPMS', 'npmsRevenue', 'freq_NPMS', 'unique_branch_serviced','otherbranch_services','SinglePMS',
                    'has_complaint_history','num_past_complaints','avg_complaint_resolution_days','max_complaint_resolution_days',
                    'recent_complaint_resolution_days'] + top_branches

    finalserv40ka[fill_columns] = finalserv40ka[fill_columns].fillna(0)

    # Final processing
    logger.info("Performing final data transformations")
    finalserv40ka = finalserv40ka.drop(['Total_Revenue','PMS_Service'], axis=1,errors='ignore')
    finalserv40ka = finalserv40ka.rename(columns={'Vin_No': 'VIN'})
    # finalserv40ka.to_csv(f'validatecode/finalservdata_{svc}k.csv', index=False)
    # Merge with master train data
    logger.info("Merging with master train data")
    filtered_dfnew = pd.merge(edaM, finalserv40ka, on=['VIN'], how='left')
    filtered_dfnew[fill_columns] = filtered_dfnew[fill_columns].fillna(0)
    filtered_dfnew = filtered_dfnew.drop('Service_Num', axis=1)
    filtered_dfnew['latest_complaint_category'] = filtered_dfnew['latest_complaint_category'].fillna('unknown')
    filtered_dfnew[revenu.columns[1:]] = filtered_dfnew[revenu.columns[1:]].fillna(0)
    # filtered_dfnew.to_csv(f'validatecode/CheckPreddata{svc}_1.csv', index=False)
    # Save output
    # output_path = 'validatecode/finalmerged40k.csv'
    # logger.info(f"Saving final output to {output_path}")
    filtered_dfnew = filtered_dfnew[filtered_dfnew['VIN'].isin(edaM['VIN'].unique())]
    filtered_dfnew[filtered_dfnew["Service_Num_x"] == filtered_dfnew["Service_Num_y"]]
    filtered_dfnew = filtered_dfnew.drop_duplicates(subset=['Vehicle Key'])
    filtered_dfnew = filtered_dfnew.drop(['Service_Num_y','Branch_List'], axis=1)
    filtered_dfnew = filtered_dfnew.rename(columns={'Service_Num_x': 'Service_Num'})
    # for i in filtered_dfnew.columns:
    #     print(i)
    filtered_dfnew['Last PMS Mileage'] = pd.to_numeric(filtered_dfnew['Last PMS Mileage'], errors='coerce')
    filtered_dfnew["LastNonPMSMileage"] = filtered_dfnew["LastNonPMSMileage"] \
                                      .fillna(filtered_dfnew["Last PMS Mileage"])
    if last_service_code == 20:
        filtered_dfnew['Avg_Mileage_Interval_PMS'] = filtered_dfnew['Last PMS Mileage']
    print(f"PMS Mileage dtype: {filtered_dfnew['Last PMS Mileage'].dtype}")

    filtered_dfnew = map_vhc_history(filtered_dfnew, serv, svc)
    vhcfill = ['Survey Score','VHC Quoted','VHC Sold','VHC Lost Sale','VHC Lost Red Sale','VHC Deferred','VHC Amber Deferred','VHC Completed_Flag']
    filtered_dfnew[vhcfill] = filtered_dfnew[vhcfill].fillna(0)

    appointdfN['Due Date IN'] = pd.to_datetime(appointdfN['Due Date IN'],format='mixed',dayfirst=True,errors='coerce')

    ltappoins = compute_late_appointment_metrics(appointdfN, servM,filter_date=filterdate,svc=svc)
    appoinstat = compute_last_appointment_status_with_constant_service_code(appointdfN,serv,svc,filter_date=filterdate)
    fnlappnt = derive_appointment_show_features(appointdfN,servM,svc,filter_date=filterdate)
    filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Key"].str.split("-", n=1).str[1]
    filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Magic"].str.split("-", n=1).str[0]
    filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Magic"].astype(int)

    filtered_dfnew = filtered_dfnew.merge(appoinstat, on='Vehicle Magic',how='left').merge(ltappoins,on='Vehicle Magic',how='left').merge(fnlappnt,on='Vehicle Magic',how='left')
    nservappbk = count_service_appointments_booked(
                appointdfN,
                filtered_dfnew,
                svc,
                filterdate,
                vehicle_col="Vehicle Magic",
                booking_date_col="WIP Booking Date"
            )
    filtered_dfnew = filtered_dfnew.merge(nservappbk,on='Vehicle Magic',how='left')
    filtered_dfnew['no_of_service_appointments_booked'] = filtered_dfnew['no_of_service_appointments_booked'].fillna(0)

    filtered_dfnew = filtered_dfnew.drop(['Vehicle Magic'],axis=1)
    # ltappoins.to_csv(f'validatecode/lateappointments{svc}kpred.csv', index=False)
    # appoinstat.to_csv(f'validatecode/lastappointmentstatus{svc}kpred.csv', index=False)
    filtered_dfnew[['last_appointment_delay_days', 'no_of_late_appointments','appointment_booked_showed_up_atleastonce','total_appointments_showed_up','no_of_appointments_booked_but_not_showed_up','Avg_appointment_delay_days']] = filtered_dfnew[['last_appointment_delay_days', 'no_of_late_appointments','appointment_booked_showed_up_atleastonce','total_appointments_showed_up','no_of_appointments_booked_but_not_showed_up','Avg_appointment_delay_days']].fillna(0)
    filtered_dfnew['last_appointment_status'] = filtered_dfnew['last_appointment_status'].fillna('No Appointment')
    # filtered_dfnew.loc[
    #             filtered_dfnew["Service_Num"] == 10,
    #             "Avg_Mileage_Interval_PMS"
    #         ] = (((last_service_code * 1000) - filtered_dfnew.loc[
    #             filtered_dfnew["Service_Num"] == 10, "Last PMS Mileage"
    #         ]) + (filtered_dfnew.loc[
    #             filtered_dfnew["Service_Num"] == 10, "Last PMS Mileage"
    #         ])) /2
    
    # Apply row-wise
    filtered_dfnew[f"Next{last_service_code}K_Due"] = filtered_dfnew.apply(
    lambda row: calc_next_due(row["Last Service Date - PMS"], row["predicted_interval_months"]),
    axis=1
    )
    filtered_dfnew[f"Next{last_service_code}K_Due"] = pd.to_datetime(filtered_dfnew[f"Next{last_service_code}K_Due"], format='mixed', errors='coerce')
    fnlupd = filtered_dfnew[(filtered_dfnew[f'Next{last_service_code}K_Due'] > pd.Timestamp('2025-06-30')) & 
    (filtered_dfnew[f'Next{last_service_code}K_Due'] <= pd.Timestamp('2025-09-30'))]
    threshmil = (last_service_code * 1000) + 10000
    fnlupd = fnlupd.query("`Last Service Mileage` < @threshmil")

    # fnlupd.to_csv(f'validatecode/CheckPreddata{svc}.csv', index=False)
    fnlupd = fnlupd.merge(rfmdf[['Customer ID','RFM_segments']],on=['Customer ID'],how='left')
    fnlupd['Number of Cylinders'] = fnlupd['Number of Cylinders'].astype('object')
    rem = ['New / Used Category','Current Customer','First Service Date',
        'Last Service Date', 'Next Service Date','Vehicle Lifetime in Years',
        'Last Service - PMS','Sale Invoice Year','Invoice date','Target Revenue','Potential Revenue',
            'Final Revenue']
    fnlupd= fnlupd.drop(rem,axis=1,errors='ignore')
    pms = fnlupd
    missfeats = []
    hifunfeats = []
    for j in pms.columns:
        print(j)
        print('\n')
        if '-' in pms[j].values:
            hifunfeats.append(j)
        elif pms[j].isnull().sum() > 0:
            missfeats.append(j)
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
        'Battery Purchase Interval','70K R.1','LastServicePMS']
    pms = pms.drop(missfeatsdrop,axis=1,errors = 'ignore')

    hifunfeats1,missfeats1 = [],[]
    for j in pms.columns:
        print(j)
        print('\n')
        if '-' in pms[j].values:
            hifunfeats1.append(j)
        elif pms[j].isnull().sum() > 0:
            missfeats1.append(j)
        print('\n')
    
    for i in hifunfeats1:
        pms[i] = pms[i].replace('-',pd.NA)

    pms['Purchase Age'] = pd.to_numeric(pms['Purchase Age'], errors='coerce')
    pms['Current Age'] = pd.to_numeric(pms['Current Age'], errors='coerce')
    pms['Vehicle Age'] = pd.to_numeric(pms['Vehicle Age'], errors='coerce')

    pms.loc[(pms['Current Age'].isnull()) & (pms['Purchase Age'].notnull()), 'Current Age'] = pms['Purchase Age'] + pms['Vehicle Age']
    pms.loc[(pms['Purchase Age'].isnull()) & (pms['Current Age'].notnull()), 'Purchase Age'] = pms['Current Age'] - pms['Vehicle Age']
    print(f'Dtype of Vehicle_Key_ExpectedServices: {pms["Vehicle_Key_ExpectedServices"].dtype}')
    pms['Vehicle_Key_Actual_Service'] = pd.to_numeric(pms['Vehicle_Key_Actual_Service'], errors='coerce')
    print(f'Dtype of Vehicle_Key_Actual_Service: {pms["Vehicle_Key_Actual_Service"].dtype}')
    pms['PMS_Delay'] = pms['Vehicle_Key_ExpectedServices'] - pms['Vehicle_Key_Actual_Service']
    catfeats = ['New / Used','Gender','Warranty Status', 'Number of Cylinders']
    nationalitymap = pms[['Vehicle Key','Nationality']]
    vehmodelmap = pms[['Vehicle Key','Model']]
    vehvariantmap = pms[['Vehicle Key','Variant']]
    # rfmpms= pms[['Vehicle Key','RFM_segments']]
    # nationalitymap.to_csv('validatecode/nationalitymap40.csv', index=False)
    # vehmodelmap.to_csv('validatecode/vehmodelmap40.csv', index=False)
    # vehvariantmap.to_csv('validatecode/vehvariantmap40.csv', index=False)
    # rfmpms.to_csv('validatecode/rfmpms40.csv', index=False)
    
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
    pms = pms.drop(['Nationality','Model','Variant'],axis=1,errors='ignore')
    
    natres = map_cluster(nationalitymap, natclus,'Nationality','Nationality','Nationality_Cluster')
    modres = map_cluster(vehmodelmap, modclus,'Model','Model','Model_Cluster')
    varres = map_cluster(vehvariantmap, varclus,'Variant','Variant','Variant_Cluster')
    pms = pms.merge(natres[['Vehicle Key','Nationality_Cluster']],on='Vehicle Key',how='left')
    pms = pms.merge(modres[['Vehicle Key','Model_Cluster']],on='Vehicle Key',how='left')
    pms = pms.merge(varres[['Vehicle Key','Variant_Cluster']],on='Vehicle Key',how='left')
    pms['Nationality_Cluster'] = pms['Nationality_Cluster'].fillna('Unknown')
    pms['Model_Cluster'] = pms['Model_Cluster'].fillna('Unknown')       
    pms['Variant_Cluster'] = pms['Variant_Cluster'].fillna('Unknown')
    pms['Nationality_Cluster'] = pms['Nationality_Cluster'].astype(str)
    pms['Model_Cluster'] = pms['Model_Cluster'].astype(str)
    pms['Variant_Cluster'] = pms['Variant_Cluster'].astype(str) 

    pmsnew = one_hot(pms)
    pmsnew = pmsnew.rename(columns=lambda c: re.sub(r'\.0$', '', c))
    pmsnew = pmsnew.reset_index(drop=True)
    pmsnew["LowMileageFreqUsers"] = np.where(
        (pmsnew["Avg_Service_Interval_PMS"].between(0, 7)) & (pmsnew["Last PMS Mileage"].between(0, (svc-10)*1000)),1,0)
    uy = ['Total Promoter','Total Passive','Total Detractor','Total Survey']
    for i in uy:
        pmsnew[i] = pmsnew[i].fillna(0)
    cols = pmsnew.pop('Last Service Date - PMS')
    pmsnew.insert(1, cols.name, cols)
    pmsnew = pmsnew.drop('70K R.1',axis=1,errors='ignore')
    # pmsnew.to_csv('validatecode/finalmerged40kv1.csv', index=False)
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
    pms_imputed = pms_imputed.drop(['Mileage','Brake Points','Vehicle_Key_ExpectedServices',
                                'Tyre Points','Battery Points'],axis=1,errors='ignore')
    pms_imputed.rename(columns={'Avg_Service_Interval_PMS':'Avg_Service_Interval_PMS_per10k'},inplace=True)
    pms_imputed.rename(columns={'Avg_Service_Interval_PMS1':'Avg_Service_Interval_PMS'},inplace=True)
    coltpmsmil = pms_imputed.pop("Last PMS Mileage")      
    pms_imputed.insert(len(pms_imputed.columns)-1, "Last PMS Mileage", coltpmsmil)
    pms_imputed["Last Service Mileage"] = pms_imputed[["Last PMS Mileage", "LastNonPMSMileage"]].max(axis=1)
    # pms_imputed.to_csv(f'validatecode/finalservdata_{svc}k_imputed.csv', index=False)

    finaldf.append(pms_imputed)

maindf = pd.concat(finaldf,axis=0)
maindf = maindf.fillna(0)
maindf.to_csv('chckpredMAin.csv', index=False)
# print(f'Results Shape: {results}')

nompmsbase = NonPMS(rfmdf,filterdate,last_service_code,custom_date,natclus,varclus,modclus,appoinshow,appoinoshow,appointdfN,servcode_desc)
nompmsbase.to_csv('chckpredMAinNonPms.csv', index=False)

finalbase = pd.concat([maindf, nompmsbase], axis=0)
finalbase = finalbase.fillna(0)
finalbase = finalbase.drop(columns=[col for col in finalbase.columns if "unknown" in col.lower()])
finalbase = finalbase.merge(digitalfts,on=['Vehicle Key','Customer ID'],how='left')
finalbase.iloc[:,-4:] = finalbase.iloc[:,-4:].fillna(0)
finalbase['Vehicles Owned'] = finalbase['Vehicles Owned'].fillna(1)
vhcfill = ['Survey Score','VHC Quoted','VHC Sold','VHC Lost Sale','VHC Lost Red Sale','VHC Deferred','VHC Amber Deferred','VHC Completed_Flag']

finalbase = finalbase.drop(vhcfill, axis=1, errors='ignore')
vhcdf = pd.read_csv('VHC PMS till 2025.csv',encoding="ISO-8859-1",low_memory=False)
vhcdata = vhcpreparation(vhcdf,last_service_code-10,finalbase)
finalbase = finalbase.merge(vhcdata,on=['Vehicle Key','Service_Num'],how='left')
finalbase = finalbase.fillna(0)

servMain = df
servMain = servMain.query("Service_Num < @last_service_code")
Npmsevents,non_pms_countn = get_non_pms_events(servMain, servcode_desc, finalbase,last_service_code)
Npmsevents = Npmsevents.rename(columns={'Vin_No':'VIN'})
non_pms_countn = non_pms_countn.rename(columns={'Vin_No':'VIN'})
# Npmsevents,npms_counts = get_non_pms_events(servM, servcode_desc, finalbase, last_service_code)
finalbase = finalbase.drop('nNPMS', axis=1, errors='ignore')
finalbase = finalbase.merge(Npmsevents,on='VIN',how='left').fillna(0)
finalbase = finalbase.merge(non_pms_countn,on='VIN',how='left').fillna(0)

finalbase.to_csv('chckpred1.csv', index=False)
# finalbase.to_csv(f'validatecode/FinalValidateset{last_service_code}k.csv', index=False) #Output Final Prediction Dataset

