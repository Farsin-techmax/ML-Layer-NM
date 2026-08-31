"""
features.py — Shared feature engineering functions for PMS Training & Prediction pipelines.

Sections:
  1. Parsing & Utilities
  2. PMS / NPMS Date Features
  3. Mileage Features
  4. Service Intervals
  5. NPMS Aggregates
  6. Branch Features
  7. Revenue Features
  8. Appointment Features
  9. Complaint Features
  10. VHC Features
  11. Non-PMS Events
  12. Encoding & Imputation
  13. Clustering & Mapping Helpers
  14. Prediction Helpers
"""

import os
import re
import ast
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import Counter, defaultdict


def _debug(msg):
    """Progress trace for avg_mileage_interval_pms(). Off unless PMS_DEBUG is set.

    These calls used to append to debug.txt unconditionally on every invocation. debug.txt is a
    tracked file, so any feature-engineering run left the working tree dirty with noise nobody
    reads. Enable with:  set PMS_DEBUG=1  (PowerShell: $env:PMS_DEBUG=1)
    """
    if os.environ.get("PMS_DEBUG"):
        with open("debug.txt", "a") as fh:
            fh.write(msg + "\n")



# ═══════════════════════════════════════════════════════════════════════════════
# 1. PARSING & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_k1(service_str):
    """Convert '50k' -> 50."""
    try:
        return int(service_str.replace("k", "").strip())
    except Exception:
        return 0


def extract_kk(service_str):
    """Extract numeric from Service History description (e.g. '21-30' -> 30)."""
    try:
        return int(service_str.split('-')[-1])
    except Exception:
        return 0

# Alias used in training script
extract_k = extract_kk


def derive_servcode(value):
    """Convert service code to nearest 10-multiple or 0."""
    try:
        num = int(value)
        return num if num % 10 == 0 else 0
    except (ValueError, TypeError):
        return 0


def ensure_list(x):
    """Wrap scalar in a list if not already a list."""
    return x if isinstance(x, list) else [x]


def normalize_to_list(val):
    """
    Normalize val into a Python list of strings.
    Handles: actual lists, string repr like "['A','B']", comma/semicolon separated, NaN.
    """
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed]
            except Exception:
                pass
        if ',' in s:
            return [x.strip() for x in s.split(',') if x.strip()]
        if ';' in s:
            return [x.strip() for x in s.split(';') if x.strip()]
        return [s]
    return [str(val)]


def clean_parts(x):
    """Explode, drop NaN/empty/dash values from a Series of lists."""
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return (
        x.explode().dropna().astype(str).str.strip()
        .loc[lambda s: (s != '-') & (s != '') & (s != 'nan')]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PMS / NPMS DATE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def _months_diff(start_date, end_date):
    """Helper: months between two dates."""
    rd = relativedelta(end_date, start_date)
    return rd.years * 12 + rd.months


def derive_pms_features1(serv1, last_service_code, reference_date=None):
    """
    PMS date features per VIN: Months_Since_Last_PMS, SinglePMS flag.
    Filters Service_Num > 0 and < last_service_code.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.today()

    df = serv1.sort_values(by=['Vin_No', 'Service_Date']).copy()
    df = df.query('Service_Num > 0 and Service_Num < @last_service_code')

    first = df.groupby('Vin_No')['Service_Date'].min().reset_index(name='First_PMS_Date')
    last = df.groupby('Vin_No')['Service_Date'].max().reset_index(name='Last_PMS_Date')

    df = df.merge(first, on='Vin_No', how='left').merge(last, on='Vin_No', how='left')
    df['Months_Since_First_PMS'] = df['First_PMS_Date'].apply(lambda d: _months_diff(d, reference_date))
    df['Months_Since_Last_PMS'] = df['Last_PMS_Date'].apply(lambda d: _months_diff(d, reference_date))

    df = df.groupby('Vin_No').agg(
        Months_Since_First_PMS=('Months_Since_First_PMS', 'mean'),
        Months_Since_Last_PMS=('Months_Since_Last_PMS', 'mean')
    ).reset_index()

    df['SinglePMS'] = (df['Months_Since_First_PMS'] == df['Months_Since_Last_PMS']).astype(int)
    return df[['Vin_No', 'Months_Since_Last_PMS', 'SinglePMS']]


def derive_npms_features(serv1, reference_date=None):
    """Non-PMS date features per VIN: Months_Since_Last_NPMS."""
    if reference_date is None:
        reference_date = pd.Timestamp.today()

    df = serv1.sort_values(by=['Vin_No', 'Service_Date']).copy()
    df = df.query('Service_Num == 0')

    first = df.groupby('Vin_No')['Service_Date'].min().reset_index(name='First_NPMS_Date')
    last = df.groupby('Vin_No')['Service_Date'].max().reset_index(name='Last_NPMS_Date')

    df = df.merge(first, on='Vin_No', how='left').merge(last, on='Vin_No', how='left')
    df['Months_Since_First_NPMS'] = df['First_NPMS_Date'].apply(lambda d: _months_diff(d, reference_date))
    df['Months_Since_Last_NPMS'] = df['Last_NPMS_Date'].apply(lambda d: _months_diff(d, reference_date))

    df = df.groupby('Vin_No').agg(
        Months_Since_First_NPMS=('Months_Since_First_NPMS', 'mean'),
        Months_Since_Last_NPMS=('Months_Since_Last_NPMS', 'mean')
    ).reset_index()

    return df[['Vin_No', 'Months_Since_Last_NPMS']]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MILEAGE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

# _avg_mile_interval removed in favor of vectorized .diff()
def derive_pms_mileage_features(serv1, dfpmsdate, last_service_code, svc=None):
    """
    PMS mileage features per VIN. SinglePMS == 1 rows get replaced with group mean.
    
    Args:
        svc: If provided and == 1, skips skipped_blocks logic (prediction first-service case).
             Otherwise applies skipped_blocks multiplier.
    """
    dfpmsmil = serv1[['Vin_No', 'Service_Num', 'Mileage']].copy()
    _debug("1. Filtering")
    upper = svc if svc is not None else last_service_code
    if svc is not None:
        dfpmsmil = dfpmsmil.query('Service_Num > 0 and Service_Num <= @upper')
    else:
        dfpmsmil = dfpmsmil.query('Service_Num > 0 and Service_Num < @last_service_code')

    _debug("2. Groupby Diff")
    dfpmsmil = dfpmsmil.copy()
    dfpmsmil['Mileage_Diff'] = dfpmsmil.groupby('Vin_No')['Mileage'].diff()
    dfpmsmil['Mileage_Diff'] = dfpmsmil['Mileage_Diff'].fillna(dfpmsmil['Mileage'])
    _debug("3. Groupby Mean")
    avg_mileage_interval = (
        dfpmsmil.groupby('Vin_No')['Mileage_Diff']
        .mean()
        .reset_index(name='Avg_Mileage_Interval_PMS')
    )

    _debug("4. Merges")
    avg_mileage_interval = avg_mileage_interval.merge(
        dfpmsdate[['Vin_No', 'SinglePMS']], on='Vin_No', how='left'
    )

    df_max_service = dfpmsmil.groupby("Vin_No", as_index=False)["Service_Num"].max()
    avg_mileage_interval = avg_mileage_interval.merge(df_max_service, on='Vin_No', how='left')

    _debug("5. Replacement Means")
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
    _debug("6. Return")
    # For prediction first-service (svc==1), return early without skipped_blocks
    if svc is not None and svc == 1:
        return avg_mileage_interval.drop(columns=['SinglePMS', 'Service_Num'])

    # Apply skipped_blocks multiplier
    avg_mileage_interval['skipped_blocks'] = (
        (last_service_code - 10 - avg_mileage_interval['Service_Num']) // 10
    ).clip(lower=0)
    avg_mileage_interval['skipped_blocks'] = np.where(
        avg_mileage_interval['skipped_blocks'] == 1, 2, avg_mileage_interval['skipped_blocks']
    )
    avg_mileage_interval['predicted_interval'] = (
        avg_mileage_interval['Avg_Mileage_Interval_PMS'] * avg_mileage_interval['skipped_blocks']
    )
    avg_mileage_interval.loc[
        avg_mileage_interval['skipped_blocks'] == 0, 'predicted_interval'
    ] = avg_mileage_interval.loc[
        avg_mileage_interval['skipped_blocks'] == 0, 'Avg_Mileage_Interval_PMS'
    ]
    avg_mileage_interval = avg_mileage_interval.rename(columns={
        'Avg_Mileage_Interval_PMS': 'Avg_Mileage_Interval_PMSold',
        'predicted_interval': 'Avg_Mileage_Interval_PMS'
    })
    avg_mileage_interval = avg_mileage_interval.drop(
        columns=['SinglePMS', 'Service_Num', 'skipped_blocks'], errors='ignore'
    )
    return avg_mileage_interval


def derive_npms_mileage_features(serv1):
    """Non-PMS average mileage interval per VIN."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SERVICE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════════

def derive_pms_service_intervals(serv1, last_service_code):
    """
    PMS service interval features per VIN.
    Returns: Vin_No, Avg_Service_Interval_PMS, Avg_Service_Interval_PMS1, Service_Num.
    """
    dfpmsmil = serv1.copy()
    dfpmsmil = dfpmsmil.query("Service_Num > 0 and Service_Num < @last_service_code")

    service_max_dates = (
        dfpmsmil.groupby(['Vin_No', 'Service_Num'], as_index=False)
        .agg({'Service_Date': 'max', 'Mileage': 'max'})
    )
    df_max_service = dfpmsmil.groupby("Vin_No", as_index=False)["Service_Num"].max()

    service_max_dates = service_max_dates.sort_values(['Vin_No', 'Service_Date'])
    
    # Calculate differences across sorted DataFrame
    service_max_dates['Date_Diff_Days'] = service_max_dates.groupby('Vin_No')['Service_Date'].diff().dt.days
    service_max_dates['Mileage_Diff'] = service_max_dates.groupby('Vin_No')['Mileage'].diff()
    
    # Average Month Interval Pure
    service_max_dates['Months_Diff'] = service_max_dates['Date_Diff_Days'] / 30.44
    avg1 = (
        service_max_dates.groupby('Vin_No')['Months_Diff']
        .mean()
        .reset_index(name='Avg_Service_Interval_PMS1')
    )
    
    # Average Month Interval Mileage Norm
    service_max_dates['Norm_Diff'] = service_max_dates['Months_Diff'] / (service_max_dates['Mileage_Diff'] / 10000)
    avg0 = (
        service_max_dates.groupby('Vin_No')['Norm_Diff']
        .mean()
        .reset_index(name='Avg_Service_Interval_PMS')
    )

    result = avg0.merge(avg1, on='Vin_No', how='left').merge(df_max_service, on='Vin_No', how='left')

    mask_invalid = (
        result['Avg_Service_Interval_PMS'].isna()
        | np.isinf(result['Avg_Service_Interval_PMS'])
        | (result['Avg_Service_Interval_PMS'] < 0.5)
    )
    result.loc[mask_invalid, 'Avg_Service_Interval_PMS'] = result.loc[mask_invalid, 'Avg_Service_Interval_PMS1']

    return result[['Vin_No', 'Avg_Service_Interval_PMS', 'Avg_Service_Interval_PMS1', 'Service_Num']]


def adjust_service_intervals(df, prediction_km):
    """
    Apply distance-based multipliers to service interval features.
    Unified version — works for both training and prediction.
    """
    assert prediction_km % 10 == 0 and prediction_km >= 20, \
        "prediction_km must be >=20 and a multiple of 10"

    cols = ["Avg_Service_Interval_PMS", "Avg_Service_Interval_PMS1"]
    df = df.copy()

    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    df = df[df["Service_Num"] < prediction_km]

    for col in cols:
        df[col] = df[col].where(df[col].between(0.5, 60))

    for col in cols:
        df[col] = df.groupby("Service_Num")[col].transform(lambda x: x.fillna(x.mean()))

    distance = (prediction_km - df["Service_Num"]) // 10
    df["Multiplier"] = np.where(
        df["Service_Num"] < 10, 1.0,
        np.select(
            [distance == 1, distance == 2, distance == 3, distance >= 4],
            [1.0, 1.5, 2.0, distance - 1.0],
            default=1.0
        )
    )

    df["Avg_Service_Interval_PMSnew"] = df["Avg_Service_Interval_PMS"] * df["Multiplier"]
    df["Avg_Service_Interval_PMS1new"] = df["Avg_Service_Interval_PMS1"] * df["Multiplier"]

    return df[[
        "Vin_No", "Service_Num", "Multiplier",
        "Avg_Service_Interval_PMS", "Avg_Service_Interval_PMS1",
        "Avg_Service_Interval_PMSnew", "Avg_Service_Interval_PMS1new",
    ]]

# Alias — pred script used this name
adjust_service_intervalspred = adjust_service_intervals


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NPMS AGGREGATES
# ═══════════════════════════════════════════════════════════════════════════════

def derive_npms_features2(serv1):
    """NPMS frequency, counts, revenue, PMS counts per VIN."""
    serv1 = serv1.copy()
    serv1['Revenue'] = abs(serv1['Revenue'])

    pms_df = serv1[serv1['Service_Num'] > 0]
    npms_df = serv1[serv1['Service_Num'] == 0].copy()

    pms_agg = pms_df.groupby(['Vin_No', 'Service_Num', 'Service_Date']).agg(
        Mileage=('Mileage', 'mean'), Total_Revenue=('Revenue', 'sum')
    ).reset_index()

    npms_df['Service_Date'] = pd.to_datetime(npms_df['Service_Date'], dayfirst=True, errors='coerce')
    npms_df['Mileage'] = pd.to_numeric(npms_df['Mileage'], errors='coerce')

    npms_agg = npms_df.groupby(['Vin_No', 'Service_Num', 'Service_Date']).agg(
        Mileage=('Mileage', 'mean'), npmsRevenue=('Revenue', 'sum')
    ).reset_index()

    npms_counts = npms_agg.groupby('Vin_No').size().reset_index(name='nNPMS')
    pms_counts = pms_agg.groupby('Vin_No').size().reset_index(name='nPMS')
    npmsrevenue = npms_agg.groupby('Vin_No').agg(npmsRevenue=('npmsRevenue', 'sum')).reset_index()

    first_last = npms_df.groupby('Vin_No').agg(
        First_NPMS_Date=('Service_Date', 'min'), Last_NPMS_Date=('Service_Date', 'max')
    ).reset_index()

    freq_npms = first_last.merge(npms_counts, on='Vin_No', how='left')
    freq_npms['freq_NPMS'] = (
        (freq_npms['Last_NPMS_Date'].dt.to_period("M") - freq_npms['First_NPMS_Date'].dt.to_period("M"))
        .apply(lambda x: x.n)
    ) / freq_npms['nNPMS']

    return freq_npms[['Vin_No', 'freq_NPMS']], npms_counts, npmsrevenue, pms_counts


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BRANCH FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def branch_visit_features(serv1, last_service_code, top_n=7):
    """Branch visit count features per VIN. Returns (pivot_df, top_branches_list)."""
    df = serv1.sort_values(by=['Vin_No', 'Service_Date'])
    df = df.query("Service_Num < @last_service_code")

    counts = df.groupby(['Vin_No', 'Service_Branch_Name']).size().reset_index(name='Visit_Count')
    pivot = counts.pivot(index='Vin_No', columns='Service_Branch_Name', values='Visit_Count')
    pivot = pivot.fillna(0).astype(int).reset_index()

    totals = pivot.drop(columns=['Vin_No']).sum().sort_values(ascending=False)
    top_branches = totals.head(top_n).index.tolist()

    other = [c for c in pivot.columns if c not in ['Vin_No'] + top_branches]
    pivot['otherbranch_services'] = pivot[other].sum(axis=1)
    pivot = pivot[['Vin_No'] + top_branches + ['otherbranch_services']]

    return pivot, top_branches


def branch_diversity_features(serv1, last_service_code):
    """Unique branch count per VIN."""
    df = serv1.sort_values(by=['Vin_No', 'Service_Date'])
    df = df.query("Service_Num < @last_service_code")

    grouped = (
        df.groupby('Vin_No')['Service_Branch_Name']
        .apply(lambda x: list(set(x)))
        .reset_index(name='Branch_List')
    )
    grouped['unique_branch_serviced'] = grouped['Branch_List'].apply(len)
    return grouped


def calculate_bodyshop_count(df, branch_col='Service_Branch_Name', vin_col='Vin_No', keyword='Bodyshop'):
    """Count Bodyshop visits per VIN."""
    filtered = df[df[branch_col].str.contains(keyword, case=False, na=False)]
    counts = filtered.groupby(vin_col).size().rename('Bodyshop_Services').reset_index()
    return counts.rename(columns={vin_col: 'VIN'})


# ═══════════════════════════════════════════════════════════════════════════════
# 7. REVENUE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_service_features(serv, filterdate, last_service_num=None):
    """PMS and Non-PMS revenue features per Vin_No."""
    serv = serv.copy()
    serv['Service_Date'] = pd.to_datetime(serv['Service_Date'])
    serv['Revenue'] = pd.to_numeric(serv['Revenue'], errors='coerce').abs()
    serv['Service_Num'] = pd.to_numeric(serv['Service_Num'], errors='coerce')

    if last_service_num is not None:
        serv = serv.query("Service_Num != @last_service_num")

    serv.sort_values(by=['Vin_No', 'Service_Date', 'Service_Num'], ascending=True, inplace=True)

    reference_date = pd.to_datetime(filterdate) + pd.Timedelta(days=1)

    # Non-PMS
    non_pms = serv[serv['Service_Num'] == 0].sort_values('Service_Date')
    last_non_pms = (
        non_pms.groupby('Vin_No', as_index=False).last()[['Vin_No', 'Service_Date', 'Revenue']]
        .rename(columns={'Service_Date': 'Last_NonPMS_Date', 'Revenue': 'Last_NonPMS_Revenue'})
    )
    last_non_pms['Days_Since_Last_NonPMS'] = (reference_date - last_non_pms['Last_NonPMS_Date']).dt.days

    # PMS
    pms = serv[serv['Service_Num'] != 0].sort_values('Service_Date')
    last_pms_rev = (
        pms.groupby('Vin_No', as_index=False).last()[['Vin_No', 'Revenue']]
        .rename(columns={'Revenue': 'Last_PMS_Revenue'})
    )
    pms_agg = (
        pms.groupby('Vin_No')['Revenue']
        .agg(Max_PMS_Revenue='max', Min_PMS_Revenue='min', StdDev_PMS_Revenue='std')
        .reset_index()
    )

    all_vins = pd.DataFrame({'Vin_No': serv['Vin_No'].unique()})
    result = (
        all_vins
        .merge(last_non_pms[['Vin_No', 'Last_NonPMS_Revenue', 'Last_NonPMS_Date', 'Days_Since_Last_NonPMS']],
               on='Vin_No', how='left')
        .merge(last_pms_rev, on='Vin_No', how='left')
        .merge(pms_agg, on='Vin_No', how='left')
    )
    return result.drop('Last_NonPMS_Date', axis=1).fillna(0)


def derive_pms_frequency(serv, filterdate, min_years=1.0):
    """Non-leaky replacement for EDA's 'Service Frequency': PMS visits per year of tenure,
    derived from service history and cut at the training cutoff.

    Caller must pass `serv` already filtered to PMS visits before the target milestone and before
    the cutoff -- process_service_data does this upstream (Service_Date <= filterdate,
    Service_Num < last_service_code, Description != 'Others'), so counting rows here needs no
    extra filtering. A VIN with zero prior PMS visits simply does not appear in `serv` and is
    absent from the result; callers must fillna(0) after merging.

    Returns one row per Vin_No:
      PMS_Count_Prior       -- count of qualifying prior PMS visits
      Years_Since_First_PMS -- tenure since the VIN's first PMS, floored at `min_years` so a
                                single recent visit cannot produce an inflated rate
      PMS_Freq_PerYear      -- PMS_Count_Prior / Years_Since_First_PMS
    """
    serv = serv.copy()
    serv['Service_Date'] = pd.to_datetime(serv['Service_Date'], errors='coerce')
    serv = serv.dropna(subset=['Service_Date'])

    reference_date = pd.to_datetime(filterdate)

    grouped = serv.groupby('Vin_No')['Service_Date'].agg(
        PMS_Count_Prior='count', First_PMS_Date='min',
    ).reset_index()

    years = (reference_date - grouped['First_PMS_Date']).dt.days / 365.25
    grouped['Years_Since_First_PMS'] = years.clip(lower=min_years)
    grouped['PMS_Freq_PerYear'] = grouped['PMS_Count_Prior'] / grouped['Years_Since_First_PMS']

    return grouped[['Vin_No', 'PMS_Freq_PerYear', 'PMS_Count_Prior', 'Years_Since_First_PMS']]


def calculate_revenue_spend(df, last_service):
    """Sum service milestone revenue columns (10K..target) into PMSRevenue."""
    df = df.copy()
    service_cols = [col for col in df.columns if col.endswith("k")]
    df[service_cols] = df[service_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    cols_to_sum = [f"{i}K" for i in range(10, last_service - 10 + 1, 10) if f"{i}K" in df.columns]
    df["PMSRevenue"] = df[cols_to_sum].sum(axis=1) if cols_to_sum else 0
    return df




def red_revenue_split(crit_list, status_list, revenue_list):
    """Split Red-criticality revenue by Lost/Deferred/Invoiced percentage."""
    red_lost = red_deferred = red_invoiced = 0.0
    for c, s, r in zip(crit_list, status_list, revenue_list):
        if c == 'Red':
            if s == 'Lost':
                red_lost += float(r)
            elif s == 'Deferred':
                red_deferred += float(r)
            elif s == 'Invoiced':
                red_invoiced += float(r)

    total = red_lost + red_deferred + red_invoiced
    if total == 0:
        return pd.Series({'Red_Lost_Revenue_Pct': 0.0, 'Red_Deferred_Revenue_Pct': 0.0, 'Red_Invoiced_Revenue_Pct': 0.0})
    return pd.Series({
        'Red_Lost_Revenue_Pct': (red_lost / total) * 100,
        'Red_Deferred_Revenue_Pct': (red_deferred / total) * 100,
        'Red_Invoiced_Revenue_Pct': (red_invoiced / total) * 100,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 8. APPOINTMENT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def map_appointments_to_services(appointments_df, service_df, svc=None):
    """Map each appointment to whether a service occurred within the booking window."""
    appointments_df = appointments_df.copy()
    service_df = service_df.copy()

    if svc is not None:
        service_df = service_df.query("Service_Num <= @svc")

    appointments_df.rename(columns={'WIP_VEH_CHASSIS': 'Vin_No'}, inplace=True)
    appointments_df["WIP_DATECREATED"] = pd.to_datetime(appointments_df["WIP_DATECREATED"], dayfirst=True)
    appointments_df["Date_Due_In_+10 Days"] = pd.to_datetime(appointments_df["Date_Due_In_+10 Days"], dayfirst=True)
    service_df["Service_Date"] = pd.to_datetime(service_df["Service_Date"], dayfirst=True)
    appointments_df = appointments_df.query("WIP_DATECREATED <= '2025-06-30'")
    appointments_df = appointments_df[appointments_df['Vin_No'].isin(service_df['Vin_No'].unique())]

    results = []
    for _, appt in appointments_df.iterrows():
        vin = appt["Vin_No"]
        start = appt["WIP_DATECREATED"]
        end = appt["Date_Due_In_+10 Days"]

        matched = service_df[
            (service_df["Vin_No"] == vin) &
            (service_df["Service_Date"] >= start) &
            (service_df["Service_Date"] <= end)
        ]
        service_nums = matched["Service_Num"].tolist()
        results.append({
            "Vin_No": vin,
            "WIP_DATECREATED": start,
            "Date_Due_In_+10 Days": end,
            "appointment_booked_showed_up": int(len(matched) > 0),
            "matched_service_count": len(set(service_nums)),
            "matched_service_nums": service_nums,
        })
    return pd.DataFrame(results)


def build_final_pms_appointment_summary(appt_df, service_df):
    """Summarize PMS appointment show-up per VIN."""
    last_pms = service_df.rename(columns={'VIN': 'Vin_No'})
    merged = appt_df.merge(last_pms, on="Vin_No", how="left")

    results = []
    for vin, group in merged.groupby("Vin_No"):
        pms_service = group["Service_Num"].iloc[0]
        showed = int(group["matched_service_nums"].apply(lambda lst: pms_service in lst).any())
        total = int(group["appointment_booked_showed_up"].sum())
        results.append({
            "Vin_No": vin, "PMS_Service": pms_service,
            "appointment_booked_showed_up": showed,
            "total_appointments_showed_up": total,
        })
    return pd.DataFrame(results)


def adjust_for_target_pms(final_df, appt_df, target_pms):
    """Adjust appointment counts for target PMS prediction."""
    prev_pms = target_pms - 10
    final_df = final_df.copy()
    appt_df = appt_df.copy()

    final_df["total_appointments_showed_up"] -= final_df["appointment_booked_showed_up"]

    prev_flag = (
        appt_df.groupby("Vin_No")["matched_service_nums"]
        .apply(lambda lists: int(any(prev_pms in lst for lst in lists)))
        .reset_index().rename(columns={"matched_service_nums": "showed_prev_pms"})
    )
    final_df = final_df.merge(prev_flag, on="Vin_No", how="left")
    mask = final_df["PMS_Service"] == target_pms
    final_df.loc[mask, "appointment_booked_showed_up"] = final_df.loc[mask, "showed_prev_pms"]
    return final_df.drop(columns=["showed_prev_pms"])


def _derive_vehicle_magic(service_df):
    """Extract Vehicle Magic from Vehicle_Key."""
    service_df = service_df.copy()
    service_df["Vehicle Magic"] = service_df["Vehicle_Key"].str.split("-", n=1).str[1]
    service_df["Vehicle Magic"] = service_df["Vehicle Magic"].str.split("-", n=1).str[0]
    service_df["Vehicle Magic"] = service_df["Vehicle Magic"].astype(int)
    return service_df


def derive_appointment_show_features(appt_df, service_df, serv_code, filter_date,
                                     vehicle_col="Vehicle Magic", status_col="WIP Status New",
                                     booking_date_col="Due Date IN"):
    """Show/no-show appointment features per Vehicle Magic."""
    df = appt_df.copy()
    df["Service_Num"] = df["WIP_SERVCODE"].apply(derive_servcode)

    service_df = _derive_vehicle_magic(service_df)
    df = df[df['Vehicle Magic'].isin(service_df['Vehicle Magic'].unique())]

    df[booking_date_col] = pd.to_datetime(df[booking_date_col], errors="coerce", dayfirst=True)
    filter_date = pd.to_datetime(filter_date)

    df = df[df[booking_date_col] <= filter_date]
    df = df.query("Service_Num < @serv_code")

    df[status_col] = df[status_col].astype(str).str.strip().str.title()
    df["show_flag"] = (df[status_col] == "Show").astype(int)
    df["no_show_flag"] = (df[status_col] == "No Show").astype(int)

    features = df.groupby(vehicle_col).agg(
        total_appointments_showed_up=("show_flag", "sum"),
        no_of_appointments_booked_but_not_showed_up=("no_show_flag", "sum"),
    ).reset_index()
    features["appointment_booked_showed_up_atleastonce"] = (features["total_appointments_showed_up"] > 0).astype(int)
    return features


def compute_late_appointment_metrics(appt_df, service_df, filter_date, svc=None):
    """Late appointment count and avg delay per Vehicle Magic."""
    appt_df = appt_df.copy()
    service_df = service_df.copy()

    if svc is not None and svc != 0:
        service_df = service_df.query("Service_Num > 1 and Service_Num < @svc")
    elif svc is None:
        service_df = service_df.query("Service_Num > 1")

    service_df = _derive_vehicle_magic(service_df)

    appt_df["Due Date IN"] = pd.to_datetime(appt_df["Due Date IN"], errors="coerce", dayfirst=True)
    service_df["Service_Date"] = pd.to_datetime(service_df["Service_Date"], errors="coerce", dayfirst=True)
    appt_df = appt_df[appt_df['Vehicle Magic'].isin(service_df["Vehicle Magic"].unique())]

    appt_df["Service_Num"] = appt_df["WIP_SERVCODE"].apply(derive_servcode)
    if svc is not None and svc != 0:
        appt_df = appt_df.query("`Due Date IN` <= @filter_date and Service_Num > 1")
    elif svc is None:
        appt_df = appt_df.query("`Due Date IN` <= @filter_date and Service_Num > 1")
    else:
        appt_df = appt_df.query("`Due Date IN` <= @filter_date")

    showed_up = appt_df[appt_df["No Show"] == 0]
    service_dedup = service_df.groupby(["Vehicle Magic", "Service_Num"], as_index=False).agg({"Service_Date": "min"})
    merged = showed_up.merge(service_dedup, on=["Vehicle Magic", "Service_Num"], how="left")
    merged["appoin_delay_days"] = (merged["Service_Date"] - merged["Due Date IN"]).dt.days
    merged["is_late_appointment"] = (merged["appoin_delay_days"] > 0).astype(int)

    return (
        merged[merged["is_late_appointment"] == 1]
        .groupby("Vehicle Magic")
        .agg(no_of_late_appointments=("is_late_appointment", "sum"), Avg_appointment_delay_days=("appoin_delay_days", "mean"))
        .reset_index()
    )


def compute_last_appointment_status_with_constant_service_code(appt_df, service_df, last_service_code, filter_date):
    """Last appointment status and delay per Vehicle Magic."""
    appt_df = appt_df.copy()
    service_df = service_df.copy()

    if last_service_code != 0:
        service_df = service_df.query("Service_Num > 1 and Service_Num < @last_service_code")

    service_df = _derive_vehicle_magic(service_df)

    appt_df["Due Date IN"] = pd.to_datetime(appt_df["Due Date IN"], errors="coerce", dayfirst=True)
    service_df["Service_Date"] = pd.to_datetime(service_df["Service_Date"], errors="coerce", dayfirst=True)
    appt_df = appt_df[appt_df['Vehicle Magic'].isin(service_df["Vehicle Magic"].unique())]

    appt_df["Service_Num"] = appt_df["WIP_SERVCODE"].apply(derive_servcode)
    if last_service_code != 0:
        appt_df = appt_df.query("`Due Date IN` <= @filter_date and Service_Num > 1")
    else:
        appt_df = appt_df.query("`Due Date IN` <= @filter_date")

    service_dedup = service_df.groupby(["Vehicle Magic", "Service_Num"], as_index=False).agg({"Service_Date": "min"})

    eligible = service_dedup[service_dedup["Service_Num"] < last_service_code]
    effective_last = (
        eligible.sort_values(["Vehicle Magic", "Service_Num"])
        .groupby("Vehicle Magic").tail(1)
        .rename(columns={"Service_Num": "Effective_Last_Service_Num"})
    )

    appt_last = appt_df.merge(
        effective_last[["Vehicle Magic", "Effective_Last_Service_Num"]],
        left_on=["Vehicle Magic", "Service_Num"],
        right_on=["Vehicle Magic", "Effective_Last_Service_Num"],
        how="left"
    )
    appt_last = appt_last[appt_last["Effective_Last_Service_Num"].notna()]

    merged = appt_last.merge(service_dedup, on=["Vehicle Magic", "Service_Num"], how="left")

    if merged.empty:
        merged["last_appointment_status"] = pd.Series(dtype="object")
        merged["last_appointment_delay_days"] = pd.Series(dtype="float64")
    else:
        delay = (merged["Service_Date"] - merged["Due Date IN"]).dt.days
        merged["last_appointment_delay_days"] = np.where(delay > 0, delay, 0)
        
        cond_no_show = (merged["No Show"] == 1) | merged["Service_Date"].isna()
        cond_late = delay > 0
        
        merged["last_appointment_status"] = np.where(
            cond_no_show, "NO_SHOW",
            np.where(cond_late, "LATE_SHOW", "IN_DUE_TIME")
        )
        merged.loc[cond_no_show, "last_appointment_delay_days"] = np.nan
    
    merged = merged.drop_duplicates(subset="Vehicle Magic", keep="first")

    return merged[["Vehicle Magic", "last_appointment_status", "last_appointment_delay_days"]]


def compute_nonpms_appointment_metrics(appt_df, service_df):
    """Appointment metrics for Non-PMS vehicles (Service_Num == 0)."""
    appt_df = appt_df.copy()
    service_df = service_df.copy()

    # Derive Vehicle Magic — handle both column variants
    if "Vehicle_Key" in service_df.columns:
        service_df["Vehicle Magic"] = service_df["Vehicle_Key"].str.split("-", n=1).str[1].str.split("-", n=1).str[0].astype(int)
    elif "VehMagic WIP" in service_df.columns:
        service_df["Vehicle Magic"] = service_df["VehMagic WIP"].str.split("-", n=1).str[0].astype(int)

    appt_df["Due Date IN"] = pd.to_datetime(appt_df["Due Date IN"], errors="coerce", dayfirst=True)
    service_df["Service Date"] = pd.to_datetime(service_df["Service_Date"], errors="coerce", dayfirst=True)
    appt_df["Service_Num"] = appt_df["WIP_SERVCODE"].apply(derive_servcode)

    appt_df = appt_df[appt_df["Service_Num"] == 0]
    service_df = service_df[service_df["Service_Num"] == 0]

    service_dedup = service_df.groupby(["Vehicle Magic", "Service_Num"], as_index=False).agg({"Service_Date": "min"})
    appt_df = appt_df[appt_df['Vehicle Magic'].isin(service_df['Vehicle Magic'].unique())]

    events = []
    showed_up = appt_df[appt_df["No Show"] == 0]
    showed_merged = showed_up.merge(service_dedup, on=["Vehicle Magic", "Service_Num"], how="left")
    showed_merged["Service_Date"] = pd.to_datetime(showed_merged["Service_Date"], errors="coerce", dayfirst=True)
    showed_merged["Due Date IN"] = pd.to_datetime(showed_merged["Due Date IN"], errors="coerce", dayfirst=True)
    showed_merged["delay_days"] = (showed_merged["Service_Date"] - showed_merged["Due Date IN"]).dt.days

    for _, r in showed_merged.iterrows():
        if pd.isna(r["Service_Date"]):
            continue
        events.append({
            "Vehicle Magic": r["Vehicle Magic"],
            "event_date": r["Service_Date"],
            "appointment_status": "LATE_SHOW" if r["delay_days"] > 0 else "IN_DUE_TIME",
            "delay_days": max(r["delay_days"], 0),
        })

    for _, r in appt_df[appt_df["No Show"] == 1].iterrows():
        events.append({
            "Vehicle Magic": r["Vehicle Magic"],
            "event_date": r["Due Date IN"],
            "appointment_status": "NO_SHOW",
            "delay_days": np.nan,
        })

    events_df = pd.DataFrame(events)
    results = []
    for vm, grp in events_df.groupby("Vehicle Magic"):
        late = grp.loc[grp["appointment_status"] == "LATE_SHOW", "delay_days"]
        last_event = grp.sort_values("event_date").iloc[-1]
        results.append({
            "Vehicle Magic": vm,
            "no_of_late_appointments": len(late),
            "Avg_appointment_delay_days": late.mean() if len(late) > 0 else np.nan,
            "last_appointment_status": last_event["appointment_status"],
            "last_appointment_delay_days": last_event["delay_days"],
        })
    return pd.DataFrame(results)


def compute_no_show_appointments_test(master_df, noshow_df, filter_date, vin_col="Vin_No", wip_deleted_col="WIP Deleted"):
    """No-show appointment count per VIN for test data."""
    master_df = master_df.copy()
    noshow_df = noshow_df.copy()

    if "VEHICLE CHASSIS" in noshow_df.columns:
        noshow_df.rename(columns={"VEHICLE CHASSIS": "VIN"}, inplace=True)

    noshow_df = noshow_df[noshow_df["VIN"].isin(master_df[vin_col].unique())]
    noshow_df[wip_deleted_col] = pd.to_datetime(noshow_df[wip_deleted_col], dayfirst=True)
    filter_date = pd.to_datetime(filter_date)
    noshow_df = noshow_df[noshow_df[wip_deleted_col].notna()]

    valid = noshow_df[noshow_df[wip_deleted_col] <= filter_date]
    counts = valid.groupby("VIN").size().reset_index(name="no_of_appointments_booked_but_not_showed_up")

    final = master_df.merge(counts, left_on=vin_col, right_on="VIN", how="left")
    final["no_of_appointments_booked_but_not_showed_up"] = final["no_of_appointments_booked_but_not_showed_up"].fillna(0).astype(int)
    final.rename(columns={"VIN": "Vin_No"}, inplace=True)
    return final[['Vin_No', "no_of_appointments_booked_but_not_showed_up"]]


def count_service_appointments_booked(appt_df, service, serv_code, filter_date,
                                      vehicle_col="Vehicle Magic", booking_date_col="WIP Booking Date"):
    """Count total service appointments booked before filter_date."""
    df = appt_df.copy()
    df = df[df['Vehicle Magic'].isin(service['Vehicle Magic'].unique())]
    df[booking_date_col] = pd.to_datetime(df[booking_date_col], format="mixed", errors="coerce")
    filter_date = pd.to_datetime(filter_date)
    df["Service_Num"] = df["WIP_SERVCODE"].apply(derive_servcode)
    df = df[df[booking_date_col] <= filter_date]
    df = df.query("Service_Num < @serv_code")
    return df.groupby(vehicle_col).size().reset_index(name="no_of_service_appointments_booked")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. COMPLAINT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def transform_complaint_features(main_df, service_df):
    """Complaint features per VIN based on last PMS service."""
    main_df = main_df.copy()
    service_df = service_df.copy()
    main_df.rename(columns={'VIN': 'Vin_No', 'Service_Num': 'Last_PMS_Service'}, inplace=True)

    if "Service_Date" in service_df.columns:
        service_df["Service_Date"] = pd.to_datetime(service_df["Service_Date"], errors="coerce", dayfirst=True)
        
    if "Complaint Closed in" not in service_df.columns:
        service_df["Complaint Closed in"] = np.nan
    if "Complaint Category" not in service_df.columns:
        service_df["Complaint Category"] = np.nan
        
    service_df["Complaint Closed in"] = pd.to_numeric(service_df["Complaint Closed in"], errors="coerce")

    merged = service_df.merge(main_df[["Vin_No", "Last_PMS_Service"]], on="Vin_No", how="inner")
    merged = merged[merged["Service_Num"] < merged["Last_PMS_Service"]]
    merged = merged.sort_values(["Vin_No", "Service_Date"])

    results = []
    for vin, grp in merged.groupby("Vin_No"):
        cats = grp["Complaint Category"].dropna()
        res_vals = grp["Complaint Closed in"].dropna()

        latest = cats.iloc[-1] if len(cats) > 0 else 'unknown'
        recent = res_vals.iloc[-1] if len(res_vals) > 0 else 0
        avg_res = res_vals.mean() if len(res_vals) > 0 else 0
        max_res = res_vals.max() if len(res_vals) > 0 else 0
        has_hist = int(len(cats) > 0)

        if has_hist == 1:
            avg_res = max(avg_res, 1)
            max_res = max(max_res, 1)
            recent = max(recent, 1)

        results.append({
            "Vin_No": vin,
            "latest_complaint_category": latest,
            "has_complaint_history": has_hist,
            "num_past_complaints": len(cats),
            "avg_complaint_resolution_days": avg_res,
            "max_complaint_resolution_days": max_res,
            "recent_complaint_resolution_days": recent,
        })
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. VHC FEATURES
# ═══════════════════════════════════════════════════════════════════════════════




def vhcpreparation(df, last_service_code, maindf):
    """Prepare VHC features: revenue buckets, criticality flags, top parts OHE (Vectorized)."""
    df = df[df['Vehicle Key'].isin(maindf['Vehicle Key'].unique())].copy()
    df = df.query("`Service Code` < @last_service_code").copy()
    df['VHC Revenue'] = pd.to_numeric(df['VHC Revenue'], errors="coerce").fillna(0)
    df['VHS Status'] = df['VHS Status'].replace({'Deleted': 'Lost'})

    vhs_to_category = {
        'Lost': 'Lost Revenue', 'Deferred': 'Deferred Revenue', 'Open': 'Deferred Revenue',
        'Pending': 'Deferred Revenue', 'In Progress': 'Deferred Revenue',
        'Invoiced': 'Invoiced Revenue', 'Closed': 'Invoiced Revenue',
    }
    df['VHC_Category'] = df['VHS Status'].map(vhs_to_category).fillna('Other VHC Revenue')

    # 1. Base Aggregation
    grouped = df.groupby(['Vehicle Key', 'Service Code'], as_index=False).agg(
        vhc_Lost=('VHS Status', lambda x: int((x == 'Lost').any())),
        vhc_Deferred=('VHS Status', lambda x: int((x == 'Deferred').any())),
        vhc_Invoiced=('VHS Status', lambda x: int((x == 'Invoiced').any())),
        VHC_Criticality_Red=('Criticality', lambda x: int((x == 'Red').any())),
        VHC_Criticality_Amber=('Criticality', lambda x: int((x == 'Amber').any()))
    )

    # 2. Revenue Pivot
    rev_pivot = df.pivot_table(
        index=['Vehicle Key', 'Service Code'], columns='VHC_Category', values='VHC Revenue', aggfunc='sum', fill_value=0
    ).reset_index()
    for c in ['Lost Revenue', 'Deferred Revenue', 'Invoiced Revenue', 'Other VHC Revenue']:
        if c not in rev_pivot.columns: rev_pivot[c] = 0.0
    
    rev_pivot['Total VHC Revenue'] = rev_pivot[['Lost Revenue', 'Deferred Revenue', 'Invoiced Revenue', 'Other VHC Revenue']].sum(axis=1)
    
    # Safe division with np.where to avoid NaNs being 0 if total is 0, wait, fillna(0) is fine
    rev_pivot['VHC Revenue Convertion'] = np.where(rev_pivot['Total VHC Revenue'] > 0, (rev_pivot['Invoiced Revenue'] / rev_pivot['Total VHC Revenue']) * 100, 0.0)
    rev_pivot['Deferred Revenue Ratio'] = np.where(rev_pivot['Total VHC Revenue'] > 0, rev_pivot['Deferred Revenue'] / rev_pivot['Total VHC Revenue'], 0.0)
    rev_pivot['VHC Revenue at Risk(%)'] = np.where(rev_pivot['Total VHC Revenue'] > 0, ((rev_pivot['Deferred Revenue'] + rev_pivot['Lost Revenue']) / rev_pivot['Total VHC Revenue']) * 100, 0.0)
    
    grouped = grouped.merge(rev_pivot, on=['Vehicle Key', 'Service Code'], how='left')

    # 3. Red Criticality Revenue
    red_df = df[df['Criticality'] == 'Red']
    if not red_df.empty:
        red_pivot = red_df.pivot_table(
            index=['Vehicle Key', 'Service Code'], columns='VHS Status', values='VHC Revenue', aggfunc='sum', fill_value=0
        ).reset_index()
        for c in ['Lost', 'Deferred', 'Invoiced']:
            if c not in red_pivot.columns: red_pivot[c] = 0.0
        
        red_pivot['Total_Red'] = red_pivot[['Lost', 'Deferred', 'Invoiced']].sum(axis=1)
        red_pivot['Red_Lost_Revenue_Pct'] = np.where(red_pivot['Total_Red'] > 0, (red_pivot['Lost'] / red_pivot['Total_Red']) * 100, 0.0)
        red_pivot['Red_Deferred_Revenue_Pct'] = np.where(red_pivot['Total_Red'] > 0, (red_pivot['Deferred'] / red_pivot['Total_Red']) * 100, 0.0)
        red_pivot['Red_Invoiced_Revenue_Pct'] = np.where(red_pivot['Total_Red'] > 0, (red_pivot['Invoiced'] / red_pivot['Total_Red']) * 100, 0.0)
        
        grouped = grouped.merge(red_pivot[['Vehicle Key', 'Service Code', 'Red_Lost_Revenue_Pct', 'Red_Deferred_Revenue_Pct', 'Red_Invoiced_Revenue_Pct']], on=['Vehicle Key', 'Service Code'], how='left')
    
    for c in ['Red_Lost_Revenue_Pct', 'Red_Deferred_Revenue_Pct', 'Red_Invoiced_Revenue_Pct']:
        if c not in grouped.columns: grouped[c] = 0.0
        else: grouped[c] = grouped[c].fillna(0.0)

    # 4. Top Parts OHE
    desc_col = 'Refined Description (Enhanced)' if 'Refined Description (Enhanced)' in df.columns else 'Refined Description'
    
    if desc_col in df.columns:
        parts_df = df.dropna(subset=[desc_col]).copy()
        parts_df[desc_col] = parts_df[desc_col].astype(str).str.strip()
        
        TOP_K = 20
        for label, col_name in [('Lost', 'LostPart__'), ('Invoiced', 'InvoicedPart__'), ('Deferred', 'DeferredPart__')]:
            sub_df = parts_df[parts_df['VHS Status'] == label]
            if not sub_df.empty:
                top_parts = [p for p, _ in Counter(sub_df[desc_col]).most_common(TOP_K)]
                sub_df = sub_df[sub_df[desc_col].isin(top_parts)]
                if not sub_df.empty:
                    crosstab = pd.crosstab([sub_df['Vehicle Key'], sub_df['Service Code']], sub_df[desc_col]).clip(upper=1)
                    crosstab.columns = [f'{col_name}{c}' for c in crosstab.columns]
                    grouped = grouped.merge(crosstab.reset_index(), on=['Vehicle Key', 'Service Code'], how='left')
                    
    # Fill NAs for parts with 0
    part_cols = [c for c in grouped.columns if c.startswith('LostPart__') or c.startswith('InvoicedPart__') or c.startswith('DeferredPart__')]
    grouped[part_cols] = grouped[part_cols].fillna(0).astype(int)

    grouped = grouped.rename(columns={'Service Code': 'Service_Num'})
    grouped['VHC Completed_Flag'] = 1
    return grouped


def map_vhc_history(main_df, history_df, svc=None):
    """Merge VHC + survey features into main dataset per VIN + Service_Num."""
    history_df = history_df.copy()
    if svc is not None:
        history_df = history_df.query("Service_Num <= @svc")
    history_df.rename(columns={'Vin_No': 'VIN'}, inplace=True)

    replace_cols = ["Survey Score", "Survey Status", "VHC Quoted", "VHC Sold", "VHC Lost Sale",
                    "VHC Lost Red Sale", "VHC Deferred", "VHC Amber Deferred", "VHC Completed"]
    history_df[replace_cols] = history_df[replace_cols].replace("-", np.nan)

    vhc_cols = ["VHC Quoted", "VHC Sold", "VHC Lost Sale", "VHC Lost Red Sale", "VHC Deferred", "VHC Amber Deferred"]
    history_df[vhc_cols] = history_df[vhc_cols].apply(lambda c: pd.to_numeric(c, errors="coerce"))
    history_df["Survey Score"] = pd.to_numeric(history_df["Survey Score"], errors="coerce")
    history_df["Survey Status"] = history_df["Survey Status"].fillna("Unknown")

    history_df["VHC Completed"] = pd.to_datetime(history_df["VHC Completed"], errors="coerce", dayfirst=True)
    history_df["VHC Completed"] = history_df["VHC Completed"].replace('-', np.nan)
    history_df["VHC Completed_Flag"] = history_df["VHC Completed"].notna().astype(int)
    history_df["Service_Date"] = pd.to_datetime(history_df["Service_Date"], errors="coerce")
    history_df = history_df.sort_values(["VIN", "Service_Date"])

    map_cols = ["VIN", "Service_Num", "Survey Score", "Survey Status", "VHC Completed_Flag"] + vhc_cols
    numeric_cols = ["Survey Score"] + vhc_cols
    categorical_cols = ["Survey Status", "VHC Completed_Flag"]

    agg_dict = {
        **{col: "last" for col in numeric_cols},
        **{col: (lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan) for col in categorical_cols},
    }
    df_out = history_df[map_cols].groupby(["VIN", "Service_Num"]).agg(agg_dict).reset_index()
    return main_df.merge(df_out, on=["VIN", "Service_Num"], how="left")


def backfill_vhc_for_target_rows(merged_df, history_df, target_service_num, vhc_cols, survey_cols,
                                  vin_col="VIN", service_col="Service_Num", targetflag_col="TargetFlag"):
    """Backfill VHC features for TargetFlag==1 rows from previous service."""
    merged_df = merged_df.copy()
    history_df = history_df.copy()
    history_df = history_df.rename(columns={'Vin_No': 'VIN'})
    history_df[vhc_cols + ["Survey Score", "Survey Status"]] = history_df[vhc_cols + ["Survey Score", "Survey Status"]].replace("-", 0)
    history_df['Survey Status'] = history_df['Survey Status'].replace(0, 'Unknown')
    history_df["VHC Completed"] = history_df["VHC Completed"].replace('-', np.nan)
    history_df["VHC Completed_Flag"] = history_df["VHC Completed"].notna().astype(int)

    history_past = history_df[history_df[service_col] < target_service_num]
    last_past = history_past.sort_values([vin_col, service_col]).groupby(vin_col).tail(1).set_index(vin_col)

    fill_cols = vhc_cols + survey_cols
    mask = (merged_df[targetflag_col] == 1) & (merged_df[service_col] == target_service_num)

    for col in fill_cols:
        merged_df.loc[mask, col] = merged_df.loc[mask, vin_col].map(last_past[col]).fillna(merged_df.loc[mask, col])
    return merged_df


def backfill_vhc_leakage_safe(merged_df, history_df, vhc_cols,
                               vin_col="VIN", service_col="Service_Num", targetflag_col="TargetFlag"):
    """Backfill VHC features — leakage-safe version using last service per VIN."""
    merged_df = merged_df.copy()
    history_df = history_df.copy()

    history_df[vhc_cols] = history_df[vhc_cols].replace("-", 0)

    last_svc = history_df.groupby(vin_col)[service_col].max().reset_index().rename(columns={service_col: "Last_Service_Num"})
    merged_df = merged_df.merge(last_svc, on=vin_col, how="left")

    mask = (merged_df[targetflag_col] == 1) & (merged_df[service_col] == merged_df["Last_Service_Num"])

    prev = history_df.merge(last_svc, on=vin_col, how="left")
    prev = prev[prev[service_col] < prev["Last_Service_Num"]]
    last_prev = prev.sort_values([vin_col, service_col]).groupby(vin_col).tail(1).set_index(vin_col)

    for col in vhc_cols:
        merged_df.loc[mask, col] = merged_df.loc[mask, vin_col].map(last_prev[col]).fillna(merged_df.loc[mask, col])

    return merged_df.drop(columns=["Last_Service_Num"])


# ═══════════════════════════════════════════════════════════════════════════════
# 11. NON-PMS EVENTS
# ═══════════════════════════════════════════════════════════════════════════════
import os
def get_non_pms_events(serv, servcode_desc, mastertrain, last_service_code):
    """
    Non-PMS event features: last event OHE, flag, and count.
    Works for both train and pred (parameter is last_service_code / svc).
    """
    if isinstance(servcode_desc, str):
        if os.path.exists(servcode_desc):
            servcode_desc = pd.read_csv(servcode_desc)
        else:
            raise FileNotFoundError(f"servcode_desc path not found: {servcode_desc}")
        
    if isinstance(servcode_desc, pd.DataFrame):
        servcode_desc = servcode_desc.dropna(how='all').copy()
        cols_map = {c.lower().strip(): c for c in servcode_desc.columns}
        key_col = cols_map.get('so_co_code') or cols_map.get('service_code')
        val_col = cols_map.get('so_co_descripn_001')
        if key_col is None or val_col is None:
            raise KeyError(f"Service code file missing expected columns. Found: {list(servcode_desc.columns)}")
        servcode_desc = dict(
            zip(
                servcode_desc[key_col].astype(str).str.strip(),
                servcode_desc[val_col].astype(str).str.strip()
            )
        )
    
    elif isinstance(servcode_desc, dict):
        pass
    else:
        try:
            servcode_desc = dict(servcode_desc)
        except Exception:
            raise TypeError("servcode_desc must be filepath, DataFrame or dict")
    serv = serv.copy()
    serv['Service_Date'] = pd.to_datetime(serv['Service_Date'], dayfirst=True, errors="coerce")
    serv['Revenue'] = pd.to_numeric(serv['Revenue'], errors='coerce').abs()

    veh = list(mastertrain['Vehicle Key'].unique())

    exclude_codes = {'ADP', 'BR4', 'CON', 'CRE', 'EXC', 'ME', 'BES', 'MES', 'C01', 'NS4', 'PNA', 'VAS', 'VAT', 'VCP', 'VHP'}
    serv = serv.sort_values(by=['Vin_No', 'Service_Date'], ascending=True)
    serv = serv.query("Service_Code not in @exclude_codes").reset_index(drop=True)
    serv = serv[serv['Vehicle_Key'].isin(veh)]

    svc_cutoff = (
        serv[serv['Service_Num'] == last_service_code]
        .sort_values('Service_Date')
        .groupby('Vin_No')['Service_Date'].first()
        .reset_index()
        .rename(columns={'Service_Date': f'First_{last_service_code}k_Date'})
    )
    serv = serv.merge(svc_cutoff, on='Vin_No', how='left')
    filtered = serv[
        (serv[f'First_{last_service_code}k_Date'].isna()) |
        (serv['Service_Date'] <= serv[f'First_{last_service_code}k_Date'])
    ].copy()
    filtered.drop(columns=[f'First_{last_service_code}k_Date'], inplace=True)

    filtered['Is_Non_PMS'] = (
        (filtered['Service_Num'] == 0) &
        (~filtered['Service_Code'].astype(str).str.contains(r'\\*', na=False))
    )

    filtered = filtered.sort_values(['Vehicle_Key', 'Service_Date'])
    lstflag = filtered.groupby('Vehicle_Key').tail(1)

    last_non_pms = lstflag[lstflag['Is_Non_PMS'].values].copy()
    last_non_pms['Last_NonPMS_Event'] = last_non_pms['Service_Code']

    top_events = last_non_pms['Last_NonPMS_Event'].value_counts().head(last_non_pms['Last_NonPMS_Event'].nunique()).index
    last_non_pms['Event_Mapped'] = last_non_pms['Last_NonPMS_Event'].apply(lambda x: x if x in top_events else 'Others')
    last_non_pms['Event_Npms'] = last_non_pms['Event_Mapped'].map(servcode_desc)

    # ── OHE: last Non-PMS event type ──────────────────────────────────
    event_ohe = pd.get_dummies(last_non_pms['Event_Npms'], prefix='LastNPMS_Event', dtype=int)
    ohe_df = pd.concat([last_non_pms[['Vin_No']], event_ohe], axis=1)

    # ── Revenue: Top 5 last Non-PMS events (matches training pipeline) ─
    top_5_events = (
        last_non_pms['Last_NonPMS_Event']
        .value_counts()
        .head(5)
        .index
    )
    last_non_pms['Event_Mapped_5'] = last_non_pms['Last_NonPMS_Event'].apply(
        lambda x: x if x in top_5_events else 'Oth'
    )
    last_non_pms['Event_Npms_5'] = (
        last_non_pms['Event_Mapped_5']
        .map(servcode_desc)
        .fillna(last_non_pms['Event_Mapped_5'])   # fallback to code if no description
    )
    last_non_pms['Rev_Col'] = 'LastNPMS_Rev_' + last_non_pms['Event_Npms_5']

    rev_pivot = (
        last_non_pms
        .assign(Revenue=last_non_pms['Revenue'].abs())
        .pivot_table(
            index='Vin_No',
            columns='Rev_Col',
            values='Revenue',
            aggfunc='last'
        )
        .reset_index()
        .fillna(0)
    )
    rev_pivot.columns.name = None

    # ── Flag column ───────────────────────────────────────────────────
    lstflag = lstflag.copy()
    lstflag['Last_Service_Non_PMS_Flag'] = lstflag['Is_Non_PMS'].astype(int)
    lstflag = lstflag[['Vin_No', 'Last_Service_Non_PMS_Flag']]

    # ── Final merge ───────────────────────────────────────────────────
    Npmsevents = (
        lstflag
        .merge(ohe_df, on='Vin_No', how='left')
        .merge(rev_pivot, on='Vin_No', how='left')
        .fillna(0)
    )

    non_pms_count = (
        filtered[filtered['Is_Non_PMS']]
        .groupby(['Vin_No', 'Service_Date']).size()
        .reset_index(name='nNPMS')
    )
    non_pms_countn = non_pms_count.groupby('Vin_No').size().reset_index(name='nNPMS')

    return Npmsevents, non_pms_countn


def get_last_nonpms_mileage(df_nonpms, mastertrain, vin_col="Vin_No", mileage_col="Mileage", date_col="Service_Date"):
    """Last Non-PMS mileage for TargetFlag==0 VINs (training)."""
    vins = mastertrain.query("TargetFlag == 0")['VIN'].unique()
    last_pms_df = mastertrain.query("TargetFlag == 0")[['VIN', 'Last PMS Mileage']].rename(columns={'VIN': 'Vin_No'})

    df_nonpms = df_nonpms.copy()
    df_nonpms[date_col] = pd.to_datetime(df_nonpms[date_col])
    df_nonpms = df_nonpms[df_nonpms['Vin_No'].isin(vins)]
    df_nonpms = df_nonpms.sort_values([vin_col, date_col], ascending=[True, False])
    df_nonpms = df_nonpms.query("Service_Num == 0")

    last_nonpms = df_nonpms.groupby(vin_col).first().reset_index()
    last_nonpms.rename(columns={mileage_col: 'LastNonPMSMileage'}, inplace=True)
    last_nonpms = last_nonpms.merge(last_pms_df, on='Vin_No', how='left')
    last_nonpms["LastNonPMSMileage"] = last_nonpms["LastNonPMSMileage"].fillna(last_nonpms["Last PMS Mileage"])
    return last_nonpms[[vin_col, 'LastNonPMSMileage']]


def get_last_nonpms_before_targetpms(df_nonpms, vin_col, mileage_col, date_col, mastertrain):
    """Last Non-PMS mileage before target PMS date for TargetFlag==1 VINs."""
    vins = mastertrain.query("TargetFlag == 1")['VIN'].unique()
    last_pms_df = mastertrain.query("TargetFlag == 1")[['VIN', 'Last Service Date - PMS', 'Last PMS Mileage']].rename(columns={'VIN': 'Vin_No'})

    df_nonpms = df_nonpms.copy()
    df_nonpms[date_col] = pd.to_datetime(df_nonpms[date_col])
    df_nonpms = df_nonpms[df_nonpms['Vin_No'].isin(vins)]

    merged = df_nonpms.merge(last_pms_df, on=vin_col, how="left")
    valid = merged[merged[date_col] < merged["Last Service Date - PMS"]]

    last_nonpms = (
        valid.sort_values([vin_col, date_col], ascending=[True, False])
        .groupby(vin_col).first()[[mileage_col]]
        .rename(columns={mileage_col: "LastNonPMSMileage"})
    )

    result = last_pms_df.merge(last_nonpms, on=vin_col, how="left")
    return result[['Vin_No', 'LastNonPMSMileage']]


def get_last_nonpms_mileagepred(df_nonpms, svc, mastertrain, vin_col="Vin_No", mileage_col="Mileage", date_col="Service_Date"):
    """Last Non-PMS mileage for prediction pipeline."""
    vins = mastertrain['VIN'].unique()
    last_pms_df = mastertrain[['VIN', 'Last PMS Mileage']].rename(columns={'VIN': 'Vin_No'})

    df_nonpms = df_nonpms.copy()
    df_nonpms[date_col] = pd.to_datetime(df_nonpms[date_col])
    df_nonpms = df_nonpms.query("Service_Num <= @svc")
    df_nonpms = df_nonpms[df_nonpms['Vin_No'].isin(vins)]
    df_nonpms = df_nonpms.sort_values([vin_col, date_col], ascending=[True, False])
    df_nonpms = df_nonpms.query("Service_Num == 0")

    last_nonpms = df_nonpms.groupby(vin_col).first().reset_index()
    last_nonpms.rename(columns={mileage_col: 'LastNonPMSMileage'}, inplace=True)
    last_nonpms = last_nonpms.merge(last_pms_df, on='Vin_No', how='left')
    last_nonpms["LastNonPMSMileage"] = last_nonpms["LastNonPMSMileage"].fillna(last_nonpms["Last PMS Mileage"])
    return last_nonpms[[vin_col, 'LastNonPMSMileage']]


# ═══════════════════════════════════════════════════════════════════════════════
# 12. ENCODING & IMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def one_hot(df):
    """One-hot encode all object columns (starting from 4th)."""
    cat_cols = df.select_dtypes(include='O').keys().tolist()
    for column in cat_cols[3:]:
        temp = pd.get_dummies(df[column], prefix=column, dtype=int)
        df = pd.merge(left=df, right=temp, left_index=True, right_index=True)
        df = df.drop(columns=column)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 13. CLUSTERING & MAPPING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def map_cluster(df_vehicles, df_clusters, vehicle_nat_col, cluster_nat_col, cluster_id_col):
    """Map vehicle nationality/model/variant to cluster ID via exploded lookup."""
    df2 = df_clusters.copy()
    df1 = df_vehicles.copy()

    df2[cluster_nat_col] = df2[cluster_nat_col].apply(normalize_to_list)
    df2_exploded = df2.explode(cluster_nat_col).reset_index(drop=True)

    df2_exploded[cluster_nat_col] = df2_exploded[cluster_nat_col].astype(str).str.strip()
    df1[vehicle_nat_col] = df1[vehicle_nat_col].astype(str).str.strip()

    merged = df1.merge(
        df2_exploded[[cluster_id_col, cluster_nat_col]],
        left_on=vehicle_nat_col, right_on=cluster_nat_col, how='left'
    )
    return merged.drop(columns=[cluster_nat_col])


# ═══════════════════════════════════════════════════════════════════════════════
# 14. PREDICTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def months_to_days(months, start_date=None):
    """Convert months to approximate days."""
    if start_date is None:
        start_date = datetime.today()
    return (start_date + relativedelta(months=int(months)) - start_date).days


def calc_next_due(last_date, interval_months):
    """Calculate next due date from last service date + interval months."""
    if pd.isna(last_date) or pd.isna(interval_months):
        return pd.NaT
    whole = int(interval_months)
    frac = interval_months - whole
    next_due = last_date + relativedelta(months=whole)
    days_in_month = pd.Period(next_due.strftime("%Y-%m")).days_in_month
    next_due += pd.Timedelta(days=frac * days_in_month)
    return next_due.strftime("%Y-%m-%d")


def calculatenxt_service_interval(dfaa, target_mileage):
    """Compute predicted_interval_months = Avg_Service_Interval_PMS1 * remaining service units."""
    dfaa = dfaa.copy()
    dfaa['last_service_no'] = dfaa['Service_Num'] / 10
    target_no = target_mileage / 10
    dfaa['predicted_interval_months'] = dfaa['Avg_Service_Interval_PMS1'] * (target_no - dfaa['last_service_no'])
    return dfaa[['Vin_No', 'predicted_interval_months']]


def find_previous_service_mileage(service_histories, target, original_target=None):
    """Find the mileage of the closest previous service before target."""
    if not service_histories:
        return None
    if original_target is None:
        original_target = target
    buffer = original_target * 1000 + 10000
    current = target - 10

    while current >= 10:
        matches = [s for s in service_histories if s.get('Service_Num', 0) == current]
        if matches:
            best = min(matches, key=lambda s: abs((s.get('Mileage') or 0) - buffer))
            return {
                'Vin_No': best.get('Vin_No'),
                'Mileage': best.get('Mileage'),
                'Service_Date': best.get('Service_Date'),
                'Service_Num': best.get('Service_Num'),
            }
        current -= 10
    return None


def group_by_vin(service_records):
    """Group service records list by Vin_No."""
    groups = defaultdict(list)
    for rec in service_records:
        groups[rec.get("Vin_No")].append(rec)
    return groups


def find_previous_service_for_all_vins(records, target_service_num):
    """Find previous service mileage for all VINs."""
    groups = group_by_vin(records)
    return {vin: find_previous_service_mileage(hist, target_service_num) for vin, hist in groups.items()}

def ensure_list(x):
    if isinstance(x, list):
        return x
    return [x]

def compute_revenue_buckets(vhs_list, revenue_list, vhs_to_category):
    buckets = {
        'Lost Revenue': 0.0,
        'Deferred Revenue': 0.0,
        'Invoiced Revenue': 0.0,
        'Other VHC Revenue': 0.0
    }
    
    for status, rev in zip(vhs_list, revenue_list):
        category = vhs_to_category.get(status, 'Other VHC Revenue')
        buckets[category] += float(rev)
    
    return pd.Series(buckets)

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

def compute_pms_delay(df, target_milestone):
    """Return a FRAME of four candidate PMS_Delay definitions, for A/B on identical rows.

    All four are "services behind schedule" = expected - actual. They differ in where each term
    comes from, and every difference is a leak question.

      PMS_Delay            EDAexp - EDAact               legacy / current production
      PMS_Delay_Doc        EDAexp - EDAact - TargetFlag  the doc's turn-up -1
      PMS_Delay_DerivedA   EDAexp - sum(has_x)           actual from cutoff-bound history
      PMS_Delay_DerivedB   2*YSFP - sum(has_x)           both terms from cutoff-bound history

    THE ACTUAL TERM. Measured 2026-08-16: Vehicle_Key_Actual_Service == sum(has_x) + TargetFlag,
    to three decimals, on both matrices --

        60k neg 2.506/2.501, 60k pos 3.363/4.350 (gap 0.987)
        20k neg 1.000/1.000, 20k pos 0.720/1.720 (gap 1.000)

    i.e. the EDA count includes the target service for turn-ups and only for turn-ups. So the
    legacy formula ALREADY carries a -1 on positives, implicitly; PMS_Delay_Doc makes it -2.
    sum(has_x) is that same count with the label term removed.

    THE EXPECTED TERM. Vehicle_Key_ExpectedServices == 2 * `Vehicle Age` -- 53.2% exact, 99.90%
    within 1 (a semi-annual schedule; Vehicle Age is an integer 0-10). `Vehicle Age` is in
    LEAK_COLS, so PMS_Delay smuggles it back in exactly as it smuggles Vehicle_Key_Actual_Service
    back in. It is also measured at EXTRACT, not at the cutoff: reproducing Vehicle Age as
    floor((ref - Invoice date)/365.25) matches 4.88% at ref=2025-12-31 (the training cutoff) but
    90.44% at ref=2026-12-06 (the extract). Re-deriving from sale date is not viable -- Invoice
    date parses for only 60,411/130,759 EDA rows (46%) -- but Years_Since_First_PMS, built by
    derive_pms_frequency() and anchored on filterdate, is 100% populated, so 2*YSFP stands in.

    A CONSTANT expected term (target_milestone / 10) was used here until 2026-08-12. That made
    PMS_Delay a pure linear function of Vehicle_Key_Actual_Service -- measured correlation exactly
    -1.0000 -- and it agreed with the legacy definition only 0.4346. On the 20k Q1 test set the
    rule `PMS_Delay <= 0` then reproduced TargetFlag on 98.84% of rows, beating the trained model.
    Do not reintroduce the constant.

    Caller must invoke this while Vehicle_Key_ExpectedServices is still present -- both pipelines
    drop it later (refactored:833 / legacy:2675) -- and after the has_x and Years_Since_First_PMS
    merges. Selection of which variant reaches a model happens later, in
    pms_model.select_pms_delay_variant(); this function only emits the candidates.

    20k CAVEAT: only has_10 exists (range(10, 20, 10)), so sum(has_x) is 0/1 and the derived
    variants are coarse there. The 1k service (Description == '1') is off the 10k grid and has no
    flag of its own.
    """
    out = pd.DataFrame(index=df.index)

    if 'Vehicle_Key_Actual_Service' in df.columns:
        actual_services = pd.to_numeric(df['Vehicle_Key_Actual_Service'], errors='coerce').fillna(0)
    elif 'Service_Num' in df.columns:
        actual_services = pd.to_numeric(df['Service_Num'], errors='coerce').fillna(0)
    else:
        actual_services = pd.Series(0, index=df.index)

    if 'Vehicle_Key_ExpectedServices' in df.columns:
        expected_services = pd.to_numeric(df['Vehicle_Key_ExpectedServices'], errors='coerce')
        n_missing = int(expected_services.isna().sum())
        if n_missing:
            # Per-row fallback only. Filling the whole column with the constant would silently
            # restore the leaky definition, so say so loudly when it happens at all.
            print(f"WARNING compute_pms_delay: Vehicle_Key_ExpectedServices missing for "
                  f"{n_missing}/{len(df)} rows; those fall back to the constant "
                  f"{target_milestone / 10:g}")
            expected_services = expected_services.fillna(target_milestone / 10)
    else:
        print("WARNING compute_pms_delay: Vehicle_Key_ExpectedServices absent -- falling back to "
              "the CONSTANT expected term. PMS_Delay will be a relabelled copy of "
              "Vehicle_Key_Actual_Service (a label leak). Fix the caller.")
        expected_services = pd.Series(target_milestone / 10, index=df.index)

    out['PMS_Delay'] = expected_services - actual_services

    if 'TargetFlag' in df.columns:
        out['PMS_Delay_Doc'] = out['PMS_Delay'] - pd.to_numeric(
            df['TargetFlag'], errors='coerce').fillna(0)
    else:
        # Inference frames have no label. Emitting the column unchanged keeps the schema stable;
        # emitting a zero-shifted copy would silently make the variant a different feature.
        print("WARNING compute_pms_delay: TargetFlag absent -- PMS_Delay_Doc falls back to "
              "PMS_Delay. Expected at inference, a bug during training.")
        out['PMS_Delay_Doc'] = out['PMS_Delay']

    has_cols = [f'has_{x}' for x in range(10, target_milestone, 10) if f'has_{x}' in df.columns]
    if has_cols:
        prior = df[has_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
        out['PMS_Delay_DerivedA'] = expected_services - prior

        if 'Years_Since_First_PMS' in df.columns:
            ysfp = pd.to_numeric(df['Years_Since_First_PMS'], errors='coerce').fillna(0)
            out['PMS_Delay_DerivedB'] = (2 * ysfp).clip(0, 21) - prior
        else:
            print("WARNING compute_pms_delay: Years_Since_First_PMS absent -- skipping "
                  "PMS_Delay_DerivedB. Run derive_pms_frequency() and merge it first.")
    else:
        # Emitting zeros here would make a derived variant look like a leak-free copy of the
        # expected term. Skip the columns instead so the omission is visible downstream.
        print(f"WARNING compute_pms_delay: no has_* columns for milestone {target_milestone} -- "
              "skipping PMS_Delay_DerivedA and _DerivedB. Merge the milestone flags first.")

    return out


#: Every column compute_pms_delay() can emit. The frame carries all of them through feature-eng;
#: pms_model.select_pms_delay_variant() collapses them to one named `PMS_Delay` at train/score time.
PMS_DELAY_COLS = ['PMS_Delay', 'PMS_Delay_Doc', 'PMS_Delay_DerivedA', 'PMS_Delay_DerivedB']


#: Median months from first service to each milestone, measured 2026-08-16 over the full
#: Service_History_Q2-2026 file (14,972-47,851 vehicles per milestone, lags clipped to 0-180 months).
#: The hardcoded schedule assumes 6 months per 10k; reality is 3.25-4.40 and drifts DOWN as the
#: milestone rises, so the error compounds: 12 vs 8.8 months at 20k, 36 vs 22.1 at 60k, 60 vs 32.5
#: at 100k. That bias is why --label-mode window lost AUC -- it filtered on due dates that ran up to
#: a year late, worst at 40k/60k. See the CLAUDE.md gotcha.
MILESTONE_MONTHS_CALIBRATED = {20: 9, 30: 12, 40: 16, 50: 19, 60: 22, 70: 25, 80: 27, 90: 30,
                               100: 32}


def expected_milestone_months(milestone, date_model='schedule'):
    """Months from first service to when `milestone` is expected. Both pipelines must agree.

    'schedule'   -- 6 months per 10k, the original hardcoded rule. Default, so nothing moves
                    unless asked.
    'calibrated' -- the measured medians above, falling back to the schedule for any milestone not
                    in the table.
    """
    if date_model not in ('schedule', 'calibrated'):
        raise ValueError(f"date_model must be 'schedule' or 'calibrated', got {date_model!r}")
    schedule = int(milestone / 10) * 6
    if date_model == 'calibrated':
        return MILESTONE_MONTHS_CALIBRATED.get(int(milestone), schedule)
    return schedule

from sklearn.metrics import silhouette_score

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
from scipy.cluster.hierarchy import linkage, fcluster
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
        # 'Service Frequency' (EDA) replaced 2026-08-13 by the derived, non-leaky
        # 'PMS_Freq_PerYear' -- see derive_pms_frequency().
        "PMS_Freq_PerYear": "mean",
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
    # scipy linkage expects a condensed (1D) distance matrix; squareform converts
    from scipy.spatial.distance import squareform
    gower_condensed = squareform(gower_dist, checks=False)
    silhouette_scores = []
    K = range(k_min, k_max+1)
    # silhouette_score requires 2 <= n_labels <= n_samples-1. n_samples here is the number of
    # DISTINCT `feat` categories (groupby above), not the cohort's row count -- a small quarterly
    # cohort (candidates mode reaches back to 2016, with cohorts as small as ~7-165 rows) can easily
    # have fewer than k_max+1 distinct Nationality/Model/Variant values, so a fixed range(2, 11)
    # crashes for k > n_samples-1. Harmless to skip: best_k is hardcoded to 4 below, so these scores
    # are never actually used to choose anything -- this loop only needs to not crash.
    n_samples = gower_dist.shape[0]

    for k in K:
        linkage_matrix = linkage(gower_condensed, method=method)
        labels = fcluster(linkage_matrix, k, criterion="maxclust")

        if 1 < k <= n_samples - 1:
            silhouette_scores.append(silhouette_score(gower_dist, labels, metric="precomputed"))
        else:
            silhouette_scores.append(None)

    # --- Step 4: Select best k (max silhouette) ---
    # best_k = K[np.argmax(silhouette_scores)]
    best_k = 4

    # --- Step 5: Final clustering ---
    linkage_matrix = linkage(gower_condensed, method=method)
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

