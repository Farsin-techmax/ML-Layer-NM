"""
predservicemil_4_refactored.py — PMS Prediction Pipeline (Refactored)

This file builds the prediction dataset for the PMS service model.
All shared feature-engineering functions have been moved to features.py
and are imported here instead of being redefined.

REMOVED DUPLICATES (all now imported from features.py):
  ┌──────────────────────────────────────────────────┬────────────────────────────────────┐
  │ Function (was in predservicemil 4.py)             │ features.py equivalent             │
  ├──────────────────────────────────────────────────┼────────────────────────────────────┤
  │ extract_kk()                                     │ extract_kk()                       │
  │ extract_k1()                                     │ extract_k1()                       │
  │ derive_servcode()                                │ derive_servcode()                  │
  │ ensure_list()                                    │ ensure_list()                      │
  │ normalize_to_list()                              │ normalize_to_list()                │
  │ clean_parts()                                    │ clean_parts()                      │
  │ derive_pms_features1()                           │ derive_pms_features1()             │
  │ derive_npms_features()                           │ derive_npms_features()             │
  │ derive_pms_mileage_features()                    │ derive_pms_mileage_features()      │
  │ derive_npms_mileage_features()                   │ derive_npms_mileage_features()     │
  │ derive_pms_service_intervals()                   │ derive_pms_service_intervals()     │
  │ adjust_service_intervalspred()                   │ adjust_service_intervals()         │
  │ derive_npms_features2()                          │ derive_npms_features2()            │
  │ branch_visit_features()                          │ branch_visit_features()            │
  │ branch_diversity_features()                      │ branch_diversity_features()        │
  │ calculate_bodyshop_count()                       │ calculate_bodyshop_count()         │
  │ compute_service_features()                       │ compute_service_features()         │
  │ calculate_revenue_spend()                        │ calculate_revenue_spend()          │
  │ compute_revenue_buckets()                        │ compute_revenue_buckets()          │
  │ red_revenue_split()                              │ red_revenue_split()                │
  │ split_parts_by_status()                          │ split_parts_by_status()            │
  │ vhcpreparation()                                 │ vhcpreparation()                   │
  │ map_vhc_history()                                │ map_vhc_history()                  │
  │ get_non_pms_events()                             │ get_non_pms_events()               │
  │ get_last_nonpms_mileagepred()                    │ get_last_nonpms_mileagepred()      │
  │ map_appointments_to_services()                   │ map_appointments_to_services()     │
  │ build_final_pms_appointment_summary()            │ build_final_pms_appointment_summary() │
  │ compute_no_show_appointments_test()              │ compute_no_show_appointments_test()│
  │ compute_late_appointment_metrics()               │ compute_late_appointment_metrics() │
  │ compute_last_appointment_status_...()            │ compute_last_appointment_status_...() │
  │ derive_appointment_show_features()               │ derive_appointment_show_features() │
  │ count_service_appointments_booked()              │ count_service_appointments_booked()│
  │ compute_nonpms_appointment_metrics() [x2 defs]   │ compute_nonpms_appointment_metrics() │
  │ transform_complaint_features()                   │ transform_complaint_features()     │
  │ one_hot()                                        │ one_hot()                          │
  │ map_cluster()                                    │ map_cluster()                      │
  │ months_to_days()                                 │ months_to_days()                   │
  │ calc_next_due()                                  │ calc_next_due()                    │
  │ calculatenxt_service_interval()                  │ calculatenxt_service_interval()    │
  └──────────────────────────────────────────────────┴────────────────────────────────────┘

NOTE: compute_nonpms_appointment_metrics() was defined TWICE in the original file
      (lines 1078 and 1719) with slightly different Vehicle Magic derivation logic.
      features.py has a single unified version that handles both column variants.
"""

import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
import sys
import re
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


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS FROM SHARED MODULE
# All 40+ functions below were previously copy-pasted into this file.
# They now live in features.py as the single source of truth.
# ═══════════════════════════════════════════════════════════════════════════════
from features import (
    # --- Parsing & Utilities ---
    extract_kk,
    extract_k1,
    derive_servcode,
    ensure_list,
    normalize_to_list,
    clean_parts,

    # --- PMS / NPMS Date Features ---
    derive_pms_features1,
    derive_npms_features,

    # --- Mileage Features ---
    derive_pms_mileage_features,
    derive_npms_mileage_features,

    # --- Service Intervals ---
    derive_pms_service_intervals,
    adjust_service_intervals as adjust_service_intervalspred,  # alias kept for backward compat
    calculatenxt_service_interval,

    # --- NPMS Aggregates ---
    derive_npms_features2,

    # --- Branch Features ---
    branch_visit_features,
    branch_diversity_features,
    calculate_bodyshop_count,

    # --- Revenue Features ---
    compute_service_features,
    calculate_revenue_spend,
    compute_revenue_buckets,
    red_revenue_split,

    # --- Appointment Features ---
    map_appointments_to_services,
    build_final_pms_appointment_summary,
    compute_no_show_appointments_test,
    compute_late_appointment_metrics,
    compute_last_appointment_status_with_constant_service_code,
    derive_appointment_show_features,
    count_service_appointments_booked,
    compute_nonpms_appointment_metrics,

    # --- Complaint Features ---
    # transform_complaint_features,


    # --- VHC Features ---
    split_parts_by_status,
    vhcpreparation,
    map_vhc_history,

    # --- Non-PMS Events ---
    get_non_pms_events,
    get_last_nonpms_mileagepred,

    # --- Encoding ---
    one_hot,

    # --- Clustering & Mapping ---
    map_cluster,

    # --- Prediction Helpers ---
    months_to_days,
    calc_next_due,
    compute_pms_delay
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Line %(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('pms_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION-SPECIFIC FUNCTION: NonPMS
# This function handles vehicles that have NEVER had a PMS service.
# It estimates when their first PMS would be due based on invoice date.
# This is NOT in features.py because it's specific to the prediction pipeline.
# ═══════════════════════════════════════════════════════════════════════════════

def NonPMS(rfmdf, filterdate, last_service_code, custom_date, natclus, varclus, modclus,
           appoinshow, appoinoshow, appointdfN, servcode_desc):
    """
    Build prediction features for vehicles with NO PMS history (Non-PMS cohort).

    These are vehicles that were sold (New Only) but have never visited for a
    scheduled PMS service. Their next-due date is estimated from invoice date
    using a default interval of 6 months per 10k service.

    Parameters
    ----------
    rfmdf : pd.DataFrame - RFM segments per customer
    filterdate : str - data cutoff date (e.g. '2025-09-30')
    last_service_code : int - target service code (e.g. 60)
    custom_date : datetime - filterdate + 1 day
    natclus, varclus, modclus : pd.DataFrame - cluster mapping files
    appoinshow, appoinoshow : pd.DataFrame - appointment data
    appointdfN : pd.DataFrame - full appointment data
    servcode_desc : pd.DataFrame - service code descriptions
    """
    eda = pd.read_csv('data/EDA Datasheet Till 2025.csv', low_memory=False, encoding='ISO-8859-1')
    print(f'''EDA shape before filtering PMS: {eda.shape}''')
    serv1 = pd.read_csv('data/Service History Feb 2026.csv', low_memory=False)
    serv1['Service_Date'] = pd.to_datetime(serv1['Service_Date'], format='mixed', dayfirst=True, errors='coerce')
    serv1 = serv1.query("Service_Date <= @filterdate")
    serv = serv1.query("Description != 'Others'")

    # --- Filter to: New vehicles, never had PMS, with valid sale invoice ---
    neweda = eda[
        (eda['New / Used Category'] == 'New Only') &
        (eda['New / Used'] == 'NEW') &
        (eda['Last Service - PMS'] == '-') &
        (~eda['Sale Invoice Year'].isna()) &
        (eda['Sale Invoice Year'] != '-')
    ]
    neweda = neweda[~neweda['VIN'].isin(serv['Vin_No'].unique())]
    print(f'''Neweda shape after filtering PMS: {neweda.shape}''')

    # --- Merge RFM segments and estimate due date ---
    neweda = neweda.merge(rfmdf[['Customer ID', 'RFM_segments']], on=['Customer ID'], how='left')
    neweda['Invoice date'] = pd.to_datetime(neweda['Invoice date'], format='mixed', dayfirst=True)

    # Cutoff: only vehicles sold recently enough to still be candidates
    # e.g., for 60k target: (60/10)*6 = 36 months back from custom_date
    cutoff_date = custom_date - pd.DateOffset(months=(last_service_code / 10) * 6)
    print(f'Cutoff Date: {cutoff_date}')
    neweda = neweda[neweda['Invoice date'] >= cutoff_date]
    print(f'Neweda shape after filtering Invoice date: {neweda.shape}')

    # Estimated due date = Invoice date + default interval
    print(f'Months to days function output: {months_to_days((last_service_code / 10) * 6)}')
    neweda[f'Next{last_service_code}K_Due'] = neweda['Invoice date'] + pd.Timedelta(
        days=months_to_days((last_service_code / 10) * 6))

    # --- Filter to vehicles due within the prediction window ---
    fnlupd = neweda[
        (neweda[f'Next{last_service_code}K_Due'] > pd.Timestamp(filterdate)) &
        (neweda[f'Next{last_service_code}K_Due'] <= pd.Timestamp('2026-12-31'))
    ]
    print(f'FilterDate: {filterdate}')
    print(f'Fnupd shape before filtering: {fnlupd.shape}')

    fnlupd['Last Service Mileage'] = fnlupd['Last Service Mileage'].fillna(0)
    lscode = last_service_code / 10
    fnlupd = fnlupd.query('Vehicle_Key_ExpectedServices == @lscode')

    # Mileage sanity: exclude vehicles already past target + 10k buffer
    threshmil = (last_service_code * 1000) + 10000
    print(f'Threshmil: {threshmil}')
    fnlupd['Last Service Date - PMS'] = fnlupd['Invoice date']
    fnlupd = fnlupd.query("`Last Service Mileage` < @threshmil")

    fnlupd['Number of Cylinders'] = pd.to_numeric(fnlupd['Number of Cylinders'], errors='coerce')
    rem = ['New / Used Category', 'Current Customer', 'First Service Date',
           'Last Service Date', 'Next Service Date', 'Vehicle Lifetime in Years',
           'Last Service - PMS', 'Sale Invoice Year', 'Invoice date', 'Target Revenue',
           'Potential Revenue', 'Final Revenue']
    fnlupd = fnlupd.drop(rem, axis=1, errors='ignore')
    pms = fnlupd

    # --- Identify columns with '-' values or missing values ---
    missfeats = []
    hifunfeats = []
    for j in pms.columns:
        print(j)
        print('\n')
        if '-' in pms[j].values:
            hifunfeats.append(j)
        elif pms[j].isnull().sum() > 0:
            missfeats.append(j)

    # --- Wearables flag ---
    cond1 = pms['Brake Purchase Interval'] > 0
    cond2 = pms['Tyre Purchase Interval'] > 0
    cond3 = pms['Battery Purchase Interval'] > 0
    pms['wearablesBought'] = np.where(cond1 | cond2 | cond3, 1, 0)

    # --- Revenue features (imported from features.py) ---
    pms = calculate_revenue_spend(pms, last_service_code)

    missfeatsdrop = ['10K', '20K', '30K', '40K', '50K', '60K', '70K', '80K', '90K', '100K', '110K',
                     '10K R', '20K R', '30K R', '40K R', '50K R', '60K R', '70K R', '80K R', '90K R', '100K R', '110K R',
                     '10K SC', '20K SC', '30K SC', '40K SC', '50K SC', '60K SC', '70K SC', '80K SC', '90K SC', '100K SC',
                     '110K SC', '120K', '120K R', '120K SC', '130K', '130K R', '130K SC', '140K', '140K R', '140K SC',
                     '150K', '150K R', '150K SC', '160K', '160K R', '160K SC', '170K', '170K R', '170K SC',
                     '180K', '180K R', '180K SC', '190K', '190K R', '190K SC', '200K', '200K R', '200K SC',
                     '200K+', '200K+   R', '200K+   SC', 'Brake Purchase Interval', 'Tyre Purchase Interval',
                     'Battery Purchase Interval', '70K R.1', 'LastServicePMS']
    pms = pms.drop(missfeatsdrop, axis=1, errors='ignore')

    hifunfeats1, missfeats1 = [], []
    for j in pms.columns:
        print(j)
        print('\n')
        if '-' in pms[j].values:
            hifunfeats1.append(j)
        elif pms[j].isnull().sum() > 0:
            missfeats1.append(j)
        print('\n')

    for i in hifunfeats1:
        pms[i] = pms[i].replace('-', pd.NA)

    pms['Purchase Age'] = pd.to_numeric(pms['Purchase Age'], errors='coerce')
    pms['Current Age'] = pd.to_numeric(pms['Current Age'], errors='coerce')
    pms['Vehicle Age'] = pd.to_numeric(pms['Vehicle Age'], errors='coerce')

    pms.loc[(pms['Current Age'].isnull()) & (pms['Purchase Age'].notnull()), 'Current Age'] = pms['Purchase Age'] + pms['Vehicle Age']
    pms.loc[(pms['Purchase Age'].isnull()) & (pms['Current Age'].notnull()), 'Purchase Age'] = pms['Current Age'] - pms['Vehicle Age']
    print(f'Dtype of Vehicle_Key_ExpectedServices: {pms["Vehicle_Key_ExpectedServices"].dtype}')
    pms['Vehicle_Key_Actual_Service'] = pd.to_numeric(pms['Vehicle_Key_Actual_Service'], errors='coerce')
    print(f'Dtype of Vehicle_Key_Actual_Service: {pms["Vehicle_Key_Actual_Service"].dtype}')
    pms['Vehicle_Key_Actual_Service'] = pms['Vehicle_Key_Actual_Service'].fillna(0)
    pms['PMS_Delay'] = compute_pms_delay(pms, last_service_code)

    pms['Service_Num'] = 0

    catfeats = ['New / Used', 'Gender', 'Warranty Status', 'Number of Cylinders']
    nationalitymap = pms[['Vehicle Key', 'Nationality']]
    vehmodelmap = pms[['Vehicle Key', 'Model']]
    vehvariantmap = pms[['Vehicle Key', 'Variant']]

    obint = ['Total Promoter', 'Total Passive', 'Total Detractor', 'Total Survey',
             'CC', 'Weight', 'Height', 'Wheel Base', 'Service Frequency']
    for i in obint:
        pms[i] = pd.to_numeric(pms[i], errors='coerce')

    # --- Load non-PMS service history (Description == 'Others') ---
    serv2 = pd.read_csv('data/Service History Feb 2026.csv', low_memory=False)
    serv2 = serv2.query("Description == 'Others'")
    serv2['Service_Num'] = serv2['Description'].apply(extract_kk)
    serv2['Service_Date'] = pd.to_datetime(serv2['Service_Date'], format='mixed', dayfirst=True, errors='coerce')
    serv2 = serv2.query("Service_Date <= @filterdate")
    serv2['Mileage'] = pd.to_numeric(serv2['Mileage'], errors='coerce')
    serv2['Revenue'] = pd.to_numeric(serv2['Revenue'], errors='coerce')
    serv2 = serv2[serv2['Vin_No'].isin(pms['VIN'].unique())]
    serv2.to_csv('Filtered_NonPMS_Service_History.csv', index=False)

    newserv20k = serv2.groupby(['Vin_No']).agg(
        Service_Date=('Service_Date', 'max'),
        Mileage=('Mileage', 'max'),
        Total_Revenue=('Revenue', 'sum'),
    ).reset_index()
    newserv20k = newserv20k[newserv20k['Vin_No'].isin(pms['VIN'].unique())]

    # --- All feature derivation functions below are imported from features.py ---
    dfnpmsdate = derive_npms_features(serv2)
    avg_mileage_interval_non_pms = derive_npms_mileage_features(serv2)
    vin_branch_pivot, top_branches = branch_visit_features(serv2, last_service_code)
    lastnonpmsmil = get_last_nonpms_mileagepred(serv2, 0, pms, vin_col="Vin_No", mileage_col="Mileage", date_col="Service_Date")
    branch_grouped = branch_diversity_features(serv2, last_service_code)
    freq_npms, npms_counts, npmsrevenue, pms_counts = derive_npms_features2(serv2)
    bodyshop_counts = calculate_bodyshop_count(serv2, branch_col='Service_Branch_Name', vin_col='Vin_No', keyword='Bodyshop')
    bodyshop_counts = bodyshop_counts.rename(columns={'VIN': 'Vin_No'})
    revenu = compute_service_features(serv2, filterdate)
    # complaint_features = transform_complaint_features(pms[['VIN', 'Service_Num']], serv2)

    # --- Merge all service features ---
    newserv20ka = newserv20k
    newserv20ka = pd.merge(newserv20ka, npms_counts, on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka, dfnpmsdate, on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka, revenu, on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka, avg_mileage_interval_non_pms, on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka, npmsrevenue, on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka, vin_branch_pivot, on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka, bodyshop_counts, on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka, freq_npms, on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka, branch_grouped, on=['Vin_No'], how='left')
    newserv20ka = pd.merge(newserv20ka, lastnonpmsmil, on=['Vin_No'], how='left')
    # newserv20ka = pd.merge(newserv20ka, complaint_features, on=['Vin_No'], how='left')
    newserv20ka[['nNPMS', 'npmsRevenue', 'Bodyshop_Services']] = newserv20ka[['nNPMS', 'npmsRevenue', 'Bodyshop_Services']].fillna(0)
    # complfts = ['has_complaint_history', 'num_past_complaints', 'avg_complaint_resolution_days',
    #             'max_complaint_resolution_days', 'recent_complaint_resolution_days']

    newserv20ka = newserv20ka.rename(columns={'Vin_No': 'VIN'})
    pmsnew = pd.merge(pms, newserv20ka, on=['VIN'], how='left')
    pmsnew[['nNPMS', 'npmsRevenue', 'Bodyshop_Services', 'unique_branch_serviced', 'otherbranch_services', 'freq_NPMS']] = \
        pmsnew[['nNPMS', 'npmsRevenue', 'Bodyshop_Services', 'unique_branch_serviced', 'otherbranch_services', 'freq_NPMS']].fillna(0)
    pmsnew[top_branches] = pmsnew[top_branches].fillna(0)
    pmsnew[revenu.columns[1:]] = pmsnew[revenu.columns[1:]].fillna(0)
    # pmsnew[complfts] = pmsnew[complfts].fillna(0)
    # pmsnew['latest_complaint_category'] = pmsnew['latest_complaint_category'].fillna('unknown')
    pmsnew = pmsnew.drop(['Service_Date', 'Mileage', 'Total_Revenue'], axis=1)
    pmsnew['Last PMS Mileage'] = 0
    pmsnew[['Total Promoter', 'Total Passive', 'Total Detractor', 'Total Survey']] = \
        pmsnew[['Total Promoter', 'Total Passive', 'Total Detractor', 'Total Survey']].fillna(0)
    pmsnew['Warranty Status'] = pmsnew['Warranty Status'].fillna('No')
    pmsnew['RFM_segments'] = pmsnew['RFM_segments'].fillna('Unknown')

    pmsnew[['Months_Since_Last_NPMS', 'Avg_Mileage_Interval_NPMS']] = \
        pmsnew[['Months_Since_Last_NPMS', 'Avg_Mileage_Interval_NPMS']].fillna(0)
    colssrvd = pmsnew.pop('Last Service Date - PMS')
    pmsnew.insert(3, colssrvd.name, colssrvd)
    pmsnew['Last Service Date - PMS'] = pd.to_datetime(pmsnew['Last Service Date - PMS'], format='mixed', dayfirst=True, errors='coerce')
    pmsnew[f'Next{last_service_code}K_Due'] = pd.to_datetime(pmsnew[f'Next{last_service_code}K_Due'], format='mixed', dayfirst=True, errors='coerce')
    colsdue = pmsnew.pop(f'Next{last_service_code}K_Due')
    pmsnew.insert(4, colsdue.name, colsdue)
    pmsnew = pmsnew.drop(['Nationality', 'Model', 'Variant', 'Branch_List'], axis=1, errors='ignore')

    # --- Cluster mapping (imported from features.py) ---
    natres = map_cluster(nationalitymap, natclus, 'Nationality', 'Nationality', 'Nationality_Cluster')
    modres = map_cluster(vehmodelmap, modclus, 'Model', 'Model', 'Model_Cluster')
    varres = map_cluster(vehvariantmap, varclus, 'Variant', 'Variant', 'Variant_Cluster')
    pmsnew = pmsnew.merge(natres[['Vehicle Key', 'Nationality_Cluster']], on='Vehicle Key', how='left')
    pmsnew = pmsnew.merge(modres[['Vehicle Key', 'Model_Cluster']], on='Vehicle Key', how='left')
    pmsnew = pmsnew.merge(varres[['Vehicle Key', 'Variant_Cluster']], on='Vehicle Key', how='left')
    pmsnew['Nationality_Cluster'] = pmsnew['Nationality_Cluster'].fillna('Unknown')
    pmsnew['Model_Cluster'] = pmsnew['Model_Cluster'].fillna('Unknown')
    pmsnew['Variant_Cluster'] = pmsnew['Variant_Cluster'].fillna('Unknown')
    pmsnew['Nationality_Cluster'] = pmsnew['Nationality_Cluster'].astype(str)
    pmsnew['Model_Cluster'] = pmsnew['Model_Cluster'].astype(str)
    pmsnew['Variant_Cluster'] = pmsnew['Variant_Cluster'].astype(str)

    # --- VHC history (imported from features.py) ---
    # pmsnew = map_vhc_history(pmsnew, serv2, 0)

    # --- Appointment features (all imported from features.py) ---
    appointdfN['Due Date IN'] = pd.to_datetime(appointdfN['Due Date IN'], format='mixed', dayfirst=True, errors='coerce')
    appoinstat = compute_nonpms_appointment_metrics(appointdfN, serv2)
    fnlappnt = derive_appointment_show_features(appointdfN, serv2, last_service_code, filter_date=filterdate)

    pmsnew["Vehicle Magic"] = pmsnew["Vehicle Key"].str.split("-", n=1).str[1]
    pmsnew["Vehicle Magic"] = pmsnew["Vehicle Magic"].str.split("-", n=1).str[0]
    pmsnew["Vehicle Magic"] = pmsnew["Vehicle Magic"].astype(int)
    nservappbk = count_service_appointments_booked(
        appointdfN, pmsnew, last_service_code, filterdate,
        vehicle_col="Vehicle Magic", booking_date_col="WIP Booking Date"
    )

    pmsnew = pmsnew.merge(appoinstat, on='Vehicle Magic', how='left') \
                    .merge(fnlappnt, on='Vehicle Magic', how='left') \
                    .merge(nservappbk, on='Vehicle Magic', how='left')
    pmsnew[['last_appointment_delay_days', 'no_of_late_appointments',
            'appointment_booked_showed_up_atleastonce', 'total_appointments_showed_up',
            'no_of_appointments_booked_but_not_showed_up', 'Avg_appointment_delay_days',
            'no_of_service_appointments_booked']] = pmsnew[[
        'last_appointment_delay_days', 'no_of_late_appointments',
        'appointment_booked_showed_up_atleastonce', 'total_appointments_showed_up',
        'no_of_appointments_booked_but_not_showed_up', 'Avg_appointment_delay_days',
        'no_of_service_appointments_booked']].fillna(0)
    pmsnew['last_appointment_status'] = pmsnew['last_appointment_status'].fillna('No Appointment')
    pmsnew = pmsnew.drop('Vehicle Magic', axis=1, errors='ignore')

    # --- One-hot encoding (imported from features.py) ---
    pmsnew = one_hot(pmsnew)
    pmsnew = pmsnew.reset_index(drop=True)
    pmsnew = pmsnew.rename(columns=lambda c: re.sub(r'\.0$', '', c))
    pmsnew["LowMileageFreqUsers"] = 0

    cols = pmsnew.pop('Last Service Date - PMS')
    pmsnew.insert(1, cols.name, cols)
    pmsnew = pmsnew.drop('70K R.1', axis=1, errors='ignore')
    # vhcfill = ['Survey Score', 'VHC Quoted', 'VHC Sold', 'VHC Lost Sale', 'VHC Lost Red Sale',
    #            'VHC Deferred', 'VHC Amber Deferred', 'VHC Completed_Flag']
    # pmsnew[vhcfill] = pmsnew[vhcfill].fillna(0)

    # --- Imputation (IterativeImputer with Lasso) ---
    imputer = IterativeImputer(random_state=0, estimator=Lasso(), max_iter=1)
    pms_imputed = imputer.fit_transform(pmsnew.iloc[:, 5:])
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
    pms_imputed = pms_imputed.drop(['Mileage', 'Brake Points', 'Vehicle_Key_ExpectedServices',
                                     'Tyre Points', 'Battery Points'], axis=1, errors='ignore')

    # For NonPMS vehicles, set service interval features to default values
    # (6 months per 10k, so for 60k target: 6 * 6 = 36 months)
    pms_imputed[['Avg_Service_Interval_PMS', 'Avg_Service_Interval_PMS1old',
                 'Avg_Service_Interval_PMSper10k', 'Avg_Service_Interval_PMSold',
                 'Months_Since_Last_PMS', 'predicted_interval_months']] = 6 * (last_service_code / 10)
    pms_imputed['Avg_Mileage_Interval_PMS'] = last_service_code * 1000

    return pms_imputed


# ═══════════════════════════════════════════════════════════════════════════════
#                          MAIN PREDICTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
# This is the top-level script that:
#   1. Loads all input data files
#   2. Loops through service milestones (10k, 20k, ..., 50k for target=60k)
#   3. For each milestone: finds vehicles at that stage, computes features,
#      estimates next-due date, filters to the prediction window
#   4. Concatenates all milestones + NonPMS vehicles into one final dataset
# ═══════════════════════════════════════════════════════════════════════════════

filterdate = '2026-06-30'  # Input: data cutoff date
custom_date = datetime.strptime(filterdate, '%Y-%m-%d') + timedelta(days=1)

service_interval = 10
initial = 8  # initial lookback window for the highest service milestone

# --- Load service history ---
df = pd.read_csv('data/Service History Feb 2026.csv', low_memory=False)
servhistory = df
df['Service_Date'] = pd.to_datetime(df['Service_Date'], format='mixed', dayfirst=True, errors='coerce')
df = df.query("Service_Date <= @filterdate")
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
df['Service_Num'] = df['Description'].apply(extract_kk)
df['Service_Num'] = np.where(df['Description'] == '<=10', 10, df['Service_Num'])

finaldf = []
last_service_code = int(sys.argv[1]) if len(sys.argv) > 1 else 60  # Input: Target service code (e.g., 60 for 60k)
os.makedirs(f'predictions/{last_service_code}k', exist_ok=True)

# service_history = [10, 20, 30, 40, 50] for target=60k
# These are the milestones we look at to predict who is due for 60k
service_history = list(range(service_interval, last_service_code, service_interval))

# --- Load reference datasets ---
eda = pd.read_csv('data/EDA Datasheet Till 2025.csv', low_memory=False, encoding='ISO-8859-1')
rfmdf = pd.read_csv('data/RFM till 2025.csv', low_memory=False, encoding='ISO-8859-1')
modclus = pd.read_csv(f'validatecode/Model_clusters_{last_service_code}.csv')
varclus = pd.read_csv(f'validatecode/Variant_clusters_{last_service_code}.csv')
natclus = pd.read_csv(f'validatecode/Nationality_clusters_{last_service_code}.csv')
# appoinshow and appoinoshow are NOT used — appointdfN covers all appointment data
appointdfN = pd.read_csv('data/Appoinments2025.csv', low_memory=False)
digitalfts = pd.read_csv('data/DigitalData2025.csv', low_memory=False)
servcode_desc = pd.read_csv('data/Service Code Desc.csv')

# ═══════════════════════════════════════════════════════════════════════════════
# LOOP: Process each service milestone (e.g., 10k → 50k for target 60k)
# For each milestone, we find vehicles whose LAST PMS was at that milestone
# and haven't yet done a higher service. These are candidates for the target.
# ═══════════════════════════════════════════════════════════════════════════════

for idx, svc in enumerate(service_history):
    # Lookback window: higher milestones get shorter lookback
    # e.g., for target=60k: 10k gets 38mo, 20k gets 32mo, ..., 50k gets 8mo
    months_back = initial + (len(service_history) - idx - 1) * 6
    cutoff_date = custom_date - pd.DateOffset(months=months_back)

    # --- Find vehicles at this milestone ---
    # vinsrem: vehicles that already went BEYOND this milestone (exclude them)
    vinsrem = df.query(f"Service_Num > {svc}")['Vin_No'].unique()
    pms30k = df.query(f"Service_Num == {svc}").copy()
    pms30k['Service_Date'] = pd.to_datetime(pms30k['Service_Date'], format='mixed', dayfirst=True)

    pms40k = df.query(f"Service_Num == {svc}")['Vin_No'].unique()
    pms30kupd = pms30k[pms30k['Vin_No'].isin(pms40k)]

    # Only keep vehicles that had this service after the cutoff date
    filtpms30k = pms30kupd[pms30kupd['Service_Date'] >= cutoff_date]
    filtpms30k['Mileage'] = pd.to_numeric(filtpms30k['Mileage'], errors='coerce')

    filtpms30kupd = filtpms30k

    # Handle multi-owner vehicles: keep only the latest owner
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

    # Aggregate per VIN: max mileage, latest date, total revenue
    filtpms30kupdM = filtpms30kupd.groupby(['Vin_No']).agg(
        Mileage=('Mileage', 'max'),
        Service_Date=('Service_Date', 'max'),
        Total_Revenue=('Revenue', 'sum'),
    ).reset_index()

    # Remove vehicles that already went beyond this milestone
    finalserv40k = filtpms30kupdM[~filtpms30kupdM['Vin_No'].isin(vinsrem)]
    finalserv40k['Service_Num'] = svc

    # --- Match with EDA data ---
    eda40k = eda.copy()
    eda40k['Last Service Date - PMS'] = pd.to_datetime(eda40k['Last Service Date - PMS'], format='mixed', dayfirst=True)
    eda40k = eda40k.rename(columns={'Last Service - PMS': 'LastServicePMS'})
    eda40k['Service_Num'] = eda40k['LastServicePMS'].apply(extract_k1)
    eda40k = eda40k.query("Service_Num == @svc")
    edaM = eda40k[eda40k['VIN'].isin(list(finalserv40k['Vin_No'].unique()))]

    # --- Prepare service history for feature engineering ---
    logger.info("Reading service history data")
    serv1 = servhistory
    logger.debug(f"Service history initial shape: {serv1.shape}")

    logger.info("Processing service history data")
    serv1['Service_Date'] = pd.to_datetime(serv1['Service_Date'], format='mixed', dayfirst=True, errors='coerce')
    invalid_dates = serv1['Service_Date'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Found {invalid_dates} invalid dates in service history")

    serv1 = serv1.query(f"Service_Date <= @filterdate")
    logger.info(f"Service history shape after date filtering: {serv1.shape}")

    logger.info("Converting numeric columns")
    serv1['Mileage'] = pd.to_numeric(serv1['Mileage'], errors='coerce')
    serv1['Revenue'] = pd.to_numeric(serv1['Revenue'], errors='coerce')

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

    # ═══════════════════════════════════════════════════════════════════════════
    # FEATURE ENGINEERING — all functions imported from features.py
    # ═══════════════════════════════════════════════════════════════════════════

    logger.info(f"Deriving PMS features for last service code {svc}k")
    dfpmsdate = derive_pms_features1(serv1, last_service_code)

    logger.info(f"Deriving NPMS features for last service code {svc}k")
    dfnpmsdate = derive_npms_features(serv1)

    logger.info(f"Deriving LastNonPMSMileage for last service code {svc}k")
    lastnonpmsmil = get_last_nonpms_mileagepred(serv1, svc, edaM, vin_col="Vin_No", mileage_col="Mileage", date_col="Service_Date")

    logger.info(f"Calculating average mileage intervals for PMS - last service code {svc}k")
    avg_mileage_interval = derive_pms_mileage_features(serv1, dfpmsdate, last_service_code, svc)

    logger.info(f"Calculating average mileage intervals for NPMS - last service code {svc}k")
    avg_mileage_interval_non_pms = derive_npms_mileage_features(serv1)

    logger.info(f"Deriving PMS service intervals - last service code {svc}k")
    avg_monthly_interval = derive_pms_service_intervals(serv, last_service_code)

    logger.info(f"Adjusting service intervals - last service code {svc}k")
    avg_monthly_interval1 = adjust_service_intervalspred(avg_monthly_interval, last_service_code)
    avg_monthly_interval1.to_csv(f'validatecode/avg_monthly_interval1_{svc}k.csv', index=False)

    # calculatenxt_service_interval: estimates how many months until next target service
    # based on avg interval * remaining service units
    nextservinterv = calculatenxt_service_interval(avg_monthly_interval1, last_service_code)

    avg_monthly_interval1 = avg_monthly_interval1.rename(columns={
        "Avg_Service_Interval_PMS": "Avg_Service_Interval_PMSold",
        "Avg_Service_Interval_PMS1": "Avg_Service_Interval_PMS1old",
        "Avg_Service_Interval_PMSnew": "Avg_Service_Interval_PMSper10k",
        "Avg_Service_Interval_PMS1new": "Avg_Service_Interval_PMS"
    })
    avg_monthly_interval1.drop(['Multiplier', 'last_service_no', 'predicted_interval_months'], axis=1, inplace=True, errors='ignore')

    logger.info(f"Calculating NPMS metrics - last service code {svc}k")
    freq_npms, npms_counts, npmsrevenue, pms_counts = derive_npms_features2(serv1)

    logger.info("Calculating branch-related features")
    vin_branch_pivot, top_branches = branch_visit_features(serv, last_service_code)
    branch_grouped = branch_diversity_features(serv, last_service_code)

    logger.info("Calculating NPMS/PMS Revenue metrics")
    revenu = compute_service_features(servM, filterdate)

    # complaint_features = transform_complaint_features(edaM[['VIN', 'Service_Num']], servM)
    finalserv40k = finalserv40k[finalserv40k['Vin_No'].isin(list(edaM['VIN'].unique()))]

    finalserv40ka = finalserv40k

    # --- Merge all computed features into the main dataframe ---
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
        (lastnonpmsmil, 'Last non-PMS mileage')
        # (complaint_features, 'Complaint features')
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
    fill_columns = ['nNPMS', 'npmsRevenue', 'freq_NPMS', 'unique_branch_serviced', 'otherbranch_services', 'SinglePMS'] + top_branches
    finalserv40ka[fill_columns] = finalserv40ka[fill_columns].fillna(0)

    # Final processing
    logger.info("Performing final data transformations")
    finalserv40ka = finalserv40ka.drop(['Total_Revenue', 'PMS_Service'], axis=1, errors='ignore')
    finalserv40ka = finalserv40ka.rename(columns={'Vin_No': 'VIN'})

    # Merge with master EDA data
    logger.info("Merging with master train data")
    filtered_dfnew = pd.merge(edaM, finalserv40ka, on=['VIN'], how='left')
    filtered_dfnew[fill_columns] = filtered_dfnew[fill_columns].fillna(0)
    filtered_dfnew = filtered_dfnew.drop('Service_Num', axis=1)
    # filtered_dfnew['latest_complaint_category'] = filtered_dfnew['latest_complaint_category'].fillna('unknown')
    filtered_dfnew[revenu.columns[1:]] = filtered_dfnew[revenu.columns[1:]].fillna(0)

    filtered_dfnew = filtered_dfnew[filtered_dfnew['VIN'].isin(edaM['VIN'].unique())]
    filtered_dfnew[filtered_dfnew["Service_Num_x"] == filtered_dfnew["Service_Num_y"]]
    filtered_dfnew = filtered_dfnew.drop_duplicates(subset=['Vehicle Key'])
    filtered_dfnew = filtered_dfnew.drop(['Service_Num_y', 'Branch_List'], axis=1)
    filtered_dfnew = filtered_dfnew.rename(columns={'Service_Num_x': 'Service_Num'})

    filtered_dfnew['Last PMS Mileage'] = pd.to_numeric(filtered_dfnew['Last PMS Mileage'], errors='coerce')
    filtered_dfnew["LastNonPMSMileage"] = filtered_dfnew["LastNonPMSMileage"].fillna(filtered_dfnew["Last PMS Mileage"])
    if last_service_code == 20:
        filtered_dfnew['Avg_Mileage_Interval_PMS'] = filtered_dfnew['Last PMS Mileage']
    print(f"PMS Mileage dtype: {filtered_dfnew['Last PMS Mileage'].dtype}")

    # --- VHC history ---
    # filtered_dfnew = map_vhc_history(filtered_dfnew, serv, svc)  # VHC cols not in service history; handled by vhcpreparation() later
    # vhcfill = ['Survey Score', 'VHC Quoted', 'VHC Sold', 'VHC Lost Sale', 'VHC Lost Red Sale',
    #            'VHC Deferred', 'VHC Amber Deferred', 'VHC Completed_Flag']
    # filtered_dfnew[vhcfill] = filtered_dfnew[vhcfill].fillna(0)

    # --- Appointment features ---
    appointdfN['Due Date IN'] = pd.to_datetime(appointdfN['Due Date IN'], format='mixed', dayfirst=True, errors='coerce')

    ltappoins = compute_late_appointment_metrics(appointdfN, servM, filter_date=filterdate, svc=svc)
    appoinstat = compute_last_appointment_status_with_constant_service_code(appointdfN, serv, svc, filter_date=filterdate)
    fnlappnt = derive_appointment_show_features(appointdfN, servM, svc, filter_date=filterdate)

    filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Key"].str.split("-", n=1).str[1]
    filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Magic"].str.split("-", n=1).str[0]
    filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Magic"].astype(int)

    filtered_dfnew = filtered_dfnew.merge(appoinstat, on='Vehicle Magic', how='left') \
                                   .merge(ltappoins, on='Vehicle Magic', how='left') \
                                   .merge(fnlappnt, on='Vehicle Magic', how='left')
    nservappbk = count_service_appointments_booked(
        appointdfN, filtered_dfnew, svc, filterdate,
        vehicle_col="Vehicle Magic", booking_date_col="WIP Booking Date"
    )
    filtered_dfnew = filtered_dfnew.merge(nservappbk, on='Vehicle Magic', how='left')
    filtered_dfnew['no_of_service_appointments_booked'] = filtered_dfnew['no_of_service_appointments_booked'].fillna(0)

    filtered_dfnew = filtered_dfnew.drop(['Vehicle Magic'], axis=1)

    filtered_dfnew[['last_appointment_delay_days', 'no_of_late_appointments',
                     'appointment_booked_showed_up_atleastonce', 'total_appointments_showed_up',
                     'no_of_appointments_booked_but_not_showed_up', 'Avg_appointment_delay_days']] = \
        filtered_dfnew[['last_appointment_delay_days', 'no_of_late_appointments',
                         'appointment_booked_showed_up_atleastonce', 'total_appointments_showed_up',
                         'no_of_appointments_booked_but_not_showed_up', 'Avg_appointment_delay_days']].fillna(0)
    filtered_dfnew['last_appointment_status'] = filtered_dfnew['last_appointment_status'].fillna('No Appointment')

    # ═══════════════════════════════════════════════════════════════════════════
    # COHORT SELECTION — This is the 3-month window filter
    # calc_next_due: converts predicted_interval_months → concrete due date
    # The filter below keeps only vehicles due within the prediction window
    # ═══════════════════════════════════════════════════════════════════════════

    # Step 1: Compute the predicted next-service due date
    filtered_dfnew[f"Next{last_service_code}K_Due"] = filtered_dfnew.apply(
        lambda row: calc_next_due(row["Last Service Date - PMS"], row["predicted_interval_months"]),
        axis=1
    )
    filtered_dfnew[f"Next{last_service_code}K_Due"] = pd.to_datetime(
        filtered_dfnew[f"Next{last_service_code}K_Due"], format='mixed', errors='coerce')

    # Step 2: Filter to vehicles due within the prediction window (currently 3 months)
    # Change '2025-09-30' to '2025-12-31' for a 6-month window
    fnlupd = filtered_dfnew[
        (filtered_dfnew[f'Next{last_service_code}K_Due'] > pd.Timestamp('2026-06-30')) &
        (filtered_dfnew[f'Next{last_service_code}K_Due'] <= pd.Timestamp('2026-12-31'))
    ]

    # Step 3: Mileage sanity filter — exclude vehicles already past target + 10k buffer
    threshmil = (last_service_code * 1000) + 10000
    fnlupd = fnlupd.query("`Last Service Mileage` < @threshmil")

    # --- Post-filter processing ---
    fnlupd = fnlupd.merge(rfmdf[['Customer ID', 'RFM_segments']], on=['Customer ID'], how='left')
    fnlupd['Number of Cylinders'] = pd.to_numeric(fnlupd['Number of Cylinders'], errors='coerce')
    rem = ['New / Used Category', 'Current Customer', 'First Service Date',
           'Last Service Date', 'Next Service Date', 'Vehicle Lifetime in Years',
           'Last Service - PMS', 'Sale Invoice Year', 'Invoice date', 'Target Revenue',
           'Potential Revenue', 'Final Revenue']
    fnlupd = fnlupd.drop(rem, axis=1, errors='ignore')
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
    pms['wearablesBought'] = np.where(cond1 | cond2 | cond3, 1, 0)

    pms = calculate_revenue_spend(pms, last_service_code)

    missfeatsdrop = ['10K', '20K', '30K', '40K', '50K', '60K', '70K', '80K', '90K', '100K', '110K',
                     '10K R', '20K R', '30K R', '40K R', '50K R', '60K R', '70K R', '80K R', '90K R', '100K R', '110K R',
                     '10K SC', '20K SC', '30K SC', '40K SC', '50K SC', '60K SC', '70K SC', '80K SC', '90K SC', '100K SC',
                     '110K SC', '120K', '120K R', '120K SC', '130K', '130K R', '130K SC', '140K', '140K R', '140K SC',
                     '150K', '150K R', '150K SC', '160K', '160K R', '160K SC', '170K', '170K R', '170K SC',
                     '180K', '180K R', '180K SC', '190K', '190K R', '190K SC', '200K', '200K R', '200K SC',
                     '200K+', '200K+   R', '200K+   SC', 'Brake Purchase Interval', 'Tyre Purchase Interval',
                     'Battery Purchase Interval', '70K R.1', 'LastServicePMS']
    pms = pms.drop(missfeatsdrop, axis=1, errors='ignore')

    hifunfeats1, missfeats1 = [], []
    for j in pms.columns:
        print(j)
        print('\n')
        if '-' in pms[j].values:
            hifunfeats1.append(j)
        elif pms[j].isnull().sum() > 0:
            missfeats1.append(j)
        print('\n')

    for i in hifunfeats1:
        pms[i] = pms[i].replace('-', pd.NA)

    pms['Purchase Age'] = pd.to_numeric(pms['Purchase Age'], errors='coerce')
    pms['Current Age'] = pd.to_numeric(pms['Current Age'], errors='coerce')
    pms['Vehicle Age'] = pd.to_numeric(pms['Vehicle Age'], errors='coerce')

    pms.loc[(pms['Current Age'].isnull()) & (pms['Purchase Age'].notnull()), 'Current Age'] = pms['Purchase Age'] + pms['Vehicle Age']
    pms.loc[(pms['Purchase Age'].isnull()) & (pms['Current Age'].notnull()), 'Purchase Age'] = pms['Current Age'] - pms['Vehicle Age']
    print(f'Dtype of Vehicle_Key_ExpectedServices: {pms["Vehicle_Key_ExpectedServices"].dtype}')
    pms['Vehicle_Key_Actual_Service'] = pd.to_numeric(pms['Vehicle_Key_Actual_Service'], errors='coerce')
    print(f'Dtype of Vehicle_Key_Actual_Service: {pms["Vehicle_Key_Actual_Service"].dtype}')
    pms['PMS_Delay'] = compute_pms_delay(pms, last_service_code)


    catfeats = ['New / Used', 'Gender', 'Warranty Status', 'Number of Cylinders']
    nationalitymap = pms[['Vehicle Key', 'Nationality']]
    vehmodelmap = pms[['Vehicle Key', 'Model']]
    vehvariantmap = pms[['Vehicle Key', 'Variant']]

    obint = ['Total Promoter', 'Total Passive', 'Total Detractor', 'Total Survey',
             'CC', 'Weight', 'Height', 'Wheel Base', 'Service Frequency']
    for i in obint:
        pms[i] = pd.to_numeric(pms[i], errors='coerce')
    pms = pms.drop('Service_Date', axis=1, errors='ignore')
    pms[['Months_Since_Last_NPMS', 'Avg_Mileage_Interval_NPMS']] = \
        pms[['Months_Since_Last_NPMS', 'Avg_Mileage_Interval_NPMS']].fillna(0)
    colssrvd = pms.pop('Last Service Date - PMS')
    pms.insert(3, colssrvd.name, colssrvd)
    pms['Last Service Date - PMS'] = pd.to_datetime(pms['Last Service Date - PMS'], format='mixed', dayfirst=True, errors='coerce')
    pms[f'Next{last_service_code}K_Due'] = pd.to_datetime(pms[f'Next{last_service_code}K_Due'], format='mixed', dayfirst=True, errors='coerce')
    colsdue = pms.pop(f'Next{last_service_code}K_Due')
    pms.insert(4, colsdue.name, colsdue)
    pms = pms.drop(['Nationality', 'Model', 'Variant'], axis=1, errors='ignore')

    # --- Cluster mapping ---
    natres = map_cluster(nationalitymap, natclus, 'Nationality', 'Nationality', 'Nationality_Cluster')
    modres = map_cluster(vehmodelmap, modclus, 'Model', 'Model', 'Model_Cluster')
    varres = map_cluster(vehvariantmap, varclus, 'Variant', 'Variant', 'Variant_Cluster')
    pms = pms.merge(natres[['Vehicle Key', 'Nationality_Cluster']], on='Vehicle Key', how='left')
    pms = pms.merge(modres[['Vehicle Key', 'Model_Cluster']], on='Vehicle Key', how='left')
    pms = pms.merge(varres[['Vehicle Key', 'Variant_Cluster']], on='Vehicle Key', how='left')
    pms['Nationality_Cluster'] = pms['Nationality_Cluster'].fillna('Unknown')
    pms['Model_Cluster'] = pms['Model_Cluster'].fillna('Unknown')
    pms['Variant_Cluster'] = pms['Variant_Cluster'].fillna('Unknown')
    pms['Nationality_Cluster'] = pms['Nationality_Cluster'].astype(str)
    pms['Model_Cluster'] = pms['Model_Cluster'].astype(str)
    pms['Variant_Cluster'] = pms['Variant_Cluster'].astype(str)

    # --- One-hot encoding ---
    pmsnew = one_hot(pms)
    pmsnew = pmsnew.rename(columns=lambda c: re.sub(r'\.0$', '', c))
    pmsnew = pmsnew.reset_index(drop=True)
    pmsnew["LowMileageFreqUsers"] = np.where(
        (pmsnew["Avg_Service_Interval_PMS"].between(0, 7)) & (pmsnew["Last PMS Mileage"].between(0, (svc - 10) * 1000)), 1, 0)
    uy = ['Total Promoter', 'Total Passive', 'Total Detractor', 'Total Survey']
    for i in uy:
        pmsnew[i] = pmsnew[i].fillna(0)
    cols = pmsnew.pop('Last Service Date - PMS')
    pmsnew.insert(1, cols.name, cols)
    pmsnew = pmsnew.drop('70K R.1', axis=1, errors='ignore')

    # --- Imputation ---
    imputer = IterativeImputer(random_state=0, estimator=Lasso(), max_iter=1)
    pms_imputed = imputer.fit_transform(pmsnew.iloc[:, 5:])
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
    bs = serv1[serv1['Description'] == 'Others']
    bodyshop_counts = calculate_bodyshop_count(bs, branch_col='Service_Branch_Name', vin_col='Vin_No', keyword='Bodyshop')
    pms_imputed = pms_imputed.merge(bodyshop_counts, on='VIN', how='left')
    pms_imputed['Bodyshop_Services'] = pms_imputed['Bodyshop_Services'].fillna(0).astype(int)
    pms_imputed = pms_imputed.drop(['Mileage', 'Brake Points', 'Vehicle_Key_ExpectedServices',
                                     'Tyre Points', 'Battery Points'], axis=1, errors='ignore')
    pms_imputed.rename(columns={'Avg_Service_Interval_PMS': 'Avg_Service_Interval_PMS_per10k'}, inplace=True)
    pms_imputed.rename(columns={'Avg_Service_Interval_PMS1': 'Avg_Service_Interval_PMS'}, inplace=True)
    coltpmsmil = pms_imputed.pop("Last PMS Mileage")
    pms_imputed.insert(len(pms_imputed.columns) - 1, "Last PMS Mileage", coltpmsmil)
    pms_imputed["Last Service Mileage"] = pms_imputed[["Last PMS Mileage", "LastNonPMSMileage"]].max(axis=1)

    finaldf.append(pms_imputed)

# ═══════════════════════════════════════════════════════════════════════════════
# COMBINE ALL MILESTONES + NON-PMS VEHICLES
# ═══════════════════════════════════════════════════════════════════════════════

maindf = pd.concat(finaldf, axis=0)
maindf = maindf.fillna(0)
maindf.to_csv(f'predictions/{last_service_code}k/pms_milestones.csv', index=False)

# --- Non-PMS vehicles (never had any PMS service) ---
nompmsbase = NonPMS(rfmdf, filterdate, last_service_code, custom_date, natclus, varclus, modclus,
                    None, None, appointdfN, servcode_desc)  # appoinshow/appoinoshow unused
nompmsbase.to_csv(f'predictions/{last_service_code}k/nonpms_vehicles.csv', index=False)

# --- Final merge: PMS milestones + Non-PMS ---
finalbase = pd.concat([maindf, nompmsbase], axis=0)
finalbase = finalbase.fillna(0)
finalbase = finalbase.drop(columns=[col for col in finalbase.columns if "unknown" in col.lower()])
finalbase = finalbase.merge(digitalfts, on=['Vehicle Key', 'Customer ID'], how='left')
finalbase.iloc[:, -4:] = finalbase.iloc[:, -4:].fillna(0)
finalbase['Vehicles Owned'] = finalbase['Vehicles Owned'].fillna(1)

vhcfill = ['Survey Score', 'VHC Quoted', 'VHC Sold', 'VHC Lost Sale', 'VHC Lost Red Sale',
           'VHC Deferred', 'VHC Amber Deferred', 'VHC Completed_Flag']
finalbase = finalbase.drop(vhcfill, axis=1, errors='ignore')

# --- VHC features from dedicated VHC dataset ---
vhcdf = pd.read_csv('data/VHC PMS till 2025.csv', encoding="ISO-8859-1", low_memory=False)
vhcdata = vhcpreparation(vhcdf, last_service_code - 10, finalbase)
finalbase = finalbase.merge(vhcdata, on=['Vehicle Key', 'Service_Num'], how='left')
finalbase = finalbase.fillna(0)

# --- Non-PMS event features ---
servMain = df
servMain = servMain.query("Service_Num < @last_service_code")
Npmsevents, non_pms_countn = get_non_pms_events(servMain, servcode_desc, finalbase, last_service_code)
Npmsevents = Npmsevents.rename(columns={'Vin_No': 'VIN'})
non_pms_countn = non_pms_countn.rename(columns={'Vin_No': 'VIN'})
finalbase = finalbase.drop('nNPMS', axis=1, errors='ignore')
finalbase = finalbase.merge(Npmsevents, on='VIN', how='left').fillna(0)
finalbase = finalbase.merge(non_pms_countn, on='VIN', how='left').fillna(0)

finalbase.to_csv(f'predictions/{last_service_code}k/prediction_dataset.csv', index=False)
print(f'\n[DONE] Prediction dataset saved: predictions/{last_service_code}k/prediction_dataset.csv')
print(f'Shape: {finalbase.shape}')
