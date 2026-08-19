import pandas as pd
import numpy as np
import os
import sys
import argparse
from pmstrainfeatureeng_refactored import process_service_data, extract_k1, extract_kk
from features import expected_milestone_months

def create_candidates_test_cohort(eda_path, service_history_path, milestone, start_date, end_date,
                                  grace_days, band_days=None):
    """Test-set counterpart of build_candidates_cohort() (--label-mode candidates on the training
    side): due date = whichever comes first of the schedule and a service-1 -> service-10 burn-rate
    projection (due_date.burn_rate_date), invoice-date-guarded so FirstSrvDate is always a genuine
    sale date.

    Cohort MEMBERSHIP is unchanged either way -- still "due date lands in [start_date, end_date]".
    Only the LABEL rule changes:

    `band_days=None` (default): TargetFlag = 1 if completed on or before window_end + grace_days,
    with NO exclusion for early completion -- an EarlyCompleter column flags those instead (needed
    for Stage D's early / not-early / combined slicing), matching the training side's convention
    exactly rather than build_history_cohort()'s "drop early completers" rule.

    `band_days=N`: a symmetric band around the quarter, [start_date - N, end_date + N]. Vehicles
    that completed the milestone BEFORE the band opened are dropped from the cohort outright (they
    were never really "due this quarter" -- the wide forward-only grace above is what let them
    count as automatic positives). TargetFlag = 1 only for a completion inside the band; a
    completion after the band closes is a 0, same as never completing. No EarlyCompleter column --
    the band absorbs what that flag used to separate out.

    Deliberately a separate function rather than a branch bolted onto create_test_cohort(): the due
    date model, the invoice guard and the label rule are all different from every existing mode
    there, and threading a fourth set of conditionals through that function was worse than keeping
    them apart.
    """
    from due_date import burn_rate_date, earliest_date

    print(f"Loading EDA data from {eda_path}")
    df = pd.read_csv(eda_path, low_memory=False, encoding="ISO-8859-1")
    df = df.query("`Last Service - PMS` != '-'").copy()

    df["Invoice date"] = pd.to_datetime(df["Invoice date"].replace("-", pd.NA), format="mixed",
                                        dayfirst=True, errors="coerce")
    df["First Service Date"] = pd.to_datetime(df["First Service Date"].replace("-", pd.NA),
                                              format="mixed", dayfirst=True, errors="coerce")
    df["FirstSrvDate"] = df["Invoice date"].fillna(df["First Service Date"])
    n_before = len(df)
    df = df[df["Invoice date"].notna()].copy()
    print(f"invoice-date guard: {n_before:,} -> {len(df):,} EDA vehicles "
          f"({n_before - len(df):,} dropped, no genuine sale date)")

    serv = pd.read_csv(service_history_path, low_memory=False, encoding="ISO-8859-1")
    serv["Service_Date"] = pd.to_datetime(serv["Service_Date"], format="mixed", dayfirst=True,
                                          errors="coerce")
    serv["Mileage"] = pd.to_numeric(serv["Mileage"], errors="coerce")
    serv = serv.dropna(subset=["Service_Date"])
    # Raw Service_Code, NOT extract_kk(Description) -- matches build_candidates_cohort() exactly
    # (see that function's docstring for why extract_kk's '<=10' trap is avoided here regardless).
    serv["Service_Num"] = pd.to_numeric(serv["Service_Code"], errors="coerce")

    vins = df["VIN"].values
    first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
    months = expected_milestone_months(milestone, "schedule")
    due_sched = first_srv + pd.DateOffset(months=months)
    cutoff = serv["Service_Date"].max()
    due_burn = burn_rate_date(vins, serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]],
                              milestone, cutoff)
    due = earliest_date(due_sched, due_burn)
    due_source = np.where(due_burn.notna().to_numpy() & (due_burn.to_numpy() <= due_sched.to_numpy()),
                          "burn-rate", "schedule")

    at_ms = serv[serv["Service_Num"] == milestone]
    date_ms = (at_ms.sort_values("Service_Date").groupby("Vin_No")["Service_Date"]
              .first().reindex(vins))

    ms_col = f"Actual{milestone}kDate"
    df[f"Expected{milestone}kDate"] = due.values
    df["DueSource"] = due_source
    df[ms_col] = date_ms.values

    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    cohort = df[
        (df[f"Expected{milestone}kDate"] >= start_dt) &
        (df[f"Expected{milestone}kDate"] <= end_dt) &
        (df["Vehicle Service Status"].isin(["InActive", "Lapsed", "Active"]))
    ].copy()

    if band_days is not None:
        band_lo = start_dt - pd.Timedelta(days=band_days)
        band_hi = end_dt + pd.Timedelta(days=band_days)
        ms = cohort[ms_col]
        done_far_early = ms.notna() & (ms < band_lo)
        n_dropped = int(done_far_early.sum())
        cohort = cohort[~done_far_early].copy()

        ms = cohort[ms_col]
        cohort["TargetFlag"] = (ms.notna() & (ms >= band_lo) & (ms <= band_hi)).astype(int)

        turnup = int(cohort["TargetFlag"].sum())
        print(f"Band: {band_lo.date()} .. {band_hi.date()} (+/-{band_days}d around the quarter)")
        print(f"Dropped {n_dropped} that completed {milestone}k before the band opened "
              f"(no longer 'due this quarter')")
        print(f"Found {len(cohort)} vehicles due for {milestone}k between {start_date} and "
              f"{end_date} (candidates due-date model, banded)")
        print(f"Ground truth: {turnup} positive ({100*turnup/max(len(cohort),1):.1f}%)")
        return cohort

    ms = cohort[ms_col]
    cohort["TargetFlag"] = (ms.notna() & (ms <= end_dt + pd.Timedelta(days=grace_days))).astype(int)
    cohort["EarlyCompleter"] = (ms.notna() & (ms < start_dt)).astype(int)

    turnup = int(cohort["TargetFlag"].sum())
    early = int(cohort["EarlyCompleter"].sum())
    print(f"Found {len(cohort)} vehicles due for {milestone}k between {start_date} and {end_date} "
          f"(candidates due-date model)")
    print(f"Ground truth: {turnup} positive ({100*turnup/max(len(cohort),1):.1f}%), "
          f"{early} early completers ({100*early/max(len(cohort),1):.1f}%)")
    return cohort


def create_test_cohort(eda_path, service_history_path, milestone, start_date, end_date, cutoff_date, use_mileage_projection=True, date_model="schedule", grace_days=None):
    """
    Identifies vehicles due for the milestone in the target window,
    and assigns ground truth based on actual service history in that window.

    `date_model` must match the training run's --date-model, or the two sides disagree on who is
    due when. See features.expected_milestone_months().

    NOTE on use_mileage_projection: measured 2026-08-16, it is WORSE than the time-based date, not
    better -- median lag to the actual turn-up -17.8 months vs -3.9 on 20k, and the IQR nearly
    quadruples (29.9 vs 7.9). It collapses for vehicles already past the milestone mileage:
    Miles_Remaining goes negative, Days_Remaining is clipped to NaN then filled with 0, so
    Projected_Date becomes the last service date. The CLI flag defaults it off; leave it off.

    For the 'candidates' label mode (matching --label-mode candidates on the training side), use
    create_candidates_test_cohort() instead -- the due-date model, invoice guard and label rule are
    all different, and this function is unchanged for every other mode.
    """
    print(f"Loading EDA data from {eda_path}")
    df = pd.read_csv(eda_path, low_memory=False, encoding="ISO-8859-1")
    df = df.query("`Last Service - PMS` != '-'").copy()

    # 1. Compute Expected Date for the milestone
    df["FirstSrvDate"] = df["Invoice date"].fillna(df["First Service Date"])
    df["FirstSrvDate"] = pd.to_datetime(df["FirstSrvDate"], format="mixed", errors="coerce")

    months_to_add = expected_milestone_months(milestone, date_model)
    print(f"Expected-date model: {date_model} ({months_to_add} months from first service)")
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
    
    at_milestone = serv[serv["Service_Num"] == milestone]

    if grace_days is not None:
        # --- grace-band rule: mirrors build_history_cohort() in the training builder ---
        # A positive turned up within `grace_days` either side of the due window. This asks "did
        # they come roughly when due", not "did they ever come" -- and it is the rule the 'history'
        # training mode uses, so train and test finally answer the same question.
        #
        # Two deliberate differences from the 'ever' branch below:
        #  - the already-done exclusion keys on `== milestone`, not `>= milestone`, so vehicles that
        #    SKIPPED this milestone and went straight past it stay in as negatives. They are the
        #    clearest no-shows in the data and the old rule threw them away.
        #  - the exclusion cuts at the band start, not at the feature cutoff, because that is the
        #    first moment the outcome could occur.
        band_lo = pd.to_datetime(start_date) - pd.Timedelta(days=grace_days)
        band_hi = pd.to_datetime(end_date) + pd.Timedelta(days=grace_days)
        print(f"Grace band: {band_lo.date()} .. {band_hi.date()} (+/-{grace_days}d)")

        done_early = set(at_milestone.loc[at_milestone["Service_Date"] < band_lo, "Vin_No"].unique())
        n_early = int(cohort["VIN"].isin(done_early).sum())
        cohort = cohort[~cohort["VIN"].isin(done_early)].copy()
        print(f"Dropped {n_early} that already completed {milestone}k before the band opened")

        in_band = set(at_milestone.loc[(at_milestone["Service_Date"] >= band_lo) &
                                       (at_milestone["Service_Date"] <= band_hi),
                                       "Vin_No"].unique())
        cohort["TargetFlag"] = cohort["VIN"].isin(in_band).astype(int)
        n_skip = int((~cohort["VIN"].isin(set(at_milestone["Vin_No"].unique()))).sum())
        print(f"Never do {milestone}k at all (skippers + no-shows): {n_skip}")
    else:
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

        # 4. Label with Ground Truth (Did they EVER show up for this milestone?)
        print(f"Assigning ground truth...")
        vins_that_showed_up = set(at_milestone["Vin_No"].unique())
        cohort["TargetFlag"] = cohort["VIN"].apply(lambda x: 1 if x in vins_that_showed_up else 0)

    print(f"Found {len(cohort)} vehicles due for {milestone}k between {start_date} and {end_date}")
    turnup_count = cohort["TargetFlag"].sum()
    print(f"Ground Truth assigned: {turnup_count} showed up ({turnup_count/len(cohort)*100:.1f}%), {len(cohort)-turnup_count} did not.")

    return cohort

def window_label(start_date, end_date):
    """Name a prediction window for the output filename.

    Exactly one calendar quarter -> 'Q1_2026'. Anything else -> '2026-01-15_2026-02-28', so an
    ad-hoc window can never collide with a quarter's file.

    This exists because the output name used to be hardcoded to Q1_2026 while the window came from
    argv: building a Q2 cohort silently overwrote the Q1 file with Q2 rows, under the Q1 name.
    """
    s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
    q_start = pd.Timestamp(year=s.year, month=3 * s.quarter - 2, day=1)
    q_end = q_start + pd.offsets.QuarterEnd(0)
    if s.normalize() == q_start and e.normalize() == q_end.normalize():
        return f"Q{s.quarter}_{s.year}"
    return f"{s.date()}_{e.date()}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PMS Test Set")
    parser.add_argument("--milestone", type=int, default=20)
    parser.add_argument("--window-start", type=str, default="2026-01-01", help="Start of prediction window")
    parser.add_argument("--window-end", type=str, default="2026-03-31", help="End of prediction window")
    parser.add_argument("--train-cutoff", type=str, default="2025-12-31", help="Date to cut off feature generation")
    parser.add_argument("--enable-mileage-projection", action="store_true", help="Use dynamic mileage projection for expected date (measured WORSE than time-based -- see create_test_cohort)")
    parser.add_argument("--date-model", choices=["schedule", "calibrated"], default="schedule",
                        help="Must match pmstrainfeatureeng_refactored.py --date-model.")
    parser.add_argument("--grace-days", type=int, default=None,
                        help="Label a positive as 'turned up within N days either side of the "
                             "window' instead of 'ever turned up'. Must match the training run's "
                             "--grace-days when training used --label-mode history, or train and "
                             "test are answering different questions. Writes a _h{N} file so the "
                             "'ever' cohorts and every metric measured on them survive untouched. "
                             "Required (forward-only band) when --label-mode candidates.")
    parser.add_argument("--label-mode", choices=["ever", "candidates"], default="ever",
                        help="'ever' (default): existing create_test_cohort() logic, --date-model "
                             "and --grace-days apply as documented on those flags. 'candidates': "
                             "matches --label-mode candidates on the training side -- due date = "
                             "whichever comes first of the schedule or a service-1->service-10 "
                             "burn-rate projection, invoice-date guard, and TargetFlag counts early "
                             "completers as positive (flagged via EarlyCompleter) instead of "
                             "excluding them. Ignores --date-model and --enable-mileage-projection. "
                             "Writes a _cand file.")
    parser.add_argument("--band-days", type=int, default=None,
                        help="candidates mode only: replace the forward-only "
                             "'completed by window_end + grace_days, flag early completers' rule "
                             "with a symmetric band [window_start - N, window_end + N]. Vehicles "
                             "that completed before the band opened are DROPPED (not flagged), and "
                             "there is no EarlyCompleter column. Overrides --grace-days. Writes a "
                             "_cand_b{N} file.")
    args = parser.parse_args()

    if args.label_mode == "candidates" and args.grace_days is None and args.band_days is None:
        sys.exit("--label-mode candidates requires --grace-days (the labelling plan uses 15) "
                 "or --band-days")
    if args.band_days is not None and args.label_mode != "candidates":
        sys.exit("--band-days only applies to --label-mode candidates")
    
    # Files
    # Must stay identical to the block in pmstrainfeatureeng_refactored.py -- see the note there.
    eda_path = os.path.join("data", "EDA_Q2-2026.csv")
    service_history_path = os.path.join("data", "Service_History_Q2-2026.csv")
    rfm_path = "data/RFM_Q2-2026.csv"
    appointdf_path = "data/Appointments_Q2-2026.csv"
    digidf_path = "data/Digital_Sessions_Q2-2026.csv"
    vhc_path = "data/VHC_Q2-2026.csv"
    servcode_path = "data/Service_Code_Desc.csv"
    
    print(f"--- Building Test Set for {args.milestone}k ---")
    print(f"Prediction Window: {args.window_start} to {args.window_end}")
    print(f"Feature Cutoff Date: {args.train_cutoff}")
    print(f"Dynamic Mileage Projection: {'Enabled' if args.enable_mileage_projection else 'Disabled'}")
    
    # 1. Build Cohort and Label Ground Truth
    if args.label_mode == "candidates":
        cohort = create_candidates_test_cohort(
            eda_path=eda_path,
            service_history_path=service_history_path,
            milestone=args.milestone,
            start_date=args.window_start,
            end_date=args.window_end,
            grace_days=args.grace_days,
            band_days=args.band_days,
        )
    else:
        cohort = create_test_cohort(
            eda_path=eda_path,
            service_history_path=service_history_path,
            milestone=args.milestone,
            start_date=args.window_start,
            end_date=args.window_end,
            cutoff_date=args.train_cutoff,
            use_mileage_projection=args.enable_mileage_projection,
            date_model=args.date_model,
            grace_days=args.grace_days,
        )

    # 2. Process Service Data to generate features
    # CRITICAL: We pass filter_date=train_cutoff so it only uses history BEFORE the prediction window
    #
    # Under --grace-days the outcome window opens `grace` days BEFORE the prediction window, so the
    # cutoff has to move back with it -- otherwise the last `grace` days of feature history sit
    # inside the period being predicted. Same reasoning as run_history_mode() in the training
    # builder; keep the two in step. NOT true for candidates mode: there the outcome window only
    # extends FORWARD from window_end (early completions are read from history that already predates
    # the cutoff, and the milestone's own service record is excluded from features regardless via
    # serv1's Service_Num < milestone filter), matching run_candidates_mode()'s filter_date=q_start.
    feature_cutoff = args.train_cutoff
    if args.label_mode != "candidates" and args.grace_days is not None:
        band_lo = pd.to_datetime(args.window_start) - pd.Timedelta(days=args.grace_days)
        feature_cutoff = min(pd.to_datetime(args.train_cutoff), band_lo).strftime("%Y-%m-%d")
        if feature_cutoff != args.train_cutoff:
            print(f"Feature cutoff pulled back {args.train_cutoff} -> {feature_cutoff} "
                  f"(band opens {band_lo.date()})")
    elif args.label_mode == "candidates" and args.band_days is not None:
        # band_days makes the outcome window reach BACKWARD from window_start for the first time in
        # candidates mode -- the docstring's "outcome window only extends forward" no longer holds,
        # so this needs the same cutoff pull-back the 'ever'+grace-days branch above already has.
        band_lo = pd.to_datetime(args.window_start) - pd.Timedelta(days=args.band_days)
        feature_cutoff = min(pd.to_datetime(args.train_cutoff), band_lo).strftime("%Y-%m-%d")
        if feature_cutoff != args.train_cutoff:
            print(f"Feature cutoff pulled back {args.train_cutoff} -> {feature_cutoff} "
                  f"(band opens {band_lo.date()})")
    print(f"\nProcessing features using data up to {feature_cutoff}...")

    # Ensure Service_Num is present as process_service_data expects it
    cohort["Service_Num"] = cohort["Last Service - PMS"].apply(extract_k1)

    # Bookkeeping columns (candidates mode only): process_service_data() treats every non-ID object
    # column as a categorical to one-hot encode and has no notion of "bookkeeping" -- see
    # run_candidates_mode()'s identical guard on the training side for the exact failure mode this
    # avoids (OHE junk columns, a dropped high-cardinality date column, and an EarlyCompleter/_x/_y
    # collision on re-merge). Strip before the call, re-attach by VIN after.
    bookkeeping_cols = ["DueSource", "EarlyCompleter", f"Actual{args.milestone}kDate"]
    bookkeeping_cols = [c for c in bookkeeping_cols if c in cohort.columns]
    bk = cohort[["VIN"] + bookkeeping_cols].set_index("VIN") if bookkeeping_cols else None

    test_features = process_service_data(
        mastersheet=cohort.drop(columns=bookkeeping_cols, errors="ignore"),
        servhistory=service_history_path,
        rfm=rfm_path,
        servcode=servcode_path,
        appointdf=appointdf_path,
        digidf=digidf_path,
        vhc=vhc_path,
        filter_date=feature_cutoff,
        last_service_code=args.milestone,
        is_test=True
    )
    if bk is not None:
        test_features = test_features.drop(columns=bookkeeping_cols, errors="ignore").merge(
            bk, left_on="VIN", right_index=True, how="left")

    # 3. Write the FULL feature matrix -- do NOT filter to a feature list.
    # This step used to filter by models/{m}k/selected_features.json (the feature_sel() MI
    # shortlist, ~33 cols), which cut the output to 35 columns and left 67-78% of every model's
    # inputs zero-filled at inference. Filtering by the model's real list
    # (models/selected_features_{m}k.json) would not fix it either: that list contains columns
    # derived at inference time (months_to_10k, Max_PMS_Revenue, Last Service Mileage, ...) which
    # process_service_data never produces.
    # Both consumers already handle a wide input -- retrain.py intersects it with the training
    # columns, score_milestone.py zero-fills what is missing and warns with a count.
    missing_keys = [c for c in ("VIN", "TargetFlag") if c not in test_features.columns]
    if missing_keys:
        print(f"Error: process_service_data did not return {missing_keys}; cannot use this cohort.")
        sys.exit(1)

    lead = ["VIN", "TargetFlag"]
    final_test_set = test_features[lead + [c for c in test_features.columns if c not in lead]]

    os.makedirs("test_sets", exist_ok=True)
    label = window_label(args.window_start, args.window_end)
    if args.label_mode == "candidates":
        # A different due-date model AND a different label rule -- its own suffix, never near the
        # '_h{N}' grace-band files (those match --label-mode history, a different training run).
        # band_days gets a further suffix: same cohort membership as plain '_cand', different label
        # rule and no EarlyCompleter column, so it must never collide with the '_cand' file either.
        date_suffix = f"_cand_b{args.band_days}" if args.band_days is not None else "_cand"
    else:
        # A calibrated cohort is a DIFFERENT set of vehicles (the due-date model decides who is in
        # the window), so it gets its own file rather than overwriting the schedule one -- otherwise
        # two runs' metrics would silently be measured on different populations.
        date_suffix = "" if args.date_model == "schedule" else "_cal"
        # A grace-band cohort is a different population AND a different label rule, so it never
        # overwrites the 'ever' file -- the existing metrics stay reproducible.
        if args.grace_days is not None:
            date_suffix += f"_h{args.grace_days}"
    test_path = os.path.join("test_sets", f"test_{args.milestone}k_{label}{date_suffix}.csv")
    if os.path.exists(test_path):
        print(f"Note: overwriting existing {test_path}")
    final_test_set.to_csv(test_path, index=False)

    print(f"\nSuccess! Test set saved to {test_path}")
    print(f"Final test shape: {final_test_set.shape}")
