import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from features import extract_k1, resolve_service_num
from due_date import burn_rate_date, earliest_date
from date_utils import parse_dates



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
    label_mode="ever",
    label_start=None,
    date_model="schedule",
    drop_mileage_cap=False,
):
    """Build the labelled training cohort. Two labelling rules, `label_mode`:

    'ever' (default, the POC convention decided 2026-08-12)
    - Positives: ALL vehicles whose EDA master sheet shows they completed the target PMS, with NO
      date condition.
    - Negatives: vehicles pending/overdue for the target PMS within a sliding date window.
    - The while loop slides that window until the positive ratio lands in target_range.

    'window' (added 2026-08-16)
    - Positives AND negatives must both be DUE in the same window: Expected{svc}Date within
      [start_date, quarter_end]. Label is then simply "did they complete it".
    - The sliding loop is SKIPPED -- the base rate is whatever the cohort naturally gives. Tuning
      the window to hit a target ratio would defeat the point, since the window is now what defines
      both classes.

    Why 'window' exists. Under 'ever' the positives are drawn from all time while the negatives come
    from one window, so the two groups differ systematically in how much service history they have,
    and history-derived features end up carrying the OPPOSITE sign in training to the one they have
    at test time. Measured 2026-08-16 on 20k: `has_10` raw AUC 0.3602 train vs 0.6885 test,
    `PMS_Count_Prior` 0.3792 vs 0.7224; group-permuting the whole schedule-keeping family RAISES
    20k test AUC by 0.19. The test sets never had this problem -- create_test_cohort() already
    builds a due-in-window cohort -- so 'window' is what makes training match test.

    `label_start` (window mode only) overrides the start of that shared window. The default from
    get_quarter_dates() is Jan 1 of the previous year, which on 20k leaves only 1,343 rows once the
    positives are windowed too -- too few to train on. Widening it is legitimate: what matters is
    that BOTH classes see the same window, not how long it is.

    'all' (added 2026-08-17)
    - Positives unchanged: every vehicle that ever completed the milestone, no date filter.
    - Negatives: every vehicle still pending for it, ALSO with no date filter -- the due-date window
      is simply not applied. The definitional filters stay (`Service_Num < svc`, not already past
      via `dfafter`, and the mileage cap unless `drop_mileage_cap`), because without them a
      "negative" is not a pending vehicle at all.
    - The sliding loop is SKIPPED. With the window gone there is nothing left to slide, and the base
      rate is whatever the two full populations give (20k: 7,591 pos / 6,884 neg = 52.4%).

    This is the mirror image of 'window'. Both fix the same asymmetry -- under 'ever' the positives
    are drawn from all time while the negatives come from one quarter -- but 'window' fixes it by
    restricting the positives, and 'all' by releasing the negatives. 'all' keeps far more data
    (14,475 rows on 20k vs 8,046) and needs no ratio tuning.
    """
    if label_mode not in ("ever", "window", "all"):
        raise ValueError(f"label_mode must be 'ever', 'window' or 'all', got {label_mode!r}")
    start_date_dt, quarter_end_dt = get_quarter_dates(selected_quarter, year)
    if label_mode == "window" and label_start:
        start_date_dt = pd.to_datetime(label_start)
        print(f"[window] shared due-date window overridden to start {start_date_dt.date()}")

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
    serv["Service_Num"] = resolve_service_num(serv)
    serv['Service_Date'] = parse_dates(serv['Service_Date'])
    
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

        # In 'window' mode the positives must sit in the SAME due-date window as the negatives, so
        # they need an Expected{svc}Date of their own. Built exactly as the negatives' is below
        # (:165): first-service date + 6 months per 10k of milestone. Under 'ever' this column is
        # left unset and back-filled from `Last Service Date - PMS` after the concat, as before.
        if label_mode == "window":
            turnup["FirstSrvDate"] = turnup["Invoice date"].fillna(turnup["First Service Date"])
            turnup[f"Expected{svc}Date"] = (turnup["FirstSrvDate"]
                                            + pd.DateOffset(months=expected_milestone_months(
                                                svc_num, date_model)))
            n_dated = int(turnup[f"Expected{svc}Date"].notna().sum())
            print(f"  [window] {len(turnup)} positives, {n_dated} with a usable expected date")

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
        months_to_add = expected_milestone_months(svc_num, date_model)
        missed[f"Expected{svc}Date"] = missed["FirstSrvDate"] + pd.DateOffset(months=months_to_add)
        
        missed = missed[missed["Vehicle Service Status"].isin(["InActive", "Lapsed", "Active"])]
        pending_base = missed.query(f"Service_Num < {svc_num}").copy()
        dfafter = df.query(f"Service_Num >= {svc_num}")["VIN"].unique()
        pending_base = pending_base[~pending_base["VIN"].isin(dfafter)]
        pending_base = pending_base.drop(["FirstSrvDate"], axis=1)
        milthreshold = svc_num * 1000
        if drop_mileage_cap:
            # `Last Service Mileage` is an EDA current-state field, so this cap excludes vehicles
            # that are PAST the milestone mileage but never had the service -- i.e. the most overdue
            # customers in the file. On 20k it removes 1,571 of 8,455 candidate negatives.
            print(f"  [mileage cap OFF] keeping {len(pending_base)} negatives; the cap would have "
                  f"cut them to {len(pending_base.query('`Last Service Mileage` < @milthreshold'))}")
        else:
            pending_base = pending_base.query("`Last Service Mileage` < @milthreshold")
        
        while iteration < max_iterations:
            print(f'{svc} - Iteration {iteration + 1}: Start Date = {start_date_adj.date()}')
            
            # 'all': no due-date window on the negatives at all, matching the positives, which have
            # never had one. This is the whole point of the mode -- the window is the single filter
            # that made the two classes come from different time periods.
            if label_mode == "all":
                pending = pending_base
            else:
                pending = pending_base[
                    (pending_base[f"Expected{svc}Date"] >= start_date_adj) &
                    (pending_base[f"Expected{svc}Date"] <= quarter_end_dt)
                ]
            
            # ── Combine Positives + Negatives ──
            if not pending.empty:
                pending = pending.copy()
                pending["TargetFlag"] = 0

            # 'window': cut the positives to the same due-date window as the negatives, so both
            # classes answer "due in this window -- did they turn up?" rather than comparing
            # all-time turn-ups against one quarter's pending list.
            if label_mode == "window":
                turnup_w = turnup[
                    (turnup[f"Expected{svc}Date"] >= start_date_adj) &
                    (turnup[f"Expected{svc}Date"] <= quarter_end_dt)
                ].copy()
            else:
                turnup_w = turnup

            finaltr = pd.concat([turnup_w, pending], axis=0)
            finaltr[f"Expected{svc}Date"] = finaltr[f"Expected{svc}Date"].fillna(finaltr["Last Service Date - PMS"])
            
            # Calculate ratio
            if len(finaltr) > 0:
                ratio = finaltr["TargetFlag"].sum() / len(finaltr)
            else:
                ratio = 0
            print(f'Target Range: {target_range[0]} to {target_range[1]}')
            print(f'Iteration {iteration + 1}: TargetFlag ratio = {round(ratio * 100, 2)}% | Rows: {len(finaltr)}')

            # 'all': there is no window left to slide, so the loop has nothing to tune. The base
            # rate is whatever the two full populations give.
            if label_mode == "all":
                print(f'  [all] single pass, natural base rate {round(ratio * 100, 2)}% '
                      f'({int(finaltr["TargetFlag"].sum())} pos / {len(finaltr)} rows) -- '
                      f'no due-date filter on either class')
                break

            # 'window': the window now DEFINES both classes, so sliding it to hit a target ratio
            # would be manufacturing the base rate. Take the cohort as it comes, one pass.
            if label_mode == "window":
                print(f'  [window] single pass, natural base rate {round(ratio * 100, 2)}% '
                      f'({int(finaltr["TargetFlag"].sum())} pos / {len(finaltr)} rows) -- '
                      f'window {start_date_adj.date()} .. {quarter_end_dt.date()}')
                break

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


def quarter_bounds(year: int, q: int):
    """(start, end) Timestamps for calendar quarter q of `year`."""
    start = pd.Timestamp(year=year, month=3 * q - 2, day=1)
    return start, start + pd.offsets.QuarterEnd(0)


def build_history_cohort(df, serv, milestone, q_start, q_end, grace_days=45,
                         date_model="schedule"):
    """Label ONE quarterly cohort straight from raw service history.

    This is the 'history' label mode. It exists because the 'ever' mode reads its label from EDA's
    `Last Service - PMS`, which means the vehicle's MOST RECENT PMS -- so a vehicle that did its 20k
    and then went on to 30k is neither a positive (last PMS is 30k, not 20k) nor a negative
    (`Service_Num < 20` fails). Measured on 20k: 52,533 vehicles ever did the service, only 7,591
    are labelled positive, and 44,991 fall out of the data entirely. The surviving positives are
    exactly the customers who did their 20k and never came back, which inverts every
    history-derived feature relative to the test cohorts.

    Here instead:
      cohort   every vehicle whose expected milestone date lands in [q_start, q_end], that has not
               already completed THIS milestone before the band opens (nothing left to predict).
               Vehicles that skipped the milestone and went further ARE kept -- they are the
               clearest negatives available, and the 'ever' mode discards 25,132 of them on 20k.
      positive the milestone appears in raw service history inside [q_start - grace, q_end + grace].
      negative anything else in the cohort.

    `grace_days` is the band either side of the due quarter, 45 by default. It is deliberately
    tight: it asks "did they turn up roughly when due", not "did they ever turn up". Cost of that
    choice, measured over the eligible population: the positive rate lands near 22% on 20k
    (+/-45d around a quarter is ~+/-90d around its centre) and falls to ~6% on 60k. There is no
    ratio-tuning loop here -- with the band fixed and the window defining both classes, sliding it
    would just be manufacturing the base rate.

    The caller runs this once per quarter and pairs each cohort with `filter_date=q_start`, so every
    feature in that cohort is cut before its own due date. That is what makes a multi-year training
    set safe; see refactored_test_dir/feature_cutoff_audit.md.
    """
    grace = pd.Timedelta(days=grace_days)
    band_lo, band_hi = q_start - grace, q_end + grace
    months = expected_milestone_months(milestone, date_model)

    c = df.copy()
    c[f"Expected{milestone}kDate"] = c["FirstSrvDate"] + pd.DateOffset(months=months)
    c = c[(c[f"Expected{milestone}kDate"] >= q_start) & (c[f"Expected{milestone}kDate"] <= q_end)]
    c = c[c["Vehicle Service Status"].isin(["InActive", "Lapsed", "Active"])]
    n_due = len(c)

    at_milestone = serv[serv["Service_Num"] == milestone]
    # already done before the band opened -> nothing to predict, drop. Mirrors
    # prepare_test_set.create_test_cohort(), except that it keys on `Service_Num >= milestone`;
    # here it is `== milestone` on purpose, so skippers stay in as negatives.
    done_early = set(at_milestone.loc[at_milestone["Service_Date"] < band_lo, "Vin_No"].unique())
    n_early = int(c["VIN"].isin(done_early).sum())
    c = c[~c["VIN"].isin(done_early)]

    in_band = set(at_milestone.loc[(at_milestone["Service_Date"] >= band_lo) &
                                   (at_milestone["Service_Date"] <= band_hi), "Vin_No"].unique())
    c["TargetFlag"] = c["VIN"].isin(in_band).astype(int)

    # vehicles that never do this milestone at all -- includes the skippers who go straight past it
    n_skip = int((~c["VIN"].isin(set(at_milestone["Vin_No"].unique()))).sum())
    print(f"  {q_start.date()}..{q_end.date()}  due {n_due:>6,} | "
          f"-{n_early:>5,} already done | cohort {len(c):>6,} | "
          f"pos {int(c['TargetFlag'].sum()):>5,} ({100*c['TargetFlag'].mean():>5.1f}%) | "
          f"never-did-it {n_skip:>6,}")
    return c.drop(columns=["FirstSrvDate"], errors="ignore")


def build_candidates_cohort(df, serv, milestone, win_start, win_end, grace_days=15,
                            date_model="schedule"):
    """Label the FULL 'candidates' cohort in one global vectorised pass.

    due date = whichever comes first of the schedule projection (FirstSrvDate + N months) and the
    service-1 -> service-10 burn-rate projection (due_date.burn_rate_date). `df` must already be
    invoice-date-guarded -- FirstSrvDate == a genuine `Invoice date`, never the First Service Date
    fallback -- and `serv` must carry Service_Num from features.resolve_service_num(), which is
    now what every path uses: Service_Code as the source of truth, Description only as a fallback.
    (This used to be two rival derivations -- raw Service_Code here, extract_kk(Description) in
    process_service_data() -- and they disagreed on 1,983 rows.)

    TargetFlag = 1 if the milestone was completed on or before quarter_end + grace_days, with NO
    restriction on how early the completion happened -- an "early completer" (already done before
    the due quarter even opened) is still counted a positive, flagged via EarlyCompleter, per the
    user's deliberate labelling decision (see the labelling plan's Context / "known weakness"
    section). This is the one substantive difference from build_history_cohort(), which drops early
    completers outright rather than keeping+flagging them.

    Ported from and verified byte-for-byte against a scratchpad reference build: 0 due-date
    mismatches across all 41,983 candidates, 2016Q1-2025Q4, grace 15d -> 23,269 positive / 18,714
    negative. The reproduction depends on due_date.burn_rate_date() sorting codes {1, 10, milestone}
    together (a handful of VINs carry two same-day records for the same Service_Num with different
    Mileage, and the "first visit" tie-break is sensitive to exactly what is sorted together with
    what) -- see that function's docstring.
    """
    vins = df["VIN"].values
    first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)

    months = expected_milestone_months(milestone, date_model)
    due_sched = first_srv + pd.DateOffset(months=months)

    # No cutoff restriction on the burn-rate anchor here -- this mirrors the verified reference
    # build exactly (see due_date.burn_rate_date()'s docstring on why `cutoff` exists at all: it is
    # a defensive check against a caller passing a looser serv_cut than intended, not something this
    # global, all-history candidate pass needs to restrict further).
    cutoff = serv["Service_Date"].max()
    due_burn = burn_rate_date(vins, serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]],
                              milestone, cutoff)
    due = earliest_date(due_sched, due_burn)
    due_source = np.where(due_burn.notna().to_numpy() & (due_burn.to_numpy() <= due_sched.to_numpy()),
                          "burn-rate", "schedule")

    at_milestone = serv[serv["Service_Num"] == milestone]
    date_ms = (at_milestone.sort_values("Service_Date")
              .groupby("Vin_No")["Service_Date"].first().reindex(vins))

    ms_col = f"Actual{milestone}kDate"
    c = df.copy()
    c[f"Expected{milestone}kDate"] = due.values
    c["DueSource"] = due_source
    c[ms_col] = date_ms.values
    c = c.dropna(subset=[f"Expected{milestone}kDate"])

    inwin = ((c[f"Expected{milestone}kDate"] >= win_start) &
             (c[f"Expected{milestone}kDate"] <= win_end))
    c = c[inwin].copy()

    quarter = c[f"Expected{milestone}kDate"].dt.to_period("Q")
    c["CohortQuarter"] = quarter.astype(str)
    c["QuarterEnd"] = quarter.dt.end_time.dt.normalize()
    q_start = quarter.dt.start_time

    grace = pd.Timedelta(days=grace_days)
    ms = c[ms_col]
    c["TargetFlag"] = (ms.notna() & (ms <= c["QuarterEnd"] + grace)).astype(int)
    c["EarlyCompleter"] = (ms.notna() & (ms < q_start)).astype(int)

    print(f"  candidates {len(c):>7,} | pos {int(c['TargetFlag'].sum()):>7,} "
          f"({100*c['TargetFlag'].mean():>5.1f}%) | early completers "
          f"{int(c['EarlyCompleter'].sum()):>7,} ({100*c['EarlyCompleter'].mean():>5.1f}%) | "
          f"burn-rate due {int((c['DueSource']=='burn-rate').sum()):>7,}")
    return c.drop(columns=["FirstSrvDate"], errors="ignore")


def run_candidates_mode(args, milestone, label_suffix, eda_path, service_history_path,
                        rfm_path, appointdf_path, digidf_path, vhc_path, servcode_path):
    """Build the 'candidates' training matrix (Stage A+B of the labelling plan).

    Stage A: build_candidates_cohort() labels the WHOLE 2016Q1-2025Q4 (by default) population in one
    vectorised pass and assigns each candidate to its due quarter (CohortQuarter).

    Stage B: reuses run_history_mode()'s per-quarter discipline -- one process_service_data() call
    per present quarter, features cut at that quarter's START (not q_start - grace like 'history'
    mode: here the outcome window only extends FORWARD from the quarter, since early completers are
    counted from raw history that predates the cutoff and their OWN service record is already
    excluded from features via serv1's `Service_Num < milestone` filter, so nothing about the
    labelling window sits inside the feature-cutoff gap). Two changes over run_history_mode(), both
    from the plan: the service history is read ONCE and passed as a DataFrame (not re-read per
    quarter), and the 15 EDA current-state columns (refactored_test_dir/eda_currentstate_features.json)
    are dropped from every quarter's output -- safe to drop only 8 quarters back, unsafe over a
    10-year span (see feature_cutoff_audit.md).
    """
    import time
    import json as _json

    def _parse_yq(s, name):
        s = s.upper().strip()
        if len(s) != 6 or s[4] != 'Q' or not s[:4].isdigit() or s[5] not in '1234':
            sys.exit(f"--{name} must look like 2016Q1, got {s!r}")
        return int(s[:4]), int(s[5])

    from_y, from_q = _parse_yq(args.quarters_from or "2016Q1", "quarters-from")
    to_y, to_q = _parse_yq(args.quarters_to or "2025Q4", "quarters-to")
    win_start, _ = quarter_bounds(from_y, from_q)
    _, win_end = quarter_bounds(to_y, to_q)
    grace_days = args.grace_days

    print(f"\n--- candidates mode: due date in [{win_start.date()}, {win_end.date()}], "
          f"grace +{grace_days}d ---")

    df = pd.read_csv(eda_path, low_memory=False, encoding="ISO-8859-1")
    df = df.query("`Last Service - PMS` != '-'").copy()
    df["Last Service Mileage"] = pd.to_numeric(df["Last Service Mileage"], errors="coerce")
    df["Last PMS Mileage"] = pd.to_numeric(df["Last PMS Mileage"], errors="coerce")
    for col in ["Invoice date", "First Service Date", "Last Service Date - PMS",
                "Last Service Date", "Next Service Date"]:
        df[col] = parse_dates(df[col])
    df["Service_Num"] = df["Last Service - PMS"].apply(extract_k1)
    df["FirstSrvDate"] = df["Invoice date"].fillna(df["First Service Date"])
    df = df.dropna(subset=["FirstSrvDate"]).drop_duplicates(subset="VIN")

    # Invoice-date guard: without a genuine sale date, "first service" is merely the first visit
    # that happens to fall inside the file, and the vehicle may have years of unseen history
    # (including its own milestone completion) before the file starts. See the labelling plan's
    # Context section 4 for the measured effect (63.2% "never did a 20k" without a sale date vs
    # 16.5% with one).
    n_before_guard = len(df)
    df = df[df["Invoice date"].notna()].copy()
    print(f"invoice-date guard: {n_before_guard:,} -> {len(df):,} EDA vehicles "
          f"({n_before_guard - len(df):,} dropped, no genuine sale date)")

    serv_raw = pd.read_csv(service_history_path, low_memory=False, encoding="ISO-8859-1")
    serv_raw["Service_Date"] = parse_dates(serv_raw["Service_Date"])
    serv_raw["Mileage"] = pd.to_numeric(serv_raw["Mileage"], errors="coerce")
    serv_raw = serv_raw.dropna(subset=["Service_Date"])
    # Service_Code first, Description as fallback -- the same rule process_service_data() uses, so
    # candidate labelling and feature derivation can no longer disagree about what service a row is.
    serv_raw["Service_Num"] = resolve_service_num(serv_raw)
    print(f"EDA rows {len(df):,} | service history rows {len(serv_raw):,}\n")

    cohort = build_candidates_cohort(df, serv_raw, milestone, win_start, win_end,
                                     grace_days=grace_days, date_model=args.date_model)

    ms_col = f"Actual{milestone}kDate"
    bookkeeping_cols = ["CohortQuarter", "EarlyCompleter", "DueSource", "QuarterEnd", ms_col]

    currentstate_path = "refactored_test_dir/eda_currentstate_features.json"
    currentstate_cols = _json.load(open(currentstate_path)) if os.path.exists(currentstate_path) else []
    if currentstate_cols:
        print(f"\nwill drop {len(currentstate_cols)} EDA current-state columns from every cohort "
              f"(recomputed per extract; unsafe over a multi-year span -- see feature_cutoff_audit.md)")
    else:
        print(f"\nWARNING: {currentstate_path} not found -- EDA current-state columns NOT dropped")

    quarters = sorted(cohort["CohortQuarter"].unique())
    print(f"\n{len(quarters)} quarters present, {quarters[0]} .. {quarters[-1]}\n\ncohorts:")
    parts, t_start = [], time.time()
    for cq in quarters:
        sub = cohort[cohort["CohortQuarter"] == cq]
        if len(sub) < 5:
            print(f"    {cq} skipped -- only {len(sub)} rows")
            continue
        y, q = int(cq[:4]), int(cq[5])
        q_start, q_end = quarter_bounds(y, q)
        # process_service_data() treats every non-ID object column as a categorical to one-hot
        # encode, and has no notion of "bookkeeping" -- passing these through would OHE
        # CohortQuarter/DueSource into junk per-quarter dummies, silently drop Actual{m}kDate
        # (high-cardinality), and collide on re-merge with EarlyCompleter (an int column, so it
        # passes through untouched and then gets suffixed _x/_y against the merge below). Strip them
        # before the call and re-attach by VIN afterward instead of trusting any of that survives.
        bk = sub[["VIN"] + bookkeeping_cols].set_index("VIN")
        feats = process_service_data(
            mastersheet=sub.drop(columns=bookkeeping_cols, errors="ignore"),
            servhistory=serv_raw, rfm=rfm_path,
            servcode=servcode_path, appointdf=appointdf_path, digidf=digidf_path, vhc=vhc_path,
            filter_date=q_start.strftime("%Y-%m-%d"),
            last_service_code=milestone,
            is_test=True,   # every row is due AFTER q_start by construction -- the master-sheet
                            # date filter would otherwise delete the whole cohort
        )
        feats = feats.drop(columns=bookkeeping_cols, errors="ignore").merge(
            bk, left_on="VIN", right_index=True, how="left")
        if currentstate_cols:
            feats = feats.drop(columns=[c for c in currentstate_cols if c in feats.columns],
                               errors="ignore")
        parts.append(feats)
        print(f"    {cq} -> {feats.shape[0]:,} rows x {feats.shape[1]} cols "
              f"({time.time()-t_start:.0f}s elapsed)")

    if not parts:
        sys.exit("candidates mode produced no cohorts -- check --quarters-from/--quarters-to")

    # Same union-reindex-zero-fill discipline as run_history_mode(): cohorts emit different
    # one-hot/per-category columns, and a category genuinely absent from a quarter is correctly 0,
    # not NaN.
    union = list(dict.fromkeys(c for p in parts for c in p.columns))
    filled = {}
    for i, p in enumerate(parts):
        missing = [c for c in union if c not in p.columns]
        for c in missing:
            filled[c] = filled.get(c, 0) + len(p)
        parts[i] = p.reindex(columns=union, fill_value=0)
    if filled:
        print(f"\n{len(filled)} per-category columns absent from at least one cohort, "
              f"zero-filled there ({sum(filled.values()):,} cells):")
        for c, n in sorted(filled.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {c:<45} {n:>6,} rows")

    out = pd.concat(parts, axis=0, ignore_index=True, sort=False)
    dupes = out["VIN"].duplicated().sum()
    if dupes:
        print(f"WARNING: {dupes} duplicate VINs across cohorts -- keeping the earliest")
        out = out.drop_duplicates(subset="VIN", keep="first")

    path = f"refactored_test_dir/final_processed_{milestone}k{label_suffix}.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    print(f"  {len(out):,} rows x {out.shape[1]} cols | "
          f"positives {int(out['TargetFlag'].sum()):,} ({100*out['TargetFlag'].mean():.1f}%) | "
          f"early completers {int(out['EarlyCompleter'].sum()):,} "
          f"({100*out['EarlyCompleter'].mean():.1f}%) | {time.time()-t_start:.0f}s total")
    print(f"  rows per cohort:\n{out['CohortQuarter'].value_counts().sort_index().to_string()}")


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
    # mastertrain reaches here either as an in-memory frame (dates already datetime) or read back
    # from a CSV (dates written out as ISO). parse_dates handles both, and would still be right if
    # it ever sees the raw day-first EDA text -- hardcoding either convention here breaks one case.
    master_dates['FirstSrvDate'] = parse_dates(master_dates['Invoice date'].fillna(master_dates['First Service Date']))
    
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
        if not isinstance(servhistory, pd.DataFrame) and not os.path.exists(servhistory):
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
            mastertrain['Last Service Date - PMS'] = parse_dates(mastertrain['Last Service Date - PMS'])
            mastertrain[col_name] = parse_dates(mastertrain[col_name])
            
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
            # Accept a preloaded DataFrame the same way `mastersheet` already does, so a caller
            # running many process_service_data() calls over the same history (e.g. one per
            # quarterly cohort) does not re-read and re-parse a 124MB CSV every time. `.copy()`
            # because this function mutates serv1 in place below.
            if isinstance(servhistory, pd.DataFrame):
                serv1 = servhistory.copy()
            else:
                serv1 = pd.read_csv(servhistory,low_memory=False) #Input Service History
            logger.debug(f"Service history initial shape: {serv1.shape}")
            
            logger.info("Processing service history data")
            serv1['Service_Date'] = parse_dates(serv1['Service_Date'])
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
            # Service_Code first, Description as fallback. The '<=10' np.where patch that used
            # to sit here is folded into resolve_service_num(), which also fixes the Q1-2026-only
            # '11_20' spelling that bare extract_kk reads as service 1120.
            serv1['Service_Num'] = resolve_service_num(serv1)
            
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

            logger.info("Calculating PMS frequency (non-leaky replacement for EDA Service Frequency)")
            pms_freq = derive_pms_frequency(serv, filterdate)

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
                (pms_freq, 'PMS frequency'),
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
                            'recent_complaint_resolution_days',
                            'PMS_Freq_PerYear', 'PMS_Count_Prior', 'Years_Since_First_PMS'] + top_branches
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
            # 'Vehicle Age' was in this list until 2026-08-12; legacy/pmstrainfeatureEng.py:2520-2523
            # is otherwise identical and does NOT drop it. Removed so the column survives
            # feature-eng and can actually be evaluated -- it is plausible behavioural signal
            # (older vehicles service differently), not obviously a label leak.
            # NOTE it is still listed in LEAK_COLS (pms_model.py), so retrain.py drops it before
            # training. Take it out of there too once its leakiness has been measured.
            # 'Service Frequency' added 2026-08-13: the EDA column is a current-state snapshot and
            # leaks the outcome (test AUC 0.9449 alone, vs 0.8300 on train -- stronger on unseen
            # data is the signature of post-outcome information). Replaced by the non-leaky
            # PMS_Freq_PerYear derived above from pre-cutoff service history.
            rem = ['New / Used Category','Current Customer','First Service Date',
                'Last Service Date', 'Next Service Date','Vehicle Lifetime in Years','MileagePMS',
                'Last Service - PMS','Sale Invoice Year','Invoice date','Target Revenue','Potential Revenue',
                  'Final Revenue', 'Service Frequency']
            filtered_dfnew= filtered_dfnew.drop(rem,axis=1,errors='ignore')

            # PMS_Delay must be computed HERE, while Vehicle_Key_ExpectedServices is still present
            # (it is dropped at line ~833). See compute_pms_delay() for why the expected term has to
            # be that per-vehicle column and not a constant.
            # All four candidate definitions are emitted in ONE run so the A/B compares
            # byte-identical rows -- no re-derivation, no seed drift. Requires has_* (built :347,
            # merged :534) and Years_Since_First_PMS (:494 -> :534), both present by now.
            # pms_model.select_pms_delay_variant() collapses them to one column at train/score time.
            pms_delay_variants = compute_pms_delay(filtered_dfnew, last_service_code)
            for _c in pms_delay_variants.columns:
                filtered_dfnew[_c] = pms_delay_variants[_c]
            logger.info(f"PMS_Delay variants emitted: {list(pms_delay_variants.columns)}")
            
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


            # 'Service Frequency' removed 2026-08-13 -- the column no longer exists at this point
            # (dropped above in `rem`), and pms[i] here indexes directly so it would KeyError.
            obint = ['Total Promoter','Total Passive','Total Detractor','Total Survey',
            'CC','Weight','Height','Wheel Base']
            for i in obint:
                pms[i] = pd.to_numeric(pms[i], errors='coerce')
            pms = pms.drop('Service_Date',axis=1,errors = 'ignore')
            pms[['Months_Since_Last_NPMS','Avg_Mileage_Interval_NPMS']] = pms[['Months_Since_Last_NPMS','Avg_Mileage_Interval_NPMS']].fillna(0)
            colssrvd = pms.pop('Last Service Date - PMS')
            pms.insert(3, colssrvd.name, colssrvd)
            pms['Last Service Date - PMS'] = parse_dates(pms['Last Service Date - PMS'])
            pms[f'Next{last_service_code}K_Due'] = parse_dates(pms[f'Next{last_service_code}K_Due'])
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
            
            
            # NOTE: `& (pmsnew["TargetFlag"]==1)` removed 2026-08-13 -- it leaked the label into the
            # feature (the flag could only ever be 1 for positives) and mismatched the
            # prediction-side definition in predservicemil_4.py:2679, which has no label term and so
            # means something entirely different at inference -- causing systematic over-prediction.
            # Verified before the fix on the 60k matrix: 208 rows had the flag set, ALL 208 were
            # TargetFlag==1, zero exceptions. This restores the definition already corrected in
            # legacy/pmstrainfeatureEng.py:2621-2625, where the same bug was found and documented.
            pmsnew["LowMileageFreqUsers"] = np.where(
             (pmsnew["Avg_Service_Interval_PMS"].between(0, 7)) & (pmsnew["Last Service Mileage"].between(0, (last_service_code-10)*1000)),1,0)
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


def run_history_mode(args, milestone, label_suffix, eda_path, service_history_path,
                     rfm_path, appointdf_path, digidf_path, vhc_path, servcode_path):
    """Build the 'history' training matrix: one labelled cohort per quarter, features cut at that
    quarter's start, then concatenated.

    The per-quarter loop is the whole point. Under the 'ever' label mode every row shares one global
    cutoff, which is fine only because the cohort is roughly contemporaneous. Labelling from service
    history pulls in vehicles whose milestone fell anywhere in 2016-2025, and 88.2% of them complete
    at least one further service before a 2025-12-31 cutoff (median 5). A single cutoff would hand
    the model a snapshot taken five services after the event it is meant to predict.

    Cost is one process_service_data() call per quarter, ~55s per 4,000 rows.
    """
    import time

    n_q = args.quarters
    end_year, end_q = args.year, int(args.quarter.upper().lstrip("Q"))
    # walk back n_q quarters from the one BEFORE the prediction quarter -- the prediction quarter
    # itself belongs to the test set, never to training
    quarters, y, q = [], end_year, end_q
    for _ in range(n_q):
        q -= 1
        if q == 0:
            q, y = 4, y - 1
        quarters.append((y, q))
    quarters.reverse()

    print(f"\n--- history mode: {n_q} quarterly cohorts, "
          f"{quarters[0][0]}Q{quarters[0][1]} .. {quarters[-1][0]}Q{quarters[-1][1]}, "
          f"grace +/-{args.grace_days}d ---")

    df = pd.read_csv(eda_path, low_memory=False, encoding="ISO-8859-1")
    df = df.query("`Last Service - PMS` != '-'").copy()
    df["Last Service Mileage"] = pd.to_numeric(df["Last Service Mileage"], errors="coerce")
    df["Last PMS Mileage"] = pd.to_numeric(df["Last PMS Mileage"], errors="coerce")
    for col in ["Invoice date", "First Service Date", "Last Service Date - PMS",
                "Last Service Date", "Next Service Date"]:
        df[col] = parse_dates(df[col])
    df["Service_Num"] = df["Last Service - PMS"].apply(extract_k1)
    df["FirstSrvDate"] = df["Invoice date"].fillna(df["First Service Date"])
    df = df.dropna(subset=["FirstSrvDate"]).drop_duplicates(subset="VIN")

    serv = pd.read_csv(service_history_path, low_memory=False, encoding="ISO-8859-1")
    serv["Service_Num"] = resolve_service_num(serv)
    serv["Service_Date"] = parse_dates(serv["Service_Date"])
    serv = serv.dropna(subset=["Service_Date"])

    print(f"EDA rows {len(df):,} | service history rows {len(serv):,}\n")
    print("cohorts:")
    parts, t_start = [], time.time()
    for y, q in quarters:
        q_start, q_end = quarter_bounds(y, q)
        cohort = build_history_cohort(df, serv, milestone, q_start, q_end,
                                      grace_days=args.grace_days, date_model=args.date_model)
        if len(cohort) < 50:
            print(f"    skipped -- only {len(cohort)} rows")
            continue
        # Cut features at the moment the OUTCOME WINDOW OPENS, not at the quarter start. The band
        # runs [q_start - grace, q_end + grace], so a cutoff of q_start would leave its first
        # `grace` days inside the feature window -- the model could see ordinary visits made during
        # the period it is being asked to predict. The milestone's own visit could never leak
        # (serv1 is restricted to Service_Num < milestone at :514), but neighbouring visits could.
        cutoff = q_start - pd.Timedelta(days=args.grace_days)
        feats = process_service_data(
            mastersheet=cohort, servhistory=service_history_path, rfm=rfm_path,
            servcode=servcode_path, appointdf=appointdf_path, digidf=digidf_path, vhc=vhc_path,
            filter_date=cutoff.strftime("%Y-%m-%d"),
            last_service_code=milestone,
            is_test=True,   # the master-sheet date filter would delete the whole cohort: every row
                            # is due AFTER q_start by construction
        )
        feats["CohortQuarter"] = f"{y}Q{q}"
        parts.append(feats)
        print(f"    -> {feats.shape[0]:,} rows x {feats.shape[1]} cols "
              f"({time.time()-t_start:.0f}s elapsed)")

    if not parts:
        sys.exit("history mode produced no cohorts -- check the year/quarter arguments")

    # Cohorts emit different one-hot/per-category columns -- a quarter with no Renault-Alain visit
    # produces no dummy for it. A plain concat leaves those NaN for that quarter, which is not the
    # same as 0 to the imputer downstream. Reindex every part to the union and fill 0 EXPLICITLY:
    # the category genuinely did not occur, and for the per-category COUNT columns
    # (InvoicedPart__*, DeferredPart__*, LostPart__*) a count of 0 is equally correct. Only absent
    # columns are touched; NaN inside a column a cohort did produce is left for the imputer.
    union = list(dict.fromkeys(c for p in parts for c in p.columns))
    filled = {}
    for i, p in enumerate(parts):
        missing = [c for c in union if c not in p.columns]
        for c in missing:
            filled[c] = filled.get(c, 0) + len(p)
        parts[i] = p.reindex(columns=union, fill_value=0)
    if filled:
        print(f"\n{len(filled)} per-category columns absent from at least one cohort, "
              f"zero-filled there ({sum(filled.values()):,} cells):")
        for c, n in sorted(filled.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {c:<45} {n:>6,} rows")

    out = pd.concat(parts, axis=0, ignore_index=True, sort=False)
    # A VIN can be due in only one quarter, so duplicates mean an upstream merge fanned out.
    dupes = out["VIN"].duplicated().sum()
    if dupes:
        print(f"WARNING: {dupes} duplicate VINs across cohorts -- keeping the earliest")
        out = out.drop_duplicates(subset="VIN", keep="first")

    path = f"refactored_test_dir/final_processed_{milestone}k{label_suffix}.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    print(f"  {len(out):,} rows x {out.shape[1]} cols | "
          f"positives {int(out['TargetFlag'].sum()):,} ({100*out['TargetFlag'].mean():.1f}%) | "
          f"{time.time()-t_start:.0f}s total")
    print(f"  rows per cohort:\n{out['CohortQuarter'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="Run PMS Training Feature Engineering")
    parser.add_argument("year", type=int, help="Prediction Year (e.g., 2026)")
    parser.add_argument("quarter", type=str, help="Prediction Quarter (e.g., Q1, Q2, Q3, Q4)")
    parser.add_argument("milestone", type=str, help="Service milestone (e.g., 20, 20k, 30)")
    parser.add_argument("--train-cutoff", type=str, help="Optional: YYYY-MM-DD date to filter service history (e.g., 2025-12-31). Defaults to the end of the previous quarter.", default=None)
    parser.add_argument("--drop-mileage-cap", action="store_true",
                        help="Keep negatives whose Last Service Mileage is already past the "
                             "milestone. The cap removes 1,571 of 8,455 candidate negatives on 20k "
                             "-- arguably the most overdue customers in the file.")
    parser.add_argument("--label-mode", choices=["ever", "window", "all", "history", "candidates"],
                        default="ever",
                        help="'ever' (default): a positive is any vehicle that completed the "
                             "milestone at any time. 'window': positives and negatives must both "
                             "be DUE in the same window, which is what the test cohorts already "
                             "do. See prepare_pms_datasets(). 'history': labels come from raw "
                             "service history instead of EDA's `Last Service - PMS`, one cohort "
                             "per quarter with the feature cutoff set to that quarter's start. "
                             "See build_history_cohort() and run_history_mode(). 'candidates': like "
                             "'history', but the due date is whichever comes first of the schedule "
                             "or a service-1->service-10 burn-rate projection, requires a genuine "
                             "EDA Invoice date, spans --quarters-from/--quarters-to (years, not a "
                             "walk-back count), and counts early completers as positives (flagged "
                             "via EarlyCompleter) instead of dropping them. See "
                             "build_candidates_cohort() and run_candidates_mode().")
    parser.add_argument("--quarters", type=int, default=8,
                        help="history mode only: how many quarterly cohorts to build, walking back "
                             "from the quarter BEFORE the prediction quarter. Default 8 (two "
                             "years). More quarters buys rows and older, less representative "
                             "customers.")
    parser.add_argument("--quarters-from", type=str, default=None,
                        help="candidates mode only: first due-quarter to include, e.g. 2016Q1. "
                             "Default 2016Q1.")
    parser.add_argument("--quarters-to", type=str, default=None,
                        help="candidates mode only: last due-quarter to include, e.g. 2025Q4. "
                             "Default 2025Q4.")
    parser.add_argument("--grace-days", type=int, default=45,
                        help="history/candidates mode: how far past the due quarter a completion "
                             "still counts as turning up (candidates mode: forward only -- an early "
                             "completion is a positive regardless of how early, see 'candidates' "
                             "above). Default 45 for history mode; the labelling plan calls for 15 "
                             "with --label-mode candidates. Tight on purpose in history mode -- it "
                             "asks 'did they turn up roughly when due', not 'did they ever'. "
                             "Widening it raises the positive rate steeply (20k over the eligible "
                             "population: 11.2%% at +/-45d, 22.4%% at +/-90d, 41.8%% at +/-180d) "
                             "but drifts back toward the 'ever' question.")
    parser.add_argument("--label-start", type=str, default=None,
                        help="window mode only: YYYY-MM-DD start of the shared due-date window. "
                             "Defaults to Jan 1 of the previous year. Widening it buys rows but "
                             "costs GRACE-PERIOD consistency: the label is still 'ever completed' "
                             "against data ending 2026-12, so a vehicle due in 2018 had 8 years to "
                             "turn up while one due in 2025 had 1 -- and the test cohorts (due "
                             "Q1/Q2 2026) get ~1. Measured: a 2018 start cost 40k 0.13 AUC.")
    parser.add_argument("--date-model", choices=["schedule", "calibrated"], default="schedule",
                        help="How Expected{svc}Date is computed. 'schedule' (default): 6 months "
                             "per 10k, the original rule. 'calibrated': measured medians from "
                             "service history (features.MILESTONE_MONTHS_CALIBRATED) -- the "
                             "schedule runs up to a year late (36 vs 22.1 months at 60k). MUST "
                             "match prepare_test_set.py --date-model or train and test disagree "
                             "on who is due when.")
    parser.add_argument("--label-tag", type=str, default="",
                        help="extra suffix on the output matrix, e.g. --label-tag narrow -> "
                             "final_processed_20k_window_narrow.csv. Lets two window builds with "
                             "different --label-start coexist instead of overwriting each other.")
    args = parser.parse_args()

    milestone_val = int(args.milestone.lower().replace('k', ''))
    # 'window' writes to its own matrix so an A/B never destroys the 'ever' one. retrain.py picks
    # the matching file up via $env:PMS_LABEL_MODE.
    # The suffix has to carry the date model too: 'calibrated' moves Expected{svc}Date for BOTH
    # label modes (it defines the negatives' window under 'ever' as well), so a calibrated build
    # must not overwrite a schedule one.
    label_suffix = "" if args.label_mode == "ever" else f"_{args.label_mode}"
    if args.date_model != "schedule":
        label_suffix += "_cal"
    if args.drop_mileage_cap:
        label_suffix += "_nomil"
    # a different grace band selects a different positive set, so it must not overwrite the default
    if args.label_mode == "history" and args.grace_days != 45:
        label_suffix += f"_g{args.grace_days}"
    if args.label_tag:
        label_suffix += f"_{args.label_tag}"

    print(f"--- Running Training Data Preparation for {milestone_val}k "
          f"(label mode: {args.label_mode}) ---")
    
    # Files
    # The Q2-2026 snapshot is the live data set. Keep this block identical to the one in
    # prepare_test_set.py -- if training and test read different vintages they describe different
    # businesses, and nothing in the pipeline will tell you.
    eda_path = os.path.join("data", "EDA_Q2-2026.csv")
    service_history_path = os.path.join("data", "Service_History_Q2-2026.csv")
    rfm_path = "data/RFM_Q2-2026.csv"
    appointdf_path = "data/Appointments_Q2-2026.csv"
    digidf_path = "data/Digital_Sessions_Q2-2026.csv"
    vhc_path = "data/VHC_Q2-2026.csv"
    servcode_path = "data/Service_Code_Desc.csv"
    
    if not os.path.exists(eda_path):
        print(f"Error: Could not find {eda_path}")
        sys.exit(1)
    if not os.path.exists(service_history_path):
        print(f"Error: Could not find {service_history_path}")
        sys.exit(1)
    
    # 'history' takes a completely different route: it builds one labelled cohort PER QUARTER and
    # runs the feature pipeline once per cohort with that quarter's start as the cutoff, so a
    # multi-year training set never lets a feature see past its own row's due date. Everything below
    # this block (prepare_pms_datasets -> one process_service_data call) assumes a single global
    # cutoff, which is only safe when every row is due at roughly the same time.
    if args.label_mode == "history":
        run_history_mode(args, milestone_val, label_suffix, eda_path, service_history_path,
                         rfm_path, appointdf_path, digidf_path, vhc_path, servcode_path)
        sys.exit(0)

    # 'candidates': see run_candidates_mode()'s docstring. Labels the whole
    # --quarters-from/--quarters-to span in one vectorised pass (build_candidates_cohort()), then
    # reuses the same per-quarter process_service_data() discipline as 'history' mode.
    if args.label_mode == "candidates":
        run_candidates_mode(args, milestone_val, label_suffix, eda_path, service_history_path,
                            rfm_path, appointdf_path, digidf_path, vhc_path, servcode_path)
        sys.exit(0)

    # 1. Prepare PMS Datasets (old-style target labelling)
    final_df = prepare_pms_datasets(
        df_path=eda_path,
        service_history_path=service_history_path,
        service_types=[f"{milestone_val}k"],
        selected_quarter=args.quarter,
        year=args.year,
        target_range=(0.35, 0.45),
        output_prefix="TestRefactored",
        label_mode=args.label_mode,
        label_start=args.label_start,
        date_model=args.date_model,
        drop_mileage_cap=args.drop_mileage_cap,
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
    raw_processed_path = rf"refactored_test_dir\final_processed_{milestone_val}k{label_suffix}.csv"
    final_df_after_process.to_csv(raw_processed_path, index=False)
    
    # 3. Feature Selection -- kept for inspection only.
    #
    # RETIRED 2026-08-12: this block used to write a second, competing feature list to
    # models/{m}k/selected_features.json (~33 cols) alongside a narrowed training_features.csv.
    # Neither is consumed by anything: retrain.py reads the FULL matrix written above
    # (refactored_test_dir/final_processed_{m}k.csv) and derives the real feature list itself, then
    # saves it to models/selected_features_{m}k.json -- one file per model, at the models/ root.
    # Having two files with the same name and incompatible contents was a live trap: filtering a
    # test set by the MI shortlist left 67-78% of the model's inputs zero-filled and still ran clean.
    print("\n--- Running Feature Selection (MI scores, for inspection only) ---")
    selected_features, mi_scores_df = feature_sel(final_df_after_process, cumulative_threshold=0.85)
    print(f"MI shortlist: {len(selected_features)} of {final_df_after_process.shape[1]} columns "
          f"(not used for training -- retrain.py selects its own)")

    model_dir_path = f"models/{milestone_val}k"
    os.makedirs(model_dir_path, exist_ok=True)

    # MI scores stay -- they are diagnostics, not a feature list, so nothing can mistake them.
    mi_scores_path = f"{model_dir_path}/mi_scores.csv"
    mi_scores_df.to_csv(mi_scores_path, index=False)
    print(f"Saved MI scores to {mi_scores_path}")

    # --- commented out: retired outputs, see note above ---
    # final_cols = ["VIN", "TargetFlag"] + selected_features
    # train_master = final_df_after_process[final_cols]
    # training_dir = f"{model_dir_path}/training"
    # os.makedirs(training_dir, exist_ok=True)
    # train_path = f"{training_dir}/training_features.csv"
    # train_master.to_csv(train_path, index=False)
    # print(f"Saved training set to {train_path}")
    #
    # feature_list_path = f"{model_dir_path}/selected_features.json"
    # with open(feature_list_path, 'w') as f:
    #     json.dump(selected_features, f)
    # print(f"Saved selected feature list to {feature_list_path}")