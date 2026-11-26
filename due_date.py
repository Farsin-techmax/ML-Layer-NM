"""due_date.py — candidate due-date models for a PMS milestone.

Built for the Phase 1 bake-off in the approved plan (see
refactored_test_dir/due_date_bakeoff.md for the scored results). Five candidates:

  schedule    FirstSrvDate + 6 months x (milestone/10)          -- existing default
                (features.expected_milestone_months(milestone, 'schedule'))
  calibrated  FirstSrvDate + MILESTONE_MONTHS_CALIBRATED[milestone]  -- existing, biased early
                (features.expected_milestone_months(milestone, 'calibrated'))
  mileage     last service date + remaining km / daily mileage rate. FIXED relative to
                create_test_cohort(use_mileage_projection=True): a vehicle already past the
                target mileage, or with under 30 days / no history before the cutoff, is EXCLUDED
                (NaT) rather than collapsed onto the last service date.
  interval    last completed lower-milestone date + remaining km / (per-vehicle
                Avg_Mileage_Interval_PMS km-per-block / population median days-per-PMS-interval).
                Mirrors features.derive_pms_mileage_features's Mileage_Diff logic (a single prior
                PMS record's "diff" is its own odometer reading, i.e. km covered in that first
                block) but adds a time axis, which that function does not produce.
  earliest    min(schedule, mileage) per vehicle, NaT-safe -- "every N km or M months, whichever
                comes first," which is how manufacturer service schedules are actually written.

A sixth candidate, `burn_rate_date`, was added later for the "candidates" labelling plan
(pmstrainfeatureeng_refactored.py --label-mode candidates / build_candidates_cohort()): the
most-recent-interval rate (1->10 for a 20k projection, 10->20 for 30k, 20->30 for 40k, ... i.e.
(M-20)->(M-10) for milestone M, generalized 2026-09-09) projected to the target mileage, distinct
from both `mileage` (a lifetime rate) and `interval` (a population-median-based rate) above. See
its own docstring.

A seventh, `last_completed_plus_months`, was added 2026-10-08 from a vehicle-level trace
(JN8AY2NY9R9499028, see CLAUDE.md's Gotchas): the vehicle's last completed milestone below the
target + a flat number of months (default 6), no rate computed at all. See its own docstring for
why this beats both `schedule` and `burn_rate` on vehicles whose pace changed after their earlier
services.

An eighth, `avg_interval_date`, was added 2026-10-09: same "reset the clock at the last completed
milestone" idea as `last_completed_plus_months`, but instead of a flat 6 months, projects forward by
the vehicle's own average historical block length in months (FirstSrvDate->1, 1->10, 10->20, ...,
up to its highest completed milestone below the target). See its own docstring.

All functions that touch service history take a `serv_cut` frame the CALLER has already restricted
to `Service_Date <= cutoff` -- nothing here re-filters by date, so passing an uncut frame leaks
future information into the prediction. Required serv_cut columns: Vin_No, Service_Date, Mileage,
Service_Num (see pmstrainfeatureeng_refactored.py for the '<=10' -> 10 fix that must be applied
before calling).
"""
import numpy as np
import pandas as pd


def first_srv_date(df, serv, vin_col="VIN"):
    """Earliest evidence the vehicle existed, collapsed to exactly ONE date per VIN.

    Every due-date model in this module is anchored on this date, so an anchor that is late by N
    months moves the whole cohort by N months. The old rule -- per EDA ROW,
    `Invoice date` else `First Service Date` -- was wrong two ways:

    1. **One VIN can hold several EDA rows.** When a vehicle is resold the dealer group issues a new
       Vehicle_Key with the sequence bumped (`03-557292-0` -> `03-557292-1`) and EDA carries BOTH,
       each with its OWN `First Service Date` -- the new owner's first visit, not the vehicle's.
       See the company-rule note in CLAUDE.md. 1,102 VINs in EDA_Q2-2026 have >1 row (325 survive
       the `Last Service - PMS != '-'` filter), so the same VIN got two different due dates and
       could land in two different quarters as two rows. The training side hid half of this with a
       `.drop_duplicates(subset='VIN')` that kept whichever row came first in the CSV -- arbitrary,
       not earliest -- and the test-set builder did not dedupe at all.
    2. **EDA's `First Service Date` is not always the first service.** On `5N1DR3MR9RC248028` the
       older key reads 2025-04-19, which is that vehicle's code-10 visit; its true first service is
       2024-07-03 (code 1, 1,073 km). So picking the earlier EDA row is not enough either --
       service history has to be consulted directly.

    Hence: take the minimum over ALL evidence for the VIN -- every key's `Invoice date`, every key's
    `First Service Date`, and the earliest `Service_Date` in raw service history. Service history
    alone would be a regression: for 21.8% of VINs a genuine sale date legitimately precedes the
    first service, and that sale date is the better zero-point.

    Measured against the old rule, months from the derived due date to the actual turn-up
    (completers only, schedule model):

        milestone    median old -> new      p10 old -> new
        20k             -4.2 -> -3.4       -10.3 ->  -8.6
        30k             -6.5 -> -5.8       -14.1 -> -12.9
        40k             -9.3 -> -8.4       -22.8 -> -19.4
        60k            -15.1 -> -14.1      -30.7 -> -29.2

    Better at every milestone, and each VIN now resolves to one deterministic anchor.

    `df` must already have `Invoice date` / `First Service Date` parsed by date_utils.parse_dates,
    and `serv` must have a parsed `Service_Date`; nothing here re-parses. Returns a Series indexed
    by a UNIQUE VIN index -- callers reindex onto their own row order.
    """
    per_vin = df.groupby(vin_col)[["Invoice date", "First Service Date"]].min()
    hist = serv.groupby("Vin_No")["Service_Date"].min().rename("HistFirstService")
    cand = per_vin.join(hist, how="left")
    return cand.min(axis=1, skipna=True)


def mileage_date(vins, first_srv, serv_cut, milestone):
    """Last service date + (target mileage - last mileage) / daily mileage rate.

    Returns a Series indexed by `vins`. NaT where the vehicle is already at/past the target
    mileage as of the cutoff, has no usable rate (< 30 days of history, non-positive rate), or has
    no service history at all before the cutoff -- these are EXCLUDED, not collapsed onto the last
    service date (that collapse was the bug measured in create_test_cohort()).
    """
    target = milestone * 1000
    s = serv_cut.dropna(subset=["Mileage", "Service_Date"]).sort_values("Service_Date")
    last = s.groupby("Vin_No").tail(1).set_index("Vin_No")[["Service_Date", "Mileage"]]
    last = last.reindex(vins)
    fsv = first_srv.reindex(vins)

    # Build everything on plain float/NaN (never NaT/inf arithmetic) and only touch datetimes for
    # rows that survive every guard -- overflow-safe by construction rather than by cleanup after.
    days_since_start = (last["Service_Date"] - fsv).dt.days.astype(float)
    has_history = last["Mileage"].notna() & days_since_start.notna() & (days_since_start > 30)

    daily_rate = pd.Series(np.nan, index=vins)
    daily_rate[has_history] = last.loc[has_history, "Mileage"] / days_since_start[has_history]

    miles_remaining = target - last["Mileage"]
    include = (
        has_history
        & (miles_remaining > 0)
        & daily_rate.notna() & np.isfinite(daily_rate) & (daily_rate > 0)
    )
    days_remaining = pd.Series(np.nan, index=vins)
    days_remaining[include] = miles_remaining[include] / daily_rate[include]
    # Cap at 10 years out, same bound create_test_cohort() uses for its mileage projection -- a
    # rate-implied date further out than that is not a meaningful prediction, and without a cap the
    # addition below can push a real Timestamp past pandas' 2262-04-11 ceiling and overflow int64.
    include = include & days_remaining.notna() & np.isfinite(days_remaining) & (days_remaining.abs() <= 3650)

    predicted = pd.Series(pd.NaT, index=vins, dtype="datetime64[ns]")
    predicted[include] = last.loc[include, "Service_Date"] + pd.to_timedelta(days_remaining[include], unit="D")
    return predicted


def interval_date(vins, first_srv, serv_cut, milestone, min_days_sample=30, fallback_days=182.0):
    """Last completed lower-milestone date + remaining km / (per-vehicle km-per-block /
    population median days-per-block).

    Population median is measured on THIS cutoff's serv_cut only (no peeking), from whichever
    vehicles happen to have more than one dated record at the same lower-milestone Service_Num
    (e.g. a warranty comeback). Falls back to `fallback_days` (~6 months) when that sample is
    smaller than `min_days_sample` -- for a single prior milestone (e.g. 10k for a 20k due date)
    most vehicles contribute no diff at all, so the fallback is expected to fire often; see the
    bake-off report for how often.
    """
    target = milestone * 1000
    pms = serv_cut[(serv_cut["Service_Num"] > 0) & (serv_cut["Service_Num"] < milestone)].copy()
    pms = pms.dropna(subset=["Mileage", "Service_Date"]).sort_values(["Vin_No", "Service_Date"])

    if pms.empty:
        return pd.Series(pd.NaT, index=vins, dtype="datetime64[ns]"), np.nan, 0

    last_pms = pms.groupby("Vin_No").tail(1).set_index("Vin_No")[["Service_Date", "Mileage"]]

    pms["Mileage_Diff"] = pms.groupby("Vin_No")["Mileage"].diff()
    pms["Mileage_Diff"] = pms["Mileage_Diff"].fillna(pms["Mileage"])
    pms["Days_Diff"] = pms.groupby("Vin_No")["Service_Date"].diff().dt.days

    days_sample = pms["Days_Diff"].dropna()
    median_days = days_sample.median() if len(days_sample) >= min_days_sample else np.nan
    if pd.isna(median_days) or median_days <= 0:
        median_days = fallback_days

    avg_km = pms.groupby("Vin_No")["Mileage_Diff"].mean()
    pop_km_per_block = pms["Mileage_Diff"].median()
    if pd.isna(pop_km_per_block) or pop_km_per_block <= 0:
        pop_km_per_block = 10000.0

    last_pms = last_pms.reindex(vins)
    avg_km = avg_km.reindex(vins)
    fsv = first_srv.reindex(vins)

    anchor_date = last_pms["Service_Date"].fillna(fsv)
    anchor_mileage = last_pms["Mileage"].fillna(0)
    rate = (avg_km.fillna(pop_km_per_block) / median_days)

    remaining = target - anchor_mileage
    include = (
        (remaining > 0)
        & rate.notna() & np.isfinite(rate) & (rate > 0)
        & anchor_date.notna()
    )
    days_remaining = pd.Series(np.nan, index=vins)
    days_remaining[include] = remaining[include] / rate[include]
    # Same 10-year cap as mileage_date(), for the same overflow-safety reason.
    include = include & days_remaining.notna() & np.isfinite(days_remaining) & (days_remaining.abs() <= 3650)

    predicted = pd.Series(pd.NaT, index=vins, dtype="datetime64[ns]")
    predicted[include] = anchor_date[include] + pd.to_timedelta(days_remaining[include], unit="D")
    return predicted, median_days, len(days_sample)


def burn_rate_date(vins, serv_cut, milestone, cutoff, lo_fallback=False):
    """Cumulative pooled burn rate, service-1 to the highest completed milestone below the
    target, projected to the target mileage.

    `lo_fallback=True`: a vehicle with no Service_Num==1 record anchors `lo` on its earliest PMS
    visit instead (e.g. its 10k), so a missing first-service record no longer blanks the date.
    Still NaT when that earliest PMS visit is `hi` itself (no span to measure a rate over).

    History: this function originally used a fixed service-1 -> service-10 pair for every
    milestone; then (briefly, 2026-09-09) a milestone-relative single-interval pair
    (M-20 -> M-10); now (2026-09-09, same day, superseded per the user's explicit "not just the
    last milestone" instruction) this cumulative version. Rationale for each vehicle: pool every
    completed interval it actually has (1->10, 10->20, 20->30, ... up to whatever it has reached
    below the target) into ONE rate rather than reading only the most recent block. Concretely:
        lo = the vehicle's Service_Num==1 (first service) visit.
        hi = the vehicle's HIGHEST completed Service_Num strictly below `milestone` (whatever that
             happens to be per vehicle -- 10, 20, 30, ... -- not a fixed offset from `milestone`).
        burn = (mileage_hi - mileage_lo) / days(date_lo -> date_hi)
        due  = date_hi + (target - mileage_hi) / burn
    This `burn` is a total-km/total-days rate over the whole lo->hi span, which is mathematically
    identical (by telescoping sums) to a DAYS-WEIGHTED average of every individual interval rate
    the vehicle completed along the way -- there is no need to enumerate 1->10, 10->20, 20->30
    separately, reading the two endpoints gives the same number. Chosen over an unweighted average
    of interval rates (which would let a short interval count as much as a long one) per the user's
    explicit choice of a days-weighted/pooled rate, 2026-09-09.

    For milestone 20 this reduces to exactly the original 1->10 pair (nothing below 20 other than
    10 and 1 can exist, per the discrete PMS_SERVICE_NUMS steps), so 20k's due-date behavior is
    unchanged by this generalization -- verified in due_date_burn_rate_check.py.

    This is deliberately NOT mileage_date() (a lifetime rate: last mileage / days since first
    service -- doesn't stop at the milestone boundary below target) and NOT interval_date() (a
    population-median days-per-block with a fallback) -- it is a purely per-vehicle pooled rate,
    no population fallback and no fallback to a more distant/nearer pair.

    Guards, all required:
      - both anchors present (Service_Num == 1 and some Service_Num in (0, milestone), each with a
        date and mileage) -- a vehicle with NO completed milestone below the target at all (e.g. a
        brand-new car with only its first service) is EXCLUDED (NaT), not given a fabricated rate.
      - days(date_lo -> date_hi) > 0, km(mileage_lo -> mileage_hi) > 0, burn > 0
      - days-to-target clipped to [0, 3650]: a vehicle already at/past the target mileage at its
        `hi` visit is due IMMEDIATELY, not in the past, and an unclipped rate can overflow pandas'
        timestamp range on a near-zero burn.
      - date_hi <= cutoff: the anchor visit must already have happened as of the point in time this
        projection is being made from. `serv_cut` is expected to already be restricted to
        Service_Date <= cutoff by the caller (module convention); this is an explicit second check
        so a caller who passes a looser frame does not silently anchor a projection on post-cutoff
        data.

    Note this function only ever projects a FUTURE due date from the highest milestone the vehicle
    has completed BELOW the target -- it has no way to know, and does not check, whether the
    vehicle has since gone on to complete the target milestone itself via some other route (e.g. a
    duplicate/split vehicle record). Excluding already-completed vehicles from a forward-looking
    candidate list is a separate, deliberate post-filter applied by the caller, not this function's
    job (see prepare_test_set.py's stale-completion filter, added 2026-09-09 alongside this change).

    Returns a Series indexed by `vins`, NaT where any guard fails.
    """
    target = milestone * 1000
    cutoff_ts = pd.Timestamp(cutoff)

    s = serv_cut.dropna(subset=["Service_Date"]).sort_values("Service_Date", kind="mergesort")

    # lo: first (earliest-dated) Service_Num==1 visit per VIN.
    lo_rows = s[s["Service_Num"] == 1]
    g_lo = lo_rows.groupby("Vin_No").first()[["Service_Date", "Mileage"]]
    date_lo = g_lo["Service_Date"].reindex(vins)
    mil_lo = g_lo["Mileage"].reindex(vins)
    if lo_fallback:
        g_any = s[s["Service_Num"] > 0].groupby("Vin_No").first()[["Service_Date", "Mileage"]]
        no_lo = date_lo.isna()
        date_lo = date_lo.where(~no_lo, g_any["Service_Date"].reindex(vins))
        mil_lo = mil_lo.where(~no_lo, g_any["Mileage"].reindex(vins))

    # hi: among completed milestones strictly below the target (0 < Service_Num < milestone), the
    # HIGHEST Service_Num per VIN; ties on Service_Num broken by earliest Service_Date (stable sort
    # above keeps this deterministic).
    below = s[(s["Service_Num"] > 0) & (s["Service_Num"] < milestone)]
    if below.empty:
        date_hi = pd.Series(pd.NaT, index=vins, dtype="datetime64[ns]")
        mil_hi = pd.Series(np.nan, index=vins)
    else:
        max_sn = below.groupby("Vin_No")["Service_Num"].transform("max")
        hi_candidates = below[below["Service_Num"] == max_sn]
        g_hi = hi_candidates.groupby("Vin_No").first()[["Service_Date", "Mileage"]]
        date_hi = g_hi["Service_Date"].reindex(vins)
        mil_hi = g_hi["Mileage"].reindex(vins)

    have_both = date_lo.notna() & date_hi.notna() & mil_lo.notna() & mil_hi.notna()
    within_cutoff = date_hi.notna() & (date_hi <= cutoff_ts)

    days = (date_hi - date_lo).dt.days.astype("float64")
    km = (mil_hi - mil_lo).astype("float64")
    ok = (have_both & within_cutoff & (days > 0) & (km > 0)).fillna(False)

    burn = pd.Series(np.nan, index=vins)
    burn[ok] = km[ok] / days[ok]

    remaining = (target - mil_hi).astype("float64")
    include = ok & burn.notna() & np.isfinite(burn) & (burn > 0) & remaining.notna() & np.isfinite(remaining)

    days_remaining = pd.Series(np.nan, index=vins)
    days_remaining[include] = remaining[include] / burn[include]
    days_remaining[include] = days_remaining[include].clip(lower=0, upper=3650)
    include = include & days_remaining.notna() & np.isfinite(days_remaining)

    predicted = pd.Series(pd.NaT, index=vins, dtype="datetime64[ns]")
    predicted[include] = date_hi[include] + pd.to_timedelta(days_remaining[include], unit="D")
    return predicted


def last_completed_plus_months(vins, serv_cut, milestone, months=6):
    """Last completed milestone below the target + a flat number of months -- no rate, no history
    before that one anchor visit.

    Anchor: the vehicle's HIGHEST completed Service_Num strictly below `milestone`, same "hi"
    selection as `burn_rate_date()` (ties on Service_Num broken by earliest Service_Date, stable
    sort keeps it deterministic). due = date_hi + `months` calendar months.

    Why this exists alongside the other two forward-looking legs, not instead of them: `schedule`
    anchors on the vehicle's very FIRST service and assumes a flat 6-months-per-10k pace the whole
    way through, so any drift from that pace compounds over every milestone since -- a vehicle
    that's fallen behind looks "already overdue" long before it realistically is.  `burn_rate_date`
    fixes that by computing a real rate, but pools it over the vehicle's ENTIRE history from
    service-1 to `hi`, so an early fast (or slow) stretch still drags the projection the wrong way
    even if the vehicle's recent pace has changed. This leg does neither: it only has to be right
    about the NEXT interval, and resets the clock at whatever the vehicle's most recent actual visit
    was.

    Traced on VIN JN8AY2NY9R9499028 (2026-10-08): `schedule` said 2025-07-08 (6 months too early --
    the vehicle hadn't even had its 10k yet), `burn_rate` said 2026-11-24 (11 months too late -- its
    early low-mileage stretch dragged the pooled rate down), this leg says 2026-01-22 -- 20 days from
    the real completion, 2026-01-02. Measured against the 444 (20k) / 460 (30k) Q1-2026 vehicles the
    existing `union(schedule, burn_rate)` rule missed entirely: on its own, this leg would additionally
    have placed 36% (20k) / 61% (30k) of them inside a +-45 day window -- see CLAUDE.md for the
    full numbers and for why it is NOT yet adopted (see `earliest_date`'s module docstring for status).

    Returns a Series indexed by `vins`, NaT where the vehicle has no completed milestone below the
    target at all (e.g. a brand-new vehicle with only its first service) -- same exclusion
    `burn_rate_date()` uses, not a fabricated rate.
    """
    s = serv_cut.dropna(subset=["Service_Date"]).sort_values("Service_Date", kind="mergesort")
    below = s[(s["Service_Num"] > 0) & (s["Service_Num"] < milestone)]
    if below.empty:
        return pd.Series(pd.NaT, index=vins, dtype="datetime64[ns]")
    max_sn = below.groupby("Vin_No")["Service_Num"].transform("max")
    hi_candidates = below[below["Service_Num"] == max_sn]
    date_hi = hi_candidates.groupby("Vin_No").first()["Service_Date"].reindex(vins)
    return date_hi + pd.DateOffset(months=months)


def avg_interval_date(vins, first_srv, serv_cut, milestone, include_first_srv_segment=True,
                      scale_by_steps=False):
    """Last completed milestone below the target + the vehicle's OWN average historical block
    length in months (not a flat 6, not a mileage rate) -- "how long has this particular vehicle
    actually been taking between services, on average".

    Breakpoints, in order: FirstSrvDate -> date(Service_Num==1) -> date(Service_Num==10) ->
    date(Service_Num==20) -> ... -> date_hi, where `hi` is the vehicle's HIGHEST completed
    Service_Num strictly below `milestone` (same "hi" as `burn_rate_date()` /
    `last_completed_plus_months()`). Each consecutive pair is one segment; a segment is counted only
    when BOTH its endpoints are on record and its length is positive -- a segment missing either
    breakpoint (e.g. no Service_Num==10 record) is dropped from both the sum and the count, so the
    average is taken over whatever segments the vehicle actually has, never padded or backfilled.
    due = date_hi + mean(available segment lengths in months).

    Distinct from `burn_rate_date()` (one pooled km/day rate over the whole lo->hi span -- a single
    number, weighted toward whichever segment covers more days) and from
    `last_completed_plus_months()` (a flat offset, same for every vehicle): this keeps each
    historical block as its own data point and averages their LENGTHS, in months, with no mileage
    involved at all -- immune to burn_rate's stop-start-driving mileage noise, and personalized
    per-vehicle unlike the flat offset.

    `scale_by_steps` (added 2026-10-21, default False to keep every prior measurement in this module
    and in CLAUDE.md reproducible): `hi` can sit arbitrarily far below `milestone` -- e.g. a vehicle
    whose only completed service is its 10k has hi=10 whether `milestone` is 20, 30, ... or 100 --
    but the single-shot `due = date_hi + avg_months` applies the SAME one-interval offset regardless
    of how many 10k-steps away the target actually is. Caught from a Q2-2026 candidate-list review:
    the SAME vehicle, hi=10, was listed for every milestone 20k through 100k with the IDENTICAL due
    date, because the offset never scaled with distance -- 95.0% of that list's 8,341 VINs were
    duplicated this way (7,923 VINs on 2+ milestone lists, all with copy-pasted dates, not genuinely
    staggered ones). `scale_by_steps=True` fixes it by multiplying the per-vehicle average by how
    many steps away the target is: `due = date_hi + avg_months * ((milestone - hi) / 10)`. Measured
    effect on that same cohort: cross-milestone duplication fell from 95.0% to 5.8% of listed VINs,
    and of the ones still appearing on 2+ lists, 100% now carry genuinely different (correctly
    staggered) due dates per milestone -- i.e. it keeps the small genuine population of fast-pace
    vehicles that really can come due for two consecutive milestones inside one quarter, while fixing
    the vastly larger population of vehicles the unscaled version duplicated nonsensically across
    every milestone above their current step.

    Returns a Series indexed by `vins`, NaT where the vehicle has no completed milestone below the
    target at all (no `hi`, same exclusion the other two "reset" legs use) or has zero usable
    segments (e.g. `hi` exists but neither FirstSrvDate->1 nor 1->10 has both endpoints dated).
    """
    below = serv_cut[(serv_cut["Service_Num"] > 0) & (serv_cut["Service_Num"] < milestone)]
    if below.empty:
        return pd.Series(pd.NaT, index=vins, dtype="datetime64[ns]")

    s = serv_cut.dropna(subset=["Service_Date"]).sort_values("Service_Date", kind="mergesort")

    def _date_for(sn):
        rows = s[s["Service_Num"] == sn]
        return rows.groupby("Vin_No")["Service_Date"].first().reindex(vins)

    hi_step = below.groupby("Vin_No")["Service_Num"].max().reindex(vins)
    max_sn = below.groupby("Vin_No")["Service_Num"].transform("max")
    hi_candidates = below[below["Service_Num"] == max_sn]
    date_hi = hi_candidates.groupby("Vin_No").first()["Service_Date"].reindex(vins)

    lower_milestones = list(range(10, milestone, 10))  # 10, 20, ... strictly below target
    # include_first_srv_segment=False drops FirstSrvDate -> service 1 (usually invoice -> first
    # check-up, a delivery gap rather than a service interval) and averages from service 1 onward.
    breakpoints = ([first_srv.reindex(vins)] if include_first_srv_segment else []) + \
        [_date_for(1)] + [_date_for(m) for m in lower_milestones]

    month_sum = pd.Series(0.0, index=vins)
    month_count = pd.Series(0, index=vins)
    DAYS_PER_MONTH = 30.4368  # 365.25 / 12, average calendar month
    for start, end in zip(breakpoints[:-1], breakpoints[1:]):
        days = (end - start).dt.days.astype(float)
        ok = start.notna() & end.notna() & (days > 0)
        month_sum[ok] = month_sum[ok] + days[ok] / DAYS_PER_MONTH
        month_count[ok] = month_count[ok] + 1

    have_avg = month_count > 0
    avg_months = pd.Series(np.nan, index=vins)
    avg_months[have_avg] = month_sum[have_avg] / month_count[have_avg]

    steps_away = pd.Series(1.0, index=vins)
    if scale_by_steps:
        steps_away = (milestone - hi_step) / 10.0

    include = date_hi.notna() & have_avg
    predicted = pd.Series(pd.NaT, index=vins, dtype="datetime64[ns]")
    offset_days = (avg_months[include] * DAYS_PER_MONTH * steps_away[include]).round()
    predicted[include] = date_hi[include] + pd.to_timedelta(offset_days, unit="D")
    return predicted


def earliest_date(*legs):
    """min(...) per vehicle across any number of due-date legs, NaT-safe -- 'whichever comes
    first'. Originally min(schedule, mileage); takes a variable number of legs since 2026-10-08 so a
    third (or later, a fourth) candidate can be added without a signature change at every call site."""
    combined = pd.concat([s.rename(i) for i, s in enumerate(legs)], axis=1)
    return combined.min(axis=1, skipna=True)
