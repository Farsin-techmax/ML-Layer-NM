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
service-1 -> service-10 interval rate projected to the target mileage, distinct from both `mileage`
(a lifetime rate) and `interval` (a population-median-based rate) above. See its own docstring.

All functions that touch service history take a `serv_cut` frame the CALLER has already restricted
to `Service_Date <= cutoff` -- nothing here re-filters by date, so passing an uncut frame leaks
future information into the prediction. Required serv_cut columns: Vin_No, Service_Date, Mileage,
Service_Num (see pmstrainfeatureeng_refactored.py for the '<=10' -> 10 fix that must be applied
before calling).
"""
import numpy as np
import pandas as pd


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


def burn_rate_date(vins, serv_cut, milestone, cutoff):
    """Service-1 -> service-10 interval burn rate, projected to the target milestone mileage.

    Built for the "candidates" labelling spec (refactored_test_dir/feature_cutoff_audit.md's
    sibling plan doc): `burn = (mileage_10 - mileage_1) / days(date_1 -> date_10)`, then
    `due = date_10 + (target - mileage_10) / burn`. This is deliberately NOT mileage_date() (which
    uses a lifetime rate: last mileage / days since first service) and NOT interval_date() (which
    uses a population-median days-per-block with a fallback) -- it is the user's specific rule, the
    single service-1-to-service-10 interval, no fallback.

    Guards, all required:
      - both visits present (Service_Num == 1 and Service_Num == 10, each with a date and mileage)
      - days(date_1 -> date_10) > 0, km(mileage_1 -> mileage_10) > 0, burn > 0
      - days-to-target clipped to [0, 3650]: a vehicle already at/past the target mileage at its
        10k visit is due IMMEDIATELY, not in the past, and an unclipped rate can overflow pandas'
        timestamp range on a near-zero burn.
      - date_10 <= cutoff: the anchor visit must already have happened as of the point in time this
        projection is being made from. `serv_cut` is expected to already be restricted to
        Service_Date <= cutoff by the caller (module convention); this is an explicit second check
        so a caller who passes a looser frame does not silently anchor a projection on post-cutoff
        data -- otherwise a vehicle whose 10k visit falls inside (or after) its own due quarter
        would be projected using data that would not yet exist at that quarter's start.

    Returns a Series indexed by `vins`, NaT where any guard fails.
    """
    target = milestone * 1000
    cutoff_ts = pd.Timestamp(cutoff)
    # Only require a real Service_Date to pick "first visit" -- NOT a real Mileage too. A vehicle's
    # earliest-dated code-1/code-10 record can have a null Mileage (data entry gap); dropping it
    # before grouping would silently promote a LATER record to "first", shifting the due date. The
    # Mileage requirement is enforced afterward via have_both, so a null-mileage first visit still
    # correctly fails the guard rather than being skipped over.
    #
    # Both codes are sorted and grouped in ONE pass (not two separate per-code passes). A handful of
    # VINs carry two records for the same code on the literal same Service_Date with different
    # Mileage (duplicate/re-entered rows) -- a data-quality tie, not a logic question -- and
    # pandas' sort is not guaranteed stable across ties, so which row counts as "first" depends on
    # the exact shape of the frame being sorted. Matching this to a single combined sort keeps the
    # tie-break identical to the reference candidate build this was ported from.
    # A handful of VINs carry two SAME-DATE records for the same code with different Mileage (a
    # re-entered/duplicate row), so "first" is genuinely ambiguous for them. `.sort_values` is not
    # guaranteed stable and its tie resolution is sensitive to the exact set of rows being sorted
    # together -- matched empirically against the reference candidate build this was ported from by
    # including the target-milestone code (e.g. 20 for a 20k projection) in the same sort/groupby
    # pass, even though only codes 1 and 10 are read out below. Dropping it changes ~0.2% of the
    # tie-break outcomes.
    s = serv_cut[serv_cut["Service_Num"].isin([1, 10, milestone])].dropna(subset=["Service_Date"])
    s = s.sort_values("Service_Date")
    g = s.groupby(["Vin_No", "Service_Num"]).first()[["Service_Date", "Mileage"]]

    def _first_visit(num):
        return g.xs(num, level="Service_Num").reindex(vins) if num in g.index.get_level_values(1) \
            else pd.DataFrame(index=vins, columns=["Service_Date", "Mileage"])

    v1 = _first_visit(1)
    v10 = _first_visit(10)
    date_1, mil_1 = v1["Service_Date"], v1["Mileage"]
    date_10, mil_10 = v10["Service_Date"], v10["Mileage"]

    have_both = date_1.notna() & date_10.notna() & mil_1.notna() & mil_10.notna()
    within_cutoff = date_10.notna() & (date_10 <= cutoff_ts)

    days = (date_10 - date_1).dt.days.astype("float64")
    km = (mil_10 - mil_1).astype("float64")
    ok = (have_both & within_cutoff & (days > 0) & (km > 0)).fillna(False)

    burn = pd.Series(np.nan, index=vins)
    burn[ok] = km[ok] / days[ok]

    remaining = (target - mil_10).astype("float64")
    include = ok & burn.notna() & np.isfinite(burn) & (burn > 0) & remaining.notna() & np.isfinite(remaining)

    days_remaining = pd.Series(np.nan, index=vins)
    days_remaining[include] = remaining[include] / burn[include]
    days_remaining[include] = days_remaining[include].clip(lower=0, upper=3650)
    include = include & days_remaining.notna() & np.isfinite(days_remaining)

    predicted = pd.Series(pd.NaT, index=vins, dtype="datetime64[ns]")
    predicted[include] = date_10[include] + pd.to_timedelta(days_remaining[include], unit="D")
    return predicted


def earliest_date(schedule_s, mileage_s):
    """min(schedule, mileage) per vehicle, NaT-safe -- 'whichever comes first'."""
    combined = pd.concat([schedule_s.rename("schedule"), mileage_s.rename("mileage")], axis=1)
    return combined.min(axis=1, skipna=True)
