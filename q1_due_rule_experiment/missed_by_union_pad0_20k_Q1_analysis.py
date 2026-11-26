"""Companion analysis for missed_by_union_pad0_20k_Q1.csv -- digs into WHY each bucket was missed:
direction/size of the due-date miss for the "both legs dated" bucket, and service-history shape for
the "no burn-rate leg" bucket. Read-only, prints to stdout; feeds the stakeholder artifact.
"""
import pandas as pd

from build_unionstale_20k import load_base, compute_due_legs
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONE = 20
Q_START, Q_END = quarter_bounds(2026, 1)

df, serv = load_base()
due_sched, due_burn, _ = compute_due_legs(df, serv, MILESTONE, include_last_milestone_leg=False)

missed = pd.read_csv("q1_due_rule_experiment/missed_by_union_pad0_20k_Q1.csv")
missed["actual_20k_completion_date"] = pd.to_datetime(missed["actual_20k_completion_date"])
missed["due_schedule"] = pd.to_datetime(missed["due_schedule"])
missed["due_burnrate"] = pd.to_datetime(missed["due_burnrate"])

both = missed[missed["due_schedule"].notna() & missed["due_burnrate"].notna()].copy()
no_burn = missed[missed["due_burnrate"].isna()].copy()

print("=== bucket A: both legs dated, both missed the window ===")
print(f"n = {len(both)}")
for leg in ["due_schedule", "due_burnrate"]:
    early = (both[leg] < Q_START).sum()
    late = (both[leg] > Q_END).sum()
    print(f"  {leg}: {early} projected EARLY (before Q1), {late} projected LATE (after Q1)")
    # days from window
    dist = both[leg].apply(lambda d: (Q_START - d).days if d < Q_START else (d - Q_END).days)
    print(f"    days outside window: median {dist.median():.0f}, p25 {dist.quantile(.25):.0f}, "
          f"p75 {dist.quantile(.75):.0f}, max {dist.max():.0f}")

# closer leg per row
both["closer_leg_dist_days"] = both.apply(
    lambda r: min(
        (Q_START - r.due_schedule).days if r.due_schedule < Q_START else (r.due_schedule - Q_END).days,
        (Q_START - r.due_burnrate).days if r.due_burnrate < Q_START else (r.due_burnrate - Q_END).days,
    ), axis=1)
print(f"  closer-of-the-two-legs miss distance: median {both['closer_leg_dist_days'].median():.0f}d, "
      f"within 45d: {(both['closer_leg_dist_days'] <= 45).sum()} "
      f"({100*(both['closer_leg_dist_days'] <= 45).mean():.1f}%)")

print(f"\n  Number_of_Services: mean {both['Number_of_Services'].mean():.2f}, "
      f"median {both['Number_of_Services'].median():.0f}")

print("\n=== bucket B: no burn-rate leg (no completed milestone below 20k on record) ===")
print(f"n = {len(no_burn)}")
print(f"  Number_of_Services: mean {no_burn['Number_of_Services'].mean():.2f}, "
      f"median {no_burn['Number_of_Services'].median():.0f}, "
      f"value counts:\n{no_burn['Number_of_Services'].value_counts().sort_index().to_string()}")

# schedule leg direction for bucket B
early = (no_burn["due_schedule"] < Q_START).sum()
late = (no_burn["due_schedule"] > Q_END).sum()
print(f"  schedule leg: {early} EARLY, {late} LATE")
dist = no_burn["due_schedule"].apply(lambda d: (Q_START - d).days if d < Q_START else (d - Q_END).days)
print(f"    days outside window: median {dist.median():.0f}, p25 {dist.quantile(.25):.0f}, "
      f"p75 {dist.quantile(.75):.0f}, max {dist.max():.0f}")

# compare against the overlap population (caught by union pad0) for contrast
at_ms = serv[serv["Service_Num"] == MILESTONE]
date_ms = at_ms.sort_values("Service_Date").groupby("Vin_No")["Service_Date"].first()
actual = date_ms[(date_ms >= Q_START) & (date_ms <= Q_END)]
actual = actual[actual.index.isin(df["VIN"])]
caught_vins = [v for v in actual.index if v not in set(missed["VIN"])]
n_services_all = serv.groupby("Vin_No").size()
caught_n_services = n_services_all.reindex(caught_vins)
print(f"\n=== contrast: the 1,073 CAUGHT completers ===")
print(f"  Number_of_Services: mean {caught_n_services.mean():.2f}, median {caught_n_services.median():.0f}")

# how many of bucket B have ANY completed milestone at all (any Service_Num>0), vs truly only PMS-code-1
has_any_lower = serv[serv["Vin_No"].isin(no_burn["VIN"]) & (serv["Service_Num"] > 0) & (serv["Service_Num"] < 20)]
print(f"\n  of bucket B, VINs with ANY Service_Num in (0,20) on record: "
      f"{has_any_lower['Vin_No'].nunique()} / {len(no_burn)}")
