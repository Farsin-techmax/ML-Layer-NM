"""Two questions from the user, both measured against the PRE-dedup lists (no prev-step gate):

1. Upper-arm mileage filter: a vehicle is only considered for milestone m if its last recorded
   service mileage (any service type, as of the 2026-03-31 cutoff) is UNDER (m*1000 + 10000) --
   e.g. under 110,000 km for the 100k list. How many rows does this remove, on its own, without
   the prev-step gate?

2. Why does the prev-step gate work and this filter (mostly) doesn't catch the same problem?
   Direct evidence: how many vehicles whose ONLY completed PMS service is a low step (10k, 20k, ...)
   got listed for a much higher milestone, and what is their actual mileage -- do they even come
   close to tripping the upper-arm filter?

Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe Q2/mileage_filter_analysis.py
"""
import pandas as pd

from build_q2_avg_interval import CUTOFF, MILESTONES, load

df, serv, sc = load("data/EDA_Q1-2026.csv")

# Last recorded service mileage per VIN as of the cutoff -- ANY service, PMS or non-PMS, the plain
# "what does the odometer say the last time this car was in" reading.
last_mileage = (sc.dropna(subset=["Mileage", "Service_Date"])
                   .sort_values("Service_Date")
                   .groupby("Vin_No")["Mileage"].last())

print("=" * 100)
print("1. UPPER-ARM FILTER: keep only if last service mileage < target + 10,000 km")
print("=" * 100)
rows = []
for m in MILESTONES:
    pre = pd.read_csv(f"Q2/pre_dedup/lists/avg_interval_{m}k_Q2_2026.csv")
    pre["last_mileage"] = pre["VIN"].map(last_mileage)
    cap = m * 1000 + 10000
    over_cap = pre["last_mileage"] >= cap
    no_mileage = pre["last_mileage"].isna()
    kept = pre[~over_cap.fillna(False)]
    rows.append({
        "milestone": f"{m}k", "cap_km": cap, "listed_before": len(pre),
        "removed_over_cap": int(over_cap.fillna(False).sum()),
        "no_mileage_record": int(no_mileage.sum()),
        "listed_after_upperarm": len(kept),
        "precision_before": round(100 * pre["CameInQ2"].mean(), 1),
        "precision_after_upperarm": round(100 * kept["CameInQ2"].mean(), 1) if len(kept) else None,
    })
out1 = pd.DataFrame(rows)
out1.to_csv("Q2/mileage_upperarm_filter_Q2_2026.csv", index=False)
print(out1.to_string(index=False))

print()
print("=" * 100)
print("2. RECORDED CASES: vehicles whose highest completed PMS step is way below the target")
print("   -- do they even approach the upper-arm cap?")
print("=" * 100)
rows2 = []
for m in MILESTONES:
    pre = pd.read_csv(f"Q2/pre_dedup/lists/avg_interval_{m}k_Q2_2026.csv")
    pre["last_mileage"] = pre["VIN"].map(last_mileage)
    # "way below": last completed step is <= m - 30 (at least 3 milestones short of the target)
    far = pre[pre["last_pms_step"] <= m - 30]
    if len(far) == 0:
        continue
    cap = m * 1000 + 10000
    rows2.append({
        "milestone": f"{m}k", "n_vehicles_3plus_steps_short": len(far),
        "pct_of_list": round(100 * len(far) / len(pre), 1),
        "median_last_mileage": far["last_mileage"].median(),
        "max_last_mileage": far["last_mileage"].max(),
        "cap_km": cap,
        "would_any_trip_upperarm_cap": int((far["last_mileage"] >= cap).sum()),
    })
out2 = pd.DataFrame(rows2)
out2.to_csv("Q2/mileage_lowstep_evidence_Q2_2026.csv", index=False)
print(out2.to_string(index=False))

# Headline example: vehicles whose ONLY completed step is the very first 10k, listed for 100k.
only10 = pd.read_csv("Q2/pre_dedup/lists/avg_interval_100k_Q2_2026.csv")
only10 = only10[only10["last_pms_step"] == 10].copy()
only10["last_mileage"] = only10["VIN"].map(last_mileage)
print(f"\nVehicles last serviced at 10k, listed for the 100k quarter: {len(only10):,}")
print(only10["last_mileage"].describe().to_string())
print(f"Cap for 100k is 110,000 km -- how many of these are anywhere near it? "
      f"{int((only10['last_mileage'] >= 50000).sum())} are even over 50,000 km.")

print()
print("=" * 100)
print("3. TWO-SIDED mileage band (an alternative to the step-count prev-step gate): keep only if")
print("   last service mileage is within +-10,000 km of the target, i.e. [target-10k, target+10k]")
print("=" * 100)
rows3 = []
for m in MILESTONES:
    pre = pd.read_csv(f"Q2/pre_dedup/lists/avg_interval_{m}k_Q2_2026.csv")
    pre["last_mileage"] = pre["VIN"].map(last_mileage)
    target = m * 1000
    lo, hi = target - 10000, target + 10000
    in_band = pre["last_mileage"].between(lo, hi)
    kept = pre[in_band.fillna(False)]
    rows3.append({
        "milestone": f"{m}k", "band_km": f"{lo}-{hi}", "listed_before": len(pre),
        "listed_after_band": len(kept), "removed": len(pre) - len(kept),
        "precision_before": round(100 * pre["CameInQ2"].mean(), 1),
        "precision_after_band": round(100 * kept["CameInQ2"].mean(), 1) if len(kept) else None,
        "recall_vs_before": round(100 * kept["CameInQ2"].sum() / max(pre["CameInQ2"].sum(), 1), 1),
    })
out3 = pd.DataFrame(rows3)
out3.to_csv("Q2/mileage_band_filter_Q2_2026.csv", index=False)
print(out3.to_string(index=False))

# Compare the mileage-band survivors to the prev-step-gate-equivalent (post-dedup) survivors --
# do the two rules agree on WHICH vehicles belong on each list?
print("\nOverlap between the mileage-band filter and the prev-step-gate dedup (post-dedup lists):")
for m in MILESTONES:
    pre = pd.read_csv(f"Q2/pre_dedup/lists/avg_interval_{m}k_Q2_2026.csv")
    pre["last_mileage"] = pre["VIN"].map(last_mileage)
    target = m * 1000
    band_vins = set(pre.loc[pre["last_mileage"].between(target - 10000, target + 10000), "VIN"])
    dedup_vins = set(pd.read_csv(f"Q2/lists/avg_interval_{m}k_Q2_2026.csv")["VIN"])
    inter = band_vins & dedup_vins
    print(f"  {m}k: band={len(band_vins):,} dedup={len(dedup_vins):,} "
          f"overlap={len(inter):,} ({100*len(inter)/max(len(dedup_vins),1):.1f}% of dedup's list)")
