"""One-off: list real 20k Q1-2026 completers that the PLAIN union rule (schedule OR burn-rate,
NO +-45d padding) never listed -- i.e. the false negatives of the "union only (pad 0)" arm from
the all-methods comparison (q1_due_rule_experiment/all_methods_all_milestones_Q1_2026.csv: 1,946
actual, 1,073 overlap -> 873 missed). Reuses build_unionstale_20k.py's load_base()/compute_due_legs()
so the due-date math matches exactly. Adds a "Number_of_Services" column (lifetime service-history
row count per VIN) and a reason_missed explanation. Run from repo root.
"""
import pandas as pd

from build_unionstale_20k import load_base, compute_due_legs
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONE = 20
Q_START, Q_END = quarter_bounds(2026, 1)

df, serv = load_base()
due_sched, due_burn, _ = compute_due_legs(df, serv, MILESTONE, include_last_milestone_leg=False)

# actual Q1-2026 completers, strict quarter, within the EDA gate (matches the all-methods table)
at_ms = serv[serv["Service_Num"] == MILESTONE]
date_ms = at_ms.sort_values("Service_Date").groupby("Vin_No")["Service_Date"].first()
actual = date_ms[(date_ms >= Q_START) & (date_ms <= Q_END)]
actual = actual[actual.index.isin(df["VIN"])]
print(f"actual Q1 2026 20k completers (in gate): {len(actual)}")

# plain union, pad0: due_sched OR due_burn lands in the strict Q1 window
in_sched = due_sched.notna() & (due_sched >= Q_START) & (due_sched <= Q_END)
in_burn = due_burn.notna() & (due_burn >= Q_START) & (due_burn <= Q_END)
union_listed = set(due_sched.index[in_sched]) | set(due_burn.index[in_burn])
print(f"union (pad0) listed: {len(union_listed)}")

missed_vins = [v for v in actual.index if v not in union_listed]
print(f"missed (actual but not in union pad0 list): {len(missed_vins)}")

n_services = serv.groupby("Vin_No").size()

rows = []
for vin in missed_vins:
    sched = due_sched.get(vin)
    burn = due_burn.get(vin)
    if pd.isna(sched) and pd.isna(burn):
        reason = "both legs undated (no FirstSrvDate and/or no completed milestone below 20k)"
    elif pd.isna(burn):
        reason = "burn-rate leg undated (no completed milestone below 20k on record); schedule leg projected outside Q1"
    elif pd.isna(sched):
        reason = "schedule leg undated (no FirstSrvDate on record); burn-rate leg projected outside Q1"
    else:
        reason = "both legs dated but both projected outside the Q1 window"
    rows.append({
        "VIN": vin,
        "actual_20k_completion_date": actual[vin].date(),
        "due_schedule": sched.date() if pd.notna(sched) else None,
        "due_burnrate": burn.date() if pd.notna(burn) else None,
        "Number_of_Services": int(n_services.get(vin, 0)),
        "reason_missed": reason,
    })

out = pd.DataFrame(rows).sort_values("VIN")
out_path = "q1_due_rule_experiment/missed_by_union_pad0_20k_Q1.csv"
out.to_csv(out_path, index=False)
print(f"wrote {len(out)} rows -> {out_path}")
print(out["Number_of_Services"].describe())
print(out["reason_missed"].value_counts())
