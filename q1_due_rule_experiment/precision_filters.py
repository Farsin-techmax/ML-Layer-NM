"""Part 4 of the precision-drop diagnosis: what-if filters on the D' + prev-step list. Reads
precision_drop_cars_2026.csv (from precision_drop_diagnosis.py) and the roi_check CSVs for the
actual counts. Run from repo root:
    ./venv/Scripts/python.exe q1_due_rule_experiment/precision_filters.py

Filters (all use only information available at the list's cutoff):
  burn_gap<=20   burn leg only counts when the car is at most 2 steps behind (gap = m - last PMS <= 20);
                 cars 3+ steps behind are dropped unless another (prev-step) leg lists them
  visit<=365d    drop cars with no visit of any kind in the 365 days before the cutoff
  visit<=270d / visit<=180d   stricter versions
  EDA Active     keep only EDA 'Vehicle Service Status' == Active. Q3 ONLY: the EDA snapshot is dated
                 2026-06-29, so for Q1 it was measured AFTER the quarter (leaks the outcome)
Pass rule: recall loss <= 2pp at every milestone, in both quarters.
"""
import numpy as np
import pandas as pd

D = "q1_due_rule_experiment/"
cars = pd.read_csv(D + "precision_drop_cars_2026.csv")
roi = pd.concat([pd.read_csv(D + f) for f in ["roi_check_2026.csv", "roi_check_2026_70k_80k.csv",
                                               "roi_check_2026_60k_90k_100k.csv"]])
roi = roi[roi["rule"] == "D' prev-step (burn, last, avg)"]
actual = {(s, int(m[:-1])): a for s, m, a in zip(roi["setup"], roi["milestone"], roi["actual"])}

by_other = cars["by_last6"] | cars["by_avg"]
gap_ok = by_other | (cars["by_burn"] & (cars["gap"] <= 20))
FILTERS = {
    "D' + prev-step (base)": np.ones(len(cars), bool),
    "burn_gap<=20": gap_ok,
    "visit<=365d": cars["days_since_visit"] <= 365,
    "visit<=270d": cars["days_since_visit"] <= 270,
    "visit<=180d": cars["days_since_visit"] <= 180,
    "burn_gap<=20 + visit<=365d": gap_ok & (cars["days_since_visit"] <= 365),
    "burn_gap<=20 + visit<=270d": gap_ok & (cars["days_since_visit"] <= 270),
    "EDA Active only (Q3 only)": cars["status"] == "Active",
}

rows = []
for name, keep in FILTERS.items():
    k = cars[np.asarray(keep)]
    g = k.groupby(["setup", "milestone"])["tp"].agg(["size", "sum"]).reset_index()
    for _, r in g.iterrows():
        a = actual[(r["setup"], r["milestone"])]
        rows.append({"filter": name, "setup": r["setup"], "milestone": r["milestone"],
                     "actual": a, "listed": r["size"], "overlap": r["sum"],
                     "recall_pct": round(100 * r["sum"] / a, 1),
                     "precision_pct": round(100 * r["sum"] / r["size"], 1)})
res = pd.DataFrame(rows)
base = res[res["filter"] == "D' + prev-step (base)"].set_index(["setup", "milestone"])
res["d_recall_pp"] = res.apply(lambda r: round(r["recall_pct"] - base.loc[(r["setup"], r["milestone"]), "recall_pct"], 1), axis=1)
res["d_precision_pp"] = res.apply(lambda r: round(r["precision_pct"] - base.loc[(r["setup"], r["milestone"]), "precision_pct"], 1), axis=1)
res.to_csv(D + "precision_filters_2026.csv", index=False)

for setup in ["Q1 forward", "Q3 forward"]:
    t = res[res["setup"] == setup]
    print(f"\n===== {setup}: precision % (recall change pp) =====")
    t = t.assign(cell=t["precision_pct"].map("{:.1f}".format) + " (" + t["d_recall_pp"].map("{:+.1f}".format) + ")")
    print(t.pivot(index="filter", columns="milestone", values="cell").to_string())
summ = res.groupby("filter").agg(worst_recall_loss=("d_recall_pp", "min"),
                                 avg_precision_gain=("d_precision_pp", "mean")).round(1)
print("\n", summ.to_string())
