"""Step 2: what-if filters on top of the baseline (D' + prev-step + burn-gap<=20 + visit<=365d).
Reads precision_levers_cars_2026.csv. Candidates come from the step-1 profile: buckets that read
low precision in BOTH Q1 and Q3. Recall is reported against D' + prev-step with NO filter, so the
2pp budget is cumulative (the baseline already spends up to 1.0pp). Run from repo root:
    ./venv/Scripts/python.exe q1_due_rule_experiment/precision_levers_filters.py
"""
import numpy as np
import pandas as pd

D = "q1_due_rule_experiment/"
c = pd.read_csv(D + "precision_levers_cars_2026.csv")
roi = pd.concat([pd.read_csv(D + f) for f in ["roi_check_2026.csv", "roi_check_2026_70k_80k.csv",
                                               "roi_check_2026_60k_90k_100k.csv"]])
roi = roi[roi["rule"] == "D' prev-step (burn, last, avg)"]
key = list(zip(roi["setup"], roi["milestone"].str[:-1].astype(int)))
actual = dict(zip(key, roi["actual"]))
d_caught = dict(zip(key, roi["caught"]))      # D' + prev-step, no filter

single = c["n_legs"] == 1
DROP = {
    "baseline": np.zeros(len(c), bool),
    "a: last PMS > 8 months ago": c["months_since_pms"] > 8,
    "b: rhythm < 1 at quarter end": c["rhythm_qend"] < 1,
    "c: one leg + only 1 visit in 365d": single & (c["visits_365d"] == 1),
    "d: burn-only, 2 steps behind": single & c["by_burn"] & (c["gap"] == 20),
    "e: one leg + car 5+ years old": single & (c["age_years"] >= 5),
    "f: one leg + leg date >30d from mid-quarter": single & (c["centre_days"] > 30),
}
DROP["a+b"] = DROP["a: last PMS > 8 months ago"] | DROP["b: rhythm < 1 at quarter end"]
DROP["a+b+c"] = DROP["a+b"] | DROP["c: one leg + only 1 visit in 365d"]
DROP["a+b+c+e"] = DROP["a+b+c"] | DROP["e: one leg + car 5+ years old"]

rows = []
for name, drop in DROP.items():
    k = c[~np.asarray(drop, bool)]
    g = k.groupby(["setup", "milestone"])["tp"].agg(["size", "sum"])
    for (s, m), r in g.iterrows():
        a = actual[(s, m)]
        rows.append({"filter": name, "setup": s, "milestone": m, "listed": r["size"], "overlap": r["sum"],
                     "recall_pct": round(100 * r["sum"] / a, 1),
                     "precision_pct": round(100 * r["sum"] / r["size"], 1),
                     "recall_loss_vs_Dprev_pp": round(100 * (r["sum"] - d_caught[(s, m)]) / a, 1)})
res = pd.DataFrame(rows)
res.to_csv(D + "precision_levers_filters_2026.csv", index=False)

for s in ["Q1 forward", "Q3 forward"]:
    t = res[res["setup"] == s]
    t = t.assign(cell=t["precision_pct"].map("{:.1f}".format) + " (" + t["recall_loss_vs_Dprev_pp"].map("{:+.1f}".format) + ")")
    print(f"\n===== {s}: precision % (recall change vs D'+prev-step, pp) =====")
    print(t.pivot(index="filter", columns="milestone", values="cell").reindex(list(DROP)).to_string())
base = res[res["filter"] == "baseline"].set_index(["setup", "milestone"])["precision_pct"]
res["prec_gain_vs_baseline"] = res.apply(lambda r: r["precision_pct"] - base[(r["setup"], r["milestone"])], axis=1)
print("\n", res.groupby("filter").agg(worst_cum_recall_loss=("recall_loss_vs_Dprev_pp", "min"),
                                     avg_prec_gain_vs_baseline=("prec_gain_vs_baseline", "mean"))
      .round(1).reindex(list(DROP)).to_string())
