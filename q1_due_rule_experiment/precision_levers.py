"""Step 1 of the next precision round: profile the list that survives the baseline filter.
Baseline = D' + prev-step (burn | last+6mo[prev] | avg[prev], stale-filtered) minus cars that are
burn-only AND 3+ steps behind, minus cars with no visit in the 365 days before the cutoff.
Forward-built exactly as roi_check.py / precision_drop_diagnosis.py; 20k-100k; Q1 and Q3 2026.
Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe q1_due_rule_experiment/precision_levers.py

Writes precision_levers_cars_2026.csv (one row per baseline-listed car with every trait below) and
precision_levers_profile_2026.csv (precision by trait bucket, milestone band, quarter).
Traits, all as of the cutoff (no peeking):
  legs           which legs list it; n_legs
  centre_days    smallest |listing-leg date - quarter midpoint|
  rhythm_qend    (quarter end - last PMS date) / own usual gap (avg-interval gap)
  months_since_pms, nonpms_since_pms, visits_365d, km_per_day, age_years
  EDA (static-ish): Franchise, New / Used, Contract_PurchaseStatus_Final, fleet size (VINs per
  Customer ID). Warranty / Service Contract Status are extract-time, so Q1 readings may leak.
"""
import numpy as np
import pandas as pd

from build_unionstale_20k import load_base
from date_utils import parse_dates
from due_date import burn_rate_date, last_completed_plus_months, avg_interval_date
from features import resolve_service_num
from pmstrainfeatureeng_refactored import quarter_bounds

MILESTONES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
OUT = "q1_due_rule_experiment/precision_levers"

df, serv = load_base()
vins = df["VIN"].values
gate = set(vins)
eda = df.drop_duplicates("VIN").set_index("VIN")
fs_all = eda["FirstSrvDate"].reindex(vins)
fleet = eda.groupby("Customer ID").size()
fleet_size = eda["Customer ID"].map(fleet)
serv["Mileage"] = serv["Mileage"].where(serv["Mileage"] <= 500_000)
data_end = serv["Service_Date"].max()

q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)

SETUPS = {"Q1 forward": dict(q=(2026, 1), cutoff=pd.Timestamp("2025-12-31")),
          "Q3 forward": dict(q=(2026, 3), cutoff=data_end)}


def in_q(due, qs, qe):
    return set(due.index[due.notna() & (due >= qs) & (due <= qe)])


cars = []
for setup, cfg in SETUPS.items():
    qs, qe = quarter_bounds(*cfg["q"])
    q_mid = qs + (qe - qs) / 2
    cutoff = cfg["cutoff"]
    sc = serv.loc[serv["Service_Date"] <= cutoff]
    sc4 = sc[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]
    fs = fs_all.where(fs_all <= cutoff)

    visits = sc.groupby(["Vin_No", "Service_Date"])["Service_Num"].max().reset_index()
    last_visit = visits.groupby("Vin_No")["Service_Date"].max().reindex(vins)
    visits_365 = visits[visits["Service_Date"] > cutoff - pd.Timedelta(days=365)].groupby("Vin_No").size().reindex(vins).fillna(0)
    odo = sc[sc["Mileage"] > 0].sort_values("Service_Date").groupby("Vin_No").last()
    age_at_odo = (odo["Service_Date"].reindex(vins) - fs).dt.days
    kmday = odo["Mileage"].reindex(vins) / age_at_odo.where(age_at_odo >= 30)

    for m in MILESTONES:
        burn = burn_rate_date(vins, sc4, m, cutoff)
        last = last_completed_plus_months(vins, sc4, m, months=6)
        avg = avg_interval_date(vins, fs, sc4, m)
        below = sc[(sc["Service_Num"] > 0) & (sc["Service_Num"] < m)]
        hi = below.groupby("Vin_No")["Service_Num"].max().reindex(vins)
        hi_date = (below.sort_values(["Service_Num", "Service_Date"]).groupby("Vin_No")["Service_Date"]
                   .last().reindex(vins))
        last_pms = below.groupby("Vin_No")["Service_Date"].max().reindex(vins)
        prev = (hi == m - 10).values
        b, lp, ap = in_q(burn, qs, qe), in_q(last.where(prev), qs, qe), in_q(avg.where(prev), qs, qe)
        done_before = set(serv.loc[(serv["Service_Num"] == m) & (serv["Service_Date"] < qs), "Vin_No"])
        if cfg["q"] == (2026, 3):
            w = q3[(q3["Service_Date"] >= qs) & (q3["Service_Date"] <= qe)]
            actual = set(w.loc[w["Service_Num"] == m, "Vin_No"])
        else:
            d = serv[serv["Service_Num"] == m].groupby("Vin_No")["Service_Date"].min()
            actual = set(d.index[(d >= qs) & (d <= qe)]) & gate

        L = sorted((b | lp | ap) - done_before)
        f = pd.DataFrame(index=pd.Index(L, name="VIN"))
        f["by_burn"], f["by_last6"], f["by_avg"] = f.index.isin(b), f.index.isin(lp), f.index.isin(ap)
        step = hi.reindex(L).fillna(-1).replace(1, 0)
        f["gap"] = np.where(step >= 0, m - step, np.nan)
        f["days_since_visit"] = (cutoff - last_visit.reindex(L)).dt.days
        # baseline filter
        keep = (f["by_last6"] | f["by_avg"] | (f["by_burn"] & (f["gap"] <= 20))) & (f["days_since_visit"] <= 365)
        f = f[keep].copy()
        V = f.index
        f["setup"], f["milestone"], f["tp"] = setup, m, V.isin(list(actual))
        f["n_legs"] = f[["by_burn", "by_last6", "by_avg"]].sum(axis=1)
        leg_dates = pd.DataFrame({"b": burn.reindex(V).where(f["by_burn"]),
                                  "l": last.reindex(V).where(f["by_last6"]),
                                  "a": avg.reindex(V).where(f["by_avg"])})
        f["centre_days"] = leg_dates.apply(lambda c: (c - q_mid).dt.days.abs()).min(axis=1)
        gap_days = (avg.reindex(V) - hi_date.reindex(V)).dt.days
        f["rhythm_qend"] = (qe - hi_date.reindex(V)).dt.days / gap_days.where(gap_days > 0)
        f["months_since_pms"] = (cutoff - last_pms.reindex(V)).dt.days / 30.44
        vv = visits[visits["Vin_No"].isin(V)].copy()
        vv["lp"] = vv["Vin_No"].map(last_pms)
        f["nonpms_since_pms"] = vv[(vv["Service_Num"] == 0) & (vv["lp"].isna() | (vv["Service_Date"] > vv["lp"]))].groupby("Vin_No").size().reindex(V).fillna(0)
        f["visits_365d"] = visits_365.reindex(V).values
        f["km_per_day"] = kmday.reindex(V).values
        f["age_years"] = ((cutoff - fs.reindex(V)).dt.days / 365.25).values
        for c in ["Franchise", "New / Used", "Contract_PurchaseStatus_Final", "Warranty Status",
                  "Service Contract Status"]:
            f[c] = eda[c].reindex(V).astype(str).values
        f["fleet_size"] = fleet_size.reindex(V).values
        cars.append(f.reset_index())
        print(f"{setup} {m}k: baseline listed {len(f):,}, TP {int(f['tp'].sum()):,}, "
              f"precision {100 * f['tp'].mean():.1f}%")

cars = pd.concat(cars, ignore_index=True)
cars.to_csv(f"{OUT}_cars_2026.csv", index=False)

cars["band"] = pd.cut(cars["milestone"], [0, 40, 70, 100], labels=["20-40k", "50-70k", "80-100k"]).astype(str)
B = {
    "n_legs": cars["n_legs"].astype(str),
    "legs": np.select([cars.by_burn & ~cars.by_last6 & ~cars.by_avg, ~cars.by_burn & cars.by_last6 & ~cars.by_avg,
                       ~cars.by_burn & ~cars.by_last6 & cars.by_avg], ["burn only", "last6 only", "avg only"], "2+ legs"),
    "gap": cars["gap"].map(lambda g: "no PMS" if pd.isna(g) else f"{int(g)}"),
    "centre_days": pd.cut(cars["centre_days"], [-1, 15, 30, 45, 1e9], labels=["<=15", "16-30", "31-45", ">45"]),
    "rhythm_qend": pd.cut(cars["rhythm_qend"], [-1e9, 0.75, 1.0, 1.25, 2, 1e9], labels=["<0.75", "0.75-1", "1-1.25", "1.25-2", ">2"]),
    "months_since_pms": pd.cut(cars["months_since_pms"], [-1, 4, 8, 12, 18, 1e9], labels=["<=4", "4-8", "8-12", "12-18", ">18"]),
    "nonpms_since_pms": cars["nonpms_since_pms"].clip(upper=3).astype(int).astype(str).replace("3", "3+"),
    "visits_365d": cars["visits_365d"].clip(upper=4).astype(int).astype(str).replace("4", "4+"),
    "km_per_day": pd.cut(cars["km_per_day"], [-1, 30, 50, 80, 1e9], labels=["<30", "30-50", "50-80", "80+"]),
    "age_years": pd.cut(cars["age_years"], [-1, 2, 3, 4, 5, 100], labels=["<2", "2-3", "3-4", "4-5", "5+"]),
    "fleet_size": pd.cut(cars["fleet_size"], [0, 1, 3, 10, 1e9], labels=["1", "2-3", "4-10", "11+"]),
    **{c: cars[c] for c in ["Franchise", "New / Used", "Contract_PurchaseStatus_Final", "Warranty Status",
                            "Service Contract Status"]},
}
prof = []
for trait, vals in B.items():
    t = cars.assign(bucket=pd.Series(vals, index=cars.index).astype(object).fillna("unknown").astype(str))
    g = t.groupby(["setup", "band", "bucket"])["tp"].agg(["size", "sum"]).reset_index()
    tot = t.groupby(["setup", "band"])["tp"].agg(["size", "sum"])
    g["trait"] = trait
    g["share_names_pct"] = (100 * g["size"] / g.set_index(["setup", "band"]).index.map(tot["size"])).round(1)
    g["share_tp_pct"] = (100 * g["sum"] / g.set_index(["setup", "band"]).index.map(tot["sum"])).round(1)
    g["precision_pct"] = (100 * g["sum"] / g["size"]).round(1)
    prof.append(g)
prof = pd.concat(prof).rename(columns={"size": "names", "sum": "tp"})
prof.to_csv(f"{OUT}_profile_2026.csv", index=False)
base_prec = cars.groupby(["setup", "band"])["tp"].mean().mul(100).round(1)
print("\nbaseline precision by band:\n", base_prec.unstack().to_string())
for trait in B:
    p = prof[prof["trait"] == trait].pivot_table(index="bucket", columns=["setup", "band"],
                                                  values=["precision_pct", "share_names_pct"])
    print(f"\n--- {trait} ---")
    print(p.to_string())
