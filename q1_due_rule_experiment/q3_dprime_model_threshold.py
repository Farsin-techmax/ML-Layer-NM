"""Score the Q3 2026 D' + prev-step lists with the PRODUCTION models and find the max-F1 threshold.
Run from repo root (slow: one feature build per milestone):
    PYTHONPATH=. ./venv/Scripts/python.exe q1_due_rule_experiment/q3_dprime_model_threshold.py

Q3 only: it is the one genuinely forward quarter (EDA snapshot and history end 2026-06-29, outcomes
come from Q3/Service_History_Q3-2026.csv). Q1 would read post-outcome EDA fields -- the documented
Q1/Q2 inflation -- so it is not used to pick a threshold.

Per milestone 20k-100k:
  1. D' + prev-step list (burn | last+6mo[prev] | avg[prev]), stale-filtered -- as roi_check.py.
  2. Cohort = EDA rows of those VINs, Expected{m}kDate = earliest in-quarter listing-leg date,
     TargetFlag = 0 for every row (the forward / outcome-unknown feature path), feature cutoff
     2026-06-30 -- same as the delivered Q3 lists. process_service_data(), then the production
     model (production_models/{m}k) via build_inference_matrix + predict_proba.
  3. Real outcome = distinct VIN with Service_Num == m in Q3.
  4. Threshold sweep 0.01..0.99: F1 / precision / recall on the listed cars. In-sample best, plus an
     honest 2-fold split by VIN (pick on one half, score the other, both ways, averaged).
     Reference rows: shipped 0.5, and "flag everyone".
Also reported for the baseline-filtered list (burn-gap <= 20 + visit <= 365d) from
precision_levers_cars_2026.csv.
Writes q3_dprime_scored_{m}k.csv per milestone and q3_dprime_threshold_2026_{tag}.csv.
Pass milestones to limit the run, e.g. `... q3_dprime_model_threshold.py 70 80`.
"""
import sys

import numpy as np
import pandas as pd

from date_utils import parse_dates
from due_date import burn_rate_date, last_completed_plus_months, avg_interval_date
from features import resolve_service_num
from pms_model import load_artifacts, build_inference_matrix, predict_proba, split_missing
from pmstrainfeatureeng_refactored import (process_service_data, quarter_bounds, extract_k1,
                                           _apply_first_srv_date)

MILESTONES = [int(x) for x in sys.argv[1:]] or [20, 30, 40, 50, 60, 70, 80, 90, 100]
TAG = "_".join(f"{m}k" for m in MILESTONES)
D = "q1_due_rule_experiment/"
EDA, SERV = "data/EDA_Q2-2026.csv", "data/Service_History_Q2-2026.csv"
FEATURE_CUTOFF = "2026-06-30"
QS, QE = quarter_bounds(2026, 3)
THR = np.round(np.arange(0.01, 1.00, 0.01), 2)

serv = pd.read_csv(SERV, low_memory=False, encoding="ISO-8859-1")
serv["Service_Date"] = parse_dates(serv["Service_Date"])
serv["Mileage"] = pd.to_numeric(serv["Mileage"], errors="coerce")
serv = serv.dropna(subset=["Service_Date"])
serv["Service_Num"] = resolve_service_num(serv)
df = pd.read_csv(EDA, low_memory=False, encoding="ISO-8859-1")
df = df.query("`Last Service - PMS` != '-'").copy()
for c in ["Invoice date", "First Service Date", "Last Service Date"]:
    df[c] = parse_dates(df[c])
df = _apply_first_srv_date(df, serv)
vins = df["VIN"].values
fs = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
cutoff = serv["Service_Date"].max()
sc4 = serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]

q3 = pd.read_csv("Q3/Service_History_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
q3["Service_Date"] = parse_dates(q3["Service_Date"])
q3["Service_Num"] = resolve_service_num(q3)
q3 = q3[(q3["Service_Date"] >= QS) & (q3["Service_Date"] <= QE)]
base_cars = pd.read_csv(D + "precision_levers_cars_2026.csv")
base_cars = base_cars[base_cars["setup"] == "Q3 forward"]


def f1_at(y, p, t):
    flag = p >= t
    tp = int((flag & (y == 1)).sum())
    prec = tp / flag.sum() if flag.sum() else 0.0
    rec = tp / y.sum() if y.sum() else 0.0
    return (2 * prec * rec / (prec + rec) if prec + rec else 0.0), prec, rec, int(flag.sum())


def best_thr(y, p):
    f = [f1_at(y, p, t)[0] for t in THR]
    return THR[int(np.argmax(f))]


def sweep(y, p, vin):
    out = {}
    t_in = best_thr(y, p)
    out["best_thr_in_sample"] = t_in
    out["f1_in_sample"], out["prec_in_sample"], out["rec_in_sample"], out["flagged_in_sample"] = f1_at(y, p, t_in)
    rng = np.random.default_rng(42)
    half = rng.permutation(len(y)) % 2 == 0
    f_hold, picks = [], []
    for a, b in [(half, ~half), (~half, half)]:
        t = best_thr(y[a], p[a])
        picks.append(t)
        f_hold.append(f1_at(y[b], p[b], t)[0])
    out["thr_picked_on_halves"] = "/".join(f"{t:.2f}" for t in picks)
    out["f1_holdout"] = float(np.mean(f_hold))
    out["f1_at_0.5"], out["prec_at_0.5"], out["rec_at_0.5"], out["flagged_at_0.5"] = f1_at(y, p, 0.5)
    out["f1_flag_everyone"] = f1_at(y, p, 0.0)[0]
    return out


rows = []
for m in MILESTONES:
    burn = burn_rate_date(vins, sc4, m, cutoff)
    last = last_completed_plus_months(vins, sc4, m, months=6)
    avg = avg_interval_date(vins, fs, sc4, m)
    below = serv[(serv["Service_Num"] > 0) & (serv["Service_Num"] < m)]
    prev = (below.groupby("Vin_No")["Service_Num"].max().reindex(vins) == m - 10).values
    legs = pd.DataFrame({"b": burn, "l": last.where(prev), "a": avg.where(prev)}).reindex(vins)
    inq = legs.apply(lambda c: c.where((c >= QS) & (c <= QE)))
    done_before = set(serv.loc[(serv["Service_Num"] == m) & (serv["Service_Date"] < QS), "Vin_No"])
    listed = inq.notna().any(axis=1)
    listed = listed[listed].index.difference(list(done_before))

    cohort = df[df["VIN"].isin(listed)].copy()
    cohort[f"Expected{m}kDate"] = inq.min(axis=1).reindex(cohort["VIN"]).values
    cohort["TargetFlag"] = 0
    cohort["Service_Num"] = cohort["Last Service - PMS"].apply(extract_k1)
    print(f"\n=== {m}k: D' + prev-step list {len(cohort):,} cars -> building features ===")
    feats = process_service_data(mastersheet=cohort, servhistory=SERV,
                                 rfm="data/RFM_Q2-2026.csv", servcode="data/Service_Code_Desc.csv",
                                 appointdf="data/Appointments_Q2-2026.csv",
                                 digidf="data/Digital_Sessions_Q2-2026.csv",
                                 vhc="data/VHC_Q2-2026.csv", filter_date=FEATURE_CUTOFF,
                                 last_service_code=m, is_test=True)
    mdir = f"production_models/{m}k"
    model, imputer, scaler, sel = load_artifacts(m, mdir=mdir, features_file=f"{mdir}/selected_features.json")
    X, missing = build_inference_matrix(feats.copy(), m, sel)
    _, gaps = split_missing(missing, feats.columns)
    p = predict_proba(model, imputer, scaler, X)
    came = set(q3.loc[q3["Service_Num"] == m, "Vin_No"])
    y = feats["VIN"].isin(came).astype(int).values
    pd.DataFrame({"VIN": feats["VIN"].values, "prob": p, "came_in": y}).to_csv(
        D + f"q3_dprime_scored_{m}k.csv", index=False)
    print(f"{m}k scored {len(y):,} rows (list {len(cohort):,}), real feature gaps {len(gaps)}: {gaps[:8]}")

    r = {"milestone": f"{m}k", "list": "D' + prev-step", "rows": len(y), "came_in": int(y.sum()),
         "base_rate_pct": round(100 * y.mean(), 1), "feature_gaps": len(gaps), **sweep(y, p, feats["VIN"].values)}
    rows.append(r)
    keep = feats["VIN"].isin(set(base_cars.loc[base_cars["milestone"] == m, "VIN"])).values
    rows.append({"milestone": f"{m}k", "list": "+ baseline filter", "rows": int(keep.sum()),
                 "came_in": int(y[keep].sum()), "base_rate_pct": round(100 * y[keep].mean(), 1),
                 "feature_gaps": len(gaps), **sweep(y[keep], p[keep], feats["VIN"].values[keep])})
    print(pd.DataFrame(rows[-2:]).to_string(index=False))

out = pd.DataFrame(rows)
for c in out.columns:
    if out[c].dtype == float:
        out[c] = out[c].round(3)
out.to_csv(D + f"q3_dprime_threshold_2026_{TAG}.csv", index=False)
print("\n", out.to_string(index=False))
