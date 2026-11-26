"""20k: which features tell a listed customer will NOT come in (push no-shows down the ranking)?

Read-only diagnostic -- no pipeline file, model or test set is changed. Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe noshow_feature_review/noshow_features_20k.py

Populations (TargetFlag = came in for 20k):
  train    final_processed_20k_candidates.csv, Next20K_Due <= 2025-12-31 (same rows retrain.py uses);
           feature cutoff = start of each row's CohortQuarter (as process_service_data was called)
  q1, q2   test_20k_Q{1,2}_2026_cand.csv, NOT-early rows only, cutoff 2025-12-31
  q3       test_20k_Q3_2026_REAL.csv -- the delivered Q3 list with real Q3 outcomes, cutoff 2026-06-30

Part 1  single-feature AUC of the 160 production features on each population, plus AUC inside the
        rows the production model flagged (prob >= 0.5) on q3 -- the no-shows we want pushed down.
        AUC is RAW: below 0.5 = higher value means MORE likely a no-show.
Part 2  candidate new "no-show" features from raw service history, each cut at the row's own
        feature cutoff (no peeking): where the four due-date legs sit vs the quarter, how many land in
        it, recency, own-rhythm overdue ratio, projected km, non-PMS visits since last PMS, keys, gaps.
Part 3  would they help if added to feature engineering? Same learner (HistGradientBoosting, 3 seeds)
        trained on the training rows with the 160 production features vs 160 + new, scored on q1/q2/q3.
        Proxy for a retrain -- the production model is a neural net; this isolates the features.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from build_unionstale_20k import load_base
from date_utils import parse_dates
from due_date import burn_rate_date, last_completed_plus_months, avg_interval_date
from features import expected_milestone_months
from pms_model import load_artifacts, build_inference_matrix, predict_proba
from pmstrainfeatureeng_refactored import quarter_bounds

M = 20
OUT = "noshow_feature_review"
MDIR = "production_models/20k"
SEEDS = [0, 1, 2]

model, imputer, scaler, FEATS = load_artifacts(M, mdir=MDIR,
                                               features_file=f"{MDIR}/selected_features.json")
eda, serv = load_base()
first_srv_all = eda.drop_duplicates("VIN").set_index("VIN")["FirstSrvDate"]

# ---- populations ---------------------------------------------------------------------------
tr = pd.read_csv("refactored_test_dir/final_processed_20k_candidates.csv", low_memory=False)
tr = tr[parse_dates(tr[f"Next{M}K_Due"]) <= pd.Timestamp("2025-12-31")].reset_index(drop=True)
q1 = pd.read_csv(f"test_sets/test_{M}k_Q1_2026_cand.csv", low_memory=False)
q2 = pd.read_csv(f"test_sets/test_{M}k_Q2_2026_cand.csv", low_memory=False)
q1 = q1[q1["EarlyCompleter"] == 0].reset_index(drop=True)
q2 = q2[q2["EarlyCompleter"] == 0].reset_index(drop=True)
q3 = pd.read_csv(f"test_sets/test_{M}k_Q3_2026_REAL.csv", low_memory=False)
POPS = {"train": tr, "q1": q1, "q2": q2, "q3": q3}
TEST_WIN = {"q1": (2026, 1, "2025-12-31"), "q2": (2026, 2, "2025-12-31"),
            "q3": (2026, 3, "2026-06-30")}
for k, d in POPS.items():
    print(f"{k}: {len(d):,} rows, came in {d['TargetFlag'].mean():.1%}")

all_vins = set().union(*[set(d["VIN"]) for d in POPS.values()])
serv_small = serv[serv["Vin_No"].isin(all_vins)][
    ["Vin_No", "Vehicle_Key", "Service_Date", "Mileage", "Service_Num"]].copy()
serv_small["Mileage"] = serv_small["Mileage"].where(serv_small["Mileage"] <= 500_000)


# ---- part 2 feature builder ----------------------------------------------------------------
def new_features(vins, cutoff, q_start, q_end):
    cutoff = pd.Timestamp(cutoff)
    uv = pd.unique(vins)
    sc = serv_small[serv_small["Vin_No"].isin(uv) & (serv_small["Service_Date"] <= cutoff)]
    sc4 = sc[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]
    fs = first_srv_all.reindex(uv)
    fs = fs.where(fs <= cutoff)
    q_mid = q_start + (q_end - q_start) / 2

    legs = {
        "sched": fs + pd.DateOffset(months=expected_milestone_months(M, "schedule")),
        "burn": burn_rate_date(uv, sc4, M, cutoff),
        "last6": last_completed_plus_months(uv, sc4, M, months=6),
        "avg": avg_interval_date(uv, fs, sc4, M),
    }
    L = pd.DataFrame({k: pd.to_datetime(pd.Series(v).reindex(uv)).dt.normalize().values
                      for k, v in legs.items()}, index=uv)
    F = pd.DataFrame(index=uv)
    F["n_legs_in_q"] = L.apply(lambda c: c.between(q_start, q_end)).sum(axis=1)
    F["n_legs_avail"] = L.notna().sum(axis=1)
    off = L.apply(lambda c: (c - q_mid).dt.days)
    for k in legs:
        F[f"{k}_days_from_qmid"] = off[k]
    F["legs_median_days_from_qmid"] = off.median(axis=1)
    F["legs_spread_days"] = (L.max(axis=1) - L.min(axis=1)).dt.days

    last_any = sc.groupby("Vin_No")["Service_Date"].max().reindex(uv)
    F["days_since_any_visit"] = (cutoff - last_any).dt.days
    pms = sc[(sc["Service_Num"] > 0) & (sc["Service_Num"] < M)]
    last_pms = pms.groupby("Vin_No")["Service_Date"].max().reindex(uv)
    F["days_since_last_pms"] = (cutoff - last_pms).dt.days
    hi = (pms.sort_values(["Service_Num", "Service_Date"]).groupby("Vin_No")["Service_Date"]
          .last().reindex(uv))
    gap = (L["avg"] - hi).dt.days
    gap = gap.where(gap > 0)
    F["own_gap_days"] = gap
    F["rhythm_ratio_at_cutoff"] = (cutoff - hi).dt.days / gap
    F["rhythm_ratio_at_qend"] = (q_end - hi).dt.days / gap

    mk = sc[sc["Mileage"] > 0].sort_values("Service_Date").groupby("Vin_No").last()
    last_km = mk["Mileage"].reindex(uv)
    last_km_date = mk["Service_Date"].reindex(uv)
    life = (last_km_date - fs).dt.days
    rate = last_km / life.where(life >= 30)
    F["km_per_day_life"] = rate
    F["proj_km_at_qend"] = last_km + rate * (q_end - last_km_date).dt.days
    F["km_short_at_qend"] = M * 1000 - F["proj_km_at_qend"]
    F["last_km_minus_target"] = last_km - M * 1000

    vis = sc.groupby(["Vin_No", "Service_Date"])["Service_Num"].max().reset_index()
    vis["lp"] = vis["Vin_No"].map(last_pms)
    after = (vis["Service_Num"] == 0) & (vis["lp"].isna() | (vis["Service_Date"] > vis["lp"]))
    F["nonpms_visits_since_last_pms"] = vis[after].groupby("Vin_No").size().reindex(uv).fillna(0)
    recent = vis["Service_Date"] > cutoff - pd.Timedelta(days=365)
    F["visits_last_365d"] = vis[recent].groupby("Vin_No").size().reindex(uv).fillna(0)
    F["n_keys"] = sc.groupby("Vin_No")["Vehicle_Key"].nunique().reindex(uv).fillna(0)
    steps = pms.groupby("Vin_No")["Service_Num"].apply(
        lambda s: len(set(s.astype(int)) & set(range(10, M, 10))))
    F["missing_10k_steps"] = (M // 10 - 1) - steps.reindex(uv).fillna(0)
    return F.reindex(vins).reset_index(drop=True)


NEW = {}
parts = []
for cq, sub in tr.groupby("CohortQuarter"):
    qs, qe = quarter_bounds(int(cq[:4]), int(cq[5]))
    f = new_features(sub["VIN"].values, qs, qs, qe)
    f.index = sub.index
    parts.append(f)
NEW["train"] = pd.concat(parts).sort_index()
for k, (y, q, cut) in TEST_WIN.items():
    qs, qe = quarter_bounds(y, q)
    NEW[k] = new_features(POPS[k]["VIN"].values, cut, qs, qe)
NEW_COLS = list(NEW["q3"].columns)

# ---- production model scores + existing feature matrices -----------------------------------
X, PROB = {}, {}
for k, d in POPS.items():
    X[k], _ = build_inference_matrix(d.copy(), M, FEATS)
    PROB[k] = predict_proba(model, imputer, scaler, X[k])
Y = {k: d["TargetFlag"].astype(int).values for k, d in POPS.items()}
flag3 = PROB["q3"] >= 0.5
print("production NN AUC:", {k: round(roc_auc_score(Y[k], PROB[k]), 4) for k in POPS})
print(f"q3 flagged at 0.5: {flag3.sum()} rows, {Y['q3'][flag3].mean():.1%} came in")


def auc(y, x):
    x = pd.to_numeric(pd.Series(np.asarray(x)), errors="coerce")
    ok = x.notna().values
    yy = np.asarray(y)[ok]
    if ok.sum() < 50 or len(set(yy)) < 2 or x[ok].nunique() < 2:
        return np.nan
    return round(roc_auc_score(yy, x[ok]), 4)


def auc_table(cols, getcol):
    rows = []
    for c in cols:
        r = {"feature": c}
        for k in POPS:
            r[f"auc_{k}"] = auc(Y[k], getcol(k, c))
        r["auc_q3_flagged"] = auc(Y["q3"][flag3], np.asarray(getcol("q3", c))[flag3])
        r["coverage_q3"] = round(float(pd.Series(np.asarray(getcol("q3", c))).notna().mean()), 3)
        a, b = r["auc_train"], r["auc_q3"]
        r["flips_train_vs_q3"] = bool(pd.notna(a) and pd.notna(b) and (a - .5) * (b - .5) < 0
                                      and abs(a - .5) > .03 and abs(b - .5) > .03)
        rows.append(r)
    t = pd.DataFrame(rows)
    t["q3_strength"] = (t["auc_q3"] - 0.5).abs()
    return t.sort_values("q3_strength", ascending=False)


t_old = auc_table(FEATS, lambda k, c: X[k][c].values)
t_new = auc_table(NEW_COLS, lambda k, c: NEW[k][c].values)
t_old.to_csv(f"{OUT}/existing_feature_auc_20k.csv", index=False)
t_new.to_csv(f"{OUT}/new_feature_auc_20k.csv", index=False)
show = ["feature", "auc_train", "auc_q1", "auc_q2", "auc_q3", "auc_q3_flagged", "flips_train_vs_q3"]
print("\nEXISTING features, strongest on q3:")
print(t_old[show].head(25).to_string(index=False))
print(f"\nexisting features that flip direction train vs q3: {int(t_old['flips_train_vs_q3'].sum())}")
print(t_old[t_old["flips_train_vs_q3"]][show].to_string(index=False))
print("\nNEW candidate features:")
print(t_new[show + ["coverage_q3"]].to_string(index=False))

# save everything part 3 / the follow-up A/B needs, so it can be re-run without rebuilding
pd.to_pickle({"X": X, "NEW": NEW, "PROB": PROB, "Y": Y,
              "VIN": {k: d["VIN"].values for k, d in POPS.items()}},
             f"{OUT}/frames_20k.pkl")
print(f"saved {OUT}/frames_20k.pkl")

# ---- part 3: same learner, with vs without the new features ----------------------------------
GROUPS = {
    "legs": [c for c in NEW_COLS if "legs" in c or "days_from_qmid" in c],
    "rhythm_recency": ["days_since_any_visit", "days_since_last_pms", "own_gap_days",
                       "rhythm_ratio_at_cutoff", "rhythm_ratio_at_qend", "visits_last_365d",
                       "nonpms_visits_since_last_pms"],
    "mileage": ["km_per_day_life", "proj_km_at_qend", "km_short_at_qend", "last_km_minus_target"],
    "other": ["n_keys", "missing_10k_steps"],
}
ARMS = {"base_160": [], "+all_new": NEW_COLS, **{f"+{g}": cols for g, cols in GROUPS.items()}}


def frame(k, extra):
    f = X[k].reset_index(drop=True)
    return pd.concat([f, NEW[k][extra].reset_index(drop=True)], axis=1) if extra else f


ab = []
for arm, extra in ARMS.items():
    for s in SEEDS:
        clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=True,
                                             validation_fraction=0.1, random_state=s)
        clf.fit(frame("train", extra), Y["train"])
        for k in ["q1", "q2", "q3"]:
            p = clf.predict_proba(frame(k, extra))[:, 1]
            n_flag = int((PROB[k] >= 0.5).sum())  # same call volume as production at 0.5
            top = np.argsort(-p)[:n_flag]
            bottom = np.argsort(p)[: len(p) // 5]
            ab.append({"arm": arm, "seed": s, "pop": k, "auc": roc_auc_score(Y[k], p),
                       "precision_at_prod_volume": Y[k][top].mean(),
                       "noshow_share_bottom20pct": 1 - Y[k][bottom].mean()})
    print(f"  arm {arm} done")
ab = pd.DataFrame(ab)
summ = ab.groupby(["arm", "pop"])[["auc", "precision_at_prod_volume",
                                   "noshow_share_bottom20pct"]].mean().round(4).reset_index()
prod = []
for k in ["q1", "q2", "q3"]:
    bottom = np.argsort(PROB[k])[: len(PROB[k]) // 5]
    prod.append({"arm": "production_NN", "pop": k, "auc": round(roc_auc_score(Y[k], PROB[k]), 4),
                 "precision_at_prod_volume": round(Y[k][PROB[k] >= 0.5].mean(), 4),
                 "noshow_share_bottom20pct": round(1 - Y[k][bottom].mean(), 4)})
summ = pd.concat([pd.DataFrame(prod), summ], ignore_index=True)
summ.to_csv(f"{OUT}/hgb_ab_20k.csv", index=False)
print("\nA/B (HGB mean of 3 seeds; production NN for reference):")
print(summ.pivot(index="arm", columns="pop").to_string())
