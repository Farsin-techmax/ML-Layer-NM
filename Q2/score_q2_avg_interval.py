"""Score the Q2 2026 average-interval lists with the promoted production_models/{m}k and write the
delivery files. Run AFTER build_q2_avg_interval.py.

Decided with the user 2026-10-21:
  - model = production_models/{m}k (model.pt + its own selected_features.json), threshold from its
    threshold.json (0.5 for all nine)
  - features cut at 2026-03-31, the same day the list was built from
  - built as a FORWARD list: TargetFlag = 0 for every row while features are built, because
    process_service_data() takes a different code path for TargetFlag == 1 rows; the real Q2
    outcome is attached only afterwards, for evaluation
  - EDA = the Q1 2026 snapshot (data/EDA_Q1-2026.csv), so EDA-derived features carry no Q2 info

Per milestone, writes:
  Q2/features/features_{m}k_Q2_2026.csv   full process_service_data() matrix (scorer input)
  Q2/scored/scored_{m}k_Q2_2026.csv        score_milestone.py output (prob_turnup, pred)
  Q2/expected_cohort_{m}k_Q2_2026.csv      8-column delivery file, same layout as salman/
and once for all milestones:
  Q2/eval_model_Q2_2026.csv                confusion matrix + metrics against real Q2 visits

Run from repo root:
    PYTHONPATH=. ./venv/Scripts/python.exe Q2/score_q2_avg_interval.py [--milestones 20,30]
"""
import argparse
import os
import subprocess
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_q2_avg_interval import CUTOFF, MILESTONES, load  # noqa: E402

from pmstrainfeatureeng_refactored import extract_k1, process_service_data  # noqa: E402

RFM_PATH = "data/RFM_Q2-2026.csv"
APPT_PATH = "data/Appointments_Q2-2026.csv"
DIGI_PATH = "data/Digital_Sessions_Q2-2026.csv"
VHC_PATH = "data/VHC_Q2-2026.csv"
SERVCODE_PATH = "data/Service_Code_Desc.csv"


def build_features(df, serv_cut, lst, m, out_path):
    cohort = df[df["VIN"].isin(lst["VIN"])].copy()
    cohort[f"Expected{m}kDate"] = cohort["VIN"].map(lst.set_index("VIN")["NextServiceDate"])
    cohort["TargetFlag"] = 0
    cohort["Service_Num"] = cohort["Last Service - PMS"].apply(extract_k1)
    feats = process_service_data(
        mastersheet=cohort, servhistory=serv_cut, rfm=RFM_PATH, servcode=SERVCODE_PATH,
        appointdf=APPT_PATH, digidf=DIGI_PATH, vhc=VHC_PATH,
        filter_date=CUTOFF.strftime("%Y-%m-%d"), last_service_code=m, is_test=True,
    )
    lead = ["VIN", "TargetFlag"]
    feats = feats[lead + [c for c in feats.columns if c not in lead]]
    feats.to_csv(out_path, index=False)
    return feats


def score(m, feat_path, out_path):
    mdir = f"production_models/{m}k"
    if os.path.exists(os.path.join(mdir, "best_model.pt")):
        sys.exit(f"{mdir}/best_model.pt exists and would shadow model.pt -- stop and check")
    cmd = [sys.executable, "score_milestone.py", str(m), "--test", feat_path,
           "--model-dir", mdir, "--features", os.path.join(mdir, "selected_features.json"),
           "--out", out_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("\n".join(l for l in res.stdout.splitlines()
                    if any(k in l for k in ("rows=", "zero-filled", "WARNING", "Predicted"))))
    if res.returncode != 0:
        sys.exit(res.stderr[-3000:])
    return pd.read_csv(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda", default="data/EDA_Q1-2026.csv")
    ap.add_argument("--out-dir", default="Q2")
    ap.add_argument("--milestones", default=",".join(str(m) for m in MILESTONES))
    args = ap.parse_args()
    milestones = [int(x) for x in args.milestones.split(",")]

    df, serv, _ = load(args.eda)
    serv_cut = serv[serv["Service_Date"] <= CUTOFF]
    for sub in ("features", "scored"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    rows = []
    for m in milestones:
        print(f"\n===== {m}k =====")
        lst = pd.read_csv(os.path.join(args.out_dir, "lists", f"avg_interval_{m}k_Q2_2026.csv"))
        feat_path = os.path.join(args.out_dir, "features", f"features_{m}k_Q2_2026.csv")
        feats = build_features(df, serv_cut, lst, m, feat_path)
        lost = set(lst["VIN"]) - set(feats["VIN"])
        print(f"list {len(lst):,} VINs -> feature rows {len(feats):,} (lost {len(lost)})")

        sc = score(m, feat_path, os.path.join(args.out_dir, "scored", f"scored_{m}k_Q2_2026.csv"))
        due = pd.to_datetime(sc[f"Next{m}K_Due"], format="mixed").dt.strftime("%Y-%m-%d")
        lst_idx = lst.set_index("VIN")
        deliver = pd.DataFrame({
            "Predicted Quarter": "Q2", "Predicted Year": 2026,
            "Predicted Service Mileage": f"{m}k", "Predicted Service_Code": m,
            "NextServiceDate": due, "VIN": sc["VIN"],
            "predicted_prob": sc["prob_turnup"], "predicted_flag": sc["pred"],
            # Ground truth, added 2026-10-21 for review -- never fed to the model. ActualServiceDate
            # is the vehicle's real first-ever completion of this milestone if the data shows one
            # (service history runs to 2026-06-29, so any date present here already falls inside Q2
            # by construction -- a listed VIN that completed before Q2 was already stale-dropped).
            # ActualTargetFlag is 1 only when that completion falls inside the strict Q2 window.
            "ActualServiceDate": sc["VIN"].map(lst_idx[f"Actual{m}kDate"]),
            "ActualTargetFlag": sc["VIN"].map(lst_idx["CameInQ2"]),
            # The exact per-vehicle average service interval (months) the due date was built from,
            # before scaling by steps_away -- added 2026-10-21 for transparency into the method.
            "AvgServiceIntervalMonths": sc["VIN"].map(lst_idx["AvgServiceIntervalMonths"]),
        })
        deliver.to_csv(os.path.join(args.out_dir, f"expected_cohort_{m}k_Q2_2026.csv"), index=False)

        # Evaluation only: real Q2 outcome from the list file, never seen by the model.
        y = sc["VIN"].map(lst.set_index("VIN")["CameInQ2"]).astype(int)
        p, f = sc["prob_turnup"], sc["pred"]
        visitors_all = serv[serv["Service_Num"] == m].groupby("Vin_No")["Service_Date"].min()
        n_all = int(((visitors_all >= pd.Timestamp("2026-04-01")) &
                     (visitors_all <= pd.Timestamp("2026-06-30"))).sum())
        tp, fp = int(((f == 1) & (y == 1)).sum()), int(((f == 1) & (y == 0)).sum())
        fn, tn = int(((f == 0) & (y == 1)).sum()), int(((f == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec_list = tp / (tp + fn) if tp + fn else float("nan")
        rec_all = tp / n_all if n_all else float("nan")
        rows.append({
            "milestone": f"{m}k", "listed": len(sc), "came_in_from_list": int(y.sum()),
            "all_Q2_visitors": n_all, "flagged": tp + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision_pct": round(100 * prec, 1),
            "recall_in_list_pct": round(100 * rec_list, 1),
            "recall_of_all_pct": round(100 * rec_all, 1),
            "f1_vs_all": round(2 * prec * rec_all / (prec + rec_all), 3) if tp else 0.0,
            "AUC": round(roc_auc_score(y, p), 4) if y.nunique() == 2 else None,
            "mean_prob": round(p.mean(), 3),
        })
        print(rows[-1])

    ev = pd.DataFrame(rows)
    ev_path = os.path.join(args.out_dir, "eval_model_Q2_2026.csv")
    if os.path.exists(ev_path) and len(milestones) < len(MILESTONES):
        old = pd.read_csv(ev_path)
        ev = pd.concat([old[~old["milestone"].isin(ev["milestone"])], ev], ignore_index=True)
    ev.to_csv(ev_path, index=False)
    print("\n" + ev.to_string(index=False))


if __name__ == "__main__":
    main()
