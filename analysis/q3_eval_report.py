"""q3_eval_report.py -- straight evaluation of the delivered Q3 2026 candidate lists.

For each milestone: take the list we shipped (salman/expected_cohort_{m}k_Q3_2026.csv, which
already carries predicted_prob and predicted_flag), stamp the real outcome from the Q3 service
history, and report the confusion matrix and the standard metrics.

No model is loaded and no features are computed -- the predictions already exist in the
delivered files. Re-scoring would measure a different prediction than the one shipped.

TargetFlag here = the vehicle has a Service_Code == milestone record dated inside
2026-07-01..2026-09-30. Strict window, no grace band.

Writes two files into --out-dir:
    q3_metrics_summary.csv   one row per milestone: counts, confusion matrix, AUC, accuracy, F1
    q3_predictions_all.csv   one row per vehicle: milestone, VIN, predicted_prob,
                             predicted_flag, TargetFlag

Run from the repo root:  PYTHONPATH=. python analysis/q3_eval_report.py
"""
import argparse
import os

import pandas as pd
from sklearn.metrics import roc_auc_score

from date_utils import parse_dates
from features import resolve_service_num

MILESTONES = [20, 30, 40, 50, 60, 70, 80, 90, 100]


def load_actuals(path, lo, hi):
    """Q3 service records, restricted to the window.

    Service_Code is authoritative and Description only a fallback -- resolve_service_num()
    handles the '11-20' / 'Nov-20' / '11_20' spellings that int() silently misreads.
    """
    s = pd.read_csv(path, low_memory=False, encoding="ISO-8859-1")
    s["Service_Date"] = parse_dates(s["Service_Date"])
    s["Service_Num"] = resolve_service_num(s)
    s = s.dropna(subset=["Service_Date"])
    s = s[(s["Service_Date"] >= lo) & (s["Service_Date"] <= hi)]
    print(f"actuals: {len(s):,} rows in window, {s['Vin_No'].nunique():,} VINs, "
          f"{s['Service_Date'].min().date()} .. {s['Service_Date'].max().date()}")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actuals", default=os.path.join("Q3", "Service_History_Q3-2026.csv"))
    ap.add_argument("--cohort-dir", default="salman")
    ap.add_argument("--quarter", default="Q3")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--window-start", default="2026-07-01")
    ap.add_argument("--window-end", default="2026-09-30")
    ap.add_argument("--out-dir", default="reconcile_q3_2026")
    ap.add_argument("--milestones", default=",".join(str(m) for m in MILESTONES))
    args = ap.parse_args()

    lo, hi = pd.Timestamp(args.window_start), pd.Timestamp(args.window_end)
    milestones = [int(m) for m in args.milestones.split(",") if m.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"--- Q3 {args.year} evaluation, window {lo.date()} .. {hi.date()} ---")
    act = load_actuals(args.actuals, lo, hi)

    summary, per_vin = [], []
    for m in milestones:
        turned_up = set(act.loc[act["Service_Num"] == m, "Vin_No"].unique())
        path = os.path.join(args.cohort_dir,
                            f"expected_cohort_{m}k_{args.quarter}_{args.year}.csv")
        d = pd.read_csv(path)
        d["TargetFlag"] = d["VIN"].isin(turned_up).astype(int)

        tp = int(((d.predicted_flag == 1) & (d.TargetFlag == 1)).sum())
        fp = int(((d.predicted_flag == 1) & (d.TargetFlag == 0)).sum())
        fn = int(((d.predicted_flag == 0) & (d.TargetFlag == 1)).sum())
        tn = int(((d.predicted_flag == 0) & (d.TargetFlag == 0)).sum())

        n = len(d)
        acc = (tp + tn) / n if n else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        auc = (roc_auc_score(d.TargetFlag, d.predicted_prob)
               if d.TargetFlag.nunique() > 1 else float("nan"))

        summary.append({
            "milestone": f"{m}k", "cohort_size": n,
            "predicted_positive": int((d.predicted_flag == 1).sum()),
            "actual_positive": int(d.TargetFlag.sum()),
            "base_rate": round(d.TargetFlag.mean(), 4),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "specificity": round(spec, 4),
            "f1_score": round(f1, 4), "auc": round(auc, 4),
        })

        out = d[["VIN", "predicted_prob", "predicted_flag", "TargetFlag"]].copy()
        out.insert(0, "milestone", f"{m}k")
        per_vin.append(out)

    s = pd.DataFrame(summary)
    v = pd.concat(per_vin, ignore_index=True)

    sp = os.path.join(args.out_dir, "q3_metrics_summary.csv")
    vp = os.path.join(args.out_dir, "q3_predictions_all.csv")
    s.to_csv(sp, index=False)
    v.to_csv(vp, index=False)

    pd.set_option("display.width", 220)
    print()
    print(s.to_string(index=False))
    print()
    tp, tn = s.TP.sum(), s.TN.sum()
    fp, fn = s.FP.sum(), s.FN.sum()
    tot = tp + tn + fp + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"POOLED  n={tot:,}  TP={tp:,} TN={tn:,} FP={fp:,} FN={fn:,}  "
          f"accuracy={(tp+tn)/tot:.4f}  precision={prec:.4f}  recall={rec:.4f}  "
          f"f1={2*prec*rec/(prec+rec):.4f}")
    print(f"\nWritten: {sp}  ({len(s)} rows)")
    print(f"Written: {vp}  ({len(v):,} rows)")


if __name__ == "__main__":
    main()
