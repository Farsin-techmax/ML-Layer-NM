"""Remove cross-milestone duplicates from the Q2 avg-interval lists and delivery files.

Root cause: avg_interval_date()'s due date for a vehicle depends only on its highest COMPLETED
milestone below the target ("hi") and the average length of its own past service segments -- NOT
on which target milestone m is being asked about. So a vehicle whose last completed step is 10k
gets the identical due date whether m=20, 30, ..., or 100 (hi=10 either way, same segments feed the
same average). With the prev-step gate off (per the user's explicit first-principles choice), that
means the SAME vehicle, with the SAME due date, lands on every milestone list above its current
step -- 95.0% of all 8,341 listed VINs were duplicated this way (verified 2026-10-21).

Fix: each VIN keeps exactly one listing, on the LOWEST milestone it appears on across all nine
lists. That milestone always equals last_pms_step + 10 (verified: 0/8,341 exceptions) -- i.e. the
vehicle's genuine immediate next service -- so this is not an arbitrary tie-break, it recovers the
correct target milestone for every vehicle. Dropped from every higher list.

Run from repo root (after build_q2_avg_interval.py and score_q2_avg_interval.py):
    PYTHONPATH=. ./venv/Scripts/python.exe Q2/dedupe_milestones.py
"""
import os
import shutil

import pandas as pd
from sklearn.metrics import roc_auc_score

from build_q2_avg_interval import MILESTONES

OUT_DIR = "Q2"
BACKUP_DIR = os.path.join(OUT_DIR, "pre_dedup")


def backup(path):
    dest = os.path.join(BACKUP_DIR, os.path.relpath(path, OUT_DIR))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):  # don't clobber a backup with an already-deduped file on rerun
        shutil.copy2(path, dest)


def main():
    list_paths = {m: os.path.join(OUT_DIR, "lists", f"avg_interval_{m}k_Q2_2026.csv") for m in MILESTONES}
    cohort_paths = {m: os.path.join(OUT_DIR, f"expected_cohort_{m}k_Q2_2026.csv") for m in MILESTONES}
    for p in list(list_paths.values()) + list(cohort_paths.values()):
        backup(p)

    lists = {m: pd.read_csv(list_paths[m]) for m in MILESTONES}
    for m in MILESTONES:
        lists[m]["milestone"] = m
    allv = pd.concat(lists.values(), ignore_index=True)

    min_ms = allv.groupby("VIN")["milestone"].min()

    print(f"Total distinct VINs listed (pre-dedup): {min_ms.shape[0]:,}")
    dupes = allv.groupby("VIN")["milestone"].nunique()
    print(f"VINs that were on 2+ lists: {(dupes > 1).sum():,} ({100*(dupes>1).mean():.1f}%)")

    rows = []
    for m in MILESTONES:
        lst = lists[m].drop(columns="milestone")
        keep_vins = set(min_ms.index[min_ms == m])
        before = len(lst)
        lst_dedup = lst[lst["VIN"].isin(keep_vins)].reset_index(drop=True)
        lst_dedup.to_csv(list_paths[m], index=False)

        cohort = pd.read_csv(cohort_paths[m])
        cohort_dedup = cohort[cohort["VIN"].isin(keep_vins)].reset_index(drop=True)
        cohort_dedup.to_csv(cohort_paths[m], index=False)

        came = int(lst_dedup["CameInQ2"].sum()) if "CameInQ2" in lst_dedup.columns else None
        visitors_all = lst_dedup["all_Q2_visitors"].iloc[0] if "all_Q2_visitors" in lst_dedup.columns else None

        y = lst_dedup.set_index("VIN")["CameInQ2"] if "CameInQ2" in lst_dedup.columns else None
        flag = cohort_dedup.set_index("VIN")["predicted_flag"]
        prob = cohort_dedup.set_index("VIN")["predicted_prob"]
        yv = y.reindex(flag.index)
        tp = int(((flag == 1) & (yv == 1)).sum())
        fp = int(((flag == 1) & (yv == 0)).sum())
        fn = int(((flag == 0) & (yv == 1)).sum())
        tn = int(((flag == 0) & (yv == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")

        rows.append({
            "milestone": f"{m}k", "listed_before": before, "listed_after": len(lst_dedup),
            "removed": before - len(lst_dedup), "came_in_from_list": came,
            "flagged": tp + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision_pct": round(100 * prec, 1) if tp + fp else None,
            "AUC": round(roc_auc_score(yv, prob), 4) if yv.nunique() == 2 else None,
        })
        print(f"{m}k: {before:,} -> {len(lst_dedup):,} ({before - len(lst_dedup):,} removed)")

    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(OUT_DIR, "eval_model_Q2_2026.csv"), index=False)
    print("\n" + ev.to_string(index=False))

    summary = pd.read_csv(os.path.join(OUT_DIR, "summary_avg_interval_Q2_2026.csv"))
    summary = summary.merge(ev[["milestone", "listed_after", "removed", "precision_pct"]],
                            on="milestone", suffixes=("_orig", ""))
    summary.to_csv(os.path.join(OUT_DIR, "summary_avg_interval_Q2_2026.csv"), index=False)
    print(f"\nOriginals backed up under {BACKUP_DIR}/")


if __name__ == "__main__":
    main()
