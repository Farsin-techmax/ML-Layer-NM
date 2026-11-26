"""reconcile_quarter.py -- reconcile a delivered forward candidate list against real outcomes.

Nine per-milestone binary classifiers produced nine delivered lists
(salman/expected_cohort_{m}k_Q3_2026.csv) BEFORE Q3 2026 happened -- forward projections of who
would complete each PMS milestone. Q3 actuals have
now arrived (Q3/Service_History_Q3-2026.csv). This script answers one question per milestone: of
everyone who really turned up, how many did we list, and of everyone we listed, how many really
turned up -- plus a breakdown of WHY the misses happened.

NO MODEL IS LOADED AND NO FEATURE IS COMPUTED HERE. predicted_prob / predicted_flag already exist
as columns in the delivered salman/ CSVs -- that is the prediction that was actually shipped to the
business. Re-deriving features and re-scoring would measure a different model run than the one that
went out the door. The job is strictly: read two files, do set arithmetic, compute metrics.

THE THREE-BUCKET FUNNEL
Every actual completer of milestone m falls into exactly one bucket (they sum to the actual total):
  out_gate          completer was never even in the EDA-gate universe the pipeline draws from --
                    a STRUCTURAL CEILING, the model could never have listed them no matter what.
  in_gate_unlisted  completer was gate-eligible but the due-date projection put them in a different
                    quarter -- a COHORT ERROR, the business logic's fault, not the classifier's.
  overlap           completer WAS on the delivered list -- this is the only bucket the per-VIN
                    predicted_flag / predicted_prob can be judged against.

THREE CONFUSION MATRICES, THREE DIFFERENT DENOMINATORS
  A   listed only              rows = the delivered list, exactly as shipped.
  B1  listed union actual      every actual completer the list missed enters as an extra row with
                                predicted_flag=0, predicted_prob=0.0 -- "how did we do against every
                                real completer, not just the ones we had a chance to flag."
  B2  listed union eligible pool   rows = the list plus every gate-eligible VIN that had not already
                                completed milestone m before the window opened -- "how did we do
                                against the whole addressable population."
B2 is NOT a superset of B1: the eligible pool excludes out-of-gate completers (never eligible) and
excludes anyone who had already completed m before the window (not part of "who could become due
this quarter"), so B1 and B2 do not even contain the same count of actual positives.

TP and FP are invariant across all three matrices BY CONSTRUCTION: every row added beyond matrix A
carries a forced predicted_flag=0, so no added row can ever become a new TP or FP. Consequently
PRECISION (TP / (TP+FP)) is also invariant across A/B1/B2 -- it is reported once. FN is NOT
invariant: an unlisted actual completer, forced to predicted_flag=0, is an actual positive predicted
negative, i.e. an FN, never a TN -- so FN (and therefore RECALL) grows as the denominator widens
from "who we listed" to "who really turned up" to "who could ever have turned up". That recall
progression (A -> B1 -> B2) is the headline number this script produces: it separates "did the flag
work on who we listed" from "how much of real demand did we reach at all". B1 and B2 recall are not
directly comparable to each other either, since they count actual positives over different
populations (see above) -- B2's recall can legitimately exceed B1's simply because B2's denominator
excludes vehicles that were never reachable in the first place.

Run from the repo root: PYTHONPATH=. python analysis/reconcile_quarter.py
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from date_utils import parse_dates
from features import resolve_service_num

MILESTONES = [20, 30, 40, 50, 60, 70, 80, 90, 100]

DEFAULT_ACTUALS = os.path.join("Q3", "Service_History_Q3-2026.csv")
DEFAULT_ACTUALS_EDA = os.path.join("Q3", "EDA_Q3-2026.csv")
DEFAULT_EDA = os.path.join("data", "EDA_Q2-2026.csv")
DEFAULT_BASE_HISTORY = os.path.join("data", "Service_History_Q2-2026.csv")

# Measured once (2026-10-06) on the default Q3 2026 files. Asserted only when the script is run
# against those exact default paths -- a later quarter's reconciliation reuses this same script
# against different files, where these specific counts no longer apply.
EXPECTED_ACTUALS_ROWS = 20726
EXPECTED_ACTUALS_VINS = 16028
EXPECTED_ACTUALS_MIN_DATE = "2026-07-01"
EXPECTED_ACTUALS_MAX_DATE = "2026-09-29"
EXPECTED_SERVICE_NUM_COUNTS = {0: 5794, 1: 926, 10: 1604, 20: 1746, 30: 1846, 40: 1713,
                                50: 1499, 60: 1299, 70: 947, 80: 743, 90: 529, 100: 416}
EXPECTED_GATE_VINS = 86481


def load_actuals(path):
    """Q3 service-history actuals -- the ground truth for 'did this VIN really turn up'.

    Same two silent-corruption traps as everywhere else in this repo: parse_dates() (never a bare
    pd.to_datetime -- the raw file is day-first) and resolve_service_num() (never int() on
    Description -- PEP 515 mangles the underscore-spelled service codes).
    """
    act = pd.read_csv(path, low_memory=False, encoding="ISO-8859-1")
    act["Service_Date"] = parse_dates(act["Service_Date"])
    act["Service_Num"] = resolve_service_num(act)
    n_nat = int(act["Service_Date"].isna().sum())
    print(f"actuals: {path}")
    print(f"  {len(act):,} rows, {act['Vin_No'].nunique():,} unique VINs, "
          f"{act['Service_Date'].min().date()} .. {act['Service_Date'].max().date()}, "
          f"{n_nat} unparsed dates")

    if os.path.normpath(path) == os.path.normpath(DEFAULT_ACTUALS):
        assert len(act) == EXPECTED_ACTUALS_ROWS, \
            f"row count drifted: {len(act)} != {EXPECTED_ACTUALS_ROWS}"
        assert act["Vin_No"].nunique() == EXPECTED_ACTUALS_VINS, \
            f"unique VIN count drifted: {act['Vin_No'].nunique()} != {EXPECTED_ACTUALS_VINS}"
        assert n_nat == 0, f"{n_nat} unparsed Service_Date rows -- was 0"
        assert act["Service_Date"].min() == pd.Timestamp(EXPECTED_ACTUALS_MIN_DATE), \
            f"min date drifted: {act['Service_Date'].min()} != {EXPECTED_ACTUALS_MIN_DATE}"
        assert act["Service_Date"].max() == pd.Timestamp(EXPECTED_ACTUALS_MAX_DATE), \
            f"max date drifted: {act['Service_Date'].max()} != {EXPECTED_ACTUALS_MAX_DATE}"
        counts = act["Service_Num"].value_counts()
        for sn, expected in EXPECTED_SERVICE_NUM_COUNTS.items():
            got = int(counts.get(sn, 0))
            assert got == expected, f"Service_Num {sn}: {got} != {expected}"
        print("  baseline assertions: PASS (matches the measured Q3 2026 facts)")
    else:
        print("  (not the default Q3 2026 file -- skipping hardcoded baseline assertions)")
    return act


def load_gate(path):
    """EDA gate: the universe the delivered lists could ever have drawn a candidate from.

    data/EDA_Q2-2026.csv -- the file the Q3 lists were actually built from -- NEVER
    Q3/EDA_Q3-2026.csv, which reflects post-quarter state and would retroactively change who was
    ever eligible.
    """
    eda = pd.read_csv(path, low_memory=False, encoding="ISO-8859-1")
    gate = eda[(eda["Last Service - PMS"] != "-") &
               (eda["Vehicle Service Status"].isin(["InActive", "Lapsed", "Active"]))]
    vins = set(gate["VIN"].unique())
    print(f"EDA gate: {path}")
    print(f"  {len(vins):,} unique VINs")
    if os.path.normpath(path) == os.path.normpath(DEFAULT_EDA):
        assert len(vins) == EXPECTED_GATE_VINS, f"gate size drifted: {len(vins)} != {EXPECTED_GATE_VINS}"
        print("  baseline assertion: PASS")
    return vins


def load_base_history(path, win_lo):
    """Base service history, parsed once -- used only to find who already completed a milestone
    BEFORE the window opened (the B2 eligible-pool exclusion). NOT the actuals source: this file
    ends 2026-06-29, before the Q3 window even starts.
    """
    hist = pd.read_csv(path, low_memory=False, encoding="ISO-8859-1")
    hist["Service_Date"] = parse_dates(hist["Service_Date"])
    hist["Service_Num"] = resolve_service_num(hist)
    print(f"base history: {path}")
    print(f"  {len(hist):,} rows, {hist['Service_Date'].min().date()} .. "
          f"{hist['Service_Date'].max().date()}")
    before = hist[hist["Service_Date"] < pd.Timestamp(win_lo)]
    print(f"  {len(before):,} rows before window start {win_lo}")
    return before


def load_cohort(cohort_dir, quarter, year, m):
    """One delivered list. Guards against duplicate VINs even though none exist in the 9 known
    Q3 2026 files -- a future quarter's file is not guaranteed to be as clean.
    """
    path = os.path.join(cohort_dir, f"expected_cohort_{m}k_{quarter}_{year}.csv")
    df = pd.read_csv(path, low_memory=False, encoding="ISO-8859-1")
    dup = int(df["VIN"].duplicated().sum())
    if dup:
        print(f"  WARNING: {dup} duplicate VIN rows in {path}, keeping first occurrence")
        df = df.drop_duplicates(subset="VIN", keep="first")
    df["predicted_flag"] = df["predicted_flag"].astype(int)
    df["predicted_prob"] = df["predicted_prob"].astype(float)
    return df, path


def funnel(actual_vins, gate_vins, listed_vins):
    """Partition actual completers into the three buckets; they must sum to the actual total."""
    out_gate = actual_vins - gate_vins
    in_gate = actual_vins & gate_vins
    unlisted = in_gate - listed_vins
    overlap = actual_vins & listed_vins
    assert len(out_gate) + len(unlisted) + len(overlap) == len(actual_vins), (
        f"funnel buckets {len(out_gate)}+{len(unlisted)}+{len(overlap)} != "
        f"actual total {len(actual_vins)}")
    return {"out_gate": out_gate, "in_gate_unlisted": unlisted, "overlap": overlap}


def confusion_stats(y_true, y_pred):
    """TP/FP/FN/TN plus accuracy and precision/recall/F1 for BOTH classes, every ratio guarded."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "total": total,
        "accuracy": accuracy,
        "precision_neg": float(prec[0]), "recall_neg": float(rec[0]), "f1_neg": float(f1[0]),
        "precision_pos": float(prec[1]), "recall_pos": float(rec[1]), "f1_pos": float(f1[1]),
    }


def auc_safe(y_true, y_prob):
    """roc_auc_score, NaN when the universe is single-class (undefined otherwise)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def matrix_A(listed_df, actual_vins):
    y_true = listed_df["VIN"].isin(actual_vins).astype(int).to_numpy()
    y_pred = listed_df["predicted_flag"].to_numpy()
    y_prob = listed_df["predicted_prob"].to_numpy()
    stats = confusion_stats(y_true, y_pred)
    stats["auc"] = auc_safe(y_true, y_prob)
    stats["universe"] = len(listed_df)
    stats["actual_in_universe"] = int(y_true.sum())
    return stats


def matrix_B1(listed_df, actual_vins):
    """listed union actual completers -- unlisted completers enter forced to predicted_flag=0."""
    listed_vins = set(listed_df["VIN"])
    extra_vins = actual_vins - listed_vins
    y_true = np.concatenate([
        listed_df["VIN"].isin(actual_vins).astype(int).to_numpy(),
        np.ones(len(extra_vins), dtype=int),
    ])
    y_pred = np.concatenate([
        listed_df["predicted_flag"].to_numpy(),
        np.zeros(len(extra_vins), dtype=int),
    ])
    stats = confusion_stats(y_true, y_pred)
    stats["universe"] = len(listed_df) + len(extra_vins)
    stats["actual_in_universe"] = int(y_true.sum())
    return stats


def matrix_B2(listed_df, pool_vins, actual_vins):
    """listed union eligible pool -- pool members not on the list enter at predicted_flag=0.

    Listed VINs keep their own flag even when they fall outside the pool (e.g. an early completer
    who had already done this milestone before the window -- candidates-mode keeps those as
    legitimate positives, see CLAUDE.md).
    """
    listed_vins = set(listed_df["VIN"])
    extra_vins = pool_vins - listed_vins
    extra_true = np.fromiter((1 if v in actual_vins else 0 for v in extra_vins),
                              dtype=int, count=len(extra_vins))
    y_true = np.concatenate([
        listed_df["VIN"].isin(actual_vins).astype(int).to_numpy(),
        extra_true,
    ])
    y_pred = np.concatenate([
        listed_df["predicted_flag"].to_numpy(),
        np.zeros(len(extra_vins), dtype=int),
    ])
    stats = confusion_stats(y_true, y_pred)
    stats["universe"] = len(listed_df) + len(extra_vins)
    stats["actual_in_universe"] = int(y_true.sum())
    return stats


def build_detail(listed_df, actual_vins, actual_any_vins, fnl, date_map):
    """Per-VIN detail: the listed rows plus every unlisted actual completer (the B1 universe),
    labelled with which funnel bucket it fell into. The B2 eligible pool (tens of thousands of
    VINs) is deliberately NOT dumped per-VIN here -- only its aggregate counts are reported.
    """
    d = listed_df[["VIN", "predicted_prob", "predicted_flag"]].copy()
    d["actual_exact"] = d["VIN"].isin(actual_vins).astype(int)
    d["actual_any"] = d["VIN"].isin(actual_any_vins).astype(int)
    d["bucket"] = "listed"

    extra_vins = sorted(fnl["out_gate"] | fnl["in_gate_unlisted"])
    extra = pd.DataFrame({"VIN": extra_vins})
    extra["predicted_prob"] = 0.0
    extra["predicted_flag"] = 0
    extra["actual_exact"] = 1
    extra["actual_any"] = extra["VIN"].isin(actual_any_vins).astype(int)
    extra["bucket"] = extra["VIN"].apply(
        lambda v: "out_gate" if v in fnl["out_gate"] else "in_gate_unlisted")

    out = pd.concat([d, extra], ignore_index=True)
    out["Service_Date"] = out["VIN"].map(date_map)
    out["Service_Date"] = out["Service_Date"].apply(
        lambda ts: ts.strftime("%Y-%m-%d") if pd.notna(ts) else "")
    return out


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quarter", default="Q3")
    ap.add_argument("--year", default="2026")
    ap.add_argument("--window-start", default="2026-07-01")
    ap.add_argument("--window-end", default="2026-09-30")
    ap.add_argument("--actuals", default=DEFAULT_ACTUALS)
    ap.add_argument("--actuals-eda", default=DEFAULT_ACTUALS_EDA)
    ap.add_argument("--eda", default=DEFAULT_EDA)
    ap.add_argument("--base-history", default=DEFAULT_BASE_HISTORY)
    ap.add_argument("--cohort-dir", default="salman")
    ap.add_argument("--milestones", default=",".join(str(m) for m in MILESTONES))
    ap.add_argument("--out-dir", default="reconcile_q3_2026")
    args = ap.parse_args()

    milestones = [int(m) for m in args.milestones.split(",") if m.strip()]
    win_lo, win_hi = pd.Timestamp(args.window_start), pd.Timestamp(args.window_end)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"--- Reconciling {args.quarter} {args.year} ({args.window_start} .. "
          f"{args.window_end}) against delivered lists in {args.cohort_dir}/ ---\n")

    actuals = load_actuals(args.actuals)
    gate_vins = load_gate(args.eda)
    hist_before = load_base_history(args.base_history, win_lo)
    print()

    w = actuals[(actuals["Service_Date"] >= win_lo) & (actuals["Service_Date"] <= win_hi)]
    actual_any_vins = set(w.loc[w["Service_Num"] > 0, "Vin_No"].unique())

    eda3 = pd.read_csv(args.actuals_eda, low_memory=False, encoding="ISO-8859-1")
    eda3_counts = eda3["Service_Code"].value_counts()

    summary_rows = []
    detail_frames = {}

    for m in milestones:
        listed_df, cohort_path = load_cohort(args.cohort_dir, args.quarter, args.year, m)
        listed_vins = set(listed_df["VIN"])

        actual_vins = set(w.loc[w["Service_Num"] == m, "Vin_No"].unique())
        fnl = funnel(actual_vins, gate_vins, listed_vins)

        done_before = set(hist_before.loc[hist_before["Service_Num"] == m, "Vin_No"].unique())
        pool_vins = gate_vins - done_before

        mA = matrix_A(listed_df, actual_vins)
        mB1 = matrix_B1(listed_df, actual_vins)
        mB2 = matrix_B2(listed_df, pool_vins, actual_vins)

        assert mA["TP"] == mB1["TP"] == mB2["TP"], \
            f"{m}k: TP not invariant: A={mA['TP']} B1={mB1['TP']} B2={mB2['TP']}"
        assert mA["FP"] == mB1["FP"] == mB2["FP"], \
            f"{m}k: FP not invariant: A={mA['FP']} B1={mB1['FP']} B2={mB2['FP']}"

        fnl_any = funnel(actual_any_vins, gate_vins, listed_vins)
        mA_any = matrix_A(listed_df, actual_any_vins)

        eda_count = int(eda3_counts.get(m, 0))
        hist_count = len(actual_vins)

        date_map = w.loc[w["Service_Num"] == m].groupby("Vin_No")["Service_Date"].min()
        detail_frames[m] = build_detail(listed_df, actual_vins, actual_any_vins, fnl, date_map)
        detail_path = os.path.join(args.out_dir, f"detail_{m}k_{args.quarter}_{args.year}.csv")
        detail_frames[m].to_csv(detail_path, index=False)

        row = {
            "milestone": m,
            "listed": len(listed_df),
            "pool_size": len(pool_vins),
            "gate_size": len(gate_vins),
            "already_done_before_window": len(done_before & gate_vins),
            # exact-match funnel
            "actual_exact": len(actual_vins),
            "out_gate_exact": len(fnl["out_gate"]),
            "in_gate_unlisted_exact": len(fnl["in_gate_unlisted"]),
            "overlap_exact": len(fnl["overlap"]),
            "recall_list_pct_exact": pct(len(fnl["overlap"]), len(actual_vins)),
            "precision_list_pct_exact": pct(len(fnl["overlap"]), len(listed_df)),
        }
        for prefix, mx in (("A", mA), ("B1", mB1), ("B2", mB2)):
            for k, v in mx.items():
                row[f"{prefix}_{k}"] = v
        row["precision_invariant_ok"] = bool(
            mA["TP"] == mB1["TP"] == mB2["TP"] and mA["FP"] == mB1["FP"] == mB2["FP"])
        # secondary any-PMS funnel + matrix A
        row.update({
            "actual_any": len(actual_any_vins),
            "out_gate_any": len(fnl_any["out_gate"]),
            "in_gate_unlisted_any": len(fnl_any["in_gate_unlisted"]),
            "overlap_any": len(fnl_any["overlap"]),
            "recall_list_pct_any": pct(len(fnl_any["overlap"]), len(actual_any_vins)),
            "precision_list_pct_any": pct(len(fnl_any["overlap"]), len(listed_df)),
        })
        for k, v in mA_any.items():
            row[f"Aany_{k}"] = v
        # EDA cross-check
        row.update({
            "eda_code_count": eda_count,
            "hist_actual_exact_count": hist_count,
            "eda_vs_hist_diff": eda_count - hist_count,
        })
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, f"reconcile_summary_{args.quarter}_{args.year}.csv")
    summary.to_csv(summary_path, index=False)

    # ---------------- printed report ----------------
    def ms_label(m):
        return f"{m:>4}k"

    print("\n=== 1. FUNNEL + MATRIX A (headline, exact-match rule) ===")
    print(f"{'ms':>4} {'actual':>7} {'out_gate':>9} {'unlisted':>9} {'listed':>7} "
          f"{'overlap':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'recall%':>8} {'prec%':>7}")
    for r in summary_rows:
        print(f"{r['milestone']:>3}k {r['actual_exact']:>7,} {r['out_gate_exact']:>9} "
              f"{r['in_gate_unlisted_exact']:>9} {r['listed']:>7,} {r['overlap_exact']:>8} "
              f"{r['A_TP']:>5} {r['A_FP']:>5} {r['A_FN']:>5} {r['A_TN']:>5} "
              f"{r['recall_list_pct_exact']:>8.1f} {r['precision_list_pct_exact']:>7.1f}")

    print("\nNote: recall% = overlap/actual, prec% = overlap/listed (list-level, computed from the "
          "funnel sets directly -- NOT the same as the flag-level precision/recall below, which "
          "come from the TP/FP/FN/TN confusion matrix).")

    def print_matrix_table(title, rows, key, extra_cols=()):
        print(f"\n=== {title} ===")
        header = (f"{'ms':>4} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'acc%':>6} "
                  f"{'prec_pos':>8} {'rec_pos':>7} {'f1_pos':>6} "
                  f"{'prec_neg':>8} {'rec_neg':>7} {'f1_neg':>6}")
        for c in extra_cols:
            header += f" {c:>10}"
        print(header)
        for r in rows:
            mx = {k: r[f"{key}_{k}"] for k in
                  ("TP", "FP", "FN", "TN", "accuracy", "precision_pos", "recall_pos", "f1_pos",
                   "precision_neg", "recall_neg", "f1_neg")}
            line = (f"{r['milestone']:>3}k {mx['TP']:>5} {mx['FP']:>5} {mx['FN']:>5} "
                    f"{mx['TN']:>5} {100*mx['accuracy']:>6.1f} "
                    f"{mx['precision_pos']:>8.4f} {mx['recall_pos']:>7.4f} {mx['f1_pos']:>6.4f} "
                    f"{mx['precision_neg']:>8.4f} {mx['recall_neg']:>7.4f} {mx['f1_neg']:>6.4f}")
            for c in extra_cols:
                line += f" {r[c]:>10}"
            print(line)

    print_matrix_table("2. MATRIX A -- listed only (AUC valid here, uses predicted_prob)",
                        summary_rows, "A", extra_cols=())
    for r in summary_rows:
        pass
    # AUC printed as its own compact line since it doesn't fit the common header above
    print(f"\n{'ms':>4} {'A_auc':>8}")
    for r in summary_rows:
        auc = r["A_auc"]
        print(f"{r['milestone']:>3}k {auc:>8.4f}" if not np.isnan(auc) else
              f"{r['milestone']:>3}k {'n/a':>8}")

    print_matrix_table(
        "3. MATRIX B1 -- listed UNION actual completers "
        "(unlisted completers forced to predicted_flag=0)",
        summary_rows, "B1", extra_cols=("B1_universe", "B1_actual_in_universe"))

    print_matrix_table(
        "4. MATRIX B2 -- listed UNION eligible pool "
        "(pool = gate VINs with no prior completion before the window; pool-not-listed forced to "
        "predicted_flag=0)",
        summary_rows, "B2", extra_cols=("pool_size", "gate_size", "B2_actual_in_universe"))

    print("\n=== 5. RECALL ACROSS A / B1 / B2, side by side (the headline progression) ===")
    print("Precision is invariant across all three by construction (TP and FP never change -- no "
          "row added beyond matrix A ever carries predicted_flag=1). Recall is NOT invariant: it "
          "falls as the denominator widens from 'who we listed' (A) to 'who really turned up' (B1) "
          "to 'who could ever turn up' (B2). B1 and B2 are not directly comparable to EACH OTHER "
          "either -- they count actual positives over different populations (B2 excludes "
          "out-of-gate completers and anyone who already completed the milestone before the "
          "window), so B2 recall exceeding B1 recall is expected, not an error.")
    print(f"\n{'ms':>4} {'precision':>10} {'recall_A':>9} {'recall_B1':>9} {'recall_B2':>9} "
          f"{'actual_B1':>10} {'actual_B2':>10}")
    for r in summary_rows:
        print(f"{r['milestone']:>3}k {r['A_precision_pos']:>10.4f} {r['A_recall_pos']:>9.4f} "
              f"{r['B1_recall_pos']:>9.4f} {r['B2_recall_pos']:>9.4f} "
              f"{r['B1_actual_in_universe']:>10} {r['B2_actual_in_universe']:>10}")

    print("\n=== 6. SECONDARY RULE -- any PMS visit (Service_Num > 0), not just the exact milestone ===")
    print(f"{'ms':>4} {'actual':>7} {'out_gate':>9} {'unlisted':>9} {'listed':>7} "
          f"{'overlap':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'recall%':>8} {'prec%':>7} "
          f"{'auc':>7}")
    for r in summary_rows:
        auc = r["Aany_auc"]
        auc_s = f"{auc:>7.4f}" if not np.isnan(auc) else f"{'n/a':>7}"
        print(f"{r['milestone']:>3}k {r['actual_any']:>7,} {r['out_gate_any']:>9} "
              f"{r['in_gate_unlisted_any']:>9} {r['listed']:>7,} {r['overlap_any']:>8} "
              f"{r['Aany_TP']:>5} {r['Aany_FP']:>5} {r['Aany_FN']:>5} {r['Aany_TN']:>5} "
              f"{r['recall_list_pct_any']:>8.1f} {r['precision_list_pct_any']:>7.1f} {auc_s}")

    print("\n=== 7. EDA CROSS-CHECK (Q3/EDA_Q3-2026.csv, Vehicle_Key-keyed -- surfaced, not "
          "reconciled away; see module docstring) ===")
    print(f"{'ms':>4} {'eda_code_count':>14} {'hist_vin_count':>14} {'diff':>6}")
    for r in summary_rows:
        print(f"{r['milestone']:>3}k {r['eda_code_count']:>14,} {r['hist_actual_exact_count']:>14,} "
              f"{r['eda_vs_hist_diff']:>6,}")

    print(f"\nWritten: {summary_path}")
    for m in milestones:
        print(f"Written: {os.path.join(args.out_dir, f'detail_{m}k_{args.quarter}_{args.year}.csv')}")


if __name__ == "__main__":
    main()
