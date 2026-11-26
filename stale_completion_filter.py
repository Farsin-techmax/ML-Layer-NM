"""Post-filter for forward-looking predictions/{m}k/expected_cohort_{Q3,Q4}_2026.csv files: drop
any VIN whose raw service history already shows a real completion record for the target milestone
dated before the window opens. Decided 2026-09-09: forward lists only, no change to training/eval
labelling (EarlyCompleter stays as-is there -- that is a deliberate, documented design, see
CLAUDE.md). This is a pure post-processing step on the scored output -- it does not touch
prepare_test_set.py, the training matrix, or any promoted model. Run from repo root.

Usage: python stale_completion_filter.py <milestone> <window_start YYYY-MM-DD> <expected_cohort_csv>
Overwrites the input file in place with the stale rows removed, and writes a sibling
<name>_removed_stale_completions.csv listing exactly what was dropped and why, for auditability.
"""
import sys
import pandas as pd
from date_utils import parse_dates
from features import resolve_service_num


def filter_stale_completions(milestone, window_start, cohort_path):
    serv = pd.read_csv('data/Service_History_Q2-2026.csv', low_memory=False, encoding='ISO-8859-1')
    serv['Service_Date'] = parse_dates(serv['Service_Date'])
    serv['Service_Num'] = resolve_service_num(serv)

    completed = (serv[serv['Service_Num'] == milestone]
                 .sort_values('Service_Date')
                 .groupby('Vin_No')['Service_Date'].first())

    df = pd.read_csv(cohort_path)
    win_start_ts = pd.Timestamp(window_start)
    completion_date = df['VIN'].map(completed)
    is_stale = completion_date.notna() & (completion_date < win_start_ts)

    removed = df[is_stale].copy()
    removed['actual_completion_date'] = completion_date[is_stale].dt.date
    kept = df[~is_stale].copy()

    kept.to_csv(cohort_path, index=False)
    removed_path = cohort_path.replace('.csv', '_removed_stale_completions.csv')
    removed.to_csv(removed_path, index=False)

    print(f"{cohort_path}: {len(df)} -> {len(kept)} rows "
          f"({is_stale.sum()} stale-completion VINs removed, {kept['VIN'].duplicated().sum()} dup VINs remain)")
    print(f"  removed list -> {removed_path}")
    return len(df), len(kept), is_stale.sum()


if __name__ == '__main__':
    milestone = int(sys.argv[1])
    window_start = sys.argv[2]
    cohort_path = sys.argv[3]
    filter_stale_completions(milestone, window_start, cohort_path)
