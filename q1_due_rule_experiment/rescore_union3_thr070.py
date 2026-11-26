"""One-off: re-cut the already-scored 3-leg union+stale 20k expected bases (pad0 and pad45,
Q1 and Q2) at threshold 0.70 instead of 0.5, against strict-quarter real outcomes.
Reads q1_due_rule_experiment/union3_pad{0,45}_20k/scored_Q{1,2}.csv (prob_turnup per VIN,
already produced by score_milestone.py against production_models/20k) -- no re-scoring of the
model, just a different cutoff on the same probabilities. Run from repo root.
"""
import pandas as pd
from date_utils import parse_dates
from features import resolve_service_num

MILESTONE = 20
WINDOWS = {
    'Q1': (pd.Timestamp('2026-01-01'), pd.Timestamp('2026-03-31')),
    'Q2': (pd.Timestamp('2026-04-01'), pd.Timestamp('2026-06-30')),
}

serv = pd.read_csv('data/Service_History_Q2-2026.csv', low_memory=False, encoding='ISO-8859-1')
serv['Service_Date'] = parse_dates(serv['Service_Date'])
serv['Service_Num'] = resolve_service_num(serv)
hits = serv[serv['Service_Num'] == MILESTONE][['Vin_No', 'Service_Date']]

rows = []
for pad in ['pad0', 'pad45']:
    for q in ['Q1', 'Q2']:
        path = f'q1_due_rule_experiment/union3_{pad}_20k/scored_{q}.csv'
        df = pd.read_csv(path)
        start, end = WINDOWS[q]
        strict_hits = hits[(hits['Service_Date'] >= start) & (hits['Service_Date'] <= end)]
        actual_vins = set(strict_hits['Vin_No'].dropna())
        df['actual'] = df['VIN'].isin(actual_vins).astype(int)

        for thr, label in [(0.5, 'thr 0.50 (shipped)'), (0.70, 'thr 0.70')]:
            flag = (df['prob_turnup'] >= thr).astype(int)
            tp = int(((flag == 1) & (df['actual'] == 1)).sum())
            fp = int(((flag == 1) & (df['actual'] == 0)).sum())
            fn = int(((flag == 0) & (df['actual'] == 1)).sum())
            tn = int(((flag == 0) & (df['actual'] == 0)).sum())
            listed = int(flag.sum())
            precision = tp / listed if listed else float('nan')
            recall = tp / (tp + fn) if (tp + fn) else float('nan')
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float('nan')
            acc = (tp + tn) / len(df)
            rows.append({
                'pad': pad, 'quarter': q, 'threshold': label,
                'rows_in_cohort': len(df), 'actual_positives': int(df['actual'].sum()),
                'listed': listed, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                'precision': round(precision, 4), 'recall': round(recall, 4),
                'f1': round(f1, 4), 'accuracy': round(acc, 4),
            })

out = pd.DataFrame(rows)
out.to_csv('q1_due_rule_experiment/union3_thr070_comparison.csv', index=False)
pd.set_option('display.width', 160)
print(out.to_string(index=False))
