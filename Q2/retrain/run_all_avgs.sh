#!/usr/bin/env bash
# Replicate the 30k average-interval retrain to the other eight milestones, in order:
# lists -> features -> leak check (stops on an A/B failure) -> train -> head-to-head vs production.
set -e
cd /c/Techmax/cwf/code
export PYTHONPATH=.
PY=./venv/Scripts/python.exe
MS="20 40 50 60 70 80 90 100"
$PY Q2/retrain/build_lists_avgs.py $MS        > Q2/retrain/build_lists_avgs_rest.log 2>&1
$PY Q2/retrain/build_matrix_avgs.py $MS       > Q2/retrain/build_matrix_avgs_rest.log 2>&1
$PY Q2/retrain/test_no_label_leak_avgs.py $MS > Q2/retrain/leak_avgs_rest.log 2>&1
$PY Q2/retrain/train_avgs.py $MS              > Q2/retrain/train_avgs_rest.log 2>&1
for m in $MS; do
  $PY Q2/retrain/evaluate_q2_avgs.py $m > Q2/retrain/evaluate_q2_avgs_${m}k.log 2>&1
done
echo ALL_DONE
