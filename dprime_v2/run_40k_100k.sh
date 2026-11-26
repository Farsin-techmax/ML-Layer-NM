#!/usr/bin/env bash
# D' v2 pipeline for 40/50/60/90/100k, run end to end in order. Each step logs to dprime_v2/run_40k_100k/<step>.log;
# the run stops at the first failing step (see run_40k_100k/STATUS). Run from repo root:
#   bash dprime_v2/run_40k_100k.sh [pid-to-wait-for]
cd "$(dirname "$0")/.." || exit 1
M="40 50 60 90 100"
PY="./venv/Scripts/python.exe -u"
export PYTHONPATH=.
L=dprime_v2/run_40k_100k
mkdir -p "$L"
status() { echo "$(date '+%F %T') $*" >> "$L/STATUS"; }

if [ -n "$1" ]; then
  status "waiting for pid $1 (build_lists exact)"
  while ps -p "$1" > /dev/null 2>&1; do sleep 30; done
fi
for f in 40 50 60 90 100; do
  [ -f "dprime_v2/lists_${f}k.csv" ] || { status "FAIL lists_${f}k.csv missing after build_lists"; exit 1; }
done

step() {   # step <name> <command...>
  local name=$1; shift
  status "start $name"
  if $PY "$@" > "$L/$name.log" 2>&1; then status "done  $name"; else status "FAIL  $name (see $L/$name.log)"; exit 1; fi
}

step build_lists_pad45      dprime_v2/build_lists.py $M --pad 45
step build_matrix           dprime_v2/build_matrix.py $M
step build_matrix_pad45     dprime_v2/build_matrix.py $M --pad 45
step leak                   dprime_v2/test_no_label_leak.py $M
step leak_pad45             dprime_v2/test_no_label_leak.py $M --set v2_pad45
step train                  dprime_v2/train_models.py $M
step train_pad45            dprime_v2/train_models.py $M --set v2_pad45
step production_scores      q1_due_rule_experiment/q3_dprime_model_threshold.py $M
step evaluate_q3            dprime_v2/evaluate_q3.py $M
step best_model_confusion   dprime_v2/best_model_confusion.py $M
step stability_report       dprime_v2/stability_report.py $M
step list_padding_ceiling   dprime_v2/list_padding_ceiling.py $M
step eval_pad45             dprime_v2/eval_pad45.py $M
step feature_importance     dprime_v2/feature_importance.py $M
step success_window         dprime_v2/success_window.py $M
step success_window_importance dprime_v2/success_window_importance.py $M
step final_q3_report        dprime_v2/final_q3_report.py $M
status "ALL DONE"
