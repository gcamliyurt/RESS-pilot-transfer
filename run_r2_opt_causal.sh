#!/bin/bash
# RESS R2 (causal): optimiser at pilot-boat dispatch, past-only AIS, fold-calibrated gates. Resumable.
M="$1"; W=$(basename "$M" .csv); T="${W//-/_}_r2verified"
DONE=logs_r2/${T}.causal.done
[ -f "$DONE" ] && exit 0
python3 -W ignore -m src.core.optimize_meeting_point --events data/processed/interactions/event_context_enriched_${T}.csv \
  --master "$M" --tag "${T}_causal" --anchors data/processed/r2/optimiser_anchors_r2_dispatch.csv \
  --fold-gates data/processed/r2/gate_thresholds_folds.json --dump-candidates \
  > logs_r2/${T}.causal.log 2>&1 && { touch "$DONE"; echo "done $W"; exit 0; }
echo "FAIL $W"
