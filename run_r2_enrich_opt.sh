#!/bin/bash
# RESS R2 rebuild, stage 2: re-enrich + re-optimise a week whose detection already exists
# (rerun after the KST tide / night-flag fix). Marker file makes it resumable.
set -e
M="$1"; W=$(basename "$M" .csv); T="${W//-/_}_r2verified"
DONE=logs_r2/${T}.kstfix.done
[ -f "$DONE" ] && { echo "skip $W"; exit 0; }
E=data/processed/interactions/interceptions_${T}.csv
C=data/processed/interactions/event_context_enriched_${T}.csv
L=logs_r2/${T}.kstfix.log
for i in 1 2 3; do
  python3 -W ignore -m src.core.enrich_event_context --events "$E" --master "$M" --tag "$T" > "$L" 2>&1 \
  && python3 -W ignore -m src.core.optimize_meeting_point --events "$C" --master "$M" --tag "$T" >> "$L" 2>&1 \
  && { touch "$DONE"; echo "done $W $(date +%H:%M)"; exit 0; }
  echo "retry $i $W"; sleep 30
done
echo "FAIL $W"; exit 1
