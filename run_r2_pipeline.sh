#!/bin/bash
# RESS R2 rebuild: verified pilot fleet, one AIS week per call.
# Usage: ls data/processed/ais_master/weeks/*.csv | xargs -P 5 -n 1 ./run_r2_pipeline.sh
set -e
M="$1"; W=$(basename "$M" .csv); T="${W//-/_}_r2verified"
OUT=data/processed/meeting_point/optimized_meeting_points_${T}.csv
[ -f "$OUT" ] && { echo "skip $W"; exit 0; }
E=data/processed/interactions/interceptions_${T}.csv
C=data/processed/interactions/event_context_enriched_${T}.csv
L=logs_r2/${T}.log
python3 -W ignore -m src.core.detect_interceptions --master "$M" --pilot-fleet verified --tag "$T" > "$L" 2>&1
python3 -W ignore -m src.core.extract_event_trajectories --events "$E" --master "$M" --tag "$T" >> "$L" 2>&1
python3 -W ignore -m src.core.enrich_event_context --events "$E" --master "$M" --tag "$T" >> "$L" 2>&1
python3 -W ignore -m src.core.optimize_meeting_point --events "$C" --master "$M" --tag "$T" >> "$L" 2>&1
echo "done $W $(date +%H:%M)"
