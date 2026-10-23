#!/bin/bash
# Causal optimiser reruns (chart-based water routing): Busan full year, then the 6 US ports.
set -e
python3 - <<'P'
import pandas as pd
o=pd.read_csv("data/processed/r2/oof_predictions_r2_dispatch.csv")
f=pd.read_csv("data/processed/r2/dispatch_features_r2.csv",usecols=["event_id","t_dispatch","ship_visible","dispatch_to_meeting_min"])
a=o.merge(f,on="event_id"); a=a[a.ship_visible==1].rename(columns={"t_dispatch":"t_dec","dispatch_to_meeting_min":"lead_min"})
a.to_csv("data/processed/r2/optimiser_anchors_r2_dispatch.csv",index=False); print("anchors",len(a))
P
python3 -W ignore calibrate_r2_gates.py --folds data/processed/r2/oof_predictions_r2_dispatch.csv > logs_r2/calibrate_folds.log 2>&1
rm -f logs_r2/*.causal.done
ls data/processed/ais_master/weeks/*.csv | xargs -P 6 -n 1 ./run_r2_opt_causal.sh > logs_r2/busan_causal_run.log 2>&1
echo "busan done $(date +%H:%M)"
export MP_SET=dispatch
cd multiport
rm -f ../data/processed/multiport/opt_dispatch/*/optimized_*.csv ../data/processed/multiport/opt_dispatch/*/candidates_*.csv
for p in us_galveston us_longbeach us_port_arthur us_capehenry us_swpass us_port_everglades; do
  python3 -W ignore mp_optimise.py calibrate $p >> ../logs_r2/mp_dispatch_calibrate2.log 2>&1
done
for p in us_galveston us_longbeach us_port_arthur us_capehenry us_swpass us_port_everglades; do
  for m in 1 2 3 4 5 6 7 8 9 10 11 12; do echo "$p $m"; done
done | xargs -P 6 -n 2 sh -c 'python3 -W ignore mp_optimise.py run $0 $1 >> ../logs_r2/mp_dispatch_run2.log 2>&1'
python3 -W ignore mp_optimise.py summary > ../logs_r2/mp_dispatch_summary2.log 2>&1
echo "all done $(date +%H:%M)"
