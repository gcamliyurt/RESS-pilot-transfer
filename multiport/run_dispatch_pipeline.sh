#!/bin/bash
# Multi-port causal pipeline (pilot-dispatch decision): validation, OOF anchors, fold-calibrated gates,
# US optimiser (UKC gate off), summary. Run from pilot_boat_analysis/ after build_mp_dispatch_features.py.
set -e
export MP_SET=dispatch
export DYLD_FALLBACK_LIBRARY_PATH=$(python3 -c "import sklearn,os;print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))")
cd multiport
python3 -W ignore mp_validation.py > ../logs_r2/mp_dispatch_validation.log 2>&1
python3 -W ignore mp_oof.py > ../logs_r2/mp_dispatch_oof.log 2>&1
for p in us_galveston us_longbeach us_port_arthur us_capehenry us_swpass us_port_everglades; do
  python3 -W ignore mp_optimise.py calibrate $p >> ../logs_r2/mp_dispatch_calibrate.log 2>&1
done
for p in us_galveston us_longbeach us_port_arthur us_capehenry us_swpass us_port_everglades; do
  for m in 1 2 3 4 5 6 7 8 9 10 11 12; do echo "$p $m"; done
done | xargs -P 6 -n 2 sh -c 'python3 -W ignore mp_optimise.py run $0 $1 >> ../logs_r2/mp_dispatch_run.log 2>&1'
python3 -W ignore mp_optimise.py summary > ../logs_r2/mp_dispatch_summary.log 2>&1
echo pipeline done
