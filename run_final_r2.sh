#!/bin/bash
# Final R2 rerun: (1) dispatch-window sensitivity (search from the previous session only, no T bound);
# (2) ship-independent fold gates and the Busan + US optimisers.
set -e
export DYLD_FALLBACK_LIBRARY_PATH=$(python3 -c "import sklearn,os;print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))")
DISPATCH_WINDOW=prev python3 -W ignore build_r2_dispatch_features.py > logs_r2/final_prevwin_features.log 2>&1
FEATURE_SET=dispatch_prevwin python3 -W ignore revision_r2_validation.py > logs_r2/final_prevwin_validation.log 2>&1
echo "sensitivity done $(date +%H:%M)"
./run_causal_all.sh
echo "final done $(date +%H:%M)"
