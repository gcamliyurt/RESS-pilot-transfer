# Decision-time pilot transfer point prediction and safety-constrained relocation

Code for the revised manuscript JRESS-D-26-03347 (Reliability Engineering & System Safety).

## What it does
1. Detects pilot-boat / ship transfer sessions in AIS for the 13 verified Busan pilot boats
   (`src/core/detect_interceptions.py`) and confirms them against the Busan Harbor Pilots'
   Association pilotage log (`scrape_busanpilot_log.py`, `build_r2_dataset.py`).
2. Builds decision-time inputs at pilot-boat dispatch, using only observations at or before
   that moment (`build_r2_dispatch_features.py`; every input timestamp is asserted <= dispatch).
3. Validates the meeting-distance models with ship-grouped and temporal partitions and
   ship-clustered bootstrap intervals (`revision_r2_validation.py`, `FEATURE_SET=dispatch`).
4. Relocates the out-of-fold predicted point under hard safety gates calibrated within the
   training folds, with chart-based (water-only) routes and no use of future AIS
   (`calibrate_r2_gates.py --folds`, `src/core/optimize_meeting_point.py`, `run_r2_opt_causal.sh`).
5. Repeats prediction at 11 US and Danish stations and the optimisation at 6 US stations on
   NOAA charts (`multiport/`).

## Data (not included)
- AIS: Busan New Port (supplied under licence); MarineCadastre (US, public); Danish Maritime
  Authority (public).
- Metocean: KMA observations (supplied under licence by GCS Solution Co., Ltd.); ERA5 via Open-Meteo.
- Charts: KHOA ENC cells (licensed); NOAA ENC (public).
- Pilotage log: Busan Harbor Pilots' Association website (public).
The scripts expect these under `data/raw/` as described at the top of each script.

## Verified pilot boats (Busan New Port)
See `verified_pilot_mmsi.txt`.

## Environment
Python 3.12; see `requirements.txt`. On macOS, XGBoost can use scikit-learn's OpenMP runtime:
`export DYLD_FALLBACK_LIBRARY_PATH=$(python3 -c "import sklearn,os;print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))")`.
