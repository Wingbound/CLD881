#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Droplet-deformation validation sweep (Hu, Li & Niu, PRE 98, 033301, Sec. III B).
# Runs all 5 applied-field cases and collects the final CSV of each into sweep/.
#
# Field scaling:  H_lat = H_phys * SCALE, with SCALE fixed so that the strongest
# field (5.5 kA/m) gives Bo_m ~ 5 and xi_e < 0.1 (linear-chi limit).  Only the
# ratios of the H_lat values carry physical meaning (Bo_m matching); the common
# scale is the single calibration knob -- if the whole b/a curve is off by a
# constant factor, adjust SCALE and rerun.
# ---------------------------------------------------------------------------
set -euo pipefail

BIN=./drop                       # compiled droplet solver
OUTROOT=output_linear_field      # solver's own output root (fixed in the .cpp)
SWEEP=sweep                      # where we collect the labelled results
SCALE=0.004243                   # 0.006/sqrt(2), holds Bo_m fixed at R=25

# physical fields (kA/m) from the paper's Fig. 4 / Fig. 5
H_PHYS=(1.2 2.4 2.9 3.7 5.5)

mkdir -p "$SWEEP"

# the solver writes to a fixed subfolder name (independent of H_lat), so each run
# overwrites the previous one's CSVs -> copy out immediately after each run.
SUBDIR="$OUTROOT/output_xi0_20.000000_tauB_20.000000_H0_0.000000_wtB_0.000000"

# last output index = nSteps (must match nSteps/outputInterval in the .cpp)
NSTEPS=60000

echo "idx  H_phys(kA/m)  H_lat        Bo_m"
for idx in "${!H_PHYS[@]}"; do
  hp="${H_PHYS[$idx]}"
  hl=$(python3 -c "print(f'{$hp*$SCALE:.8f}')")
  n=$((idx+1))
  echo "----- case $n : H_phys=$hp kA/m  H_lat=$hl -----"
  "$BIN" "$hl"
  # collect the final-timestep CSV plus a couple of intermediates for the transient
  cp "$SUBDIR/output_t_${NSTEPS}.csv" "$SWEEP/field_${n}_final.csv"
  # keep the full time series too (rename with the case index)
  for f in "$SUBDIR"/output_t_*.csv; do
    t=$(basename "$f" | sed 's/output_t_//; s/.csv//')
    cp "$f" "$SWEEP/field_${n}_t_${t}.csv"
  done
done

# write a small manifest the notebook reads (case index -> H_phys, H_lat)
python3 - "$SCALE" <<'PY' > "$SWEEP/manifest.csv"
import sys
scale=float(sys.argv[1])
print("case,H_phys_kAm,H_lat")
for i,hp in enumerate([1.2,2.4,2.9,3.7,5.5],start=1):
    print(f"{i},{hp},{hp*scale:.8f}")
PY

echo
echo "Done. Collected results in $SWEEP/ :"
ls -la "$SWEEP"/field_*_final.csv "$SWEEP"/manifest.csv
