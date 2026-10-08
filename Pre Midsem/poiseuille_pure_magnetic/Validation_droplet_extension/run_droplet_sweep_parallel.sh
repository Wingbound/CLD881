#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Droplet-deformation validation sweep -- PARALLEL version.
# Runs all 5 applied-field cases at once (each in its own output directory so
# concurrent runs never share files), then collects results into sweep/.
#
# Each solver process is single-threaded, so N cases run on N cores.  MAXJOBS
# caps concurrency for machines with fewer than 5 cores; set it to 5 (or leave
# the default) to run all five simultaneously.
# ---------------------------------------------------------------------------
set -euo pipefail

BIN=./drop
OUTROOT=output_linear_field
SWEEP=sweep
SCALE=0.004243                   # = 0.006/sqrt(2): holds Bo_m fixed after R 12.5->25
                                 # so the deformation physics is unchanged; only the
                                 # interface resolution (a/D) improves. H_lat(5.5)=0.0233.
NSTEPS=60000                     # must match nSteps in the .cpp

# concurrency cap.  Default: run ALL cases at once (the only speedup a bash-only
# change can give -- per-case runtime lives in the .cpp and is untouched here).
# Override with e.g.  MAXJOBS=3 ./run_droplet_sweep_parallel.sh  on a small machine.
NCORES=$(nproc 2>/dev/null || echo 4)
NCASES=5
MAXJOBS=${MAXJOBS:-$NCASES}

# --- resource sanity: 5 concurrent cases each hold a few hundred MB. If there
#     isn't enough free RAM, the OS swaps and the "parallel" run is slower than
#     serial. Warn (don't block) so you can lower MAXJOBS if needed.
if command -v free >/dev/null 2>&1; then
  FREE_GB=$(free -g | awk '/^Mem:/{print $7}')   # available GB
  NEED_GB=$(( MAXJOBS ))                          # ~<=1 GB per case, conservative
  if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt "$NEED_GB" ]; then
    echo "WARNING: only ${FREE_GB} GB RAM available for $MAXJOBS parallel cases."
    echo "         If the run thrashes/swaps, rerun with a lower MAXJOBS, e.g.:"
    echo "         MAXJOBS=$(( FREE_GB > 0 ? FREE_GB : 2 )) $0"
  fi
fi
if [ "$NCORES" -lt "$MAXJOBS" ]; then
  echo "NOTE: $NCORES cores < $MAXJOBS requested jobs -> cases will time-share cores,"
  echo "      so the sweep won't get faster than $NCORES at a time. (Per-case speed"
  echo "      is fixed in the .cpp and unchanged.)"
fi

H_PHYS=(1.2 2.4 2.9 3.7 5.5)

mkdir -p "$SWEEP"
echo "Launching ${#H_PHYS[@]} cases, up to $MAXJOBS in parallel (cores: $NCORES)."

pids=()
run_case() {
  local n="$1" hp="$2" hl="$3"
  echo "  [case $n] H_phys=$hp kA/m  H_lat=$hl  -> logs/case_${n}.log"
  # label = case index -> writes to output_linear_field/case_<n>/
  "$BIN" "$hl" "$n" > "logs/case_${n}.log" 2>&1
}

mkdir -p logs

# throttle to MAXJOBS concurrent jobs
active=0
for idx in "${!H_PHYS[@]}"; do
  hp="${H_PHYS[$idx]}"
  n=$((idx+1))
  hl=$(python3 -c "print(f'{$hp*$SCALE:.8f}')")
  run_case "$n" "$hp" "$hl" &
  pids+=($!)
  active=$((active+1))
  if (( active >= MAXJOBS )); then
    wait -n            # wait for any one job to finish before launching more
    active=$((active-1))
  fi
done

# wait for all remaining jobs
wait
echo "All cases finished."

# collect results (each case has its own dir -> no clobbering)
python3 - "$SCALE" <<'PY' > "$SWEEP/manifest.csv"
import sys
scale=float(sys.argv[1])
print("case,H_phys_kAm,H_lat")
for i,hp in enumerate([1.2,2.4,2.9,3.7,5.5],start=1):
    print(f"{i},{hp},{hp*scale:.8f}")
PY

for idx in "${!H_PHYS[@]}"; do
  n=$((idx+1))
  SUBDIR="$OUTROOT/case_${n}"
  cp "$SUBDIR/output_t_${NSTEPS}.csv" "$SWEEP/field_${n}_final.csv"
  for f in "$SUBDIR"/output_t_*.csv; do
    t=$(basename "$f" | sed 's/output_t_//; s/.csv//')
    cp "$f" "$SWEEP/field_${n}_t_${t}.csv"
  done
done

echo
echo "Done. Collected results in $SWEEP/ :"
ls -la "$SWEEP"/field_*_final.csv "$SWEEP"/manifest.csv
