#!/usr/bin/env bash
# =====================================================================
# run_Btests.sh -- compile phaseField_droplet_Btests_parallel.cpp once,
# then run all B-test cases CONCURRENTLY, each in its own output folder
# and with its own log.  No GNU parallel needed; uses a simple job
# throttle so you never oversubscribe the machine.
#
# Usage:
#   ./run_Btests.sh                 # run everything, NJOBS = #cores
#   NJOBS=4 ./run_Btests.sh         # cap at 4 concurrent sims
#   ./run_Btests.sh b1              # run only the B1 sweep (b1|b2|b3|all)
#
# Each case writes to:  output_morphology/run_MS..._FB..._.../output_t_*.csv
# and logs to:          logs/<case>.log
# =====================================================================
set -u

SRC="phaseField_droplet_Btests_parallel.cpp"
BIN="./Bpar"
WHICH="${1:-all}"                       # all | b1 | b2 | b3
NJOBS="${NJOBS:-$(nproc 2>/dev/null || echo 4)}"
NSTEPS="${NSTEPS:-80000}"               # per run; override for quick tests
OUTEVERY="${OUTEVERY:-2000}"
mkdir -p logs

# ---- compile once ----------------------------------------------------
echo "Compiling $SRC  (NJOBS=$NJOBS, NSTEPS=$NSTEPS)"
g++ -O2 -std=c++17 "$SRC" -o "$BIN" || { echo "BUILD FAILED"; exit 1; }
echo "Build OK."

# ---- job throttle: block until a slot is free ------------------------
wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do
    wait -n 2>/dev/null || sleep 0.5    # wait -n: return when any job ends
  done
}

# ---- launch one case: name + KEY=VAL env assignments -----------------
launch() {
  local name="$1"; shift
  wait_for_slot
  echo "  launching  $name   [$*]"
  # shellcheck disable=SC2086
  env "$@" NSTEPS="$NSTEPS" OUTEVERY="$OUTEVERY" RUN_TAG="$name" \
      "$BIN" > "logs/${name}.log" 2>&1 &
}

# =====================================================================
# B1 -- deformation vs field strength (pure field, no flow).
#       Sweep Ms; expect D to grow with Bo_mag ~ Ms^2.  Keep Ms < ~0.05.
# =====================================================================
run_b1() {
  echo "== B1: deformation vs field (FB=0, DC) =="
  for MS in 0.005 0.010 0.015 0.020 0.025 0.030 0.035 0.040; do
    launch "b1_Ms${MS}" MS="$MS" FB=0.0 DC_FIELD=1 XI0=5.0
  done
}

# =====================================================================
# B2 -- morphology under combined field + shear.
#       Fixed moderate Ms, sweep the body force (flow strength).
#       Watch the long-axis tilt angle 'ang' in each log.
# =====================================================================
run_b2() {
  echo "== B2: field + pressure-driven flow (DC) =="
  local MS=0.020
  for FB in 1e-7 2e-7 5e-7 1e-6 2e-6; do
    launch "b2_Fb${FB}" MS="$MS" FB="$FB" DC_FIELD=1 XI0=5.0
  done
}

# =====================================================================
# B3 -- breakup threshold: push Ms up until the drop pinches.
#       The last Ms that still reaches a steady D is the threshold.
#       These may run longer / go unstable -- that IS the result.
# =====================================================================
run_b3() {
  echo "== B3: breakup sweep (FB=0, DC, high field) =="
  for MS in 0.040 0.050 0.060 0.070 0.080; do
    launch "b3_Ms${MS}" MS="$MS" FB=0.0 DC_FIELD=1 XI0=5.0
  done
}

# ---- dispatch --------------------------------------------------------
case "$WHICH" in
  b1)  run_b1 ;;
  b2)  run_b2 ;;
  b3)  run_b3 ;;
  all) run_b1; run_b2; run_b3 ;;
  *)   echo "unknown target '$WHICH' (use: all|b1|b2|b3)"; exit 1 ;;
esac

echo "All cases launched; waiting for completion..."
wait
echo "============================================================"
echo "DONE.  Per-case final state:"
printf "  %-16s %10s %8s %5s   %s\n" "case" "D" "ang" "bdy" "note"
for f in logs/*.log; do
  tag=$(basename "$f" .log)
  last=$(grep ' D=.* bdy=' "$f" | tail -1)     # robust key=value diag line
  if echo "$last" | grep -qi 'nan'; then
    printf "  %-16s %10s %8s %5s   *** DIVERGED (NaN) -- drop shredded ***\n" "$tag" "-" "-" "-"
    continue
  fi
  D=$(echo "$last"   | sed -n 's/.* D=\([^ ]*\).*/\1/p')
  ang=$(echo "$last" | sed -n 's/.* ang=\([^ ]*\).*/\1/p')
  bdy=$(echo "$last" | sed -n 's/.* bdy=\([^ ]*\).*/\1/p')
  note=""
  [ "$bdy" = "1" ] && note="BOUNDARY CONTACT -- D is a finite-size artifact, not physics"
  printf "  %-16s %10s %8s %5s   %s\n" "$tag" "${D:-?}" "${ang:-?}" "${bdy:-?}" "$note"
done
echo "============================================================"
echo "Per-run time series: output_morphology/<run_...>/summary.csv  (iter,D,ang,bdy,...)"
echo "Full fields:          output_morphology/<run_...>/output_t_*.csv"
echo
echo "NOTE on B3 (breakup): with R=10 in a ${NSTEPS}-step 50x50 box the drop"
echo "reaches the walls (bdy=1) before true pinch-off.  For a real breakup"
echo "threshold, rerun B3 with a bigger box (edit nX,nY) or smaller DROP_R,"
echo "e.g.  DROP_R=7 ./run_Btests.sh b3 ."
