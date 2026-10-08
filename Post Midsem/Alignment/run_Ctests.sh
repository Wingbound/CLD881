#!/usr/bin/env bash
# =====================================================================
# run_Ctests.sh -- oscillating-field droplet morphology (the "C" tests).
# Builds on the same binary as the B tests, just with DC_FIELD=0.
#
#   C1  AC vs DC at matched field amplitude   -> does oscillation change D?
#   C2  one AC run with DENSE output          -> within-cycle microscopics
#                                                (M lag, tip vs equator, stress)
#   C3  frequency sweep omega*tau_B           -> where AC differs most from DC
#
# Output per run:  output_morphology/run_..._AC_<tag>/summary.csv
#   columns: iter,D,ang,bdy,phi_sum,h_iters,h_resid,
#            Happ,Mx_in,My_in,Mtip,Meq,magstress
#   Happ      applied field now (phase reference)
#   Mx_in     mean magnetisation in the drop  (lag vs Happ = the A-effect)
#   Mtip,Meq  |M| at the poles vs the equator (tip>eq drives elongation)
#   magstress mean 0.5 mu0 |M|^2 on the interface (the stress it feels)
#
# WARNING ON SPEED: with an oscillating field the magnetostatic solver
# cannot reuse the previous step's solution, so each step costs ~350 inner
# sweeps (vs 1 for a settled DC run).  AC runs are therefore ~100x slower
# per step than DC.  Step counts below are chosen accordingly; raise them
# for production and expect minutes-to-hours per case.
# =====================================================================
set -u

SRC="phaseField_droplet_Ctests_parallel.cpp"
BIN="./C"
WHICH="${1:-all}"                       # all | c1 | c2 | c3
NJOBS="${NJOBS:-$(nproc 2>/dev/null || echo 4)}"
mkdir -p logs

# shared physics (match the B runs so AC/DC are comparable)
MS="${MS:-0.02}"      # fixed saturation magnetisation (Bo ~ 0.17)
XI0="${XI0:-5.0}"     # Langevin parameter at field peak
DROP_R="${DROP_R:-10}"
FB="${FB:-0.0}"       # pure field (no pressure flow) for clean morphology

echo "Compiling $SRC  (NJOBS=$NJOBS, MS=$MS, XI0=$XI0, R=$DROP_R)"
g++ -O2 -std=c++17 "$SRC" -o "$BIN" || { echo "BUILD FAILED"; exit 1; }
echo "Build OK."

wait_for_slot(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 0.5; done; }

launch(){ # name  KEY=VAL...
  local name="$1"; shift
  wait_for_slot
  echo "  launching  $name   [$*]"
  # shellcheck disable=SC2086
  env MS="$MS" XI0="$XI0" DROP_R="$DROP_R" FB="$FB" DROPLET=1 \
      "$@" RUN_TAG="$name" "$BIN" > "logs/${name}.log" 2>&1 &
}

# =====================================================================
# C1 -- AC vs DC at matched amplitude.  Same everything; only the field
#       mode differs.  Compare the tail-mean D of the two.
# =====================================================================
run_c1(){
  echo "== C1: AC vs DC (match amplitude) =="
  local NS="${NSTEPS:-40000}" OE="${OUTEVERY:-500}"
  launch "c1_DC" DC_FIELD=1 WTB=1.5 NSTEPS="$NS" OUTEVERY="$OE"
  launch "c1_AC" DC_FIELD=0 WTB=1.5 NSTEPS="$NS" OUTEVERY="$OE"
}

# =====================================================================
# C2 -- the microscopic run.  AC field, DENSE output (every ~4 steps) over
#       the converged tail so the within-cycle M-lag, tip/equator split and
#       interface stress are resolved.  One case, the richest data.
#       P = 2*pi*tau_B/(omega*tau_B) ~ 84 steps at wtB=1.5, so OUTEVERY=4
#       gives ~21 samples per field period.
# =====================================================================
run_c2(){
  echo "== C2: microscopic within-cycle (AC, dense) =="
  local NS="${NSTEPS:-40000}" OE="${OUTEVERY:-4}"
  launch "c2_micro" DC_FIELD=0 WTB=1.5 NSTEPS="$NS" OUTEVERY="$OE"
}

# =====================================================================
# C3 -- frequency sweep.  How the AC morphology depends on omega*tau_B:
#       low wtB = quasi-static (follows field, ~DC); high wtB = drop sees
#       only the time-averaged stress.  Max lag/difference in between.
# =====================================================================
run_c3(){
  echo "== C3: frequency sweep (AC) =="
  local NS="${NSTEPS:-40000}" OE="${OUTEVERY:-500}"
  for WTB in 0.1 0.5 1.0 1.5 2.0; do
    launch "c3_wtB${WTB}" DC_FIELD=0 WTB="$WTB" NSTEPS="$NS" OUTEVERY="$OE"
  done
}

case "$WHICH" in
  c1)  run_c1 ;;
  c2)  run_c2 ;;
  c3)  run_c3 ;;
  all) run_c1; run_c2; run_c3 ;;
  *)   echo "unknown target '$WHICH' (use: all|c1|c2|c3)"; exit 1 ;;
esac

echo "All cases launched; waiting..."
wait
echo "============================================================"
echo "DONE.  Per-case final state (tail-mean D printed by the analysis step):"
printf "  %-14s %10s %8s %5s   %s\n" "case" "D(last)" "ang" "bdy" "note"
for f in logs/c*.log; do
  [ -e "$f" ] || continue
  tag=$(basename "$f" .log)
  last=$(grep ' D=.* bdy=' "$f" | tail -1)
  if echo "$last" | grep -qi nan; then
    printf "  %-14s %10s %8s %5s   *** DIVERGED ***\n" "$tag" "-" "-" "-"; continue
  fi
  D=$(echo "$last"|sed -n 's/.* D=\([^ ]*\).*/\1/p')
  ang=$(echo "$last"|sed -n 's/.* ang=\([^ ]*\).*/\1/p')
  bdy=$(echo "$last"|sed -n 's/.* bdy=\([^ ]*\).*/\1/p')
  note=""; [ "$bdy" = 1 ] && note="BOUNDARY CONTACT"
  printf "  %-14s %10s %8s %5s   %s\n" "$tag" "${D:-?}" "${ang:-?}" "${bdy:-?}" "$note"
done
echo "============================================================"
echo "Analyse with:  python3 analyze_C.py"
