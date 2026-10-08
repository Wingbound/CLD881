#!/usr/bin/env bash
# =====================================================================
# run_Etests.sh -- extended oscillating-field droplet morphology.
# One binary, many cases, all in parallel (job-throttled to NJOBS).
#
#   E1  amplitude x frequency map   D(xi0, omega*tau_B)   [AC]
#   E4  rotating field              |H| const, rotates     [ROT]
#   E5  DC + AC bias                steady + oscillation    [BIAS]
#   E6  AC + Poiseuille flow        oscillation + shear     [AC, FB>0]
#   (E2/E3 are post-processing of the dense E1 centre case + C2 -> analyze_E.py)
#
# Each run writes output_morphology/run_..._<MODE>_<tag>/summary.csv
#   columns: iter,D,ang,bdy,phi_sum,h_iters,h_resid,
#            Happ,Happ_y,Mx_in,My_in,Mtip,Meq,magstress
#
# SPEED: any oscillating field (AC/ROT/BIAS) restarts the magnetostatic
# solver every step (~350 inner sweeps/step), so these are ~100x slower
# than a settled DC run.  Step counts are modest here; raise for production.
#
# NOTE: the rotating field (E4) uses the reformulated solver -- the uniform
# applied H_y is imposed analytically on top of the periodic self-field.
# Verified: DC/AC cases are byte-identical to the old formulation.
# =====================================================================
set -u

SRC="phaseField_droplet_Etests_parallel.cpp"
BIN="./E"
WHICH="${1:-all}"                       # all | e1 | e4 | e5 | e6
NJOBS="${NJOBS:-$(nproc 2>/dev/null || echo 4)}"
mkdir -p logs

# shared physics (match the B/C runs)
MS="${MS:-0.02}"; XI0="${XI0:-5.0}"; DROP_R="${DROP_R:-10}"
NSTEPS="${NSTEPS:-40000}"; OUTEVERY="${OUTEVERY:-500}"

echo "Compiling $SRC  (NJOBS=$NJOBS, MS=$MS, R=$DROP_R, NSTEPS=$NSTEPS)"
g++ -O2 -std=c++17 "$SRC" -o "$BIN" || { echo "BUILD FAILED"; exit 1; }
echo "Build OK."

wait_for_slot(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 0.5; done; }

launch(){ # name  KEY=VAL...
  local name="$1"; shift
  wait_for_slot
  echo "  launching  $name   [$*]"
  # shellcheck disable=SC2086
  env MS="$MS" DROP_R="$DROP_R" DROPLET=1 NSTEPS="$NSTEPS" OUTEVERY="$OUTEVERY" \
      "$@" RUN_TAG="$name" "$BIN" > "logs/${name}.log" 2>&1 &
}

# =====================================================================
# E1 -- amplitude x frequency map.  AC field; sweep xi0 and omega*tau_B.
#       The 2D surface D(xi0, wtB) is the headline AC-morphology figure.
#       One centre case (xi0=5, wtB=1.5) is run DENSE for E2/E3 harmonics.
# =====================================================================
run_e1(){
  echo "== E1: amplitude x frequency map (AC) =="
  for XI in 2 5 10; do
    for WT in 0.5 1.0 1.5 2.0; do
      local OE="$OUTEVERY"
      # dense output on the centre case so E2/E3 can resolve harmonics
      [ "$XI" = 5 ] && [ "$WT" = 1.5 ] && OE=4
      launch "e1_xi${XI}_wtB${WT}" FMODE=AC XI0="$XI" WTB="$WT" FB=0 OUTEVERY="$OE"
    done
  done
}

# =====================================================================
# E4 -- rotating field.  |H| constant, direction rotates at omega.
#       Sweep frequency; watch the orientation 'ang' track the field.
#
#       IMPORTANT: a rotating field drives a BULK FLOW (net magnetic torque
#       spins the fluid) that the oscillating/steady cases do not. At the
#       Ms=0.02 used elsewhere this flow exceeds the LBM Mach limit
#       (|u|~0.27 c_s) and the run diverges to NaN.  E4 therefore uses a
#       WEAKER field, MS_ROT (default 0.005), which keeps |u| well below the
#       stability limit.  Raise MS_ROT only while watching for divergence.
# =====================================================================
run_e4(){
  local MSR="${MS_ROT:-0.005}"
  echo "== E4: rotating field (ROT, Ms=$MSR) =="
  for WT in 0.5 1.0 1.5 2.0; do
    wait_for_slot
    echo "  launching  e4_wtB${WT}   [FMODE=ROT MS=$MSR WTB=$WT]"
    env MS="$MSR" DROP_R="$DROP_R" DROPLET=1 NSTEPS="$NSTEPS" OUTEVERY="$OUTEVERY" \
        FMODE=ROT XI0="$XI0" WTB="$WT" FB=0 RUN_TAG="e4_wtB${WT}" \
        "$BIN" > "logs/e4_wtB${WT}.log" 2>&1 &
  done
}

# =====================================================================
# E5 -- DC + AC bias.  H = H0 ( b + (1-b) cos wt ).  Sweep the DC fraction
#       b: b=1 is pure DC, b=0 is pure AC.  Shows how a pre-stretched drop
#       responds to a superposed oscillation.
# =====================================================================
run_e5(){
  echo "== E5: DC + AC bias (BIAS) =="
  for B in 0.2 0.4 0.6 0.8; do
    launch "e5_bias${B}" FMODE=BIAS BIAS="$B" XI0="$XI0" WTB=1.5 FB=0
  done
}

# =====================================================================
# E6 -- AC field + Poiseuille flow.  Oscillating field competing with
#       steady shear.  Sweep the body force at fixed AC field.
# =====================================================================
run_e6(){
  echo "== E6: AC + flow (AC, FB>0) =="
  for FBV in 1e-7 5e-7 1e-6 2e-6; do
    launch "e6_Fb${FBV}" FMODE=AC XI0="$XI0" WTB=1.5 FB="$FBV"
  done
}

case "$WHICH" in
  e1) run_e1 ;;
  e4) run_e4 ;;
  e5) run_e5 ;;
  e6) run_e6 ;;
  all) run_e1; run_e4; run_e5; run_e6 ;;
  *) echo "unknown target '$WHICH' (use: all|e1|e4|e5|e6)"; exit 1 ;;
esac

echo "All cases launched; waiting..."
wait
echo "============================================================"
echo "DONE.  Per-case final state:"
printf "  %-16s %10s %8s %5s   %s\n" "case" "D(last)" "ang" "bdy" "note"
for f in logs/e*.log; do
  [ -e "$f" ] || continue
  tag=$(basename "$f" .log)
  last=$(grep ' D=.* bdy=' "$f" | tail -1)
  if echo "$last" | grep -qi nan; then
    printf "  %-16s %10s %8s %5s   *** DIVERGED ***\n" "$tag" "-" "-" "-"; continue
  fi
  D=$(echo "$last"|sed -n 's/.* D=\([^ ]*\).*/\1/p')
  ang=$(echo "$last"|sed -n 's/.* ang=\([^ ]*\).*/\1/p')
  bdy=$(echo "$last"|sed -n 's/.* bdy=\([^ ]*\).*/\1/p')
  note=""; [ "$bdy" = 1 ] && note="BOUNDARY CONTACT"
  printf "  %-16s %10s %8s %5s   %s\n" "$tag" "${D:-?}" "${ang:-?}" "${bdy:-?}" "$note"
done
echo "============================================================"
echo "Analyse with:  python3 analyze_E.py"
