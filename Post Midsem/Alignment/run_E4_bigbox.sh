#!/usr/bin/env bash
# =====================================================================
# run_E4_bigbox.sh -- rotating-field (E4) sweep in a LARGE 100x100 box.
#
# WHY A SEPARATE SCRIPT: a rotating field makes the drop wander, so in the
# 50x50 box it walls out (bdy=1) or diverges at higher frequency. The box
# must be enlarged -- and nX,nY are COMPILE-TIME constants, so this uses a
# separately-built binary (E_big, nX=nY=100). Verified: the omega*tau_B=2.0
# case that diverged at 50x50 runs clean (bdy=0) here.
#
# COST: ~4x the cells => ~4x slower per step, on top of the oscillating-field
# slowdown. These are cluster runs -- budget tens of minutes to hours each.
#
# Output: output_morphology/run_MS..._R15_..._ROT_<tag>/summary.csv
#         (R=15 keeps the drop ~35 cells clear of every wall)
# =====================================================================
set -u

SRC="phaseField_droplet_Etests_bigbox.cpp"   # the nX=nY=100 source
BIN="./E_big"
NJOBS="${NJOBS:-$(nproc 2>/dev/null || echo 4)}"
MS="${MS_ROT:-0.005}"       # rotating field drives flow -> keep Ms small
DROP_R="${DROP_R:-15}"      # big drop, but 35 cells of wall clearance
XI0="${XI0:-5.0}"
NSTEPS="${NSTEPS:-60000}"   # needs many rotation periods to reach a limit cycle
OUTEVERY="${OUTEVERY:-500}"
mkdir -p logs

echo "Compiling $SRC  (100x100 box, NJOBS=$NJOBS, Ms=$MS, R=$DROP_R)"
g++ -O2 -std=c++17 "$SRC" -o "$BIN" || { echo "BUILD FAILED"; exit 1; }
echo "Build OK."

wait_for_slot(){ while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || sleep 0.5; done; }

echo "== E4 (big box): rotating field, Ms=$MS, R=$DROP_R in 100x100 =="
for WT in 0.5 1.0 1.5 2.0; do
  wait_for_slot
  echo "  launching  e4big_wtB${WT}"
  env MS="$MS" DROP_R="$DROP_R" DROPLET=1 NSTEPS="$NSTEPS" OUTEVERY="$OUTEVERY" \
      FMODE=ROT XI0="$XI0" WTB="$WT" FB=0 RUN_TAG="e4big_wtB${WT}" \
      "$BIN" > "logs/e4big_wtB${WT}.log" 2>&1 &
done

echo "All launched; waiting..."
wait
echo "============================================================"
printf "  %-16s %10s %8s %5s   %s\n" "case" "D(last)" "ang" "bdy" "note"
for f in logs/e4big_*.log; do
  [ -e "$f" ] || continue
  tag=$(basename "$f" .log)
  last=$(grep ' D=.* bdy=' "$f" | tail -1)
  if echo "$last" | grep -qi nan; then
    printf "  %-16s %10s %8s %5s   *** DIVERGED ***\n" "$tag" - - -; continue
  fi
  D=$(echo "$last"|sed -n 's/.* D=\([^ ]*\).*/\1/p')
  ang=$(echo "$last"|sed -n 's/.* ang=\([^ ]*\).*/\1/p')
  bdy=$(echo "$last"|sed -n 's/.* bdy=\([^ ]*\).*/\1/p')
  note=""; [ "$bdy" = 1 ] && note="BOUNDARY CONTACT (box still too small -- raise nX,nY)"
  printf "  %-16s %10s %8s %5s   %s\n" "$tag" "${D:-?}" "${ang:-?}" "${bdy:-?}" "$note"
done
echo "============================================================"
echo "Analyse with:  python3 analyze_E.py   (reads these alongside the 50x50 runs)"
