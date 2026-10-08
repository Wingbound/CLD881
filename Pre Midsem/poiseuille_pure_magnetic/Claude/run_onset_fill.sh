#!/usr/bin/env bash
# Run ONLY the grid cells missing for the onset figure, in parallel.
# Onset is a fluid property (read from g_ref_nodemag), so controls are not needed:
# each cell is one kelvin run.  Nothing you already have is recomputed.
#
#   ./run_onset_fill.sh
#
# Needs, in this directory: phaseField_poiseuille_linearField_AC.cpp, sweep.py
# (uses sweep.py's patch step to build the solver, then calls the binary directly).
set -euo pipefail
cd "$(dirname "$0")"

SRC=phaseField_poiseuille_linearField_AC.cpp
WORK=onset_build
LOGS=onset_logs
RUNS=onset_runs
NJOBS=12                 # parallel simulations (you said 12-14 is safe on 16 cores)
NSTEPS=120000
NX=50; NY=50; NSUBH=64
H0=1.73; R=0.3           # bias H_d/H0; G_target auto-scales to hold kL/H_d <= 0.02
GTARGET=2e-7

# ---- the 20 missing (xi0, omega*tau_B) cells ----
CELLS=(
  "1 1.6"   "1 1.7"
  "2.5 1.6" "2.5 1.7"
  "5 1.7"
  "7.5 1.6" "7.5 1.7"
  "10 1.7"
  "12.5 1.6" "12.5 1.7"
  "15 1.7"
  "17.5 0.1" "17.5 0.5" "17.5 1.0" "17.5 1.4" "17.5 1.5" "17.5 1.6" "17.5 1.7" "17.5 2.0"
  "20 1.7"
)

mkdir -p "$WORK" "$LOGS" "$RUNS"

# build the solver once (patch adds command-line args, averaging, in-code g fit)
python3 - "$SRC" "$WORK/solver.cpp" <<'PY'
import sys
# reuse sweep.py's EDITS + size substitutions to produce a runnable solver
import importlib.util
spec = importlib.util.spec_from_file_location("sweep", "sweep.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.patch(sys.argv[1], sys.argv[2], nsteps=int("120000"), nx=50, ny=50, nsubh=64)
print("patched ->", sys.argv[2])
PY
g++ -O3 -std=c++17 -o "$WORK/solver" "$WORK/solver.cpp"
echo "built $WORK/solver"

HD=$(python3 -c "print($R*$H0)")

run_cell () {                       # run_cell "<xi0> <wtb>"
  set -- $1
  local xi0=$1 wtb=$2
  local taub; taub=$(python3 -c "print(max(20.0, 10.0*$wtb))")
  local tag="xi0_${xi0}_wtB_${wtb}"
  mkdir -p "$RUNS/$tag"
  # args: xi0 tauB wtb H_d H0 Fb(=0 -> kelvin) G_target
  ( cd "$RUNS/$tag" && "$OLDPWD/$WORK/solver" "$xi0" "$taub" "$wtb" "$HD" "$H0" 0 "$GTARGET" \
      > "$OLDPWD/$LOGS/$tag.log" 2>&1 \
      && echo "  done $tag" || echo "  FAILED $tag" )
}
export -f run_cell
export WORK RUNS LOGS HD H0 GTARGET

echo "running ${#CELLS[@]} cells, ${NJOBS} at a time..."
printf '%s\n' "${CELLS[@]}" | xargs -P "$NJOBS" -I {} bash -c 'run_cell "$@"' _ {}

echo "=== done; collecting RESULT lines ==="
grep -h "^RESULT" "$LOGS"/*.log | sort > "$LOGS/onset_results.txt" || true
echo "wrote $LOGS/onset_results.txt"
echo "append these to your master summary.csv, then plot."

# ---- merge new RESULT lines into the master summary.csv (kelvin rows) ----
python3 - <<'PY'
import csv, glob, re, os
master = "../summary.csv"      # adjust if your summary.csv lives elsewhere
if not os.path.exists(master):
    master = "summary.csv"
rows = []
for f in glob.glob("onset_logs/*.log"):
    for line in open(f):
        if line.startswith("RESULT"):
            d = dict(kv.split("=",1) for kv in line.split()[1:])
            rows.append(d)
if not rows:
    print("no RESULT lines to merge"); raise SystemExit
# read existing header/rows
existing = []
header = ["run","xi0","tau_B","wtB","H_d","H0","k_grad","Fb","Ms_over_Href",
          "eta_eff","g_sim","g_ref","g_ref_nodemag","avg_steps"]
if os.path.exists(master):
    with open(master) as fp:
        r = csv.DictReader(fp); header = r.fieldnames or header
        existing = list(r)
seen = {(x["xi0"], x["wtB"], x.get("run")) for x in existing}
added = 0
for d in rows:
    key = (d["xi0"], d["wtB"], "kelvin")
    if key in seen:  # already present
        continue
    d["run"] = "kelvin"
    existing.append({k: d.get(k, "") for k in header}); added += 1
with open(master, "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=header); w.writeheader(); w.writerows(existing)
print(f"merged {added} new rows into {master} (now {len(existing)} rows)")
PY
