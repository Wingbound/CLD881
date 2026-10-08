#!/usr/bin/env python3
"""
Patch, build, smoke-test and run the (xi0, omega*tau_B) sweep for the
linear-field ferrofluid LBM solver.

    python3 sweep.py            # smoke test, then the full sweep, then collect
    python3 sweep.py --smoke    # stop after the smoke test
    python3 sweep.py --collect  # only rebuild summary.csv from existing logs

If the smoke test fails, the reason is printed and nothing else runs.
"""
import csv, glob, math, os, shutil, subprocess, sys, time
from pathlib import Path

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
# Any of the settings below can be overridden from the environment, e.g.
#   XI0_LIST="20.0" WTB_LIST="1.5" NSTEPS=4000 NX=20 NY=16 ./run_sweep.sh
def _envf(name, default):  return float(os.environ.get(name, default))
def _envi(name, default):  return int(float(os.environ.get(name, default)))
def _envl(name, default):  return [float(v) for v in os.environ.get(name, default).split()]

SRC        = os.environ.get("SRC", "phaseField_poiseuille_linearField_AC.cpp")

XI0_LIST   = _envl("XI0_LIST", "1.0 2.5 5.0 10.0 15.0 20.0")   # Langevin parameter at H_ref
WTB_LIST   = _envl("WTB_LIST", "0.0 0.1 0.5 1.0 1.4 2.0")      # omega * tau_B
# xi0 = 0 cannot run: no magnetisation means no Kelvin drive and the solver exits.

H0          = _envf("H0", 1.73)        # oscillating field amplitude, on in every case
R_BIAS      = _envf("R_BIAS", 0.3)     # H_d / H0, the static bias that produces the drive
G_TARGET    = _envf("G_TARGET", 2e-7)  # target mean drive (also the control body force)
RUN_CONTROL = _envi("RUN_CONTROL", 1) == 1   # also run a steady-drive control per case

NSTEPS = _envi("NSTEPS", 120000)       # production settings
NX     = _envi("NX", 50)
NY     = _envi("NY", 50)
NSUBH  = _envi("NSUBH", 64)
SMOKE_NSTEPS, SMOKE_NX, SMOKE_NY = 4000, 20, 16   # smoke test

AUTO_SCALE_G = True       # shrink G_TARGET per case so that k_grad*nX/H_d <= MAX_RAMP
MAX_RAMP   = 0.02         # limit on k_grad*nX/H_d
MAX_HLAM   = 2.0          # limit on xi0/(8*tau_B), RK4 stability
TAUB_RULE  = lambda wtb: max(20.0, 10.0 * wtb)    # keeps omega*dt <= 0.1

# ----------------------------------------------------------------------------
# patching
# ----------------------------------------------------------------------------
EDITS = [
    # (1) command-line overrides: xi0 tau_B [omega_tauB] [H_d] [H0] [Fb]
    ("""  // Optional command-line override:  ./solver xi0 tau_B
  if (argc >= 3) {
    xi0   = std::atof(argv[1]);
    tau_B = std::atof(argv[2]);
  }""",
     """  // Optional command-line override:  ./solver xi0 tau_B [omega_tauB] [H_d] [H0] [Fb] [G_target]
  if (argc >= 3) {
    xi0   = std::atof(argv[1]);
    tau_B = std::atof(argv[2]);
  }
  if (argc >= 4) omega_tauB = std::atof(argv[3]);
  if (argc >= 5) H_d        = std::atof(argv[4]);
  if (argc >= 6) H0         = std::atof(argv[5]);
  if (argc >= 7) Fb_const = std::atof(argv[6]);   // control run: steady drive
  if (argc >= 8) G_target  = std::atof(argv[7]);  // per-case target drive
  if (Fb_const != 0.0) { G_target = 0.0; k_grad_input = 0.0; }"""),
    # (2) constant body force for control runs
    ("double k_grad_input = 0.0; // field gradient used when G_target <= 0",
     "double k_grad_input = 0.0; // field gradient used when G_target <= 0\n"
     "double Fb_const = 0.0;     // constant body force (control runs), set from argv[6]"),
    ("      Fb_x[i][j] = 0.0;", "      Fb_x[i][j] = Fb_const;"),
    # (3) omega = 0 must not give an infinite period in the reference
    ("""  const double period = 2.0 * acos(-1.0) / omega;
  int n_trans = int(40.0 * tau_B + 10.0 * period) + 1;
  int n_avg   = (H0 != 0.0) ? int(2000.0 * period + 0.5) : int(400.0 * tau_B) + 1;""",
     """  const double period = (omega > 0.0) ? 2.0 * acos(-1.0) / omega : 0.0;
  int n_trans = int(40.0 * tau_B + 10.0 * period) + 1;
  int n_avg   = (H0 != 0.0 && period > 0.0) ? int(2000.0 * period + 0.5)
                                            : int(400.0 * tau_B) + 1;"""),
    # (4) time averages
    ("static double Fk_x[nX][nY], Fk_y[nX][nY];   // Kelvin part alone (for post-processing)",
     """static double Fk_x[nX][nY], Fk_y[nX][nY];   // Kelvin part alone (for post-processing)
// ---- time averages over the last AVG_n steps (AC runs need them; harmless for DC)
static double AVG_ux[nX][nY], AVG_Fk[nX][nY], AVG_Fm[nX][nY], AVG_Mx[nX][nY];
long AVG_n = 0;"""),
    ("    // Equilibrium distributions",
     """    // ---- time averaging over the last AVG_win steps
    {
      const double per = (omega > 0.0) ? 2.0 * acos(-1.0) / omega : 0.0;
      long AVG_win = (per > 0.0) ? long(40.0 * per + 0.5) : 2000;
      if (AVG_win < 2000)        AVG_win = 2000;
      if (AVG_win > nSteps / 4)  AVG_win = nSteps / 4;
      if (iter > nSteps - AVG_win) {
        AVG_n++;
        for (int i = 0; i < nX; i++)
          for (int j = 0; j < nY; j++) {
            AVG_ux[i][j] += ux[i][j];
            AVG_Fk[i][j] += Fk_x[i][j];
            AVG_Fm[i][j] += Fm_x[i][j];
            AVG_Mx[i][j] += Mx[i][j];
          }
      }
    }

    // Equilibrium distributions"""),
    # (5) averaged profile + fitted viscosity
    ("""  return 0;""",
     """  // ---- averaged profile + effective viscosity from  eta_eff * u'' = -(F_kelvin + Fb)
  if (AVG_n > 0) {
    double u[nY], Fk[nY], Fm[nY], Mxb[nY];
    for (int j = 0; j < nY; j++) {
      u[j] = Fk[j] = Fm[j] = Mxb[j] = 0.0;
      for (int i = 0; i < nX; i++) {
        u[j]   += AVG_ux[i][j] / AVG_n;  Fk[j]  += AVG_Fk[i][j] / AVG_n;
        Fm[j]  += AVG_Fm[i][j] / AVG_n;  Mxb[j] += AVG_Mx[i][j] / AVG_n;
      }
      u[j] /= nX;  Fk[j] /= nX;  Fm[j] /= nX;  Mxb[j] /= nX;
    }
    double num = 0.0, den = 0.0;
    for (int j = 2; j < nY - 2; j++) {          // skip the two wall rows
      double d2u = u[j - 1] - 2.0 * u[j] + u[j + 1];
      num += -(Fk[j] + Fb_const) * d2u;  den += d2u * d2u;
    }
    double eta_eff = num / den;
    double g_sim   = (eta_eff / eta_h - 1.0) / (1.5 * phi_p);
    cout << "RESULT xi0=" << xi0 << " tau_B=" << tau_B << " wtB=" << omega_tauB
         << " H_d=" << H_d << " H0=" << H0 << " k_grad=" << k_grad
         << " Fb=" << Fb_const << " eta_eff=" << eta_eff << " g_sim=" << g_sim
         << " g_ref=" << g_ref << " g_ref_nodemag=" << g_nd
         << " Ms_over_Href=" << Ms / H_ref << " avg_steps=" << AVG_n << endl;
    FILE *fpa = fopen((subDir / "profile_avg.csv").string().c_str(), "w");
    fprintf(fpa, "# xi0=%g tau_B=%g omega_tauB=%g H_d=%g H0=%g Fb=%g k_grad=%.10e "
                 "eta_eff=%.10e g_sim=%.10e g_ref=%.10e g_ref_nodemag=%.10e avg_steps=%ld\\n",
            xi0, tau_B, omega_tauB, H_d, H0, Fb_const, k_grad, eta_eff, g_sim, g_ref, g_nd, AVG_n);
    fprintf(fpa, "j,u_mean,Fk_mean,Fm_mean,Mx_mean\\n");
    for (int j = 0; j < nY; j++)
      fprintf(fpa, "%d,%.10e,%.10e,%.10e,%.10e\\n", j, u[j], Fk[j], Fm[j], Mxb[j]);
    fclose(fpa);
  }
  return 0;"""),
]


def patch(src, dst, nsteps, nx, ny, nsubh):
    text = Path(src).read_text()
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"patch anchor matched {n} times, expected 1:\n{old[:90]}")
        text = text.replace(old, new)
    for old, new in [("int nSteps = 200000;", f"int nSteps = {nsteps};"),
                     ("const int nX = 50;", f"const int nX = {nx};"),
                     ("const int nY = 50;", f"const int nY = {ny};"),
                     ("const int NSUB_H = 64;", f"const int NSUB_H = {nsubh};")]:
        if text.count(old) != 1:
            raise RuntimeError(f"size anchor not found exactly once: {old}")
        text = text.replace(old, new)
    Path(dst).write_text(text)


def build(cpp, exe):
    r = subprocess.run(["g++", "-O3", "-std=c++17", "-o", exe, cpp],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("compilation failed:\n" + r.stderr[-3000:])


def run_case(exe, workdir, args, logfile):
    Path(workdir).mkdir(parents=True, exist_ok=True)
    with open(logfile, "w") as fp:
        return subprocess.Popen([os.path.abspath(exe)] + [str(a) for a in args],
                                cwd=workdir, stdout=fp, stderr=subprocess.STDOUT)


def parse_result(logfile):
    for line in Path(logfile).read_text().splitlines():
        if line.startswith("RESULT"):
            return dict(kv.split("=", 1) for kv in line.split()[1:])
    return None


def langevin(x):
    return 1.0 / math.tanh(x) - 1.0 / x if abs(x) > 1e-4 else x / 3.0


# ----------------------------------------------------------------------------
# smoke test
# ----------------------------------------------------------------------------
def smoke_test():
    print("=== smoke test ===")
    work = Path("smoke_build")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    try:
        patch(SRC, work / "solver_smoke.cpp", SMOKE_NSTEPS, SMOKE_NX, SMOKE_NY, NSUBH)
        print("  patch applied")
        build(str(work / "solver_smoke.cpp"), str(work / "solver_smoke"))
        print("  compiled")
    except RuntimeError as e:
        print("SMOKE TEST FAILED:", e)
        return False

    checks = [
        # (label, args, extra check)
        ("DC, H0 = 0", [20.0, 20.0, 0.0, 1.73, 0.0], "dc"),
        ("AC + ramp",  [20.0, 20.0, 1.5, R_BIAS * H0, H0], "ac"),
        ("control",    [20.0, 20.0, 1.5, R_BIAS * H0, H0, G_TARGET, G_TARGET], "ctrl"),
    ]
    ok = True
    for label, args, kind in checks:
        wd = work / f"run_{kind}"
        log = work / f"{kind}.log"
        p = run_case(work / "solver_smoke", wd, args, log)
        rc = p.wait()
        tail = "\n".join(Path(log).read_text().splitlines()[-8:])
        if rc != 0:
            print(f"SMOKE TEST FAILED: case '{label}' exited with code {rc}\n{tail}")
            return False
        res = parse_result(log)
        if res is None:
            print(f"SMOKE TEST FAILED: case '{label}' printed no RESULT line\n{tail}")
            return False
        vals = {k: float(v) for k, v in res.items()}
        if any(math.isnan(v) or math.isinf(v) for v in vals.values()):
            print(f"SMOKE TEST FAILED: case '{label}' produced NaN/Inf:\n{res}")
            return False
        prof = list(wd.glob("output_linear_field/*/profile_avg.csv"))
        if not prof:
            print(f"SMOKE TEST FAILED: case '{label}' wrote no profile_avg.csv")
            return False
        nrows = sum(1 for _ in open(prof[0])) - 2
        if nrows != SMOKE_NY:
            print(f"SMOKE TEST FAILED: profile_avg.csv has {nrows} rows, expected {SMOKE_NY}")
            return False
        if kind == "dc":
            xi = 20.0
            g_analytic = xi * langevin(xi) ** 2 / (xi - langevin(xi))
            if abs(vals["g_ref"] - g_analytic) > 1e-3:
                print(f"SMOKE TEST FAILED: static reference g = {vals['g_ref']:.6f}, "
                      f"analytic = {g_analytic:.6f}")
                return False
            print(f"  DC reference g = {vals['g_ref']:.6f} vs analytic {g_analytic:.6f}  OK")
        if kind == "ac" and vals["k_grad"] <= 0.0:
            print(f"SMOKE TEST FAILED: k_grad = {vals['k_grad']}")
            return False
        print(f"  case '{label}' ran, RESULT written  OK")
    print("=== smoke test passed ===\n")
    return ok


# ----------------------------------------------------------------------------
# pre-flight parameter check (no simulation, just the criteria)
# ----------------------------------------------------------------------------
def preflight():
    """Check the sweep limits and return {(xi0, wtb): G_case}."""
    eta_h, H_ref = 0.05, 1.73
    phi_p = 0.2
    print("=== pre-flight parameter check ===")
    print(f"{'xi0':>6} {'wtB':>6} {'tau_B':>7} {'G_case':>10} {'kL/H_d':>9} "
          f"{'xi0/(8 tau_B)':>14} {'Ms/H_ref':>10}")
    gmap, bad = {}, []
    Hd = R_BIAS * H0
    for xi0 in XI0_LIST:
        for wtb in WTB_LIST:
            taub = TAUB_RULE(wtb)
            Ms = 3 * eta_h * phi_p * xi0 / (taub * H_ref)
            Mx = Ms * langevin(xi0 * (Hd + H0) / H_ref)    # quasi-static estimate
            G = G_TARGET
            ramp = (G / Mx) * NX / Hd if Mx > 0 else float("inf")
            if AUTO_SCALE_G and ramp > MAX_RAMP:
                G *= MAX_RAMP / ramp
                ramp = MAX_RAMP
            hlam = xi0 / (8 * taub)
            flag = ""
            if ramp > MAX_RAMP:
                flag += "  RAMP TOO LARGE"; bad.append((xi0, wtb, "ramp", ramp))
            if hlam > MAX_HLAM:
                flag += "  RK4 UNSTABLE";  bad.append((xi0, wtb, "rk4", hlam))
            gmap[(xi0, wtb)] = G
            print(f"{xi0:6.2f} {wtb:6.2f} {taub:7.1f} {G:10.3e} {ramp:9.4f} "
                  f"{hlam:14.3f} {Ms/H_ref:10.5f}{flag}")
    if bad:
        print("\nWARNING: the cases flagged above still violate the limits; "
              "lower G_TARGET or adjust TAUB_RULE.")
    print()
    return gmap


# ----------------------------------------------------------------------------
# full sweep
# ----------------------------------------------------------------------------
def sweep(gmap):
    work = Path("sweep_build"); work.mkdir(exist_ok=True)
    logs = Path("logs"); logs.mkdir(exist_ok=True)
    runs = Path("runs"); runs.mkdir(exist_ok=True)
    patch(SRC, work / "solver_sweep.cpp", NSTEPS, NX, NY, NSUBH)
    build(str(work / "solver_sweep.cpp"), str(work / "solver_sweep"))
    print(f"built {work/'solver_sweep'}  (nSteps={NSTEPS}, {NX}x{NY}, NSUB_H={NSUBH})")
    Hd = R_BIAS * H0
    for xi0 in XI0_LIST:
        batches = [("kelvin", None)] + ([("ctrl", "fb")] if RUN_CONTROL else [])
        for kind, fb in batches:
            print(f"=== xi0 = {xi0}: {len(WTB_LIST)} {kind} runs in parallel ===")
            procs = []
            for wtb in WTB_LIST:
                taub = TAUB_RULE(wtb)
                tag = f"xi0_{xi0}_wtB_{wtb}_{kind}"
                G = gmap[(xi0, wtb)]
                args = [xi0, taub, wtb, Hd, H0, G if fb is not None else 0.0, G]
                procs.append((tag, run_case(work / "solver_sweep", runs / tag,
                                            args, logs / f"{tag}.log"), time.time()))
            for tag, p, t0 in procs:
                rc = p.wait()
                print(f"  {'done' if rc == 0 else 'FAILED(rc=%d)' % rc} {tag} "
                      f"({(time.time()-t0)/60:.1f} min)")


# ----------------------------------------------------------------------------
# collect
# ----------------------------------------------------------------------------
def collect():
    rows = []
    for f in sorted(glob.glob("logs/*.log")):
        res = parse_result(f)
        if res:
            res["run"] = "control" if float(res.get("Fb", 0)) != 0.0 else "kelvin"
            res["log"] = f
            rows.append(res)
    if not rows:
        print("no RESULT lines found in logs/ -- check the logs for errors")
        return
    keys = ["run", "xi0", "tau_B", "wtB", "H_d", "H0", "k_grad", "Fb", "Ms_over_Href",
            "eta_eff", "g_sim", "g_ref", "g_ref_nodemag", "avg_steps", "log"]
    with open("summary.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote summary.csv with {len(rows)} rows\n")
    print(f"{'run':8} {'xi0':>6} {'wtB':>6} {'tau_B':>7} {'g_sim':>9} {'g_ref':>9} {'g_ref_nd':>9}")
    for r in sorted(rows, key=lambda r: (float(r["xi0"]), float(r["wtB"]), r["run"])):
        print(f"{r['run']:8} {float(r['xi0']):6.2f} {float(r['wtB']):6.2f} {float(r['tau_B']):7.1f} "
              f"{float(r['g_sim']):>+9.4f} {float(r['g_ref']):>+9.4f} {float(r['g_ref_nodemag']):>+9.4f}")


def main():
    if not Path(SRC).exists():
        print(f"source file '{SRC}' not found in {os.getcwd()}")
        sys.exit(1)
    if "--collect" in sys.argv:
        collect(); return
    if not smoke_test():
        print("\nSmoke test failed: nothing else was run. Fix the problem above and rerun.")
        sys.exit(1)
    gmap = preflight()
    if "--smoke" in sys.argv:
        print("--smoke given: stopping before the sweep.")
        return
    sweep(gmap)
    collect()


if __name__ == "__main__":
    main()
