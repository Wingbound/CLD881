#!/usr/bin/env python3
"""
One command: launch the onset-fill simulations AND show live, in-place progress bars.

    python3 run_and_watch.py                     # runs the 20 onset-fill cells, 12 parallel
    python3 run_and_watch.py --jobs 14
    python3 run_and_watch.py --nsteps 120000 --nx 50 --ny 50

Unlike the earlier monitor, the bars update IN PLACE (cursor moves up and rewrites the lines)
instead of clearing the screen -- no flicker.  This script both starts the runs and watches
them, so it is the only thing you launch.  Ctrl-C stops the simulations too (they are its
children), after asking to confirm.

Needs, in this directory: phaseField_poiseuille_linearField_AC.cpp, sweep.py
"""
import argparse, os, re, subprocess, sys, time
from pathlib import Path

SRC = "phaseField_poiseuille_linearField_AC.cpp"
ITER_RE = re.compile(r"Writing iter (\d+)")
RESULT_RE = re.compile(r"^RESULT ", re.M)

# the 20 missing onset-figure cells (xi0, omega*tau_B)
CELLS = [
    (1, 1.6), (1, 1.7), (2.5, 1.6), (2.5, 1.7), (5, 1.7),
    (7.5, 1.6), (7.5, 1.7), (10, 1.7), (12.5, 1.6), (12.5, 1.7), (15, 1.7),
    (17.5, 0.1), (17.5, 0.5), (17.5, 1.0), (17.5, 1.4), (17.5, 1.5),
    (17.5, 1.6), (17.5, 1.7), (17.5, 2.0), (20, 1.7),
]

H0, R_BIAS, G_TARGET = 1.73, 0.3, 2e-7

ESC = "\033["
def up(n): return f"{ESC}{n}A"
def clear_line(): return f"{ESC}2K"
def hide(): sys.stdout.write(f"{ESC}?25l")
def show(): sys.stdout.write(f"{ESC}?25h")


def bar(frac, width=30):
    frac = max(0.0, min(1.0, frac))
    n = int(frac * width)
    return "\u2588" * n + "\u2591" * (width - n)   # solid / light blocks


def fmt(secs):
    if secs is None or secs != secs or secs < 0:
        return " --"
    secs = int(secs); h, r = divmod(secs, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def build(nsteps, nx, ny, nsubh, work):
    import importlib.util
    spec = importlib.util.spec_from_file_location("sweep", "sweep.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.patch(SRC, f"{work}/solver.cpp", nsteps=nsteps, nx=nx, ny=ny, nsubh=nsubh)
    r = subprocess.run(["g++", "-O3", "-std=c++17", "-o", f"{work}/solver", f"{work}/solver.cpp"],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit("compile failed:\n" + r.stderr[-2000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--nsteps", type=int, default=120000)
    ap.add_argument("--nx", type=int, default=50)
    ap.add_argument("--ny", type=int, default=50)
    ap.add_argument("--nsubh", type=int, default=64)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    if not Path(SRC).exists() or not Path("sweep.py").exists():
        sys.exit(f"need {SRC} and sweep.py in this directory")

    work, logs, runs = "onset_build", "onset_logs", "onset_runs"
    for d in (work, logs, runs):
        os.makedirs(d, exist_ok=True)
    print("building solver ...")
    build(args.nsteps, args.nx, args.ny, args.nsubh, work)
    solver = os.path.abspath(f"{work}/solver")
    Hd = R_BIAS * H0

    # launch pool: keep <= jobs running, feed cells as slots free
    pending = list(CELLS)
    procs = {}          # tag -> (Popen, logpath)
    tags = [f"xi0_{x}_wtB_{w}" for x, w in CELLS]
    started_at = time.time()
    hist = {}           # tag -> (wall, iter) for rate
    rate_samples = []

    def launch(cell):
        x, w = cell
        tag = f"xi0_{x}_wtB_{w}"
        taub = max(20.0, 10.0 * w)
        d = f"{runs}/{tag}"; os.makedirs(d, exist_ok=True)
        lp = f"{logs}/{tag}.log"
        fh = open(lp, "w")
        p = subprocess.Popen([solver, str(x), str(taub), str(w), str(Hd), str(H0),
                              "0", str(G_TARGET)], cwd=d, stdout=fh, stderr=subprocess.STDOUT)
        procs[tag] = (p, lp, fh)

    def read_iter(lp):
        try:
            t = open(lp, errors="ignore").read()
        except OSError:
            return 0, False
        it = 0
        for m in ITER_RE.finditer(t):
            it = max(it, int(m.group(1)))
        return it, bool(RESULT_RE.search(t))

    n = len(CELLS)
    nlines = n + 4      # header + per-cell + overall + rate/eta
    hide()
    printed = False
    try:
        while True:
            # fill free slots
            while pending and sum(1 for (p, _, _) in procs.values() if p.poll() is None) < args.jobs:
                launch(pending.pop(0))

            now = time.time()
            done = 0; running = 0; lines = []
            steps_left = 0
            for x, w in CELLS:
                tag = f"xi0_{x}_wtB_{w}"
                if tag in procs:
                    p, lp, _ = procs[tag]
                    it, has_result = read_iter(lp)
                    finished = has_result or p.poll() is not None
                    if tag in hist and not finished:
                        pt, pi = hist[tag]
                        if it - pi > 0 and now - pt > 0:
                            rate_samples.append((now - pt, it - pi))
                            rate_samples[:] = rate_samples[-40:]
                    hist[tag] = (now, it)
                    frac = 1.0 if finished else it / args.nsteps
                    if finished:
                        done += 1; state = "\u2713"
                    else:
                        running += 1; state = "\u25b6"; steps_left += max(0, args.nsteps - it)
                else:
                    frac = 0.0; state = "\u00b7"; steps_left += args.nsteps
                lines.append((tag, frac, state))

            if rate_samples:
                rs = sorted(di / dt for dt, di in rate_samples)
                per = rs[len(rs) // 2]
            else:
                per = 0.0
            active = max(1, running)
            eff = min(active, max(1, n - done))
            thru = per * eff
            eta = steps_left / thru if thru > 0 else None

            # ---- render in place ----
            buf = []
            if printed:
                buf.append(up(nlines))
            buf.append(clear_line() +
                       f"  onset fill   elapsed {fmt(now-started_at)}   "
                       f"{args.jobs} slots   {args.nsteps:,} steps/run\n")
            for tag, frac, state in lines:
                buf.append(clear_line() +
                           f"  {state} {tag:22s} {bar(frac)} {frac*100:5.1f}%\n")
            buf.append(clear_line() +
                       f"  OVERALL {bar(done/n, 40)} {done}/{n} ({done*100//n}%)\n")
            buf.append(clear_line() +
                       f"  rate ~{per:,.0f} steps/s/worker x {eff} = {thru:,.0f} steps/s\n")
            buf.append(clear_line() +
                       f"  ETC: {fmt(eta)}"
                       + (f"  (finish ~{time.strftime('%H:%M', time.localtime(now+eta))})"
                          if eta else "") + "\n")
            sys.stdout.write("".join(buf)); sys.stdout.flush()
            printed = True

            if done >= n:
                break
            time.sleep(args.interval)

        # merge results into ../summary.csv (or ./summary.csv)
        merge_results(logs)
        print("\n  all cells complete.")
    except KeyboardInterrupt:
        show()
        print("\n  Ctrl-C: stopping simulations ...")
        for p, _, _ in procs.values():
            if p.poll() is None:
                p.terminate()
        sys.exit(1)
    finally:
        show()
        for _, _, fh in procs.values():
            try: fh.close()
            except Exception: pass


def merge_results(logs):
    import csv, glob
    master = "../summary.csv" if os.path.exists("../summary.csv") else "summary.csv"
    header = ["run", "xi0", "tau_B", "wtB", "H_d", "H0", "k_grad", "Fb",
              "Ms_over_Href", "eta_eff", "g_sim", "g_ref", "g_ref_nodemag", "avg_steps"]
    existing = []
    if os.path.exists(master):
        with open(master) as fp:
            r = csv.DictReader(fp); header = r.fieldnames or header; existing = list(r)
    seen = {(x["xi0"], x["wtB"], x.get("run")) for x in existing}
    added = 0
    for f in glob.glob(f"{logs}/*.log"):
        for line in open(f):
            if line.startswith("RESULT"):
                d = dict(kv.split("=", 1) for kv in line.split()[1:]); d["run"] = "kelvin"
                if (d["xi0"], d["wtB"], "kelvin") in seen:
                    continue
                existing.append({k: d.get(k, "") for k in header}); added += 1
    with open(master, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=header); w.writeheader(); w.writerows(existing)
    print(f"  merged {added} new rows into {master}")


if __name__ == "__main__":
    main()
