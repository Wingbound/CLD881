#!/usr/bin/env python3
"""
Plot the velocity (and force) profiles of one or more runs.

    python3 plot_profiles.py runs/xi0_10.0_wtB_2.0_kelvin runs/xi0_10.0_wtB_2.0_ctrl
    python3 plot_profiles.py runs/*_wtB_2.0_* -o fig_wtB2.png
    python3 plot_profiles.py runs/xi0_10.0_wtB_2.0_kelvin --snapshot   # last raw snapshot

By default it uses profile_avg.csv (time averaged, written at the end of the run).
With --snapshot it uses the last output_t_*.csv instead and averages over x, which is
an instantaneous profile: for AC runs it still contains the oscillating flow.
"""
import argparse, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ETA_H, PHI_P = 0.05, 0.2          # must match the solver


def read_avg(run):
    f = glob.glob(os.path.join(run, "output_linear_field", "*", "profile_avg.csv"))
    if not f:
        return None
    head = open(f[0]).readline().lstrip("#").split()
    meta = {k: float(v) for k, v in (kv.split("=") for kv in head)}
    d = np.genfromtxt(f[0], delimiter=",", names=True, skip_header=1)
    return dict(y=d["j"] + 0.5, u=d["u_mean"], Fk=d["Fk_mean"],
                Frot=d["Fm_mean"] - d["Fk_mean"], Mx=d["Mx_mean"], meta=meta)


def read_snapshot(run):
    fs = glob.glob(os.path.join(run, "output_linear_field", "*", "output_t_*.csv"))
    if not fs:
        return None
    f = max(fs, key=lambda p: int(re.search(r"output_t_(\d+)", p).group(1)))
    a = np.genfromtxt(f, delimiter=",", names=True)
    nX, nY = int(a["i"].max()) + 1, int(a["j"].max()) + 1
    g = lambda k: a[k].reshape(nX, nY).mean(0)
    return dict(y=np.arange(nY) + 0.5, u=g("ux"), Fk=g("Fk_x"),
                Frot=g("Fm_x") - g("Fk_x"), Mx=g("Mx"),
                meta={"snapshot": int(re.search(r"output_t_(\d+)", f).group(1))})


def label_of(run, m):
    base = os.path.basename(run.rstrip("/"))
    if "xi0" in m:
        return (f"{base}\n$\\xi_0$={m['xi0']:g}, $\\omega\\tau_B$={m['omega_tauB']:g}"
                f", $g_{{sim}}$={m.get('g_sim', float('nan')):.3f}")
    return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run directories")
    p.add_argument("-o", "--out", default="profiles.png")
    p.add_argument("--snapshot", action="store_true", help="use the last raw snapshot")
    p.add_argument("--forces", action="store_true", help="add a force panel")
    args = p.parse_args()

    ncol = 2 if args.forces else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 4.6), squeeze=False)
    ax = axes[0][0]

    for run in args.runs:
        d = read_snapshot(run) if args.snapshot else read_avg(run)
        if d is None:
            print(f"no data in {run} (did the run finish?)")
            continue
        H = d["y"][-1] + 0.5
        line, = ax.plot(d["u"], d["y"], "o-", ms=3, lw=1, label=label_of(run, d["meta"]))
        # parabola implied by the fitted viscosity and the mean drive
        m = d["meta"]
        if "eta_eff" in m:
            G = d["Fk"][2:-2].mean() + m.get("Fb", 0.0)
            u_par = G / (2 * m["eta_eff"]) * d["y"] * (H - d["y"])
            ax.plot(u_par, d["y"], "--", lw=1, color=line.get_color(), alpha=0.6)
        print(f"{os.path.basename(run.rstrip('/')):40s} u_max={d['u'].max():.4e}"
              + (f"  eta_eff={m['eta_eff']:.4e}  g_sim={m['g_sim']:+.4f}" if "eta_eff" in m else ""))

    ax.set_xlabel("$\\bar u_x$ (lattice units)")
    ax.set_ylabel("$y$")
    ax.set_title("velocity profile" + (" (snapshot)" if args.snapshot else " (time averaged)"))
    ax.axhline(0, color="k", lw=0.8); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    if args.forces:
        axf = axes[0][1]
        for run in args.runs:
            d = read_snapshot(run) if args.snapshot else read_avg(run)
            if d is None:
                continue
            base = os.path.basename(run.rstrip("/"))
            drive = d["Fk"] + d["meta"].get("Fb", 0.0)   # Kelvin force, or Fb for controls
            l, = axf.plot(drive, d["y"], "-", lw=1, label=f"{base}: drive")
            axf.plot(d["Frot"], d["y"], "--", lw=1, color=l.get_color(), label=f"{base}: $F_{{rot}}$")
        axf.set_xlabel("mean force density"); axf.set_ylabel("$y$")
        axf.set_title("drive (solid) and rotational force (dashed)")
        axf.grid(alpha=0.3); axf.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
