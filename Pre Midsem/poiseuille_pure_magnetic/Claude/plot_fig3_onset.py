#!/usr/bin/env python3
"""
Bacri Fig. 3 layout with the negative-viscosity onset marked on each curve.

Plots g vs xi0, one curve per omega*tau_B, and on every curve that crosses from positive to
negative it marks the crossing point -- the first xi0 at which that frequency produces
negative rotational viscosity.

g is read from g_ref_nodemag by default (the fluid's demagnetisation-free rotational-viscosity
factor, present on every row -- control or kelvin).  That is the physical g; the raw Kelvin
g_sim is contaminated by the transverse-stress term and should not be used for onset.

    python3 plot_fig3_onset.py                                  # all omega*tau_B in the file
    python3 plot_fig3_onset.py --wtb 1.0,1.4,1.5,1.6,2.0        # just these frequencies
    python3 plot_fig3_onset.py --value g_sim --source control   # measured control g instead
    python3 plot_fig3_onset.py -o fig3_onset.png

The crossing on each curve is found by linear interpolation between the two bracketing xi0
points; a small open ring marks it and the xi0_onset value is printed.
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(csv_path, source, value):
    if source == "kelvin_corr":
        alt = csv_path.replace(".csv", "_corrected.csv")
        if not os.path.exists(alt):
            sys.exit(f"{alt} not found: run  correct_kelvin.py {csv_path}  first")
        csv_path = alt
    want = "control" if source == "control" else "kelvin"
    pts = {}
    for r in csv.DictReader(open(csv_path)):
        if source in ("control", "kelvin") and r.get("run") != want:
            # value columns g_ref/g_ref_nodemag are identical on both rows; g_sim is not
            if value == "g_sim":
                continue
        v = r.get("g_corr" if (source == "kelvin_corr" and r.get("g_corr")) else value, "")
        if v == "":
            continue
        pts[(float(r["xi0"]), float(r["wtB"]))] = float(v)
    if not pts:
        sys.exit(f"no usable rows for source={source}, value={value} in {csv_path}")
    return pts


def first_crossing(xs, gs):
    """First positive->negative crossing along xi0; returns (xi0_onset, g_left, g_right, i) or None."""
    for i in range(len(xs) - 1):
        if gs[i] > 0 >= gs[i + 1]:
            t = gs[i] / (gs[i] - gs[i + 1])
            return xs[i] + t * (xs[i + 1] - xs[i]), i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="summary.csv")
    ap.add_argument("--source", default="kelvin", choices=["control", "kelvin", "kelvin_corr"],
                    help="which rows; with the default value g_ref_nodemag either gives the fluid g")
    ap.add_argument("--value", default="g_ref_nodemag",
                    help="g column: g_ref_nodemag (default), g_ref, g_sim, g_corr")
    ap.add_argument("--wtb", default=None, help="comma-separated omega*tau_B to include")
    ap.add_argument("-o", "--out", default="fig3_onset.png")
    args = ap.parse_args()

    pts = load(args.csv, args.source, args.value)
    wts = sorted({w for _, w in pts})
    if args.wtb:
        keep = {float(v) for v in args.wtb.split(",")}
        wts = [w for w in wts if w in keep]
    if not wts:
        sys.exit("no omega*tau_B left after --wtb filter")

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    cmap = plt.get_cmap("viridis")
    print(f"{'wtB':>6} {'xi0_onset':>10}")
    onset_pts = []
    for n, w in enumerate(wts):
        row = sorted((x, g) for (x, ww), g in pts.items() if ww == w)
        if len(row) < 2:
            continue
        xs = np.array([r[0] for r in row]); gs = np.array([r[1] for r in row])
        col = cmap(n / max(len(wts) - 1, 1))
        ax.plot(xs, gs, "-o", color=col, ms=4, lw=1.4,
                label=f"{w:g}")
        cr = first_crossing(xs, gs)
        if cr:
            xo, i = cr
            ax.plot([xo], [0], "o", mfc="white", mec=col, ms=11, mew=2, zorder=5)
            onset_pts.append((xo, w, col))
            print(f"{w:6g} {xo:10.2f}")
        else:
            print(f"{w:6g} {'(no crossing)':>10}")

    # connect the onset points to show how the crossing moves with frequency
    if len(onset_pts) >= 2:
        onset_pts.sort()
        xo = [p[0] for p in onset_pts]; wo = [p[1] for p in onset_pts]
        # a faint guide line through the onset markers (in data coords they all sit at g=0)
        ax.plot(xo, [0] * len(xo), ":", color="0.4", lw=1, zorder=4)
        # annotate the earliest (smallest xi0_onset)
        xmin_i = int(np.argmin(xo))
        ax.annotate(f"first onset\n$\\xi_0\\approx{xo[xmin_i]:.1f}$ at $\\omega\\tau_B={wo[xmin_i]:g}$",
                    xy=(xo[xmin_i], 0), xytext=(xo[xmin_i] + 2.5, 0.18),
                    fontsize=9, color="0.2",
                    arrowprops=dict(arrowstyle="->", color="0.4"))

    ax.axhline(0, color="k", lw=1.0)
    ax.set_xlabel(r"$\xi_0$")
    ax.set_ylabel("$g$")
    src = {"control": "control", "kelvin": "reference", "kelvin_corr": "corrected"}[args.source]
    ax.set_title("Rotational-viscosity factor vs $\\xi_0$; open rings = negative-viscosity onset\n"
                 f"($g$ = {args.value}, {src} rows)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(title=r"$\omega\tau_B$", fontsize=8, ncol=2, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=190)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
