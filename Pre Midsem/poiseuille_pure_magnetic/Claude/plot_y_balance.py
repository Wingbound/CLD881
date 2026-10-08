#!/usr/bin/env python3
"""
Clear version of the transverse-balance figure: why there is no net y-motion.

For a fully developed flow u = (u(y), 0) the y-momentum equation is

        0 = -dp/dy + F_y(y)        =>        pressure force  = -F_y(y),

so the sideways magnetic force is cancelled, at every y, by a pressure gradient, and the
fluid does not move sideways.  Instead of a cloud of little arrows, this draws:

  Panel 1  F_y(y):   the magnetic sideways force as a filled curve about zero, with a few
                     labelled arrows at the peaks so the direction is obvious.
  Panel 2  the balance:  F_y (red) and the pressure force -F_y (blue) on the same axis;
                     the grey band between them is the NET sideways force, which is ~0.
  Panel 3  u_y(y):   the sideways velocity actually observed, annotated against u_x.

    python3 plot_y_balance_clear.py runs/xi0_10.0_wtB_2.0_kelvin
    python3 plot_y_balance_clear.py runs/xi0_10.0_wtB_2.0_kelvin --iter 120000 --which Fk
    python3 plot_y_balance_clear.py runs/xi0_10.0_wtB_2.0_kelvin -o balance.png

--which  Fm (default) total magnetic y-force, or  Fk  Kelvin part only.

The solver does not write p, so the pressure force is reconstructed from the balance above
(that reconstruction is the physical statement being shown).  The real evidence that nothing
flows sideways is panel 3: u_y is ~1e-8, tens of thousands of times below u_x.
"""
import argparse, glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def load_snapshot(run, iter_):
    fs = glob.glob(os.path.join(run, "output_linear_field", "*", "output_t_*.csv"))
    if not fs:
        return None
    if iter_ is None:
        f = max(fs, key=lambda p: int(re.search(r"output_t_(\d+)", p).group(1)))
    else:
        want = [p for p in fs if f"output_t_{iter_}.csv" in p]
        if not want:
            sys.exit(f"iter {iter_} not found; have "
                     + ", ".join(sorted(re.search(r'output_t_(\d+)', p).group(1) for p in fs)))
        f = want[0]
    a = np.genfromtxt(f, delimiter=",", names=True)
    nX, nY = int(a["i"].max()) + 1, int(a["j"].max()) + 1
    g = lambda k: a[k].reshape(nX, nY)
    return dict(nX=nX, nY=nY, iter=int(re.search(r"output_t_(\d+)", f).group(1)),
                Fmy=g("Fm_y"), Fky=g("Fk_y"), uy=g("uy"), ux=g("ux"))


def sci(ax, axis):
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))
    getattr(ax, f"{axis}axis").set_major_formatter(fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--which", choices=["Fm", "Fk"], default="Fm")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    d = load_snapshot(args.run, args.iter)
    if d is None:
        sys.exit(f"no snapshots in {args.run}")
    nY = d["nY"]
    y = np.arange(nY) + 0.5
    Fy = (d["Fmy"] if args.which == "Fm" else d["Fky"]).mean(0)  # force depends on y only
    Ppush = -Fy
    net = Fy + Ppush                                             # ~0 by construction
    uy = d["uy"].mean(0)
    umax = np.abs(d["ux"]).max()
    lbl = r"$F_{K,y}$" if args.which == "Fk" else r"$F_{m,y}$"
    RED, BLUE, GREEN, GREY = "#c0392b", "#2e6fb0", "#2e8b57", "#9aa0a6"

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.2), sharey=True)

    # ---- panel 1: the sideways magnetic force itself ----
    a0 = ax[0]
    a0.fill_betweenx(y, 0, Fy, where=(Fy >= 0), color=RED, alpha=0.25)
    a0.fill_betweenx(y, 0, Fy, where=(Fy < 0), color=RED, alpha=0.25)
    a0.plot(Fy, y, color=RED, lw=2)
    a0.axvline(0, color="k", lw=0.8)
    # a couple of direction arrows at the extrema
    jhi, jlo = int(np.argmax(Fy)), int(np.argmin(Fy))
    for j in (jhi, jlo):
        a0.annotate("", xy=(Fy[j], y[j]), xytext=(0, y[j]),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    a0.text(0.5, 0.97, "points away from the centreline",
            transform=a0.transAxes, ha="center", va="top", fontsize=8, color=RED)
    a0.set_title(f"sideways magnetic force  {lbl}$(y)$", fontsize=10)
    a0.set_xlabel("force density")
    a0.set_ylabel("$y$")
    sci(a0, "x")

    # ---- panel 2: force vs pressure, and the (near-zero) net ----
    a1 = ax[1]
    a1.plot(Fy, y, color=RED, lw=2, label=f"magnetic  {lbl}")
    a1.plot(Ppush, y, color=BLUE, lw=2, ls="--", label=r"pressure force  $-\partial_y p$")
    a1.plot(net, y, color=GREY, lw=2.5, label="net = magnetic + pressure")
    a1.axvline(0, color="k", lw=0.8)
    a1.set_title("they cancel at every $y$  (net $\\approx 0$)", fontsize=10)
    a1.set_xlabel("force density")
    a1.legend(fontsize=8, loc="upper right", framealpha=0.9)
    sci(a1, "x")
    a1.text(0.03, 0.03,
            f"peak $|{lbl[1:-1]}|$ = {np.abs(Fy).max():.1e}\n"
            f"peak net = {np.abs(net).max():.1e}",
            transform=a1.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.9))

    # ---- panel 3: the sideways velocity that results ----
    a2 = ax[2]
    a2.plot(uy, y, color=GREEN, lw=2)
    a2.fill_betweenx(y, 0, uy, color=GREEN, alpha=0.2)
    a2.axvline(0, color="k", lw=0.8)
    a2.set_title("resulting sideways velocity  $\\bar u_y(y)$", fontsize=10)
    a2.set_xlabel(r"$\bar u_y$")
    sci(a2, "x")
    a2.text(0.5, 0.5,
            f"max$|u_y|$ = {np.abs(uy).max():.1e}\n"
            f"$u_x^{{max}}$ = {umax:.1e}\n"
            f"ratio = {np.abs(uy).max()/umax:.0e}",
            transform=a2.transAxes, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#eafaf0", ec=GREEN, alpha=0.95))

    for a in ax:
        a.set_ylim(-1, nY + 1)
        a.axhspan(-1, 0, color="k", alpha=0.06)      # wall bands
        a.axhspan(nY, nY + 1, color="k", alpha=0.06)
        a.grid(alpha=0.25)

    tag = os.path.basename(args.run.rstrip("/"))
    fig.suptitle(f"Why there is no net sideways flow — {tag}  (iter {d['iter']})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.out or f"y_balance_clear_{tag}.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)
    print(f"peak |F_y| = {np.abs(Fy).max():.3e}   peak net (balance residual) = {np.abs(net).max():.3e}")
    print(f"max |u_y| = {np.abs(uy).max():.3e}   u_x^max = {umax:.3e}   ratio = {np.abs(uy).max()/umax:.1e}")


if __name__ == "__main__":
    main()
