#!/usr/bin/env python3
"""
Clear QUIVER version of the transverse-balance figure.

At a handful of heights y, it draws two arrows starting from the SAME point:
    * red  , pointing along the magnetic sideways force  F_y(y)
    * blue , pointing along the pressure force  -F_y(y)
They are equal and opposite, so they visibly cancel -- that is the whole point.
A faint green arrow at each height shows the resulting sideways velocity u_y (scaled up,
because it is thousands of times smaller), so you can see it is essentially nothing.

    python3 plot_y_quiver_clear.py runs/xi0_10.0_wtB_2.0_kelvin
    python3 plot_y_quiver_clear.py runs/xi0_10.0_wtB_2.0_kelvin --iter 120000 --which Fk
    python3 plot_y_quiver_clear.py runs/xi0_10.0_wtB_2.0_kelvin --rows 11 -o quiver.png

--which  Fm (default) total magnetic y-force, or  Fk  Kelvin part only.
--rows   how many heights to sample (default 9).

The solver does not write p; the pressure force is reconstructed from  0 = -dp/dy + F_y
(that reconstruction is the physical statement being shown).  The real evidence that
nothing flows sideways is the green u_y arrows and the ratio printed on the plot.
"""
import argparse, glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--which", choices=["Fm", "Fk"], default="Fm")
    ap.add_argument("--rows", type=int, default=9, help="number of sampled heights")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    d = load_snapshot(args.run, args.iter)
    if d is None:
        sys.exit(f"no snapshots in {args.run}")
    nY = d["nY"]
    yall = np.arange(nY) + 0.5
    Fy_all = (d["Fmy"] if args.which == "Fm" else d["Fky"]).mean(0)
    uy_all = d["uy"].mean(0)
    umax = np.abs(d["ux"]).max()
    lbl = r"$F_{K,y}$" if args.which == "Fk" else r"$F_{m,y}$"
    RED, BLUE, GREEN = "#c0392b", "#2e6fb0", "#2e8b57"

    # sample a manageable number of interior rows
    idx = np.linspace(2, nY - 3, args.rows).round().astype(int)
    idx = np.unique(idx)
    y = yall[idx]
    Fy = Fy_all[idx]
    uy = uy_all[idx]

    fmax = max(np.abs(Fy_all).max(), 1e-30)
    umx = max(np.abs(uy_all).max(), 1e-30)

    # geometry of the drawing: a single column at x0, arrows scaled to a fixed visual length
    x0 = 0.5
    ARROW = 0.34            # visual half-length of the biggest force arrow (axes fraction of width span)
    W = 1.0                 # plotting width
    fscale = ARROW * W / fmax
    # u_y is drawn relative to the ACTUAL axial flow u_x^max: the green arrow length is
    # (u_y / u_x^max) of the full force-arrow span, so "no sideways flow" reads as
    # "no visible green arrow".  A faint magnified ghost shows its shape.
    uscale = ARROW * W / umax
    MAG = 0.16 * ARROW * W / umx      # magnify the ghost to its own max for shape only
    GHOST = True

    fig, ax = plt.subplots(figsize=(8.6, 6.4))

    for yj, fj, uj in zip(y, Fy, uy):
        # magnetic force (red) and pressure force (blue), tail-to-tail at (x0, yj)
        ax.add_patch(FancyArrowPatch((x0, yj), (x0 + fj * fscale, yj),
                                     arrowstyle="-|>", mutation_scale=16,
                                     color=RED, lw=2.2, zorder=3))
        ax.add_patch(FancyArrowPatch((x0, yj), (x0 - fj * fscale, yj),
                                     arrowstyle="-|>", mutation_scale=16,
                                     color=BLUE, lw=2.2, ls="--", zorder=2, alpha=0.9))
        # faint magnified ghost (shape of u_y), then the true-scale arrow on top
        ax.add_patch(FancyArrowPatch((x0, yj - 0.55), (x0 + uj * MAG, yj - 0.55),
                                     arrowstyle="-|>", mutation_scale=7,
                                     color=GREEN, lw=1.0, zorder=4, alpha=0.30))
        ax.add_patch(FancyArrowPatch((x0, yj - 0.55), (x0 + uj * uscale, yj - 0.55),
                                     arrowstyle="-|>", mutation_scale=9,
                                     color=GREEN, lw=1.8, zorder=5))

    ax.axvline(x0, color="k", lw=1.0, zorder=1)
    ax.axhline(0, color="k", lw=1.2)
    ax.axhline(nY, color="k", lw=1.2)
    ax.axhspan(-1.5, 0, color="k", alpha=0.06)
    ax.axhspan(nY, nY + 1.5, color="k", alpha=0.06)
    ax.text(x0, -1.0, "wall", ha="center", va="center", fontsize=8, color="0.3")
    ax.text(x0, nY + 1.0, "wall", ha="center", va="center", fontsize=8, color="0.3")

    ax.set_xlim(x0 - 1.15 * ARROW * W, x0 + 1.15 * ARROW * W)
    ax.set_ylim(-1.6, nY + 1.6)
    ax.set_xticks([])
    ax.set_ylabel("$y$")
    ax.set_title(f"Transverse balance at each height — {os.path.basename(args.run.rstrip('/'))}"
                 f"  (iter {d['iter']})", fontsize=11)

    # legend via proxy handles
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=RED, lw=2.2, label=f"magnetic force  {lbl}"),
        Line2D([0], [0], color=BLUE, lw=2.2, ls="--", label=r"pressure force  $-\partial_y p = -F_y$"),
        Line2D([0], [0], color=GREEN, lw=1.8,
               label=r"resulting $u_y$, solid = true size vs $u_x^{max}$; faint = shape, magnified $\times$%d" % round(MAG/uscale)),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=1, fontsize=9, framealpha=0.95)

    ax.text(0.5, 1.10,
            "red and blue are equal and opposite at every height, so they cancel — "
            "the fluid barely moves sideways\n"
            f"peak $|F_y|$ = {fmax:.1e}      max $|u_y|$ = {umx:.1e}      "
            f"$u_x^{{max}}$ = {umax:.1e}      $|u_y|/u_x^{{max}}$ = {umx/umax:.0e}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f5f7fa", ec="0.6", alpha=0.95))

    fig.subplots_adjust(top=0.80)
    fig.tight_layout(rect=[0,0,1,0.88])
    out = args.out or f"y_quiver_clear_{os.path.basename(args.run.rstrip('/'))}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)
    print(f"peak |F_y| = {fmax:.3e}   max |u_y| = {umx:.3e}   u_x^max = {umax:.3e}   "
          f"ratio = {umx/umax:.1e}")
    print("NOTE: solid green u_y arrows are drawn to true size against u_x^max (so they "
          "read as near-zero); the faint green shows their shape magnified.")


if __name__ == "__main__":
    main()
