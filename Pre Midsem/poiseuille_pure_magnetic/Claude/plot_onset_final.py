#!/usr/bin/env python3
"""
Onset-of-negative-viscosity figure (Bacri Fig. 3 layout, publication version).

  * g vs xi0, one smooth curve per omega*tau_B
  * every curve anchored at the origin (xi0 = 0, g = 0 exactly: no magnetisation there)
  * markers at the sampled xi0 = 0, 1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20
  * smooth monotone-cubic (PCHIP) interpolation through the points (no overshoot)
  * open rings where each curve first crosses into negative g (the onset)
  * the two operative points highlighted:
        - global-earliest onset (smallest xi0 at which any frequency goes negative)
        - critical coupling xi0_crit (below which no frequency goes negative)

g is read from g_ref_nodemag (the fluid's demagnetisation-free rotational-viscosity factor,
present on every row).  That is the physical g; raw Kelvin g_sim is not used here.

    python3 plot_onset_final.py
    python3 plot_onset_final.py --wtb 1.0,1.4,1.5,1.6,1.7,2.0
    python3 plot_onset_final.py --csv summary.csv -o onset_final.png
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MARK_XI = [0, 1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]


def pchip(x, y, xq):
    """Monotone piecewise-cubic Hermite (Fritsch-Carlson); passes through points, no overshoot."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 2:
        return np.full_like(xq, y[0] if n else np.nan)
    h = np.diff(x); d = np.diff(y) / h; m = np.zeros(n)
    if n == 2:
        m[:] = d[0]
    else:
        for i in range(1, n - 1):
            if d[i-1] * d[i] > 0:
                w1, w2 = 2*h[i] + h[i-1], h[i] + 2*h[i-1]
                m[i] = (w1 + w2) / (w1/d[i-1] + w2/d[i])
        for i, (h0, h1, dd, dn) in ((0, (h[0], h[1] if n > 2 else h[0], d[0], d[min(1, len(d)-1)])),
                                    (n-1, (h[-1], h[-2], d[-1], d[-2]))):
            mi = ((2*h0 + h1) * dd - h0 * dn) / (h0 + h1)
            if np.sign(mi) != np.sign(dd):
                mi = 0.0
            elif np.sign(dd) != np.sign(dn) and abs(mi) > abs(3*dd):
                mi = 3*dd
            m[i] = mi
    idx = np.clip(np.searchsorted(x, xq) - 1, 0, n - 2)
    t = (xq - x[idx]) / h[idx]
    h00 = 2*t**3 - 3*t**2 + 1; h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2;    h11 = t**3 - t**2
    return h00*y[idx] + h10*h[idx]*m[idx] + h01*y[idx+1] + h11*h[idx]*m[idx+1]


def load(csv_path, value):
    d = {}
    for r in csv.DictReader(open(csv_path)):
        if r.get("run") != "kelvin":       # g_ref_nodemag identical on both; kelvin has all rows
            continue
        v = r.get(value, "")
        if v == "":
            continue
        d.setdefault(float(r["wtB"]), {})[float(r["xi0"])] = float(v)
    return d


def cross(xs, gs):
    for i in range(len(xs) - 1):
        if gs[i] > 0 >= gs[i+1]:
            t = gs[i] / (gs[i] - gs[i+1])
            return xs[i] + t * (xs[i+1] - xs[i])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="summary.csv")
    ap.add_argument("--value", default="g_ref_nodemag")
    ap.add_argument("--wtb", default=None, help="comma-separated omega*tau_B to include")
    ap.add_argument("-o", "--out", default="onset_final.png")
    args = ap.parse_args()

    data = load(args.csv, args.value)
    if not data:
        sys.exit(f"no kelvin rows with {args.value} in {args.csv}")
    wts = sorted(data)
    if args.wtb:
        keep = {float(v) for v in args.wtb.split(",")}
        wts = [w for w in wts if w in keep]

    fig, ax = plt.subplots(figsize=(9, 6.2))
    cmap = plt.get_cmap("viridis")
    onset_pts = []           # (xi0_onset, wtB, colour)
    print(f"{'wtB':>6} {'xi0_onset':>10}")
    for n, w in enumerate(wts):
        pts = data[w]
        xs = [0.0] + sorted(pts)                 # anchor at origin
        gs = [0.0] + [pts[x] for x in sorted(pts)]
        xs = np.array(xs); gs = np.array(gs)
        col = cmap(n / max(len(wts) - 1, 1))
        # smooth curve
        xq = np.linspace(0, xs.max(), 400)
        ax.plot(xq, pchip(xs, gs, xq), "-", color=col, lw=1.6, zorder=3)
        # sampled markers (only those actually present)
        mx = [x for x in MARK_XI if x == 0 or x in pts]
        mg = [0.0 if x == 0 else pts[x] for x in mx]
        ax.plot(mx, mg, "o", color=col, ms=5, zorder=4, label=f"{w:g}")
        # onset ring
        xo = cross(xs, gs)
        if xo is not None:
            ax.plot([xo], [0], "o", mfc="white", mec=col, ms=12, mew=2, zorder=6)
            onset_pts.append((xo, w, col))
            print(f"{w:6g} {xo:10.2f}")
        else:
            print(f"{w:6g} {'(none)':>10}")

    # ---- the two operative points ----
    if onset_pts:
        onset_pts.sort()
        # 1) global-earliest onset: leftmost ring
        xe, we, ce = onset_pts[0]
        ax.plot([xe], [0], "*", color="crimson", ms=20, zorder=7,
                mec="k", mew=0.5)
        ax.annotate(f"earliest onset\n$\\xi_0\\!\\approx\\!{xe:.1f}$ at $\\omega\\tau_B\\!=\\!{we:g}$",
                    xy=(xe, 0), xytext=(xe + 2.2, 0.13), fontsize=9, color="crimson",
                    arrowprops=dict(arrowstyle="->", color="crimson"), zorder=8)
        # 2) critical coupling: where the onset curve (xi0_onset vs wtB) turns back --
        #    equivalently the smallest xi0 that any curve reaches g<0.  Report the leftmost
        #    ring's xi0 as xi0_crit (they coincide to the grid resolution).
        ax.axvline(xe, color="0.5", ls=":", lw=1, zorder=1)
        ax.text(xe, ax.get_ylim()[0] * 0.92, f"$\\xi_{{0,\\mathrm{{crit}}}}\\approx{xe:.1f}$",
                rotation=90, va="bottom", ha="right", fontsize=8, color="0.4")

    ax.axhline(0, color="k", lw=1.0)
    ax.set_xlabel(r"$\xi_0$", fontsize=12)
    ax.set_ylabel(r"$g$", fontsize=12)
    ax.set_title("Rotational-viscosity factor $g(\\xi_0)$; open rings mark the onset of "
                 "negative viscosity", fontsize=11)
    ax.set_xlim(-0.5, 20.8)
    ax.grid(alpha=0.3)
    ax.legend(title=r"$\omega\tau_B$", fontsize=8, ncol=2, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
