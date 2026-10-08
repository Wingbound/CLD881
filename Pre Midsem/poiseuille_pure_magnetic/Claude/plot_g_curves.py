#!/usr/bin/env python3
"""
Plot the simulated rotational-viscosity factor g against omega*tau_B, one curve per xi0,
in the style of Bacri et al. (PRL 1995) Fig. 3.  Simulation data only: markers are the
simulated g, lines are shape-preserving (PCHIP) trend lines through them.

    python3 plot_g_curves.py                          # control runs, g vs omega*tau_B
    python3 plot_g_curves.py --vs xi0                 # Bacri's layout: g vs xi0, one curve per omega*tau_B
    python3 plot_g_curves.py --source kelvin_corr     # Kelvin runs after the eps correction
    python3 plot_g_curves.py --csv path/summary.csv -o g_curves.png
    python3 plot_g_curves.py --join-static            # also join the omega*tau_B = 0 points

The omega*tau_B = 0 runs have a static field at H_d + H0.  That is not the omega -> 0 limit
of the AC curves, which still sweep through every field value each period, so by default
those points are drawn hollow and kept out of the trend lines.

Sources:
    control      g_sim of the body-force control runs (the direct measurement; default)
    kelvin       g_sim of the Kelvin-driven runs, uncorrected
    kelvin_corr  g_corr of the Kelvin-driven runs (needs summary_corrected.csv from
                 correct_kelvin.py; static points, which need no correction, use g_sim)
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pchip(x, y, xq):
    """Monotone piecewise-cubic Hermite interpolation (Fritsch-Carlson); no overshoot."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 2:
        return np.full_like(xq, y[0] if n else np.nan)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros(n)
    if n == 2:
        m[:] = d[0]
    else:
        for i in range(1, n - 1):
            if d[i - 1] * d[i] > 0:
                w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
                m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
        for i, (hh, dd, dn) in ((0, (h[0], d[0], d[1])), (n - 1, (h[-1], d[-1], d[-2]))):
            h0, h1 = (h[0], h[1]) if i == 0 else (h[-1], h[-2])
            mi = ((2 * h0 + h1) * dd - h0 * dn) / (h0 + h1)
            if np.sign(mi) != np.sign(dd):
                mi = 0.0
            elif np.sign(dd) != np.sign(dn) and abs(mi) > abs(3 * dd):
                mi = 3 * dd
            m[i] = mi
    idx = np.clip(np.searchsorted(x, xq) - 1, 0, n - 2)
    t = (xq - x[idx]) / h[idx]
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    return h00 * y[idx] + h10 * h[idx] * m[idx] + h01 * y[idx + 1] + h11 * h[idx] * m[idx + 1]


def load(csv_path, source):
    if source == "kelvin_corr":
        alt = csv_path.replace(".csv", "_corrected.csv")
        if not os.path.exists(alt):
            sys.exit(f"{alt} not found: run  python3 correct_kelvin.py {csv_path}  first")
        csv_path = alt
    want = "control" if source == "control" else "kelvin"
    pts = []
    for r in csv.DictReader(open(csv_path)):
        if r.get("run") != want:
            continue
        g = r.get("g_sim")
        if source == "kelvin_corr" and r.get("g_corr"):
            g = r["g_corr"]
        pts.append((float(r["xi0"]), float(r["wtB"]), float(g)))
    if not pts:
        sys.exit(f"no '{want}' rows in {csv_path}")
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="summary.csv")
    ap.add_argument("--source", default="control", choices=["control", "kelvin", "kelvin_corr"])
    ap.add_argument("--wtb", default=None,
                    help="comma-separated omega*tau_B values to include, e.g. 0.1,0.5,1,1.5,2")
    ap.add_argument("--xi0", default=None,
                    help="comma-separated xi0 values to include, e.g. 1,2.5,5,10,15,20")
    ap.add_argument("--vs", default="wtB", choices=["wtB", "xi0"],
                    help="x-axis: omega*tau_B (default) or xi0 (Bacri's Fig. 3 layout)")
    ap.add_argument("--join-static", action="store_true",
                    help="join the omega*tau_B = 0 (static-field) points into the AC trend lines")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    pts = load(args.csv, args.source)
    if args.wtb:
        keep = {float(v) for v in args.wtb.split(",")}
        pts = [p for p in pts if p[1] in keep]
    if args.xi0:
        keep = {float(v) for v in args.xi0.split(",")}
        pts = [p for p in pts if p[0] in keep]
    if not pts:
        sys.exit("no points left after --wtb/--xi0 filtering; check the values match the data")
    xi_vals = sorted({p[0] for p in pts})
    wt_vals = sorted({p[1] for p in pts})

    if args.vs == "wtB":
        groups = {xi: sorted((w, g) for x, w, g in pts if x == xi) for xi in xi_vals}
        fmt, xlabel = r"$\xi_0$ = {:g}", r"$\omega\tau_B$"
    else:
        groups = {w: sorted((x, g) for x, ww, g in pts if ww == w) for w in wt_vals}
        fmt, xlabel = r"$\omega\tau_B$ = {:g}", r"$\xi_0$"

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    cmap = plt.get_cmap("viridis")
    keys = sorted(groups)
    for n, key in enumerate(keys):
        xs, gs = zip(*groups[key])
        colour = cmap(n / max(len(keys) - 1, 1))
        if args.vs == "wtB" and not args.join_static:
            # omega*tau_B = 0 is a static field at H_d + H0, not the omega -> 0 limit of the
            # AC curve (which still sweeps through all field values), so keep it separate
            ac = [(x, g) for x, g in zip(xs, gs) if x > 0.0]
            dc = [(x, g) for x, g in zip(xs, gs) if x == 0.0]
            if len(ac) >= 2:
                ax_, ag_ = zip(*ac)
                xq = np.linspace(min(ax_), max(ax_), 300)
                ax.plot(xq, pchip(ax_, ag_, xq), "-", color=colour, lw=1.4)
                ax.plot(ax_, ag_, "o", color=colour, ms=5, label=fmt.format(key))
            for x, g in dc:
                ax.plot(x, g, "o", mfc="none", mec=colour, ms=6, mew=1.2)
        else:
            static_curve = args.vs == "xi0" and key == 0.0 and not args.join_static
            xq = np.linspace(min(xs), max(xs), 300)
            ax.plot(xq, pchip(xs, gs, xq), "--" if static_curve else "-", color=colour, lw=1.4)
            ax.plot(xs, gs, "o", color=colour, ms=5,
                    mfc="none" if static_curve else colour,
                    label=("static field ($\\omega\\tau_B$ = 0)" if static_curve
                           else fmt.format(key)))
        print(fmt.format(key).replace("$", "").replace("\\omega\\tau_B", "omega*tau_B")
              .replace("\\xi_0", "xi0") + ": " +
              ", ".join(f"({a:g}, {b:+.4f})" for a, b in zip(xs, gs)))

    if args.vs == "wtB" and not args.join_static and any(w == 0.0 for _, w, _ in pts):
        ax.plot([], [], "o", mfc="none", mec="k", label="static field ($\\omega\\tau_B$ = 0)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$g$")
    src = {"control": "control runs", "kelvin": "Kelvin-driven runs",
           "kelvin_corr": "Kelvin-driven runs, corrected"}[args.source]
    ax.set_title(f"Simulated rotational-viscosity factor ({src})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    out = args.out or f"g_vs_{args.vs}_{args.source}.png"
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
