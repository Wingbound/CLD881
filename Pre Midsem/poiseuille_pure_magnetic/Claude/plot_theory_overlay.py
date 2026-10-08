#!/usr/bin/env python3
"""
Validation overlay: LBM-measured g vs the single-point MRSh (Bacri Eqs. 7-9) theory,
both at the same biased AC field.  If the simulation reproduces the theory the dotted
curves lie on the solid ones.

  solid  : g measured from the 2D LBM flow          (control g_sim, the pressure-driven
           measurement; use --measured kelvin_corr for the corrected magnetic-drive g)
  dotted : g from the solver's single-point reference (g_ref_nodemag by default) -- this IS
           Bacri's effective-field theory integrated at H_d + H0 cos(wt), computed
           independently of the flow.

Same colour per omega*tau_B as the onset figure; every curve anchored at the origin.

    python3 plot_theory_overlay.py
    python3 plot_theory_overlay.py --wtb 0.1,0.5,1.0,1.4,1.5,1.6,1.7,2.0
    python3 plot_theory_overlay.py --measured kelvin_corr --theory g_ref
    python3 plot_theory_overlay.py -o overlay.png

--measured  control (default, g_sim) | kelvin_corr (g_corr from summary_corrected.csv)
--theory    g_ref_nodemag (default) | g_ref     which reference column is the dotted line
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MARK_XI = [0, 1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]


def pchip(x, y, xq):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 2:
        return np.full_like(xq, y[0] if n else np.nan)
    h = np.diff(x); d = np.diff(y)/h; m = np.zeros(n)
    if n == 2:
        m[:] = d[0]
    else:
        for i in range(1, n-1):
            if d[i-1]*d[i] > 0:
                w1, w2 = 2*h[i]+h[i-1], h[i]+2*h[i-1]
                m[i] = (w1+w2)/(w1/d[i-1]+w2/d[i])
        for i, (h0, h1, dd, dn) in ((0, (h[0], h[1] if n > 2 else h[0], d[0], d[min(1, len(d)-1)])),
                                    (n-1, (h[-1], h[-2], d[-1], d[-2]))):
            mi = ((2*h0+h1)*dd - h0*dn)/(h0+h1)
            if np.sign(mi) != np.sign(dd): mi = 0.0
            elif np.sign(dd) != np.sign(dn) and abs(mi) > abs(3*dd): mi = 3*dd
            m[i] = mi
    idx = np.clip(np.searchsorted(x, xq)-1, 0, n-2)
    t = (xq-x[idx])/h[idx]
    return ((2*t**3-3*t**2+1)*y[idx] + (t**3-2*t**2+t)*h[idx]*m[idx]
            + (-2*t**3+3*t**2)*y[idx+1] + (t**3-t**2)*h[idx]*m[idx+1])


def load_col(csv_path, run_want, value):
    d = {}
    for r in csv.DictReader(open(csv_path)):
        if r.get("run") != run_want:
            continue
        v = r.get(value, "")
        if v != "":
            d.setdefault(float(r["wtB"]), {})[float(r["xi0"])] = float(v)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="summary.csv")
    ap.add_argument("--measured", default="control", choices=["control", "kelvin_corr"])
    ap.add_argument("--theory", default="g_ref_nodemag", choices=["g_ref_nodemag", "g_ref"])
    ap.add_argument("--wtb", default=None)
    ap.add_argument("-o", "--out", default="theory_overlay.png")
    args = ap.parse_args()

    # measured (solid)
    if args.measured == "control":
        meas = load_col(args.csv, "control", "g_sim")
        meas_label = "LBM (pressure-driven control)"
    else:
        corr = args.csv.replace(".csv", "_corrected.csv")
        if not os.path.exists(corr):
            sys.exit(f"{corr} not found: run correct_kelvin.py first")
        meas = {}
        for r in csv.DictReader(open(corr)):
            if r.get("run") == "kelvin" and (r.get("g_corr") or r.get("g_sim")):
                meas.setdefault(float(r["wtB"]), {})[float(r["xi0"])] = float(r.get("g_corr") or r["g_sim"])
        meas_label = "LBM (magnetic-drive, corrected)"

    # theory (dotted) -- from the kelvin rows, which carry the reference on every cell
    theo = load_col(args.csv, "kelvin", args.theory)
    theo_label = f"MRSh theory ({args.theory})"

    wts = sorted(set(meas) & set(theo))
    if args.wtb:
        keep = {float(v) for v in args.wtb.split(",")}
        wts = [w for w in wts if w in keep]
    if not wts:
        sys.exit("no common omega*tau_B between measured and theory")

    fig, ax = plt.subplots(figsize=(9, 6.2))
    cmap = plt.get_cmap("viridis")
    maxdev = 0.0
    for n, w in enumerate(wts):
        col = cmap(n/max(len(wts)-1, 1))
        # solid: measured
        mp = meas[w]
        xs = [0.0]+sorted(mp); gs = [0.0]+[mp[x] for x in sorted(mp)]
        xq = np.linspace(0, max(xs), 400)
        ax.plot(xq, pchip(np.array(xs), np.array(gs), xq), "-", color=col, lw=1.6, zorder=3)
        mx = [x for x in MARK_XI if x == 0 or x in mp]
        ax.plot(mx, [0 if x == 0 else mp[x] for x in mx], "o", color=col, ms=4, zorder=4,
                label=f"{w:g}")
        # dotted: theory
        tp = theo[w]
        xs2 = [0.0]+sorted(tp); gs2 = [0.0]+[tp[x] for x in sorted(tp)]
        xq2 = np.linspace(0, max(xs2), 400)
        ax.plot(xq2, pchip(np.array(xs2), np.array(gs2), xq2), ":", color=col, lw=1.6, zorder=2)
        # track max deviation where both exist
        for x in set(mp) & set(tp):
            maxdev = max(maxdev, abs(mp[x]-tp[x]))

    # proxy legend entries for line styles
    ax.plot([], [], "-", color="0.3", lw=1.6, label=meas_label)
    ax.plot([], [], ":", color="0.3", lw=1.6, label=theo_label)

    ax.axhline(0, color="k", lw=1.0)
    ax.set_xlabel(r"$\xi_0$", fontsize=12); ax.set_ylabel(r"$g$", fontsize=12)
    ax.set_title("LBM simulation (solid) vs single-point MRSh theory (dotted), same field",
                 fontsize=11)
    ax.set_xlim(-0.5, 20.8)
    ax.grid(alpha=0.3)
    ax.legend(title=r"$\omega\tau_B$", fontsize=8, ncol=2, framealpha=0.9, loc="upper left")
    ax.text(0.98, 0.02, f"max |LBM - theory| = {maxdev:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}   (max |measured - theory| = {maxdev:.4f} over {len(wts)} frequencies)")


if __name__ == "__main__":
    main()
