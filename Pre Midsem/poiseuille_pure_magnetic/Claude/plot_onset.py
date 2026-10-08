#!/usr/bin/env python3
"""
Plot the negative-viscosity ONSET curve: the omega*tau_B at which the fluid's g first
crosses from positive to negative, as a function of xi0.

g here is the fluid's true rotational-viscosity factor, read from g_ref_nodemag (the
demagnetisation-free single-point reference the solver writes on EVERY row, control or
kelvin).  That is the right quantity for onset -- the raw Kelvin g_sim is contaminated by
the transverse-stress term and must NOT be used here.

    python3 plot_onset.py                       # from summary.csv, column g_ref_nodemag
    python3 plot_onset.py --value g_ref         # include the planar demag shift
    python3 plot_onset.py --csv summary.csv -o onset.png

Output: onset omega*tau_B vs xi0, with
  * the region below the curve shaded (positive g, ordinary viscosity),
  * the region above shaded (negative g, the interesting regime),
  * the critical xi0 (weakest coupling that admits any onset) marked,
  * the minimum of the onset curve annotated (easiest operating point).
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(csv_path, value):
    """Return {xi0: {wtB: g}} from the chosen column, merging control and kelvin rows
    (they carry the same reference value; kelvin g_sim is ignored)."""
    d = {}
    for r in csv.DictReader(open(csv_path)):
        v = r.get(value, "")
        if v == "":
            continue
        x, w = float(r["xi0"]), float(r["wtB"])
        d.setdefault(x, {})[w] = float(v)   # last one wins; ref value is identical per (x,w)
    if not d:
        sys.exit(f"no rows with column '{value}' in {csv_path}")
    return d


def onset(row):
    """First positive->negative crossing in wtB; returns (wtB_onset, bracket_gap) or (None, None)."""
    ws = sorted(row)
    gs = [row[w] for w in ws]
    for i in range(len(ws) - 1):
        if gs[i] > 0 >= gs[i + 1]:
            t = gs[i] / (gs[i] - gs[i + 1])
            return ws[i] + t * (ws[i + 1] - ws[i]), ws[i + 1] - ws[i]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="summary.csv")
    ap.add_argument("--value", default="g_ref_nodemag",
                    choices=["g_ref_nodemag", "g_ref"],
                    help="fluid g column to read; g_ref_nodemag (default) is demag-free")
    ap.add_argument("-o", "--out", default="onset_curve.png")
    args = ap.parse_args()

    data = load(args.csv, args.value)
    xis, onsets, gaps, tight = [], [], [], []
    no_onset = []
    for x in sorted(data):
        z, gap = onset(data[x])
        if z is None:
            no_onset.append(x)
        else:
            xis.append(x); onsets.append(z); gaps.append(gap); tight.append(gap <= 0.11)
    if len(xis) < 2:
        sys.exit("fewer than two xi0 have an onset crossing; need a wider omega*tau_B range")
    xis = np.array(xis); onsets = np.array(onsets); tight = np.array(tight)

    # critical xi0: between the largest no-onset xi0 and the smallest with an onset
    x_first = xis.min()
    x_crit_lo = max([x for x in no_onset if x < x_first], default=None)

    # minimum of the onset curve (parabola through the three lowest points)
    imin = int(onsets.argmin())
    xmin = zmin = None
    if 0 < imin < len(xis) - 1:
        co = np.polyfit(xis[imin-1:imin+2], onsets[imin-1:imin+2], 2)
        xmin = -co[1] / (2 * co[0]); zmin = np.polyval(co, xmin)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    # shade regimes using the onset curve as the boundary
    xfine = np.linspace(xis.min(), xis.max(), 400)
    ofine = np.interp(xfine, xis, onsets)
    ymax = max(onsets.max() * 1.25, onsets.max() + 0.15)
    ymin = onsets.min() - 0.15
    ax.fill_between(xfine, ymin, ofine, color="#d9e8f5", alpha=0.8, zorder=0)
    ax.fill_between(xfine, ofine, ymax, color="#f7dede", alpha=0.8, zorder=0)
    ax.text(xis.max() * 0.62, ymin + 0.04, "positive $g$  (ordinary viscosity)",
            fontsize=9, color="#2e6fb0", va="bottom")
    ax.text(xis.max() * 0.5, ymax - 0.03, "negative $g$  (negative rotational viscosity)",
            fontsize=9, color="#c0392b", va="top")

    # the onset curve itself
    ax.plot(xis, onsets, "-", color="k", lw=1.6, zorder=3)
    ax.plot(xis[tight], onsets[tight], "o", color="k", ms=6, zorder=4,
            label="onset (tight bracket $\\leq 0.1$)")
    if (~tight).any():
        ax.plot(xis[~tight], onsets[~tight], "o", mfc="white", mec="k", ms=6, zorder=4,
                label="onset (loose bracket)")

    # minimum marker
    if xmin is not None:
        ax.plot(xmin, zmin, "*", color="crimson", ms=16, zorder=5)
        ax.annotate(f"easiest onset\n$\\xi_0\\approx{xmin:.1f}$, $\\omega\\tau_B\\approx{zmin:.2f}$",
                    xy=(xmin, zmin), xytext=(xmin + 3, zmin - 0.08),
                    fontsize=9, color="crimson",
                    arrowprops=dict(arrowstyle="->", color="crimson"))

    # critical xi0 band
    if x_crit_lo is not None:
        ax.axvspan(x_crit_lo, x_first, color="0.5", alpha=0.18, zorder=1)
        ax.text((x_crit_lo + x_first) / 2, ymax - 0.10,
                f"critical $\\xi_0$\nin ({x_crit_lo:g}, {x_first:g})",
                ha="center", va="top", fontsize=8, color="0.3")

    ax.set_xlim(max(0, (x_crit_lo or xis.min()) - 0.5), xis.max() + 0.5)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\xi_0$")
    ax.set_ylabel(r"onset $\omega\tau_B$  (where $g$ first goes negative)")
    ax.set_title("Onset of negative rotational viscosity", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.25, zorder=2)

    fig.tight_layout()
    fig.savefig(args.out, dpi=190)
    print("wrote", args.out)
    print(f"\nxi0 with no onset in range: {sorted(no_onset)}")
    print(f"critical xi0 bracketed in: ({x_crit_lo}, {x_first})" if x_crit_lo else
          f"critical xi0 <= {x_first} (no smaller xi0 sampled)")
    if xmin is not None:
        print(f"onset-curve minimum: omega*tau_B = {zmin:.3f} at xi0 ~ {xmin:.1f}")
    print(f"{'xi0':>6} {'onset':>8} {'bracket':>8}")
    for x, z, g in zip(xis, onsets, gaps):
        print(f"{x:6g} {z:8.3f} {g:8.2f}")


if __name__ == "__main__":
    main()
