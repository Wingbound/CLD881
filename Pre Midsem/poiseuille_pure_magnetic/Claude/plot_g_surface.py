#!/usr/bin/env python3
"""
3D surface of the rotational-viscosity factor g over the (xi0, omega*tau_B) plane.

Builds the surface on the LEAN grid  xi0 in {1,2.5,5,7.5,10,12.5,15,20},
omega*tau_B in {0,0.5,1,1.5,2}, where every cell has a validated control pair and
tau_B = 20 throughout (no demagnetisation drift).

Two outputs:
  * interactive Plotly HTML  (rotate, hover for values), with the g = 0 plane shown
  * static matplotlib PNG    (3D render + companion heatmap) for the report

Which g:
  --surface control      the fluid's true g, from control g_sim              (default)
  --surface kelvin_corr  the corrected Kelvin (pumped-flow) g, from g_corr
When --surface kelvin_corr, the control g is drawn as validation dots on the same axes.

    python3 plot_g_surface.py
    python3 plot_g_surface.py --surface kelvin_corr
    python3 plot_g_surface.py --wtb-max 2 --html g_surf.html --png g_surf.png

Needs summary.csv (and, for kelvin_corr, summary_corrected.csv from correct_kelvin.py).
"""
import argparse, csv, os, sys
import numpy as np

XI_GRID = [1, 2.5, 5, 7.5, 10, 12.5, 15, 20]
WT_GRID = [0, 0.5, 1.0, 1.5, 2.0]


def load(csv_path, run_want, value):
    pts = {}
    for r in csv.DictReader(open(csv_path)):
        if r.get("run") != run_want:
            continue
        v = r.get(value, "")
        if v == "":
            continue
        pts[(float(r["xi0"]), float(r["wtB"]))] = float(v)
    return pts


def grid_from(pts, xi_grid, wt_grid, label):
    Z = np.full((len(wt_grid), len(xi_grid)), np.nan)
    missing = []
    for iw, w in enumerate(wt_grid):
        for ix, x in enumerate(xi_grid):
            if (x, w) in pts:
                Z[iw, ix] = pts[(x, w)]
            else:
                missing.append((x, w))
    if missing:
        print(f"WARNING [{label}]: {len(missing)} grid cells missing: {missing}")
        print("  (surface will interpolate across them; run those cells for a clean grid)")
        # linear fill along wtB so the surface is drawable
        for ix in range(len(xi_grid)):
            col = Z[:, ix]; m = ~np.isnan(col)
            if m.sum() >= 2:
                Z[:, ix] = np.interp(wt_grid, np.array(wt_grid)[m], col[m])
    return Z


def make_html(XI, WT, Z, title, ctrl_pts, out):
    import plotly.graph_objects as go
    X, Y = np.meshgrid(XI, WT)
    fig = go.Figure()
    fig.add_surface(
        x=XI, y=WT, z=Z, colorscale="RdBu", reversescale=True,
        cmid=0.0, colorbar=dict(title="g"),
        contours={"z": {"show": True, "start": -0.5, "end": 0.9, "size": 0.1,
                        "usecolormap": True}},
        hovertemplate="xi0=%{x}<br>omega*tau_B=%{y}<br>g=%{z:.3f}<extra></extra>",
    )
    # g = 0 plane
    fig.add_surface(x=XI, y=WT, z=np.zeros_like(Z),
                    opacity=0.25, showscale=False, colorscale=[[0, "gray"], [1, "gray"]],
                    hoverinfo="skip")
    # validation dots (control g) if provided
    if ctrl_pts is not None:
        cx = [p[0] for p in ctrl_pts]; cy = [p[1] for p in ctrl_pts]; cz = [p[2] for p in ctrl_pts]
        fig.add_scatter3d(x=cx, y=cy, z=cz, mode="markers",
                          marker=dict(size=3, color="black"),
                          name="control g (validation)",
                          hovertemplate="xi0=%{x}<br>wtB=%{y}<br>control g=%{z:.3f}<extra></extra>")
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="xi0", yaxis_title="omega*tau_B", zaxis_title="g",
            camera=dict(eye=dict(x=1.7, y=-1.6, z=0.9)),
        ),
        width=950, height=750, margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out, include_plotlyjs="cdn")
    print("wrote", out)


def make_png(XI, WT, Z, title, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    X, Y = np.meshgrid(XI, WT)
    vmax = np.nanmax(np.abs(Z))
    norm = plt.Normalize(-vmax, vmax)

    fig = plt.figure(figsize=(14, 6.2))

    # 3D surface
    ax = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    surf = ax.plot_surface(X, Y, Z, facecolors=cm.RdBu_r(norm(Z)),
                           rstride=1, cstride=1, linewidth=0.3,
                           edgecolor="0.3", antialiased=True, shade=False)
    # g = 0 translucent plane
    ax.plot_surface(X, Y, np.zeros_like(Z), color="gray", alpha=0.20,
                    rstride=1, cstride=1, linewidth=0)
    # zero contour projected onto the base for the onset boundary
    try:
        ax.contour(X, Y, Z, levels=[0], colors="k", linewidths=2,
                   offset=np.nanmin(Z), zdir="z")
    except Exception:
        pass
    ax.set_xlabel(r"$\xi_0$"); ax.set_ylabel(r"$\omega\tau_B$"); ax.set_zlabel("$g$")
    ax.view_init(elev=28, azim=-120)
    ax.set_box_aspect((1.4,1,0.8))
    ax.set_title("surface", fontsize=10)
    m = cm.ScalarMappable(norm=norm, cmap="RdBu_r"); m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.6, pad=0.1, label="g")

    # companion heatmap with g=0 contour
    ax2 = fig.add_subplot(1, 2, 2)
    def edges(a):
        a = np.asarray(a, float)
        return np.concatenate([[a[0]-(a[1]-a[0])/2], (a[:-1]+a[1:])/2, [a[-1]+(a[-1]-a[-2])/2]])
    pm = ax2.pcolormesh(edges(XI), edges(WT), Z, cmap="RdBu_r", norm=norm, shading="flat")
    cs = ax2.contour(XI, WT, Z, levels=[0], colors="k", linewidths=2)
    ax2.clabel(cs, fmt="g = 0", fontsize=9)
    ax2.set_xlabel(r"$\xi_0$"); ax2.set_ylabel(r"$\omega\tau_B$")
    ax2.set_title("heatmap (read values here)", fontsize=10)
    fig.colorbar(pm, ax=ax2, label="g")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=180)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="summary.csv")
    ap.add_argument("--surface", default="control", choices=["control", "kelvin_corr"])
    ap.add_argument("--wtb-max", type=float, default=2.0)
    ap.add_argument("--html", default=None)
    ap.add_argument("--png", default=None)
    args = ap.parse_args()

    WT = [w for w in WT_GRID if w <= args.wtb_max + 1e-9]
    XI = XI_GRID

    if args.surface == "control":
        pts = load(args.csv, "control", "g_sim")
        Z = grid_from(pts, XI, WT, "control g")
        title = "Rotational-viscosity factor g (fluid, control runs)"
        ctrl_pts = None
    else:
        corr = args.csv.replace(".csv", "_corrected.csv")
        if not os.path.exists(corr):
            sys.exit(f"{corr} not found: run  correct_kelvin.py {args.csv}  first")
        # g_corr on kelvin rows; static (wtB=0) rows have blank g_corr, fall back to g_sim
        pts = {}
        for r in csv.DictReader(open(corr)):
            if r.get("run") != "kelvin":
                continue
            v = r.get("g_corr") or r.get("g_sim")
            if v:
                pts[(float(r["xi0"]), float(r["wtB"]))] = float(v)
        Z = grid_from(pts, XI, WT, "corrected kelvin g")
        title = "Corrected Kelvin (pumped-flow) g, with control g as validation dots"
        cpts = load(args.csv, "control", "g_sim")
        ctrl_pts = [(x, w, cpts[(x, w)]) for x in XI for w in WT if (x, w) in cpts]

    html = args.html or f"g_surface_{args.surface}.html"
    png = args.png or f"g_surface_{args.surface}.png"
    make_html(XI, WT, Z, title, ctrl_pts, html)
    make_png(XI, WT, Z, title, png)


if __name__ == "__main__":
    main()
