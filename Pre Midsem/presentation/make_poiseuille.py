#!/usr/bin/env python3
"""Slide 12: regenerate the Poiseuille base-flow validation figure from CSV.

Zero-field case (xi0=0, wtB=0) -> pure body-force-driven Poiseuille flow.
LBM mean profile (averaged over x) vs the analytic parabola.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = ("../poiseuille_h_dynamic/output_poiseuille/"
       "output_xi0_0.000000_wtB_0.000000/output_t_100000.csv")
OUT = "figs/poiseuille_base.png"

df = pd.read_csv(CSV)
# average ux over x (i) at each wall-normal position (j)
prof = df.groupby("j")["ux"].mean().sort_index()
y = prof.index.values.astype(float)
u = prof.values.astype(float)

# Halfway bounce-back: true no-slip walls sit half a node beyond the first/last
# fluid nodes, i.e. at y=-1/2 and y=(Ny-1)+1/2.  Poiseuille shape is then
# s(y) = (y+1/2)(yW-y+1/2), amplitude fixed by least squares.
yW = y.max()                                   # last fluid node index
shape = (y + 0.5) * (yW - y + 0.5)
A = np.dot(u, shape) / np.dot(shape, shape)    # least-squares amplitude

y_fit = np.linspace(-0.5, yW + 0.5, 400)
u_theory = A * (y_fit + 0.5) * (yW - y_fit + 0.5)

u_th_nodes = A * shape
umax = u.max()
rms = np.sqrt(np.mean((u - u_th_nodes)**2))
rms_rel = rms / umax * 100.0

fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.plot(u_theory, y_fit, "k-", lw=2,
        label=r"Poiseuille theory  $u\propto y(H-y)$")
ax.plot(u, y, "o", color="#c0392b", ms=6, mfc="#e74c3c", mec="#7b241c",
        label="LBM (mean over $x$)")
ax.set_xlabel(r"Velocity  $u_x$  (lattice units)")
ax.set_ylabel(r"Channel position  $y$")
ax.set_title(r"Poiseuille base-flow validation  ($H_0 = 0$)")
ax.grid(alpha=0.3)
ax.legend(loc="center left", frameon=True, fontsize=9)
ax.text(0.02, 0.03,
        f"RMS residual = {rms_rel:.2f}% of $u_{{max}}$",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.6"))
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}: Hwall={yW+1:.0f}, umax={umax:.3e}, RMS={rms_rel:.3f}%")
