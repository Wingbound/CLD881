#!/usr/bin/env python3
"""
Correct Kelvin-driven runs for the extra magnetic body force produced by the
transverse applied field  H_y = -k (y - y_c).

That component is required by  div H = 0  once  dHx/dx = k.  With a lagging M it
makes  <M x H>  vary linearly in y, so  F = (1/2) curl(mu0 M x H)  adds a uniform
force  eps * G  on top of the Kelvin drive  G = mu0 k <Mx>.  It does not depend on
the vorticity, so it is a body force, not a viscosity.

    python3 correct_kelvin.py summary.csv           # adds eps and g_corr columns
    python3 correct_kelvin.py --case 10 20 2 0.3 9.35779e-05

The single-point model reproduces the measured eps to about 7% (systematically high).
"""
import csv, sys
import numpy as np

ETA_H, PHI_P, H_REF, MU0 = 0.05, 0.2, 1.73, 1.0

L   = lambda x: x/3 - x**3/45 if abs(x) < 1e-3 else 1/np.tanh(x) - 1/x
dLf = lambda x: 1/3 - x*x/15 if abs(x) < 1e-3 else 1/x**2 - 1/np.sinh(min(abs(x), 300))**2


def mean_torque(tauB, Ms, xi0, wtB, Hd, H0, k, Y, Om=0.0, nper=400, NSUB=8, dt=1.0):
    """Period-averaged mu0 (M x H)_z and <Mx> at transverse offset Y, zero vorticity."""
    om = wtB / tauB
    xph = xi0 / H_REF
    per = 2 * np.pi / om
    ntr, nav = int(40 * tauB + 10 * per) + 1, int(nper * per + 0.5)
    ex, ey, my = xph * (Hd + H0), 0.0, 0.0
    hxp, hyp, sT, sM = Hd + H0, -k * Y, 0.0, 0.0
    h = dt / NSUB
    for n in range(1, ntr + nav + 1):
        hx, hy = Hd + H0 * np.cos(om * n * dt), -k * Y - my
        def rhs(s, ax, ay):
            f = s / dt
            hxx, hyy = (1 - f) * hxp + f * hx, (1 - f) * hyp + f * hy
            xtx, xty = xph * hxx, xph * hyy
            xe = np.hypot(ax, ay)
            if xe < 1e-10:
                return (-Om * ay - (ax - xtx) / tauB, Om * ax - (ay - xty) / tauB)
            nx, ny, Le, dLe = ax / xe, ay / xe, L(xe), dLf(xe)
            xpar = nx * xtx + ny * xty
            ppx, ppy = xtx - nx * xpar, xty - ny * xpar
            par, cp = -(Le / (dLe * tauB)) * (1 - xpar / xe), 0.5 * (xe / Le - 1) / tauB
            return (-Om * ay + par * nx + cp * ppx, Om * ax + par * ny + cp * ppy)
        s = 0.0
        for _ in range(NSUB):
            k1 = rhs(s, ex, ey)
            k2 = rhs(s + h/2, ex + h/2*k1[0], ey + h/2*k1[1])
            k3 = rhs(s + h/2, ex + h/2*k2[0], ey + h/2*k2[1])
            k4 = rhs(s + h,   ex + h*k3[0],   ey + h*k3[1])
            ex += h/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            ey += h/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
            s += h
        xe = np.hypot(ex, ey)
        mx, my = (Ms*ex/3, Ms*ey/3) if xe < 1e-10 else (Ms*L(xe)*ex/xe, Ms*L(xe)*ey/xe)
        if n > ntr:
            sT += MU0 * (mx * hy - my * hx)
            sM += mx
        hxp, hyp = hx, hy
    return sT / nav, sM / nav


def eps_of(xi0, tauB, wtB, r, k, Y0=8.0):
    """Extra force as a fraction of the Kelvin drive: F_extra = eps * G."""
    Ms = 3 * ETA_H * PHI_P * xi0 / (tauB * H_REF)
    Hd, H0 = r * H_REF, H_REF
    Tp, Mxm = mean_torque(tauB, Ms, xi0, wtB, Hd, H0, k, +Y0)
    Tm, _   = mean_torque(tauB, Ms, xi0, wtB, Hd, H0, k, -Y0)
    G = MU0 * k * Mxm
    return 0.5 * (Tp - Tm) / (2 * Y0) / G


def main():
    if "--case" in sys.argv:
        xi0, tauB, wtB, r, k = (float(v) for v in sys.argv[sys.argv.index("--case") + 1:][:5])
        print(f"eps = {eps_of(xi0, tauB, wtB, r, k):.4f}")
        return
    src = sys.argv[1]
    rows = list(csv.DictReader(open(src)))
    out = []
    for row in rows:
        k = float(row["k_grad"])
        if row["run"] != "kelvin" or k == 0.0 or float(row["wtB"]) == 0.0:
            row["eps"], row["g_corr"] = "", ""
            out.append(row); continue
        r = float(row["H_d"]) / float(row["H0"])
        eps = eps_of(float(row["xi0"]), float(row["tau_B"]), float(row["wtB"]), r, k)
        eta_corr = float(row["eta_eff"]) * (1.0 + eps)
        row["eps"] = f"{eps:.5f}"
        row["g_corr"] = f"{(eta_corr/ETA_H - 1.0)/(1.5*PHI_P):.5f}"
        print(f"xi0={row['xi0']:>5} wtB={row['wtB']:>5}  eps={eps:+.4f}  "
              f"g_sim={float(row['g_sim']):+.4f} -> g_corr={float(row['g_corr']):+.4f}  "
              f"(g_ref={float(row['g_ref']):+.4f})")
        out.append(row)
    dst = src.replace(".csv", "_corrected.csv")
    with open(dst, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print("wrote", dst)


if __name__ == "__main__":
    main()
