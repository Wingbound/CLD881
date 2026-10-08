#!/usr/bin/env python3
"""
Analyse the E-tests (extended oscillating-field droplet morphology).

  E1  amplitude x frequency map  -> heatmap of tail-mean D(xi0, wtB)
  E2  harmonic content of D(t)   -> FFT of the dense centre run; 2w breathing
  E3  deformation-field lag       -> phase of D(t) vs H^2(t)
  E4  rotating field              -> D and orientation vs frequency
  E5  DC+AC bias                  -> D vs DC fraction b
  E6  AC + flow                   -> D vs body force

Reads output_morphology/run_*_{AC,ROT,BIAS}_* summary.csv.
Run from the folder containing output_morphology/.  Writes E_results.png.
"""
import os, re, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE="output_morphology"; TAIL=0.5; tau_B=20.0; H0=1.73
pat=re.compile(r"run_MS([0-9.eE+-]+)_FB([0-9.eE+-]+)_R([0-9.]+)_xi([0-9.eE+-]+)_wtB([0-9.eE+-]+)_(DC|AC|ROT|BIAS)(?:_(.*))?$")

runs=[]
for d in glob.glob(os.path.join(BASE,"run_*")):
    m=pat.search(os.path.basename(d)); sp=os.path.join(d,"summary.csv")
    if not m or not os.path.exists(sp): continue
    df=pd.read_csv(sp)
    if df.empty: continue
    runs.append(dict(Ms=float(m.group(1)),Fb=float(m.group(2)),R=float(m.group(3)),
                     xi0=float(m.group(4)),wtB=float(m.group(5)),mode=m.group(6),
                     tag=m.group(7) or "",df=df,n=df.iter.max()))
if not runs: raise SystemExit(f"No E-test summary.csv under {BASE}/")

def diverged(r):
    """True if the run blew up (any NaN) or ended at exact-zero D (NaN->0)."""
    d=r["df"]
    if not np.isfinite(d.D.values).all(): return True
    if np.isfinite(d.D.values).any() and np.nanmax(np.abs(d.D.values[-3:]))==0.0: return True
    return False

# drop diverged runs up front so they never pollute any panel
_before=len(runs)
runs=[r for r in runs if not diverged(r)]
if len(runs)<_before:
    print(f"[dropped {_before-len(runs)} diverged run(s) from the analysis]")

def tail(df): n=df.iter.max(); return df[df.iter>=TAIL*n]
def pick(pred):  # longest matching run
    c=[r for r in runs if pred(r)]
    return max(c,key=lambda r:r["n"]) if c else None
def allmatch(pred): return sorted([r for r in runs if pred(r)],key=lambda r:(r["xi0"],r["wtB"],r["Fb"]))

fig=plt.figure(figsize=(14,9)); gs=fig.add_gridspec(2,3,hspace=.34,wspace=.34)
CF,CL,CA="#2a6f97","#bb3e03","#2d6a4f"

# ---------- E1: amplitude x frequency heatmap ----------
print("="*60); print("E1  amplitude x frequency map  (tail-mean D)")
e1=allmatch(lambda r: r["mode"]=="AC" and r["Fb"]==0 and r["tag"].startswith("e1"))
ax=fig.add_subplot(gs[0,0])
if e1:
    xis=sorted({r["xi0"] for r in e1}); wts=sorted({r["wtB"] for r in e1})
    Z=np.full((len(xis),len(wts)),np.nan)
    for r in e1:
        Z[xis.index(r["xi0"]),wts.index(r["wtB"])]=tail(r["df"]).D.mean()
    im=ax.imshow(Z,origin="lower",aspect="auto",cmap="viridis",
                 extent=[min(wts),max(wts),min(xis),max(xis)])
    fig.colorbar(im,ax=ax,label=r"$\bar D$")
    print("   xi0 \\ wtB:",wts)
    for i,x in enumerate(xis): print(f"   xi0={x:<4g} "+" ".join(f"{v:.3f}" if np.isfinite(v) else " -- " for v in Z[i]))
ax.set(xlabel=r"$\omega\tau_B$",ylabel=r"$\xi_0$",title="E1 — $\\bar D(\\xi_0,\\omega\\tau_B)$")

# ---------- E2/E3: harmonic content of D(t) ----------
print("="*60); print("E2/E3  harmonic content of D(t)  (dense centre run)")
dense=pick(lambda r: r["mode"]=="AC" and len(r["df"])>300)
ax=fig.add_subplot(gs[0,1]); ax2=fig.add_subplot(gs[0,2])
if dense:
    t=tail(dense["df"]).copy(); it=t.iter.values.astype(float)
    Dt=t.D.values-t.D.values.mean()
    om=dense["wtB"]/tau_B
    # uniform resample (output is uniform already), FFT
    dt=np.median(np.diff(it)); freqs=np.fft.rfftfreq(len(Dt),d=dt); amp=np.abs(np.fft.rfft(Dt))
    # express frequency in units of the drive omega
    fdrive=om/(2*np.pi)
    fnorm=freqs/ (fdrive if fdrive>0 else 1)
    ax.plot(fnorm,amp,color=CL); ax.set_xlim(0,5)
    ax.set(xlabel=r"frequency / $\omega$",ylabel="|FFT of D(t)|",
           title="E2 — D(t) spectrum (peak at $2\\omega$)")
    # report the 2w/1w/DC content
    def nearest(fr):
        i=np.argmin(np.abs(fnorm-fr)); return amp[i]
    print(f"   drive omega={om:.4f}; spectral peaks (relative):")
    print(f"     at 1w: {nearest(1):.3e}   at 2w: {nearest(2):.3e}   at 4w: {nearest(4):.3e}")
    print(f"     -> {'2w dominant (M^2 forcing)' if nearest(2)>nearest(1) else '1w dominant'}")
    # E3: D(t) vs H^2(t) phase
    H2=(H0*np.cos(om*it))**2; H2-=H2.mean()
    A=np.column_stack([np.cos(2*om*it),np.sin(2*om*it)])  # H^2 has a 2w part
    (aD,bD),*_=np.linalg.lstsq(A,Dt,rcond=None)
    (aH,bH),*_=np.linalg.lstsq(A,H2,rcond=None)
    lagD=np.degrees((np.arctan2(bD,aD)-np.arctan2(bH,aH)+np.pi)%(2*np.pi)-np.pi)
    print(f"   E3: D(t) lags H^2(t) by {lagD:.1f} deg  (interface response delay)")
    ax2.plot(it,H2/np.abs(H2).max(),color=CF,label=r"$H^2$ (norm)")
    ax2.plot(it,Dt/np.abs(Dt).max(),color=CL,label=r"$D$ (norm)")
    ax2.legend(fontsize=8); ax2.set(xlabel="iter",title=f"E3 — D lags $H^2$ by {lagD:.0f}°")
    ax2.set_xlim(it.min(),it.min()+6*2*np.pi/om)
else:
    ax.set_title("E2 — (no dense AC run)"); ax2.set_title("E3 — (no dense AC run)")

# ---------- E4: rotating field ----------
print("="*60); print("E4  rotating field")
def hit_boundary(r):
    """True if the drop touched the box over most of its converged tail."""
    t=tail(r["df"])
    return ("bdy" in t) and (t.bdy.mean() > 0.3)
# dedup by frequency; keep lowest Ms (stable regime); EXCLUDE boundary-contact
# runs -- a rotating drop wanders and can wall out; those D are finite-size
# artifacts, not physics (same discipline as the B-tests' bdy flag).
# valid ROT runs = stable (not diverged, already filtered) AND bdy=0.
valid=[r for r in runs if r["mode"]=="ROT" and not hit_boundary(r)]
n_excl=len([r for r in runs if r["mode"]=="ROT" and hit_boundary(r)])
if n_excl: print(f"   [excluded {n_excl} ROT run(s) for boundary contact]")
# Use ONE Ms for the curve: pick the Ms that has the most valid frequencies,
# so we never silently mix field strengths on one axis.
ax=fig.add_subplot(gs[1,0])
if valid:
    from collections import Counter
    best_ms=Counter(r["Ms"] for r in valid).most_common(1)[0][0]
    e4=sorted([r for r in valid if r["Ms"]==best_ms], key=lambda r:r["wtB"])
    w=[r["wtB"] for r in e4]; Dm=[tail(r["df"]).D.mean() for r in e4]
    angm=[np.abs(tail(r["df"]).ang).mean() for r in e4]
    ax.plot(w,Dm,"o-",color=CL,label=r"$\bar D$")
    print(f"   (Ms={best_ms:g}, bdy=0 only, {len(e4)} valid freq)")
    print("   wtB   D_mean   |ang|_mean")
    for r,dd,aa in zip(e4,Dm,angm): print(f"  {r['wtB']:<4g} {dd:.4f}   {aa:.1f}")
    ttl=f"E4 — rotating field: $\\bar D$ vs freq (Ms={best_ms:g})"
    if len(e4)<2: ttl+="  [thin: enlarge box for more freq]"
else:
    ttl="E4 — rotating field (all runs hit boundary; enlarge box)"
    print("   WARNING: every ROT run hit the boundary -- enlarge the box (nX,nY)")
ax.set(xlabel=r"$\omega\tau_B$",ylabel=r"$\bar D$",title=ttl)

# ---------- E5: DC+AC bias ----------
print("="*60); print("E5  DC + AC bias")
e5=sorted([r for r in runs if r["mode"]=="BIAS"],key=lambda r:r["tag"])
ax=fig.add_subplot(gs[1,1])
if e5:
    # recover bias fraction from tag e5_bias0.4
    def bval(r):
        m=re.search(r"bias([0-9.]+)",r["tag"]); return float(m.group(1)) if m else np.nan
    bs=[bval(r) for r in e5]; Dm=[tail(r["df"]).D.mean() for r in e5]
    order=np.argsort(bs); bs=np.array(bs)[order]; Dm=np.array(Dm)[order]
    ax.plot(bs,Dm,"o-",color=CA)
    print("   bias(DC frac)   D_mean");
    for b,d in zip(bs,Dm): print(f"   {b:.2f}          {d:.4f}")
ax.set(xlabel="DC fraction $b$",ylabel=r"$\bar D$",title="E5 — DC+AC: $\\bar D$ vs bias")

# ---------- E6: AC + flow ----------
print("="*60); print("E6  AC + flow")
e6=allmatch(lambda r: r["mode"]=="AC" and r["Fb"]>0)
ax=fig.add_subplot(gs[1,2])
if e6:
    fb=[r["Fb"] for r in e6]; Dm=[tail(r["df"]).D.mean() for r in e6]
    angm=[np.abs(tail(r["df"]).ang).mean() for r in e6]
    ax.plot(fb,Dm,"o-",color=CL); ax.set_xscale("log")
    print("   Fb      D_mean   |ang|_mean")
    for r,dd,aa in zip(e6,Dm,angm): print(f"  {r['Fb']:.0e} {dd:.4f}   {aa:.1f}")
ax.set(xlabel=r"$F_b$ (flow)",ylabel=r"$\bar D$",title="E6 — AC + flow: $\\bar D$ vs $F_b$")

fig.suptitle("E-tests: extended oscillating-field droplet morphology",fontsize=14,y=.99)
fig.savefig("E_results.png",dpi=150,bbox_inches="tight")
print("="*60); print("saved E_results.png")
