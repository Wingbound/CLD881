#!/usr/bin/env python3
"""
Analyse the C-test (oscillating-field morphology) runs.

Reads output_morphology/run_*_AC_* and _DC_* summary.csv files and reports:
  C1  tail-mean D for AC vs DC (does oscillation change the shape?)
  C2  the microscopic within-cycle story from the dense AC run:
        - phase lag of the drop magnetisation M behind the applied field H
        - tip-vs-equator magnetisation ratio (the elongation driver)
        - how deformation D and magnetic stress breathe over the cycle
  C3  tail-mean D vs omega*tau_B (frequency dependence)

Run from the folder that contains output_morphology/.
Produces C_results.png.
"""
import os, re, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE="output_morphology"; TAIL=0.5; tau_B=20.0; H0=1.73
pat=re.compile(r"run_MS([0-9.eE+-]+)_FB([0-9.eE+-]+)_R([0-9.]+)_xi([0-9.eE+-]+)_wtB([0-9.eE+-]+)_(AC|DC)(?:_(.*))?$")

runs=[]
for d in glob.glob(os.path.join(BASE,"run_*")):
    m=pat.search(os.path.basename(d))
    sp=os.path.join(d,"summary.csv")
    if not m or not os.path.exists(sp): continue
    df=pd.read_csv(sp)
    if df.empty: continue
    runs.append(dict(dir=d, Ms=float(m.group(1)), Fb=float(m.group(2)),
                     R=float(m.group(3)), xi0=float(m.group(4)),
                     wtB=float(m.group(5)), mode=m.group(6),
                     tag=m.group(7) or "", df=df))
if not runs:
    raise SystemExit(f"No C-test summary.csv under {BASE}/")

def tail(df):
    n=df.iter.max(); return df[df.iter>=TAIL*n]

def lag_deg(df, wtB):
    """phase lag of Mx_in behind Happ, by sinusoidal fit on the tail."""
    t=tail(df); om=wtB/tau_B; th=om*t.iter.values
    A=np.column_stack([np.cos(th),np.sin(th)])
    def fit(y):
        (a,b),*_=np.linalg.lstsq(A,y,rcond=None); return a,b
    aH,bH=fit(t.Happ.values); aM,bM=fit(t.Mx_in.values)
    phH=np.arctan2(bH,aH); phM=np.arctan2(bM,aM)
    return np.degrees((phH-phM+np.pi)%(2*np.pi)-np.pi)

# ---------------- C1: AC vs DC ----------------
print("="*60); print("C1  --  AC vs DC (tail-mean deformation)")
c1={}
for r in runs:
    if r["tag"].startswith("c1") or r["tag"]=="" :
        c1.setdefault(r["mode"], r)
for r in runs:
    if r["tag"].startswith("c1"): c1[r["mode"]]=r
for mode in ("DC","AC"):
    if mode in c1:
        t=tail(c1[mode]["df"])
        print(f"  {mode}:  D_mean={t.D.mean():.5f}  D_std={t.D.std():.5f}  "
              f"(wtB={c1[mode]['wtB']}, {len(t)} tail pts)")
if "AC" in c1 and "DC" in c1:
    dAC=tail(c1['AC']['df']).D.mean(); dDC=tail(c1['DC']['df']).D.mean()
    print(f"  --> AC/DC deformation ratio = {dAC/dDC:.3f}  "
          f"({'AC deforms less' if dAC<dDC else 'AC deforms more'})")

# ---------------- C2: microscopic ----------------
print("="*60); print("C2  --  microscopic within-cycle (dense AC run)")
c2=[r for r in runs if r["tag"].startswith("c2")]
if not c2:  # fall back to any dense AC run
    c2=[r for r in runs if r["mode"]=="AC" and len(r["df"])>200]
micro=c2[0] if c2 else None
if micro:
    df=micro["df"]; t=tail(df)
    L=lag_deg(df, micro["wtB"])
    tipeq=(t.Mtip/t.Meq).mean()
    print(f"  M lags applied field by {L:.1f} deg")
    print(f"  mean |M|_tip / |M|_eq = {tipeq:.4f}  "
          f"({'poles more magnetised -> elongation' if tipeq>1 else 'check'})")
    print(f"  D breathes: mean={t.D.mean():.4f}  amplitude~{(t.D.max()-t.D.min())/2:.4f}")
    print(f"  magstress breathes: mean={t.magstress.mean():.3e}  "
          f"max={t.magstress.max():.3e}")
else:
    print("  (no dense AC run found -- run c2)")

# ---------------- C3: frequency sweep ----------------
print("="*60); print("C3  --  frequency dependence")
c3=sorted([r for r in runs if r["tag"].startswith("c3")], key=lambda r:r["wtB"])
if c3:
    print("   wtB     D_mean    D_std    M-lag(deg)")
    for r in c3:
        t=tail(r["df"]); print(f"  {r['wtB']:.2g}   {t.D.mean():.5f}  "
                               f"{t.D.std():.5f}   {lag_deg(r['df'],r['wtB']):.1f}")
else:
    print("  (no c3 runs found)")

# ---------------- plots ----------------
fig=plt.figure(figsize=(13,8)); gs=fig.add_gridspec(2,2,hspace=.33,wspace=.26)
CF,CL="#2a6f97","#bb3e03"

# C1 bar
ax=fig.add_subplot(gs[0,0])
if "AC" in c1 and "DC" in c1:
    for k,(mode,c) in enumerate([("DC",CF),("AC",CL)]):
        t=tail(c1[mode]["df"])
        ax.bar(k,t.D.mean(),yerr=t.D.std(),color=c,capsize=5,width=.6)
    ax.set_xticks([0,1]); ax.set_xticklabels(["DC","AC"])
ax.set(ylabel="tail-mean $D$",title="C1 — AC vs DC deformation"); ax.grid(alpha=.3,axis="y")

# C2 within-cycle: Happ and Mx_in vs iter (shows the lag)
ax=fig.add_subplot(gs[0,1])
if micro:
    t=tail(df).copy()
    t=t[t.iter<=t.iter.min()+6*(2*np.pi*tau_B/micro["wtB"])]  # ~6 periods
    ax.plot(t.iter,t.Happ/H0,color=CF,label=r"$H_{app}/H_0$")
    ax.plot(t.iter,t.Mx_in/ (np.abs(t.Mx_in).max()+1e-30),color=CL,label=r"$M_x$ (norm.)")
    ax.legend(fontsize=8)
ax.set(xlabel="iter",title="C2 — $M$ lags the oscillating field"); ax.grid(alpha=.3)

# C2 tip vs equator over cycle
ax=fig.add_subplot(gs[1,0])
if micro:
    ax.plot(t.iter,t.Mtip,color="#2d6a4f",label=r"$|M|$ tip")
    ax.plot(t.iter,t.Meq,color="#9a5b13",ls="--",label=r"$|M|$ equator")
    ax.legend(fontsize=8)
ax.set(xlabel="iter",ylabel="$|M|$",title="C2 — poles vs equator magnetisation"); ax.grid(alpha=.3)

# C3 D and lag vs wtB
ax=fig.add_subplot(gs[1,1])
if c3:
    w=[r["wtB"] for r in c3]; Dm=[tail(r["df"]).D.mean() for r in c3]
    lg=[lag_deg(r["df"],r["wtB"]) for r in c3]
    ax.plot(w,Dm,"o-",color=CL,label="$D$ mean"); ax.set_xlabel(r"$\omega\tau_B$")
    ax.set_ylabel("tail-mean $D$",color=CL)
    ax2=ax.twinx(); ax2.plot(w,lg,"s--",color=CF,label="M-lag")
    ax2.set_ylabel("M lag (deg)",color=CF)
ax.set_title("C3 — frequency dependence"); ax.grid(alpha=.3)

fig.suptitle("C-tests: oscillating-field droplet morphology (microscopic)",fontsize=13,y=.99)
fig.savefig("C_results.png",dpi=150,bbox_inches="tight")
print("="*60); print("saved C_results.png")
