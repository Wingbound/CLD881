#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# ----------------------------------------------------------------------
# 1. Configuration & Parameters
# ----------------------------------------------------------------------
BASE_DIR = Path("output_poiseuille")

# SPECIFY YOUR PARAMETERS HERE
TARGET_H0 = 0.000000
TARGET_OMEGA = 0.000000

# Physical parameters from phaseField_poiseuille.cpp
Fx = 1e-4              # Body force applied in simulation

# Lattice / fluid properties
cs2 = 1.0 / 3.0        # Speed of sound squared
rho_l = 1.0
rho_h = 1.975
eta_l = 0.0025
eta_h = 0.05

# ----------------------------------------------------------------------
# 2. Corrected Kinematic Viscosity (matches update_Sf in simulation)
# ----------------------------------------------------------------------
def effective_nu(phi, rho_l, rho_h, eta_l, eta_h, cs2):
    """
    Compute the kinematic viscosity actually used by the MRT scheme.
    This mimics the logic of update_Sf() for a given phase field phi.
    """
    term_l = rho_l * cs2 / eta_l      # dt = 1
    term_h = rho_h * cs2 / eta_h
    term   = term_l + phi * (term_h - term_l)
    s_v    = term / (1.0 + 0.5 * term)    # Sf[7] = Sf[1] = Sf[2] = Sf[8] in the code
    nu     = cs2 * (1.0 / s_v - 0.5)
    return nu

# Since phi is initialised to 1.0 everywhere, we use phi = 1.0
nu_sim = effective_nu(1.0, rho_l, rho_h, eta_l, eta_h, cs2)

# ----------------------------------------------------------------------
# 3. Path Construction (matches C++ output folder naming)
# ----------------------------------------------------------------------
def get_path_for_params(h0, omega):
    """Constructs the folder path based on C++ directory naming logic."""
    folder_name = f"output_{h0:.6f}_{omega:.6f}"
    folder_path = BASE_DIR / folder_name
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Directory not found: {folder_path}")
    
    # Try to find the final output file (adjust filename if you use a different output interval)
    files = sorted(glob.glob(str(folder_path / "output_t_100000.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
        
    return Path(files[-1])

# ----------------------------------------------------------------------
# 4. Data Loading & Analytical Formula
# ----------------------------------------------------------------------
def load_profile(csv_path):
    # df = pd.read_csv(csv_path)
    df = pd.read_csv(csv_path, skipinitialspace=True)
    nX = df["i"].max() + 1
    nY = df["j"].max() + 1
    
    # Extract vertical slice from the middle of the channel
    mid_x = nX // 2
    slice_data = df[df['i'] == mid_x].sort_values('j')
    
    return nY, slice_data['j'].values, slice_data['ux'].values

def analytical_poiseuille(y, F, rho, nu, H):
    """Formula for mid-grid bounce-back (walls at -0.5 and H-0.5)."""
    return (F / (2.0 * rho * nu)) * (y + 0.5) * (H - 0.5 - y)

# ----------------------------------------------------------------------
# 5. Execution and Visualization
# ----------------------------------------------------------------------
try:
    csv_file = get_path_for_params(TARGET_H0, TARGET_OMEGA)
    H, y_sim, u_sim = load_profile(csv_file)
    
    print(f"Plotting results for H0={TARGET_H0}, Omega={TARGET_OMEGA}")
    print(f"Source: {csv_file}")
    print(f"Theoretical nu (from MRT relaxation): {nu_sim:.4f}")
    print(f"Max Velocity (Sim): {np.max(u_sim):.6f}")

    # Generate analytical curve
    y_fine = np.linspace(0, H - 1, 500)
    rho_sim = rho_l + 1.0 * (rho_h - rho_l) # Since phi=1.0
    u_theory = analytical_poiseuille(y_fine, Fx, rho_sim, nu_sim, H)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(u_theory, y_fine, 'k--', lw=2, label=f'Analytical (nu={nu_sim:.4f})')
    plt.scatter(u_sim, y_sim, color='royalblue', s=25, alpha=0.7, label='LBM Simulation')

    plt.xlabel("Velocity $u_x$", fontsize=12)
    plt.ylabel("Lattice Position $y$", fontsize=12)
    plt.title(f"Velocity Profile: $H_0={TARGET_H0}$, $\omega={TARGET_OMEGA}$", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error: {e}")


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.special import iv

# --- 1. Constants (from phaseField_poiseuille.cpp) ---
nX, nY = 100, 25
phi_val = 1.0
F_px = 1e-4
tauB = 4.0
Href = 126.1585
Ms = 40.0 / Href
chi0 = 2.2

# Fluid properties
rho_l, rho_h = 1.0, 1.975
eta_l, eta_h = 0.0025, 0.05

# Effective base properties since phi = 1.0 everywhere initially
rho_base = rho_l + phi_val * (rho_h - rho_l) # 1.975
eta_base = eta_l + phi_val * (eta_h - eta_l) # 0.05

def poiseuille_theory(y, eta_eff):
    """
    Theoretical Poiseuille flow profile. 
    y bounds are -0.5 and nY-0.5 due to mid-grid bounce back.
    """
    return (F_px / (2.0 * eta_eff)) * (y + 0.5) * (nY - 0.5 - y)

def calculate_g_theory_2D(H0_real, wtB):
    """
    2D rotational viscosity coefficient g, consistent with the phase-field LBM simulation.
    H0_real is the physical field strength (in kA/m).
    """
    H0_real_arr = np.asarray(H0_real, dtype=float)
    H0_lbm = H0_real_arr / Href
    
    # xi = 2.0 * chi0 * H0_lbm / Ms
    xi = H0_real
    
    L_2D = np.where(xi < 1e-8, 0.5*xi, iv(1, xi) / iv(0, xi))
    
    # g_lbm = Delta_eta_lbm / (phi * eta_base)
    g_lbm = (tauB * Ms * L_2D * H0_lbm * (1.0 - wtB**2)) / (4.0 * eta_base * (1.0 + wtB**2)**2)
    
    # The classical Shliomis definition for g includes a 3/2 factor multiplier.
    g_physical = g_lbm * 1.5
    
    return np.nan_to_num(g_physical)

# --- 2. Simulation Data Processing ---
base_path = "output_poiseuille"
results = []

if os.path.exists(base_path):
    for folder in os.listdir(base_path):
        if not folder.startswith("output_"):
            continue
            
        try:
            parts = folder.split('_')
            # Folder name format: output_{H0*Href}_{omega*tauB}
            h0_real = float(parts[1])
            wtB = float(parts[2])
            
            f_csv = os.path.join(base_path, folder, "output_t_100000.csv")
            if os.path.exists(f_csv):
                df = pd.read_csv(f_csv, skipinitialspace=True)
                
                # Average ux over x direction (i) to get y-profile
                u_profile = df.groupby('j')['ux'].mean().values
                
                # Fit the profile to find the effective viscosity
                popt, _ = curve_fit(poiseuille_theory, np.arange(nY), u_profile,
                                    p0=[eta_base], maxfev=5000, bounds=(1e-7, 2.0))
                
                # Calculate the simulated rotational viscosity coefficient 'g'
                g_sim_lbm = (popt[0] - eta_base) / (phi_val * eta_base)
                
                # Multiply by 1.5 to map to the classical Shliomis g definition
                g_sim_physical = g_sim_lbm * 1.5
                
                results.append({'H0_real': h0_real, 'wtB': wtB, 'g': g_sim_physical})
        except Exception as e:
            print(f"Skipping {folder}: {e}")
            continue

df = pd.DataFrame(results)

# --- 3. Plot ---
plt.figure(figsize=(10, 7))
H0_fine = np.linspace(0, 20, 300) # physical H0 values from 0 to 20 kA/m

if not df.empty:
    unique_wtBs = sorted(df['wtB'].unique())
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, max(1, len(unique_wtBs))))

    for i, wtB in enumerate(unique_wtBs):
        c = colors[i]
        # g_th = calculate_g_theory_2D(H0_fine, wtB)
        # plt.plot(H0_fine, g_th, color=c, linewidth=2, alpha=0.8,
        #          label=r'Theory $\omega\tau_B$=' + f'{wtB:.1f}')
        
        sub = df[df['wtB'] == wtB]
        clean = sub[np.abs(sub['g']) < 1.5]
        plt.scatter(clean['H0_real'], clean['g'], color=c, s=80, edgecolors='k', zorder=5, label=r'Sim $\omega\tau_B$=' + f'{wtB:.1f}')

else:
    print("No complete simulation data found yet. Plotting theoretical curves.")
    for wtB, c in zip([0.1, 0.5, 1.0, 1.5, 2.0], ['royalblue', 'cyan', 'yellow', 'orange', 'red']):
        g_th = calculate_g_theory_2D(H0_fine, wtB)
        plt.plot(H0_fine, g_th, color=c, linewidth=2, alpha=0.8, label=r'Theory $\omega\tau_B=$' + f'{wtB:.1f}')

plt.axhline(0, color='black', linewidth=1, alpha=0.5, linestyle='--')
plt.xlabel(r'Field Strength $H_0$ (kA/m)', fontsize=13)
plt.ylabel(r'Rotational Viscosity Coefficient $g$', fontsize=13)
plt.title('Validation: LBM Simulation', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(-0.3, 0.6)
plt.tight_layout()
plt.savefig("validation_plot.png")
print("Saved plot to validation_plot.png")


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.special import iv

# --- 1. Constants (from phaseField_poiseuille.cpp) ---
nX, nY = 100, 25
phi_val = 1.0
F_px = 1e-4
tauB = 4.0
Href = 126.1585
Ms = 40.0 / Href
chi0 = 2.2

# Fluid properties
rho_l, rho_h = 1.0, 1.975
eta_l, eta_h = 0.0025, 0.05

# Effective base properties since phi = 1.0 everywhere initially
rho_base = rho_l + phi_val * (rho_h - rho_l) # 1.975
eta_base = eta_l + phi_val * (eta_h - eta_l) # 0.05

def poiseuille_theory(y, eta_eff):
    """
    Theoretical Poiseuille flow profile. 
    y bounds are -0.5 and nY-0.5 due to mid-grid bounce back.
    """
    return (F_px / (2.0 * eta_eff)) * (y + 0.5) * (nY - 0.5 - y)

def calculate_g_theory_2D(H0_real, wtB):
    """
    2D rotational viscosity coefficient g, consistent with the phase-field LBM simulation.
    H0_real is the physical field strength (in kA/m).
    """
    H0_real_arr = np.asarray(H0_real, dtype=float)
    H0_lbm = H0_real_arr / Href
    
    # xi = 2.0 * chi0 * H0_lbm / Ms
    xi = H0_real
    
    L_2D = np.where(xi < 1e-8, 0.5*xi, iv(1, xi) / iv(0, xi))
    
    # g_lbm = Delta_eta_lbm / (phi * eta_base)
    g_lbm = (tauB * Ms * L_2D * H0_lbm * (1.0 - wtB**2)) / (4.0 * eta_base * (1.0 + wtB**2)**2)
    
    # The classical Shliomis definition for g includes a 3/2 factor multiplier.
    g_physical = g_lbm * 1.5
    
    return np.nan_to_num(g_physical)

# --- 2. Simulation Data Processing ---
base_path = "output_poiseuille_updated"
results = []

if os.path.exists(base_path):
    for folder in os.listdir(base_path):
        if not folder.startswith("output_"):
            continue
            
        try:
            parts = folder.split('_')
            # Folder name format: output_{H0*Href}_{omega*tauB}
            h0_real = float(parts[1])
            wtB = float(parts[2])
            
            f_csv = os.path.join(base_path, folder, "output_t_100000.csv")
            if os.path.exists(f_csv):
                df = pd.read_csv(f_csv, skipinitialspace=True)
                
                # Average ux over x direction (i) to get y-profile
                u_profile = df.groupby('j')['ux'].mean().values
                
                # Fit the profile to find the effective viscosity
                popt, _ = curve_fit(poiseuille_theory, np.arange(nY), u_profile,
                                    p0=[eta_base], maxfev=5000, bounds=(1e-7, 2.0))
                
                # Calculate the simulated rotational viscosity coefficient 'g'
                g_sim_lbm = (popt[0] - eta_base) / (phi_val * eta_base)
                
                # Multiply by 1.5 to map to the classical Shliomis g definition
                g_sim_physical = g_sim_lbm * 1.5
                
                results.append({'H0_real': h0_real, 'wtB': wtB, 'g': g_sim_physical})
        except Exception as e:
            print(f"Skipping {folder}: {e}")
            continue

df = pd.DataFrame(results)

# --- 3. Plot ---
plt.figure(figsize=(10, 7))
H0_fine = np.linspace(0, 20, 300) # physical H0 values from 0 to 20 kA/m

if not df.empty:
    unique_wtBs = sorted(df['wtB'].unique())
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, max(1, len(unique_wtBs))))

    for i, wtB in enumerate(unique_wtBs):
        c = colors[i]
        # g_th = calculate_g_theory_2D(H0_fine, wtB)
        # plt.plot(H0_fine, g_th, color=c, linewidth=2, alpha=0.8,
        #          label=r'Theory $\omega\tau_B$=' + f'{wtB:.1f}')
        
        sub = df[df['wtB'] == wtB]
        clean = sub[np.abs(sub['g']) < 1.5]
        plt.scatter(clean['H0_real'], clean['g'], color=c, s=80, edgecolors='k', zorder=5, label=r'Sim $\omega\tau_B$=' + f'{wtB:.1f}')

else:
    print("No complete simulation data found yet. Plotting theoretical curves.")
    for wtB, c in zip([0.1, 0.5, 1.0, 1.5, 2.0], ['royalblue', 'cyan', 'yellow', 'orange', 'red']):
        g_th = calculate_g_theory_2D(H0_fine, wtB)
        plt.plot(H0_fine, g_th, color=c, linewidth=2, alpha=0.8, label=r'Theory $\omega\tau_B=$' + f'{wtB:.1f}')

plt.axhline(0, color='black', linewidth=1, alpha=0.5, linestyle='--')
plt.xlabel(r'Field Strength $H_0$ (kA/m)', fontsize=13)
plt.ylabel(r'Rotational Viscosity Coefficient $g$', fontsize=13)
plt.title('Validation: LBM Simulation', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(-0.3, 0.6)
plt.tight_layout()
plt.savefig("validation_plot_updated.png")
print("Saved plot to validation_plot_updated.png")

