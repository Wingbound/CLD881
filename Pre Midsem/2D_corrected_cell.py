"""
Copy this entire file into one Jupyter cell (or run it with %run).

It evaluates the 2D effective-field equation in Cartesian reduced-
magnetization variables and plots g = Delta_eta / (eta * phi).
"""

import numpy as np
import plotly.graph_objects as go
from scipy.special import ive


# ---------------------------------------------------------------------------
# 2D Langevin function and its inverse
# ---------------------------------------------------------------------------

def langevin_2d(x):
    """L_2(x) = I_1(x) / I_0(x), evaluated stably for scalar/array x."""
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    out = np.empty_like(x)
    small = ax < 1.0e-4

    # L_2(x) = x/2 - x^3/16 + x^5/96 + O(x^7)
    out[small] = x[small] / 2.0 - x[small] ** 3 / 16.0 + x[small] ** 5 / 96.0
    if np.any(~small):
        out[~small] = np.sign(x[~small]) * ive(1, ax[~small]) / ive(0, ax[~small])
    return out


def d_langevin_2d(x):
    """dL_2/dx = 1 - L_2/x - L_2^2, with its regular zero-field limit."""
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    out = np.empty_like(x)
    small = ax < 1.0e-3

    out[small] = 0.5 - 3.0 * x[small] ** 2 / 16.0 + 5.0 * x[small] ** 4 / 96.0
    if np.any(~small):
        L = langevin_2d(x[~small])
        out[~small] = 1.0 - L / x[~small] - L**2
    return out


# A tabulated inverse makes the many RK4 evaluations practical in a notebook.
# x=100 already gives L_2(x) approximately 0.995; the asymptotic tail below
# covers the unlikely case where an intermediate RK4 state is still closer to 1.
_INVERSE_X_TABLE = np.linspace(0.0, 100.0, 50001)
_INVERSE_L_TABLE = langevin_2d(_INVERSE_X_TABLE)


def inverse_langevin_2d(reduced_magnitude):
    """
    Return x >= 0 such that L_2(x) = reduced_magnitude.

    Linear interpolation on a dense monotone table is both stable and far
    faster than a Bessel/Newton solve at every RK4 substep.  The clipping only
    protects against a finite-RK4 overshoot beyond 0 <= |ell| < 1.
    """
    q = np.clip(np.asarray(reduced_magnitude, dtype=float), 0.0, 1.0 - 1.0e-12)
    table_limit = _INVERSE_L_TABLE[-1]
    x = np.interp(np.minimum(q, table_limit), _INVERSE_L_TABLE, _INVERSE_X_TABLE)
    tail = q > table_limit
    if np.any(tail):
        x[tail] = 1.0 / (2.0 * (1.0 - q[tail]))
    return x


# ---------------------------------------------------------------------------
# Cartesian 2D effective-field dynamics: Eq. (15) of CLD880_SI_2D_derivation
# ---------------------------------------------------------------------------

# Important dimensional point:
#
# The 3D Bacri perturbation equation is
#   tau_B F_dot = 1 - xi F / 2 * (1/L_3 - 1/xi_e).
# It must NOT be copied into 2D.  Applying the d=2 coefficient
# (1 - 2 L_2/xi_e) in Eq. (15) gives instead
#   tau_B F_dot = 1 - xi F / (2 L_2).
#
# Integrating ell below implements that regular 2D equation directly, without
# dividing by L_2 at field reversals.  It is therefore the canonical 2D path.

def rhs_effective_field_2d(t, ell, xi0, omega_tau, omega_vorticity):
    """
    Right-hand side for ell = L_2(|xi_e|) * xi_e / |xi_e|.

    `omega_vorticity` is Omega*tau_B.  The physical viscosity is the linear
    response as omega_vorticity -> 0; that derivative is evaluated by a
    symmetric pair of runs below.
    """
    xi = np.zeros_like(ell)
    xi[:, 0] = xi0 * np.cos(omega_tau * t)

    q = np.linalg.norm(ell, axis=1)
    xe = inverse_langevin_2d(q)

    # r = L_2(xe)/xe, including lim_{xe->0} r = 1/2.
    L_over_xe = np.empty_like(q)
    nonzero = xe > 1.0e-6
    L_over_xe[nonzero] = q[nonzero] / xe[nonzero]
    L_over_xe[~nonzero] = 0.5 - xe[~nonzero] ** 2 / 16.0

    # n = xi_e/|xi_e|.  Its value is immaterial at q=0 because the nonlinear
    # coefficient vanishes there; choose a fixed direction for continuity.
    n = np.zeros_like(ell)
    q_nonzero = q > 1.0e-12
    n[q_nonzero] = ell[q_nonzero] / q[q_nonzero, None]
    n[~q_nonzero, 0] = 1.0

    # [xi_e x (xi_e x xi)] / |xi_e|^2 = n (n . xi) - xi.
    double_cross_over_xe2 = n * np.sum(n * xi, axis=1)[:, None] - xi
    nonlinear_prefactor = 1.0 - 2.0 * L_over_xe

    # Omega z-hat x ell.
    rotation = omega_vorticity * np.column_stack((-ell[:, 1], ell[:, 0]))

    return (
        rotation
        - ell
        + L_over_xe[:, None] * xi
        - 0.5 * nonlinear_prefactor[:, None] * double_cross_over_xe2
    )


def rk4_step_2d(t, ell, h, xi0, omega_tau, omega_vorticity):
    """One RK4 update of the Cartesian 2D effective-field equation."""
    k1 = rhs_effective_field_2d(t, ell, xi0, omega_tau, omega_vorticity)
    k2 = rhs_effective_field_2d(t + h / 2.0, ell + h * k1 / 2.0,
                                 xi0, omega_tau, omega_vorticity)
    k3 = rhs_effective_field_2d(t + h / 2.0, ell + h * k2 / 2.0,
                                 xi0, omega_tau, omega_vorticity)
    k4 = rhs_effective_field_2d(t + h, ell + h * k3,
                                 xi0, omega_tau, omega_vorticity)
    ell_next = ell + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    # The exact state obeys |ell|<1.  Prevent a rare explicit-RK4 intermediate
    # overshoot from entering the nonphysical inverse-Langevin branch.
    magnitude = np.linalg.norm(ell_next, axis=1)
    overshot = magnitude >= 1.0 - 1.0e-10
    if np.any(overshot):
        ell_next[overshot] *= ((1.0 - 1.0e-10) / magnitude[overshot])[:, None]
    return ell_next


def torque_average_2d(xi0, omega_tau, omega_vorticity,
                      steps_per_period=800, settle_periods=15,
                      average_periods=3):
    """Return <xi0 cos(omega_tau t) ell_y> after reaching periodic steady state."""
    xi0 = np.asarray(xi0, dtype=float)
    period = 2.0 * np.pi / omega_tau
    h = period / steps_per_period
    ell = np.zeros((xi0.size, 2), dtype=float)
    ell[:, 0] = langevin_2d(xi0)
    t = 0.0

    for _ in range(settle_periods * steps_per_period):
        ell = rk4_step_2d(t, ell, h, xi0, omega_tau, omega_vorticity)
        t += h

    integral = np.zeros_like(xi0)
    for _ in range(average_periods * steps_per_period):
        # RK4-consistent quadrature: use the midpoint of each completed step.
        ell_next = rk4_step_2d(t, ell, h, xi0, omega_tau, omega_vorticity)
        ell_mid = 0.5 * (ell + ell_next)
        integral += h * xi0 * np.cos(omega_tau * (t + h / 2.0)) * ell_mid[:, 1]
        ell = ell_next
        t += h

    return integral / (average_periods * period)


def compute_g_2d(xi0, omega_tau, omega_vorticity=1.0e-4,
                 steps_per_period=800, settle_periods=15, average_periods=3):
    """
    Compute g = Delta_eta/(eta*phi) using the symmetric linear response:

        g = (1/2) <xi0 cos(omega_tau t) d ell_y / d(Omega*tau_B)>.
    """
    if omega_tau <= 0.0:
        raise ValueError("omega_tau must be positive.")
    if omega_vorticity <= 0.0:
        raise ValueError("omega_vorticity must be positive.")

    plus = torque_average_2d(
        xi0, omega_tau, +omega_vorticity,
        steps_per_period, settle_periods, average_periods,
    )
    minus = torque_average_2d(
        xi0, omega_tau, -omega_vorticity,
        steps_per_period, settle_periods, average_periods,
    )
    return 0.5 * (plus - minus) / (2.0 * omega_vorticity)


# ---------------------------------------------------------------------------
# Surface requested for direct comparison with the 3D Bacri plot
# ---------------------------------------------------------------------------

xi0_values = np.linspace(0.0, 20.0, 61)
omega_tau_values = np.linspace(0.1, 2.0, 61)

XI0, OMEGA_TAU = np.meshgrid(xi0_values, omega_tau_values)
g_surface_2d = np.zeros_like(XI0)

print("Computing 2D Cartesian effective-field RK4 surface...")
for i, omega_tau in enumerate(omega_tau_values):
    g_surface_2d[i] = compute_g_2d(xi0_values, omega_tau)
    print(f"omega*tau_B = {omega_tau:.3f}")
print("Done.")

fig = go.Figure()
fig.add_trace(
    go.Surface(
        x=XI0,
        y=OMEGA_TAU,
        z=g_surface_2d,
        colorscale="Turbo",
        cmin=-0.4,
        cmax=0.9,
        opacity=0.95,
        colorbar=dict(title="g"),
        hovertemplate=(
            "<b>2D effective-field RK4</b><br>"
            "xi0 = %{x:.2f}<br>"
            "omega*tau_B = %{y:.2f}<br>"
            "g = %{z:.5f}<extra></extra>"
        ),
    )
)
fig.update_traces(contours_z=dict(show=True, usecolormap=True, project_z=True))
fig.update_layout(
    title="2D Negative-Viscosity Surface: Cartesian Effective-Field RK4",
    scene=dict(
        xaxis=dict(title="xi0", backgroundcolor="white", gridcolor="lightgray", showbackground=True),
        yaxis=dict(title="omega*tau_B", backgroundcolor="white", gridcolor="lightgray", showbackground=True),
        zaxis=dict(title="g = Delta_eta/(eta phi)", range=[-0.4, 0.9], backgroundcolor="white", gridcolor="lightgray", showbackground=True),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.6, y=-1.5, z=1.2)),
    ),
    width=1000,
    height=800,
    margin=dict(l=10, r=10, b=10, t=50),
)
fig.show()
