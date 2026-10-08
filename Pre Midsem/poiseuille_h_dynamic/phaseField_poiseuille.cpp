#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using namespace std;

// Simulation setup parameters
int nSteps = 100000;
int outputInterval = 10000;
const int nX = 100;
const int nY = 25;

// Lattice parameters
const int dt = 1; // Time step (lattice units)
const int dx = 1; // Lattice spacing (lattice units)

double c = dx / dt; // Lattice speed
const int Q = 9;
int cX[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
int cY[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
double w[Q] = {4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0, 1.0 / 9.0,
               1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};
const int opp[Q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
double c_s = c / sqrt(3.0);
double c_s2 = c_s * c_s;
double c_s4 = c_s2 * c_s2;
double c_s_inv = 1.0 / c_s;
double c_s_inv2 = 1.0 / c_s2;
double c_s_inv4 = 1.0 / c_s4;

const double Lambda = 1.0;

// ---- dimensionless targets: these define the run ---------------------
double xi0        = 20.0;   // Langevin parameter at the field peak
double omega_tauB = 0.0;    // dimensionless frequency  omega*tau_B
double phi_p      = 0.2;    // PARTICLE volume fraction (the theory's phi)

// ---- numerical scales ------------------------------------------------
double tau_B = 20.0;                // Brownian time, in lattice steps
double omega = omega_tauB / tau_B;  // field angular frequency
// double H0    = 0.1;                 // field amplitude (lattice units)
double H0x_amp = 0.1, H0y_amp = 0.0;   // set direction via amplitude ratio
double H0 = sqrt(pow(H0x_amp,2.0)+pow(H0y_amp,2.0));
inline double H0x(int iter) { return H0x_amp * cos(omega * iter); }
inline double H0y(int iter) { return H0y_amp * cos(omega * iter); }

double M_phi   = 0.1;
double epsilon = 4.0;
double chi0    = 2.2;
double mu0     = 1.0;
double D       = 4.0;
double sigma   = 0.00191875;
double Beta    = 12.0 * sigma / D;
double kappa   = 1.5 * D * sigma;
double rho_l   = 1.0;
double rho_h   = 1.975;
double eta_l   = 0.0025;
double eta_h   = 0.05;

// ---- DERIVED: fixed by  xi = mu0 tau_B Ms H / (4 eta phi_p) ----------
double Ms       = 4.0 * eta_h * phi_p * xi0 / (mu0 * tau_B * H0);
double xi_per_H = xi0 / H0;   // Langevin argument per unit field

// MRT Matrices
double M[Q][Q] = {{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                  {-4.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0},
                  {4.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0},
                  {0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0},
                  {0.0, -2.0, 0.0, 2.0, 0.0, 1.0, -1.0, -1.0, 1.0},
                  {0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0},
                  {0.0, 0.0, -2.0, 0.0, 2.0, 1.0, 1.0, -1.0, -1.00},
                  {0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0},
                  {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0}};

double Minv[Q][Q] = {
    {1.0 / 9.0, -1.0 / 9.0, 1.0 / 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.0 / 9.0, -1.0 / 36.0, -1.0 / 18.0, 1.0 / 6.0, -1.0 / 6.0, 0.0, 0.0,
     1.0 / 4.0, 0.0},
    {1.0 / 9.0, -1.0 / 36.0, -1.0 / 18.0, 0.0, 0.0, 1.0 / 6.0, -1.0 / 6.0,
     -1.0 / 4.0, 0.0},
    {1.0 / 9.0, -1.0 / 36.0, -1.0 / 18.0, -1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0,
     1.0 / 4.0, 0.0},
    {1.0 / 9.0, -1.0 / 36.0, -1.0 / 18.0, 0.0, 0.0, -1.0 / 6.0, 1.0 / 6.0,
     -1.0 / 4.0, 0.0},
    {1.0 / 9.0, 1.0 / 18.0, 1.0 / 36.0, 1.0 / 6.0, 1.0 / 12.0, 1.0 / 6.0,
     1.0 / 12.0, 0.0, 1.0 / 4.0},
    {1.0 / 9.0, 1.0 / 18.0, 1.0 / 36.0, -1.0 / 6.0, -1.0 / 12.0, 1.0 / 6.0,
     1.0 / 12.0, 0.0, -1.0 / 4.0},
    {1.0 / 9.0, 1.0 / 18.0, 1.0 / 36.0, -1.0 / 6.0, -1.0 / 12.0, -1.0 / 6.0,
     -1.0 / 12.0, 0.0, 1.0 / 4.0},
    {1.0 / 9.0, 1.0 / 18.0, 1.0 / 36.0, 1.0 / 6.0, 1.0 / 12.0, -1.0 / 6.0,
     -1.0 / 12.0, 0.0, -1.0 / 4.0}};

// Relaxation matrix (g)
double Sg3 = 1 / (M_phi * c_s_inv2 / dt + 0.5);
double Sg[Q] = {1.0, 1.1, 1.1, Sg3, Sg3, Sg3, Sg3, 1.2, 1.2};
double MinvSgM[Q][Q];

// Relaxation matrix (f)
double Sf1 = 1.0;
double Sf4 = 8 * (2.0 - Sf1) / (8.0 - Sf1);
double Sf[Q] = {0.0, Sf1, Sf1, 0.0, Sf4, 0.0, Sf4, Sf1, Sf1};
double ISf[Q] = {1.0 - 0.5 * Sf[0], 1.0 - 0.5 * Sf[1], 1.0 - 0.5 * Sf[2],
                 1.0 - 0.5 * Sf[3], 1.0 - 0.5 * Sf[4], 1.0 - 0.5 * Sf[5],
                 1.0 - 0.5 * Sf[6], 1.0 - 0.5 * Sf[7], 1.0 - 0.5 * Sf[8]};
double MinvSfM[Q][Q];
double MinvISfM[Q][Q];

// Relaxation matrix (h)
double Sh3 = 1 / (epsilon * mu0 * c_s_inv2 / dt + 0.5);
double tau_h = 1/Sh3;
double Sh[Q] = {1.0, 1.1, 1.1, Sh3, Sh3, Sh3, Sh3, 1.2, 1.2};
double MinvShM[Q][Q];

// LBM Distributions
static double f[nX][nY][Q];
static double feq[nX][nY][Q];
static double fstar[nX][nY][Q];
static double ftemp[nX][nY][Q];
static double g[nX][nY][Q];
static double geq[nX][nY][Q];
static double gstar[nX][nY][Q];
static double gtemp[nX][nY][Q];
static double h[nX][nY][Q];
static double heq[nX][nY][Q];
static double hstar[nX][nY][Q];
static double htemp[nX][nY][Q];
static double F_alpha[nX][nY][Q];

// Macroscopic variables
static double rho[nX][nY];
static double ux[nX][nY];
static double uy[nX][nY];
static double p[nX][nY];
static double vorticity_z[nX][nY];

static double phi[nX][nY];
static double ncap_x[nX][nY];
static double ncap_y[nX][nY];
static double mu_phi[nX][nY];

static double psi[nX][nY];
static double Hx[nX][nY];
static double Hy[nX][nY];
static double Mx[nX][nY];
static double My[nX][nY];
static double xie_x[nX][nY];        // effective field, x component
static double xie_y[nX][nY];        // effective field, y component

static double Fm_x[nX][nY];
static double Fm_y[nX][nY];
static double Fs_x[nX][nY];
static double Fs_y[nX][nY];
static double Fb_x[nX][nY];
static double Fb_y[nX][nY];

void matrix_multiply(double A[Q][Q], double B[Q], double C[Q][Q],
                     double result[Q][Q]) {
  double temp[Q][Q];
  for (int i = 0; i < Q; i++)
    for (int j = 0; j < Q; j++)
      temp[i][j] = A[i][j] * B[j];
  for (int i = 0; i < Q; i++)
    for (int j = 0; j < Q; j++) {
      result[i][j] = 0;
      for (int k = 0; k < Q; k++)
        result[i][j] += temp[i][k] * C[k][j];
    }
}

void MRT_matrix_multiply(double A[Q][Q], double B[Q], double C[Q]) {
  for (int i = 0; i < Q; i++) {
    double sum = 0.0;
    for (int j = 0; j < Q; j++)
      sum += A[i][j] * B[j];
    C[i] = sum;
  }
}

inline int idx_x(int i) { return (i % nX + nX) % nX; }   // x is periodic

inline int idx_y(int j) {
  if (j < 0)
    return -j;
  if (j >= nY)
    return 2 * nY - 2 - j;
  return j;
}

inline double phi_safe(int i, int j) { return phi[idx_x(i)][idx_y(j)]; }

inline double psi_safe(int i, int j, int iter) {
  int ii = idx_x(i);
  if (j < 0)
    return psi[ii][-j] + 2.0 * H0y(iter) * dx;      // ∂ψ/∂y = -H0y at bottom
  if (j >= nY)
    return psi[ii][2 * (nY - 1) - j] - 2.0 * H0y(iter) * dx;  // top
  return psi[ii][j];
}

inline double H2_safe(int i, int j) {
  int ii = idx_x(i);
  int jj = idx_y(j);
  return Hx[ii][jj] * Hx[ii][jj] + Hy[ii][jj] * Hy[ii][jj];
}

double delphix(int i, int j) {
  double term = 2.0 * (phi_safe(i + 1, j + 1) - phi_safe(i - 1, j + 1)) +
                8.0 * (phi_safe(i + 1, j) - phi_safe(i - 1, j)) +
                2.0 * (phi_safe(i + 1, j - 1) - phi_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double delphiy(int i, int j) {
  double term = 2.0 * (phi_safe(i + 1, j + 1) - phi_safe(i + 1, j - 1)) +
                8.0 * (phi_safe(i, j + 1) - phi_safe(i, j - 1)) +
                2.0 * (phi_safe(i - 1, j + 1) - phi_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double del2phi(int i, int j) {
  double term = 4.0 * (phi_safe(i + 1, j) + phi_safe(i - 1, j) +
                       phi_safe(i, j + 1) + phi_safe(i, j - 1)) +
                (phi_safe(i + 1, j + 1) + phi_safe(i + 1, j - 1) +
                 phi_safe(i - 1, j + 1) + phi_safe(i - 1, j - 1)) -
                20.0 * phi_safe(i, j);
  return term / (6.0 * dx * dx);
}

double compute_delx_H2(int i, int j) {
  double term = 2.0 * (H2_safe(i + 1, j + 1) - H2_safe(i - 1, j + 1)) +
                8.0 * (H2_safe(i + 1, j) - H2_safe(i - 1, j)) +
                2.0 * (H2_safe(i + 1, j - 1) - H2_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double compute_dely_H2(int i, int j) {
  double term = 2.0 * (H2_safe(i + 1, j + 1) - H2_safe(i + 1, j - 1)) +
                8.0 * (H2_safe(i, j + 1) - H2_safe(i, j - 1)) +
                2.0 * (H2_safe(i - 1, j + 1) - H2_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double dot_product(double ax, double ay, double bx, double by) {
  return ax * bx + ay * by;
}

void compute_ncap(int i, int j) {
  double dphix = delphix(i, j);
  double dphiy = delphiy(i, j);
  double mag = sqrt(dphix * dphix + dphiy * dphiy);
  if (mag > 1e-12) {
    ncap_x[i][j] = dphix / mag;
    ncap_y[i][j] = dphiy / mag;
  } else {
    ncap_x[i][j] = 0.0;
    ncap_y[i][j] = 0.0;
  }
}

void compute_mu_phi(int i, int j) {
  double phi_loc = phi[i][j];
  double laplacian = del2phi(i, j);
  mu_phi[i][j] = 4.0 * Beta * phi_loc * (phi_loc - 1.0) * (phi_loc - 0.5) -
                 kappa * laplacian;
}

void compute_rho(int i, int j) {
  rho[i][j] = rho_l + phi[i][j] * (rho_h - rho_l);
}

double del_rho_x(int i, int j) { return (rho_h - rho_l) * delphix(i, j); }

double del_rho_y(int i, int j) { return (rho_h - rho_l) * delphiy(i, j); }

void compute_psi(int i, int j) {
  psi[i][j] = 0.0;
  for (int k = 0; k < Q; k++)
    psi[i][j] += h[i][j][k];
}

void compute_geq(int i, int j) {
  double phi_loc = phi[i][j];
  double ux_loc = ux[i][j];
  double uy_loc = uy[i][j];
  for (int k = 0; k < Q; k++) {
    double cdotu = dot_product(cX[k], cY[k], ux_loc, uy_loc);
    double cdotn = dot_product(cX[k], cY[k], ncap_x[i][j], ncap_y[i][j]);
    geq[i][j][k] =
        w[k] * Lambda * phi_loc + w[k] * phi_loc * c_s_inv2 * cdotu +
        w[k] * 4.0 * M_phi / D * phi_loc * (1 - phi_loc) * c_s_inv2 * cdotn;
  }
}

void compute_feq(int i, int j) {
  double rho_loc = rho[i][j];
  double p_loc = p[i][j];
  double ux_loc = ux[i][j];
  double uy_loc = uy[i][j];
  double drho_dx = del_rho_x(i, j);
  double drho_dy = del_rho_y(i, j);
  double F_tot_x = (Fm_x[i][j] + Fs_x[i][j] + Fb_x[i][j]);
  double F_tot_y = (Fm_y[i][j] + Fs_y[i][j] + Fb_y[i][j]);
  for (int k = 0; k < Q; k++) {
    double cdotu = dot_product(cX[k], cY[k], ux_loc, uy_loc);
    double udotu = dot_product(ux_loc, uy_loc, ux_loc, uy_loc);
    double gamma =
        w[k] * (1 + c_s_inv2 * cdotu + 0.5 * c_s_inv4 * cdotu * cdotu -
                0.5 * udotu * c_s_inv2);
    feq[i][j][k] = w[k] * p_loc + rho_loc * c_s2 * (gamma - w[k]);
    double pre_x = (cX[k] - ux_loc);
    double pre_y = (cY[k] - uy_loc);
    // double pre_x = (cX[k] - ux_loc)*c_s_inv2;
    // double pre_y = (cY[k] - uy_loc)*c_s_inv2;
    double post_x = c_s2 * drho_dx * (gamma - w[k]) + gamma * F_tot_x;
    double post_y = c_s2 * drho_dy * (gamma - w[k]) + gamma * F_tot_y;
    F_alpha[i][j][k] = pre_x * post_x + pre_y * post_y;
  }
}

void compute_heq(int i, int j) {
  for (int k = 0; k < Q; k++)
    heq[i][j][k] = w[k] * psi[i][j];
}

// compute_H: keep the uniform-drive convention consistent with #6 below
void compute_H(int i, int j, int iter) {
  double chsumx = 0.0, chsumy = 0.0;
  for (int k = 0; k < Q; k++) { chsumx += cX[k]*h[i][j][k]; chsumy += cY[k]*h[i][j][k]; }
  Hx[i][j] = H0x(iter) + chsumx / (tau_h * c_s2 * dt);   // tau_h now a constant, not tau_h[i][j]
  Hy[i][j] = H0y(iter) + chsumy / (tau_h * c_s2 * dt);
}

inline double extrap_y(const double A[nX][nY], int i, int j) {
  int ii = idx_x(i);
  if (j < 0)   return 2.0 * A[ii][0]      - A[ii][1];
  if (j >= nY) return 2.0 * A[ii][nY - 1] - A[ii][nY - 2];
  return A[ii][j];
}

inline double Hx_safe(int i, int j) { return extrap_y(Hx, i, j); }
inline double Hy_safe(int i, int j) { return extrap_y(Hy, i, j); }
inline double Mx_safe(int i, int j) { return extrap_y(Mx, i, j); }
inline double My_safe(int i, int j) { return extrap_y(My, i, j); }

double compute_delx_Hx(int i, int j) {
  double term = 2.0 * (Hx_safe(i + 1, j + 1) - Hx_safe(i - 1, j + 1)) +
                8.0 * (Hx_safe(i + 1, j) - Hx_safe(i - 1, j)) +
                2.0 * (Hx_safe(i + 1, j - 1) - Hx_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double compute_delx_Hy(int i, int j) {
  double term = 2.0 * (Hy_safe(i + 1, j + 1) - Hy_safe(i - 1, j + 1)) +
                8.0 * (Hy_safe(i + 1, j) - Hy_safe(i - 1, j)) +
                2.0 * (Hy_safe(i + 1, j - 1) - Hy_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double compute_delx_Mx(int i, int j) {
  double term = 2.0 * (Mx_safe(i + 1, j + 1) - Mx_safe(i - 1, j + 1)) +
                8.0 * (Mx_safe(i + 1, j) - Mx_safe(i - 1, j)) +
                2.0 * (Mx_safe(i + 1, j - 1) - Mx_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double compute_delx_My(int i, int j) {
  double term = 2.0 * (My_safe(i + 1, j + 1) - My_safe(i - 1, j + 1)) +
                8.0 * (My_safe(i + 1, j) - My_safe(i - 1, j)) +
                2.0 * (My_safe(i + 1, j - 1) - My_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double compute_dely_Hx(int i, int j) {
  double term = 2.0 * (Hx_safe(i + 1, j + 1) - Hx_safe(i + 1, j - 1)) +
                8.0 * (Hx_safe(i, j + 1) - Hx_safe(i, j - 1)) +
                2.0 * (Hx_safe(i - 1, j + 1) - Hx_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double compute_dely_Hy(int i, int j) {
  double term = 2.0 * (Hy_safe(i + 1, j + 1) - Hy_safe(i + 1, j - 1)) +
                8.0 * (Hy_safe(i, j + 1) - Hy_safe(i, j - 1)) +
                2.0 * (Hy_safe(i - 1, j + 1) - Hy_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double compute_dely_Mx(int i, int j) {
  double term = 2.0 * (Mx_safe(i + 1, j + 1) - Mx_safe(i + 1, j - 1)) +
                8.0 * (Mx_safe(i, j + 1) - Mx_safe(i, j - 1)) +
                2.0 * (Mx_safe(i - 1, j + 1) - Mx_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double compute_dely_My(int i, int j) {
  double term = 2.0 * (My_safe(i + 1, j + 1) - My_safe(i + 1, j - 1)) +
                8.0 * (My_safe(i, j + 1) - My_safe(i, j - 1)) +
                2.0 * (My_safe(i - 1, j + 1) - My_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double compute_divM(int i, int j) {
  return compute_delx_Mx(i, j) + compute_dely_My(i, j);
}

void compute_magnetic_force(int i, int j) {
  double delx_Hx = compute_delx_Hx(i, j);
  double dely_Hx = compute_dely_Hx(i, j);
  double delx_Hy = compute_delx_Hy(i, j);
  double dely_Hy = compute_dely_Hy(i, j);
  double delx_Mx = compute_delx_Mx(i, j);
  double dely_Mx = compute_dely_Mx(i, j);
  double delx_My = compute_delx_My(i, j);
  double dely_My = compute_dely_My(i, j);
  double F_kelvin_x = mu0 * (Mx[i][j] * delx_Hx + My[i][j] * dely_Hx);
  double F_kelvin_y = mu0 * (Mx[i][j] * delx_Hy + My[i][j] * dely_Hy);
  // Corrected Rotational Magnetic Force
  double F_rot_x = 0.5 * mu0 *
                   ((Hy[i][j] * dely_Mx + Mx[i][j] * dely_Hy) -
                    (Hx[i][j] * dely_My + My[i][j] * dely_Hx));
  double F_rot_y = 0.5 * mu0 *
                   ((Hx[i][j] * delx_My + My[i][j] * delx_Hx) -
                    (Hy[i][j] * delx_Mx + Mx[i][j] * delx_Hy));
  Fm_x[i][j] = F_kelvin_x + F_rot_x;
  Fm_y[i][j] = F_kelvin_y + F_rot_y;
}

void compute_surface_tension_force(int i, int j) {
  Fs_x[i][j] = mu_phi[i][j] * delphix(i, j);
  Fs_y[i][j] = mu_phi[i][j] * delphiy(i, j);
}

void compute_pressure(int i, int j) {
  double fsum = 0.0;
  for (int k = 0; k < Q; k++)
    fsum += f[i][j][k];
  double udotdelrho =
      dot_product(ux[i][j], uy[i][j], del_rho_x(i, j), del_rho_y(i, j));
  p[i][j] = fsum + 0.5 * dt * c_s2 * udotdelrho;
}

void compute_velocity(int i, int j) {
  double sum_fx = 0.0, sum_fy = 0.0;
  for (int k = 0; k < Q; k++) {
    sum_fx += f[i][j][k] * cX[k];
    sum_fy += f[i][j][k] * cY[k];
  }
  ux[i][j] = sum_fx * c_s_inv2 / rho[i][j] +
             0.5 * dt * (Fm_x[i][j] + Fs_x[i][j] + Fb_x[i][j]) / rho[i][j];
  uy[i][j] = sum_fy * c_s_inv2 / rho[i][j] +
             0.5 * dt * (Fm_y[i][j] + Fs_y[i][j] + Fb_y[i][j]) / rho[i][j];
}

void update_Sf(int i, int j) {
  double term1 = rho_l * c_s2 * dt / eta_l;
  double term2 = rho_h * c_s2 * dt / eta_h;
  double interpolation_value = term1 + phi[i][j] * (term2 - term1);
  Sf[7] = interpolation_value / (1.0 + 0.5 * interpolation_value);
  Sf[8] = Sf[7];
  Sf[1] = Sf[7];
  Sf[2] = Sf[7];
  Sf[4] = 8.0 * (2.0 - Sf[7]) / (8.0 - Sf[7]);
  Sf[6] = Sf[4];
  for (int k = 0; k < Q; k++)
    ISf[k] = 1.0 - 0.5 * Sf[k];
  matrix_multiply(Minv, Sf, M, MinvSfM);
  matrix_multiply(Minv, ISf, M, MinvISfM);
}

//CHANGE Compute M
inline double langevin2D(double xi) {
  if (xi < 1e-8)
    return 0.5 * xi;
  if (xi > 100.0)
    return 1.0 - 0.5 / xi; // asymptotic, no exp() overflow
  return cyl_bessel_i(1, xi) / cyl_bessel_i(0, xi);
}

inline void compute_M(int i, int j, int iter) {
  const double XMIN = 1e-6;              // switch to the regular linear form

  double phi_loc = phi[i][j];
  if (!std::isfinite(phi_loc) || phi_loc < 1e-6) phi_loc = 1e-6;

  double Omega = vorticity_z[i][j];
  if (!std::isfinite(Omega)) Omega = 0.0;

  double ex = xie_x[i][j], ey = xie_y[i][j];
  if (!std::isfinite(ex)) ex = 0.0;
  if (!std::isfinite(ey)) ey = 0.0;

  // d(xi_e)/dt for the effective-field (MRS) closure, 2D
  auto rhs = [&](double s, double ax, double ay, double &dax, double &day) {
    // double hx  = H0 * cos(omega * (iter + s));       // uniform, x-polarised
    // double xtx = xi_per_H * hx, xty = 0.0;           // true field xi
    double hx = Hx[i][j], hy = Hy[i][j];
    double xtx = xi_per_H * hx, xty = xi_per_H * hy;
    double xe  = sqrt(ax*ax + ay*ay);

    if (xe < XMIN) {                                 // regular small-field limit
      dax = -Omega*ay - (ax - xtx)/tau_B;
      day =  Omega*ax - (ay - xty)/tau_B;
      return;
    }
    double nx = ax/xe, ny = ay/xe;
    double Lv  = langevin2D(xe);
    double dLv = 1.0 - Lv*Lv - Lv/xe;                // dL2/dxi, from I0'=I1
    double LdL = Lv/dLv;                             // (dlnL2/dxi)^-1

    double ndotx = nx*xtx + ny*xty;
    double px = xtx - nx*ndotx;                      // transverse true field
    double py = xty - ny*ndotx;

    double par = -(LdL/tau_B)*(1.0 - ndotx/xe);      // along n
    double cperp = (xe/Lv - 1.0)/tau_B;              // perpendicular

    dax = -Omega*ay + par*nx + cperp*px;
    day =  Omega*ax + par*ny + cperp*py;
  };

  const int    nsub  = 16;
  const double hstep = 1.0/nsub;
  double s = 0.0;
  for (int n = 0; n < nsub; n++) {
    double k1x,k1y,k2x,k2y,k3x,k3y,k4x,k4y;
    rhs(s,               ex,                     ey,                     k1x,k1y);
    rhs(s + 0.5*hstep,   ex + 0.5*hstep*k1x,     ey + 0.5*hstep*k1y,     k2x,k2y);
    rhs(s + 0.5*hstep,   ex + 0.5*hstep*k2x,     ey + 0.5*hstep*k2y,     k3x,k3y);
    rhs(s + hstep,       ex + hstep*k3x,         ey + hstep*k3y,         k4x,k4y);
    ex += (hstep/6.0)*(k1x + 2.0*k2x + 2.0*k3x + k4x);
    ey += (hstep/6.0)*(k1y + 2.0*k2y + 2.0*k3y + k4y);
    s  += hstep;
  }

  if (!std::isfinite(ex)) ex = 0.0;
  if (!std::isfinite(ey)) ey = 0.0;
  xie_x[i][j] = ex;  xie_y[i][j] = ey;

  // M = phi * Ms * L2(xi_e) * xi_e/|xi_e|      [(B-5) in 2D]
  double xe = sqrt(ex*ex + ey*ey);
  if (xe < XMIN) {
    Mx[i][j] = phi_loc*Ms*0.5*ex;                    // L2 -> xi/2
    My[i][j] = phi_loc*Ms*0.5*ey;
  } else {
    double Lv = langevin2D(xe);
    Mx[i][j] = phi_loc*Ms*Lv*(ex/xe);
    My[i][j] = phi_loc*Ms*Lv*(ey/xe);
  }
}

void compute_phi_all_domain() {
  for (int i = 0; i < nX; i++)
    for (int j = 0; j < nY; j++) {
      double gsum = 0.0;
      for (int k = 0; k < Q; k++)
        gsum += g[i][j][k];
      phi[i][j] = gsum;
    }
}

inline double ux_safe(int i, int j) {
  int ii = idx_x(i);
  if (j < 0)   return -ux[ii][0];        // half-way no-slip wall at y = -1/2
  if (j >= nY) return -ux[ii][nY - 1];
  return ux[ii][j];
}

inline double uy_safe(int i, int j) {
  int ii = idx_x(i);
  if (j < 0)   return -uy[ii][0];
  if (j >= nY) return -uy[ii][nY - 1];
  return uy[ii][j];
}

double compute_delx_uy(int i, int j) {
  double term = 2.0 * (uy_safe(i + 1, j + 1) - uy_safe(i - 1, j + 1)) +
                8.0 * (uy_safe(i + 1, j) - uy_safe(i - 1, j)) +
                2.0 * (uy_safe(i + 1, j - 1) - uy_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double compute_dely_ux(int i, int j) {
  double term = 2.0 * (ux_safe(i + 1, j + 1) - ux_safe(i + 1, j - 1)) +
                8.0 * (ux_safe(i, j + 1) - ux_safe(i, j - 1)) +
                2.0 * (ux_safe(i - 1, j + 1) - ux_safe(i - 1, j - 1));
  return term / (24.0 * dx);
}

double curl_U(int i, int j) {
  double delx_uy = compute_delx_uy(i, j);
  double dely_ux = compute_dely_ux(i, j);
  return delx_uy - dely_ux;
}

int main() {
  cout << "Starting simulation..." << endl;

  const fs::path baseDir = "output_poiseuille_latest";
  // const fs::path subDir = baseDir / ("output_" + std::to_string(H0 * Href) + "_" + std::to_string(omega * tau_B));
  const fs::path subDir = baseDir / ("output_xi0_" + std::to_string(xi0) + "_wtB_" + std::to_string(omega_tauB));
  if (!fs::exists(baseDir))
    fs::create_directory(baseDir);
  if (!fs::exists(subDir))
    fs::create_directory(subDir);

  // Precompute g and h MRT matrix (time-invariant)
  matrix_multiply(Minv, Sg, M, MinvSgM);
  matrix_multiply(Minv, Sh, M, MinvShM);

  // double centerX = (nX - 1) / 2.0;
  // double centerY = (nY - 1) / 2.0;
  // double radius = 12.5;
  for (int i = 0; i < nX; i++)
    for (int j = 0; j < nY; j++) {
      // double dist = sqrt((i-centerX)*(i-centerX) + (j-centerY)*(j-centerY));
      // double W = D / 2.0;
      // phi[i][j] = 0.5 * (1.0 + tanh((radius - dist)/W));
      phi[i][j] = 1.0;
    }

  // Build specular-reflection index table for h y-wall BC
  int specular_y[Q];
  for (int k = 0; k < Q; k++)
    for (int m = 0; m < Q; m++)
      if (cY[m] == -cY[k] && cX[m] == cX[k])
        specular_y[k] = m;

  // Phase-field initialisation
  for (int i = 0; i < nX; i++)
    for (int j = 0; j < nY; j++) {
      compute_ncap(i, j);
      compute_mu_phi(i, j);
      compute_geq(i, j);
      for (int k = 0; k < Q; k++)
        g[i][j][k] = geq[i][j][k];
    }

  // Magnetic field initialisation
  for (int i = 0; i < nX; i++)
    for (int j = 0; j < nY; j++) {
      // double H0_basis = H0 * cos(omega * 0.0);
      psi[i][j] = 0.0;
      Hx[i][j] = H0x(0);
      Hy[i][j] = H0y(0);
      Mx[i][j] = 0.0;
      My[i][j] = 0.0;
      xie_x[i][j] = xi0;     // Bacri's initial condition:  xi_e = xi_0
      xie_y[i][j] = 0.0;
      compute_heq(i, j);
      for (int k = 0; k < Q; k++)
        h[i][j][k] = heq[i][j][k];
    }

  // Force and flow initialisation
  for (int i = 0; i < nX; i++)
    for (int j = 0; j < nY; j++) {
      compute_magnetic_force(i, j);
      compute_surface_tension_force(i, j);
      Fb_x[i][j] = 4e-6;
      Fb_y[i][j] = 0.0;

      // Compute Density
      compute_rho(i, j);
    }

  for (int i = 0; i < nX; i++)
    for (int j = 0; j < nY; j++) {
      ux[i][j] = uy[i][j] = 0.0;
      p[i][j] = 0.0;
      vorticity_z[i][j] = 0.0;
      compute_feq(i, j);
      for (int k = 0; k < Q; k++)
        f[i][j][k] = feq[i][j][k];
    }

  cout << "Initialisation complete. Starting main loop..." << endl;

  for (int iter = 0; iter <= nSteps; iter++) {

    // Macroscopic phi (from g)
    compute_phi_all_domain();

    // Phase-field quantities
    for (int i = 0; i < nX; i++)
      for (int j = 0; j < nY; j++) {
        compute_ncap(i, j);
        compute_mu_phi(i, j);
      }

    // (1) provisional velocity from the previous step's forces
    for (int i = 0; i < nX; i++)
      for (int j = 0; j < nY; j++)
        compute_velocity(i, j);

    // (2) vorticity at the CURRENT step
    for (int i = 0; i < nX; i++)
      for (int j = 0; j < nY; j++)
        vorticity_z[i][j] = 0.5 * curl_U(i, j);

    // (3) field + magnetisation, using the current vorticity
    for (int i = 0; i < nX; i++)
      for (int j = 0; j < nY; j++) {
        compute_psi(i, j);
        compute_H(i, j, iter);
        compute_M(i, j, iter);
      }

    // (4) forces from the updated magnetisation
    for (int i = 0; i < nX; i++)
      for (int j = 0; j < nY; j++) {
        compute_magnetic_force(i, j);
        compute_surface_tension_force(i, j);
        compute_rho(i, j);
      }

    // (5) corrected velocity and pressure
    for (int i = 0; i < nX; i++)
      for (int j = 0; j < nY; j++) {
        compute_velocity(i, j);
        compute_pressure(i, j);
      }

    // Equilibrium distributions
    for (int i = 0; i < nX; i++)
      for (int j = 0; j < nY; j++) {
        compute_geq(i, j);
        compute_feq(i, j);
        compute_heq(i, j);
      }

    // Collision with MRT and forcing
    memset(gstar, 0, sizeof(gstar));
    memset(fstar, 0, sizeof(fstar));
    memset(hstar, 0, sizeof(hstar));

    for (int i = 0; i < nX; i++)
      for (int j = 0; j < nY; j++) {
        double diff_term[Q];

        // g
        double dg[Q];
        for (int k = 0; k < Q; k++)
          dg[k] = g[i][j][k] - geq[i][j][k];
        MRT_matrix_multiply(MinvSgM, dg, diff_term);
        for (int k = 0; k < Q; k++)
          gstar[i][j][k] = g[i][j][k] - diff_term[k];

        // f
        update_Sf(i, j);
        double df[Q];
        for (int k = 0; k < Q; k++)
          df[k] = f[i][j][k] - feq[i][j][k];
        MRT_matrix_multiply(MinvSfM, df, diff_term);
        double force_term[Q];
        MRT_matrix_multiply(MinvISfM, F_alpha[i][j], force_term);
        for (int k = 0; k < Q; k++)
          fstar[i][j][k] = f[i][j][k] - diff_term[k] + force_term[k];

        // h  (now solves ∇²ψ = ∇·M, MinvShM precomputed once, constant)
        double dh[Q];
        for (int k = 0; k < Q; k++)
          dh[k] = h[i][j][k] - heq[i][j][k];
        MRT_matrix_multiply(MinvShM, dh, diff_term);
        double S_div_M = compute_divM(i, j);
        for (int k = 0; k < Q; k++)
          hstar[i][j][k] = h[i][j][k] - diff_term[k] + w[k] * S_div_M * dt;
      }

    // Streaming + BCs
    memset(ftemp, 0, sizeof(ftemp));
    memset(gtemp, 0, sizeof(gtemp));
    memset(htemp, 0, sizeof(htemp));

    // f and g: full bounce-back at all four walls (no-slip, no-penetration)
    for (int i = 0; i < nX; i++)
      for (int j = 0; j < nY; j++)
        for (int k = 0; k < Q; k++) {
          int ip = i - cX[k];
          int jp = j - cY[k];
          if (jp < 0 || jp >= nY) {
            ftemp[i][j][k] = fstar[i][j][opp[k]];
            gtemp[i][j][k] = gstar[i][j][opp[k]];
          } else {
            ip = (ip + nX) % nX; // Periodic in x
            ftemp[i][j][k] = fstar[ip][jp][k];
            gtemp[i][j][k] = gstar[ip][jp][k];
          }
        }

    // h streaming BCs:
    // x: periodic (psi' is periodic)
    // y: specular reflection (∂ψ'/∂y = 0)

    for (int i = 0; i < nX; i++)
      for (int j = 0; j < nY; j++)
        for (int k = 0; k < Q; k++) {
          int ip = i - cX[k], jp = j - cY[k];

          if (jp < 0 || jp >= nY) {
            int jr = (jp < 0) ? (-jp - 1) : (2 * nY - 1 - jp);
            ip = (ip + nX) % nX;   // must wrap BEFORE indexing
            htemp[i][j][k] = hstar[ip][jr][specular_y[k]];
          } else {
            ip = (ip + nX) % nX; // periodic in x
            htemp[i][j][k] = hstar[ip][jp][k];
          }
        }

    memcpy(f, ftemp, sizeof(f));
    memcpy(g, gtemp, sizeof(g));
    memcpy(h, htemp, sizeof(h));

    // Output
    if (iter % outputInterval == 0) {
      cout << "Writing iter " << iter << " ... ";
      std::string filename =
          (subDir / ("output_t_" + std::to_string(iter) + ".csv")).string();
      FILE *fp = fopen(filename.c_str(), "w");
      fprintf(fp, "i,j,phi,rho,ux,uy,Hx,Hy,Fm_x,Fm_y,Fs_x,Fs_y,Fb_x,Fb_y\n");
      double phi_sum = 0.0;
      for (int i = 0; i < nX; i++)
        for (int j = 0; j < nY; j++) {
          phi_sum += phi[i][j];
          fprintf(fp,
                  "%d,%d,%10e,%10e,%10e,%10e,%10e,%10e,%10e,%10e,%10e,%10e,%"
                  "10e,%10e\n",
                  i, j, phi[i][j], rho[i][j], ux[i][j], uy[i][j], Hx[i][j],
                  Hy[i][j], Fm_x[i][j], Fm_y[i][j], Fs_x[i][j], Fs_y[i][j],
                  Fb_x[i][j], Fb_y[i][j]);
        }
      fclose(fp);
      cout << "phi_sum=" << phi_sum << endl;
    }
  }
  return 0;
}