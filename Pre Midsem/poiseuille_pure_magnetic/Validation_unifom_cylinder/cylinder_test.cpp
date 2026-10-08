#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

namespace fs = std::filesystem;

using namespace std;

// Simulation setup parameters
int nSteps = 300;
int outputInterval = 300;
const int nX = 200;
const int nY = 200;

const int NSUB_M = 8;
const int NSUB_H = 64;
// (For the STATIC cylinder test with FREEZE_FLOW, NSUB_M=2 / NSUB_H=20 give an
//  identical converged field and run much faster; the values above are the
//  originals kept for the dynamic runs.)

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
double xi0        = 20.0;   // Langevin parameter at the field peak (unused as driver here)
double omega_tauB = 0.0;    // dimensionless frequency  omega*tau_B  (STATIC test: 0)
double phi_p      = 0.2;    // PARTICLE volume fraction (the theory's phi)


// ---- CYLINDER TEST controls (linear magnetostatic reduction) ---------
// Susceptibility map:  chi = (1/3) phi Ms xi_per_H.  We set chi1 (inside) and
// chi2 (outside) directly and back out a per-node Ms; mu_i = mu0 (1+chi_i).
// Analytical target (Eq. 53):  H_in / H0 = 2 mu2/(mu1+mu2) = 2/((mu1/mu2)+1).
double chi1 = 1.0;          // susceptibility INSIDE the cylinder  (region 1)
double chi2 = 0.0;          // susceptibility OUTSIDE the cylinder (region 2)
double R_cyl = 20.0;        // cylinder radius (paper: R = 20)
const bool FREEZE_FLOW = true;  // pure magnetostatic test: no flow, no rotational coupling


// ---- numerical scales ------------------------------------------------
double tau_B = 20.0;                // Brownian time, in lattice steps
double omega = omega_tauB / tau_B;  // field angular frequency
// double H0 = 1.73;
// double theta = 0.0; // Angle of inclination of magnetic field with respect to x-axis
// double H0x_amp = H0 * cos(theta);
// double H0y_amp = H0 * sin(theta);
// inline double H0x(int iter) { return H0x_amp * cos(omega * iter); }
// inline double H0y(int iter) { return H0y_amp * cos(omega * iter); }

// ---- applied field: H_app = ( H_d + H0 cos(wt) + k_grad (x - x_c),  -k_grad (y - y_c) )
double H_ref    = 1.73;   // reference field: xi = xi0 at |H| = H_ref (fixes xi_per_H)
double H0       = 0.0;    // amplitude of the uniform oscillating field (STATIC: 0)
double H_d      = 0.005;  // uniform (DC) part of the driving field: SMALL -> linear limit
double G_target = -1.0;   // <=0 disables Kelvin-gradient drive; use k_grad_input
double k_grad_input = 0.0; // field gradient used when G_target <= 0 (ZERO: uniform field)
double k_grad   = 0.0;    // field gradient dHx/dx; set in main()
const double x_c = 0.5 * (nX - 1);  // field centre in x (nodes at x = i)
const double y_c = 0.5 * nY;        // field centre in y (nodes at y = j + 1/2, walls at 0, nY)

double M_phi   = 0.1;
double epsilon = 9.2; //Used only in the mag field part, can be changed
double mu0     = 1.0;
double D       = 4.0;
double sigma   = 0.00191875;
double Beta    = 12.0 * sigma / D;
double kappa   = 1.5 * D * sigma;
double rho_l   = 1.0;
double rho_h   = 1.975;
double eta_l   = 0.0025;
double eta_h   = 0.05;

// ---- DERIVED: fixed by  xi = mu0 tau_B Ms H / (3 eta phi_p) ----------
// double Ms       = 3.0 * eta_h * phi_p * xi0 / (mu0 * tau_B * H0);
double Ms       = 3.0 * eta_h * phi_p * xi0 / (mu0 * tau_B * H_ref);
// double Ms = 0.04 * H0; // Scaled Ms to match paper, inbuilt physics is not yet resolved
// double xi_per_H = xi0 / H0;   // Langevin argument per unit field
double xi_per_H = xi0 / H_ref;   // Langevin argument per unit field

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
double Sh3 = 1 / (epsilon * c_s_inv2 / dt + 0.5);
double tau_h = 1/Sh3;
double Sh[Q] = {1.0, 1.1, 1.1, Sh3, Sh3, Sh3, Sh3, 1.2, 1.2};
double MinvShM[Q][Q];


// ---- applied-field helpers (unrolled coordinates: x = i, y = j + 1/2) ----
inline double Hx_app(double x, int iter) {
  return H_d + H0 * cos(omega * iter) + k_grad * (x - x_c);
}
inline double Hy_app(double y) { return -k_grad * (y - y_c); }
inline double A_app(double x, double y, int iter) {   // B_app = curl(A_app z)
  return mu0 * Hx_app(x, iter) * (y - y_c);
}

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

static double Az[nX][nY];
static double Hx[nX][nY];
static double Hy[nX][nY];
static double Hx_prev[nX][nY];
static double Hy_prev[nX][nY];
static double Mx[nX][nY];
static double My[nX][nY];
static double xie_x[nX][nY];        // effective field, x component
static double xie_y[nX][nY];        // effective field, y component
static double Ms_map[nX][nY];       // per-node saturation magnetisation (chi map)
static double chi_map[nX][nY];      // per-node linear susceptibility (diagnostic)
static double MsG[2][nY];           // Ms in the x-ghost columns

// ---- x-ghost columns (unrolled): s = 0 -> x = -1 (image of column nX-1),
//                                  s = 1 -> x = nX (image of column 0)
static double HGx[2][nY], HGy[2][nY], HGx_prev[2][nY], HGy_prev[2][nY];
static double MGx[2][nY], MGy[2][nY], xieG_x[2][nY], xieG_y[2][nY];
// ---- seam jump of A (applied ramp + magnetisation ramp), per row j
static double dA_seam[nY], dAy_seam[nY];
double dMxw_bot = 0.0, dMxw_top = 0.0;   // seam jump of Mx extrapolated to the walls

// Jump of a post-collision h population across the x-seam, at row j:
//   h*_k(x + nX, j) = h*_k(x, j) + h_jump(k, j)
inline double h_jump(int k, int j) {
  return w[k] * (dA_seam[j] - (tau_h - 1.0) * dt * cY[k] * dAy_seam[j]);
}

static double SA[nX][nY];
double A_resid = 0.0;                          // diagnostic

static double Fm_x[nX][nY];
static double Fk_x[nX][nY], Fk_y[nX][nY];   // Kelvin part alone (for post-processing)
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

void compute_Az(int i, int j) {
  Az[i][j] = 0.0;
  for (int k = 0; k < Q; k++)
    Az[i][j] += h[i][j][k];
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
    heq[i][j][k] = w[k] * Az[i][j];
}

// compute_H: keep the uniform-drive convention consistent with #6 below
void compute_H(int i, int j, int iter) {
  double chsumx = 0.0, chsumy = 0.0;
  for (int k = 0; k < Q; k++) { chsumx += cX[k]*h[i][j][k]; chsumy += cY[k]*h[i][j][k]; }
  double inv = 1.0 / (tau_h * c_s2 * dt);
  double Bx = -chsumy * inv;        //  B_x =  dA/dy
  double By =  chsumx * inv;        //  B_y = -dA/dx
  Hx[i][j] = Bx/mu0 - Mx[i][j];
  Hy[i][j] = By/mu0 - My[i][j];
}

inline double extrap_y(const double A[nX][nY], int i, int j) {
  int ii = idx_x(i);
  if (j < 0)   return 2.0 * A[ii][0]      - A[ii][1];
  if (j >= nY) return 2.0 * A[ii][nY - 1] - A[ii][nY - 2];
  return A[ii][j];
}

// inline double Hx_safe(int i, int j) { return extrap_y(Hx, i, j); }
inline double col_extrap(const double *col, int j) {
  if (j < 0)   return 2.0 * col[0]      - col[1];
  if (j >= nY) return 2.0 * col[nY - 1] - col[nY - 2];
  return col[j];
}
// Stencil value at (i, j): interior column, or unrolled x-ghost column (i = -1 or i = nX)
inline double field_safe(const double A[nX][nY], const double G[2][nY], int i, int j) {
  if (i < 0)   return col_extrap(G[0], j);
  if (i >= nX) return col_extrap(G[1], j);
  return col_extrap(A[i], j);
}
inline double Hx_safe(int i, int j) { return field_safe(Hx, HGx, i, j); }
inline double Hy_safe(int i, int j) { return field_safe(Hy, HGy, i, j); }
inline double Mx_safe(int i, int j) { return field_safe(Mx, MGx, i, j); }
inline double My_safe(int i, int j) { return field_safe(My, MGy, i, j); }

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

double compute_curlM(int i, int j) {          // (curl M)_z
  return compute_delx_My(i, j) - compute_dely_Mx(i, j);
}
inline double Mxw_bot(int i) { return 1.5*Mx[i][0]      - 0.5*Mx[i][1];      }
inline double Mxw_top(int i) { return 1.5*Mx[i][nY-1]   - 0.5*Mx[i][nY-2];   }

void compute_source_all(int iter) {
  double fsum = 0.0, gsum = 0.0;
  for (int i = 0; i < nX; i++) {
    for (int j = 0; j < nY; j++) {
      SA[i][j] = -mu0 * compute_curlM(i, j);   // = lap(A)
      fsum += SA[i][j];
    }
    // gsum += mu0*(H0x(iter) + Mxw_top(i)) - mu0*(H0x(iter) + Mxw_bot(i));
    gsum += mu0*(Mxw_top(i) - Mxw_bot(i));   // H_app,x is identical on both walls and cancels
  }
  A_resid = (fsum - gsum) / double(nX*nY);
  for (int i = 0; i < nX; i++)
    for (int j = 0; j < nY; j++)
      SA[i][j] = -epsilon * (SA[i][j] - A_resid);   // S = -eps*lap(A)
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
  Fk_x[i][j] = F_kelvin_x;
  Fk_y[i][j] = F_kelvin_y;
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

inline double langevin3D(double x) {
  double a = fabs(x), L;
  if (a < 0.2) {
    double a2 = a * a;
    L = a * (1.0/3.0 + a2 * (-1.0/45.0 + a2 * (2.0/945.0 - a2/4725.0)));
  } else if (a > 200.0) {
    L = 1.0 - 1.0/a;
  } else {
    L = 1.0/tanh(a) - 1.0/a;
  }
  return (x < 0.0) ? -L : L;
}

inline double dlangevin3D(double x) {          // even in x
  double a = fabs(x);
  if (a < 0.2) {
    double a2 = a * a;
    return 1.0/3.0 + a2 * (-1.0/15.0 + a2 * (2.0/189.0 - a2/675.0));
  }
  if (a > 200.0) return 1.0/(a*a);
  double s = sinh(a);
  return 1.0/(a*a) - 1.0/(s*s);
}

// One MRSh step [iter-1, iter] at a single point (RK4, NSUB_M sub-steps, H linear in time).
// (ex, ey) is advanced in place; (mx, my) = phi*Ms*L(xi_e)*xi_e/|xi_e|   [Bacri Eqs. (5)-(6)]
inline void mrsh_point(double &ex, double &ey, double hx0, double hy0, double hx1,
                       double hy1, double Omega, double phi_loc, double &mx, double &my,
                       double Ms_loc = Ms) {
  const double XMIN = 1e-10;              // only guards xi_e == 0 exactly

  if (!std::isfinite(phi_loc) || phi_loc < 1e-6) phi_loc = 1e-6;
  if (!std::isfinite(Omega)) Omega = 0.0;
  if (!std::isfinite(ex)) ex = 0.0;
  if (!std::isfinite(ey)) ey = 0.0;

  auto rhs = [&](double s, double ax, double ay, double &dax, double &day) {
    const double f = s / dt;                       // s runs 0 -> dt over the step
    double hx = (1.0 - f)*hx0 + f*hx1;
    double hy = (1.0 - f)*hy0 + f*hy1;
    double xtx = xi_per_H * hx, xty = xi_per_H * hy;

    double xe = sqrt(ax*ax + ay*ay);

    if (xe < XMIN) {                      // removable singularity; the limit of
      dax = -Omega*ay - (ax - xtx)/tau_B; // the full RHS is linear relaxation
      day =  Omega*ax - (ay - xty)/tau_B;
      return;
    }

    double nx = ax/xe, ny = ay/xe;
    double Le  = langevin3D(xe);
    double dLe = dlangevin3D(xe);

    double xpar = nx*xtx + ny*xty;        // n . xi
    double ppx  = xtx - nx*xpar;          // xi_perp
    double ppy  = xty - ny*xpar;

    // As xe -> 0: L/L' -> xe, so par -> -(xe - xi_par)/tau_B and the 1/xe in
    // (1 - xi_par/xe) is cancelled by the prefactor.  No cancellation loss.
    double par   = -(Le/(dLe*tau_B)) * (1.0 - xpar/xe);
    double cperp =  0.5*(xe/Le - 1.0)/tau_B;

    dax = -Omega*ay + par*nx + cperp*ppx;
    day =  Omega*ax + par*ny + cperp*ppy;
  };

  const double hstep = 1.0/NSUB_M;
  double s = 0.0;
  for (int n = 0; n < NSUB_M; n++) {
    double k1x,k1y,k2x,k2y,k3x,k3y,k4x,k4y;
    rhs(s,             ex,               ey,               k1x,k1y);
    rhs(s+0.5*hstep,   ex+0.5*hstep*k1x, ey+0.5*hstep*k1y, k2x,k2y);
    rhs(s+0.5*hstep,   ex+0.5*hstep*k2x, ey+0.5*hstep*k2y, k3x,k3y);
    rhs(s+hstep,       ex+hstep*k3x,     ey+hstep*k3y,     k4x,k4y);
    ex += (hstep/6.0)*(k1x + 2.0*k2x + 2.0*k3x + k4x);
    ey += (hstep/6.0)*(k1y + 2.0*k2y + 2.0*k3y + k4y);
    s  += hstep;
  }

  if (!std::isfinite(ex)) ex = 0.0;
  if (!std::isfinite(ey)) ey = 0.0;

  // M = phi * Ms * L(xi_e) * xi_e/|xi_e|          [Bacri Eq. (5)]
  double xe = sqrt(ex*ex + ey*ey);
  if (xe < XMIN) {
    mx = phi_loc*Ms_loc*ex/3.0;                   // L -> xi/3
    my = phi_loc*Ms_loc*ey/3.0;
  } else {
    double Le = langevin3D(xe);
    mx = phi_loc*Ms_loc*Le*(ex/xe);
    my = phi_loc*Ms_loc*Le*(ey/xe);
  }
}

inline void compute_M(int i, int j, int iter) {
  (void)iter;
  mrsh_point(xie_x[i][j], xie_y[i][j], Hx_prev[i][j], Hy_prev[i][j], Hx[i][j], Hy[i][j],
             vorticity_z[i][j], phi[i][j], Mx[i][j], My[i][j], Ms_map[i][j]);
}

// Field in ghost column s from its image column, called after compute_H and BEFORE
// compute_M of the image column (so both M's are at the previous level):
//   Hx(x +/- L) = Hx(x) +/- k_grad*L ;  B_y periodic  =>  Hy_g = Hy + My(image) - My_g
inline void set_H_ghost(int s, int j) {
  int src = (s == 0) ? nX - 1 : 0;
  double sgn = (s == 0) ? -1.0 : 1.0;
  HGx[s][j] = Hx[src][j] + sgn * k_grad * nX;
  HGy[s][j] = Hy[src][j] + My[src][j] - MGy[s][j];
}

// Magnetisation in the ghost columns: same local ODE, driven by the ghost field and the
// (periodic) vorticity and phase field of the image column.
void compute_M_ghosts() {
  for (int s = 0; s < 2; s++) {
    int src = (s == 0) ? nX - 1 : 0;
    for (int j = 0; j < nY; j++)
      mrsh_point(xieG_x[s][j], xieG_y[s][j], HGx_prev[s][j], HGy_prev[s][j], HGx[s][j],
                 HGy[s][j], vorticity_z[src][j], phi[src][j], MGx[s][j], MGy[s][j], MsG[s][j]);
  }
}

// Seam jump of A including the magnetisation ramp:
//   A(x+L, y) - A(x, y) = mu0 k L (y - y_c) + mu0 * integral of dMx dy
//   dMx(y) = Mx(x+L, y) - Mx(x, y), averaged over both seam crossings (keeps A conserved)
void compute_seam_jump() {
  double dMx[nY];
  for (int j = 0; j < nY; j++)
    dMx[j] = 0.5 * ((Mx[nX - 1][j] - MGx[0][j]) + (MGx[1][j] - Mx[0][j]));
  double IM = 0.0;
  for (int j = 0; j < nY; j++) {
    if (j > 0) IM += 0.5 * (dMx[j - 1] + dMx[j]) * dx;
    dA_seam[j]  = mu0 * (k_grad * nX * ((j + 0.5) - y_c) + IM);
    dAy_seam[j] = mu0 * (k_grad * nX + dMx[j]);
  }
  dMxw_bot = 1.5 * dMx[0]      - 0.5 * dMx[1];
  dMxw_top = 1.5 * dMx[nY - 1] - 0.5 * dMx[nY - 2];
}

// Reference for an x-uniform channel at the channel-centre field:
//   Hx(t) = H_d + H0 cos(wt),  Hy = -My (normal B continuous),  uniform vorticity Om.
// Returns time-averaged Mx and g from  <mu0 (M x H)_z> = -6 eta_h phi_p g Om.
void mrsh_reference(double Om, double &Mx_mean, double &g_ref, bool demag = true) {
  const double period = 2.0 * acos(-1.0) / omega;
  int n_trans = int(40.0 * tau_B + 10.0 * period) + 1;
  int n_avg   = (H0 != 0.0) ? int(2000.0 * period + 0.5) : int(400.0 * tau_B) + 1;
  double ex = xi_per_H * (H_d + H0), ey = 0.0;
  double hx_prev = H_d + H0, hy_prev = 0.0, mx = 0.0, my = 0.0;
  double sum_mx = 0.0, sum_T = 0.0;
  for (int n = 1; n <= n_trans + n_avg; n++) {
    double hx = H_d + H0 * cos(omega * n);
    double hy = demag ? -my : 0.0;                // demagnetising field, one step lagged
    mrsh_point(ex, ey, hx_prev, hy_prev, hx, hy, Om, 1.0, mx, my);
    if (n > n_trans) { sum_mx += mx; sum_T += mu0 * (mx * hy - my * hx); }
    hx_prev = hx; hy_prev = hy;
  }
  Mx_mean = sum_mx / n_avg;
  g_ref   = -(sum_T / n_avg) / (6.0 * eta_h * phi_p * Om);
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

// int main() {
//   cout << "Starting simulation..." << endl;

int main(int argc, char *argv[]) {
  cout << "Starting simulation..." << endl;

  // CYLINDER TEST command-line override:  ./solver chi1 chi2
  if (argc >= 3) {
    chi1 = std::atof(argv[1]);
    chi2 = std::atof(argv[2]);
  }
  // Globals initialised from xi0 / tau_B before main() must be recomputed here
  omega    = omega_tauB / tau_B;
  Ms       = 3.0 * eta_h * phi_p * xi0 / (mu0 * tau_B * H_ref);
  xi_per_H = xi0 / H_ref;

  // ---- susceptibility map.  In mrsh_point:  M = phi_loc * Ms_local * L(xi_e),
  //      and here phi_loc = phi[i][j] = 1 (single fluid fills the domain), so
  //      M = Ms_local * L(xi) -> Ms_local * xi_per_H * H / 3 in the linear limit.
  //      Matching M = chi H gives  Ms_local = 3 chi / xi_per_H.  (No phi_p: the
  //      particle fraction is already absorbed into the target chi we choose.)
  const double xC = 0.5 * (nX - 1), yC = 0.5 * (nY - 1);
  auto Ms_of_chi = [&](double chi) { return 3.0 * chi / xi_per_H; };
  for (int i = 0; i < nX; i++)
    for (int j = 0; j < nY; j++) {
      double dr = sqrt((i - xC) * (i - xC) + (j - yC) * (j - yC));
      double chi = (dr <= R_cyl) ? chi1 : chi2;
      chi_map[i][j] = chi;
      Ms_map[i][j]  = Ms_of_chi(chi);
    }
  for (int s = 0; s < 2; s++)
    for (int j = 0; j < nY; j++)
      MsG[s][j] = Ms_of_chi(chi2);   // ghost columns are far outside -> region 2
  cout << "CYLINDER TEST: chi1=" << chi1 << " chi2=" << chi2
       << "  mu1/mu2=" << (1.0 + chi1) / (1.0 + chi2)
       << "  H_in/H0 (Eq53)=" << 2.0 * (1.0 + chi2) / ((1.0 + chi1) + (1.0 + chi2))
       << "  xi_e=" << xi_per_H * H_d << endl;

  const fs::path baseDir = "output_linear_field";
  // const fs::path subDir = baseDir / ("output_" + std::to_string(H0 * Href) + "_" + std::to_string(omega * tau_B));
  // const fs::path subDir = baseDir / ("output_xi0_" + std::to_string(xi0) + "_wtB_" + std::to_string(omega_tauB));
  const fs::path subDir = baseDir / ("output_xi0_" + std::to_string(xi0) + "_tauB_" + std::to_string(tau_B) +
                                     "_H0_" + std::to_string(H0) + "_wtB_" + std::to_string(omega_tauB));
  if (!fs::exists(baseDir))
    fs::create_directory(baseDir);
  if (!fs::exists(subDir))
    fs::create_directory(subDir);

  
  // Reference state at the channel-centre field: <Mx> and rotational-viscosity factor g
  double Mx_ref = 0.0, g_ref = 0.0;
  mrsh_reference(1e-4 / tau_B, Mx_ref, g_ref);
  if (G_target > 0.0) {
    if (fabs(Mx_ref) < 1e-12) {
      cerr << "ERROR: <Mx> = 0 at the channel centre (H_d = 0?): no net Kelvin drive." << endl;
      return 1;
    }
    k_grad = G_target / (mu0 * Mx_ref);     // mean Kelvin drive = mu0 * k_grad * <Mx>
  } else {
    k_grad = k_grad_input;
  }
  double G_ref   = mu0 * k_grad * Mx_ref;
  double eta_ref = eta_h * (1.0 + 1.5 * phi_p * g_ref);
  cout << "Ms = " << Ms << "  H_d = " << H_d << "  H0 = " << H0 << "  k_grad = " << k_grad
       << "  k_grad*nX = " << k_grad * nX << endl;
  double Mx_nd = 0.0, g_nd = 0.0;
  mrsh_reference(1e-4 / tau_B, Mx_nd, g_nd, false);   // same, without the demagnetising field
  cout << "Reference: <Mx> = " << Mx_ref << "  G = " << G_ref << "  g = " << g_ref
       << "  g(no demag) = " << g_nd << "  Ms/H_ref = " << Ms / H_ref
       << "  u_max(centreline) = " << G_ref * nY * nY / (8.0 * eta_ref) << endl;
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
      // Az[i][j] = mu0 * H0x(0) * (j + 0.5);
      // Hx[i][j] = H0x(0);
      // Hy[i][j] = H0y(0);
      // Hx_prev[i][j] = Hx[i][j];
      // Hy_prev[i][j] = Hy[i][j];
      // Mx[i][j] = 0.0;
      // My[i][j] = 0.0;
      // xie_x[i][j] = xi0;     // Bacri's initial condition:  xi_e = xi_0
      // xie_y[i][j] = 0.0;
      // compute_heq(i, j);
      // // for (int k = 0; k < Q; k++)
      // //   h[i][j][k] = heq[i][j][k];
      // for (int k = 0; k < Q; k++)
      //   h[i][j][k] = w[k]*Az[i][j] - w[k]*cY[k]*tau_h*dt*mu0*H0x(0);
      double xn = double(i), yn = j + 0.5;
      Az[i][j] = A_app(xn, yn, 0);
      Hx[i][j] = Hx_app(xn, 0);
      Hy[i][j] = Hy_app(yn);
      Hx_prev[i][j] = Hx[i][j];
      Hy_prev[i][j] = Hy[i][j];
      Mx[i][j] = 0.0;
      My[i][j] = 0.0;
      xie_x[i][j] = xi_per_H * Hx[i][j];   // xi_e = xi of the local applied field
      xie_y[i][j] = xi_per_H * Hy[i][j];
      compute_heq(i, j);
      double dAdx = -mu0 * Hy_app(yn);       //  dA/dx = -B_y
      double dAdy =  mu0 * Hx_app(xn, 0);    //  dA/dy =  B_x
      for (int k = 0; k < Q; k++)
        h[i][j][k] = w[k] * (Az[i][j] - tau_h * dt * (cX[k] * dAdx + cY[k] * dAdy));
    }

  // x-ghost columns (unrolled x = -1 and x = nX)
  for (int s = 0; s < 2; s++)
    for (int j = 0; j < nY; j++) {
      double xg = (s == 0) ? -1.0 : double(nX), yn = j + 0.5;
      HGx[s][j] = Hx_app(xg, 0);
      HGy[s][j] = Hy_app(yn);
      HGx_prev[s][j] = HGx[s][j];
      HGy_prev[s][j] = HGy[s][j];
      MGx[s][j] = 0.0;
      MGy[s][j] = 0.0;
      xieG_x[s][j] = xi_per_H * HGx[s][j];
      xieG_y[s][j] = xi_per_H * HGy[s][j];
    }

  // Force and flow initialisation
  for (int i = 0; i < nX; i++)
    for (int j = 0; j < nY; j++) {
      compute_magnetic_force(i, j);
      compute_surface_tension_force(i, j);
      // Fb_x[i][j] = 2e-7;
      Fb_x[i][j] = 0.0;
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

    // (CYLINDER TEST) freeze the flow: pure magnetostatics, no rotational coupling
    if (FREEZE_FLOW)
      for (int i = 0; i < nX; i++)
        for (int j = 0; j < nY; j++) { ux[i][j] = 0.0; uy[i][j] = 0.0; }

    // (2) vorticity at the CURRENT step
    for (int i = 0; i < nX; i++)
      for (int j = 0; j < nY; j++)
        vorticity_z[i][j] = FREEZE_FLOW ? 0.0 : 0.5 * curl_U(i, j);

    // (3) field + magnetisation, using the current vorticity
    for (int i = 0; i < nX; i++)
      for (int j = 0; j < nY; j++) {
        compute_Az(i, j);
        compute_H(i, j, iter);
        if (i == nX - 1) set_H_ghost(0, j);   // before M of the image column is updated
        if (i == 0)      set_H_ghost(1, j);
        compute_M(i, j, iter);
      }
    compute_M_ghosts();
    compute_seam_jump();

    compute_source_all(iter);

        // (3b) Relax the h lattice to the magnetostatic solution for the CURRENT M.
    //      A has no dynamics of its own -- lap(A) = -mu0 curl(M) is an
    //      instantaneous constraint -- so SA, the wall gamma and H0x(iter) are
    //      all frozen for the whole sub-cycle.  Nothing else in the time step
    //      advances here.
    // Predictor: shift h by the exact lattice solution of the change in the uniform applied
    // field between steps, so the sub-cycle only has to relax the induced (magnetisation) part.
    if (iter > 0 && H0 != 0.0) {
      double dH = H0 * (cos(omega * iter) - cos(omega * (iter - 1)));
      for (int i = 0; i < nX; i++)
        for (int j = 0; j < nY; j++) {
          double dA = mu0 * dH * ((j + 0.5) - y_c);
          for (int k = 0; k < Q; k++)
            h[i][j][k] += w[k] * (dA - tau_h * dt * cY[k] * mu0 * dH);
        }
    }

    for (int n = 0; n < NSUB_H; n++) {

      for (int i = 0; i < nX; i++)
        for (int j = 0; j < nY; j++) {
          compute_Az(i, j);
          compute_heq(i, j);
        }

      for (int i = 0; i < nX; i++)
        for (int j = 0; j < nY; j++) {
          double dh[Q], dif[Q];
          for (int k = 0; k < Q; k++) dh[k] = h[i][j][k] - heq[i][j][k];
          MRT_matrix_multiply(MinvShM, dh, dif);
          for (int k = 0; k < Q; k++)
            hstar[i][j][k] = h[i][j][k] - dif[k] + w[k] * SA[i][j] * dt;
        }

      for (int i = 0; i < nX; i++)
        for (int j = 0; j < nY; j++)
          for (int k = 0; k < Q; k++) {
            // int ip = i - cX[k], jp = j - cY[k];
            // if (jp < 0 || jp >= nY) {
            //   int jr = (jp < 0) ? (-jp - 1) : (2 * nY - 1 - jp);
            //   ip = (ip + nX) % nX;
            //   double gam = (jp < 0) ? -mu0*(H0x(iter) + Mxw_bot(ip))
            //                         : +mu0*(H0x(iter) + Mxw_top(ip));
            //   double Bw = 6.0 * dt * epsilon * gam;
            //   htemp[i][j][k] = hstar[ip][jr][specular_y[k]] + Bw * w[k];
            // } else {
            //   ip = (ip + nX) % nX;
            //   htemp[i][j][k] = hstar[ip][jp][k];
            // }
            int ip = i - cX[k], jp = j - cY[k];
            int sig = (ip < 0) ? 1 : ((ip >= nX) ? -1 : 0);   // x-seam crossing
            int ipw = (ip + nX) % nX;
            if (jp < 0 || jp >= nY) {
              int jr = (jp < 0) ? (-jp - 1) : (2 * nY - 1 - jp);
              int ks = specular_y[k];
              double xw = double(i) - 0.5 * cX[k];             // wall-hit point (unrolled)
              double Mw = (jp < 0) ? Mxw_bot(ipw) - sig * dMxw_bot    // Mx at the unrolled
                                   : Mxw_top(ipw) - sig * dMxw_top;   // source column
              double gam = (jp < 0) ? -mu0*(Hx_app(xw, iter) + Mw)
                                    : +mu0*(Hx_app(xw, iter) + Mw);
              double Bw = 6.0 * dt * epsilon * gam;
              htemp[i][j][k] = hstar[ipw][jr][ks] - sig * h_jump(ks, jr) + Bw * w[k];
            } else {
              htemp[i][j][k] = hstar[ipw][jp][k] - sig * h_jump(k, jp);
            }
          }

      memcpy(h, htemp, sizeof(h));
    }

    memcpy(Hx_prev, Hx, sizeof(Hx));
    memcpy(Hy_prev, Hy, sizeof(Hy));
    memcpy(HGx_prev, HGx, sizeof(HGx));
    memcpy(HGy_prev, HGy, sizeof(HGy));

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
        // compute_heq(i, j);
      }

    // Collision with MRT and forcing
    memset(gstar, 0, sizeof(gstar));
    memset(fstar, 0, sizeof(fstar));
    // memset(hstar, 0, sizeof(hstar));

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
        // double dh[Q];
        // for (int k = 0; k < Q; k++)
        //   dh[k] = h[i][j][k] - heq[i][j][k];
        // MRT_matrix_multiply(MinvShM, dh, diff_term);
        // for (int k = 0; k < Q; k++)
        //   hstar[i][j][k] = h[i][j][k] - diff_term[k] + w[k] * SA[i][j] * dt;
      }

    // Streaming + BCs
    memset(ftemp, 0, sizeof(ftemp));
    memset(gtemp, 0, sizeof(gtemp));
    // memset(htemp, 0, sizeof(htemp));

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
    // x: periodic (Az' is periodic)
    // y: specular reflection (∂ψ'/∂y = 0)

    // for (int i = 0; i < nX; i++)
    //   for (int j = 0; j < nY; j++)
    //     for (int k = 0; k < Q; k++) {
    //       int ip = i - cX[k], jp = j - cY[k];

    //       // if (jp < 0 || jp >= nY) {
    //       //   int jr = (jp < 0) ? (-jp - 1) : (2 * nY - 1 - jp);
    //       //   ip = (ip + nX) % nX;   // must wrap BEFORE indexing
    //       //   htemp[i][j][k] = hstar[ip][jr][specular_y[k]];
    //       if (jp < 0 || jp >= nY) {
    //         int jr = (jp < 0) ? (-jp - 1) : (2 * nY - 1 - jp);
    //         ip = (ip + nX) % nX;
    //         // dA/dn = mu0 (t.H0 + t.M),  t = n x z  ->  -x at bottom, +x at top
    //         double gam = (jp < 0) ? -mu0*(H0x(iter) + Mxw_bot(ip))
    //                               : +mu0*(H0x(iter) + Mxw_top(ip));
    //         double Bw = 6.0 * dt * epsilon * gam;     // 6 = 1/W_in for a flat D2Q9 wall
    //         htemp[i][j][k] = hstar[ip][jr][specular_y[k]] + Bw * w[k];
    //       } else {
    //         ip = (ip + nX) % nX; // periodic in x
    //         htemp[i][j][k] = hstar[ip][jp][k];
    //       }
    //     }

    memcpy(f, ftemp, sizeof(f));
    memcpy(g, gtemp, sizeof(g));
    // memcpy(h, htemp, sizeof(h));

    // Output
    if (iter % outputInterval == 0) {
      cout << "Writing iter " << iter << " ... ";
      std::string filename =
          (subDir / ("output_t_" + std::to_string(iter) + ".csv")).string();
      FILE *fp = fopen(filename.c_str(), "w");
      fprintf(fp, "i,j,phi,rho,ux,uy,Hx,Hy,Fm_x,Fm_y,Fk_x,Fk_y,Fs_x,Fs_y,Fb_x,Fb_y,Az,SA,Mx,My\n");
      double phi_sum = 0.0;
      double H_max_delta = 0.0;
      for (int i = 0; i < nX; i++)
        for (int j = 0; j < nY; j++) {
          phi_sum += phi[i][j];
          // H_max_delta = fmax(H_max_delta, fabs(Hx[i][j] - H0x(iter))/fabs(H0x_amp));
                    // Hx at this point was recovered from h relaxed with the field of step iter-1
          H_max_delta = fmax(H_max_delta,
                             fabs(Hx[i][j] - Hx_app(double(i), (iter > 0) ? iter - 1 : 0)) / H_ref);
          fprintf(fp,
                  "%d,%d,%10e,%10e,%10e,%10e,%10e,%10e,%10e,%10e,%10e,%10e,%"
                  "10e,%10e,%10e,%10e,%10e,%10e,%10e,%10e\n",
                  i, j, phi[i][j], rho[i][j], ux[i][j], uy[i][j], Hx[i][j],
                  Hy[i][j], Fm_x[i][j], Fm_y[i][j], Fk_x[i][j], Fk_y[i][j],
                  Fs_x[i][j], Fs_y[i][j],
                  Fb_x[i][j], Fb_y[i][j],Az[i][j],SA[i][j],Mx[i][j],My[i][j]);
        }
      fclose(fp);
      // cout << "phi_sum=" << phi_sum << endl;
      cout << "phi_sum=" << phi_sum << " A_resid=" << A_resid << " H_max_delta=" << H_max_delta << endl;
    }
  }
  return 0;
}