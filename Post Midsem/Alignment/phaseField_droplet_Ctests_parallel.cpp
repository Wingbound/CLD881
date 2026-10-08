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

const int nX = 50;              // compile-time: sizes the static arrays
const int nY = 50;

// ===================== RUNTIME CONTROL via ENV VARS ==================
// Every parameter a sweep varies is read from an environment variable at
// startup, with the old default as fallback.  This lets ONE binary run
// many cases in parallel -- the shell script sets  MS=.. FB=.. ./Bpar  per
// case, and each writes to its own output folder (RUN_TAG below).  With no
// env vars set the defaults reproduce the verified standalone run.
//
//   env var     meaning                         default
//   --------    -----------------------------   -------
//   DROPLET     1 two-phase / 0 Bacri phi==1     1
//   DC_FIELD    1 steady / 0 AC cosine           1
//   DROP_R      droplet radius (lattice units)   10
//   MS          saturation magnetisation Ms      0.01   (Bo~0.17; <~0.1 only!)
//   FB          body force Fb_x                   0.0    (pure field; 2e-7 adds flow)
//   XI0         Langevin parameter at peak        5.0
//   WTB         omega*tau_B (AC only)             1.5
//   NSTEPS      total steps                       80000
//   OUTEVERY    output interval                   2000
//   RUN_TAG     folder suffix (keep runs apart)   ""    (auto if empty)
// =====================================================================
#include <cstdlib>
static double env_d(const char* k, double def){ const char* v=getenv(k); return v? atof(v):def; }
static int    env_i(const char* k, int    def){ const char* v=getenv(k); return v? atoi(v):def; }
static bool   env_b(const char* k, bool   def){ const char* v=getenv(k); return v? (atoi(v)!=0):def; }
static std::string env_s(const char* k, const char* def){ const char* v=getenv(k); return v? std::string(v):std::string(def); }

const bool   DROPLET     = env_b("DROPLET", true);
const bool   DC_FIELD    = env_b("DC_FIELD", true);
const double DROP_RADIUS = env_d("DROP_R", 10.0);
const double Ms_FIXED    = env_d("MS", 0.01);
const double FB_X        = env_d("FB", 0.0);

int nSteps         = env_i("NSTEPS", 80000);
int outputInterval = env_i("OUTEVERY", 2000);

const int NSUB_M = 8;

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
double xi0        = env_d("XI0", 5.0);   // Langevin parameter at the field peak
double omega_tauB = env_d("WTB", 1.5);   // dimensionless frequency  omega*tau_B
double phi_p      = 0.2;    // PARTICLE volume fraction (the theory's phi)


// ---- numerical scales ------------------------------------------------
double tau_B = 20.0;                // Brownian time, in lattice steps
double omega = omega_tauB / tau_B;  // field angular frequency
double H0x_amp = 1.73;
double H0y_amp = 0.0;   // set direction via amplitude ratio
double H0 = sqrt(pow(H0x_amp,2.0)+pow(H0y_amp,2.0));
inline double H0x(int iter) { return DC_FIELD ? H0x_amp : H0x_amp * cos(omega * iter); }
inline double H0y(int iter) { return DC_FIELD ? H0y_amp : H0y_amp * cos(omega * iter); }

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
double theta   = 0.0;               //Angle of inclination of magnetic field with respect to x-axis

// Ms: for the Bacri validation it is tied to xi0 (uniform magnetic fluid).
// For a droplet it is a fixed MATERIAL property of the ferrofluid phase and
// must NOT scale with the sweep parameter -- otherwise the drop's saturation
// magnetisation changes every time the field amplitude changes.
double Ms       = DROPLET ? Ms_FIXED
                          : 3.0 * eta_h * phi_p * xi0 / (mu0 * tau_B * H0);
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
double Sh3 = 1 / (epsilon * c_s_inv2 / dt + 0.5);
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

static double Az[nX][nY];
static double Hx[nX][nY];
static double Hy[nX][nY];
static double Hx_prev[nX][nY];
static double Hy_prev[nX][nY];
static double Mx[nX][nY];
static double My[nX][nY];
static double xie_x[nX][nY];        // effective field, x component
static double xie_y[nX][nY];        // effective field, y component

static double SA[nX][nY];
double A_resid = 0.0;                          // diagnostic

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
    gsum += mu0*(H0x(iter) + Mxw_top(i)) - mu0*(H0x(iter) + Mxw_bot(i));
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

inline void compute_M(int i, int j, int iter) {
  const double XMIN = 1e-10;              // only guards xi_e == 0 exactly

  double phi_loc = phi[i][j];
  // Floor to 0 (not 1e-6): M ~ phi, so the non-magnetic carrier (phi=0)
  // must give M=0 exactly.  A 1e-6 floor would make the carrier weakly
  // magnetic and pollute the field solve.
  if (!std::isfinite(phi_loc) || phi_loc < 0.0) phi_loc = 0.0;

  double Omega = vorticity_z[i][j];
  if (!std::isfinite(Omega)) Omega = 0.0;

  double ex = xie_x[i][j], ey = xie_y[i][j];
  if (!std::isfinite(ex)) ex = 0.0;
  if (!std::isfinite(ey)) ey = 0.0;

  const double t0 = double(iter) - 1.0;   // integrate [iter-1, iter]

  auto rhs = [&](double s, double ax, double ay, double &dax, double &day) {
    // double hx = Hx[i][j], hy = Hy[i][j];
    const double f = s / dt;                       // s runs 0 -> dt over the step
    double hx = (1.0 - f)*Hx_prev[i][j] + f*Hx[i][j];
    double hy = (1.0 - f)*Hy_prev[i][j] + f*Hy[i][j];
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
  xie_x[i][j] = ex;  xie_y[i][j] = ey;

  // M = phi * Ms * L(xi_e) * xi_e/|xi_e|          [Bacri Eq. (5)]
  double xe = sqrt(ex*ex + ey*ey);
  if (xe < XMIN) {
    Mx[i][j] = phi_loc*Ms*ex/3.0;                 // L -> xi/3
    My[i][j] = phi_loc*Ms*ey/3.0;
  } else {
    double Le = langevin3D(xe);
    Mx[i][j] = phi_loc*Ms*Le*(ex/xe);
    My[i][j] = phi_loc*Ms*Le*(ey/xe);
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

  // Per-case output folder: encodes every swept parameter plus an optional
  // RUN_TAG, so concurrently-running cases NEVER write to the same directory.
  const fs::path baseDir = env_s("OUTBASE", "output_morphology");
  char buf[256];
  snprintf(buf, sizeof(buf),
           "run_MS%.4g_FB%.3g_R%.3g_xi%.3g_wtB%.3g_%s%s",
           Ms_FIXED, FB_X, DROP_RADIUS, xi0, omega_tauB,
           DC_FIELD ? "DC" : "AC",
           DROPLET ? "" : "_FILLED");
  std::string tag = env_s("RUN_TAG", "");
  std::string leaf = std::string(buf) + (tag.empty() ? "" : ("_" + tag));
  const fs::path subDir = baseDir / leaf;
  fs::create_directories(subDir);

  // Echo the resolved parameters so each run's log is self-describing.
  cout << "PARAMS  droplet=" << DROPLET << " dc_field=" << DC_FIELD
       << " R=" << DROP_RADIUS << " Ms=" << Ms_FIXED << " Fb=" << FB_X
       << " xi0=" << xi0 << " wtB=" << omega_tauB
       << " nSteps=" << nSteps << "\n"
       << "OUTDIR  " << subDir.string() << endl;

  // Fresh summary.csv for this run (truncate any leftover from a prior run
  // into the same folder, so rows are never duplicated).
  { std::string sp=(subDir/"summary.csv").string(); FILE* s=fopen(sp.c_str(),"w"); if(s) fclose(s); }

  // Precompute g and h MRT matrix (time-invariant)
  matrix_multiply(Minv, Sg, M, MinvSgM);
  matrix_multiply(Minv, Sh, M, MinvShM);

  double centerX = (nX - 1) / 2.0;
  double centerY = (nY - 1) / 2.0;
  for (int i = 0; i < nX; i++)
    for (int j = 0; j < nY; j++) {
      if (DROPLET) {
        double dist = sqrt((i-centerX)*(i-centerX) + (j-centerY)*(j-centerY));
        double W = D / 2.0;                         // interface half-width
        phi[i][j] = 0.5 * (1.0 + tanh((DROP_RADIUS - dist)/W));
      } else {
        phi[i][j] = 1.0;
      }
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
      Az[i][j] = mu0 * H0x(0) * (j + 0.5);
      Hx[i][j] = H0x(0);
      Hy[i][j] = H0y(0);
      Hx_prev[i][j] = Hx[i][j];
      Hy_prev[i][j] = Hy[i][j];
      Mx[i][j] = 0.0;
      My[i][j] = 0.0;
      // Seed the effective field only where the magnetic phase is present.
      // In the carrier (phi=0) xi_e starts at 0; inside the drop it starts
      // aligned with the applied field, as in the Bacri run.
      xie_x[i][j] = (DROPLET ? phi[i][j] : 1.0) * xi0;
      xie_y[i][j] = 0.0;
      compute_heq(i, j);
      for (int k = 0; k < Q; k++)
        h[i][j][k] = w[k]*Az[i][j] - w[k]*cY[k]*tau_h*dt*mu0*H0x(0);
    }

  // Force and flow initialisation
  for (int i = 0; i < nX; i++)
    for (int j = 0; j < nY; j++) {
      compute_magnetic_force(i, j);
      compute_surface_tension_force(i, j);
      Fb_x[i][j] = FB_X;
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
        compute_Az(i, j);
        compute_H(i, j, iter);
        compute_M(i, j, iter);
      }
    
    compute_source_all(iter);

        // (3b) Relax h to the magnetostatic solution for the CURRENT M.
    const double H_TOL = 1e-6;     // relative L2 change in Az (tightened for droplet;
                                   // the 1e-4 value was validated only for phi==1)
    static double Az_old[nX][nY];
    int h_iters = 0;
    double resid;

    // baseline Az before the first sub-step (h unchanged since step (3))
    for (int i = 0; i < nX; i++)
      for (int j = 0; j < nY; j++) {
        compute_Az(i, j);
        Az_old[i][j] = Az[i][j];
      }

    do {
      for (int i = 0; i < nX; i++)
        for (int j = 0; j < nY; j++)
          compute_heq(i, j);

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
            int ip = i - cX[k], jp = j - cY[k];
            if (jp < 0 || jp >= nY) {
              int jr = (jp < 0) ? (-jp - 1) : (2 * nY - 1 - jp);
              ip = (ip + nX) % nX;
              double gam = (jp < 0) ? -mu0*(H0x(iter) + Mxw_bot(ip))
                                    : +mu0*(H0x(iter) + Mxw_top(ip));
              double Bw = 6.0 * dt * epsilon * gam;
              htemp[i][j][k] = hstar[ip][jr][specular_y[k]] + Bw * w[k];
            } else {
              ip = (ip + nX) % nX;
              htemp[i][j][k] = hstar[ip][jp][k];
            }
          }

      memcpy(h, htemp, sizeof(h));
      h_iters++;

      // new Az from streamed h; compare to previous, then roll forward
      double num = 0.0, den = 0.0;
      for (int i = 0; i < nX; i++)
        for (int j = 0; j < nY; j++) {
          compute_Az(i, j);
          double d = Az[i][j] - Az_old[i][j];
          num += d * d;
          den += Az[i][j] * Az[i][j];
          Az_old[i][j] = Az[i][j];
        }
      resid = sqrt(num / (den + 1e-30));
    } while (resid > H_TOL);

    memcpy(Hx_prev, Hx, sizeof(Hx));
    memcpy(Hy_prev, Hy, sizeof(Hy));

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
      }

    // Collision with MRT and forcing
    memset(gstar, 0, sizeof(gstar));
    memset(fstar, 0, sizeof(fstar));

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
      }

    // Streaming + BCs
    memset(ftemp, 0, sizeof(ftemp));
    memset(gtemp, 0, sizeof(gtemp));

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

    memcpy(f, ftemp, sizeof(f));
    memcpy(g, gtemp, sizeof(g));

    // Output
    if (iter % outputInterval == 0) {
      cout << "Writing iter " << iter << " ... ";
      std::string filename =
          (subDir / ("output_t_" + std::to_string(iter) + ".csv")).string();
      FILE *fp = fopen(filename.c_str(), "w");
      fprintf(fp, "i,j,phi,rho,ux,uy,Hx,Hy,Fm_x,Fm_y,Fs_x,Fs_y,Fb_x,Fb_y,Az,SA,Mx,My\n");
      double phi_sum = 0.0;
      double H_max_delta = 0.0;
      for (int i = 0; i < nX; i++)
        for (int j = 0; j < nY; j++) {
          phi_sum += phi[i][j];
          H_max_delta = fmax(H_max_delta, fabs(Hx[i][j] - H0x(iter))/fabs(H0x_amp));
          fprintf(fp,
                  "%d,%d,%10e,%10e,%10e,%10e,%10e,%10e,%10e,%10e,%10e,%10e,%"
                  "10e,%10e,%10e,%10e,%10e,%10e\n",
                  i, j, phi[i][j], rho[i][j], ux[i][j], uy[i][j], Hx[i][j],
                  Hy[i][j], Fm_x[i][j], Fm_y[i][j], Fs_x[i][j], Fs_y[i][j],
                  Fb_x[i][j], Fb_y[i][j],Az[i][j],SA[i][j],Mx[i][j],My[i][j]);
        }
      fclose(fp);

      // ---- Taylor deformation from the second moments of the phi MASK ----
      // Three corrections over the naive version:
      //  (1) MASK to the drop (phi>0.5), weight w=phi, so the diffuse
      //      background and the carrier do not enter the moments.
      //  (2) PERIODIC x centroid via the circular-mean trick: map i to an
      //      angle, average on the unit circle, map back.  Without this a
      //      drop that translates across the x-boundary (B2 with flow!) gets
      //      a hugely inflated Cxx and a spurious D ~ 0.5.  y is a hard wall,
      //      so y uses the ordinary mean.
      //  (3) BOUNDARY-CONTACT flag: if any masked cell touches a y-wall or
      //      the drop's x-extent approaches nX, D is a finite-size artifact
      //      (the drop hit the box) -- reported as bdy=1, not a real shape.
      const double TWO_PI = 2.0*M_PI;
      double m=0, Sx_c=0, Sx_s=0, my=0;   // circular accumulators for x
      int ymin_hit=0, ymax_hit=0;
      for (int i=0;i<nX;i++) for(int j=0;j<nY;j++){
        if (phi[i][j] <= 0.5) continue;           // mask: drop interior only
        double w_=phi[i][j];
        double ax = TWO_PI*i/nX;
        m   += w_;
        Sx_c+= w_*cos(ax);  Sx_s+= w_*sin(ax);
        my  += w_*j;
        if (j==0)    ymin_hit=1;
        if (j==nY-1) ymax_hit=1;
      }
      double D_taylor=0, ang_drop=0; int bdy = (ymin_hit||ymax_hit)?1:0;
      if (m>1e-12) {
        double xbar = atan2(Sx_s, Sx_c);          // mean angle in x
        if (xbar<0) xbar+=TWO_PI;
        double xc = xbar*nX/TWO_PI;                // periodic x centroid
        double yc = my/m;
        double Cxx=0,Cyy=0,Cxy=0;
        double xext_min=1e9, xext_max=-1e9;
        for (int i=0;i<nX;i++) for(int j=0;j<nY;j++){
          if (phi[i][j] <= 0.5) continue;
          double w_=phi[i][j];
          double dxp = i - xc;                     // periodic minimal image in x
          dxp -= nX*round(dxp/nX);
          double dy  = j - yc;
          Cxx+=w_*dxp*dxp; Cyy+=w_*dy*dy; Cxy+=w_*dxp*dy;
          if (dxp<xext_min) xext_min=dxp;
          if (dxp>xext_max) xext_max=dxp;
        }
        Cxx/=m; Cyy/=m; Cxy/=m;
        double tr=Cxx+Cyy, dif=Cxx-Cyy;
        double rad=sqrt(dif*dif+4*Cxy*Cxy);
        double lam1=0.5*(tr+rad), lam2=0.5*(tr-rad);
        double a=sqrt(fmax(lam1,0.0)), b=sqrt(fmax(lam2,0.0));
        D_taylor=(a-b)/(a+b+1e-30);
        ang_drop=0.5*atan2(2*Cxy, dif)*180.0/M_PI;   // deg from x-axis
        // x-extent within ~2 cells of filling the box -> boundary contact
        if ((xext_max-xext_min) > nX-4) bdy=1;
      }

      // ===== C-TEST microscopic diagnostics (oscillating-field morphology) =====
      // All measured on the drop interior / interface at THIS instant; combined
      // with the dense within-cycle sampling they give the particle-level story:
      //   Happ      applied field now (phase reference)  H0x(iter)
      //   Mx_in,My_in : mean magnetisation inside the drop  -> global M-H lag
      //   |M|_tip   : |M| near the x-tips (poles, along the field axis)
      //   |M|_eq    : |M| near the y-extremes (equator)
      //             tip>eq means alignment concentrates at the poles that the
      //             field pulls on -- the microscopic driver of elongation.
      //   magstress : mean 0.5*mu0*|M|^2 over interface cells (0.1<phi<0.9)
      //               -- the magnetic normal stress the interface actually feels.
      //   Mlag_x    : instantaneous sign/size proxy = Mx_in / Happ (|.|<1 and
      //               its cycle-phase vs Happ gives the lag once sampled densely)
      double Happ = H0x(iter);
      double mm=0, Mxs=0, Mys=0;
      // locate drop centroid again (cheap) for tip/equator binning
      double xcen=0, ycen=0, mc=0;
      for (int i=0;i<nX;i++) for(int j=0;j<nY;j++) if(phi[i][j]>0.5){
        mc+=phi[i][j]; xcen+=phi[i][j]*i; ycen+=phi[i][j]*j; }
      if (mc>1e-12){ xcen/=mc; ycen/=mc; }
      double Mtip=0, ntip=0, Meq=0, neq=0, mstress=0, nif=0;
      for (int i=0;i<nX;i++) for(int j=0;j<nY;j++){
        double ph=phi[i][j];
        double Mmag=sqrt(Mx[i][j]*Mx[i][j]+My[i][j]*My[i][j]);
        if (ph>0.5){ mm+=ph; Mxs+=ph*Mx[i][j]; Mys+=ph*My[i][j];
          double dxp=i-xcen; dxp-=nX*round(dxp/nX); double dy=j-ycen;
          // tip: |dx|>|dy| and far out;  equator: |dy|>|dx| and far out
          double r=sqrt(dxp*dxp+dy*dy);
          if (r>3.0){
            if (fabs(dxp)>fabs(dy)){ Mtip+=Mmag; ntip++; }
            else                   { Meq +=Mmag; neq++;  }
          }
        }
        if (ph>0.1 && ph<0.9){ mstress+=0.5*mu0*Mmag*Mmag; nif++; }
      }
      double Mx_in = mm>1e-12? Mxs/mm : 0.0;
      double My_in = mm>1e-12? Mys/mm : 0.0;
      double Mtip_m = ntip>0? Mtip/ntip : 0.0;
      double Meq_m  = neq >0? Meq /neq  : 0.0;
      double magstress = nif>0? mstress/nif : 0.0;

      // Robust key=value line (explicit spaces; no glued fields).
      printf("phi_sum=%.6f A_resid=%.3e H_max_delta=%.4e "
             "h_iters=%d h_resid=%.3e D=%.6f ang=%.3f bdy=%d "
             "Happ=%.5f Mx_in=%.4e My_in=%.4e Mtip=%.4e Meq=%.4e magstress=%.4e\n",
             phi_sum, A_resid, H_max_delta, h_iters, resid,
             D_taylor, ang_drop, bdy,
             Happ, Mx_in, My_in, Mtip_m, Meq_m, magstress);
      fflush(stdout);

      // Append one row per snapshot to a per-run summary CSV (easy to plot).
      {
        std::string spath = (subDir / "summary.csv").string();
        bool needhdr = !fs::exists(spath) || fs::file_size(spath)==0;
        FILE* sf = fopen(spath.c_str(), "a");
        if (sf) {
          if (needhdr) fprintf(sf, "iter,D,ang,bdy,phi_sum,h_iters,h_resid,"
                                   "Happ,Mx_in,My_in,Mtip,Meq,magstress\n");
          fprintf(sf, "%d,%.8e,%.6f,%d,%.6f,%d,%.3e,"
                      "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                  iter, D_taylor, ang_drop, bdy, phi_sum, h_iters, resid,
                  Happ, Mx_in, My_in, Mtip_m, Meq_m, magstress);
          fclose(sf);
        }
      }
    }
  }
  return 0;
}