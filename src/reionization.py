"""
Core reionization solver and UVLF model.
Wang & Shan (2026) — "The Reionization Escape Fraction Tension is a Degeneracy"
"""
import numpy as np
from scipy.interpolate import interp1d

# Cosmological parameters (Planck 2018)
H0_km = 67.74
Om = 0.3089
Ob = 0.0486
OL = 1 - Om
Yp = 0.2454
XH = 1 - Yp
fb = Ob / Om
sigma8 = 0.811

# Physical constants
mp_g = 1.6726e-24
sigma_T = 6.6524e-25
c_cm = 2.9979e10
km_cm = 1e5
Mpc_cm = 3.0857e24
H0_s = H0_km * km_cm / Mpc_cm
rho_crit_0 = 3 * H0_s**2 / (8 * np.pi * 6.674e-8)
nH0 = rho_crit_0 * Ob * XH / mp_g
fe_he = 1 + Yp / (4 * XH)
aB = 2.56e-13 * 2**(-0.75)  # case-B at T=2e4 K
rho_m0 = Om * 2.775e11 * (H0_km / 100)**2

# Model parameters
MP = 1e11  # pivot mass for SFE
NDOT_REF = 10**(-22.12)  # calibrated emissivity normalisation
PRODUCT = 0.13 * 0.019  # f_esc * f_star baseline product


def H(z):
    """Hubble parameter at redshift z [s^-1]."""
    return H0_s * np.sqrt(Om * (1 + z)**3 + OL)


def dtdz(z):
    """dt/dz [s]."""
    return -1.0 / (H(z) * (1 + z))


def C_HII(z):
    """Clumping factor (Shull+2012)."""
    return 2.9 * ((1 + z) / 6)**(-1.1)


def rho_sfr_norm(z):
    """Normalised SFRD evolution."""
    r = (1 + z)**3.0 / (1 + ((1 + z) / 2.5)**4.0)
    return r / (9**3 / (1 + (9 / 2.5)**4))


def growth(z):
    """Linear growth factor D(z)/D(0)."""
    a = 1 / (1 + z)
    Omz = Om * (1 + z)**3 / (Om * (1 + z)**3 + OL)
    return 2.5 * a * Omz / (Omz**(4./7) - (1 - Omz) + (1 + Omz/2) * (1 + (1 - Omz)/70))


# Halo mass function (Sheth-Tormen with ad hoc normalisation)
logM_arr = np.linspace(8, 14, 250)
M_arr = 10**logM_arr


def sheth_tormen(M, z):
    """Sheth-Tormen HMF dn/dlog10M [Mpc^-3 dex^-1]."""
    sm = sigma8 * (M / 8e13)**(-0.2) * growth(z) / growth(0)
    nu = 1.686 / sm
    A, a, p = 0.3222, 0.707, 0.3
    return (20 * A * np.sqrt(2 * a / np.pi) * nu *
            (1 + (a * nu**2)**(-p)) *
            np.exp(-a * nu**2 / 2) * rho_m0 / M * 0.2)


def build_hmf_cache(redshifts=[9, 10, 11, 12]):
    """Pre-compute HMF at each redshift."""
    return {z: np.array([sheth_tormen(m, z) for m in M_arr]) for z in redshifts}


def solve_reionization(fesc, fstar0, nsteps=1500, zmin=5.0, zmax=30.0):
    """
    Solve the reionization ODE. Returns (tau, xHI_interp).
    """
    zs = np.linspace(zmax, zmin, nsteps)
    Q = 0.0
    zo, qo = [], []
    for i in range(1, len(zs)):
        z = zs[i]
        dt = abs(dtdz(z) * (zs[i] - zs[i-1]))
        ndot = fesc * (fstar0 * (1 + z)**0.13 / 0.019) * NDOT_REF * rho_sfr_norm(z)
        rec = C_HII(z) * aB * nH0 * (1 + z)**3 * Q
        Q = min(max(Q + (ndot / nH0 - rec) * dt, 0), 1)
        zo.append(z)
        qo.append(Q)
    za = np.array(zo)
    xa = 1 - np.array(qo)
    # Thomson optical depth
    tau = sum(
        fe_he * nH0 * (1 + za[i])**2 * (1 - xa[i]) * sigma_T * c_cm / H(za[i]) *
        abs(za[i] - za[i-1])
        for i in range(1, len(za))
    )
    xf = interp1d(za, xa, fill_value=(1, 0), bounds_error=False)
    return tau, xf


def chi2_reion(fesc, fstar0, tau_obs=0.054, sigma_tau=0.007):
    """
    chi^2 from tau_e and x_HI measurements.
    """
    tau, xf = solve_reionization(fesc, fstar0)
    c2 = ((tau - tau_obs) / sigma_tau)**2
    xHI_data = [(5.9, 0.04, 0.03), (7, 0.25, 0.12), (7.5, 0.40, 0.13),
                (8, 0.55, 0.15), (9, 0.80, 0.10)]
    for zd, xd, sd in xHI_data:
        c2 += ((float(xf(zd)) - xd) / sd)**2
    return c2


def uvlf_gaussian(m_uv_bins, z, fstar0, alpha_lo, sigma_uv, hmf_cache):
    """
    Predicted UVLF with Gaussian magnitude scatter.
    Returns log10(phi) at each m_uv_bin.
    """
    fs0z = fstar0 * (1 + z)**0.13
    fstar = 2 * fs0z / ((M_arr / MP)**(-alpha_lo) + (M_arr / MP)**0.5)
    SFR = np.maximum(fstar * fb * 25.3 * (M_arr / 1e12)**1.1 * (1 + z)**2.5, 1e-10)
    MUV = -17.7 - 2.5 * np.log10(SFR)
    hmf = hmf_cache[z]
    logphi = np.zeros(len(m_uv_bins))
    for j, muv in enumerate(m_uv_bins):
        kernel = np.exp(-0.5 * ((muv - MUV) / sigma_uv)**2) / (sigma_uv * np.sqrt(2 * np.pi))
        logphi[j] = np.log10(max(np.trapezoid(hmf * kernel, logM_arr), 1e-12))
    return logphi


def uvlf_lognormal(m_uv_bins, z, fstar0, alpha_lo, sigma_uv, hmf_cache):
    """
    Predicted UVLF with log-normal (asymmetric) scatter.
    Bright side sigma = 1.3*sigma_uv, faint side = 0.7*sigma_uv.
    """
    fs0z = fstar0 * (1 + z)**0.13
    fstar = 2 * fs0z / ((M_arr / MP)**(-alpha_lo) + (M_arr / MP)**0.5)
    SFR = np.maximum(fstar * fb * 25.3 * (M_arr / 1e12)**1.1 * (1 + z)**2.5, 1e-10)
    MUV_mean = -17.7 - 2.5 * np.log10(SFR)
    hmf = hmf_cache[z]
    logphi = np.zeros(len(m_uv_bins))
    for j, muv in enumerate(m_uv_bins):
        delta = muv - MUV_mean
        sig = np.where(delta > 0, sigma_uv * 0.7, sigma_uv * 1.3)
        kernel = np.exp(-0.5 * (delta / sig)**2) / (sig * np.sqrt(2 * np.pi))
        logphi[j] = np.log10(max(np.trapezoid(hmf * kernel, logM_arr), 1e-12))
    return logphi


def uvlf_dutycycle(m_uv_bins, z, fstar0, alpha_lo, f_burst, A_burst, hmf_cache):
    """
    Two-component duty-cycle UVLF model.
    Fraction (1-f_burst) at quiescent, f_burst at burst (A_burst x brighter).
    """
    fs0z = fstar0 * (1 + z)**0.13
    fstar = 2 * fs0z / ((M_arr / MP)**(-alpha_lo) + (M_arr / MP)**0.5)
    SFR_q = np.maximum(fstar * fb * 25.3 * (M_arr / 1e12)**1.1 * (1 + z)**2.5, 1e-10)
    MUV_q = -17.7 - 2.5 * np.log10(SFR_q)
    MUV_b = MUV_q - 2.5 * np.log10(max(A_burst, 1.1))
    hmf = hmf_cache[z]
    sig_q = 0.3
    logphi = np.zeros(len(m_uv_bins))
    for j, muv in enumerate(m_uv_bins):
        k_q = np.exp(-0.5 * ((muv - MUV_q) / sig_q)**2) / (sig_q * np.sqrt(2 * np.pi))
        k_b = np.exp(-0.5 * ((muv - MUV_b) / sig_q)**2) / (sig_q * np.sqrt(2 * np.pi))
        kernel = (1 - f_burst) * k_q + f_burst * k_b
        logphi[j] = np.log10(max(np.trapezoid(hmf * kernel, logM_arr), 1e-12))
    return logphi
