"""
Scan the redshift-dependent product P(z) = P0 * ((1+z)/(1+z_piv))^beta.
"""
import numpy as np
from scipy.interpolate import interp1d
from src.reionization import (
    H, dtdz, C_HII, rho_sfr_norm, nH0, fe_he, sigma_T, c_cm, aB, NDOT_REF
)


def solve_reion_beta(fesc_0, fstar0, beta, z_piv=8.0):
    """Solve reionization ODE with z-dependent product scaling."""
    zs = np.linspace(30, 5.0, 1500)
    Q = 0.0
    zo, qo = [], []
    for i in range(1, len(zs)):
        z = zs[i]
        dt = abs(dtdz(z) * (zs[i] - zs[i-1]))
        p_ratio = ((1 + z) / (1 + z_piv))**beta
        fesc_z = fesc_0 * p_ratio
        ndot = fesc_z * (fstar0 * (1 + z)**0.13 / 0.019) * NDOT_REF * rho_sfr_norm(z)
        Q = min(max(Q + (ndot / nH0 - C_HII(z) * aB * nH0 * (1 + z)**3 * Q) * dt, 0), 1)
        zo.append(z)
        qo.append(Q)
    za = np.array(zo)
    xa = 1 - np.array(qo)
    tau = sum(
        fe_he * nH0 * (1 + za[i])**2 * (1 - xa[i]) * sigma_T * c_cm / H(za[i]) *
        abs(za[i] - za[i-1])
        for i in range(1, len(za))
    )
    xf = interp1d(za, xa, fill_value=(1, 0), bounds_error=False)
    return tau, xf


def chi2_beta(beta, fesc_0=0.13, fstar0=0.019):
    """chi^2 for a given beta."""
    tau, xf = solve_reion_beta(fesc_0, fstar0, beta)
    c2 = ((tau - 0.054) / 0.007)**2
    for zd, xd, sd in [(5.9, 0.04, 0.03), (7, 0.25, 0.12), (7.5, 0.40, 0.13),
                        (8, 0.55, 0.15), (9, 0.80, 0.10)]:
        c2 += ((float(xf(zd)) - xd) / sd)**2
    return c2


def run_beta_scan():
    """Scan beta and find 2-sigma range."""
    betas = np.linspace(-0.8, 0.8, 33)
    chi2s = np.array([chi2_beta(b) for b in betas])
    chi2_min = np.min(chi2s)
    dc2 = chi2s - chi2_min
    beta_best = betas[np.argmin(chi2s)]

    print(f'Best beta = {beta_best:.3f}, chi2_min = {chi2_min:.2f}\n')
    print(f'{"beta":>8s} {"chi2":>8s} {"dchi2":>8s} {"tau":>8s} {"xHI(7)":>8s}')
    for b in [-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6]:
        tau, xf = solve_reion_beta(0.13, 0.019, b)
        c2 = chi2_beta(b)
        print(f'{b:+8.2f} {c2:8.2f} {c2-chi2_min:+8.2f} {tau:8.4f} {float(xf(7)):8.3f}')

    # 2-sigma bounds
    dc2_fn = interp1d(betas, dc2)
    lo, hi = -0.8, beta_best
    for _ in range(30):
        mid = (lo + hi) / 2
        if dc2_fn(mid) > 4: lo = mid
        else: hi = mid
    beta_lo = (lo + hi) / 2

    lo, hi = beta_best, 0.8
    for _ in range(30):
        mid = (lo + hi) / 2
        if dc2_fn(mid) < 4: lo = mid
        else: hi = mid
    beta_hi = (lo + hi) / 2

    print(f'\n2-sigma range: beta in [{beta_lo:.2f}, {beta_hi:.2f}]')
    for b in [beta_lo, 0, beta_hi]:
        ratio = ((1 + 12) / (1 + 8))**b
        print(f'  beta={b:+.2f}: P(z=12)/P(z=8) = {ratio:.3f} ({(ratio-1)*100:+.1f}%)')

    return betas, dc2, beta_best, beta_lo, beta_hi


if __name__ == '__main__':
    run_beta_scan()
