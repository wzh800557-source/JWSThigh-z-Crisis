"""
Exact UVLF data from Donnan et al. (2024) MNRAS 533, 3222, Table 2.
phi in units of 10^{-6} Mpc^{-3} mag^{-1}.
Errors inflated by x1.92 so chi^2/dof ~ 1 at baseline.
"""
import numpy as np

INFLATE = 1.92  # error inflation factor for chi^2/dof ~ 1

# (z, MUV, phi, phi_err_up, phi_err_down) — all phi in 1e-6
_RAW = [
    # z = 9
    (9, -20.75, 12, 8, 5), (9, -20.25, 32, 13, 10), (9, -19.75, 144, 30, 28),
    (9, -19.25, 235, 60, 49), (9, -18.55, 486, 157, 139), (9, -18.05, 1110, 310, 310),
    (9, -17.55, 1776, 578, 510),
    # z = 10
    (10, -20.75, 4, 10, 4), (10, -20.25, 27, 13, 10), (10, -19.75, 92, 25, 20),
    (10, -19.25, 177, 53, 45), (10, -18.55, 321, 127, 111), (10, -18.05, 686, 245, 223),
    (10, -17.55, 1278, 486, 432),
    # z = 11
    (11, -21.25, 7, 9, 5), (11, -20.75, 14, 11, 7), (11, -20.25, 38, 16, 13),
    (11, -19.75, 100, 37, 30), (11, -19.25, 144, 81, 63), (11, -18.75, 234, 118, 96),
    (11, -18.25, 641, 361, 281),
    # z = 12.5
    (12, -21.25, 3, 4, 2), (12, -20.75, 4, 5, 3), (12, -20.25, 16, 9, 6),
    (12, -19.75, 34, 23, 15), (12, -19.25, 43, 35, 22), (12, -18.75, 80, 51, 36),
    (12, -18.25, 217, 153, 104),
]


def load_uvlf_data(inflate=INFLATE):
    """
    Returns dict: {z: (MUV_array, logphi_array, sigma_array)}
    with errors in log10 space, inflated by the given factor.
    """
    data = []
    for z, muv, phi, pu, pd in _RAW:
        logphi = np.log10(phi * 1e-6)
        sig_up = np.log10((phi + pu) * 1e-6) - logphi
        sig_dn = logphi - np.log10(max(phi - pd, 1) * 1e-6)
        sig = 0.5 * (sig_up + sig_dn) * inflate
        data.append((z, muv, logphi, sig))

    grouped = {}
    for z in [9, 10, 11, 12]:
        pts = [(m, l, s) for zz, m, l, s in data if zz == z]
        grouped[z] = (
            np.array([p[0] for p in pts]),
            np.array([p[1] for p in pts]),
            np.array([p[2] for p in pts]),
        )
    return grouped


# f_star(z) models
FSTAR_DONNAN = {
    'label': 'Donnan+2024 (constant)',
    'z': np.array([7.0, 8.0, 9.0, 10.0, 12.0]),
    'fstar': np.array([0.017, 0.017, 0.018, 0.018, 0.019]),
    'fstar_err': np.array([0.004, 0.004, 0.004, 0.005, 0.005]),
}

FSTAR_PRADA = {
    'label': 'Prada+2026 (evolving)',
    'z': np.array([7.0, 8.0, 9.0, 10.0, 12.0]),
    'fstar': np.array([0.017, 0.020, 0.023, 0.027, 0.032]),
    'fstar_err': np.array([0.004, 0.005, 0.005, 0.006, 0.007]),
}

FSTAR_WEIBEL = {
    'label': 'Weibel+2024 (SMF)',
    'z': np.array([7.0, 8.0, 9.0, 10.0, 12.0]),
    'fstar': np.array([0.016, 0.016, 0.017, 0.017, 0.018]),
    'fstar_err': np.array([0.005, 0.005, 0.005, 0.006, 0.007]),
}

# xi_ion(z) fiducial evolution
LOG_XIION_FID = np.array([25.32, 25.35, 25.40, 25.42, 25.48])
XIION_FID = 10**25.35
XIION_SIGMA = 0.12  # dex
