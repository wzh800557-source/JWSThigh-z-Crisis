"""
f_esc(z) reconstruction from the degeneracy ridge + JWST f_star(z).
Includes correlated MC draws.
"""
import numpy as np
from src.data import (FSTAR_DONNAN, FSTAR_PRADA, LOG_XIION_FID,
                      XIION_FID, XIION_SIGMA)
from src.reionization import PRODUCT


def reconstruct_fesc(fstar_model, N_mc=5000, rho_corr=0.0, seed=42):
    """
    Reconstruct f_esc(z) via Monte Carlo.

    Parameters
    ----------
    fstar_model : dict with 'z', 'fstar', 'fstar_err'
    N_mc : number of MC draws
    rho_corr : correlation coefficient between P0 and f_star
    seed : random seed

    Returns
    -------
    dict with z, fesc_median, fesc_lo, fesc_hi arrays
    """
    np.random.seed(seed)
    z_pts = fstar_model['z']
    fs_mean = fstar_model['fstar']
    fs_err = fstar_model['fstar_err']

    fesc_med, fesc_lo, fesc_hi = [], [], []

    for i in range(len(z_pts)):
        sp = PRODUCT * 0.13  # sigma on product
        sf = fs_err[i]

        if rho_corr > 0:
            # Correlated bivariate draw
            cov = [[sp**2, rho_corr * sp * sf],
                   [rho_corr * sp * sf, sf**2]]
            draws = np.random.multivariate_normal([PRODUCT, fs_mean[i]], cov, N_mc)
            p_draws = draws[:, 0]
            fs_draws = np.clip(draws[:, 1], 0.005, 0.060)
        else:
            p_draws = np.random.normal(PRODUCT, sp, N_mc)
            fs_draws = np.clip(np.random.normal(fs_mean[i], sf, N_mc), 0.005, 0.060)

        xi_draws = 10**np.random.normal(LOG_XIION_FID[i], XIION_SIGMA, N_mc)
        fe_draws = np.clip(p_draws / fs_draws * (XIION_FID / xi_draws), 0.01, 0.60)

        med = np.median(fe_draws)
        fesc_med.append(med)
        fesc_lo.append(med - np.percentile(fe_draws, 16))
        fesc_hi.append(np.percentile(fe_draws, 84) - med)

    return {
        'z': z_pts,
        'fesc': np.array(fesc_med),
        'fesc_lo': np.array(fesc_lo),
        'fesc_hi': np.array(fesc_hi),
        'label': fstar_model['label'],
    }


def run_reconstruction():
    """Run reconstruction for all three models, print Table II."""
    from src.data import FSTAR_WEIBEL
    print('=== f_esc RECONSTRUCTION (Table II) ===\n')

    r_d = reconstruct_fesc(FSTAR_DONNAN)
    r_p = reconstruct_fesc(FSTAR_PRADA)
    r_w = reconstruct_fesc(FSTAR_WEIBEL)

    print(f'{"z":>4s}  {"fesc(D)":>18s}  {"fesc(P)":>18s}  {"fesc(W)":>18s}')
    for i in range(len(r_d['z'])):
        z = r_d['z'][i]
        fd = f"{r_d['fesc'][i]:.3f}+{r_d['fesc_hi'][i]:.3f}/-{r_d['fesc_lo'][i]:.3f}"
        fp = f"{r_p['fesc'][i]:.3f}+{r_p['fesc_hi'][i]:.3f}/-{r_p['fesc_lo'][i]:.3f}"
        fw = f"{r_w['fesc'][i]:.3f}+{r_w['fesc_hi'][i]:.3f}/-{r_w['fesc_lo'][i]:.3f}"
        print(f"{z:4.0f}  {fd:>18s}  {fp:>18s}  {fw:>18s}")

    # Correlated MC test
    print('\n=== Correlated MC (constant SFE, z=12) ===')
    for rho in [0.0, 0.3, 0.5]:
        r = reconstruct_fesc(FSTAR_DONNAN, rho_corr=rho)
        i = -1  # z=12
        print(f'  rho={rho}: fesc={r["fesc"][i]:.4f} '
              f'+{r["fesc_hi"][i]:.4f}/-{r["fesc_lo"][i]:.4f}  '
              f'width={r["fesc_lo"][i]+r["fesc_hi"][i]:.4f}')

    # Intrinsic f_esc (Move 2)
    print('\n=== Intrinsic f_esc (xi_ion removed) ===')
    from src.reionization import PRODUCT
    for label, fs in [('Donnan', FSTAR_DONNAN['fstar']),
                      ('Prada', FSTAR_PRADA['fstar']),
                      ('Weibel', FSTAR_WEIBEL['fstar'])]:
        fesc_int = PRODUCT / fs
        slope = (fesc_int[-1] - fesc_int[0]) / (12.0 - 7.0)
        print(f'  {label}: fesc_int(z=7)={fesc_int[0]:.3f}, '
              f'fesc_int(z=12)={fesc_int[-1]:.3f}, slope={slope:+.4f}/z')

    return r_d, r_p, r_w


if __name__ == '__main__':
    run_reconstruction()
