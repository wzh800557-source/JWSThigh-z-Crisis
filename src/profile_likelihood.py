"""
Joint profile likelihood scan over (f_star, f_esc).
Produces the 2D grid (Fig 1) and 1D profile (Fig 2a).
"""
import numpy as np
from scipy.optimize import minimize
from src.reionization import (
    chi2_reion, uvlf_gaussian, uvlf_lognormal, uvlf_dutycycle,
    build_hmf_cache, PRODUCT
)
from src.data import load_uvlf_data


def chi2_uvlf(fstar0, alpha_lo, sigma_uv, uvlf_data, hmf_cache, model='gaussian'):
    """Total UVLF chi^2 across all redshifts."""
    c2 = 0
    for z, (ms, ls, ss) in uvlf_data.items():
        if model == 'gaussian':
            mod = uvlf_gaussian(ms, z, fstar0, alpha_lo, sigma_uv, hmf_cache)
        elif model == 'lognormal':
            mod = uvlf_lognormal(ms, z, fstar0, alpha_lo, sigma_uv, hmf_cache)
        c2 += np.sum((mod - ls)**2 / ss**2)
    return c2


def profile_at(fstar, fesc, uvlf_data, hmf_cache, model='gaussian'):
    """
    Profile over nuisance parameters at a given (fstar, fesc).
    Returns (total_chi2, [alpha_lo, sigma_uv]).
    """
    cr = chi2_reion(fesc, fstar)
    best = 1e10
    bx = [2.2, 0.8]
    for a0 in [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 4.0]:
        for s0 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]:
            try:
                r = minimize(
                    lambda x: chi2_uvlf(fstar, max(x[0], 1.0), max(x[1], 0.1),
                                        uvlf_data, hmf_cache, model) + cr,
                    [a0, s0], method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 0.01}
                )
                if r.fun < best:
                    best = r.fun
                    bx = list(r.x)
            except Exception:
                pass
    return best, bx


def profile_dutycycle_at(fstar, fesc, uvlf_data, hmf_cache):
    """Profile over (alpha_lo, f_burst, A_burst) for duty-cycle model."""
    cr = chi2_reion(fesc, fstar)
    best = 1e10
    bx = [2.2, 0.1, 3.0]

    def chi2_dc(fstar0, alo, fb, ab):
        c2 = 0
        for z, (ms, ls, ss) in uvlf_data.items():
            mod = uvlf_dutycycle(ms, z, fstar0, alo, fb, ab, hmf_cache)
            c2 += np.sum((mod - ls)**2 / ss**2)
        return c2

    for a0 in [1.5, 2.0, 2.5, 3.0, 3.5]:
        for fb0 in [0.05, 0.10, 0.20, 0.30, 0.50]:
            for ab0 in [2.0, 3.0, 5.0, 8.0]:
                try:
                    r = minimize(
                        lambda x: chi2_dc(fstar, max(x[0], 1.0),
                                          min(max(x[1], 0.01), 0.99),
                                          max(x[2], 1.1)) + cr,
                        [a0, fb0, ab0], method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 0.01}
                    )
                    if r.fun < best:
                        best = r.fun
                        bx = list(r.x)
                except Exception:
                    pass
    return best, bx


def run_2d_scan(nfs=14, nfe=16, model='gaussian'):
    """Run full 2D profile likelihood scan."""
    uvlf_data = load_uvlf_data()
    hmf_cache = build_hmf_cache()
    fs_g = np.geomspace(0.008, 0.055, nfs)
    fe_g = np.geomspace(0.03, 0.28, nfe)
    chi2_2d = np.full((nfs, nfe), np.nan)

    for i, fs in enumerate(fs_g):
        for j, fe in enumerate(fe_g):
            c2, _ = profile_at(fs, fe, uvlf_data, hmf_cache, model)
            chi2_2d[i, j] = c2
            print(f'\r  {i*nfe+j+1}/{nfs*nfe}', end='', flush=True)

    print()
    c2min = np.nanmin(chi2_2d)
    dc2 = chi2_2d - c2min
    np.savez('data/joint_profile_2d.npz', fs=fs_g, fe=fe_g, dc2=dc2, c2min=c2min)
    print(f'Saved data/joint_profile_2d.npz (min chi2={c2min:.1f})')
    return fs_g, fe_g, dc2


def run_1d_scan(model='gaussian'):
    """Run 1D profile along the ridge."""
    uvlf_data = load_uvlf_data()
    hmf_cache = build_hmf_cache()
    fstar_scan = np.array([0.008, 0.010, 0.012, 0.015, 0.019, 0.022, 0.025,
                           0.027, 0.030, 0.033, 0.035, 0.040, 0.050])
    results = []
    for fs in fstar_scan:
        fe = PRODUCT / fs
        c2, xo = profile_at(fs, fe, uvlf_data, hmf_cache, model)
        results.append({'fstar': fs, 'chi2': c2, 'alpha_lo': xo[0], 'sigma_uv': xo[1]})
        print(f'  f*={fs:.4f}: chi2={c2:.2f}, alo={xo[0]:.3f}, suv={xo[1]:.3f}')

    np.savez('data/profile_1d.npz', results=results)
    return results


def run_scatter_test():
    """Test Gaussian, log-normal, and duty-cycle at baseline and crisis."""
    uvlf_data = load_uvlf_data()
    hmf_cache = build_hmf_cache()

    results = {}
    for label, fs in [('baseline', 0.019), ('crisis', 0.035)]:
        fe = PRODUCT / fs
        c2_g, x_g = profile_at(fs, fe, uvlf_data, hmf_cache, 'gaussian')
        c2_l, x_l = profile_at(fs, fe, uvlf_data, hmf_cache, 'lognormal')
        c2_d, x_d = profile_dutycycle_at(fs, fe, uvlf_data, hmf_cache)
        results[label] = {
            'gaussian': (c2_g, x_g), 'lognormal': (c2_l, x_l), 'dutycycle': (c2_d, x_d)
        }
        print(f'\n{label} (f*={fs}):')
        print(f'  Gaussian:    chi2={c2_g:.2f}')
        print(f'  Log-normal:  chi2={c2_l:.2f}')
        print(f'  Duty-cycle:  chi2={c2_d:.2f}')

    for model in ['gaussian', 'lognormal', 'dutycycle']:
        dc2 = results['crisis'][model][0] - results['baseline'][model][0]
        print(f'\nDelta-chi2 ({model}): {dc2:.1f}')
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '2d':
        run_2d_scan()
    elif len(sys.argv) > 1 and sys.argv[1] == 'scatter':
        run_scatter_test()
    else:
        run_1d_scan()
