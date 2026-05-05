#!/usr/bin/env python3
"""Generate all figures for Wang & Shan (2026)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.reionization import PRODUCT
from src.fesc_reconstruction import reconstruct_fesc
from src.data import FSTAR_DONNAN, FSTAR_PRADA

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'font.family': 'serif',
    'xtick.labelsize': 11, 'ytick.labelsize': 11
})


def plot_fig1(datafile='data/joint_profile_2d.npz'):
    """Fig 1: 2D joint profile likelihood."""
    d = np.load(datafile)
    fs_g, fe_g, dc2 = d['fs'], d['fe'], d['dc2']
    fig, ax = plt.subplots(figsize=(9, 7.5))
    FS, FE = np.meshgrid(fs_g, fe_g, indexing='ij')
    levels = [0, 1, 4, 9, 16, 25, 50]
    colors = ['#c8e6c9', '#e8f5e9', '#fff9c4', '#ffe0b2', '#ffccbc', '#f8bbd0', '#e1bee7']
    cf = ax.contourf(FS, FE, dc2, levels=levels, colors=colors, extend='max')
    cl = ax.contour(FS, FE, dc2, levels=[1, 4, 9], colors=['#444', '#222', '#444'],
                    linewidths=[0.8, 1.5, 0.8], linestyles=['--', '-', '--'])
    ax.clabel(cl, fmt={1: r'$1\sigma$', 4: r'$2\sigma$', 9: r'$3\sigma$'}, fontsize=9)
    fp = np.geomspace(0.008, 0.055, 200)
    valid = (PRODUCT / fp > 0.034) & (PRODUCT / fp < 0.26)
    ax.plot(fp[valid], PRODUCT / fp[valid], ':', color='#2e7d32', lw=1.8, alpha=0.5)
    ax.axvspan(0.035, 0.06, alpha=0.08, color='red', zorder=0)
    ax.text(0.048, 0.16, 'BK23 crisis zone', fontsize=9, ha='center',
            color='#b71c1c', style='italic', alpha=0.8, rotation=-45)
    ax.text(0.0105, 0.21, r'Allowed ($\varepsilon < 2.5\%$)', fontsize=9, ha='center',
            color='#1b5e20', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#e8f5e9', ec='#2e7d32', alpha=0.8))
    LIT = {'Robertson+15': (0.018, 0.13, 's', '#d32f2f'),
           'Naidu+20': (0.012, 0.20, '^', '#1565C0'),
           'Finkelstein+19': (0.025, 0.05, 'D', '#2e7d32'),
           'Prada+26': (0.035, 0.05, '*', '#7b1fa2'),
           'Ma+20 (FIRE-2)': (0.017, 0.10, 'o', '#e65100'),
           'Mason+23': (0.015, 0.12, 'v', '#00695c'),
           'Dekel+23': (0.040, 0.04, 'P', '#b71c1c')}
    for lb, (fs, fe, m, c) in LIT.items():
        ax.plot(fs, fe, m, color=c, ms=13, mec='k', mew=1.2, zorder=10)
    ax.text(0.036, 0.090, r'$\tau_e$ ridge', fontsize=9, color='#2e7d32',
            rotation=-42, alpha=0.6, style='italic')
    plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02,
                 label=r'$\Delta\chi^2_{\rm prof}$ (UVLF + $\tau_e$ + $\bar{x}_{\rm HI}$)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.008, 0.055); ax.set_ylim(0.034, 0.26)
    ax.set_xlabel(r'$f_{\star,0}$'); ax.set_ylabel(r'$f_{\rm esc}$')
    ax2 = ax.twiny(); ax2.set_xscale('log'); ax2.set_xlim(ax.get_xlim())
    et = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    ax2.set_xticks(et); ax2.set_xticklabels([f'{e*100:.0f}%' for e in et])
    ax2.set_xlabel(r'$\varepsilon$ (baryon conversion efficiency)', fontsize=12, labelpad=8)
    fig.savefig('figures/fig1_ridge.pdf', bbox_inches='tight')
    print('Saved figures/fig1_ridge.pdf')
    plt.close()


def plot_fig3():
    """Fig 3: f_esc(z) reconstruction."""
    r_d = reconstruct_fesc(FSTAR_DONNAN)
    r_p = reconstruct_fesc(FSTAR_PRADA)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a)
    ax1.errorbar(r_d['z'], r_d['fesc'], yerr=[r_d['fesc_lo'], r_d['fesc_hi']],
                 fmt='D-', color='#1565C0', ms=8, capsize=4, lw=2, label='Constant $f_\\star$')
    ax1.errorbar(r_p['z'], r_p['fesc'], yerr=[r_p['fesc_lo'], r_p['fesc_hi']],
                 fmt='s-', color='#7b1fa2', ms=8, capsize=4, lw=2, label='Evolving $f_\\star$')
    ax1.axhspan(0.10, 0.16, alpha=0.15, color='#4caf50', label='Robertson+15')
    # Low-z points
    for z, fe, err in [(0.3, 0.10, 0.04), (3.0, 0.09, 0.03), (3.0, 0.05, 0.02), (3.5, 0.07, 0.03)]:
        ax1.errorbar(z, fe, yerr=err, fmt='o', color='gray', ms=6, capsize=3, alpha=0.6)
    ax1.set_xlabel('Redshift $z$'); ax1.set_ylabel(r'$f_{\rm esc}$')
    ax1.set_xlim(-0.5, 13); ax1.set_ylim(0, 0.30)
    ax1.legend(fontsize=9)
    ax1.set_title(r'(a) $f_{\rm esc}(z)$ reconstruction')

    # Panel (b)
    for r, c, m, lab in [(r_d, '#1565C0', 'D', 'Donnan+24'), (r_p, '#7b1fa2', 's', 'Prada+26')]:
        fs = FSTAR_DONNAN['fstar'] if 'Donnan' in r['label'] else FSTAR_PRADA['fstar']
        ax2.plot(fs, r['fesc'], f'{m}-', color=c, ms=8, lw=2, label=lab + ' path')
        for i in range(len(r['z'])):
            ax2.annotate(f"z={r['z'][i]:.0f}", (fs[i], r['fesc'][i]),
                        fontsize=7, ha='left', va='bottom', color=c)
    fp = np.geomspace(0.008, 0.055, 200)
    valid = (PRODUCT / fp > 0.034) & (PRODUCT / fp < 0.35)
    ax2.plot(fp[valid], PRODUCT / fp[valid], ':', color='#2e7d32', lw=1.5, alpha=0.5)
    ax2.axvspan(0.035, 0.06, alpha=0.08, color='red')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel(r'$f_{\star,0}(z)$'); ax2.set_ylabel(r'$f_{\rm esc}(z)$')
    ax2.legend(fontsize=9)
    ax2.set_title(r'(b) JWST $f_\star(z)$ traces a path along the ridge')

    fig.savefig('figures/fig3_reconstruction.pdf', bbox_inches='tight')
    print('Saved figures/fig3_reconstruction.pdf')
    plt.close()


if __name__ == '__main__':
    import os
    if os.path.exists('data/joint_profile_2d.npz'):
        plot_fig1()
    else:
        print('Run "python -m src.profile_likelihood 2d" first to generate 2D scan data.')
    plot_fig3()
