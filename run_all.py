#!/usr/bin/env python3
"""
Master script: reproduce all results in Wang & Shan (2026).
Run from the repo root: python run_all.py
"""
import os
import sys

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print('=' * 60)
print('Wang & Shan (2026) — Reionization f_esc Reconstruction')
print('=' * 60)

# 1. Degeneracy verification
print('\n[1/5] Verifying degeneracy ridge...')
from src.reionization import solve_reionization, PRODUCT
for fs, fe in [(0.012, 0.20), (0.019, 0.13), (0.025, 0.10), (0.035, 0.07), (0.049, 0.05)]:
    tau, xf = solve_reionization(fe, fs)
    print(f'  f*={fs:.3f}, fesc={fe:.2f}: tau={tau:.4f}, xHI(7)={float(xf(7)):.3f}')

# 2. 1D profile + scatter model test
print('\n[2/5] Running 1D profile likelihood...')
from src.profile_likelihood import run_1d_scan, run_scatter_test
run_1d_scan()

print('\n[3/5] Testing scatter models at crisis point...')
run_scatter_test()

# 3. f_esc reconstruction
print('\n[4/5] Running f_esc reconstruction...')
from src.fesc_reconstruction import run_reconstruction
run_reconstruction()

# 4. Beta scan
print('\n[5/5] Running beta scan...')
from src.beta_scan import run_beta_scan
run_beta_scan()

print('\n' + '=' * 60)
print('All computations complete.')
print('To generate figures, run: python plot_figures.py')
print('To run the full 2D scan (~2 min): python -m src.profile_likelihood 2d')
print('=' * 60)
