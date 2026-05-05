# JWSThigh-z-Crisis

**The JWST Early Galaxy Crisis Resolved by a Reionization Degeneracy**

Wang & Shan (2026) — [arXiv:26XX.XXXXX]

## Summary

Published estimates of the ionizing photon escape fraction at z > 6 span a factor of four (5–21%). We show that every published value is correct: the apparent disagreement is structural, not observational. Reionization observables constrain only the product f_esc × f★, defining a degeneracy ridge along which all estimates are mutually consistent. JWST UV luminosity function measurements break this degeneracy, enabling the first empirical reconstruction of f_esc(z) at z = 7–12. The Boylan-Kolchin (2023) crisis threshold is excluded at >3σ under Gaussian, log-normal, and duty-cycle scatter models (4.5σ for duty-cycle bursts). Three independent lines of evidence disfavour evolving star formation efficiency: Thomson optical depth rejection at 2.6σ, a 47% decline in intrinsic ISM porosity contradicting RHD simulations, and convergence of two of three f★ methods on constant efficiency.

## Repository Structure

```
├── src/
│   ├── reionization.py          # Core solver: ODE, HMF, UVLF models
│   ├── data.py                  # Exact Donnan+2024 Table 2 UVLF data
│   ├── profile_likelihood.py    # Joint profile likelihood scan
│   ├── fesc_reconstruction.py   # f_esc(z) MC reconstruction
│   └── beta_scan.py             # z-dependent product constraint
├── data/                        # Output data products (generated)
├── figures/                     # Output figures (generated)
├── paper/                       # LaTeX source (not included)
├── run_all.py                   # Reproduce all results (~3 min)
├── plot_figures.py              # Generate all figures
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/wzh800557-source/JWSThigh-z-Crisis.git
cd JWSThigh-z-Crisis
pip install -r requirements.txt

# Reproduce all key numbers (~3 min)
python run_all.py

# Full 2D scan for Figure 1 (~5 min)
python -m src.profile_likelihood 2d

# Generate figures
python plot_figures.py
```

## Key Results

| Quantity | Value |
|---|---|
| Calibrated product | f_esc × f★ = 0.00247 ± 0.00032 |
| ε_max (95% CL) | 2.5% |
| BK23 Δχ² (Gaussian) | 13.4 (3.7σ) |
| BK23 Δχ² (log-normal) | 11.8 (3.4σ) |
| BK23 Δχ² (duty-cycle) | 20.2 (4.5σ) |
| f_esc(z=8, const. f★) | 0.146 +0.069/−0.045 |
| f_esc(z=12, const. f★) | 0.097 +0.052/−0.031 |
| f_esc(z=12, evol. f★) | 0.058 +0.026/−0.018 |
| f_esc(z=12, SMF f★) | 0.105 +0.081/−0.040 |
| β range (2σ) | [−0.80, +0.13] |
| τ_e (evolving SFE) | 0.036 → rejected at 2.6σ |
| f_esc,int decline (evol.) | 47% (contradicts RHD sims) |

## Data Sources

- **UVLF**: Donnan et al. (2024) MNRAS 533, 3222, Table 2 — 28 data points at z = 9, 10, 11, 12.5
- **f★ (constant)**: Donnan et al. (2024) Appendix A model
- **f★ (evolving)**: Prada et al. (2026) arXiv:2604.18683, Uchuu-UniverseMachine
- **ξ_ion**: Simmonds et al. (2023), Endsley et al. (2024)
- **τ_e**: Planck 2018 (Aghanim et al. 2020)
- **x_HI(z)**: Bosman et al. (2022), Greig et al. (2022), Umeda et al. (2024)

## Citation

```bibtex
@article{Wang:2026jwst,
    author = "Wang, Zihan and Shan, Huanyuan",
    title = "{The JWST Early Galaxy Crisis Resolved by a Reionization Degeneracy}",
    year = "2026",
    eprint = "26XX.XXXXX",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO"
}
```

## License

MIT
