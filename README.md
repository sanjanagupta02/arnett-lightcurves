# arnett-lightcurves

One-zone Arnett (1982) bolometric lightcurve model for supernovae.
Includes an interactive HTML explorer, a statistical Python fitter with MCMC, a
matplotlib slider tool, and a Jupyter notebook with a full derivation walkthrough.

The complete mathematical derivation is in `derivation.pdf`.
Explore the webtool at [https://sanjanagupta02.github.io/arnett-lightcurves/explorer.html](https://sanjanagupta02.github.io/arnett-lightcurves/explorer.html)

---

## Requirements

```
numpy >= 1.21
scipy >= 1.7
matplotlib >= 3.4
emcee >= 3.0        # required for MCMC; falls back to curve_fit only if absent
corner >= 2.0       # required for corner plot; skipped with a warning if absent
```

Install all at once:

```bash
pip install numpy scipy matplotlib emcee corner
```

The HTML explorer runs entirely in the browser with no Python. It uses
[Chart.js 4.4.1](https://www.chartjs.org/), loaded from a CDN.

The notebook additionally requires `jupyter` or `jupyterlab`.

---

## Files

| File | Description |
|------|-------------|
| `explorer.html` | Browser tool: 8 sliders, data upload, auto-fit button, live chi-squared card |
| `arnett_fit.py` | Statistical fitter: `curve_fit` seed + emcee MCMC, corner plot, residuals panel |
| `sn_lightcurve_arnett.ipynb` | Jupyter notebook: derivation walkthrough, parameter study, Arnett rule verification |
| `derivation.pdf` | Full mathematical derivation |
| `example_lc.csv` | 21-point bolometric lightcurve of SN 2011fe (Type Ia, Pereira et al. 2013), days 5.8–40.8 |

---

## Quick start

```bash
# Full MCMC fit on the included SN 2011fe example (produces fit figure + corner plot)
# R0=0, E0=0: compact WD progenitor with no extended envelope
python arnett_fit.py example_lc.csv --v-kms 10000 --kappa 0.10 --R0-Rsun 0 --E0 0

# Fast curve_fit only (no MCMC)
python arnett_fit.py example_lc.csv --v-kms 10000 --kappa 0.10 --R0-Rsun 0 --E0 0 --no-mcmc

# HTML explorer: open in any browser, no installation needed
open explorer.html
```

---

## Fitting workflow

### Step 1: visual exploration in the HTML explorer

Open `explorer.html` in a browser. Upload your bolometric lightcurve with the
**Upload data** button. The file should have two or three columns:

```
time_days   luminosity   [luminosity_error]
```

Lines starting with `#` are skipped. Comma, tab, or whitespace delimited.
Luminosity can be in $L_\odot$ or erg s$^{-1}$ (toggle the units dropdown
before uploading). Once data is loaded, a live **chi-squared/dof** card appears
and updates as you move the sliders, giving immediate feedback on fit quality.

Adjust the sliders in roughly this order:

1. **Shape:** start with $M_\mathrm{ej}$ and $v_\mathrm{ej}$. Together they set
   $t_m \propto \sqrt{M_\mathrm{ej}/v_\mathrm{ej}}$, which controls the peak
   width and post-peak decline rate.

2. **Amplitude:** adjust $M_\mathrm{Ni}$. This scales $L$ almost linearly at peak
   without strongly shifting the peak time.

3. **Epoch alignment ($\Delta t$):** shift the model to align the model peak with
   the data peak, accounting for explosion date uncertainty.

4. **Overall normalisation ($\Delta\log L$):** moves the model up or down in log
   space by a constant factor. Useful for distance modulus or bolometric correction
   offsets. Note this shifts $L(t)$ only, not $\dot{Q}(t)$.

5. **Fine-tune:** revisit $\kappa$, $R_0$, and $E_0$ to improve the early-time
   behaviour. $R_0$ controls $t_0 = R_0/v$ (most relevant for extended RSG
   progenitors); $E_0$ adds shock-deposited thermal energy to the early rise.

Once the visual fit looks reasonable, click **Fit data** to run an automatic
Nelder-Mead optimisation over $M_\mathrm{Ni}$, $M_\mathrm{ej}$, and $\Delta t$
(with $\kappa$, $v_\mathrm{ej}$, $R_0$, $E_0$ held fixed from the sliders). The
sliders snap to the best-fit values and the status bar reports the result with
chi-squared/dof.

### Step 2: statistical fit with `arnett_fit.py`

`arnett_fit.py` is the main fitting tool and what you should use to get more accurate results.
It is more accurate than the HTML tool because it uses a finer integration
grid (N=600 vs N=400) and a more precise optimiser (Levenberg-Marquardt via
`curve_fit`, seeded into emcee MCMC).

**Free parameters (fitted):** $M_\mathrm{Ni}$ (M$_\odot$), $t_m$ (days),
and $\ln\sigma$ (log of an additive noise floor in the same units as the data).
Sampling $\ln\sigma$ as a nuisance parameter correctly inflates uncertainties when
the data scatter exceeds the formal error bars.

**Fixed parameters (CLI flags):** $v_\mathrm{ej}$, $\kappa$, $R_0$, $E_0$.

**Derived:** $M_\mathrm{ej} = 4\pi v c\, t_m^2 / 3\kappa$ from posterior samples of $t_m$.

**Log-likelihood:**

$$\ln\mathcal{L} = -\frac{1}{2}\sum_i \left[\frac{(L_{\rm obs,i} - L_{\rm mod,i})^2}{s_i^2} + \ln(2\pi s_i^2)\right], \qquad s_i^2 = \sigma_{{\rm data},i}^2 + \sigma_{\rm floor}^2$$

If no error column is provided, $\sigma_{{\rm data},i} = 0$ and $\sigma_{\rm floor}$
absorbs all scatter.

```bash
# Type Ia (compact WD progenitor -- R0 and E0 both zero):
python arnett_fit.py example_lc.csv \
    --v-kms   10000 \
    --kappa   0.10  \
    --R0-Rsun 0     \
    --E0      0

# Type IIb/Ib/Ic (extended or stripped progenitor -- adjust R0 and E0):
python arnett_fit.py data.csv \
    --v-kms   5500 \
    --kappa   0.07 \
    --R0-Rsun 200  \
    --E0      1.0
```

The fitter always runs `curve_fit` first to find a good starting point, then
initialises 32 emcee walkers by scattering around that solution. Burn-in is
500 steps; production is 3000 steps. Autocorrelation times are printed to
verify chain convergence.

**Output figures:**
- `arnett_fit.png` -- two-panel figure: log $L(t)$ with shaded 1-sigma posterior
  credible band (top), and weighted residuals $(L_{\rm obs} - L_{\rm mod})/\sigma_{\rm eff}$
  vs time (bottom). If no error bars are provided, the residual panel shows raw
  $(L_{\rm obs} - L_{\rm mod})$ in $L_\odot$ without any assumed noise floor.
- `arnett_corner.png` -- corner plot of the joint posterior of $M_\mathrm{Ni}$,
  $t_m$, and derived $M_\mathrm{ej}$ with 16/50/84th percentile markers.

**Skip MCMC for a quick result:**

```bash
python arnett_fit.py data.csv --v-kms 5500 --kappa 0.10 --no-mcmc
```

This runs `curve_fit` only and produces the same two-panel figure without the
posterior credible band.

**What to look for in the residuals.** A good fit has residuals scattered randomly
around zero. A U-shaped pattern means $t_m$ is underestimated (try larger $M_\mathrm{ej}$
or lower $v_\mathrm{ej}$). A tilt on the decline suggests the opacity or velocity
is off. Systematic early-time residuals often point to $E_0$ or $R_0$.

### Step 3: interpret the corner plot

The corner plot shows the joint posterior of $M_\mathrm{Ni}$, $t_m$, and derived $M_\mathrm{ej}$.
Check for:
- **Gaussian marginals** on the diagonal -- a sign the posterior is well-behaved and
  the quoted 16/84th percentile uncertainties are meaningful.
- **Positive $M_\mathrm{Ni}$--$t_m$ correlation** -- expected because a larger $t_m$
  shifts the model peak later, where the heating rate per unit mass is lower (more
  nickel has decayed), so a higher $M_\mathrm{Ni}$ is needed to match the observed
  peak luminosity.
- **Parabolic $M_\mathrm{ej}$--$t_m$ panel** -- $M_\mathrm{ej} \propto t_m^2$ exactly
  (given fixed $\kappa$ and $v_\mathrm{ej}$), so this panel traces a deterministic
  curve rather than a 2D posterior distribution.
- **Long tails** -- can indicate the data do not strongly constrain a parameter, or
  that the chain has not fully converged (check the printed autocorrelation time).

---

## Caveats and applicability

**Early shock cooling phase.** The $E_0$ term is a rough proxy for shock-deposited
energy, not a proper shock-cooling calculation. For observations within a few days
of explosion, use a dedicated shock-cooling model or exclude the earliest points.

**The one-zone approximation.** The model assumes well-mixed ejecta and a single
diffusion timescale. Real SNe have density and composition gradients. The Arnett
fit gives useful global averages for $M_\mathrm{Ni}$ and $M_\mathrm{ej}$, but
sub-parameter-level structure should not be over-interpreted.

**Bolometric luminosity required.** The model operates on the bolometric luminosity.
Fitting raw filter photometry without a bolometric correction will produce biased
results.

---

## Example data

`example_lc.csv` contains the bolometric lightcurve of **SN 2011fe** (Type Ia,
M101, 6.4 Mpc) from Table 2 of Pereira et al. 2013, A&A 554, A27
(first 4 points dropped due to early surface-Ni excess not captured by the one-zone model).

---

## References

Arnett, W.D. 1982, ApJ, 253, 785.
Nadyozhin, D.K. 1994, ApJS, 92, 527.
Pereira, R. et al. 2013, A&A, 554, A27.
Cano, Z. 2013, MNRAS, 434, 1098.
Lyman, J.D. et al. 2016, MNRAS, 457, 328.
Prentice, S.J. et al. 2019, MNRAS, 485, 1559.
