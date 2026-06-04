"""
arnett_fit.py
=============
Fit the Arnett one-zone model to a supernova bolometric lightcurve.

Physics
-------
Energy equation (Arnett 1982):

    dE/dt = -E/(t+t0) - E*(t+t0)/tm^2 + Q(t)

where  t0 = R0/v,  tm = sqrt(3*kappa*M_ej / 4*pi*v*c),
and    Q(t) = M_Ni * [eps_Ni*exp(-t/tau_Ni) + eps_Co*(exp(-t/tau_Co) - exp(-t/tau_Ni))]

Exact integral solution (numerically stable O(N^2) form -- see derivation.pdf):

    L(t) = (1/tm^2) * [ t0*E0*exp((t0^2 - (t+t0)^2)/2tm^2)
             + integral_0^t (t'+t0)*exp(((t'+t0)^2-(t+t0)^2)/2tm^2)*Q(t')dt' ]

Fitting strategy
----------------
Primary fitter: emcee MCMC with 3 free parameters:
  - M_Ni  (Msun)    nickel mass
  - t_m   (days)    Arnett diffusion timescale
  - ln_sigma        log of additive noise floor (erg/s)

Fixed parameters (CLI): v_ej, kappa, R0, E0.
curve_fit is always run first to seed the emcee walkers.
If emcee is not installed, falls back to curve_fit only.

Derived: M_ej = tm^2 * 4*pi*v*c / (3*kappa)

Usage
-----
    python arnett_fit.py data.csv --v-kms 5500 --kappa 0.10 --R0-Rsun 600 --E0 2.0
    python arnett_fit.py data.csv --no-mcmc    # curve_fit only, faster
"""

import argparse
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset' : 'stix',
    'axes.labelsize'   : 13,
    'axes.titlesize'   : 12,
    'xtick.labelsize'  : 11,
    'ytick.labelsize'  : 11,
    'legend.fontsize'  : 11,
    'figure.dpi'       : 130,
})
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False

# ── Physical constants (CGS) ──────────────────────────────────────────────────
CLIGHT = 3.0e10
MSUN   = 1.989e33
RSUN   = 6.957e10
LSUN   = 3.828e33
DAY    = 86400.0

# Ni-56 / Co-56 decay chain (Nadyozhin 1994)
TAU_NI = 8.8   * DAY
TAU_CO = 111.3 * DAY
EPS_NI = 3.90e10
EPS_CO = 6.78e9

# Palette
C_MODEL   = '#1E90FF'  # dodgerblue  -- L(t)
C_DATA    = '#6B8E23'  # olivedrab   -- data
C_HEATING = '#FF4500'  # orangered   -- Q-dot
C_PEAK    = '#800000'  # maroon      -- t_peak
C_CI      = '#1E90FF'  # same as model for shaded band


# ── Physics ───────────────────────────────────────────────────────────────────

def heating_rate(t_s, Mni_g):
    """Radioactive heating rate (erg/s) from Ni56 -> Co56 -> Fe56."""
    Ni_term = EPS_NI * np.exp(-t_s / TAU_NI)
    Co_term = EPS_CO * (np.exp(-t_s / TAU_CO) - np.exp(-t_s / TAU_NI))
    return Mni_g * (Ni_term + Co_term)


def arnett_luminosity(t_s_arr, Mni_g, tm_s, t0_s, E0_erg):
    """
    Bolometric luminosity from the Arnett integral (numerically stable O(N^2)).

    Every exponent in the kernel is bounded <= 0 so exp() never overflows.
    Do not replace this with the cumulative_trapezoid form.

    Parameters
    ----------
    t_s_arr : ndarray  Evaluation times in seconds (uniform grid).
    Mni_g   : float    Nickel mass in grams.
    tm_s    : float    Arnett diffusion timescale in seconds.
    t0_s    : float    Initial light-crossing time R0/v in seconds.
    E0_erg  : float    Shock-deposited thermal energy in erg.

    Returns
    -------
    L_cgs : ndarray  Bolometric luminosity in erg/s.
    """
    N    = len(t_s_arr)
    tm2  = tm_s ** 2
    dt   = t_s_arr[1] - t_s_arr[0]
    Q    = heating_rate(t_s_arr, Mni_g)

    t_t0    = t_s_arr + t0_s
    half_sq = t_t0 ** 2 / (2.0 * tm2)

    # diff[i,j] = half_sq[j] - half_sq[i] <= 0 for j <= i
    diff    = half_sq[np.newaxis, :] - half_sq[:, np.newaxis]
    exp_mat = np.exp(np.minimum(diff, 0.0))

    intgd = t_t0[np.newaxis, :] * exp_mat * Q[np.newaxis, :]
    intgd = np.tril(intgd)

    I   = dt * (intgd.sum(axis=1) - 0.5 * intgd[:, 0] - 0.5 * np.diag(intgd))
    E0c = t0_s * E0_erg * np.exp(np.minimum(t0_s ** 2 / (2.0 * tm2) - half_sq, 0.0))

    return np.maximum((E0c + I) / tm2, 0.0)


def tm_to_mej(tm_s, v_cms, kappa):
    """Derive M_ej (grams) from the Arnett timescale."""
    return tm_s ** 2 * 4.0 * np.pi * v_cms * CLIGHT / (3.0 * kappa)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path, units='lsun'):
    """
    Load a 2- or 3-column lightcurve file.

    Columns: time_days  luminosity  [luminosity_error]
    Delimiter: comma, tab, or whitespace.  Lines starting with # are skipped.
    Returns t_days, L_cgs, Lerr_cgs (all in CGS units internally).
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in ('#', '%', '!'):
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) < 2:
                continue
            try:
                t   = float(parts[0])
                L   = float(parts[1])
                err = float(parts[2]) if len(parts) >= 3 else 0.0
            except ValueError:
                continue
            if not (np.isfinite(t) and np.isfinite(L) and L > 0):
                continue
            rows.append((t, L, abs(err)))

    if not rows:
        sys.exit(f'[arnett_fit] No valid data found in {path}')

    t_d, L, Lerr = np.array(rows).T
    scale = LSUN if units == 'lsun' else 1.0
    return t_d, L * scale, Lerr * scale


# ── curve_fit wrapper ─────────────────────────────────────────────────────────

def eval_model_at_data(t_data_days, Mni_sun, tm_d, t0_s, E0_erg, npts=600):
    """
    Evaluate the Arnett model on a fine uniform grid and interpolate to data times.

    arnett_luminosity requires a uniform time grid (uses a single dt for trapezoid
    integration).  Passing non-uniform data points directly gives wrong integrals.
    This wrapper always builds its own grid and interpolates.
    """
    t_max = max(t_data_days.max() * 1.3, 50.0)
    t_grid = np.linspace(0.01, t_max, npts)
    L_grid = arnett_luminosity(t_grid * DAY, Mni_sun * MSUN, tm_d * DAY, t0_s, E0_erg)
    return np.interp(t_data_days, t_grid, L_grid)


def arnett_initial_guess(t_days, L_cgs):
    """
    Estimate starting parameters from the data using Arnett's rule.

    Arnett's rule: L(t_peak) ≈ Q(t_peak).
    - p0_tm  : t_peak from the data (t_m ≈ t_peak for compact progenitors)
    - p0_Mni : L_peak / Q_unit(t_peak), where Q_unit is the heating rate per gram
    """
    i_pk   = np.argmax(L_cgs)
    t_pk_s = t_days[i_pk] * DAY
    L_pk   = L_cgs[i_pk]

    q_unit = (EPS_NI * np.exp(-t_pk_s / TAU_NI)
              + EPS_CO * (np.exp(-t_pk_s / TAU_CO) - np.exp(-t_pk_s / TAU_NI)))
    p0_Mni = np.clip(L_pk / (q_unit * MSUN), 0.001, 2.0)
    p0_tm  = max(t_days[i_pk], 2.0)   # t_m ≈ t_peak (exact for t0=0, good approx otherwise)
    return p0_Mni, p0_tm


def run_curve_fit(t_days, L_cgs, Lerr_cgs, t0_s, E0_erg, p0_Mni=None, p0_tm=None):
    """
    Least-squares fit using scipy.optimize.curve_fit.

    Returns popt = [Mni_Msun, tm_days], perr = 1-sigma uncertainties.
    """
    # Auto-estimate starting guesses from data if not supplied
    _p0_Mni, _p0_tm = arnett_initial_guess(t_days, L_cgs)
    if p0_Mni is None:
        p0_Mni = _p0_Mni
    if p0_tm is None:
        p0_tm = _p0_tm

    sigma = Lerr_cgs.copy()
    sigma[sigma <= 0] = np.median(L_cgs) * 0.1   # 10% floor where no errors given

    def model(t_d, Mni_sun, tm_d):
        return eval_model_at_data(t_d, Mni_sun, tm_d, t0_s, E0_erg)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        popt, pcov = curve_fit(
            model, t_days, L_cgs,
            p0     = [p0_Mni, p0_tm],
            sigma  = sigma,
            absolute_sigma = True,
            bounds = ([0.001, 1.0], [2.0, 500.0]),
            maxfev = 20000,
        )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


# ── emcee MCMC ───────────────────────────────────────────────────────────────

def make_log_prob(t_days, L_cgs, Lerr_cgs, t0_s, E0_erg):
    """
    Return the log-posterior function for emcee.

    Free params: theta = [Mni_Msun, tm_days, ln_sigma]
    ln_sigma = ln(sigma_floor [erg/s]) -- additive noise floor absorbing
    scatter not captured by the formal error bars.

    The ln_sigma prior window is derived from the data median so no absolute
    CGS values are hardcoded.  Sigma is allowed to range from 1e-4 to 1e4
    times the data median luminosity, which is physically unconstrained.
    The Mni and tm bounds are astrophysically motivated hard limits.
    """
    L_scale = np.median(L_cgs)
    # Prior on ln_sigma: sigma in [1e-4, 1e4] * L_scale -- fully data-driven
    ln_sig_lo = np.log(L_scale) - 4.0 * np.log(10.0)   # 1e-4 * L_scale
    ln_sig_hi = np.log(L_scale) + 4.0 * np.log(10.0)   # 1e+4 * L_scale

    def log_prob(theta):
        Mni_sun, tm_d, ln_sig = theta
        # Astrophysically motivated bounds (no absolute CGS values here)
        if not (0.001 <= Mni_sun <= 2.0 and 1.0 <= tm_d <= 500.0
                and ln_sig_lo <= ln_sig <= ln_sig_hi):
            return -np.inf
        sigma_floor = np.exp(ln_sig)
        s2 = Lerr_cgs ** 2 + sigma_floor ** 2
        L_mod = eval_model_at_data(t_days, Mni_sun, tm_d, t0_s, E0_erg)
        if np.any(~np.isfinite(L_mod)):
            return -np.inf
        resid = L_cgs - L_mod
        return -0.5 * np.sum(resid ** 2 / s2 + np.log(2.0 * np.pi * s2))

    return log_prob


def run_mcmc(t_days, L_cgs, Lerr_cgs, t0_s, E0_erg, cf_popt, cf_perr,
             nwalkers=32, nburn=500, nprod=3000):
    """
    Run emcee MCMC seeded from the curve_fit solution.

    Returns the flattened production chain, shape (nwalkers*nprod, 3).
    Columns: [Mni_Msun, tm_days, ln_sigma_frac].
    ln_sigma_frac = ln(sigma_floor / L_ref) where L_ref = median(L_cgs).
    """
    log_prob = make_log_prob(t_days, L_cgs, Lerr_cgs, t0_s, E0_erg)

    Mni0, tm0 = cf_popt
    # Seed ln_sigma from the data errors (CGS, no normalization needed)
    sigma0 = (np.median(Lerr_cgs[Lerr_cgs > 0]) if np.any(Lerr_cgs > 0)
              else np.std(L_cgs) * 0.1)
    ln_sig0 = np.log(sigma0)

    L_scale = np.median(L_cgs)
    ln_sig_lo = np.log(L_scale) - 4.0 * np.log(10.0)
    ln_sig_hi = np.log(L_scale) + 4.0 * np.log(10.0)

    # Scatter walkers around the curve_fit point
    rng = np.random.default_rng(42)
    p0 = np.column_stack([
        rng.normal(Mni0,    max(cf_perr[0], Mni0 * 0.05), nwalkers),
        rng.normal(tm0,     max(cf_perr[1], tm0  * 0.05), nwalkers),
        rng.normal(ln_sig0, 1.0, nwalkers),
    ])
    # Clip to prior bounds
    p0[:, 0] = np.clip(p0[:, 0], 0.001, 2.0)
    p0[:, 1] = np.clip(p0[:, 1], 1.0,   500.0)
    p0[:, 2] = np.clip(p0[:, 2], ln_sig_lo, ln_sig_hi)

    sampler = emcee.EnsembleSampler(nwalkers, 3, log_prob)

    print(f'\n  Running emcee burn-in  ({nburn} steps x {nwalkers} walkers)...')
    sampler.run_mcmc(p0, nburn, progress=True)
    sampler.reset()

    print(f'  Running emcee production  ({nprod} steps x {nwalkers} walkers)...')
    sampler.run_mcmc(None, nprod, progress=True)

    # Autocorrelation diagnostics
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        if np.any(~np.isfinite(tau)):
            print('\n  Autocorrelation times: could not be estimated (chain may not have converged).')
            print('  Try increasing --nprod or check that the model fits the data well.')
        else:
            print(f'\n  Autocorrelation times:  Mni = {tau[0]:.1f} steps,'
                  f'  tm = {tau[1]:.1f} steps,  ln_sigma = {tau[2]:.1f} steps')
            if np.any(tau * 50 > nprod):
                print('  WARNING: chain may be too short relative to autocorrelation time.')
                print(f'  Recommend at least {int(np.max(tau) * 50)} production steps.')
    except emcee.autocorr.AutocorrError:
        print('  Autocorrelation time could not be estimated (chain may be too short).')

    flat_chain = sampler.get_chain(flat=True)   # (nwalkers*nprod, 3)
    return flat_chain


# ── Summary statistics ────────────────────────────────────────────────────────

def chain_summary(flat_chain, v_cms, kappa):
    """
    Compute median and 16/84th percentiles for Mni, tm, and derived Mej.

    Returns a dict with keys: Mni, tm, Mej -- each a (med, lo, hi) tuple.
    """
    Mni_chain = flat_chain[:, 0]
    tm_chain  = flat_chain[:, 1]
    Mej_chain = tm_to_mej(tm_chain * DAY, v_cms, kappa) / MSUN

    def pct(arr):
        lo, med, hi = np.percentile(arr, [16, 50, 84])
        return med, med - lo, hi - med   # (median, -1sigma, +1sigma)

    return {
        'Mni': pct(Mni_chain),
        'tm' : pct(tm_chain),
        'Mej': pct(Mej_chain),
    }


def print_results(label, summary):
    print(f'\n  {label}')
    print('  ' + '-' * 52)
    Mni = summary['Mni']
    tm  = summary['tm']
    Mej = summary['Mej']
    print(f'  M_Ni  = {Mni[0]:.4f}  -{Mni[1]:.4f}  +{Mni[2]:.4f}   Msun')
    print(f'  t_m   = {tm[0]:.2f}   -{tm[1]:.2f}   +{tm[2]:.2f}    days')
    print(f'  M_ej  = {Mej[0]:.2f}   -{Mej[1]:.2f}   +{Mej[2]:.2f}    Msun  (derived)')


# ── Model evaluation on dense grid ───────────────────────────────────────────

def eval_model_grid(Mni_sun, tm_d, t0_s, E0_erg, t_max_d, npts=600):
    t_plot = np.linspace(0.01, t_max_d, npts)
    L_mod  = arnett_luminosity(t_plot * DAY, Mni_sun * MSUN, tm_d * DAY, t0_s, E0_erg)
    Q_mod  = heating_rate(t_plot * DAY, Mni_sun * MSUN)
    cross  = np.where(np.diff(np.sign(L_mod - Q_mod)))[0]
    t_peak = t_plot[cross[0]] if len(cross) else t_plot[np.argmax(L_mod)]
    return t_plot, L_mod, Q_mod, t_peak


# ── Plot 1: fit + residuals ───────────────────────────────────────────────────

def plot_fit_residuals(t_days, L_cgs, Lerr_cgs,
                       flat_chain, t0_s, E0_erg, v_kms, kappa, Mej_sun,
                       save_path='arnett_fit.png'):
    """
    Two-panel figure: top = log L vs time with credible band,
    bottom = weighted residuals.
    """
    # Posterior medians for the median model curve
    Mni_med = np.median(flat_chain[:, 0])
    tm_med  = np.median(flat_chain[:, 1])
    sig_med = np.exp(np.median(flat_chain[:, 2]))  # CGS erg/s, no renormalization needed

    t_max = max(t_days) * 1.3
    t_plot, L_med, Q_med, t_peak = eval_model_grid(Mni_med, tm_med, t0_s, E0_erg, t_max)

    # 1-sigma credible band: evaluate model at 300 random posterior draws
    rng   = np.random.default_rng(0)
    idx   = rng.choice(len(flat_chain), size=min(300, len(flat_chain)), replace=False)
    L_samp = np.array([
        arnett_luminosity(t_plot * DAY, flat_chain[i, 0] * MSUN,
                          flat_chain[i, 1] * DAY, t0_s, E0_erg)
        for i in idx
    ])
    L_lo = np.percentile(L_samp, 16, axis=0)
    L_hi = np.percentile(L_samp, 84, axis=0)

    # Effective sigma per data point for residuals
    L_data_med = eval_model_at_data(t_days, Mni_med, tm_med, t0_s, E0_erg)
    has_err = np.any(Lerr_cgs > 0)
    s_eff = np.sqrt(Lerr_cgs ** 2 + sig_med ** 2)
    weighted_resid = (L_cgs - L_data_med) / s_eff

    fig, axes = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08},
        sharex=True, constrained_layout=True
    )
    ax, ax_r = axes

    # ── Top panel ──
    # Credible band
    ax.fill_between(t_plot, L_lo / LSUN, L_hi / LSUN,
                    color=C_CI, alpha=0.18, label='1σ posterior')
    # Median model
    ax.plot(t_plot, L_med / LSUN, color=C_MODEL, lw=2.0, label='Median model  L(t)')
    # Heating rate
    ax.plot(t_plot, Q_med / LSUN, color=C_HEATING, lw=1.4, ls='--',
            alpha=0.6, label=r'$\dot{Q}(t)$')
    # Data
    has_err = np.any(Lerr_cgs > 0)
    if has_err:
        ax.errorbar(t_days, L_cgs / LSUN, yerr=Lerr_cgs / LSUN,
                    fmt='o', color=C_DATA, ms=5, elinewidth=1.2,
                    capsize=3, label='Data', zorder=4)
    else:
        ax.scatter(t_days, L_cgs / LSUN, color=C_DATA, s=22,
                   zorder=4, label='Data')
    # Peak line
    ax.axvline(t_peak, color=C_PEAK, lw=1.2, ls=':',
               label=f'$t_{{\\rm peak}}$ = {t_peak:.0f} d')

    ax.set_yscale('log')
    ax.set_ylabel(r'$L\ (L_\odot)$')
    # Y-limits: data range with a comfortable margin (avoid wasted decades)
    L_data_lsun = L_cgs / LSUN
    ylo = L_data_lsun.min() / 3.0
    yhi = L_data_lsun.max() * 5.0
    ax.set_ylim(ylo, yhi)
    smry = chain_summary(flat_chain, v_kms * 1e5, kappa)
    Mni = smry['Mni']
    tm  = smry['tm']
    Mej = smry['Mej']
    ax.set_title(
        fr'$M_{{\rm Ni}} = {Mni[0]:.3f}^{{+{Mni[2]:.3f}}}_{{-{Mni[1]:.3f}}}\,M_\odot$'
        fr'  |  $t_m = {tm[0]:.1f}^{{+{tm[2]:.1f}}}_{{-{tm[1]:.1f}}}$ d'
        fr'  |  $M_{{\rm ej}} = {Mej[0]:.1f}^{{+{Mej[2]:.1f}}}_{{-{Mej[1]:.1f}}}\,M_\odot$',
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

    # ── Bottom panel ──
    ax_r.scatter(t_days, weighted_resid, color=C_DATA, s=20, zorder=3)
    if has_err:
        ax_r.errorbar(t_days, weighted_resid,
                      yerr=np.ones_like(t_days),
                      fmt='none', color=C_DATA, elinewidth=0.8, capsize=2, alpha=0.5)
    ax_r.axhline(0, color='#888', lw=0.8, ls='--')
    ax_r.set_xlabel('Time since explosion  (days)')
    ax_r.set_ylabel(r'$(L_{\rm obs} - L_{\rm mod})\,/\,\sigma_{\rm eff}$')
    ax_r.grid(True, alpha=0.15)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n  Fit figure saved to  {save_path}')
    plt.show()


# ── Plot 2: corner plot ───────────────────────────────────────────────────────

def plot_corner(flat_chain, v_cms, kappa, save_path='arnett_corner.png'):
    """
    Corner plot of Mni, tm, and derived Mej posterior.
    """
    if not HAS_CORNER:
        print('  [corner] package not installed -- skipping corner plot.')
        print('  Install with:  pip install corner')
        return

    Mni_chain = flat_chain[:, 0]
    tm_chain  = flat_chain[:, 1]
    Mej_chain = tm_to_mej(tm_chain * DAY, v_cms, kappa) / MSUN

    samples = np.column_stack([Mni_chain, tm_chain, Mej_chain])
    labels  = [r'$M_{\rm Ni}\ (M_\odot)$',
               r'$t_m\ (\rm days)$',
               r'$M_{\rm ej}\ (M_\odot)$']

    quantiles = [0.16, 0.50, 0.84]
    fig = corner.corner(
        samples,
        labels      = labels,
        quantiles   = quantiles,
        show_titles = True,
        title_fmt   = '.3f',
        color       = C_MODEL,
        title_kwargs= {'fontsize': 11},
        label_kwargs= {'fontsize': 12},
    )
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  Corner plot saved to  {save_path}')
    plt.show()


# ── curve_fit-only plot (no MCMC) ─────────────────────────────────────────────

def plot_curvefit_only(t_days, L_cgs, Lerr_cgs,
                       popt, perr, t0_s, E0_erg, v_kms, kappa,
                       save_path='arnett_fit.png'):
    """
    Two-panel fit + residuals using curve_fit result only (no posterior band).
    """
    Mni_sun, tm_d = popt
    has_err = np.any(Lerr_cgs > 0)

    t_max = max(t_days) * 1.3
    t_plot, L_mod, Q_mod, t_peak = eval_model_grid(Mni_sun, tm_d, t0_s, E0_erg, t_max)
    L_data = eval_model_at_data(t_days, Mni_sun, tm_d, t0_s, E0_erg)

    resid_raw = (L_cgs - L_data) / LSUN        # always in Lsun
    if has_err:
        s_eff = Lerr_cgs.copy()
        s_eff[s_eff <= 0] = np.median(L_cgs) * 0.01
        w_res = (L_cgs - L_data) / s_eff       # normalised
    else:
        w_res = resid_raw

    fig, axes = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08},
        sharex=True, constrained_layout=True
    )
    ax, ax_r = axes

    ax.plot(t_plot, L_mod / LSUN, color=C_MODEL, lw=2.0, label='Arnett fit  L(t)')
    ax.plot(t_plot, Q_mod / LSUN, color=C_HEATING, lw=1.4, ls='--',
            alpha=0.6, label=r'$\dot{Q}(t)$')
    if np.any(Lerr_cgs > 0):
        ax.errorbar(t_days, L_cgs / LSUN, yerr=Lerr_cgs / LSUN,
                    fmt='o', color=C_DATA, ms=5, elinewidth=1.2,
                    capsize=3, label='Data', zorder=4)
    else:
        ax.scatter(t_days, L_cgs / LSUN, color=C_DATA, s=22, zorder=4, label='Data')
    ax.axvline(t_peak, color=C_PEAK, lw=1.2, ls=':',
               label=f'$t_{{\\rm peak}}$ = {t_peak:.0f} d')

    Mej_sun = tm_to_mej(tm_d * DAY, v_kms * 1e5, kappa) / MSUN
    ax.set_yscale('log')
    ax.set_ylabel(r'$L\ (L_\odot)$')
    # Y-limits: data range with a comfortable margin (avoid wasted decades)
    L_data_lsun = L_cgs / LSUN
    ylo = L_data_lsun.min() / 3.0
    yhi = L_data_lsun.max() * 5.0
    ax.set_ylim(ylo, yhi)
    ax.set_title(
        fr'$M_{{\rm Ni}} = {Mni_sun:.3f} \pm {perr[0]:.3f}\,M_\odot$'
        fr'  |  $t_m = {tm_d:.1f} \pm {perr[1]:.1f}$ d'
        fr'  |  $M_{{\rm ej}} \approx {Mej_sun:.1f}\,M_\odot$  (curve_fit)',
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

    ax_r.scatter(t_days, w_res, color=C_DATA, s=20, zorder=3)
    if has_err:
        ax_r.errorbar(t_days, w_res, yerr=np.ones_like(t_days),
                      fmt='none', color=C_DATA, elinewidth=0.8, capsize=2, alpha=0.5)
    ax_r.axhline(0, color='#888', lw=0.8, ls='--')
    ax_r.set_xlabel('Time since explosion  (days)')
    res_ylabel = (r'$(L_{\rm obs} - L_{\rm mod})\,/\,\sigma_i$' if has_err
                  else r'$L_{\rm obs} - L_{\rm mod}\ (L_\odot)$')
    ax_r.set_ylabel(res_ylabel)
    ax_r.grid(True, alpha=0.15)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n  Fit figure saved to  {save_path}')
    plt.show()


# ── Command-line interface ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Fit the Arnett one-zone model to a SN bolometric lightcurve.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data_file',
                        help='Path to data file (time_days, L, [L_err])')
    parser.add_argument('--units',    default='lsun', choices=['lsun', 'ergs'],
                        help='Luminosity units in the data file')
    parser.add_argument('--v-kms',   type=float, default=5000.0,
                        help='Ejecta velocity (km/s) -- held fixed')
    parser.add_argument('--kappa',   type=float, default=0.10,
                        help='Opacity (cm^2/g) -- held fixed')
    parser.add_argument('--R0-Rsun', type=float, default=500.0,
                        help='Progenitor radius (Rsun) -- held fixed')
    parser.add_argument('--E0',      type=float, default=1.0,
                        help='Initial shock energy (units of 10^49 erg) -- held fixed')
    parser.add_argument('--p0-Mni',  type=float, default=None,
                        help='Initial guess for M_Ni (Msun); auto-estimated from data via Arnett rule if omitted')
    parser.add_argument('--p0-tm',   type=float, default=None,
                        help='Initial guess for t_m (days); auto-estimated from data peak time if omitted')
    parser.add_argument('--nwalkers', type=int, default=32,
                        help='Number of emcee walkers')
    parser.add_argument('--nburn',    type=int, default=500,
                        help='Burn-in steps per walker')
    parser.add_argument('--nprod',    type=int, default=3000,
                        help='Production steps per walker')
    parser.add_argument('--no-mcmc',  action='store_true',
                        help='Skip MCMC -- use curve_fit only (faster)')
    parser.add_argument('--output',   default='arnett_fit.png',
                        help='Output filename for the fit + residuals figure')
    parser.add_argument('--corner-output', default='arnett_corner.png',
                        help='Output filename for the corner plot')
    args = parser.parse_args()

    # ── Load ──
    print(f'\n  Loading  {args.data_file}')
    t_d, L_cgs, Lerr_cgs = load_data(args.data_file, units=args.units)
    print(f'  {len(t_d)} points  |  t in [{t_d.min():.1f}, {t_d.max():.1f}] days'
          f'  |  data peak at t = {t_d[np.argmax(L_cgs)]:.1f} days')

    v_cms  = args.v_kms * 1e5
    t0_s   = args.R0_Rsun * RSUN / v_cms
    E0_erg = args.E0 * 1e49

    print('\n  Fixed parameters:')
    print(f'    v_ej   = {args.v_kms:.0f}  km/s')
    print(f'    kappa  = {args.kappa}  cm^2/g')
    print(f'    R_0    = {args.R0_Rsun:.0f}  Rsun')
    print(f'    E_0    = {args.E0}  x 10^49 erg')
    print(f'    t_0    = {t0_s / DAY:.3f}  days')

    # ── curve_fit ──
    print('\n  Running curve_fit...')
    cf_popt, cf_perr = run_curve_fit(
        t_d, L_cgs, Lerr_cgs, t0_s, E0_erg,
        p0_Mni=args.p0_Mni, p0_tm=args.p0_tm
    )
    Mej_cf = tm_to_mej(cf_popt[1] * DAY, v_cms, args.kappa) / MSUN
    dMej_cf = Mej_cf * 2.0 * (cf_perr[1] / cf_popt[1])

    # chi2/dof for the curve_fit solution (using formal errors; floor-free)
    L_cf_mod  = eval_model_at_data(t_d, cf_popt[0], cf_popt[1], t0_s, E0_erg)
    sigma_chi2 = np.where(Lerr_cgs > 0, Lerr_cgs, np.median(L_cgs) * 0.1)
    chi2_cf   = np.sum(((L_cgs - L_cf_mod) / sigma_chi2) ** 2)
    dof_cf    = len(t_d) - 2
    chi2dof_cf = chi2_cf / dof_cf

    print('\n  curve_fit results:')
    print('  ' + '-' * 52)
    print(f'  M_Ni  = {cf_popt[0]:.4f}  +/-  {cf_perr[0]:.4f}   Msun')
    print(f'  t_m   = {cf_popt[1]:.2f}   +/-  {cf_perr[1]:.2f}    days')
    print(f'  M_ej  = {Mej_cf:.2f}   +/-  {dMej_cf:.2f}    Msun  (derived)')
    print(f'  chi2/dof = {chi2dof_cf:.2f}  ({chi2_cf:.1f} / {dof_cf})')
    if chi2dof_cf > 5.0:
        print('  NOTE: chi2/dof >> 1 -- the one-zone model may not match the data shape,')
        print('        as it is an approximate model.  Systematic residuals are expected')
        print('        for most types of supernovae.  The MCMC noise floor will absorb the scatter.')

    # ── MCMC ──
    use_mcmc = not args.no_mcmc

    if use_mcmc and not HAS_EMCEE:
        print('\n  WARNING: emcee not installed -- falling back to curve_fit only.')
        print('  Install with:  pip install emcee')
        use_mcmc = False

    if use_mcmc:
        flat_chain = run_mcmc(
            t_d, L_cgs, Lerr_cgs, t0_s, E0_erg,
            cf_popt, cf_perr,
            nwalkers = args.nwalkers,
            nburn    = args.nburn,
            nprod    = args.nprod,
        )
        smry = chain_summary(flat_chain, v_cms, args.kappa)
        print_results('MCMC posterior (median + 16/84th pct)', smry)

        plot_fit_residuals(
            t_d, L_cgs, Lerr_cgs,
            flat_chain, t0_s, E0_erg,
            args.v_kms, args.kappa, smry['Mej'][0],
            save_path=args.output
        )
        plot_corner(flat_chain, v_cms, args.kappa, save_path=args.corner_output)

    else:
        plot_curvefit_only(
            t_d, L_cgs, Lerr_cgs,
            cf_popt, cf_perr, t0_s, E0_erg,
            args.v_kms, args.kappa,
            save_path=args.output
        )


if __name__ == '__main__':
    main()
