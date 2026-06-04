"""
arnett_interactive.py
=====================
Interactive matplotlib lightcurve explorer with parameter sliders.
Physics imported from arnett_fit.py -- no duplication.

Usage
-----
    python arnett_interactive.py
    python arnett_interactive.py data.csv
    python arnett_interactive.py data.csv --units ergs
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.rcParams.update({
    'font.family'    : 'serif',
    'font.serif'     : ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize' : 13,
    'axes.titlesize' : 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.facecolor': '#0d0d0d',
    'axes.facecolor'  : '#111111',
    'axes.edgecolor'  : '#333333',
    'xtick.color'     : '#888888',
    'ytick.color'     : '#888888',
    'axes.labelcolor' : '#cccccc',
    'text.color'      : '#cccccc',
    'grid.color'      : '#222222',
    'grid.linewidth'  : 0.6,
})
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from arnett_fit import (
    heating_rate, arnett_luminosity, load_data,
    CLIGHT, MSUN, RSUN, LSUN, DAY,
)

COLORS = {
    'L'     : '#1E90FF',  # dodgerblue
    'Q'     : '#FF4500',  # orangered
    'data'  : '#6B8E23',  # olivedrab
    'peak'  : '#800000',  # maroon
    'slider': '#1E90FF',
}


# ── Model wrapper (always uses a uniform grid for display) ────────────────────

def compute_model(Mej_sun, kappa, Mni_sun, v_kms, R0_Rsun, E0_e49,
                  t_max_d=400, N=600):
    Mej = Mej_sun * MSUN
    Mni = Mni_sun * MSUN
    v   = v_kms   * 1e5
    R0  = R0_Rsun * RSUN
    E0  = E0_e49  * 1e49
    t0  = R0 / v
    tm  = np.sqrt(3.0 * kappa * Mej / (4.0 * np.pi * v * CLIGHT))

    t_s   = np.linspace(1e-4 * DAY, t_max_d * DAY, N)
    L_cgs = arnett_luminosity(t_s, Mni, tm, t0, E0)
    Q_cgs = heating_rate(t_s, Mni)

    i_pk = np.argmax(L_cgs)
    return {
        't_d'   : t_s / DAY,
        'L_cgs' : L_cgs,
        'Q_cgs' : Q_cgs,
        'tm_d'  : tm / DAY,
        't0_d'  : t0 / DAY,
        'tPk_d' : t_s[i_pk] / DAY,
        'Lpk'   : L_cgs[i_pk],
    }


# ── Interactive plot ──────────────────────────────────────────────────────────

DEFAULTS = dict(
    Mej_sun  = 10.0,
    kappa    = 0.10,
    Mni_sun  = 0.05,
    v_kms    = 5000.0,
    R0_Rsun  = 500.0,
    E0_e49   = 1.0,
    t_shift  = 0.0,
    logL_dex = 0.0,
)


def build_interactive(data_t=None, data_L=None, data_Lerr=None):

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor('#0d0d0d')

    ax_main = fig.add_axes([0.07, 0.30, 0.59, 0.63])
    ax_main.set_yscale('log')
    ax_main.set_xlabel('Time since explosion  (days)')
    ax_main.set_ylabel(r'$L\ (L_\odot)$')
    ax_main.grid(True)

    ax_info = fig.add_axes([0.69, 0.30, 0.28, 0.63])
    ax_info.set_axis_off()
    info_text = ax_info.text(
        0.05, 0.96, '', transform=ax_info.transAxes,
        va='top', ha='left', fontsize=11,
        color='#cccccc', linespacing=1.9,
        fontfamily='monospace',
    )

    m = compute_model(**{k: DEFAULTS[k] for k in
                         ['Mej_sun', 'kappa', 'Mni_sun', 'v_kms', 'R0_Rsun', 'E0_e49']})

    lL, = ax_main.plot(m['t_d'], m['L_cgs'] / LSUN,
                       color=COLORS['L'], lw=2.0, label=r'$L(t)$')
    lQ, = ax_main.plot(m['t_d'], m['Q_cgs'] / LSUN,
                       color=COLORS['Q'], lw=1.5, ls='--', alpha=0.6,
                       label=r'$\dot{Q}(t)$')
    lPk = ax_main.axvline(m['tPk_d'],
                          color=COLORS['peak'], lw=1.2, ls=':', alpha=0.8)

    if data_t is not None:
        has_err = data_Lerr is not None and np.any(data_Lerr > 0)
        if has_err:
            ax_main.errorbar(data_t, data_L / LSUN, yerr=data_Lerr / LSUN,
                             fmt='o', color=COLORS['data'], ms=4.5,
                             elinewidth=1.0, capsize=2.5,
                             label='Data', zorder=5)
        else:
            ax_main.scatter(data_t, data_L / LSUN,
                            color=COLORS['data'], s=18, zorder=5, label='Data')

    ax_main.legend(facecolor='#1a1a1a', edgecolor='#333', loc='upper right')

    def fmt_card(m, tsh, ldex):
        dL = 10.0 ** ldex
        Lp = m['Lpk'] * dL / LSUN
        e  = int(np.floor(np.log10(max(Lp, 1e-30))))
        mn = Lp / 10 ** e
        return (
            f"  tₘ  = {m['tm_d']:.1f} d\n"
            f"  t₀  = {m['t0_d']:.2f} d\n"
            f"  t_peak  = {m['tPk_d'] + tsh:.1f} d\n"
            f"  L_peak  = {mn:.2f}e{e} L☉"
        )
    info_text.set_text(fmt_card(m, 0.0, 0.0))

    slider_defs = [
        # label                                  key        rect                         vmin   vmax    step   init
        (r'$M_{\rm ej}$ ($M_\odot$)',          'Mej_sun',  [0.07, 0.215, 0.38, 0.025],  0.1,   40.0,  0.1,   10.0),
        (r'$\kappa$ (cm$^2$ g$^{-1}$)',        'kappa',    [0.07, 0.175, 0.38, 0.025],  0.01,  0.40,  0.005, 0.10),
        (r'$v_{\rm ej}$ (km s$^{-1}$)',        'v_kms',    [0.07, 0.135, 0.38, 0.025],  500,   40000, 500,   5000),
        (r'$R_0$ ($R_\odot$)',                  'R0_Rsun',  [0.07, 0.095, 0.38, 0.025],  0,     3000,  10,    500 ),
        (r'$M_{\rm Ni}$ ($M_\odot$)',           'Mni_sun',  [0.55, 0.215, 0.38, 0.025],  0.001, 1.5,   0.005, 0.05),
        (r'$E_0$ ($10^{49}$ erg)',              'E0_e49',   [0.55, 0.175, 0.38, 0.025],  0.0,   20.0,  0.1,   1.0 ),
        (r'$\Delta t$ (days)',                  't_shift',  [0.55, 0.135, 0.38, 0.025], -300,   300,   0.5,   0.0 ),
        (r'$\Delta \log L$ (dex)',              'logL_dex', [0.55, 0.095, 0.38, 0.025], -4.0,   4.0,   0.05,  0.0 ),
    ]

    sliders = {}
    for label, key, rect, vmin, vmax, vstep, vinit in slider_defs:
        ax_sl = fig.add_axes(rect, facecolor='#1a1a1a')
        sl = Slider(ax_sl, label, vmin, vmax, valinit=vinit,
                    valstep=vstep, color=COLORS['slider'])
        sl.label.set_fontsize(10)
        sl.label.set_color('#aaaaaa')
        sl.valtext.set_fontsize(10)
        sl.valtext.set_color('#cccccc')
        sl.track.set_alpha(0.3)
        sliders[key] = sl

    def update(_=None):
        Mej  = sliders['Mej_sun'].val
        kap  = sliders['kappa'].val
        Mni  = sliders['Mni_sun'].val
        v    = sliders['v_kms'].val
        R0   = sliders['R0_Rsun'].val
        E0   = sliders['E0_e49'].val
        tsh  = sliders['t_shift'].val
        ldex = sliders['logL_dex'].val
        dL   = 10.0 ** ldex

        m = compute_model(Mej, kap, Mni, v, R0, E0)

        lL.set_xdata(m['t_d'] + tsh)
        lL.set_ydata(np.maximum(m['L_cgs'] * dL / LSUN, 1e-30))
        lQ.set_xdata(m['t_d'] + tsh)
        lQ.set_ydata(np.maximum(m['Q_cgs'] / LSUN, 1e-30))
        lPk.set_xdata([m['tPk_d'] + tsh, m['tPk_d'] + tsh])

        all_y = np.concatenate([m['L_cgs'] * dL / LSUN, m['Q_cgs'] / LSUN])
        if data_L is not None:
            all_y = np.concatenate([all_y, data_L / LSUN])
        pos = all_y[all_y > 0]
        if pos.size > 0:
            ymax = pos.max()
            ymin = pos.min()
            ax_main.set_ylim(max(ymin * 0.05, ymax * 1e-5), ymax * 8)

        info_text.set_text(fmt_card(m, tsh, ldex))
        fig.canvas.draw_idle()

    for sl in sliders.values():
        sl.on_changed(update)

    update()

    ax_reset = fig.add_axes([0.47, 0.04, 0.06, 0.03])
    ax_reset.set_facecolor('#1a1a1a')
    btn = Button(ax_reset, 'Reset', color='#1a1a1a', hovercolor='#2a2a2a')
    btn.label.set_color('#aaaaaa')
    btn.label.set_fontsize(10)

    def reset(_):
        for key, sl in sliders.items():
            sl.set_val(DEFAULTS[key])
    btn.on_clicked(reset)

    plt.figtext(0.07, 0.01,
                r'Arnett (1982) one-zone model  |  '
                r'$t_m = \sqrt{3\kappa M_{\rm ej}/4\pi v c}$  |  '
                r'True peak: $L(t_{\rm peak}) = \dot{Q}(t_{\rm peak})$',
                fontsize=9, color='#555555')

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Interactive Arnett lightcurve explorer with matplotlib sliders.'
    )
    parser.add_argument('data_file', nargs='?', default=None,
                        help='Optional data file to overlay (time_days, L, [L_err])')
    parser.add_argument('--units', default='lsun', choices=['lsun', 'ergs'])
    args = parser.parse_args()

    data_t = data_L = data_Lerr = None
    if args.data_file:
        data_t, data_L, data_Lerr = load_data(args.data_file, units=args.units)
        print(f'Loaded {len(data_t)} data points from {args.data_file}')

    fig = build_interactive(data_t, data_L, data_Lerr)
    plt.show()


if __name__ == '__main__':
    main()
