import matplotlib.pyplot as plt
import numpy as np

def rv_vs_mjd(t, rv, err, instrument, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(t, rv, yerr=err, fmt='o' if instrument=='FLAMES' else 's',
                label=instrument, capsize=3)
    ax.set_xlabel('MJD')
    ax.set_ylabel('RV (km s⁻¹)')
    ax.legend()
    return ax

def line_overlay(wave, flux, line_wave, window=15, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    mask = (wave >= line_wave-window) & (wave <= line_wave+window)
    ax.plot(wave[mask], flux[mask], lw=1)
    ax.axvline(line_wave, ls=':', color='k')
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Norm. flux')
    return ax
