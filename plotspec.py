"""Standalone observed-frame full-spectrum overlay (Python 3.7).
No Gaussian fits, RV removal, resampling, residuals or file saving.
Colours follow the original convention: chronological full-sample E01 onward,
tab20 for <=20 exposures, otherwise turbo (viridis fallback on old Matplotlib).
Select the same FITS sample as before to retain identical labels and colours.
Assumes primary 1D flux, wavelength header and BJD; see settings below.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
try:
    from matplotlib.widgets import RangeSlider, Button
except ImportError:
    RangeSlider = None

# ---------------- settings ----------------
SPECTRA_DIR = Path(__file__).resolve().parent / 'spectra'
FILE_PATTERN = '*.fits'
WAVELENGTH_MODE = 'loge'       # 'loge', 'log10', or 'linear'
BJD_KEY = 'BJD'
EPOCH_SELECTION = '5,8,11,13'  # 'all' or original chronological numbers
WAVE_RANGE = None             # full spectrum; or e.g. (4300, 5100)
FLUX_LIMITS = (0.0, 1.15)     # None for automatic limits
SHOW_LINE_LABELS = True
# AIR-wavelength guides, not automatic identifications.
GUIDE_LINES = {
    'Hδ': 4101.734, 'Hγ': 4340.472, 'He I 4388': 4387.929,
    'He I 4471': 4471.48, 'He II 4542': 4541.59,
    'He II 4686': 4685.68, 'He I 4713': 4713.146,
    'Hβ': 4861.333, 'He I 4922': 4921.931, 'He I 5016': 5015.678,
    'He II 5412': 5411.52, 'He I 5876': 5875.62,
    'Na I D': 5892.0, 'Hα': 6562.80, 'He I 6678': 6678.149,
    'He I 7065': 7065.19,
}


def read_spectrum(path):
    with fits.open(str(path), memmap=False) as hdul:
        header = hdul[0].header
        flux = np.asarray(hdul[0].data, dtype=float).squeeze()
        if flux.ndim != 1:
            raise ValueError('Expected 1D primary-HDU flux.')
        step = header.get('CDELT1', header.get('CD1_1'))
        if step is None:
            raise ValueError('Missing CDELT1/CD1_1.')
        pixel = np.arange(flux.size) + 1.0
        coordinate = float(header['CRVAL1']) + (pixel - float(header.get('CRPIX1', 1))) * float(step)
        if WAVELENGTH_MODE == 'loge':
            wave = np.exp(coordinate)
        elif WAVELENGTH_MODE == 'log10':
            wave = 10.0**coordinate
        elif WAVELENGTH_MODE == 'linear':
            wave = coordinate
        else:
            raise ValueError('Invalid WAVELENGTH_MODE.')
    valid_wave = np.isfinite(wave) & (wave > 0)
    wave, flux = wave[valid_wave], flux[valid_wave]
    flux[~np.isfinite(flux)] = np.nan
    order = np.argsort(wave)
    wave, flux = wave[order], flux[order]
    wave, unique = np.unique(wave, return_index=True)
    flux = flux[unique]
    if wave.size < 20 or np.isfinite(flux).sum() < 20:
        raise ValueError('Too few usable pixels.')
    if not 1000 < np.median(wave) < 30000:
        raise ValueError('Unexpected wavelength scale: check WAVELENGTH_MODE and Å units.')
    # Insert NaNs to avoid connecting across substantial wavelength gaps.
    spacing = np.diff(np.log(wave))
    typical = np.median(spacing[spacing > 0])
    gaps = np.flatnonzero(spacing > 10 * typical)
    if gaps.size:
        positions = gaps + 1
        flux = np.insert(flux, positions, np.nan)
        wave = np.insert(wave, positions, np.sqrt(wave[gaps] * wave[gaps + 1]))
    return wave, flux


def main():
    paths = sorted(SPECTRA_DIR.glob(FILE_PATTERN))
    if not paths:
        raise FileNotFoundError('No {} files in {}. Check SPECTRA_DIR.'.format(FILE_PATTERN, SPECTRA_DIR))
    records, failures = [], []
    for path in paths:
        try:
            bjd = float(fits.getheader(str(path), 0)[BJD_KEY])
            if not np.isfinite(bjd):
                raise ValueError('Invalid BJD.')
            records.append((bjd, path))
        except (KeyError, ValueError, OSError) as exc:
            failures.append('{}: {}'.format(path.name, exc))
    records.sort(key=lambda item: (item[0], str(item[1])))
    n_all = len(records)
    if not n_all:
        raise RuntimeError('No valid BJD headers. ' + '; '.join(failures))
    # Assign full-sample chronological labels and colours before selection.
    selection = str(EPOCH_SELECTION).strip().lower()
    numbers = list(range(1, n_all + 1)) if selection == 'all' else [
        int(item.strip()) for item in selection.split(',') if item.strip()]
    if not numbers or len(numbers) != len(set(numbers)) or any(k < 1 or k > n_all for k in numbers):
        raise ValueError('Select unique epoch numbers between 1 and {}.'.format(n_all))
    wanted = set(numbers)
    if n_all <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        try:
            cmap = plt.get_cmap('turbo')
        except ValueError:
            cmap = plt.get_cmap('viridis')
            print('turbo unavailable: using viridis (colours will differ from turbo).')
    profiles = []
    for number, (bjd, path) in enumerate(records, 1):
        if number not in wanted:
            continue
        label = 'E{:02d}'.format(number)
        try:
            wave, flux = read_spectrum(path)
            colour = cmap(number - 1) if n_all <= 20 else cmap((number - 1) / max(n_all - 1, 1))
            profiles.append(dict(label=label, BJD=bjd, file=path.name,
                                 wave=wave, flux=flux, colour=colour))
        except (KeyError, ValueError, OSError) as exc:
            failures.append('{} {}: {}'.format(label, path.name, exc))
    if not profiles:
        raise RuntimeError('No usable selected spectra. ' + '; '.join(failures))
    wmin = min(p['wave'][0] for p in profiles)
    wmax = max(p['wave'][-1] for p in profiles)
    limits = (wmin, wmax) if WAVE_RANGE is None else (max(wmin, WAVE_RANGE[0]), min(wmax, WAVE_RANGE[1]))
    if limits[0] >= limits[1]:
        raise ValueError('WAVE_RANGE does not overlap the spectra.')
    fig, ax = plt.subplots(figsize=(16, 6))
    for p in profiles:
        ax.plot(p['wave'], p['flux'], color=p['colour'], lw=0.8,
                label='{} | {:.5f}'.format(p['label'], p['BJD']))
    ax.axhline(1, color='0.5', ls=':', lw=0.7)
    ax.set_xlabel('Wavelength [Å]')
    ax.set_ylabel('Input normalised flux')
    ax.set_title('Observed frame — {} selected exposures; original epoch colours'.format(len(profiles)))
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax.grid(alpha=0.15)
    if FLUX_LIMITS is not None:
        ax.set_ylim(*FLUX_LIMITS)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    annotations = []
    if SHOW_LINE_LABELS:
        for name, wave in GUIDE_LINES.items():
            if wmin <= wave <= wmax:
                ax.axvline(wave, color='0.6', lw=0.5, alpha=0.5)
                text = ax.text(wave, 0.99, name, rotation=90, fontsize=7, va='top', ha='right',
                               transform=ax.get_xaxis_transform())
                annotations.append((wave, text))
    fig.subplots_adjust(left=0.07, right=0.82, top=0.9, bottom=0.2)
    def update(lim):
        if lim[0] >= lim[1]:
            return
        ax.set_xlim(*lim)
        for wave, text in annotations:
            text.set_visible(lim[0] <= wave <= lim[1])
        fig.canvas.draw_idle()
    if RangeSlider is not None:
        slider = RangeSlider(fig.add_axes([0.13, 0.065, 0.61, 0.025]), 'Wavelength [Å]',
                             wmin, wmax, valinit=limits, valfmt='%0.1f')
        button = Button(fig.add_axes([0.76, 0.055, 0.065, 0.045]), 'Full range')
        slider.on_changed(update)
        button.on_clicked(lambda event: slider.set_val((wmin, wmax)))
        fig._viewer_widgets = (slider, button)
    update(limits)
    if failures:
        print('Excluded: ' + '; '.join(failures))
    plt.show()
    return profiles


if __name__ == '__main__':
    spectrum_profiles = main()
