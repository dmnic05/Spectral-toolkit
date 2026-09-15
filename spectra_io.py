import os
import numpy as np
from astropy.io import fits

def load_fits_spectrum(path):
    """Return (wave[Å], flux, mjd) from a normalized FITS file."""
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        flux = hdul[0].data.astype(float)

        crval = hdr['CRVAL1']
        cdelt = hdr['CDELT1']
        naxis = hdr['NAXIS1']
        wave = crval + np.arange(naxis) * cdelt          # Å
        mjd = hdr.get('MJD-OBS')
    return wave, flux, mjd

def collect_spectra(folder):
    """Read every *.fits* in *folder* and return a list of dicts."""
    specs = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.fits'):
            continue
        wave, flux, mjd = load_fits_spectrum(os.path.join(folder, fn))
        specs.append({'file': fn, 'wave': wave, 'flux': flux, 'mjd': mjd})
    return specs
