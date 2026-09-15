"""
Spectral line fitting and RV extraction.
"""

import numpy as np
from lmfit import Model

c_kms = 299792.458

def gaussian(x, amp, mu, sigma, cont):
    """Gaussian profile for absorption/emission lines."""
    return cont + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def doppler_rv(lambda_obs, lambda_rest):
    """Convert observed wavelength to RV."""
    return c_kms * (lambda_obs - lambda_rest) / lambda_rest

def measure_rv_epoch(
    wave, flux, lambda_rest,
    window=15.0,
    emission=False,
    return_fit=False,
    min_points=20,
    min_depth=0.01,
    min_sigma=0.25,
    max_sigma=5.0,
    mu_guard=3.0,
    max_rv_err=30.0,
):
    """
    Gaussian RV fit with quality controls.
    
    Parameters
    ----------
    wave, flux : array
        Wavelength (Å) and normalized flux
    lambda_rest : float
        Rest wavelength (Å)
    window : float
        Fitting window (Å) around lambda_rest
    emission : bool
        If True, fit emission line (positive amplitude)
    return_fit : bool
        If True, also return lmfit result object
    min_depth, mu_guard, max_rv_err : float
        Quality cuts (see oldcode for details)
    
    Returns
    -------
    rv : float
        Radial velocity (km/s)
    rv_err : float
        RV uncertainty (km/s)
    result : lmfit.ModelResult or None
        Fit result (only if return_fit=True)
    """
    
    mask = (wave >= lambda_rest - window) & (wave <= lambda_rest + window)
    if np.sum(mask) < min_points:
        return (np.nan, np.nan, None) if return_fit else (np.nan, np.nan)

    x = wave[mask]
    y = flux[mask]

    # Find initial guess from local extremum
    if emission:
        idx0 = np.nanargmax(y)
        mu0 = x[idx0]
        depth = np.nanmax(y) - 1
    else:
        idx0 = np.nanargmin(y)
        mu0 = x[idx0]
        depth = 1 - np.nanmin(y)

    # Pre-fit rejection
    if not np.isfinite(depth) or depth < min_depth:
        return (np.nan, np.nan, None) if return_fit else (np.nan, np.nan)

    # Initial amplitude
    amp0 = (+depth if emission else -depth)
    sigma0 = 0.8

    # Set up model
    gmodel = Model(gaussian)
    params = gmodel.make_params(amp=amp0, mu=mu0, sigma=sigma0, cont=1.0)

    params["sigma"].set(min=min_sigma, max=max_sigma)
    params["cont"].set(min=0.5, max=1.5)

    if emission:
        params["amp"].set(min=+min_depth, max=+1.0)
    else:
        params["amp"].set(min=-1.0, max=-min_depth)

    params["mu"].set(min=mu0 - mu_guard, max=mu0 + mu_guard)

    # Fit
    try:
        result = gmodel.fit(y, params, x=x)
    except:
        return (np.nan, np.nan, None) if return_fit else (np.nan, np.nan)

    if result is None or not result.success:
        return (np.nan, np.nan, result) if return_fit else (np.nan, np.nan)

    mu_fit = result.params["mu"].value
    mu_err = result.params["mu"].stderr

    rv = doppler_rv(mu_fit, lambda_rest)

    if mu_err is None or not np.isfinite(mu_err):
        rv_err = np.nan
    else:
        rv_err = (c_kms / lambda_rest) * mu_err

    # Reject absurd errors
    if np.isfinite(rv_err) and rv_err > max_rv_err:
        return (np.nan, np.nan, result) if return_fit else (np.nan, np.nan)

    return (rv, rv_err, result) if return_fit else (rv, rv_err)


def compute_sn_normalized(wave, flux, cont_windows):
    """
    Compute S/N from normalized spectrum using continuum windows.
    
    Parameters
    ----------
    wave : array
        Wavelength (Å)
    flux : array
        Normalized flux
    cont_windows : list of tuples
        [(w1, w2), ...] wavelength ranges for continuum
    
    Returns
    -------
    sn : float
        Signal-to-noise ratio (1 / std of continuum)
    """
    cont_flux = []
    for w1, w2 in cont_windows:
        mask = (wave >= w1) & (wave <= w2)
        if np.sum(mask) > 10:
            cont_flux.append(flux[mask])
    
    if len(cont_flux) == 0:
        return np.nan
    
    cont_flux = np.concatenate(cont_flux)
    sigma = np.nanstd(cont_flux)
    
    if sigma <= 0:
        return np.nan
    
    return 1.0 / sigma
