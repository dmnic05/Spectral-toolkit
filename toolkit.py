"""
Complete RV Analysis Toolkit
Workflow: Load → Normalize → Plot → Fit Gaussians → RV per line → 
Store RVs → Plot RV(t) → Fit Period → Mass Estimate
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import pandas as pd
from lmfit import Model
from scipy.optimize import curve_fit, fsolve
from astropy.timeseries import LombScargle

c_kms = 299792.458

# ============================================================
# 1. SPECTRUM LOADING AND NORMALIZATION
# ============================================================

def load_fits_spectrum(path):
    """Load FITS, reconstruct wavelength, normalize flux."""
    with fits.open(path) as hdul:
        header = hdul[0].header
        flux_raw = np.asarray(hdul[0].data, dtype=float).squeeze()
    
    crval = header['CRVAL1']
    cdelt = header['CDELT1']
    crpix = header.get('CRPIX1', 1.0)
    pixel = np.arange(flux_raw.size) + 1.0
    log_wavelength = crval + (pixel - crpix) * cdelt
    wavelength = np.exp(log_wavelength)
    
    valid = np.isfinite(flux_raw)
    if np.sum(valid) > 0:
        continuum = np.median(flux_raw[valid])
        flux = flux_raw / continuum
    else:
        flux = flux_raw
    
    mjd = header.get('MJD-OBS', np.nan)
    return wavelength, flux, mjd


def collect_spectra(folder):
    """Load all FITS files from folder."""
    fits_files = sorted(glob.glob(f"{folder}/*.fits"))
    specs = []
    for fits_file in fits_files:
        wave, flux, mjd = load_fits_spectrum(fits_file)
        specs.append({'file': fits_file, 'wave': wave, 'flux': flux, 'mjd': mjd})
    return specs


# ============================================================
# 2. SPECTRUM PLOTTING
# ============================================================

def plot_normalized_spectrum(wave, flux, title="Spectrum", line_dict=None, 
                            ylim=(0, 1.5), figsize=(16, 6), save_path=None):
    """Plot normalized spectrum with optional line markers."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(wave, flux, 'b-', linewidth=0.7, label='Spectrum')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='Continuum')
    
    if line_dict is not None:
        for line_name, lam in line_dict.items():
            if lam > wave.min() and lam < wave.max():
                ax.axvline(lam, color='red', alpha=0.3, linewidth=0.8)
    
    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('Normalized Flux', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(ylim)
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    return fig, ax


def plot_line_fit(wave, flux, lambda_rest, result, window=15.0, 
                  title="Line Fit", save_path=None):
    """Plot spectrum with Gaussian fit overlay."""
    mask = (wave >= lambda_rest - window) & (wave <= lambda_rest + window)
    x = wave[mask]
    y = flux[mask]
    
    x_fine = np.linspace(x.min(), x.max(), 200)
    y_fit = result.model.eval(result.params, x=x_fine)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', linewidth=1.5, label='Spectrum')
    ax.plot(x_fine, y_fit, 'r-', linewidth=2, label='Gaussian fit')
    ax.axvline(lambda_rest, color='gray', linestyle='--', alpha=0.5, label='Rest λ')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('Normalized Flux', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    return fig, ax


# ============================================================
# 3. GAUSSIAN LINE FITTING
# ============================================================

def gaussian(x, amp, mu, sigma, cont):
    """Gaussian profile for absorption/emission lines."""
    return cont + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def doppler_rv(lambda_obs, lambda_rest):
    """Convert observed wavelength to RV."""
    return c_kms * (lambda_obs - lambda_rest) / lambda_rest


def measure_rv_epoch(wave, flux, lambda_rest, window=15.0, emission=False,
                     return_fit=False, min_points=20, min_depth=0.01,
                     min_sigma=0.25, max_sigma=5.0, mu_guard=3.0, max_rv_err=30.0):
    """Gaussian RV fit with quality controls."""
    
    mask = (wave >= lambda_rest - window) & (wave <= lambda_rest + window)
    if np.sum(mask) < min_points:
        return (np.nan, np.nan, None) if return_fit else (np.nan, np.nan)

    x = wave[mask]
    y = flux[mask]

    if emission:
        idx0 = np.nanargmax(y)
        mu0 = x[idx0]
        depth = np.nanmax(y) - 1
    else:
        idx0 = np.nanargmin(y)
        mu0 = x[idx0]
        depth = 1 - np.nanmin(y)

    if not np.isfinite(depth) or depth < min_depth:
        return (np.nan, np.nan, None) if return_fit else (np.nan, np.nan)

    amp0 = (+depth if emission else -depth)
    sigma0 = 0.8

    gmodel = Model(gaussian)
    params = gmodel.make_params(amp=amp0, mu=mu0, sigma=sigma0, cont=1.0)

    params["sigma"].set(min=min_sigma, max=max_sigma)
    params["cont"].set(min=0.5, max=1.5)

    if emission:
        params["amp"].set(min=+min_depth, max=+1.0)
    else:
        params["amp"].set(min=-1.0, max=-min_depth)

    params["mu"].set(min=mu0 - mu_guard, max=mu0 + mu_guard)

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

    if np.isfinite(rv_err) and rv_err > max_rv_err:
        return (np.nan, np.nan, result) if return_fit else (np.nan, np.nan)

    return (rv, rv_err, result) if return_fit else (rv, rv_err)


# ============================================================
# 4. MULTI-LINE RV MEASUREMENT (store RVs from dictionary)
# ============================================================

def measure_rv_multiline(wave, flux, line_dict, window_dict=None, 
                         emission=False, verbose=False):
    """
    Measure RV from multiple lines in single epoch.
    Store RVs from all lines in dictionary.
    """
    if window_dict is None:
        window_dict = {name: 15.0 for name in line_dict}
    
    rv_dict = {}
    rvs = []
    
    for line_name, lambda_rest in line_dict.items():
        window = window_dict.get(line_name, 15.0)
        
        rv, rv_err, result = measure_rv_epoch(
            wave, flux, lambda_rest, window=window, emission=emission,
            return_fit=True, min_depth=0.01, mu_guard=3.0, max_rv_err=30.0
        )
        
        rv_dict[line_name] = {'rv': rv, 'rv_err': rv_err, 'result': result}
        
        if np.isfinite(rv):
            rvs.append(rv)
            if verbose:
                print(f"  {line_name:15s}: RV = {rv:7.2f}±{rv_err:6.2f} km/s")
        else:
            if verbose:
                print(f"  {line_name:15s}: FAILED")
    
    stats = {
        'median': np.median(rvs) if len(rvs) > 0 else np.nan,
        'std': np.std(rvs) if len(rvs) > 0 else np.nan,
        'n_lines': len(rvs),
        'all_rvs': rvs
    }
    
    return rv_dict, stats


def measure_rv_all_epochs(specs, line_dict, window_dict=None, 
                          emission=False, verbose=True):
    """Measure RV for all epochs."""
    if window_dict is None:
        window_dict = {name: 15.0 for name in line_dict}
    
    results = []
    
    for i, spec in enumerate(specs, 1):
        filename = spec['file'].split('\\')[-1] if '\\' in spec['file'] else spec['file'].split('/')[-1]
        print(f"[{i}/{len(specs)}] {filename}", end='', flush=True)
        
        rv_dict, stats = measure_rv_multiline(
            spec['wave'], spec['flux'], line_dict,
            window_dict=window_dict, emission=emission, verbose=False
        )
        
        if stats['n_lines'] > 0:
            result = {
                'mjd': spec['mjd'],
                'file': spec['file'],
                'rv_median': stats['median'],
                'rv_std': stats['std'],
                'n_lines': stats['n_lines'],
                'rv_dict': rv_dict
            }
            results.append(result)
            print(f" ✓ RV = {stats['median']:7.2f}±{stats['std']:5.2f} km/s (n={stats['n_lines']})")
        else:
            print(f" (no lines detected)")
    
    return results


# ============================================================
# 5. TIME SERIES ANALYSIS
# ============================================================

def plot_rv_timeseries(results, title="RV Time Series", save_path=None, figsize=(12, 6)):
    """Plot RV vs MJD."""
    mjds = np.array([r['mjd'] for r in results])
    rvs = np.array([r['rv_median'] for r in results])
    errs = np.array([r['rv_std'] for r in results])
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(mjds, rvs, yerr=errs, fmt='o', capsize=5, markersize=6, linewidth=1.5)
    ax.set_xlabel('MJD', fontsize=12)
    ax.set_ylabel('Radial Velocity (km/s)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    return fig, ax


def print_rv_summary(results):
    """Print RV summary statistics."""
    rvs = np.array([r['rv_median'] for r in results])
    print("\n" + "="*60)
    print("RV SUMMARY STATISTICS")
    print("="*60)
    print(f"Total epochs: {len(results)}")
    print(f"RV range: {rvs.min():.1f} to {rvs.max():.1f} km/s")
    print(f"RV mean: {rvs.mean():.1f} km/s")
    print(f"RV std dev: {rvs.std():.1f} km/s")
    print(f"RV scatter: {rvs.max() - rvs.min():.1f} km/s")
    print("="*60 + "\n")


def results_to_dataframe(results):
    """Convert results to pandas DataFrame."""
    data = []
    for r in results:
        data.append({
            'mjd': r['mjd'],
            'rv_median': r['rv_median'],
            'rv_std': r['rv_std'],
            'n_lines': r['n_lines'],
        })
    return pd.DataFrame(data)


# ============================================================
# 6. PERIOD FITTING
# ============================================================

def lomb_scargle_period(t, rv, err, fmin=1/2000, fmax=1/2, npp=10):
    """Compute Lomb-Scargle periodogram."""
    freq, power = LombScargle(t, rv, err).autopower(
        minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=npp,
    )
    return 1/freq, power


def sinusoid(t, gamma, K, phi, P, t0=0):
    """Circular orbital model: RV = gamma + K*sin(2π(t-t0)/P + phi)"""
    return gamma + K * np.sin(2*np.pi*(t - t0)/P + phi)


def fit_fixed_period(t, rv, err, P, t0=0):
    """Fit gamma, K, phi for a given period P."""
    def model(t, gamma, K, phi):
        return gamma + K * np.sin(2*np.pi*(t - t0)/P + phi)
    
    K0 = 0.5 * (rv.max() - rv.min())
    
    try:
        popt, pcov = curve_fit(
            model, t, rv, p0=[np.mean(rv), K0, 0.0],
            sigma=err, absolute_sigma=True,
            bounds=([50, 0, -np.pi], [300, 200, np.pi]), maxfev=50000,
        )
        perr = np.sqrt(np.diag(pcov))
        chi2 = np.sum(((rv - model(t, *popt))/err)**2)
        redchi = chi2 / (len(rv) - 3)
        return popt, perr, redchi
    except:
        return None, None, None


def plot_period_fit(t, rv, err, P, popt, title="Period Fit", save_path=None):
    """Plot RV data with best-fit sinusoid."""
    t_fine = np.linspace(t.min(), t.max(), 500)
    rv_fit = sinusoid(t_fine, *popt, P=P)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(t, rv, yerr=err, fmt='o', capsize=5, markersize=6, linewidth=1.5, label='Data')
    ax.plot(t_fine, rv_fit, 'r-', linewidth=2, label=f'Fit (P={P:.2f} d)')
    
    ax.set_xlabel('MJD', fontsize=12)
    ax.set_ylabel('Radial Velocity (km/s)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    return fig, ax


# ============================================================
# 7. MASS ESTIMATE
# ============================================================

def mass_function(P_days, K_kms, e=0.0):
    """Compute mass function f(M) in solar masses."""
    G = 6.67430e-11
    M_sun = 1.98847e30
    
    P_sec = P_days * 86400
    K_ms = K_kms * 1e3
    
    f_M = (P_sec * K_ms**3 * (1 - e**2)**1.5) / (2 * np.pi * G) / M_sun
    return f_M


def estimate_companion_mass(P_days, K_kms, M_primary, e=0.0, i_deg=90):
    """Estimate companion mass from mass function."""
    f_M = mass_function(P_days, K_kms, e=e)
    sin_i = np.sin(np.radians(i_deg))
    
    if np.abs(sin_i - 1.0) < 0.01:
        M_comp = M_primary * f_M / (1 - f_M / M_primary) if f_M < M_primary else np.inf
    else:
        def eq(M_c):
            return M_c * sin_i**3 / (M_primary + M_c)**2 - f_M
        M_comp = fsolve(eq, 10.0)[0]
    
    return M_comp, f_M


def print_period_fit_summary(P, popt, perr, redchi):
    """Print period fit results."""
    gamma, K, phi = popt
    gamma_err, K_err, phi_err = perr
    
    print("\n" + "="*60)
    print("PERIOD FIT RESULTS")
    print("="*60)
    print(f"Period: P = {P:.3f} days")
    print(f"Systemic RV: γ = {gamma:.2f}±{gamma_err:.2f} km/s")
    print(f"Semi-amplitude: K = {K:.2f}±{K_err:.2f} km/s")
    print(f"Phase: φ = {phi:.3f}±{phi_err:.3f} rad")
    print(f"Reduced χ²: {redchi:.3f}")
    print("="*60 + "\n")
