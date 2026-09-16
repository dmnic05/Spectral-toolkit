import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from scipy.interpolate import splrep, splev
from scipy.optimize import curve_fit, brentq
from lmfit import Model
from astropy.io import fits
from astropy.timeseries import LombScargle

# Additional imports required by the original UI function
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import re

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 14})
c_kms = 299792.458

# ============================================================
# 10. GAUSSIAN RV FITTING
# ============================================================

def gaussian(x, amp, mu, sigma, cont):
    """Gaussian spectral-line profile."""
    return cont + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def doppler_rv(lambda_obs, lambda_rest):
    """Classical Doppler RV in km/s."""
    return c_kms * (lambda_obs - lambda_rest) / lambda_rest


def measure_rv_epoch(wave, flux, lambda_rest, window=15.0, emission=False, return_fit=False, min_points=20, min_depth=0.01, min_sigma=0.25, max_sigma=5.0, mu_guard=3.0, max_rv_err=30.0):
    """Fit one Gaussian line and return RV and RV uncertainty."""
    wave, flux = np.asarray(wave, dtype=float), np.asarray(flux, dtype=float)

    mask = (wave >= lambda_rest - window) & (wave <= lambda_rest + window) & np.isfinite(wave) & np.isfinite(flux)

    if np.sum(mask) < min_points:
        result = (np.nan, np.nan, None)
        return result if return_fit else result[:2]

    x, y = wave[mask], flux[mask]

    if emission:
        idx = np.argmax(y)
        mu0 = x[idx]
        amp0 = max(y[idx] - 1.0, min_depth)
        amp_min, amp_max = min_depth, max(1.0, amp0 * 5)
    else:
        idx = np.argmin(y)
        mu0 = x[idx]
        amp0 = min(y[idx] - 1.0, -min_depth)
        amp_min, amp_max = min(-min_depth, amp0 * 5), -min_depth

    if emission:
        depth = np.nanmax(y) - 1.0
    else:
        depth = 1.0 - np.nanmin(y)

    if not np.isfinite(depth) or depth < min_depth:
        result = (np.nan, np.nan, None)
        return result if return_fit else result[:2]

    model = Model(gaussian)
    params = model.make_params(amp=amp0, mu=mu0, sigma=0.8, cont=1.0)
    params["amp"].set(min=amp_min, max=amp_max)
    params["mu"].set(min=mu0 - mu_guard, max=mu0 + mu_guard)
    params["sigma"].set(min=min_sigma, max=max_sigma)
    params["cont"].set(min=0.5, max=1.5)

    try:
        result = model.fit(y, params, x=x)
    except Exception:
        result = None

    if result is None or not result.success:
        output = (np.nan, np.nan, result)
        return output if return_fit else output[:2]

    mu = result.params["mu"].value
    mu_err = result.params["mu"].stderr
    rv = doppler_rv(mu, lambda_rest)
    rv_err = np.nan if mu_err is None else abs(c_kms * mu_err / lambda_rest)

    if np.isfinite(rv_err) and rv_err > max_rv_err:
        rv, rv_err = np.nan, np.nan

    output = (rv, rv_err, result)
    return output if return_fit else output[:2]


def plot_line_fit(wave, flux, lambda_rest, result, window=15.0, title="Line Fit", save_path=None):
    """Plot one Gaussian line fit."""
    mask = (wave >= lambda_rest - window) & (wave <= lambda_rest + window)
    x, y = np.asarray(wave)[mask], np.asarray(flux)[mask]

    if len(x) == 0:
        raise ValueError("No data inside line-fitting window.")

    xfine = np.linspace(x.min(), x.max(), 300)
    yfit = result.model.eval(result.params, x=xfine)

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(x, y, "k-", label="Spectrum")
    axis.plot(xfine, yfit, "r-", lw=2, label="Gaussian fit")
    axis.axvline(lambda_rest, ls="--", alpha=0.5, label="Rest wavelength")
    axis.axhline(1, ls=":", alpha=0.5)
    axis.set_xlabel(r"Wavelength [$\AA$]")
    axis.set_ylabel("Normalized Flux")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axis


# ============================================================
# 11. MULTI-LINE RV
# ============================================================

def measure_rv_multiline(wave, flux, line_dict, window_dict=None, emission=False, verbose=False):
    """Measure RV independently from multiple spectral lines."""
    if window_dict is None:
        window_dict = {name: 15.0 for name in line_dict}

    rv_dict = {}
    rvs = []
    rv_errs = []

    for name, lambda_rest in line_dict.items():
        rv, rv_err, result = measure_rv_epoch(wave, flux, lambda_rest, window=window_dict.get(name, 15.0), emission=emission, return_fit=True)

        rv_dict[name] = {"rv": rv, "rv_err": rv_err, "result": result}

        if np.isfinite(rv):
            rvs.append(rv)
            if np.isfinite(rv_err) and rv_err > 0:
                rv_errs.append(rv_err)

        if verbose:
            if np.isfinite(rv):
                print(f"  {name:15s}: {rv:8.2f} ± {rv_err:6.2f} km/s")
            else:
                print(f"  {name:15s}: FAILED")

    rvs = np.asarray(rvs, dtype=float)

    stats = {
        "median": np.nanmedian(rvs) if len(rvs) else np.nan,
        "std": np.nanstd(rvs, ddof=1) if len(rvs) > 1 else np.nan,
        "stderr": np.nanstd(rvs, ddof=1) / np.sqrt(len(rvs)) if len(rvs) > 1 else (rv_errs[0] if rv_errs else np.nan),
        "n_lines": len(rvs),
        "all_rvs": rvs,
        "line_errors": np.asarray(rv_errs)
    }

    return rv_dict, stats


def measure_rv_all_epochs(specs, line_dict, window_dict=None, emission=False, verbose=True):
    """Measure multi-line RV for every epoch."""
    results = []

    for i, spec in enumerate(specs, 1):
        filename = os.path.basename(spec["file"])
        flux = spec.get("flux_norm", spec.get("flux"))

        if verbose:
            print(f"[{i}/{len(specs)}] {filename}")

        rv_dict, stats = measure_rv_multiline(spec["wave"], flux, line_dict, window_dict, emission, verbose)

        if stats["n_lines"]:
            results.append({
                "mjd": spec["mjd"],
                "file": spec["file"],
                "rv_median": stats["median"],
                "rv_std": stats["std"],
                "rv_err": stats["stderr"],
                "n_lines": stats["n_lines"],
                "rv_dict": rv_dict
            })

            if verbose:
                print(f"  -> Epoch RV = {stats['median']:.2f} ± {stats['stderr']:.2f} km/s\n")
        elif verbose:
            print("  -> No valid lines.\n")

    return results


# ============================================================
# 12. RV TIME SERIES
# ============================================================

def results_to_dataframe(results):
    """Convert RV results to a DataFrame."""
    return pd.DataFrame([{
        "mjd": r.get("mjd", np.nan),
        "rv_median": r.get("rv_median", np.nan),
        "rv_std": r.get("rv_std", np.nan),
        "rv_err": r.get("rv_err", np.nan),
        "n_lines": r.get("n_lines", 0),
        "file": r.get("file", "")
    } for r in results])


def save_rv_results(results, output_file):
    """Save RV results to CSV."""
    df = results_to_dataframe(results)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved RV results to {output_file}")
    return df


def plot_rv_timeseries(results, title="RV Time Series", save_path=None, figsize=(12, 6)):
    """Plot radial velocity against MJD."""
    if not results:
        raise ValueError("No RV results supplied.")

    df = results_to_dataframe(results)
    valid = np.isfinite(df["mjd"]) & np.isfinite(df["rv_median"])

    if not valid.any():
        raise ValueError("No finite RV/MJD values.")

    t = df.loc[valid, "mjd"].to_numpy()
    rv = df.loc[valid, "rv_median"].to_numpy()
    err = df.loc[valid, "rv_err"].to_numpy()

    plot_err = np.where(np.isfinite(err) & (err > 0), err, 0)

    fig, axis = plt.subplots(figsize=figsize)
    axis.errorbar(t, rv, yerr=plot_err, fmt="o", capsize=4)
    axis.set_xlabel("MJD")
    axis.set_ylabel("Radial Velocity [km/s]")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axis


def print_rv_summary(results):
    """Print basic RV statistics."""
    df = results_to_dataframe(results)
    rv = df["rv_median"].to_numpy()
    rv = rv[np.isfinite(rv)]

    if len(rv) == 0:
        print("No finite RV measurements.")
        return

    print("\n" + "=" * 55)
    print("RV SUMMARY")
    print("=" * 55)
    print(f"Epochs       : {len(rv)}")
    print(f"Minimum RV   : {np.min(rv):.2f} km/s")
    print(f"Maximum RV   : {np.max(rv):.2f} km/s")
    print(f"Mean RV      : {np.mean(rv):.2f} km/s")
    print(f"Median RV    : {np.median(rv):.2f} km/s")
    print(f"Std. dev.    : {np.std(rv, ddof=1) if len(rv) > 1 else np.nan:.2f} km/s")
    print(f"Peak-to-peak : {np.ptp(rv):.2f} km/s")
    print("=" * 55 + "\n")


# ============================================================
# 13. LOMB-SCARGLE
# ============================================================

def lomb_scargle_period(t, rv, err=None, fmin=1 / 2000, fmax=1 / 2, npp=10):
    """Compute Lomb-Scargle periodogram."""
    t, rv = np.asarray(t, dtype=float), np.asarray(rv, dtype=float)

    if err is None:
        err = np.ones_like(rv)
    else:
        err = np.asarray(err, dtype=float)

    valid = np.isfinite(t) & np.isfinite(rv) & np.isfinite(err) & (err > 0)

    t, rv, err = t[valid], rv[valid], err[valid]

    if len(t) < 3:
        raise ValueError("At least three valid RV points are required.")

    baseline = np.ptp(t)

    if baseline <= 0:
        raise ValueError("Observation times must span a non-zero baseline.")

    min_frequency = max(float(fmin), 1.0 / (10.0 * baseline))
    max_frequency = float(fmax)

    if min_frequency >= max_frequency:
        raise ValueError("Invalid frequency range.")

    ls = LombScargle(t, rv, err)
    frequency, power = ls.autopower(minimum_frequency=min_frequency, maximum_frequency=max_frequency, samples_per_peak=npp)

    periods = 1.0 / frequency
    best_idx = np.argmax(power)

    return periods, power, periods[best_idx], frequency, ls


def plot_periodogram(periods, power, best_period=None, title="Lomb-Scargle Periodogram", save_path=None):
    """Plot Lomb-Scargle power versus period."""
    periods, power = np.asarray(periods), np.asarray(power)

    fig, axis = plt.subplots(figsize=(12, 6))
    axis.plot(periods, power, "k-")
    axis.set_xlabel("Period [days]")
    axis.set_ylabel("Lomb-Scargle Power")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.set_xscale("log")

    if best_period is not None:
        idx = np.nanargmin(np.abs(periods - best_period))
        axis.axvline(best_period, ls="--", label=f"Best P = {best_period:.3f} d")
        axis.scatter([best_period], [power[idx]], zorder=5)
        axis.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axis


# ============================================================
# 14. SINUSOIDAL RV FIT
# ============================================================

def sinusoid(t, gamma, K, phi, P):
    """Sinusoidal RV model."""
    return gamma + K * np.sin(2 * np.pi * t / P + phi)


def fit_fixed_period(t, rv, err, P):
    """Fit gamma, K and phase for a fixed period."""
    t, rv, err = map(np.asarray, (t, rv, err))
    valid = np.isfinite(t) & np.isfinite(rv) & np.isfinite(err) & (err > 0)

    t, rv, err = t[valid], rv[valid], err[valid]

    if len(t) < 3:
        raise ValueError("At least three valid RV points are required.")

    gamma0 = np.nanmedian(rv)
    K0 = 0.5 * (np.nanmax(rv) - np.nanmin(rv))

    def model(t, gamma, K, phi):
        return sinusoid(t, gamma, K, phi, P)

    sigma_floor = max(np.nanmedian(err[err > 0]) * 0.1, 1e-6)
    err = np.maximum(err, sigma_floor)

    bounds = ([-np.inf, 0.0, -2 * np.pi], [np.inf, max(10.0 * max(K0, 1.0), 1e3), 2 * np.pi])

    popt, pcov = curve_fit(model, t, rv, p0=[gamma0, K0, 0.0], sigma=err, absolute_sigma=True, bounds=bounds, maxfev=50000)

    perr = np.sqrt(np.maximum(np.diag(pcov), 0))

    return popt, perr, pcov


def plot_period_fit(t, rv, err, P, popt, title=None, save_path=None):
    """Plot RV measurements and fitted sinusoid."""
    t, rv, err = map(np.asarray, (t, rv, err))
    order = np.argsort(t)
    t, rv, err = t[order], rv[order], err[order]

    tfine = np.linspace(t.min(), t.max(), 1000)
    fit = sinusoid(tfine, popt[0], popt[1], popt[2], P)

    fig, axis = plt.subplots(figsize=(12, 6))
    axis.errorbar(t, rv, yerr=np.where(np.isfinite(err) & (err > 0), err, 0), fmt="o", capsize=4, label="RV measurements")
    axis.plot(tfine, fit, "r-", lw=2, label=f"P = {P:.4f} d")
    axis.set_xlabel("MJD")
    axis.set_ylabel("Radial Velocity [km/s]")
    axis.set_title(title or "Sinusoidal RV Fit")
    axis.grid(alpha=0.3)
    axis.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axis


# ============================================================
# 15. MASS FUNCTION
# ============================================================

def mass_function(P_days, K_kms):
    """
    Binary mass function in solar masses.

    f(M) = P K^3 / (2 pi G)
    """
    G = 6.67430e-11
    M_sun = 1.98847e30
    P_seconds = P_days * 86400.0
    K_ms = K_kms * 1000.0

    f_kg = P_seconds * K_ms**3 / (2 * np.pi * G)
    return f_kg / M_sun


def estimate_companion_mass(f_mass, primary_mass, inclination_deg=90.0):
    """
    Solve f(M) = M2^3 sin^3(i) / (M1 + M2)^2 for M2.

    Returns companion mass in solar masses.
    """
    if f_mass <= 0 or primary_mass <= 0:
        raise ValueError("Mass function and primary mass must be positive.")

    sini = np.sin(np.radians(inclination_deg))

    if sini <= 0:
        raise ValueError("Inclination must be greater than 0 degrees.")

    def equation(m2):
        return m2**3 * sini**3 / (primary_mass + m2)**2 - f_mass

    upper = max(primary_mass, 1.0)

    while equation(upper) < 0:
        upper *= 2

        if upper > 1e6:
            raise RuntimeError("Could not bracket companion-mass solution.")

    return brentq(equation, 1e-12, upper)


def print_period_fit_summary(P, popt, perr=None, primary_mass=None, inclination_deg=90.0):
    """Print fitted orbital parameters and optional companion mass."""
    gamma, K, phi = popt
    f_mass = mass_function(P, K)

    print("\n" + "=" * 60)
    print("PERIOD / ORBITAL FIT SUMMARY")
    print("=" * 60)
    print(f"Period       : {P:.6f} days")
    print(f"Systemic RV  : {gamma:.3f} km/s")
    print(f"RV semiamp.  : {K:.3f} km/s")
    print(f"Phase        : {phi:.4f} rad")
    print(f"Mass function: {f_mass:.6f} M_sun")

    if perr is not None:
        print(f"Gamma error  : {perr[0]:.3f} km/s")
        print(f"K error      : {perr[1]:.3f} km/s")
        print(f"Phase error  : {perr[2]:.4f} rad")

    if primary_mass is not None:
        companion = estimate_companion_mass(f_mass, primary_mass, inclination_deg)
        print(f"Primary mass : {primary_mass:.3f} M_sun")
        print(f"Inclination  : {inclination_deg:.1f} deg")
        print(f"Companion min mass: {companion:.3f} M_sun")

    print("=" * 60 + "\n")


# ============================================================
# 16. CONVENIENCE RV PIPELINE
# ============================================================

def run_rv_pipeline(specs, line_dict, window_dict=None, emission=False, period_range=(2, 2000), samples_per_peak=10, verbose=True):
    """
    Run the complete RV analysis after spectra have been normalized.

    Returns:
        results, period_data, fit_data
    """
    results = measure_rv_all_epochs(specs, line_dict, window_dict, emission, verbose)

    if len(results) < 3:
        raise ValueError("At least three valid epochs are required.")

    df = results_to_dataframe(results)
    t = df["mjd"].to_numpy()
    rv = df["rv_median"].to_numpy()
    err = df["rv_err"].to_numpy()

    good_err = np.isfinite(err) & (err > 0)
    fallback_err = np.nanmedian(err[good_err]) if np.any(good_err) else 1.0
    err = np.where(good_err, err, fallback_err)

    periods, power, best_period, frequency, ls = lomb_scargle_period(t, rv, err, fmin=1 / period_range[1], fmax=1 / period_range[0], npp=samples_per_peak)

    popt, perr, pcov = fit_fixed_period(t, rv, err, best_period)

    fit_data = {"period": best_period, "popt": popt, "perr": perr, "pcov": pcov}

    return results, {"periods": periods, "power": power, "frequency": frequency, "ls": ls, "best_period": best_period}, fit_data

