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
# CORE GAUSSIAN MODEL
# ============================================================

def gaussian(x, amp, mu, sigma, cont):
    """Gaussian spectral-line profile."""
    return cont + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def doppler_rv(lambda_obs, lambda_rest):
    """Classical Doppler RV in km/s."""
    return c_kms * (lambda_obs - lambda_rest) / lambda_rest


# ============================================================
# SINGLE-LINE RV MEASUREMENT
# ============================================================

def measure_rv_epoch(wave, flux, lambda_rest, window=15.0,
                     emission=False, return_fit=False,
                     min_points=20, min_depth=0.01,
                     min_sigma=0.25, max_sigma=5.0,
                     mu_guard=3.0, max_rv_err=30.0,
                     expected_rv=0.0):
    """
    Fit one Gaussian line profile and return RV and RV uncertainty.

    Parameters
    ----------
    wave, flux    : array-like  normalised spectrum
    lambda_rest   : float       rest wavelength [AA]
    window        : float       half-window width [AA]
    emission      : bool        True for emission lines
    return_fit    : bool        if True, return (rv, rv_err, lmfit_result)
    min_points    : int         minimum pixels inside window
    min_depth     : float       minimum line depth (fraction of continuum)
    min_sigma     : float       minimum allowed Gaussian sigma [AA]
    max_sigma     : float       maximum allowed Gaussian sigma [AA]
    mu_guard      : float       max allowed centre shift from expected [AA]
    max_rv_err    : float       reject fit if RV uncertainty > this [km/s]
    expected_rv   : float       prior on systemic RV used to centre mu bounds

    Returns
    -------
    (rv, rv_err)                        if return_fit=False
    (rv, rv_err, lmfit.ModelResult)     if return_fit=True
    NaN values are returned on failure.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    mask = (
        (wave >= lambda_rest - window) &
        (wave <= lambda_rest + window) &
        np.isfinite(wave) &
        np.isfinite(flux)
    )

    if np.sum(mask) < min_points:
        out = (np.nan, np.nan, None)
        return out if return_fit else out[:2]

    x, y = wave[mask], flux[mask]

    # Initial parameter guesses
    if emission:
        idx   = np.argmax(y)
        mu0   = x[idx]
        amp0  = max(y[idx] - 1.0, min_depth)
        amp_min, amp_max = min_depth, max(1.0, amp0 * 5)
        depth = np.nanmax(y) - 1.0
    else:
        idx   = np.argmin(y)
        mu0   = x[idx]
        amp0  = min(y[idx] - 1.0, -min_depth)
        amp_min, amp_max = min(-min_depth, amp0 * 5), -min_depth
        depth = 1.0 - np.nanmin(y)

    if not np.isfinite(depth) or depth < min_depth:
        out = (np.nan, np.nan, None)
        return out if return_fit else out[:2]

    # Centre bounds: guard against locking onto a neighbouring line
    mu_expected = lambda_rest * (1.0 + expected_rv / c_kms)
    mu_min      = mu_expected - mu_guard
    mu_max      = mu_expected + mu_guard
    mu_initial  = np.clip(mu0, mu_min + 1e-4, mu_max - 1e-4)

    model  = Model(gaussian)
    params = model.make_params(
        amp=amp0,
        mu=mu_initial,
        sigma=0.8,
        cont=1.0,
    )
    params["amp"].set(min=amp_min,  max=amp_max)
    params["mu"].set( min=mu_min,   max=mu_max)
    params["sigma"].set(min=min_sigma, max=max_sigma)
    params["cont"].set( min=0.5,    max=1.5)

    try:
        result = model.fit(y, params, x=x)
    except Exception:
        result = None

    if result is None or not result.success:
        out = (np.nan, np.nan, result)
        return out if return_fit else out[:2]

    mu     = result.params["mu"].value
    mu_err = result.params["mu"].stderr
    rv     = doppler_rv(mu, lambda_rest)
    rv_err = np.nan if mu_err is None else abs(c_kms * mu_err / lambda_rest)

    if np.isfinite(rv_err) and rv_err > max_rv_err:
        rv, rv_err = np.nan, np.nan

    out = (rv, rv_err, result)
    return out if return_fit else out[:2]


def plot_line_fit(wave, flux, lambda_rest, result, window=15.0,
                  title="Line Fit", save_path=None, ax=None):
    """Plot one Gaussian fit, optionally on an existing subplot."""
    wave = np.asarray(wave)
    flux = np.asarray(flux)

    mask = (
        (wave >= lambda_rest - window)
        & (wave <= lambda_rest + window)
        & np.isfinite(wave)
        & np.isfinite(flux)
    )
    x, y = wave[mask], flux[mask]

    if x.size == 0:
        raise ValueError("No data inside line-fitting window.")

    if ax is None:
        fig, axis = plt.subplots(figsize=(10, 6))
    else:
        axis = ax
        fig = axis.figure

    xfine = np.linspace(x.min(), x.max(), 300)
    yfit = result.model.eval(result.params, x=xfine)

    axis.plot(x, y, "k-", lw=0.8, label="Spectrum")
    axis.plot(xfine, yfit, "r-", lw=1.2, label="Gaussian fit")
    axis.axvline(
        lambda_rest, color="gray", ls="--",
        lw=0.7, alpha=0.5, label="Rest wavelength",
    )
    axis.axhline(1, color="gray", ls=":", lw=0.7, alpha=0.5)
    axis.set_xlabel("Wavelength [Å]")
    axis.set_ylabel("Normalized flux")
    axis.set_title(title)
    axis.grid(alpha=0.3)

    # Avoid adding a legend and applying layout for every grid panel.
    if ax is None:
        axis.legend()
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axis

# ============================================================
# SELECTED-LINE EPOCH WORKFLOW
# Uses the existing measure_rv_epoch; no second RV fitter.
# ============================================================

def measure_epoch(
    wave,
    flux,
    filename,
    bjd,
    *,
    primary_lines,
    secondary_lines,
    expected_rv,
    min_lines_for_mean,
    special_params=None,
    verbose=True,
):
    """Measure selected lines and preserve the notebook's group statistics.

    Line dictionaries:
        line name -> (rest wavelength [Angstrom], half-window [Angstrom])

    Preserves:
    - existing single-line fitting implementation;
    - max_sigma and mu_guard per-line settings;
    - equal-weight group and combined means;
    - combined sample standard deviation;
    - existing returned dictionary keys.

    Group names are legacy labels, not stellar identifications.
    Does not read files, plot or save outputs.
    """
    lines = {**primary_lines, **secondary_lines}
    special_params = {} if special_params is None else special_params

    rvs = {}
    errors = {}
    fits_by_line = {}

    for line_name, (lambda_rest, window) in lines.items():
        settings = special_params.get(line_name, {})

        try:
            rv, rv_err, result = measure_rv_epoch(
                wave,
                flux,
                lambda_rest,
                window=window,
                return_fit=True,
                expected_rv=expected_rv,
                max_sigma=settings.get("max_sigma", 5.0),
                mu_guard=settings.get("mu_guard", 3.0),
            )
        except Exception as exc:
            if verbose:
                print(f"  {line_name}: fitting failed: {exc}")

            rv, rv_err, result = np.nan, np.nan, None

        rvs[line_name] = (
            float(rv)
            if rv is not None and np.isfinite(rv)
            else np.nan
        )
        errors[line_name] = (
            float(rv_err)
            if rv_err is not None and np.isfinite(rv_err)
            else np.nan
        )
        fits_by_line[line_name] = result

        if verbose:
            if np.isfinite(rvs[line_name]):
                suffix = (
                    f" +/- {errors[line_name]:.2f}"
                    if np.isfinite(errors[line_name])
                    else " (uncertainty unavailable)"
                )
                print(
                    f"  {line_name:14s}: "
                    f"{rvs[line_name]:8.2f} km/s" + suffix
                )
            else:
                print(f"  {line_name:14s}: FAILED")

    primary_values = [
        rvs[name]
        for name in primary_lines
        if np.isfinite(rvs[name])
    ]
    secondary_values = [
        rvs[name]
        for name in secondary_lines
        if np.isfinite(rvs[name])
    ]
    candidate_values = [
        rvs[name]
        for name in lines
        if np.isfinite(rvs[name])
    ]

    if len(candidate_values) >= min_lines_for_mean:
        mean_rv = float(np.mean(candidate_values))
        line_scatter = (
            float(np.std(candidate_values, ddof=1))
            if len(candidate_values) > 1
            else np.nan
        )
    else:
        mean_rv = np.nan
        line_scatter = np.nan

    return {
        "filename": filename,
        "BJD": bjd,
        "wave": wave,
        "flux": flux,
        "rvs": rvs,
        "errors": errors,
        "fits": fits_by_line,
        "mean_RV_kms": mean_rv,
        "line_scatter_kms": line_scatter,
        "N_mean_lines": len(candidate_values),
        "primary_mean_RV_kms": (
            float(np.mean(primary_values))
            if primary_values else np.nan
        ),
        "N_primary_lines": len(primary_values),
        "secondary_mean_RV_kms": (
            float(np.mean(secondary_values))
            if secondary_values else np.nan
        ),
        "N_secondary_lines": len(secondary_values),
    }


# ============================================================
# VELOCITY-SPACE GAUSSIAN
# Distinct from the existing wavelength-space gaussian().
# ============================================================

def gaussian_vel(v, continuum, slope, depth, centre, width):
    """Absorption Gaussian in velocity space with a linear continuum.

    v, centre and width: km/s.
    width: Gaussian sigma, not FWHM.
    depth: positive for absorption.
    slope: uses the original v / 200 scaling.
    """
    return (
        continuum
        + slope * v / 200.0
        - depth * np.exp(
            -0.5 * ((v - centre) / width) ** 2
        )
    )


# ============================================================
# PROFILE REGISTRATION
# Preserves the notebook's registration fitting procedure.
# ============================================================

def fit_registration_profile(
    wave,
    flux,
    initial,
    *,
    rest_a,
    shift_kms,
    view_half,
    fit_half,
    cont_bounds,
    local_continuum,
):
    """Fit one profile for apparent velocity registration.

    Parameters
    ----------
    wave, flux
        Wavelength [Angstrom] and supplied normalized flux.
    initial
        Initial RV [km/s], also used to centre the selection windows.
    rest_a
        Rest wavelength in the same convention as wave.
    shift_kms
        Explicit additive velocity correction.
    view_half, fit_half
        Viewing and fitting half-ranges [km/s].
    cont_bounds
        Inner and outer continuum-sideband offsets [km/s].
    local_continuum
        Whether to divide by a clipped linear sideband continuum.

    Returns
    -------
    dict
        v, flux, RV, bound_hit and parameters.

    File identifiers, BJD, epoch labels and colours are supplied separately
    by the calling notebook.

    Returned RV is a registration diagnostic, not a separated stellar RV.
    Does not load spectra, infer wavelength conventions, plot or save.
    """
    if not np.isfinite(initial) or not np.isfinite(shift_kms):
        raise ValueError(
            "Invalid initial RV or velocity correction."
        )

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    v = c_kms * (wave / rest_a - 1.0) + shift_kms

    extraction_half = max(
        view_half + abs(initial) + 150.0,
        cont_bounds[1] + abs(initial),
    )

    use = (
        np.isfinite(v)
        & np.isfinite(flux)
        & (np.abs(v) <= extraction_half)
    )
    v, f = v[use], flux[use]

    side = (
        (np.abs(v - initial) >= cont_bounds[0])
        & (np.abs(v - initial) <= cont_bounds[1])
    )

    if (
        side.sum() < 10
        or not np.any(v[side] < initial)
        or not np.any(v[side] > initial)
    ):
        raise ValueError(
            "Insufficient continuum on both sides."
        )

    continuum = np.ones_like(f)

    if local_continuum:
        keep = side.copy()
        x_local = (v - initial) / 200.0

        for _ in range(3):
            coef = np.polyfit(
                x_local[keep],
                f[keep],
                1,
            )
            continuum = np.polyval(coef, x_local)
            residual = f - continuum

            centre_resid = np.median(residual[keep])
            scatter = 1.4826 * np.median(
                np.abs(residual[keep] - centre_resid)
            )

            if not np.isfinite(scatter) or scatter <= 0:
                break

            updated = side & (
                np.abs(residual - centre_resid)
                < 3.0 * scatter
            )

            if (
                updated.sum() < 10
                or not np.any(v[updated] < initial)
                or not np.any(v[updated] > initial)
            ):
                break

            keep = updated

        continuum = np.polyval(
            np.polyfit(x_local[keep], f[keep], 1),
            x_local,
        )

    if np.any(continuum <= 0):
        raise ValueError("Invalid continuum.")

    f = f / continuum

    fit_mask = np.abs(v - initial) <= fit_half

    if fit_mask.sum() < 15:
        raise ValueError("Too few Gaussian-fit pixels.")

    vv, ff = v[fit_mask], f[fit_mask]

    depth0 = float(
        np.clip(
            1.0 - np.percentile(ff, 5),
            0.01,
            0.8,
        )
    )

    lower = np.array([
        0.85,
        -0.15,
        0.001,
        initial - 100.0,
        8.0,
    ])
    upper = np.array([
        1.15,
        0.15,
        0.95,
        initial + 100.0,
        150.0,
    ])

    solutions = []

    for width0 in (25.0, 50.0, 90.0):
        try:
            pars, _ = curve_fit(
                gaussian_vel,
                vv,
                ff,
                p0=[
                    1.0,
                    0.0,
                    depth0,
                    initial,
                    width0,
                ],
                bounds=(lower, upper),
                maxfev=30000,
            )

            rss = np.sum(
                (ff - gaussian_vel(vv, *pars)) ** 2
            )
            solutions.append((rss, pars))

        except (RuntimeError, ValueError):
            continue

    if not solutions:
        raise RuntimeError(
            "All Gaussian initialisations failed."
        )

    _, pars = min(
        solutions,
        key=lambda item: item[0],
    )

    boundary_distance = np.minimum(
        (pars - lower) / (upper - lower),
        (upper - pars) / (upper - lower),
    )

    return {
        "v": v,
        "flux": f,
        "RV": float(pars[3]),
        "bound_hit": bool(
            np.any(boundary_distance < 0.001)
        ),
        "parameters": pars,
    }


# ============================================================
# UNWEIGHTED LINEAR TRIAL-PERIOD FIT
# Distinct from the existing weighted fit_fixed_period().
# ============================================================

def fit_trial_period(t, rv, period):
    """Fit a sinusoid at a fixed trial period by linear least squares.

    Inputs
    ------
    t
        Finite times [days], preferably relative to an explicit reference.
    rv
        Paired finite RV values [km/s].
    period
        Positive trial period [days].

    Returns
    -------
    rss
        Residual sum of squares; infinity for a rank-deficient design.
    coefficients
        [offset, cosine coefficient, sine coefficient].
        NOT the [gamma, K, phi] returned by fit_fixed_period().
    predicted
        Model at the supplied times; NaNs for a rank-deficient design.

    Uses no measurement weights, amplitude bounds, period uncertainty
    or significance calculation. Does not read, scan, plot or save.
    """
    t = np.asarray(t, dtype=float)
    rv = np.asarray(rv, dtype=float)

    if t.ndim != 1 or rv.ndim != 1 or t.size != rv.size:
        raise ValueError(
            "t and rv must be paired one-dimensional arrays."
        )

    if (
        not np.all(np.isfinite(t))
        or not np.all(np.isfinite(rv))
    ):
        raise ValueError(
            "Select finite t/RV observations before fitting."
        )

    if not np.isfinite(period) or period <= 0:
        raise ValueError(
            "period must be finite and positive."
        )

    angle = 2.0 * np.pi * t / period

    design = np.column_stack((
        np.ones(len(t)),
        np.cos(angle),
        np.sin(angle),
    ))

    coefficients, _, rank, _ = np.linalg.lstsq(
        design,
        rv,
        rcond=None,
    )

    if rank < 3:
        return (
            np.inf,
            coefficients,
            np.full(len(rv), np.nan),
        )

    predicted = design @ coefficients
    rss = float(np.sum((rv - predicted) ** 2))

    return rss, coefficients, predicted

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


# ============================================================
# 17. EXAMPLE NORMALIZATION FUNCTION FOR ONE FITS FILE
# ============================================================

def normalize_fits_file_interactive(fits_file, output_dir, template_dir=None, image_dir=None):
    """Load one FITS spectrum and launch the interactive normalizer."""
    if template_dir is None:
        template_dir = os.path.join(output_dir, "templates")

    if image_dir is None:
        image_dir = os.path.join(output_dir, "images")

    wave, flux, mjd = load_fits_spectrum(fits_file)

    return normalize_spectrum_interactive(
        wave,
        flux,
        filename=os.path.basename(fits_file),
        output_dir=output_dir,
        template_dir=template_dir,
        image_dir=image_dir
    )


# ============================================================
# 18. MAIN -- OPTIONAL TXT NORMALIZATION MODE
# ============================================================

if __name__ == "__main__":
    print("toolkit.py loaded.")
    print("Import this module and call normalize_spectrum_interactive()")
    print("or normalize_fits_file_interactive() for interactive normalization.")
