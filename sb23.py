"""HD 195592 template utilities and explicitly LEGACY Gaussian experiments.

Save as s23.py beside hd195592_common.py. rvfuncs.py must contain gaussian_vel.

ACTIVE: template reading, illustrative overlays, finite-window EW screening.
LEGACY: per-line shared-shape 2G/3G and joint-He transfer/ambiguity experiments.
Legacy component labels are template labels, NOT established stellar identities.

Importing this module performs no file I/O, fitting, plotting, style changes or
output writing. Functions read inputs only when called. Plot functions return
figures; none calls show(), savefig(), mkdir() or any output writer.

Dependencies: numpy, pandas, matplotlib, scipy, astropy. rvfuncs is imported
lazily only when a legacy Gaussian model is evaluated. Its original import-time
side effects remain unchanged; they are not added to this module's import.

No automatic air/vacuum or barycentric correction, additional broadening,
orbit inference, bootstrap, disentangling or physical SB2-template fit is added.
All line selections, paths, epochs and configuration objects are explicit inputs.
"""
from dataclasses import dataclass
from pathlib import Path
from itertools import permutations
from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import least_squares, linear_sum_assignment
from hd195592_common import C_KMS, load_hermes, wavelength_to_velocity
from rvfuncs import gaussian_vel


@dataclass(frozen=True)
class EWConfig:
    integration_range: tuple = (-220.0, 120.0)
    continuum_ranges: tuple = ((-390.0, -320.0), (330.0, 390.0))
    test_ranges: tuple = ((-180.0, 100.0), (-220.0, 120.0), (-260.0, 140.0))
    neighbourhood: tuple = (-450.0, 450.0)
    plot_range: tuple = (-400.0, 400.0)
    clip_sigma: float = 3.0
    min_side_pixels: int = 8


@dataclass(frozen=True)
class LegacyGaussianConfig:
    local_continuum: bool = False
    gauss_fit_half: float = 160.0
    fit_half: float = 300.0
    rv_range: float = 200.0
    cont_bounds: tuple = (240.0, 340.0)
    search_half: float = 200.0
    additional_shift_kms: float = 0.0
    n_starts: int = 36
    seed: int = 35


@dataclass(frozen=True)
class LegacyJointConfig:
    train_starts: int = 36
    epoch_starts: int = 24
    max_nfev: int = 6000
    seed: int = 351026
    sigma_bounds: tuple = (8.0, 150.0)
    depth_max: float = 0.95
    cont_offset: float = 0.04
    cont_tilt: float = 0.04
    near_delta: float = 10.0
    max_train_branches: int = 3
    mode_distance: float = 1.0
    match_rv: float = 15.0
    match_sigma: float = 10.0
    match_depth: float = 0.025
    core_half: float = 150.0
    core_rms_limit: float = 1.3
    core_degradation: float = 0.15
    rv_span_limit: float = 25.0  # Retained setting; original runner did not apply it.
    bound_fraction: float = 0.001


# =====================================================================
# ACTIVE: READERS / CLEANING
# =====================================================================
def clean_spectrum(wave, flux, label="spectrum", *, min_points=2, verbose=True):
    """Filter, stable-sort and average duplicate wavelengths.

    This is the ONE template/EW cleaner. min_points=2 reproduces the template
    inspection requirement; min_points=3 reproduces the EW requirement.
    It is NOT applied implicitly to the common RV reader, which retains its
    original duplicate-handling behaviour.
    """
    wave, flux = np.asarray(wave, float).ravel(), np.asarray(flux, float).ravel()
    if wave.size != flux.size:
        raise ValueError(f"{label}: wavelength and flux lengths differ.")
    good = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    removed = int(wave.size - good.sum())
    wave, flux = wave[good], flux[good]
    if wave.size < min_points:
        raise ValueError(f"{label}: fewer than {min_points} valid samples.")
    sorted_before = bool(np.all(np.diff(wave) >= 0))
    order = np.argsort(wave, kind="stable")
    wave, flux = wave[order], flux[order]
    unique, first, counts = np.unique(wave, return_index=True, return_counts=True)
    duplicates = int(wave.size - unique.size)
    if duplicates:
        flux, wave = np.add.reduceat(flux, first) / counts, unique
    if verbose and (removed or duplicates or not sorted_before):
        print(f"{label}: removed {removed}; averaged {duplicates} duplicates; "
              f"originally sorted={sorted_before}")
    if wave.size < min_points or not np.all(np.diff(wave) > 0):
        raise ValueError(f"{label}: invalid wavelength grid after cleaning.")
    return wave, flux


def read_template(path, *, exact_columns=None, min_points=2, verbose=True):
    """Read wavelength/normalized flux; extra columns allowed unless specified."""
    path = Path(path)
    arr = np.loadtxt(path, comments="#", ndmin=2)
    if arr.shape[1] < 2 or (exact_columns is not None and arr.shape[1] != exact_columns):
        raise ValueError(f"{path.name}: unexpected number of columns ({arr.shape[1]}).")
    return clean_spectrum(arr[:, 0], arr[:, 1], path.name,
                          min_points=min_points, verbose=verbose)


def read_powr(path, *, wave_format="auto", verbose=True):
    """Read the supplied two-column PoWR format, including Fortran D exponents.

    auto is numerical inference of Angstrom/log10-Angstrom, NOT documentary
    verification. Flux is not converted from logarithmic/calibrated flux.
    """
    path, rows = Path(path), []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for number, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.startswith(("*", "#", "!", ";")):
                continue
            try:
                values = [float(s.replace("D", "E").replace("d", "e"))
                          for s in text.split()]
            except ValueError:
                continue
            if len(values) != 2:
                raise ValueError(f"Numeric row {number} has {len(values)} columns.")
            rows.append(values)
    if len(rows) < 3:
        raise ValueError(f"{path.name}: no two-column spectrum found.")
    arr = np.asarray(rows)
    x, flux = arr[:, 0], arr[:, 1]
    finite = x[np.isfinite(x)]
    if finite.size < 3:
        raise ValueError("Invalid wavelength column.")
    fmt = wave_format
    if fmt == "auto":
        if 2 < np.median(finite) < 6 and 0 < finite.min() and finite.max() < 8:
            fmt = "log10_angstrom"
        elif np.median(finite) > 1000:
            fmt = "angstrom"
        else:
            raise ValueError("Cannot infer wavelength units; set wave_format explicitly.")
        if verbose:
            print(f"{path.name}: numerically inferred {fmt}; verify documentation.")
    if fmt == "log10_angstrom":
        wave = 10.0 ** x
    elif fmt == "angstrom":
        wave = x
    else:
        raise ValueError("wave_format must be auto, angstrom or log10_angstrom.")
    wave, flux = clean_spectrum(wave, flux, path.name, min_points=3, verbose=verbose)
    if np.median(flux) <= 0:
        raise ValueError("Flux inconsistent with supplied normalized-flux assumption.")
    return wave, flux


def read_model(path, kind, *, powr_wave_format="auto", verbose=True):
    """EW reader: Shenar requires exactly two columns and at least 3 points."""
    if kind == "shenar":
        return read_template(path, exact_columns=2, min_points=3, verbose=verbose)
    if kind == "powr":
        return read_powr(path, wave_format=powr_wave_format, verbose=verbose)
    raise ValueError("kind must be shenar or powr.")


def read_hermes_inspection(path, *, verbose=True):
    """CTYPE-aware inspection reader; retains the supplied inspection policy.

    Distinct from common.load_hermes: accepts natural-log or explicitly linear
    CTYPE1, supports CD1_1 fallback and averages duplicate wavelengths.
    Wavelength units/convention are reported, NOT converted or inferred.
    """
    path = Path(path)
    with fits.open(path, mode="readonly", memmap=False) as hdul:
        header = hdul[0].header.copy()
        flux = np.asarray(hdul[0].data, float).squeeze()
    if flux.ndim != 1:
        raise ValueError(f"{path.name}: expected 1D primary-HDU flux.")
    if "CDELT1" in header:
        step = float(header["CDELT1"])
    elif "CD1_1" in header:
        step = float(header["CD1_1"])
    else:
        raise ValueError(f"{path.name}: missing CDELT1 / CD1_1.")
    crpix = float(header.get("CRPIX1", 1.0))
    coord = float(header["CRVAL1"]) + (np.arange(flux.size) + 1.0 - crpix) * step
    ctype = str(header.get("CTYPE1", "")).strip().lower()
    if ctype == "log(wavelength)":
        wave = np.exp(coord)
    elif ctype in {"wavelength", "awav", "wave"}:
        wave = coord
    else:
        raise ValueError(f"{path.name}: unsupported CTYPE1={ctype!r}; axis not guessed.")
    wave, flux = clean_spectrum(wave, flux, path.name, verbose=verbose)
    bjd = header.get("BJD", None)
    if verbose:
        print(f"{path.name}: CTYPE1={ctype!r}, CUNIT1={header.get('CUNIT1', 'unspecified')!r}, "
              f"CRPIX1={crpix}, BJD={bjd}")
    return wave, flux, bjd


def load_spectra(spectra_dir, *, pattern="*.fits"):
    """Legacy natural-log batch reader using common.load_hermes.

    Retains *.fits-only selection, abort-on-invalid-BJD, (BJD, filename) sort,
    and E-label assignment of the supplied read_all(). No duplicate averaging.
    """
    root, spectra = Path(spectra_dir), []
    for path in sorted(root.glob(pattern)):
        wave, flux, bjd = load_hermes(path)
        if not np.isfinite(bjd):
            raise ValueError(f"{path.name}: missing BJD.")
        spectra.append(dict(file=path.name, wave=wave, flux=flux, bjd=bjd))
    if not spectra:
        raise FileNotFoundError(f"No matching spectra in {root}")
    spectra.sort(key=itemgetter("bjd", "file"))
    for i, spectrum in enumerate(spectra, 1):
        spectrum["label"] = f"E{i:02d}"
    return spectra


def load_templates(model_specs, *, powr_wave_format="auto", verbose=True):
    """model_specs: iterable of (label, path, kind); returns (models, failures)."""
    models, failures = [], []
    for i, (label, path, kind) in enumerate(model_specs):
        try:
            wave, flux = read_model(path, kind, powr_wave_format=powr_wave_format,
                                    verbose=verbose)
            models.append(dict(label=label, filename=Path(path).name, wave=wave,
                               flux=flux, color=plt.get_cmap("tab10")(i),
                               kind=kind, observational=False))
        except (OSError, ValueError) as exc:
            failures.append(dict(label=label, path=str(path), error=str(exc)))
            if verbose:
                print(f"Could not load {label}: {exc}")
    return models, failures


def load_observations(obs_dir, obs_files, *, reader=read_hermes_inspection):
    """Explicit epoch->filename mapping; returns (datasets, failures).

    reader must return (wave, flux, BJD). Pass common.load_hermes explicitly
    if its natural-log-only assumptions are the intended policy.
    """
    datasets, failures = {}, []
    for i, (epoch, filename) in enumerate(obs_files.items()):
        try:
            wave, flux, bjd = reader(Path(obs_dir) / filename)
            datasets[epoch] = dict(label=epoch, filename=filename, wave=wave,
                flux=flux, bjd=bjd, observational=True, kind="observation",
                color="black" if i == 0 else "dimgray")
        except (OSError, ValueError, KeyError) as exc:
            failures.append(dict(epoch=epoch, file=filename, error=str(exc)))
    return datasets, failures


# =====================================================================
# ACTIVE: ILLUSTRATIVE TEMPLATE CONTRIBUTIONS / PLOTS
# =====================================================================
def check_wavelength_systems(template_system, obs_system, *, reference_system="air"):
    """Report conventions; never convert wavelengths or infer them from FITS."""
    allowed = {"unknown", "air", "vacuum"}
    if template_system not in allowed or obs_system not in allowed:
        raise ValueError("Wavelength systems must be unknown, air or vacuum.")
    if "unknown" in {template_system, obs_system}:
        print("WARNING: air/vacuum conventions unconfirmed; offsets are not measured RVs.")
    elif template_system != obs_system:
        raise ValueError("Different wavelength conventions; convert one explicitly.")
    if any(s != "unknown" and s != reference_system for s in (template_system, obs_system)):
        print("WARNING: configured reference wavelengths do not match all declared systems.")


def doppler_factor(rv_kms):
    """Relativistic template wavelength shift; NOT the classical velocity axis."""
    beta = float(rv_kms) / C_KMS
    if not np.isfinite(beta) or abs(beta) >= 1:
        raise ValueError("RV must be finite and smaller than c in magnitude.")
    return np.sqrt((1.0 + beta) / (1.0 - beta))


def template_contribution(model, query_wave, *, rv_kms=0.0, light_fraction=1.0):
    """Alternative single-star-plus-flat-continuum overlay, NOT an SB2 sum."""
    if not np.isfinite(light_fraction) or not 0 <= light_fraction <= 1:
        raise ValueError("light_fraction must lie between zero and one.")
    shifted = np.asarray(model["wave"]) * doppler_factor(rv_kms)
    intrinsic = np.interp(query_wave, shifted, model["flux"], left=np.nan, right=np.nan)
    return 1.0 + light_fraction * (intrinsic - 1.0)


def _inspection_lines(lines):
    return [(name, rest) for name, rest, *_ in lines]


def plot_template_profiles(models, lines, *, velocity_limits=(-400.0, 400.0)):
    items = _inspection_lines(lines)
    fig, axes = plt.subplots(1, len(items), figsize=(6.5 * len(items), 5),
                             sharey=True, squeeze=False)
    for ax, (name, rest) in zip(axes[0], items):
        for model in models:
            v = wavelength_to_velocity(model["wave"], rest)
            mask = (v >= velocity_limits[0]) & (v <= velocity_limits[1])
            if mask.sum() > 1:
                ax.plot(v[mask], model["flux"][mask], color=model["color"],
                        lw=1.4, label=model["label"])
        ax.axhline(1, color="0.6", lw=0.8, ls="--")
        ax.axvline(0, color="0.7", lw=0.8, ls=":")
        ax.set(xlim=velocity_limits, title=name,
               xlabel="Velocity relative to configured reference [km/s]")
        ax.grid(alpha=0.15)
    axes[0, 0].set_ylabel("Intrinsic normalised template flux")
    handles = [plt.Line2D([0], [0], color=m["color"], label=m["label"]) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Intrinsic templates: existing broadening retained")
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    return fig


def plot_template_comparison(models, observations, lines, *, velocity_limits=(-400., 400.),
                             rv_kms=0., light_fraction=1., rv_by_file=None,
                             fraction_by_file=None):
    if not observations:
        raise ValueError("No observations supplied.")
    items = _inspection_lines(lines)
    rv_by_file = {} if rv_by_file is None else rv_by_file
    fraction_by_file = {} if fraction_by_file is None else fraction_by_file
    fig, axes = plt.subplots(len(observations), len(items), squeeze=False,
                             figsize=(6.5 * len(items), 4 * len(observations)))
    for row, (epoch, obs) in enumerate(observations.items()):
        for col, (name, rest) in enumerate(items):
            ax = axes[row, col]
            v = wavelength_to_velocity(obs["wave"], rest)
            mask = (v >= velocity_limits[0]) & (v <= velocity_limits[1])
            ax.set_title(f"{epoch}: {name}")
            if mask.sum() < 2:
                ax.text(0.5, 0.5, "No observed coverage", ha="center", transform=ax.transAxes)
                continue
            query = obs["wave"][mask]
            ax.plot(v[mask], obs["flux"][mask], "k", lw=1.3, zorder=5)
            for model in models:
                contribution = template_contribution(model, query,
                    rv_kms=rv_by_file.get(model["filename"], rv_kms),
                    light_fraction=fraction_by_file.get(model["filename"], light_fraction))
                ax.plot(v[mask], contribution, color=model["color"], lw=1.1, alpha=0.8)
                if not np.isfinite(contribution).all():
                    print(f"WARNING: missing shifted coverage: {epoch}, {name}, {model['filename']}")
            ax.axhline(1, color="0.6", lw=0.8, ls="--")
            ax.axvline(0, color="0.7", lw=0.8, ls=":")
            ax.set(xlim=velocity_limits, ylabel="Normalised flux",
                   xlabel="Velocity relative to configured reference [km/s]")
            ax.grid(alpha=0.15)
    handles = [plt.Line2D([0], [0], color="black", label="Observed")]
    handles += [plt.Line2D([0], [0], color=m["color"], label=m["label"]) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Alternative template overlays — NOT an SB2 fit")
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    return fig


# =====================================================================
# ACTIVE: ONE EW IMPLEMENTATION FOR BOTH PREVIOUS EW BLOCKS
# =====================================================================
def _between(x, interval):
    return (x >= interval[0]) & (x <= interval[1])


def _mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def ew_continuum(wave, flux, rest, offset_kms, config):
    """Clipped linear continuum; positive offset moves selection windows redward."""
    wave, flux = clean_spectrum(wave, flux, min_points=3, verbose=False)
    v = wavelength_to_velocity(wave, rest) - offset_kms
    near = _between(v, config.neighbourhood)
    wave, flux, v = wave[near], flux[near], v[near]
    if wave.size < 3:
        raise ValueError("Line neighbourhood not covered.")
    sides = [_between(v, interval) for interval in config.continuum_ranges]
    if len(sides) != 2 or any(m.sum() < config.min_side_pixels for m in sides):
        raise ValueError("Insufficient coverage in the two continuum sidebands.")
    original, retained = sides[0] | sides[1], (sides[0] | sides[1]).copy()
    design = np.column_stack((np.ones_like(v), v / 400.0))
    for _ in range(8):
        coef = np.linalg.lstsq(design[retained], flux[retained], rcond=None)[0]
        residual = flux - design @ coef
        sigma = _mad_sigma(residual[retained])
        if not np.isfinite(sigma) or sigma <= 1e-12:
            break
        candidate = original & (np.abs(residual - np.median(residual[retained]))
                                <= config.clip_sigma * sigma)
        if any((candidate & side).sum() < config.min_side_pixels for side in sides):
            break
        if np.array_equal(candidate, retained):
            break
        retained = candidate
    coef = np.linalg.lstsq(design[retained], flux[retained], rcond=None)[0]
    continuum = design @ coef
    if np.any(continuum <= 0) or not np.isfinite(continuum).all():
        raise ValueError("Invalid fitted continuum.")
    residual = flux[retained] - continuum[retained]
    sigma = _mad_sigma(residual)
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = float(np.std(residual, ddof=2))
    covariance = sigma ** 2 * np.linalg.pinv(design[retained].T @ design[retained])
    return dict(wave=wave, flux=flux, v=v, continuum=continuum, design=design,
                sigma=sigma, covariance=covariance, retained=retained)


def ew_integral(local, interval, *, observational=False):
    """Finite-window EW and original formal error, with interpolated boundaries.

    Positive is net absorption. Formal errors assume independent original
    pixels and linearized continuum coefficients; correlated noise, blends,
    cross-line covariance and systematic continuum errors are not included.
    """
    wave, flux, v = local["wave"], local["flux"], local["v"]
    continuum, design = local["continuum"], local["design"]
    lo, hi = interval
    if not v[0] <= lo < hi <= v[-1]:
        raise ValueError("Integration interval not covered.")
    mask = (v > lo) & (v < hi)
    if mask.sum() < 2:
        raise ValueError("Too few integration points.")
    boundaries = np.interp([lo, hi], v, wave)
    iw = np.r_[boundaries[0], wave[mask], boundaries[1]]
    ff, cc = np.interp(iw, wave, flux), np.interp(iw, wave, continuum)
    dd = np.column_stack((np.ones_like(iw), np.interp(iw, wave, design[:, 1])))
    delta = np.diff(iw)
    weights = np.empty(iw.size)
    weights[0], weights[-1] = delta[0] / 2, delta[-1] / 2
    weights[1:-1] = (delta[:-1] + delta[1:]) / 2
    value, error = float(np.sum(weights * (1 - ff / cc))), np.nan
    if observational:
        right = np.clip(np.searchsorted(wave, iw, side="right"), 1, wave.size - 1)
        left = right - 1
        t = (iw - wave[left]) / (wave[right] - wave[left])
        pixel_gradient = np.zeros(wave.size)
        np.add.at(pixel_gradient, left, -weights * (1 - t) / cc)
        np.add.at(pixel_gradient, right, -weights * t / cc)
        var_pixel = local["sigma"] ** 2 * np.sum(pixel_gradient ** 2)
        gradient = np.sum((weights * ff / cc ** 2)[:, None] * dd, axis=0)
        var_cont = float(gradient @ local["covariance"] @ gradient)
        error = float(np.sqrt(max(0.0, var_pixel + var_cont)))
    return value, error


def screen_ews(datasets, lines, config, *, offsets_kms=None, verbose=True):
    """datasets: label->dict(wave, flux, observational, optional color/kind).

    lines: name->rest wavelength. Returns results, local_profiles, failures.
    All models/observations use the same requested continuum/window settings.
    """
    offsets_kms = {} if offsets_kms is None else offsets_kms
    results, profiles, failures = {}, {}, []
    for label, data in datasets.items():
        results[label] = {}
        for name, rest in lines.items():
            try:
                local = ew_continuum(data["wave"], data["flux"], rest,
                                     offsets_kms.get(label, 0.0), config)
                value, error = ew_integral(local, config.integration_range,
                                            observational=data["observational"])
                tests = [ew_integral(local, interval,
                         observational=data["observational"])[0]
                         for interval in config.test_ranges]
                spread = max(tests) - min(tests)
                results[label][name] = dict(EW_A=value, formal_error_A=error,
                    test_EWs_A=tests, window_spread_A=spread,
                    continuum_sigma=local["sigma"])
                profiles[label, name] = local
                if verbose and value and spread / abs(value) > 0.10:
                    print(f"CAUTION: {label}, {name}: >10% EW sensitivity to integration limits.")
            except (ValueError, np.linalg.LinAlgError) as exc:
                failures.append(dict(dataset=label, line=name, error=str(exc)))
                if verbose:
                    print(f"Could not measure {label}, {name}: {exc}")
    return results, profiles, failures


def ew_ratio(first, second):
    """Positive-EW ratio and formal propagated error; cross-line covariance omitted."""
    a, b = first["EW_A"], second["EW_A"]
    if a <= 0 or b <= 0:
        return np.nan, np.nan
    ea, eb = first["formal_error_A"], second["formal_error_A"]
    value = a / b
    error = value * np.sqrt((ea / a) ** 2 + (eb / b) ** 2) if np.isfinite(ea) and np.isfinite(eb) else np.nan
    return value, error


def ew_table(results, first_line, second_line):
    rows = []
    for label, result in results.items():
        if first_line not in result or second_line not in result:
            continue
        a, b = result[first_line], result[second_line]
        ratio, error = ew_ratio(a, b)
        rows.append(dict(spectrum=label, first_EW_A=a["EW_A"],
            first_error_A=a["formal_error_A"], second_EW_A=b["EW_A"],
            second_error_A=b["formal_error_A"], ratio=ratio, ratio_error=error))
    return pd.DataFrame(rows)


def observed_template_ratios(results, datasets, lines):
    """Diagnostic EW ratios, NOT fitted light fractions or binary parameters."""
    rows = []
    for epoch, data in datasets.items():
        if not data["observational"]:
            continue
        for label, model in datasets.items():
            if model["observational"]:
                continue
            for name in lines:
                if name in results[epoch] and name in results[label]:
                    denom = results[label][name]["EW_A"]
                    if denom > 0:
                        rows.append(dict(epoch=epoch, template=label, line=name,
                            observed_over_template=results[epoch][name]["EW_A"] / denom))
    return pd.DataFrame(rows)


def plot_ew_screen(datasets, profiles, results, lines, config):
    if len(lines) != 2:
        raise ValueError("The EW-plane display requires exactly two lines.")
    names = list(lines)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, name in zip(axes[:2], names):
        for i, (label, data) in enumerate(datasets.items()):
            if (label, name) not in profiles:
                continue
            p = profiles[label, name]
            ax.plot(p["v"], p["flux"] / p["continuum"],
                color=data.get("color", plt.get_cmap("tab10")(i)),
                ls="--" if label == "E13" else "-",
                lw=1.6 if data["observational"] or data.get("kind") == "powr" else 1,
                alpha=1 if data["observational"] else 0.8, label=label)
        ax.axvspan(*config.integration_range, color="darkorange", alpha=0.06)
        ax.axhline(1, color="0.7", lw=0.8, ls=":")
        ax.axvline(0, color="0.7", lw=0.8, ls=":")
        ax.set(xlim=config.plot_range, title=f"{name}: locally normalised",
               xlabel="Velocity relative to configured reference [km/s]", ylabel="Flux")
        ax.grid(alpha=0.12)
    ax = axes[2]
    for i, (label, data) in enumerate(datasets.items()):
        if not all(name in results[label] for name in names):
            continue
        a, b = (results[label][name] for name in names)
        color = data.get("color", plt.get_cmap("tab10")(i))
        if data["observational"]:
            ax.errorbar(a["EW_A"], b["EW_A"], xerr=a["formal_error_A"],
                        yerr=b["formal_error_A"], fmt="*", color=color, ms=12, capsize=3)
        else:
            ax.scatter(a["EW_A"], b["EW_A"], color=color, s=55,
                       marker="o" if data.get("kind") == "powr" else "s")
        ax.annotate(label, (a["EW_A"], b["EW_A"]), xytext=(5, 6),
                    textcoords="offset points", fontsize=8, color=color)
    ax.set(xlabel=f"Finite-window W({names[0]}) [Å]",
           ylabel=f"Finite-window W({names[1]}) [Å]", title="EW strengths: models versus observations")
    ax.margins(0.20)
    ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Template EW screening — NOT a binary fit")
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    return fig


def plot_ew_diagnostics(datasets, profiles, results, lines, config):
    fig, axes = plt.subplots(len(lines), len(datasets), squeeze=False,
                             figsize=(5 * len(datasets), 4 * len(lines)))
    for col, label in enumerate(datasets):
        for row, name in enumerate(lines):
            ax = axes[row, col]
            ax.set_title(f"{label}: {name}")
            if (label, name) not in profiles:
                ax.text(0.5, 0.5, "Measurement unavailable", transform=ax.transAxes, ha="center")
                continue
            p = profiles[label, name]
            v, raw, cont, retained = p["v"], p["flux"], p["continuum"], p["retained"]
            ax.plot(v, raw, color="0.65", lw=1, label="Input flux")
            ax.plot(v, raw / cont, "k", lw=1.2, label="Locally normalised")
            ax.plot(v, cont, color="royalblue", ls="--", label="Fitted continuum")
            ax.scatter(v[retained], raw[retained], s=7, color="royalblue", alpha=0.45)
            ax.axvspan(*config.integration_range, color="darkorange", alpha=0.12, label="EW interval")
            for interval in config.continuum_ranges:
                ax.axvspan(*interval, color="royalblue", alpha=0.08)
            ax.axhline(1, color="0.5", lw=0.8, ls=":")
            ax.axvline(0, color="0.7", lw=0.8, ls=":")
            ax.set(xlim=config.plot_range, xlabel="Velocity minus window offset [km/s]",
                   ylabel="Flux", title=f"{label}: {name}\nW = {results[label][name]['EW_A']:.4f} Å")
            ax.grid(alpha=0.12)
    handles, labels = [], []
    for ax in axes.flat:
        hh, ll = ax.get_legend_handles_labels()
        for h, label in zip(hh, ll):
            if label not in labels:
                handles.append(h)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Finite-window EWs: continuum and integration diagnostics")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return fig


# =====================================================================
# LEGACY: SHARED OPTIMISATION UTILITIES / VELOCITY GAUSSIAN REUSE
# =====================================================================
def near_bound(p, lo, hi, *, fraction=0.001, return_mask=False):
    p, lo, hi = np.asarray(p), np.asarray(lo), np.asarray(hi)
    mask = np.minimum((p - lo) / (hi - lo), (hi - p) / (hi - lo)) < fraction
    return mask if return_mask else bool(np.any(mask))


def solve_multistart(residual, starts, lo, hi, *, max_nfev=50000, require_success=False):
    """One least_squares driver for both legacy workflows.

    require_success=False retains the original per-line script's acceptance
    of finite-chi2 nonconverged fits (success is still reported).
    require_success=True retains the joint script's converged-only policy.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    solutions, failed = [], 0
    for start in starts:
        eps = 1e-9 if require_success else 1e-8
        q0 = np.clip(start, lo + eps * (hi - lo), hi - eps * (hi - lo))
        try:
            fit = least_squares(residual, q0, bounds=(lo, hi),
                                x_scale="jac", max_nfev=max_nfev)
        except (ValueError, FloatingPointError):
            failed += 1
            continue
        if require_success and (not fit.success or not np.isfinite(fit.fun).all()):
            failed += 1
            continue
        chi = float(np.sum(fit.fun ** 2))
        if not require_success and not np.isfinite(chi):
            failed += 1
            continue
        solutions.append(dict(chi2=chi, theta=fit.x.copy(),
                              success=bool(fit.success), nfev=int(fit.nfev)))
    if not solutions:
        raise RuntimeError(f"All optimisation starts failed ({failed}).")
    solutions.sort(key=itemgetter("chi2"))
    return dict(best=solutions[0], solutions=solutions, bounds=(lo, hi),
                failed_starts=failed, attempted_starts=len(starts))


def gaussian_terms(v, shape, rv):
    """Component absorption deficits; reuses rvfuncs.gaussian_vel by broadcasting."""
    s = np.asarray(shape).reshape(-1, 2)
    return -gaussian_vel(np.asarray(v)[None, :], 0.0, 0.0,
                         s[:, 0, None], np.asarray(rv)[:, None], s[:, 1, None])


def curve(v, shape, rv):
    """Legacy constant-unit-continuum multi-Gaussian model."""
    return 1.0 - gaussian_terms(v, shape, rv).sum(axis=0)


def normalise_locally(v, f, centre, cont_bounds):
    """Legacy 3-pass / v-over-200 continuum, distinct from the 8-pass EW fit."""
    side = (np.abs(v - centre) >= cont_bounds[0]) & (np.abs(v - centre) <= cont_bounds[1])
    def adequate(mask):
        return mask.sum() >= 10 and np.any(v[mask] < centre) and np.any(v[mask] > centre)
    if not adequate(side):
        raise ValueError("Insufficient continuum on both sides.")
    x, keep = (v - centre) / 200.0, side.copy()
    for _ in range(3):
        residual = f - np.polyval(np.polyfit(x[keep], f[keep], 1), x)
        mid, scatter = np.median(residual[keep]), _mad_sigma(residual[keep])
        if not np.isfinite(scatter) or scatter <= 0:
            break
        new = side & (np.abs(residual - mid) < 3 * scatter)
        if not adequate(new):
            break
        keep = new
    continuum = np.polyval(np.polyfit(x[keep], f[keep], 1), x)
    if np.any(continuum <= 0):
        raise ValueError("Invalid local continuum.")
    return f / continuum


def measure_profile(spectrum, rest, config):
    """LEGACY least_squares centroid + 1G baseline; not the RV registration fitter.

    Reuses gaussian_vel but retains this experiment's smoothing, least_squares
    centroid method, noise estimation, window and 1G-baseline procedure.
    """
    v = wavelength_to_velocity(spectrum["wave"], rest,
                                shift_kms=config.additional_shift_kms)
    f = np.asarray(spectrum["flux"])
    near = np.abs(v) <= config.search_half
    if near.sum() < 20:
        raise ValueError("Line outside coverage.")
    smooth = np.convolve(f, np.ones(9) / 9, mode="same")
    start = v[near][np.argmin(smooth[near])]
    use = np.abs(v - start) <= max(config.fit_half + 120, config.cont_bounds[1] + 120)
    v, f = v[use], f[use]
    if config.local_continuum:
        f = normalise_locally(v, f, start, config.cont_bounds)
    mask = np.abs(v - start) <= config.gauss_fit_half
    if mask.sum() < 15:
        raise ValueError("Too few centroid-fit pixels.")
    d0 = float(np.clip(1 - np.percentile(f[mask], 5), 0.01, 0.8))
    sols = solve_multistart(lambda q: gaussian_vel(v[mask], *q) - f[mask],
        [[1, 0, d0, start, w] for w in (25, 50, 90)],
        [0.85, -0.15, 0.001, start - 100, 8],
        [1.15, 0.15, 0.95, start + 100, 150])
    rv = float(sols["best"]["theta"][3])
    dist = np.abs(v - rv)
    side = (dist >= config.cont_bounds[0]) & (dist <= config.cont_bounds[1])
    win = dist <= config.fit_half
    if side.sum() < 10 or win.sum() < 20:
        raise ValueError("Insufficient fit window or noise sidebands.")
    noise = float(np.std(f[side], ddof=1))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Invalid noise estimate.")
    p = dict(label=spectrum["label"], bjd=spectrum["bjd"],
             v=v[win], f=f[win], rv=rv, noise=noise)
    d0 = float(np.clip(1 - np.percentile(p["f"], 5), 0.01, 0.8))
    single = solve_multistart(lambda q: (curve(p["v"], q[:2], q[2:]) - p["f"]) / noise,
        [[d0, w, rv] for w in (35, 60, 100)],
        [0, 8, rv - config.rv_range], [0.95, 150, rv + config.rv_range])
    p["chi2_1g"] = single["best"]["chi2"]
    return p


def train_model(profiles, nc, config, shape_seed=None):
    """LEGACY per-line shared shapes; components ordered by depth*sigma area."""
    if nc not in (2, 3) or not profiles:
        raise ValueError("Require profiles and nc=2 or 3.")
    n, ns = len(profiles), 2 * nc
    centres = np.array([p["rv"] for p in profiles])
    lo = np.r_[np.tile([0, 8], nc), np.repeat(centres - config.rv_range, nc)]
    hi = np.r_[np.tile([0.95, 150], nc), np.repeat(centres + config.rv_range, nc)]
    rng = np.random.default_rng(config.seed)
    depth = float(np.clip(max(1 - np.percentile(p["f"], 5) for p in profiles), 0.05, 0.8))
    starts = []
    for j in range(config.n_starts):
        if nc == 2:
            widths = ((45, 45), (35, 65), (65, 35))[j % 3]
            frac = (0.35, 0.5, 0.65)[(j // 3) % 3]
            shape = np.array([[depth * frac, widths[0]], [depth * (1 - frac), widths[1]]])
            sep = np.zeros(n) if j % 5 == 0 else rng.choice([40, 80, 120, 160], n)
            sign = rng.choice([-1, 1], n)
            rvs = centres[:, None] + (sign * sep / 2)[:, None] * np.array([-1, 1])
        else:
            shape = np.column_stack([depth * rng.dirichlet([2, 2, 2]),
                                     rng.choice([25, 45, 65, 90], nc)])
            sep = rng.choice([40, 80, 120, 160], n)
            rvs = centres[:, None] + sep[:, None] * np.array([-1, 0, 1])
            if j % 5 == 0:
                rvs = np.repeat(centres[:, None], nc, axis=1)
            for row in rvs:
                rng.shuffle(row)
        starts.append(np.r_[shape.ravel(), rvs.ravel()])
        if shape_seed is not None:
            starts.append(np.r_[np.asarray(shape_seed).ravel(), rvs.ravel()])
    def residual(theta):
        rv = theta[ns:].reshape(n, nc)
        return np.concatenate([(curve(p["v"], theta[:ns], rv[i]) - p["f"]) / p["noise"]
                               for i, p in enumerate(profiles)])
    sols = solve_multistart(residual, starts, lo, hi)
    best = sols["best"]
    shape = best["theta"][:ns].reshape(nc, 2).copy()
    rv = best["theta"][ns:].reshape(n, nc).copy()
    order = np.argsort(-(shape[:, 0] * shape[:, 1]))
    return dict(shape=shape[order].ravel(), rv=rv[:, order], success=best["success"],
        chi2=best["chi2"], k=ns + nc * n, npix=sum(len(p["v"]) for p in profiles),
        shape_bound=near_bound(best["theta"][:ns], lo[:ns], hi[:ns]),
        chi_range=(sols["solutions"][0]["chi2"], sols["solutions"][-1]["chi2"]))


def validate_model(p, shape, nc, config):
    starts = []
    for sep in (0, 40, 80, 120, 160):
        offsets = [-sep / 2, sep / 2] if nc == 2 else [-sep, 0, sep]
        for perm in sorted(set(permutations(offsets))):
            starts.append(p["rv"] + np.array(perm))
    if nc == 3:
        rng = np.random.default_rng(config.seed)
        starts += [p["rv"] + rng.uniform(-160, 160, nc) for _ in range(12)]
    lo, hi = np.full(nc, p["rv"] - config.rv_range), np.full(nc, p["rv"] + config.rv_range)
    best = solve_multistart(lambda q: (curve(p["v"], shape, q) - p["f"]) / p["noise"],
                            starts, lo, hi)["best"]
    return best["theta"], best["success"]


def plot_legacy_shared_line(name, profiles, labels, counts, fitted, train, config):
    fig, axes = plt.subplots(2 * len(counts), len(labels), squeeze=False,
        figsize=(3.6 * len(labels), 5 * len(counts)),
        gridspec_kw={"height_ratios": [3, 1] * len(counts)})
    colors = ["steelblue", "firebrick", "darkorange"]
    fmin = min(p["f"].min() for p in profiles.values()) - 0.01
    fmax = max(p["f"].max() for p in profiles.values()) + 0.01
    rmax = max(np.max(np.abs(profiles[k]["f"] - fitted[nc, k][2]))
               for nc in counts for k in labels) * 1.1
    for row, nc in enumerate(counts):
        for col, label in enumerate(labels):
            p, (shape, rv, model) = profiles[label], fitted[nc, label]
            a, b = axes[2 * row, col], axes[2 * row + 1, col]
            a.plot(p["v"], p["f"], color="0.55", lw=0.8, label="Data")
            a.plot(p["v"], model, "k", lw=1.4, label=f"{nc}G total")
            for j, term in enumerate(gaussian_terms(p["v"], shape, rv)):
                a.plot(p["v"], 1 - term, "--", color=colors[j], lw=1,
                       label=f"RV{j + 1}={rv[j]:.1f}")
            a.set_ylim(fmin, fmax)
            a.set_title(f"{label}: {'TRAIN' if label in train else 'VALIDATE'} | {nc}G", fontsize=10)
            a.legend(fontsize=7)
            b.plot(p["v"], p["f"] - model, "k", lw=0.8)
            b.axhline(0, color="0.4", lw=0.6)
            b.axhspan(-p["noise"], p["noise"], color="0.6", alpha=0.2)
            b.set(ylim=(-rmax, rmax), xlabel="Velocity [km/s]")
        axes[2 * row, 0].set_ylabel("Normalised flux")
        axes[2 * row + 1, 0].set_ylabel("Data - model")
    fig.suptitle(f"LEGACY {name}: shared shapes | LOCAL_CONTINUUM={config.local_continuum}")
    fig.tight_layout()
    return fig


def plot_normalized_depth(profiles, lines, epochs=("E05", "E13")):
    """Original preliminary depth-normalized line comparison; no velocity alignment."""
    fig, axes = plt.subplots(1, len(epochs), figsize=(5.5 * len(epochs), 4),
                             sharey=True, squeeze=False)
    for ax, epoch in zip(axes[0], epochs):
        for name in lines:
            p = profiles[name, epoch]
            y = 1 - p["f"]
            ax.plot(p["v"], y / y.max(), lw=1, label=name)
        ax.axvline(0, ls=":", color="0.5")
        ax.set(title=epoch, xlabel="Velocity [km/s], observed frame")
    axes[0, 0].set_ylabel("Normalised absorption depth")
    axes[0, 0].legend(fontsize=8)
    return fig


def _validate_epochs(spectra, train, validate):
    if not train or set(train) & set(validate):
        raise ValueError("TRAIN must be nonempty and disjoint from VALIDATE.")
    if len(set(train)) != len(train) or len(set(validate)) != len(validate):
        raise ValueError("Duplicate TRAIN/VALIDATE labels.")
    mapping = {s["label"]: s for s in spectra}
    if set(train + validate) - set(mapping):
        raise ValueError(f"Unknown epochs: {sorted(set(train + validate) - set(mapping))}")
    return mapping


def run_legacy_shared_gaussian(spectra, lines, train, validate, counts, config,
                              *, make_plots=True, verbose=True):
    """LEGACY per-line experiment. Explicit data/settings; reference 6678 seeds only."""
    train, validate, counts = list(train), list(validate), list(counts)
    mapping = _validate_epochs(spectra, train, validate)
    if not lines or not counts or any(n not in (2, 3) for n in counts):
        raise ValueError("Select lines and component counts 2 and/or 3.")
    labels = train + validate
    reference, work = "He I 6678", {"He I 6678": 6678.149}
    work.update(lines)
    reference_shapes, tables, all_fits, all_profiles, figures, failures = {}, [], {}, {}, [], []
    training = {}
    for name, rest in work.items():
        try:
            profiles = {k: measure_profile(mapping[k], rest, config) for k in labels}
            fitted, rows = {}, []
            for nc in counts:
                result = train_model([profiles[k] for k in train], nc, config,
                    reference_shapes.get(nc) if name != reference else None)
                training[name, nc] = result
                shape = result["shape"]
                if name == reference:
                    reference_shapes[nc] = shape.copy()
                if verbose and name in lines:
                    print(f"LEGACY {name} {nc}G: shape={np.round(shape, 4)}, "
                          f"bound={result['shape_bound']}, chi2 range={result['chi_range']}, "
                          f"full-overlap flux={1 - shape.reshape(nc, 2)[:, 0].sum():.4f}")
                for label in labels:
                    p = profiles[label]
                    if label in train:
                        rv, success = result["rv"][train.index(label)], result["success"]
                    else:
                        rv, success = validate_model(p, shape, nc, config)
                    model = curve(p["v"], shape, rv)
                    chi = float(np.sum(((p["f"] - model) / p["noise"]) ** 2))
                    fitted[nc, label] = (shape, rv, model)
                    row = dict(line=name, model=f"{nc}G", epoch=label, BJD=p["bjd"],
                               role="train" if label in train else "validate", gauss_RV=p["rv"])
                    row.update({f"RV{j + 1}": rv[j] for j in range(nc)})
                    row.update({f"sep{a + 1}{b + 1}": abs(rv[b] - rv[a])
                                for a in range(nc) for b in range(a + 1, nc)})
                    row.update(RMS_over_noise=np.sqrt(chi / len(p["v"])), chi2_model=chi,
                        chi2_1g=p["chi2_1g"], n_pixels=len(p["v"]), optimiser_success=success,
                        RV_bound_hit=near_bound(rv, np.full(nc, p["rv"] - config.rv_range),
                                               np.full(nc, p["rv"] + config.rv_range)))
                    rows.append(row)
            if name in lines:
                table = pd.DataFrame(rows)
                tables.append(table)
                all_fits[name], all_profiles[name] = fitted, profiles
                if verbose:
                    print(table.to_string(index=False, float_format="%.3f"))
                if make_plots:
                    figures.append(plot_legacy_shared_line(name, profiles, labels, counts,
                                                           fitted, train, config))
        except (ValueError, RuntimeError) as exc:
            failures.append(dict(line=name, error=str(exc)))
            if verbose:
                print(f"LEGACY {name}: skipped ({exc})")
    return dict(table=pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(),
                fitted=all_fits, profiles=all_profiles, training=training,
                figures=figures, failures=failures)


# =====================================================================
# LEGACY: JOINT TWO-LINE TRAIN / VALIDATE / INDEPENDENT-RV COMPARATOR
# =====================================================================
def j_layout(ne, nl, nc, shared, train):
    nw, nr, nd = nl * nc if train else 0, ne * nc if shared else ne * nl * nc, ne * nl * nc
    return dict(nw=nw, nr=nr, nd=nd, ne=ne, nl=nl, nc=nc,
                shared=shared, train=train, size=nw + nr + nd + ne * nl * 2)


def j_pack(state, layout):
    rv = state["rv"][:, 0, :] if layout["shared"] else state["rv"]
    parts = [state["sigma"].ravel()] if layout["train"] else []
    return np.concatenate(parts + [rv.ravel(), state["depth"].ravel(), state["cont"].ravel()])


def j_unpack(q, layout, fixed_sigma=None):
    ne, nl, nc, i = layout["ne"], layout["nl"], layout["nc"], 0
    if layout["train"]:
        sigma, i = q[:layout["nw"]].reshape(nl, nc).copy(), layout["nw"]
    else:
        sigma = np.asarray(fixed_sigma).copy()
    raw_rv = q[i:i + layout["nr"]]
    i += layout["nr"]
    rv = (np.repeat(raw_rv.reshape(ne, 1, nc), nl, axis=1) if layout["shared"]
          else raw_rv.reshape(ne, nl, nc).copy())
    depth = q[i:i + layout["nd"]].reshape(ne, nl, nc).copy()
    i += layout["nd"]
    return dict(sigma=sigma, rv=rv, depth=depth, cont=q[i:].reshape(ne, nl, 2).copy())


def j_centres(data):
    return np.array([np.mean([p["rv"] for p in epoch]) for epoch in data])


def j_bounds(data, layout, base_config, joint_config):
    ne, nl, nc = layout["ne"], layout["nl"], layout["nc"]
    centres = j_centres(data)
    repeats = nc if layout["shared"] else nl * nc
    lo, hi = [], []
    if layout["train"]:
        lo.extend([joint_config.sigma_bounds[0]] * (nl * nc))
        hi.extend([joint_config.sigma_bounds[1]] * (nl * nc))
    lo.extend(np.repeat(centres - base_config.rv_range, repeats))
    hi.extend(np.repeat(centres + base_config.rv_range, repeats))
    lo.extend([0.] * (ne * nl * nc))
    hi.extend([joint_config.depth_max] * (ne * nl * nc))
    lo.extend(np.tile([1 - joint_config.cont_offset, -joint_config.cont_tilt], ne * nl))
    hi.extend(np.tile([1 + joint_config.cont_offset, joint_config.cont_tilt], ne * nl))
    return np.asarray(lo), np.asarray(hi)


def j_curve(p, state, e, l, base_config):
    shape = np.column_stack([state["depth"][e, l], state["sigma"][l]]).ravel()
    c, tilt = state["cont"][e, l]
    return c + tilt * (p["v"] - p["rv"]) / base_config.fit_half - gaussian_terms(
        p["v"], shape, state["rv"][e, l]).sum(axis=0)


def j_residual(state, data, base_config):
    return np.concatenate([(j_curve(p, state, e, l, base_config) - p["f"]) / p["noise"]
                           for e, epoch in enumerate(data) for l, p in enumerate(epoch)])


def j_random_state(data, nc, rng, sigma=None, shared=True):
    ne, nl = len(data), len(data[0])
    widths = rng.uniform(25., 95., (nl, nc)) if sigma is None else sigma.copy()
    centres = j_centres(data)
    rv, depth = np.empty((ne, nl, nc)), np.empty((ne, nl, nc))
    for e, epoch in enumerate(data):
        offsets = np.linspace(-1, 1, nc) * rng.choice([30., 60., 100., 150.])
        rng.shuffle(offsets)
        for l, p in enumerate(epoch):
            rr = offsets if shared else rng.permutation(offsets) + rng.normal(0, 20, nc)
            rv[e, l] = centres[e] + rr
            strength = np.clip(1 - np.percentile(p["f"], 5), .02, .8)
            depth[e, l] = strength * rng.dirichlet(np.full(nc, 1.5))
    cont = np.zeros((ne, nl, 2))
    cont[..., 0] = 1.
    return dict(sigma=widths, rv=rv, depth=depth, cont=cont)


def j_solve(data, layout, starts, base_config, joint_config, fixed_sigma=None):
    lo, hi = j_bounds(data, layout, base_config, joint_config)
    result = solve_multistart(lambda x: j_residual(j_unpack(x, layout, fixed_sigma),
                             data, base_config), starts, lo, hi,
                             max_nfev=joint_config.max_nfev, require_success=True)
    for sol in result["solutions"]:
        sol["state"] = j_unpack(sol["theta"], layout, fixed_sigma)
    result["layout"] = layout
    return result


def j_reorder(state, order):
    return dict(sigma=state["sigma"][:, order].copy(), rv=state["rv"][:, :, order].copy(),
                depth=state["depth"][:, :, order].copy(), cont=state["cont"].copy())


def j_feature(state, config):
    nc = state["sigma"].shape[1]
    return np.concatenate([
        state["rv"].transpose(2, 0, 1).reshape(nc, -1) / config.match_rv,
        state["sigma"].T / config.match_sigma,
        state["depth"].transpose(2, 0, 1).reshape(nc, -1) / config.match_depth], axis=1)


def j_match(state, reference, config):
    a, b = j_feature(reference, config), j_feature(state, config)
    cost = np.mean((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    row, col = linear_sum_assignment(cost)
    return j_reorder(state, col[np.argsort(row)]), float(np.sqrt(np.mean(cost[row, col])))


def j_modes(result, config):
    near = [s for s in result["solutions"]
            if s["chi2"] <= result["best"]["chi2"] + config.near_delta]
    modes = []
    for sol in near:
        distances = [j_match(sol["state"], m["state"], config)[1] for m in modes]
        if distances and min(distances) <= config.mode_distance:
            modes[int(np.argmin(distances))]["start_count"] += 1
        else:
            modes.append(dict(state=sol["state"], chi2=sol["chi2"], start_count=1))
    return modes


def j_epoch_seed(data, sigma, training_state, nc, rng, config):
    starts = []
    for e in range(training_state["rv"].shape[0]):
        state = dict(sigma=sigma.copy(), rv=training_state["rv"][e:e + 1].copy(),
            depth=training_state["depth"][e:e + 1].copy(), cont=training_state["cont"][e:e + 1].copy())
        state["rv"] += j_centres(data)[0] - np.mean(state["rv"])
        starts.append(state)
    starts += [j_random_state(data, nc, rng, sigma, True) for _ in range(config.epoch_starts)]
    return starts


def j_independent(data, sigma, shared_result, rng, base_config, joint_config):
    nc = sigma.shape[1]
    layout = j_layout(len(data), len(data[0]), nc, False, False)
    starts = [j_pack(s["state"], layout) for s in shared_result["solutions"][:5]]
    anchor = shared_result["best"]["state"]
    for perm in permutations(range(nc)):
        trial = {k: v.copy() for k, v in anchor.items()}
        trial["rv"][:, 1, :] = anchor["rv"][:, 1, list(perm)]
        starts.append(j_pack(trial, layout))
    for i in range(joint_config.epoch_starts):
        if i % 2 == 0:
            trial = {k: v.copy() for k, v in anchor.items()}
            trial["rv"] += rng.normal(0, 30, trial["rv"].shape)
        else:
            trial = j_random_state(data, nc, rng, sigma, False)
        starts.append(j_pack(trial, layout))
    result = j_solve(data, layout, starts, base_config, joint_config, sigma)
    if result["best"]["chi2"] > shared_result["best"]["chi2"] + 1e-3:
        state = {k: v.copy() for k, v in anchor.items()}
        result["solutions"].append(dict(state=state, chi2=shared_result["best"]["chi2"],
                                       theta=j_pack(state, layout), nfev=0, success=True))
        result["solutions"].sort(key=itemgetter("chi2"))
        result["best"] = result["solutions"][0]
        result["embedded_shared_used"] = True
    return result


def j_metrics(data, joint, independent, nc, branch, epoch, lines, base_config, joint_config):
    s, si = joint["best"]["state"], independent["best"]["state"]
    modes = j_modes(joint, joint_config)
    near = [x for x in joint["solutions"]
            if x["chi2"] <= joint["best"]["chi2"] + joint_config.near_delta]
    aligned = [j_match(x["state"], s, joint_config)[0] for x in near]
    span = np.ptp(np.stack([a["rv"][0, 0] for a in aligned]), axis=0)
    centre, rows, components = j_centres(data)[0], [], []
    def flags(x, low, high):
        return near_bound(x, low, high, fraction=joint_config.bound_fraction, return_mask=True)
    for l, (name, p) in enumerate(zip(lines, data[0])):
        core = np.abs(p["v"] - p["rv"]) <= joint_config.core_half
        r = (p["f"] - j_curve(p, s, 0, l, base_config)) / p["noise"]
        ri = (p["f"] - j_curve(p, si, 0, l, base_config)) / p["noise"]
        cr, cri = np.sqrt(np.mean(r[core] ** 2)), np.sqrt(np.mean(ri[core] ** 2))
        cont_flag = bool(flags(s["cont"][0, l, 0], 1 - joint_config.cont_offset, 1 + joint_config.cont_offset)
            or flags(s["cont"][0, l, 1], -joint_config.cont_tilt, joint_config.cont_tilt))
        rows.append(dict(model=f"{nc}G", train_branch=branch, epoch=epoch, line=name,
            joint_chi2=float(r @ r), independent_chi2=float(ri @ ri),
            joint_RMS_noise=float(np.sqrt(np.mean(r ** 2))), joint_core_RMS_noise=float(cr),
            independent_core_RMS_noise=float(cri), core_RMS_degradation=float(cr - cri),
            continuum=s["cont"][0, l, 0], tilt=s["cont"][0, l, 1], continuum_bound=cont_flag,
            joint_total_delta_chi2=joint["best"]["chi2"] - independent["best"]["chi2"],
            joint_near_modes=len(modes), max_near_RV_span=float(span.max()),
            residual_screen=bool(cr <= joint_config.core_rms_limit),
            transfer_screen=bool(cr - cri <= joint_config.core_degradation)))
        for j in range(nc):
            overlap = min(abs(s["rv"][0, l, j] - s["rv"][0, l, k]) /
                          np.hypot(s["sigma"][l, j], s["sigma"][l, k])
                          for k in range(nc) if k != j)
            components.append(dict(model=f"{nc}G", train_branch=branch, epoch=epoch,
                line=name, component=j + 1, shared_RV=s["rv"][0, l, j],
                independent_RV=si["rv"][0, l, j], depth=s["depth"][0, l, j],
                sigma_kms=s["sigma"][l, j], near_RV_span=span[j],
                RV_bound=bool(flags(s["rv"][0, l, j], centre - base_config.rv_range, centre + base_config.rv_range)),
                sigma_bound=bool(flags(s["sigma"][l, j], *joint_config.sigma_bounds)),
                depth_bound=bool(flags(s["depth"][0, l, j], 0., joint_config.depth_max)),
                depth_below_pixel_noise=bool(s["depth"][0, l, j] <= p["noise"]), overlap_ratio=overlap))
    return rows, components


def j_plots(profiles, validations, train_results, lines, validate, counts, base_config, joint_config):
    figures = []
    for nc in counts:
        for l, name in enumerate(lines):
            fig, axes = plt.subplots(2, len(validate), figsize=(5 * len(validate), 7),
                squeeze=False, gridspec_kw={"height_ratios": [3, 1.5]})
            for col, epoch in enumerate(validate):
                a, b = axes[:, col]
                p = profiles[name, epoch]
                rec = validations[nc, 1, epoch]
                joint, independent = rec["joint"], rec["independent"]
                s, si = joint["best"]["state"], independent["best"]["state"]
                a.plot(p["v"], p["f"], color="0.55", lw=0.8, label="Data")
                for branch in range(1, len(train_results[nc]["branches"]) + 1):
                    alt = validations[nc, branch, epoch]["joint"]
                    for mode in j_modes(alt, joint_config)[:3]:
                        a.plot(p["v"], j_curve(p, mode["state"], 0, l, base_config),
                               color="purple", lw=0.8, alpha=0.2)
                mj, mi = j_curve(p, s, 0, l, base_config), j_curve(p, si, 0, l, base_config)
                a.plot(p["v"], mj, "k", lw=1.3, label="Joint: primary TRAIN branch")
                a.plot(p["v"], mi, color="darkorange", ls="--", lw=1.2, label="Independent RVs, same widths")
                continuum = s["cont"][0, l, 0] + s["cont"][0, l, 1] * (p["v"] - p["rv"]) / base_config.fit_half
                shape = np.column_stack([s["depth"][0, l], s["sigma"][l]]).ravel()
                for j, term in enumerate(gaussian_terms(p["v"], shape, s["rv"][0, l])):
                    a.plot(p["v"], continuum - term, ":", lw=0.8,
                           label=f"C{j + 1}: {s['rv'][0, l, j]:.1f} km/s")
                b.plot(p["v"], p["f"] - mj, "k", lw=0.8)
                b.plot(p["v"], p["f"] - mi, color="darkorange", lw=0.8, ls="--")
                b.axhspan(-p["noise"], p["noise"], color="0.5", alpha=0.15)
                b.axhline(0, color="0.4", lw=0.7)
                a.set(title=f"LEGACY {name}: {epoch}, {nc}G", ylabel="Input-normalised flux")
                b.set(ylabel="Data - model", xlabel="Velocity [km/s], original convention")
                for ax in (a, b):
                    ax.set_xlim(p["rv"] - base_config.fit_half, p["rv"] + base_config.fit_half)
                a.legend(fontsize=6)
            low = min(ax.get_ylim()[0] for ax in axes[0])
            high = max(ax.get_ylim()[1] for ax in axes[0])
            limit = max(max(abs(x) for x in ax.get_ylim()) for ax in axes[1])
            for ax in axes[0]:
                ax.set_ylim(low, high)
            for ax in axes[1]:
                ax.set_ylim(-limit, limit)
            fig.suptitle("LEGACY held-out epochs: shared vs independent RVs; purple = alternatives")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            figures.append(fig)
    return figures


def run_legacy_joint_he(spectra, lines, train, validate, counts, base_config, joint_config,
                        *, make_plots=True, verbose=True):
    """LEGACY joint-He experiment with identical-width independent-RV comparator.

    Pass the original ordered line mapping (6678 first), training/validation
    lists and both configurations explicitly. Joint agreement is ENFORCED;
    residuals, independent-fit degradation and ambiguity are the diagnostics.
    Return values retain full optimizer solutions and exploratory mode tables.
    """
    train, validate, counts = list(train), list(validate), list(counts)
    mapping = _validate_epochs(spectra, train, validate)
    if len(lines) != 2 or not validate or not counts or any(n not in (2, 3) for n in counts):
        raise ValueError("Require two lines, validation epochs, and counts 2 and/or 3.")
    labels = train + validate
    if verbose:
        print(pd.DataFrame([dict(epoch=k, BJD=mapping[k]["bjd"], file=mapping[k]["file"])
                            for k in labels]).to_string(index=False))
        print(f"LEGACY: LOCAL_CONTINUUM={base_config.local_continuum}; "
              f"explicit shift={base_config.additional_shift_kms} km/s")
    profiles = {(name, k): measure_profile(mapping[k], rest, base_config)
                for name, rest in lines.items() for k in labels}
    train_data = [[profiles[name, k] for name in lines] for k in train]
    training, validations = {}, {}
    train_rows, rows, component_rows, mode_rows = [], [], [], []
    for nc in counts:
        rng = np.random.default_rng(joint_config.seed + nc * 1000)
        layout = j_layout(len(train), len(lines), nc, True, True)
        starts = [j_pack(j_random_state(train_data, nc, rng), layout)
                  for _ in range(joint_config.train_starts)]
        tr = j_solve(train_data, layout, starts, base_config, joint_config)
        modes = j_modes(tr, joint_config)
        branches = modes[:joint_config.max_train_branches]
        if not branches:
            raise ValueError("No retained TRAIN branches; check branch settings.")
        ref = j_reorder(branches[0]["state"], np.argsort(branches[0]["state"]["rv"][0, 0]))
        for branch in branches:
            branch["state"] = j_match(branch["state"], ref, joint_config)[0]
        tr["branches"], training[nc] = branches, tr
        if verbose:
            print(f"{nc}G: converged {len(tr['solutions'])}/{tr['attempted_starts']}; "
                  f"near TRAIN modes={len(modes)}; validating {len(branches)} branches.")
            if len(modes) > len(branches):
                print("WARNING: TRAIN branch cap reached; full solutions retained.")
        for bi, branch in enumerate(branches, 1):
            ts = branch["state"]
            for l, name in enumerate(lines):
                for j in range(nc):
                    train_rows.append(dict(model=f"{nc}G", branch=bi, line=name, component=j + 1,
                        sigma_kms=ts["sigma"][l, j], train_delta_chi2=branch["chi2"] - tr["best"]["chi2"],
                        sigma_bound=near_bound(ts["sigma"][l, j], *joint_config.sigma_bounds,
                                               fraction=joint_config.bound_fraction)))
            for ei, epoch in enumerate(validate):
                if verbose:
                    print(f"{nc}G branch {bi}, {epoch}: joint + independent comparison...")
                data = [[profiles[name, epoch] for name in lines]]
                erng = np.random.default_rng(joint_config.seed + nc * 1000 + bi * 100 + ei)
                vl = j_layout(1, len(lines), nc, True, False)
                starts = [j_pack(s, vl) for s in j_epoch_seed(data, ts["sigma"], ts, nc, erng, joint_config)]
                joint = j_solve(data, vl, starts, base_config, joint_config, ts["sigma"])
                independent = j_independent(data, ts["sigma"], joint, erng, base_config, joint_config)
                validations[nc, bi, epoch] = dict(joint=joint, independent=independent)
                rr, cc = j_metrics(data, joint, independent, nc, bi, epoch, lines, base_config, joint_config)
                rows.extend(rr)
                component_rows.extend(cc)
                for mi, mode in enumerate(j_modes(joint, joint_config), 1):
                    aligned = j_match(mode["state"], joint["best"]["state"], joint_config)[0]
                    row = dict(model=f"{nc}G", train_branch=bi, epoch=epoch, mode=mi,
                        delta_chi2=mode["chi2"] - joint["best"]["chi2"], start_count=mode["start_count"])
                    row.update({f"RV{j + 1}": aligned["rv"][0, 0, j] for j in range(nc)})
                    mode_rows.append(row)
    widths, summary = pd.DataFrame(train_rows), pd.DataFrame(rows)
    components, mode_table = pd.DataFrame(component_rows), pd.DataFrame(mode_rows)
    branch_rows = []
    for nc in counts:
        for epoch in validate:
            anchor = validations[nc, 1, epoch]["joint"]["best"]["state"]
            states = [j_match(validations[nc, bi, epoch]["joint"]["best"]["state"], anchor, joint_config)[0]
                      for bi in range(1, len(training[nc]["branches"]) + 1)]
            spans = np.ptp(np.stack([s["rv"][0, 0] for s in states]), axis=0)
            branch_rows.append(dict(model=f"{nc}G", epoch=epoch, retained_train_branches=len(states),
                                    max_RV_span_across_branches=float(spans.max())))
    branch_table = pd.DataFrame(branch_rows)
    if verbose:
        for title, table in [("TRAIN WIDTHS / BRANCHES", widths), ("VALIDATION COMPARISON", summary),
            ("COMPONENT WARNINGS — NOT DETECTIONS", components), ("VALIDATION MODES", mode_table),
            ("SENSITIVITY TO TRAIN BRANCHES", branch_table)]:
            print("\n" + title)
            print(table.to_string(index=False, float_format="%.3f"))
        print("LEGACY: no files, bootstrap or orbital fit. Shared RVs are enforced.")
        print("Depth/noise, overlap and mode-span flags are exploratory heuristics, not detections or RV errors.")
        print("Comparator component columns are template labels, not established stellar tracks.")
    figures = (j_plots(profiles, validations, training, lines, validate, counts,
                       base_config, joint_config) if make_plots else [])
    return dict(profiles=profiles, training=training, validation=validations,
                widths=widths, summary=summary, components=components, modes=mode_table,
                branch_sensitivity=branch_table, figures=figures)
