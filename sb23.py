"""Gaussian line-profile analysis (2G/3G).

The Gaussian components are descriptive profile components. Their labels do
not establish stellar identities, and their depths/areas are not continuum
light fractions. The per-line and joint-He workflows are separate analyses:
- run_shared_gaussian: train shared Gaussian shapes per line, then fit RVs.
- run_joint_he: share component RVs across two lines, with line-specific
  widths/depths and compare against independent line RVs.

Importing this module performs no file I/O, fitting, plotting, style changes,
or output writing. Inputs are read only when a reader is called. Plotters return
figures and do not call show(), savefig(), mkdir(), or any output writer.

No air/vacuum or barycentric correction, added broadening, orbit inference,
bootstrap, disentangling, or physical SB2-template fit is performed here.
The rvfuncs.gaussian_vel routine is imported lazily when a Gaussian model is
first evaluated, because importing rvfuncs may have its own side effects.
"""
from dataclasses import dataclass
from pathlib import Path
from itertools import permutations
from operator import itemgetter
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import least_squares, linear_sum_assignment

from hd195592_common import wavelength_to_velocity


_GAUSSIAN_VEL = None


def _gaussian_vel(*args):
    """Load and call rvfuncs.gaussian_vel only when a Gaussian fit is requested."""
    global _GAUSSIAN_VEL
    if _GAUSSIAN_VEL is None:
        _GAUSSIAN_VEL = importlib.import_module("rvfuncs").gaussian_vel
    return _GAUSSIAN_VEL(*args)


@dataclass(frozen=True)
class GaussianConfig:
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
class JointGaussianConfig:
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
    bound_fraction: float = 0.001


# =====================================================================
# INPUT READING
# =====================================================================
def clean_spectrum(wave, flux, label="spectrum", *, min_points=2, verbose=True):
    """Filter invalid samples, stable-sort, and average duplicate wavelengths."""
    wave = np.asarray(wave, float).ravel()
    flux = np.asarray(flux, float).ravel()
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


def read_hermes_inspection(path, *, verbose=True):
    """Read a 1D HERMES spectrum using its declared FITS axis type.

    Supports natural-log wavelength and explicitly linear wavelength axes,
    plus CDELT1/CD1_1. Units and air/vacuum convention are reported, not
    guessed or converted. Duplicate wavelengths are averaged by clean_spectrum.
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
        print(f"{path.name}: CTYPE1={ctype!r}, "
              f"CUNIT1={header.get('CUNIT1', 'unspecified')!r}, "
              f"CRPIX1={crpix}, BJD={bjd}")
    return wave, flux, bjd


def load_observations(obs_dir, obs_files, *, reader=read_hermes_inspection):
    """Load explicit epoch-to-file mapping; return (datasets, failures).

    Each dataset contains label, file, wave, flux, bjd and observational=True.
    Explicit mapping avoids silently assigning epoch labels by filename order.
    """
    datasets, failures = {}, []
    for epoch, filename in obs_files.items():
        try:
            wave, flux, bjd = reader(Path(obs_dir) / filename)
            datasets[epoch] = dict(label=epoch, file=filename, filename=filename,
                wave=wave, flux=flux, bjd=bjd, observational=True,
                kind="observation")
        except (OSError, ValueError, KeyError) as exc:
            failures.append(dict(epoch=epoch, file=filename, error=str(exc)))
    return datasets, failures

# =====================================================================
# GAUSSIAN PROFILE PREPARATION AND OPTIMISATION
# =====================================================================
def near_bound(p, lo, hi, *, fraction=0.001, return_mask=False):
    p, lo, hi = np.asarray(p), np.asarray(lo), np.asarray(hi)
    mask = np.minimum((p - lo) / (hi - lo), (hi - p) / (hi - lo)) < fraction
    return mask if return_mask else bool(np.any(mask))

def solve_multistart(residual, starts, lo, hi, *, max_nfev=50000,
                     require_success=False):
    """Run bounded least-squares starts; preserve status and all solutions."""
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
        if not np.isfinite(chi):
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
    """Positive Gaussian absorption deficits for each component."""
    shape = np.asarray(shape).reshape(-1, 2)
    return -_gaussian_vel(np.asarray(v)[None, :], 0.0, 0.0,
                          shape[:, 0, None], np.asarray(rv)[:, None],
                          shape[:, 1, None])

def gaussian_curve(v, shape, rv):
    """Constant-unit-continuum multi-Gaussian absorption model."""
    return 1.0 - gaussian_terms(v, shape, rv).sum(axis=0)

def normalise_locally(v, f, centre, cont_bounds):
    """Three-pass local linear continuum normalisation used by the RV fits."""
    side = ((np.abs(v - centre) >= cont_bounds[0]) &
            (np.abs(v - centre) <= cont_bounds[1]))

    def adequate(mask):
        return mask.sum() >= 10 and np.any(v[mask] < centre) and np.any(v[mask] > centre)

    if not adequate(side):
        raise ValueError("Insufficient continuum on both sides.")
    x, keep = (v - centre) / 200.0, side.copy()
    for _ in range(3):
        residual = f - np.polyval(np.polyfit(x[keep], f[keep], 1), x)
        mid, scatter = np.median(residual[keep]), 1.4826 * np.median(
            np.abs(residual[keep] - np.median(residual[keep])))
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
    """Find a preliminary 1G centroid and prepare a fitting-window profile.

    The 1G centroid is a window-centering aid, not a component RV measurement.
    """
    v = wavelength_to_velocity(spectrum["wave"], rest,
                               shift_kms=config.additional_shift_kms)
    f = np.asarray(spectrum["flux"])
    near = np.abs(v) <= config.search_half
    if near.sum() < 20:
        raise ValueError("Line outside coverage.")
    smooth = np.convolve(f, np.ones(9) / 9, mode="same")
    start = v[near][np.argmin(smooth[near])]
    use = np.abs(v - start) <= max(config.fit_half + 120,
                                   config.cont_bounds[1] + 120)
    v, f = v[use], f[use]
    if config.local_continuum:
        f = normalise_locally(v, f, start, config.cont_bounds)
    mask = np.abs(v - start) <= config.gauss_fit_half
    if mask.sum() < 15:
        raise ValueError("Too few centroid-fit pixels.")
    depth0 = float(np.clip(1 - np.percentile(f[mask], 5), 0.01, 0.8))
    single = solve_multistart(
        lambda q: _gaussian_vel(v[mask], *q) - f[mask],
        [[1.0, 0.0, depth0, start, width] for width in (25, 50, 90)],
        [0.85, -0.15, 0.001, start - 100, 8],
        [1.15, 0.15, 0.95, start + 100, 150])
    rv = float(single["best"]["theta"][3])
    distance = np.abs(v - rv)
    side = ((distance >= config.cont_bounds[0]) &
            (distance <= config.cont_bounds[1]))
    window = distance <= config.fit_half
    if side.sum() < 10 or window.sum() < 20:
        raise ValueError("Insufficient fit window or noise sidebands.")
    noise = float(np.std(f[side], ddof=1))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Invalid noise estimate.")
    profile = dict(label=spectrum["label"], bjd=spectrum.get("bjd"),
                   v=v[window], f=f[window], rv=rv, noise=noise)
    depth0 = float(np.clip(1 - np.percentile(profile["f"], 5), 0.01, 0.8))
    baseline = solve_multistart(
        lambda q: (gaussian_curve(profile["v"], q[:2], q[2:]) - profile["f"]) / noise,
        [[depth0, width, rv] for width in (35, 60, 100)],
        [0, 8, rv - config.rv_range], [0.95, 150, rv + config.rv_range])
    profile["chi2_1g"] = baseline["best"]["chi2"]
    return profile


def train_gaussian_model(profiles, n_components, config):
    """Train shared component depths/widths for one line; RVs vary by epoch.

    Components are ordered by depth*width as a bookkeeping convention, not as
    primary/secondary star identities. The fit returns all optimizer solutions
    so branch sensitivity is inspectable.
    """
    if n_components not in (2, 3) or not profiles:
        raise ValueError("Require profiles and n_components=2 or 3.")
    n_epoch, n_shape = len(profiles), 2 * n_components
    centres = np.array([p["rv"] for p in profiles])
    lo = np.r_[np.tile([0, 8], n_components),
               np.repeat(centres - config.rv_range, n_components)]
    hi = np.r_[np.tile([0.95, 150], n_components),
               np.repeat(centres + config.rv_range, n_components)]
    rng = np.random.default_rng(config.seed)
    depth0 = float(np.clip(max(1 - np.percentile(p["f"], 5) for p in profiles),
                           0.05, 0.8))
    starts = []
    for j in range(config.n_starts):
        if n_components == 2:
            widths = ((45, 45), (35, 65), (65, 35))[j % 3]
            fraction = (0.35, 0.5, 0.65)[(j // 3) % 3]
            shape = np.array([[depth0 * fraction, widths[0]],
                              [depth0 * (1 - fraction), widths[1]]])
            separation = (np.zeros(n_epoch) if j % 5 == 0 else
                          rng.choice([40, 80, 120, 160], n_epoch))
            signs = rng.choice([-1, 1], n_epoch)
            rvs = centres[:, None] + (signs * separation / 2)[:, None] * np.array([-1, 1])
        else:
            shape = np.column_stack([depth0 * rng.dirichlet([2, 2, 2]),
                                     rng.choice([25, 45, 65, 90], n_components)])
            separation = rng.choice([40, 80, 120, 160], n_epoch)
            rvs = centres[:, None] + separation[:, None] * np.array([-1, 0, 1])
            if j % 5 == 0:
                rvs = np.repeat(centres[:, None], n_components, axis=1)
            for row in rvs:
                rng.shuffle(row)
        starts.append(np.r_[shape.ravel(), rvs.ravel()])

    def residual(theta):
        rvs = theta[n_shape:].reshape(n_epoch, n_components)
        return np.concatenate([
            (gaussian_curve(p["v"], theta[:n_shape], rvs[i]) - p["f"]) / p["noise"]
            for i, p in enumerate(profiles)])

    solutions = solve_multistart(residual, starts, lo, hi)
    best = solutions["best"]
    shape = best["theta"][:n_shape].reshape(n_components, 2).copy()
    rvs = best["theta"][n_shape:].reshape(n_epoch, n_components).copy()
    order = np.argsort(-(shape[:, 0] * shape[:, 1]))
    shape, rvs = shape[order], rvs[:, order]
    return dict(shape=shape.ravel(), rv=rvs, success=best["success"],
        chi2=best["chi2"], k=n_shape + n_components * n_epoch,
        npix=sum(len(p["v"]) for p in profiles),
        shape_bound=near_bound(best["theta"][:n_shape], lo[:n_shape], hi[:n_shape]),
        chi_range=(solutions["solutions"][0]["chi2"],
                   solutions["solutions"][-1]["chi2"]),
        solutions=solutions["solutions"], attempted_starts=solutions["attempted_starts"],
        failed_starts=solutions["failed_starts"])


def validate_gaussian_model(profile, shape, n_components, config):
    """Fit validation-epoch RVs with training-epoch component shapes fixed."""
    starts = []
    for separation in (0, 40, 80, 120, 160):
        offsets = ([-separation / 2, separation / 2] if n_components == 2
                   else [-separation, 0, separation])
        for ordering in sorted(set(permutations(offsets))):
            starts.append(profile["rv"] + np.asarray(ordering))
    if n_components == 3:
        rng = np.random.default_rng(config.seed)
        starts += [profile["rv"] + rng.uniform(-160, 160, n_components)
                   for _ in range(12)]
    lo = np.full(n_components, profile["rv"] - config.rv_range)
    hi = np.full(n_components, profile["rv"] + config.rv_range)
    result = solve_multistart(
        lambda rvs: (gaussian_curve(profile["v"], shape, rvs) - profile["f"])
                    / profile["noise"], starts, lo, hi)
    return result["best"]["theta"], result["best"]["success"], result


def plot_shared_gaussian_fit(line_name, profiles, labels, component_counts,
                             fitted, train_epochs, config):
    fig, axes = plt.subplots(2 * len(component_counts), len(labels), squeeze=False,
        figsize=(3.6 * len(labels), 5 * len(component_counts)),
        gridspec_kw={"height_ratios": [3, 1] * len(component_counts)})
    colors = ["steelblue", "firebrick", "darkorange"]
    fmin = min(p["f"].min() for p in profiles.values()) - 0.01
    fmax = max(p["f"].max() for p in profiles.values()) + 0.01
    residual_max = max(np.max(np.abs(profiles[k]["f"] - fitted[nc, k][2]))
                       for nc in component_counts for k in labels) * 1.1
    for row, n_components in enumerate(component_counts):
        for col, label in enumerate(labels):
            profile = profiles[label]
            shape, rvs, model = fitted[n_components, label]
            ax, residual_ax = axes[2 * row, col], axes[2 * row + 1, col]
            ax.plot(profile["v"], profile["f"], color="0.55", lw=0.8, label="Data")
            ax.plot(profile["v"], model, "k", lw=1.4,
                    label=f"{n_components}G total")
            for j, term in enumerate(gaussian_terms(profile["v"], shape, rvs)):
                ax.plot(profile["v"], 1 - term, "--", color=colors[j], lw=1,
                        label=f"Gaussian {j + 1}: {rvs[j]:.1f}")
            ax.set_ylim(fmin, fmax)
            role = "TRAIN" if label in train_epochs else "VALIDATE"
            ax.set_title(f"{label}: {role} | {n_components}G", fontsize=10)
            ax.legend(fontsize=7)
            residual_ax.plot(profile["v"], profile["f"] - model, "k", lw=0.8)
            residual_ax.axhline(0, color="0.4", lw=0.6)
            residual_ax.axhspan(-profile["noise"], profile["noise"],
                                color="0.6", alpha=0.2)
            residual_ax.set(ylim=(-residual_max, residual_max), xlabel="Velocity [km/s]")
        axes[2 * row, 0].set_ylabel("Normalised flux")
        axes[2 * row + 1, 0].set_ylabel("Data − model")
    fig.suptitle(f"{line_name}: per-line shared-shape Gaussian fits | "
                 f"LOCAL_CONTINUUM={config.local_continuum}")
    fig.tight_layout()
    return fig


def _validate_epochs(spectra, train, validate):
    if not train or set(train) & set(validate):
        raise ValueError("TRAIN must be nonempty and disjoint from VALIDATE.")
    if len(set(train)) != len(train) or len(set(validate)) != len(validate):
        raise ValueError("Duplicate TRAIN/VALIDATE labels.")
    mapping = {s["label"]: s for s in spectra}
    missing = set(train + validate) - set(mapping)
    if missing:
        raise ValueError(f"Unknown epochs: {sorted(missing)}")
    return mapping


def run_shared_gaussian(spectra, lines, train, validate, component_counts, config,
                        *, make_plots=True, verbose=True):
    """Per-line shared-shape 2G/3G experiment with explicit train/validation.

    `lines` is an ordered mapping from line label to configured rest wavelength.
    Each line is trained independently. Component numbers are Gaussian-shape
    bookkeeping labels, not stellar identities. Validation holds the trained
    depths/widths fixed and optimises the epoch's component RVs.
    """
    train, validate = list(train), list(validate)
    component_counts = list(component_counts)
    mapping = _validate_epochs(spectra, train, validate)
    if not lines or not component_counts or any(n not in (2, 3) for n in component_counts):
        raise ValueError("Select lines and component counts 2 and/or 3.")
    labels = train + validate
    tables, all_fits, all_profiles, figures, failures, training = [], {}, {}, [], [], {}
    for line_name, rest in lines.items():
        try:
            profiles = {epoch: measure_profile(mapping[epoch], rest, config)
                        for epoch in labels}
            fitted, rows = {}, []
            for n_components in component_counts:
                result = train_gaussian_model(
                    [profiles[epoch] for epoch in train], n_components, config)
                training[line_name, n_components] = result
                shape = result["shape"]
                if verbose:
                    print(f"{line_name} {n_components}G: shape="
                          f"{np.round(shape.reshape(n_components, 2), 4)}, "
                          f"bound={result['shape_bound']}, "
                          f"chi2 range={result['chi_range']}, "
                          f"starts={result['attempted_starts']}, "
                          f"failed={result['failed_starts']}")
                for epoch in labels:
                    profile = profiles[epoch]
                    if epoch in train:
                        rvs = result["rv"][train.index(epoch)]
                        success = result["success"]
                    else:
                        rvs, success, _ = validate_gaussian_model(
                            profile, shape, n_components, config)
                    model = gaussian_curve(profile["v"], shape, rvs)
                    chi2 = float(np.sum(((profile["f"] - model) / profile["noise"]) ** 2))
                    fitted[n_components, epoch] = (shape, rvs, model)
                    row = dict(line=line_name, model=f"{n_components}G", epoch=epoch,
                               BJD=profile["bjd"],
                               role="train" if epoch in train else "validate")
                    row.update({f"G{j + 1}_RV": rvs[j] for j in range(n_components)})
                    row.update({f"sep{a + 1}{b + 1}": abs(rvs[b] - rvs[a])
                                for a in range(n_components)
                                for b in range(a + 1, n_components)})
                    row.update(RMS_over_noise=np.sqrt(chi2 / len(profile["v"])),
                        chi2_model=chi2, chi2_1g=profile["chi2_1g"],
                        n_pixels=len(profile["v"]), optimiser_success=success,
                        RV_bound_hit=near_bound(
                            rvs, np.full(n_components, profile["rv"] - config.rv_range),
                            np.full(n_components, profile["rv"] + config.rv_range)))
                    rows.append(row)
            table = pd.DataFrame(rows)
            tables.append(table)
            all_fits[line_name], all_profiles[line_name] = fitted, profiles
            if verbose:
                print(table.to_string(index=False, float_format="%.3f"))
            if make_plots:
                figures.append(plot_shared_gaussian_fit(
                    line_name, profiles, labels, component_counts, fitted, train, config))
        except (ValueError, RuntimeError) as exc:
            failures.append(dict(line=line_name, error=str(exc)))
            if verbose:
                print(f"{line_name}: skipped ({exc})")
    return dict(table=pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(),
                fitted=all_fits, profiles=all_profiles, training=training,
                figures=figures, failures=failures)


# =====================================================================
# TWO-LINE JOINT FIT: SHARED RVs VERSUS INDEPENDENT LINE RVs
# =====================================================================
def j_layout(n_epoch, n_line, n_components, shared_rv, train):
    n_width = n_line * n_components if train else 0
    n_rv = n_epoch * n_components if shared_rv else n_epoch * n_line * n_components
    n_depth = n_epoch * n_line * n_components
    return dict(nw=n_width, nr=n_rv, nd=n_depth, ne=n_epoch, nl=n_line,
                nc=n_components, shared=shared_rv, train=train,
                size=n_width + n_rv + n_depth + n_epoch * n_line * 2)


def j_pack(state, layout):
    rvs = state["rv"][:, 0, :] if layout["shared"] else state["rv"]
    parts = [state["sigma"].ravel()] if layout["train"] else []
    return np.concatenate(parts + [rvs.ravel(), state["depth"].ravel(),
                                   state["cont"].ravel()])


def j_unpack(q, layout, fixed_sigma=None):
    ne, nl, nc = layout["ne"], layout["nl"], layout["nc"]
    i = 0
    if layout["train"]:
        sigma = q[:layout["nw"]].reshape(nl, nc).copy()
        i = layout["nw"]
    else:
        sigma = np.asarray(fixed_sigma).copy()
    raw_rv = q[i:i + layout["nr"]]
    i += layout["nr"]
    rv = (np.repeat(raw_rv.reshape(ne, 1, nc), nl, axis=1) if layout["shared"]
          else raw_rv.reshape(ne, nl, nc).copy())
    depth = q[i:i + layout["nd"]].reshape(ne, nl, nc).copy()
    i += layout["nd"]
    return dict(sigma=sigma, rv=rv, depth=depth,
                cont=q[i:].reshape(ne, nl, 2).copy())


def j_centres(data):
    return np.array([np.mean([p["rv"] for p in epoch]) for epoch in data])


def j_bounds(data, layout, gaussian_config, joint_config):
    ne, nl, nc = layout["ne"], layout["nl"], layout["nc"]
    centres = j_centres(data)
    repeats = nc if layout["shared"] else nl * nc
    lo, hi = [], []
    if layout["train"]:
        lo.extend([joint_config.sigma_bounds[0]] * (nl * nc))
        hi.extend([joint_config.sigma_bounds[1]] * (nl * nc))
    lo.extend(np.repeat(centres - gaussian_config.rv_range, repeats))
    hi.extend(np.repeat(centres + gaussian_config.rv_range, repeats))
    lo.extend([0.] * (ne * nl * nc))
    hi.extend([joint_config.depth_max] * (ne * nl * nc))
    lo.extend(np.tile([1 - joint_config.cont_offset, -joint_config.cont_tilt], ne * nl))
    hi.extend(np.tile([1 + joint_config.cont_offset, joint_config.cont_tilt], ne * nl))
    return np.asarray(lo), np.asarray(hi)


def j_curve(profile, state, epoch_index, line_index, gaussian_config):
    shape = np.column_stack([state["depth"][epoch_index, line_index],
                             state["sigma"][line_index]]).ravel()
    continuum, tilt = state["cont"][epoch_index, line_index]
    return (continuum + tilt * (profile["v"] - profile["rv"]) /
            gaussian_config.fit_half -
            gaussian_terms(profile["v"], shape,
                           state["rv"][epoch_index, line_index]).sum(axis=0))


def j_residual(state, data, gaussian_config):
    return np.concatenate([
        (j_curve(profile, state, e, line, gaussian_config) - profile["f"])
        / profile["noise"]
        for e, epoch in enumerate(data)
        for line, profile in enumerate(epoch)])


def j_random_state(data, n_components, rng, sigma=None, shared_rv=True):
    n_epoch, n_line = len(data), len(data[0])
    widths = (rng.uniform(25., 95., (n_line, n_components)) if sigma is None
              else sigma.copy())
    centres = j_centres(data)
    rvs = np.empty((n_epoch, n_line, n_components))
    depths = np.empty((n_epoch, n_line, n_components))
    for e, epoch in enumerate(data):
        offsets = np.linspace(-1, 1, n_components) * rng.choice([30., 60., 100., 150.])
        rng.shuffle(offsets)
        for line, profile in enumerate(epoch):
            line_offsets = (offsets if shared_rv else
                            rng.permutation(offsets) + rng.normal(0, 20, n_components))
            rvs[e, line] = centres[e] + line_offsets
            strength = np.clip(1 - np.percentile(profile["f"], 5), .02, .8)
            depths[e, line] = strength * rng.dirichlet(np.full(n_components, 1.5))
    continuum = np.zeros((n_epoch, n_line, 2))
    continuum[..., 0] = 1.
    return dict(sigma=widths, rv=rvs, depth=depths, cont=continuum)


def j_solve(data, layout, starts, gaussian_config, joint_config, fixed_sigma=None):
    lo, hi = j_bounds(data, layout, gaussian_config, joint_config)
    result = solve_multistart(
        lambda x: j_residual(j_unpack(x, layout, fixed_sigma), data, gaussian_config),
        starts, lo, hi, max_nfev=joint_config.max_nfev, require_success=True)
    for solution in result["solutions"]:
        solution["state"] = j_unpack(solution["theta"], layout, fixed_sigma)
    result["layout"] = layout
    return result


def j_reorder(state, order):
    return dict(sigma=state["sigma"][:, order].copy(),
                rv=state["rv"][:, :, order].copy(),
                depth=state["depth"][:, :, order].copy(),
                cont=state["cont"].copy())


def j_feature(state, config):
    n_components = state["sigma"].shape[1]
    return np.concatenate([
        state["rv"].transpose(2, 0, 1).reshape(n_components, -1) / config.match_rv,
        state["sigma"].T / config.match_sigma,
        state["depth"].transpose(2, 0, 1).reshape(n_components, -1) / config.match_depth
    ], axis=1)


def j_match(state, reference, config):
    a, b = j_feature(reference, config), j_feature(state, config)
    cost = np.mean((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    row, col = linear_sum_assignment(cost)
    return j_reorder(state, col[np.argsort(row)]), float(np.sqrt(np.mean(cost[row, col])))


def j_modes(result, config):
    near = [s for s in result["solutions"]
            if s["chi2"] <= result["best"]["chi2"] + config.near_delta]
    modes = []
    for solution in near:
        distances = [j_match(solution["state"], mode["state"], config)[1]
                     for mode in modes]
        if distances and min(distances) <= config.mode_distance:
            modes[int(np.argmin(distances))]["start_count"] += 1
        else:
            modes.append(dict(state=solution["state"], chi2=solution["chi2"],
                              start_count=1))
    return modes


def j_epoch_seed(data, sigma, training_state, rng, joint_config):
    starts = []
    target_centre = j_centres(data)[0]
    for e in range(training_state["rv"].shape[0]):
        state = dict(sigma=sigma.copy(), rv=training_state["rv"][e:e + 1].copy(),
            depth=training_state["depth"][e:e + 1].copy(),
            cont=training_state["cont"][e:e + 1].copy())
        state["rv"] += target_centre - np.mean(state["rv"])
        starts.append(state)
    n_components = sigma.shape[1]
    starts += [j_random_state(data, n_components, rng, sigma, True)
               for _ in range(joint_config.epoch_starts)]
    return starts


def j_independent(data, sigma, shared_result, rng, gaussian_config, joint_config):
    """Fit line-specific RVs with the same trained widths as the shared-RV model."""
    n_components = sigma.shape[1]
    layout = j_layout(len(data), len(data[0]), n_components, False, False)
    starts = [j_pack(solution["state"], layout)
              for solution in shared_result["solutions"][:5]]
    anchor = shared_result["best"]["state"]
    for ordering in permutations(range(n_components)):
        trial = {key: value.copy() for key, value in anchor.items()}
        trial["rv"][:, 1, :] = anchor["rv"][:, 1, list(ordering)]
        starts.append(j_pack(trial, layout))
    for i in range(joint_config.epoch_starts):
        if i % 2 == 0:
            trial = {key: value.copy() for key, value in anchor.items()}
            trial["rv"] += rng.normal(0, 30, trial["rv"].shape)
        else:
            trial = j_random_state(data, n_components, rng, sigma, False)
        starts.append(j_pack(trial, layout))
    result = j_solve(data, layout, starts, gaussian_config, joint_config, sigma)
    if result["best"]["chi2"] > shared_result["best"]["chi2"] + 1e-3:
        state = {key: value.copy() for key, value in anchor.items()}
        result["solutions"].append(dict(state=state,
            chi2=shared_result["best"]["chi2"], theta=j_pack(state, layout),
            nfev=0, success=True))
        result["solutions"].sort(key=itemgetter("chi2"))
        result["best"] = result["solutions"][0]
        result["embedded_shared_used"] = True
    return result


def j_metrics(data, joint, independent, n_components, branch, epoch, lines,
              gaussian_config, joint_config):
    state, independent_state = joint["best"]["state"], independent["best"]["state"]
    modes = j_modes(joint, joint_config)
    near = [s for s in joint["solutions"]
            if s["chi2"] <= joint["best"]["chi2"] + joint_config.near_delta]
    aligned = [j_match(solution["state"], state, joint_config)[0] for solution in near]
    rv_span = np.ptp(np.stack([a["rv"][0, 0] for a in aligned]), axis=0)
    centre = j_centres(data)[0]
    rows, components = [], []

    def bound_flags(value, lower, upper):
        return near_bound(value, lower, upper,
                          fraction=joint_config.bound_fraction, return_mask=True)

    for line_index, (line_name, profile) in enumerate(zip(lines, data[0])):
        core = np.abs(profile["v"] - profile["rv"]) <= joint_config.core_half
        residual = (profile["f"] - j_curve(profile, state, 0, line_index,
                                             gaussian_config)) / profile["noise"]
        independent_residual = (profile["f"] - j_curve(
            profile, independent_state, 0, line_index, gaussian_config)) / profile["noise"]
        core_rms = np.sqrt(np.mean(residual[core] ** 2))
        independent_core_rms = np.sqrt(np.mean(independent_residual[core] ** 2))
        continuum_at_bound = bool(
            bound_flags(state["cont"][0, line_index, 0],
                1 - joint_config.cont_offset, 1 + joint_config.cont_offset) or
            bound_flags(state["cont"][0, line_index, 1],
                -joint_config.cont_tilt, joint_config.cont_tilt))
        rows.append(dict(model=f"{n_components}G", train_branch=branch,
            epoch=epoch, line=line_name,
            joint_chi2=float(residual @ residual),
            independent_chi2=float(independent_residual @ independent_residual),
            joint_RMS_noise=float(np.sqrt(np.mean(residual ** 2))),
            joint_core_RMS_noise=float(core_rms),
            independent_core_RMS_noise=float(independent_core_rms),
            core_RMS_degradation=float(core_rms - independent_core_rms),
            continuum=state["cont"][0, line_index, 0],
            tilt=state["cont"][0, line_index, 1],
            continuum_bound=continuum_at_bound,
            joint_total_delta_chi2=joint["best"]["chi2"] - independent["best"]["chi2"],
            joint_near_modes=len(modes), max_near_RV_span=float(rv_span.max())))
        for component in range(n_components):
            overlap = min(abs(state["rv"][0, line_index, component] -
                              state["rv"][0, line_index, other]) /
                          np.hypot(state["sigma"][line_index, component],
                                   state["sigma"][line_index, other])
                          for other in range(n_components) if other != component)
            components.append(dict(model=f"{n_components}G", train_branch=branch,
                epoch=epoch, line=line_name, gaussian_component=component + 1,
                shared_RV=state["rv"][0, line_index, component],
                independent_RV=independent_state["rv"][0, line_index, component],
                depth=state["depth"][0, line_index, component],
                sigma_kms=state["sigma"][line_index, component],
                near_RV_span=rv_span[component],
                RV_bound=bool(bound_flags(state["rv"][0, line_index, component],
                    centre - gaussian_config.rv_range, centre + gaussian_config.rv_range)),
                sigma_bound=bool(bound_flags(state["sigma"][line_index, component],
                    *joint_config.sigma_bounds)),
                depth_bound=bool(bound_flags(state["depth"][0, line_index, component],
                    0., joint_config.depth_max)), overlap_ratio=overlap))
    return rows, components


def plot_joint_validation(profiles, validations, train_results, lines, validate,
                          component_counts, gaussian_config, joint_config):
    figures = []
    for n_components in component_counts:
        for line_index, line_name in enumerate(lines):
            fig, axes = plt.subplots(2, len(validate), figsize=(5 * len(validate), 7),
                squeeze=False, gridspec_kw={"height_ratios": [3, 1.5]})
            for col, epoch in enumerate(validate):
                ax, residual_ax = axes[:, col]
                profile = profiles[line_name, epoch]
                record = validations[n_components, 1, epoch]
                joint = record["joint"]["best"]["state"]
                independent = record["independent"]["best"]["state"]
                ax.plot(profile["v"], profile["f"], color="0.55", lw=0.8, label="Data")
                for branch in range(1, len(train_results[n_components]["branches"]) + 1):
                    alternate = validations[n_components, branch, epoch]["joint"]
                    for mode in j_modes(alternate, joint_config)[:3]:
                        ax.plot(profile["v"], j_curve(profile, mode["state"], 0,
                            line_index, gaussian_config), color="purple", lw=0.8, alpha=0.2)
                joint_model = j_curve(profile, joint, 0, line_index, gaussian_config)
                independent_model = j_curve(profile, independent, 0, line_index,
                                             gaussian_config)
                ax.plot(profile["v"], joint_model, "k", lw=1.3,
                        label="Joint: shared RVs")
                ax.plot(profile["v"], independent_model, color="darkorange",
                        ls="--", lw=1.2, label="Independent line RVs")
                continuum = (joint["cont"][0, line_index, 0] +
                    joint["cont"][0, line_index, 1] *
                    (profile["v"] - profile["rv"]) / gaussian_config.fit_half)
                shape = np.column_stack([joint["depth"][0, line_index],
                                         joint["sigma"][line_index]]).ravel()
                for component, term in enumerate(gaussian_terms(
                        profile["v"], shape, joint["rv"][0, line_index])):
                    ax.plot(profile["v"], continuum - term, ":", lw=0.8,
                            label=f"G{component + 1}: "
                                  f"{joint['rv'][0, line_index, component]:.1f} km/s")
                residual_ax.plot(profile["v"], profile["f"] - joint_model,
                                 "k", lw=0.8)
                residual_ax.plot(profile["v"], profile["f"] - independent_model,
                                 color="darkorange", lw=0.8, ls="--")
                residual_ax.axhspan(-profile["noise"], profile["noise"],
                                    color="0.5", alpha=0.15)
                residual_ax.axhline(0, color="0.4", lw=0.7)
                ax.set(title=f"{line_name}: {epoch}, {n_components}G",
                       ylabel="Input-normalised flux")
                residual_ax.set(ylabel="Data − model", xlabel="Velocity [km/s]")
                for panel in (ax, residual_ax):
                    panel.set_xlim(profile["rv"] - gaussian_config.fit_half,
                                   profile["rv"] + gaussian_config.fit_half)
                ax.legend(fontsize=6)
            low = min(ax.get_ylim()[0] for ax in axes[0])
            high = max(ax.get_ylim()[1] for ax in axes[0])
            residual_limit = max(max(abs(x) for x in ax.get_ylim())
                                 for ax in axes[1])
            for ax in axes[0]:
                ax.set_ylim(low, high)
            for ax in axes[1]:
                ax.set_ylim(-residual_limit, residual_limit)
            fig.suptitle(f"Held-out {n_components}G: shared versus independent line RVs")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            figures.append(fig)
    return figures


def run_joint_he(spectra, lines, train, validate, component_counts,
                 gaussian_config, joint_config, *, make_plots=True, verbose=True):
    """Jointly fit two He lines with shared RVs; compare independent line RVs.

    Widths are line-specific and trained across training epochs. Depths and
    local continuum terms are per epoch and line. The validation comparator
    changes only the RV constraint: it permits each line an independent RV.
    This is a Gaussian-profile comparison, not a physical binary/light-ratio fit.
    """
    train, validate = list(train), list(validate)
    component_counts = list(component_counts)
    mapping = _validate_epochs(spectra, train, validate)
    if len(lines) != 2 or not validate or not component_counts or any(
            n not in (2, 3) for n in component_counts):
        raise ValueError("Require two lines, validation epochs, and 2G/3G counts.")
    labels = train + validate
    if verbose:
        print(pd.DataFrame([dict(epoch=epoch, BJD=mapping[epoch].get("bjd"),
            file=mapping[epoch].get("file", mapping[epoch].get("filename", "")))
            for epoch in labels]).to_string(index=False))
        print(f"LOCAL_CONTINUUM={gaussian_config.local_continuum}; "
              f"explicit shift={gaussian_config.additional_shift_kms} km/s")
    profiles = {(line_name, epoch): measure_profile(mapping[epoch], rest, gaussian_config)
                for line_name, rest in lines.items() for epoch in labels}
    train_data = [[profiles[line_name, epoch] for line_name in lines]
                  for epoch in train]
    training, validations = {}, {}
    train_rows, rows, component_rows, mode_rows = [], [], [], []
    for n_components in component_counts:
        rng = np.random.default_rng(joint_config.seed + n_components * 1000)
        layout = j_layout(len(train), len(lines), n_components, True, True)
        starts = [j_pack(j_random_state(train_data, n_components, rng), layout)
                  for _ in range(joint_config.train_starts)]
        train_result = j_solve(train_data, layout, starts,
                               gaussian_config, joint_config)
        modes = j_modes(train_result, joint_config)
        branches = modes[:joint_config.max_train_branches]
        if not branches:
            raise ValueError("No retained TRAIN branches; check branch settings.")
        reference = j_reorder(branches[0]["state"],
            np.argsort(branches[0]["state"]["rv"][0, 0]))
        for branch in branches:
            branch["state"] = j_match(branch["state"], reference, joint_config)[0]
        train_result["branches"] = branches
        training[n_components] = train_result
        if verbose:
            print(f"{n_components}G: converged {len(train_result['solutions'])}/"
                  f"{train_result['attempted_starts']}; near TRAIN modes={len(modes)}; "
                  f"validating {len(branches)} branches.")
            if len(modes) > len(branches):
                print("WARNING: TRAIN branch cap reached; full solutions retained.")
        for branch_index, branch in enumerate(branches, 1):
            train_state = branch["state"]
            for line_index, line_name in enumerate(lines):
                for component in range(n_components):
                    train_rows.append(dict(model=f"{n_components}G", branch=branch_index,
                        line=line_name, gaussian_component=component + 1,
                        sigma_kms=train_state["sigma"][line_index, component],
                        train_delta_chi2=branch["chi2"] - train_result["best"]["chi2"],
                        sigma_bound=near_bound(train_state["sigma"][line_index, component],
                            *joint_config.sigma_bounds,
                            fraction=joint_config.bound_fraction)))
            for epoch_index, epoch in enumerate(validate):
                if verbose:
                    print(f"{n_components}G branch {branch_index}, {epoch}: "
                          "joint + independent RV comparison...")
                data = [[profiles[line_name, epoch] for line_name in lines]]
                epoch_rng = np.random.default_rng(
                    joint_config.seed + n_components * 1000 + branch_index * 100 + epoch_index)
                layout_val = j_layout(1, len(lines), n_components, True, False)
                epoch_starts = [j_pack(state, layout_val) for state in j_epoch_seed(
                    data, train_state["sigma"], train_state, epoch_rng, joint_config)]
                joint = j_solve(data, layout_val, epoch_starts,
                                gaussian_config, joint_config, train_state["sigma"])
                independent = j_independent(data, train_state["sigma"], joint, epoch_rng,
                                            gaussian_config, joint_config)
                validations[n_components, branch_index, epoch] = dict(
                    joint=joint, independent=independent)
                metric_rows, comps = j_metrics(data, joint, independent, n_components,
                    branch_index, epoch, list(lines), gaussian_config, joint_config)
                rows.extend(metric_rows)
                component_rows.extend(comps)
                for mode_index, mode in enumerate(j_modes(joint, joint_config), 1):
                    aligned, _ = j_match(mode["state"], joint["best"]["state"], joint_config)
                    row = dict(model=f"{n_components}G", train_branch=branch_index,
                        epoch=epoch, mode=mode_index,
                        delta_chi2=mode["chi2"] - joint["best"]["chi2"],
                        start_count=mode["start_count"])
                    row.update({f"G{j + 1}_RV": aligned["rv"][0, 0, j]
                                for j in range(n_components)})
                    mode_rows.append(row)
    widths = pd.DataFrame(train_rows)
    summary = pd.DataFrame(rows)
    components = pd.DataFrame(component_rows)
    mode_table = pd.DataFrame(mode_rows)
    branch_rows = []
    for n_components in component_counts:
        for epoch in validate:
            anchor = validations[n_components, 1, epoch]["joint"]["best"]["state"]
            states = [j_match(validations[n_components, branch, epoch]["joint"]["best"]["state"],
                              anchor, joint_config)[0]
                      for branch in range(1, len(training[n_components]["branches"]) + 1)]
            spans = np.ptp(np.stack([state["rv"][0, 0] for state in states]), axis=0)
            branch_rows.append(dict(model=f"{n_components}G", epoch=epoch,
                retained_train_branches=len(states),
                max_RV_span_across_branches=float(spans.max())))
    branch_table = pd.DataFrame(branch_rows)
    if verbose:
        for title, table in [("TRAIN WIDTHS / BRANCHES", widths),
            ("VALIDATION COMPARISON", summary),
            ("GAUSSIAN COMPONENT DIAGNOSTICS — NOT STELLAR IDENTIFICATIONS", components),
            ("VALIDATION MODES", mode_table),
            ("SENSITIVITY TO TRAIN BRANCHES", branch_table)]:
            print("\n" + title)
            print(table.to_string(index=False, float_format="%.3f"))
        print("No files written; no bootstrap or orbital fit is performed.")
        print("Near-mode, overlap and bound diagnostics are exploratory; they are "
              "not formal RV errors or detections.")
    figures = (plot_joint_validation(profiles, validations, training, list(lines),
                validate, component_counts, gaussian_config, joint_config)
               if make_plots else [])
    return dict(profiles=profiles, training=training, validation=validations,
        widths=widths, summary=summary, components=components, modes=mode_table,
        branch_sensitivity=branch_table, figures=figures)
