"""Shared spectrum I/O and utilities for the HD 195592 notebooks.

Importing this module does not load spectra, run fits, change plotting styles
or warning filters, create directories, display plots or save outputs.

The HERMES reader preserves the wavelength reconstruction in the supplied
RV notebook. No normalization or additional velocity correction is applied.

Dependencies: numpy and astropy.
Matplotlib is imported only when the figure-output helper is called.
"""

from pathlib import Path
import csv

import numpy as np
from astropy.io import fits


__all__ = [
    "C_KMS",
    "load_hermes",
    "read_spectrum",
    "discover_spectra",
    "locate",
    "doppler_rv",
    "wavelength_to_velocity",
    "velocity_to_wavelength",
    "write_csv",
    "save_figure",
]

C_KMS = 299792.458


# ============================================================
# OBSERVED SPECTRUM READING
# ============================================================

def load_hermes(path):
    """Return wavelength [Angstrom], flux and BJD.

    Preserves the original notebook's assumptions:
    - One-dimensional spectrum in the primary FITS HDU.
    - Natural-log wavelength reconstruction using CRVAL1/CDELT1.
    - One-based pixels; CRPIX1 defaults to 1.
    - Missing BJD returns NaN.
    - Invalid samples are removed and wavelengths sorted.

    No normalization, air/vacuum conversion, barycentric correction,
    duplicate averaging or resampling is performed.
    """
    path = Path(path)

    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header.copy()
        flux = np.asarray(hdul[0].data, dtype=float).squeeze()

    if flux.ndim != 1:
        raise ValueError(f"{path.name}: expected a 1D spectrum.")

    pixels = np.arange(flux.size, dtype=float) + 1.0

    wave = np.exp(
        float(header["CRVAL1"])
        + (pixels - float(header.get("CRPIX1", 1.0)))
        * float(header["CDELT1"])
    )

    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave = wave[valid]
    flux = flux[valid]

    order = np.argsort(wave)

    return (
        wave[order],
        flux[order],
        float(header.get("BJD", np.nan)),
    )


def read_spectrum(path):
    """Alias for load_hermes; returns (wave, flux, BJD).

    Supports existing expressions such as read_spectrum(path)[:2].

    This is specifically a HERMES observed-spectrum reader, not a generic
    ASCII or synthetic-template reader.
    """
    return load_hermes(path)


# ============================================================
# FILE DISCOVERY AND PATH RESOLUTION
# ============================================================

def discover_spectra(spectra_dir):
    """Return sorted FITS files directly inside an explicit directory.

    Uses the original notebook's *.fits, *.fit and *.fts patterns.
    Does not recurse, load headers or select epochs.
    Returns an empty list if no matching files exist.
    """
    spectra_dir = Path(spectra_dir)

    if not spectra_dir.is_dir():
        raise FileNotFoundError(
            f"Spectrum directory does not exist: {spectra_dir}"
        )

    return sorted(
        list(spectra_dir.glob("*.fits"))
        + list(spectra_dir.glob("*.fit"))
        + list(spectra_dir.glob("*.fts"))
    )


def locate(value, spectra_dir):
    """Resolve a spectrum filename using an explicit search directory.

    Resolution order:
    1. Accept an existing absolute file path.
    2. For relative paths, try spectra_dir / value.
    3. Search recursively within spectra_dir for the exact basename,
       requiring one distinct resolved match.

    Relative paths are not resolved against the notebook working directory.
    Existing absolute paths may be outside spectra_dir.
    Ambiguous matches raise an error rather than choosing silently.

    This is a new helper, not a copy of the unseen original locate().
    Intended for the local Windows paths used in this project.
    """
    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "<na>"}:
        raise ValueError("Missing spectrum filename/path.")

    supplied = Path(text)

    if supplied.is_absolute() and supplied.is_file():
        return supplied.resolve()

    spectra_dir = Path(spectra_dir)

    if not spectra_dir.is_dir():
        raise FileNotFoundError(
            f"Spectrum directory does not exist: {spectra_dir}"
        )

    if not supplied.is_absolute():
        candidate = spectra_dir / supplied

        if candidate.is_file():
            return candidate.resolve()

    # Exact filename comparison avoids interpreting filename characters
    # as wildcard patterns.
    matches = sorted({
        candidate.resolve()
        for candidate in spectra_dir.rglob("*")
        if candidate.is_file() and candidate.name == supplied.name
    })

    if not matches:
        raise FileNotFoundError(
            f"Could not locate {text!r} inside {spectra_dir}"
        )

    if len(matches) > 1:
        listing = "\n".join(str(match) for match in matches)

        raise ValueError(
            f"Ambiguous spectrum basename {supplied.name!r}; "
            f"matches:\n{listing}"
        )

    return matches[0]


# ============================================================
# CLASSICAL VELOCITY CONVERSIONS
# ============================================================

def doppler_rv(lambda_obs, lambda_rest):
    """Classical RV in km/s, preserving the original rvfuncs algebra.

    Both wavelengths must use the same units and air/vacuum convention.
    """
    return C_KMS * (lambda_obs - lambda_rest) / lambda_rest


def wavelength_to_velocity(wave, lambda_rest, *, shift_kms=0.0):
    """Convert wavelength to classical velocity and add an explicit shift.

    Mirrors the profile-diagnostic expression:
        C_KMS * (wave / lambda_rest - 1) + shift_kms

    No automatic frame correction or relativistic transform is applied.
    """
    wave = np.asarray(wave, dtype=float)

    return C_KMS * (wave / lambda_rest - 1.0) + shift_kms


def velocity_to_wavelength(velocity, lambda_rest, *, shift_kms=0.0):
    """Invert wavelength_to_velocity using the same explicit shift.

    With shift_kms=0, reproduces the original profile-plot wavelength axis.
    """
    velocity = np.asarray(velocity, dtype=float)

    return lambda_rest * (
        1.0 + (velocity - shift_kms) / C_KMS
    )


# ============================================================
# EXPLICIT OUTPUT HELPERS
# ============================================================

def write_csv(path, rows):
    """Write dictionary rows using the original CSV conventions.

    Field order follows the first row.
    Existing files are overwritten when explicitly called.
    Directories are not created automatically.
    Empty rows do not create or overwrite a file.
    """
    path = Path(path)

    if not rows:
        print(f"No rows to save: {path.name}")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_figure(
    fig,
    output_dir,
    name,
    *,
    save_figures=False,
    show_figures=True,
):
    """Save PNG/PDF and show or close a figure using explicit flags.

    Retains the original output settings:
    - PNG: 200 dpi, tight bounding box.
    - PDF: tight bounding box.

    Saving defaults OFF. Pass the notebook's SAVE_FIGURES and SHOW_FIGURES
    explicitly to retain its configured behaviour.

    Existing files are overwritten when saving is requested.
    Directories are not created automatically.
    plt.show() may display all open figures, as in the original helper.
    """
    import matplotlib.pyplot as plt

    if save_figures:
        if output_dir is None:
            raise ValueError(
                "output_dir is required when save_figures=True."
            )

        output_dir = Path(output_dir)

        fig.savefig(
            output_dir / f"{name}.png",
            dpi=200,
            bbox_inches="tight",
        )
        fig.savefig(
            output_dir / f"{name}.pdf",
            bbox_inches="tight",
        )

    if show_figures:
        plt.show()
    else:
        plt.close(fig)
