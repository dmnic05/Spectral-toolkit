
"""
This script helps you in normalising a spectrum using the original
Vysakh / Michael interactive normaliser UI.

Controls (interactive plot):
  - Left click  : add continuum point (median-approx or free)
  - Right click : remove point if clicked exactly on it
  - Right-drag rectangle : select points -> press 'd' to delete
  - Enter       : fit spline through points and show normalized spectrum
  - w           : write normalized spectrum + PNG
  - r           : reset points
  - l           : load template (template.txt in template_dir)
  - j           : recalibrate loaded template y-values to current spectrum
  - m           : toggle median-approximation
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms
import re
import warnings
from scipy.interpolate import splrep, splev
from astropy.io import fits
import glob

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 14})



# -------------------------
# Helpers
# -------------------------

def find_closest_index(arr, target):
    """Return index of element in arr closest to target."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0
    return int(np.nanargmin(np.abs(arr - target)))


def load_fits_spectrum(path):
    """
    Load 1-D FITS spectrum and reconstruct wavelength.

    Tries two conventions:
      - log-lambda: wave = exp(CRVAL1 + (pix-CRPIX1)*CDELT1)
      - linear-lambda: wave = CRVAL1 + (pix-CRPIX1)*CDELT1

    Chooses the reconstruction whose wavelength range looks like Angstroms
    (2000..20000 A typical) and is monotonic. Raises if neither is plausible.
    Returns (wave, flux, mjd) where flux is 1-D.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with fits.open(path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    if data is None:
        raise ValueError("FITS primary HDU contains no data.")

    flux = np.asarray(data, dtype=float).squeeze()
    if flux.ndim != 1:
        try:
            flux = flux.ravel()
        except Exception:
            raise ValueError(f"Expected a 1-D spectrum, got shape {data.shape}.")

    crpix = float(header.get("CRPIX1", 1.0))
    crval = header.get("CRVAL1")
    cdelt = header.get("CDELT1")

    if crval is None or cdelt is None:
        if len(hdul) > 1 and hasattr(hdul[1], "data") and hdul[1].data is not None:
            w = np.asarray(hdul[1].data).squeeze()
            if w.ndim == 1 and len(w) == len(flux):
                valid = np.isfinite(w) & np.isfinite(flux)
                return w[valid], flux[valid], float(header.get("MJD-OBS", np.nan))
        raise KeyError("FITS header missing CRVAL1/CDELT1; cannot reconstruct wavelength.")

    crval = float(crval)
    cdelt = float(cdelt)

    pix = np.arange(len(flux), dtype=float) + 1.0

    # try log-lambda
    log_wave = crval + (pix - crpix) * cdelt
    wave_log = np.exp(log_wave)

    # try linear lambda
    wave_lin = crval + (pix - crpix) * cdelt

    def plausibility(w):
        if not np.all(np.isfinite(w)):
            return -np.inf
        if not np.all(np.diff(w) > 0):
            return -np.inf
        lo, hi = np.nanmin(w), np.nanmax(w)
        score = 0
        if (lo > 100.0) and (hi < 20000.0):
            score += 2
        if (lo > 3000.0) and (hi < 10000.0):
            score += 4
        rng = hi - lo
        if rng > 100.0:
            score += 1
        return score

    score_log = plausibility(wave_log)
    score_lin = plausibility(wave_lin)

    if score_log >= score_lin and score_log > -np.inf:
        wave = wave_log
    elif score_lin > score_log:
        wave = wave_lin
    else:
        if np.all(np.diff(wave_log) > 0) and (np.nanmin(wave_log) > 100.0):
            wave = wave_log
        elif np.all(np.diff(wave_lin) > 0) and (np.nanmin(wave_lin) > 100.0):
            wave = wave_lin
        else:
            raise RuntimeError("Could not determine wavelength scale (log vs linear). Inspect header manually.")

    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave, flux = wave[valid], flux[valid]

    try:
        mjd = float(header.get("MJD-OBS", np.nan))
    except Exception:
        mjd = np.nan

    return wave, flux, mjd


# -------------------------
# Interactive normaliser
# -------------------------

def medianApporx_str(bool_val):
    if bool_val:
        return r'$\bf{press\;m\;to\;disable\;median\;approximation}$'
    else:
        return r'$\bf{press\;m\;to\;enable\;median\;approximation}$'


def normalize_spectrum_interactive(wave, flux, filename="spectrum",
                                   output_dir="normalised",
                                   template_dir="templates",
                                   image_dir="images"):
    """
    Original-style interactive normaliser with the expected keys and mouse controls.
    Returns (wave, normalized, continuum) after the interactive window closes.
    """
    # ensure arrays
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    # create dirs
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # state
    pointCOUNTER = 0
    medianAPPROX = False
    selected_rectangles = []
    TEMPLATE_FILE = os.path.join(template_dir, "template.txt")

    # set up figure
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    def plot_vertical_line(axloc, x_position, text, c='blue', ypos=0.2, boxalpha=1.0, textalpha=0.6):
        axloc.axvline(x=x_position, color=c, linestyle='--', alpha=0.3, zorder=1)
        trans = mtransforms.blended_transform_factory(axloc.transData, axloc.transAxes)
        axloc.text(x_position, ypos, text,
                   color=c,
                   bbox=dict(facecolor='white', edgecolor='none', alpha=boxalpha),
                   ha='center', va='center', transform=trans,
                   rotation=90, fontsize=11, alpha=textalpha, zorder=2, weight='bold')

    def refreshPlot(x, y, contin=[None], point_coords=None, flag='p'):
        """
        Refresh the spectrum plot (upper) and normalized plot (lower).
        - contin: list with continuum array or [None]
        - point_coords: array Nx2 or None
        - flag: 'p' (plot only), 'n' (normalised)
        """
        nonlocal pointCOUNTER
        if point_coords is None:
            point_coords = []

        ax[0].cla()
        ax[0].set_xlabel(r'$Wavelength\;[\AA]$')
        ax[0].set_ylabel(r'$Flux\;[arbitrary]$')
        ax[0].plot(x, y, "k-", label='spectrum')

        # robust y-limits using percentiles
        try:
            ymin, ymax = np.nanpercentile(y, [1, 99])
            yrng = max(1e-6, ymax - ymin)
            ax[0].set_ylim(ymin - 0.05 * yrng, ymax + 0.05 * yrng)
        except Exception:
            ax[0].set_ylim(np.min(y) - 1000, np.max(y) + 1000)

        ax[0].set_xlim(np.min(x) - 100, np.max(x) + 100)
        ax[0].grid(axis='x')
        plot_vertical_line(ax[0], 6562.81, r'$H\alpha$', ypos=0.7, c='blue')
        plot_vertical_line(ax[0], 4861.35, r'$H\beta$', ypos=0.7, c='blue')
        plot_vertical_line(ax[0], 4340.472, r'$H\gamma$', ypos=0.7, c='blue')
        ax[0].set_title('Original Spectrum', fontsize=14, weight='bold')

        if len(point_coords) != 0:
            for k in range(len(point_coords)):
                ax[0].plot(point_coords[k, 0], point_coords[k, 1], 'rs', ms=6, label='point' + str(k))
            pointCOUNTER = len(point_coords)

        if (len(point_coords) >= 3) and (contin[0] is not None):
            ax[0].plot(wave, contin[0], "r-", label="continuum")

        # Normalised panel
        ax[1].cla()
        ax[1].set_title('Normalised spectrum', fontsize=14, weight='bold')
        ax[1].set_xlabel(r'$Wavelength\;[\AA]$')
        ax[1].set_ylabel(r'$Flux\;[arbitrary]$')
        plot_vertical_line(ax[1], 6562.81, r'$H\alpha$', ypos=0.1, c='blue')
        plot_vertical_line(ax[1], 4861.35, r'$H\beta$', ypos=0.1, c='blue')
        plot_vertical_line(ax[1], 4340.472, r'$H\gamma$', ypos=0.1, c='blue')

        if len(point_coords) < 3:
            ax[1].text(6400, 0.4, "Don't be too lazy!!\nselect atleast 3 points and then press ENTER!!",
                       fontsize=12, horizontalalignment='center')
        else:
            contin_arr = contin[0]
            if contin_arr is None:
                ax[1].text(6400, 0.4, "No continuum available.", fontsize=12, horizontalalignment='center')
            else:
                normalised = y / contin_arr
                ax[1].axhline(1, ls='-.', color='b', lw=2, label=r'$y = 1$')
                # label this line so 'w' handler can find it:
                ax[1].plot(x, normalised, "k-", label="normalised")
                # robust limits for normalised
                if normalised.size > 0:
                    try:
                        lo, hi = np.nanpercentile(normalised, [1, 99])
                        rng = max(1e-6, hi - lo)
                        ax[1].set_ylim(lo - 0.1 * rng, hi + 0.1 * rng)
                    except Exception:
                        ax[1].set_ylim(np.nanmin(normalised) - 0.5, np.nanmax(normalised) + 0.5)
                ax[1].grid(axis='x')
                ax[1].legend(loc='upper right')

        plt.suptitle(filename + '\n' + medianApporx_str(medianAPPROX), horizontalalignment='center')
        plt.draw()

    # click handler
    def onClick(event):
        nonlocal pointCOUNTER, medianAPPROX
        toolbar = plt.get_current_fig_manager().toolbar if plt.get_current_fig_manager() else None
        if event.inaxes != ax[0]:
            return
        if toolbar is not None and getattr(toolbar, "mode", "") != "":
            return

        if event.button == 1:  # left click
            if medianAPPROX:
                if event.xdata is None:
                    print("That click was outside the plot")
                else:
                    window = ((event.xdata - 0.25) <= wave) & (wave <= (event.xdata + 0.25))
                    if np.any(window):
                        yval = np.median(flux[window])
                        ax[0].plot(event.xdata, yval, 'rs', ms=6, picker=10, label='point' + str(pointCOUNTER))
                        pointCOUNTER += 1
                        plt.draw()
                    else:
                        print("No data in pick window around clicked x.")
            else:
                x = event.xdata
                y = event.ydata
                if x is None or y is None:
                    print("That click was outside the plot")
                else:
                    ax[0].plot(x, y, "rs", ms=6, picker=10, label="point" + str(pointCOUNTER))
                    pointCOUNTER += 1
                    plt.draw()

    # pick event: right-click on a point to remove it
    def onPick(event):
        try:
            if event.mouseevent.button == 3:
                label = getattr(event.artist, "get_label", lambda: "")()
                if label and ("point" in str(label)):
                    event.artist.remove()
                    plt.draw()
        except Exception:
            pass

    def is_point_inside_rectangle(x, y, rect_x, rect_y, rect_width, rect_height):
        return (rect_x <= x <= rect_x + rect_width) and (rect_y <= y <= rect_y + rect_height)

    def remove_items_inside_rectangle(rectangle):
        artist_list = []
        for artist in list(ax[0].get_children()):
            label = getattr(artist, "get_label", lambda: "")()
            if label and ("point" in str(label)):
                coords = artist.get_data()
                if len(coords[0]) and len(coords[1]):
                    x_pt, y_pt = coords[0][0], coords[1][0]
                    if is_point_inside_rectangle(x_pt, y_pt, rectangle.get_x(), rectangle.get_y(), rectangle.get_width(), rectangle.get_height()):
                        artist_list.append(artist)
        for art in artist_list:
            art.remove()

    # Rectangle selector callback
    def on_select(eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        rect = Rectangle((min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0), fill=False, edgecolor='red', label='rectangle')
        ax[0].add_patch(rect)
        selected_rectangles.append(rect)
        plt.draw()

    # keypress handler
    def onType(event):
        nonlocal pointCOUNTER, medianAPPROX
        key = event.key

        if key == "enter":
            # collect points
            point_coords = []
            for artist in ax[0].get_children():
                label = getattr(artist, "get_label", lambda: "")()
                if label and ("point" in str(label)):
                    coords = artist.get_data()
                    if len(coords[0]) != 0:
                        point_coords.append([coords[0][0], coords[1][0]])
                elif label and (label == "continuum"):
                    artist.remove()
            if len(point_coords) < 3:
                point_coords = np.array(point_coords)
                print(f"Not enough points were found. {len(point_coords)} < 3")
                refreshPlot(wave, flux, point_coords=point_coords, flag='n')
            else:
                point_coords = np.array(point_coords)
                # sort & remove NaNs
                point_coords = point_coords[np.argsort(point_coords[:, 0])]
                point_coords = point_coords[~np.isnan(point_coords).any(axis=1), :]
                # save template
                try:
                    np.savetxt(TEMPLATE_FILE, point_coords, header="wavelength continuum")
                except Exception as e:
                    print("Could not save template:", e)
                # fit spline (original behaviour)
                try:
                    spline = splrep(point_coords[:, 0], point_coords[:, 1], k=2)
                    continuum = splev(wave, spline)
                except Exception as e:
                    print("Spline failed, falling back to linear interp:", e)
                    continuum = np.interp(wave, point_coords[:, 0], point_coords[:, 1])
                refreshPlot(wave, flux, contin=[continuum], point_coords=point_coords, flag='n')

        elif key == "w":
            # find normalized line in lower axis and save
            # we labelled normalised line as "normalised" in refreshPlot via label argument
            # but matplotlib Line2D label isn't accessible via get_label() here, so find the first Line2D on ax[1] that isn't the horizontal line
            try:
                # reconstruct normalized array and save
                # find continuum used in last refresh if present:
                continuum = None
                for artist in ax[0].get_children():
                    label = getattr(artist, "get_label", lambda: "")()
                    if label == "continuum":
                        try:
                            cdata = artist.get_data()
                            if len(cdata[1]) == len(wave):
                                continuum = np.asarray(cdata[1])
                        except Exception:
                            pass
                if continuum is None and os.path.exists(TEMPLATE_FILE):
                    try:
                        pc = np.genfromtxt(TEMPLATE_FILE)
                        if pc.ndim == 1 and pc.size == 2:
                            pc = pc.reshape((1, 2))
                        if pc.shape[0] >= 3:
                            spline = splrep(pc[:, 0], pc[:, 1], k=2)
                            continuum = splev(wave, spline)
                    except Exception:
                        continuum = None
                if continuum is None:
                    print("No continuum found to save normalized spectrum.")
                    return
                normalised = flux / continuum
                stem = os.path.splitext(filename)[0]
                outtxt = os.path.join(output_dir, stem + "_norm.txt")
                np.savetxt(outtxt, np.column_stack([wave, normalised]), header="wavelength normalized_flux")
                print(f"Wrote to file {outtxt}")
                imagefile = os.path.join(image_dir, stem + "_norm.png")
                plt.savefig(imagefile, bbox_inches='tight')
                print("Saved image:", imagefile)
            except Exception as e:
                print("Could not save normalized data/image:", e)

        elif key == "r":
            # remove artists labelled point* and continuum
            for artist in list(ax[0].get_children()):
                label = getattr(artist, "get_label", lambda: "")()
                if label and (("point" in str(label)) or (label == "continuum")):
                    artist.remove()
            pointCOUNTER = 0
            refreshPlot(wave, flux)

        elif key == "l":
            # load template if exists
            if os.path.exists(TEMPLATE_FILE):
                try:
                    point_coords = np.genfromtxt(TEMPLATE_FILE)
                    if point_coords.ndim == 1 and point_coords.size == 2:
                        point_coords = point_coords.reshape((1, 2))
                    refreshPlot(wave, flux, point_coords=point_coords, flag='p')
                except Exception as e:
                    print("Failed to load template:", e)
            else:
                print("No template file found at", TEMPLATE_FILE)

        elif key == "j":
            # recalibrate loaded template y-values to current spectrum using median window
            point_coords = []
            for artist in list(ax[0].get_children()):
                label = getattr(artist, "get_label", lambda: "")()
                if label and ("point" in str(label)):
                    coords = artist.get_data()
                    if len(coords[0]) != 0:
                        point_coords.append([coords[0][0], coords[1][0]])
                elif label and (label == "continuum"):
                    artist.remove()

            if len(point_coords) == 0:
                print("No points to recalibrate.")
                return

            point_coords = np.array(point_coords)
            updated_points = []
            for val in point_coords[:, 0]:
                window = ((val - 0.25) <= wave) & (wave <= (val + 0.25))
                if np.any(window):
                    yval = np.median(flux[window])
                else:
                    yval = np.nan
                updated_points.append(yval)
            point_coords[:, 1] = np.array(updated_points)

            # remove old point artists
            for artist in list(ax[0].get_children()):
                label = getattr(artist, "get_label", lambda: "")()
                if label and (("point" in str(label)) or (label == "continuum")):
                    artist.remove()
            # replot updated points
            pointCOUNTER = 0
            for xval, yval in zip(point_coords[:, 0], point_coords[:, 1]):
                ax[0].plot(xval, yval, "rs", ms=6, picker=10, label="point" + str(pointCOUNTER))
                pointCOUNTER += 1
            plt.suptitle(filename + '\n' + medianApporx_str(medianAPPROX))
            plt.draw()

        elif key == "d":
            # remove points inside stored rectangles
            for rect in list(selected_rectangles):
                remove_items_inside_rectangle(rect)
                rect.set_visible(False)
            # remove rectangle artists
            for artist in list(ax[0].get_children()):
                if getattr(artist, "get_label", lambda: "")() == "rectangle":
                    artist.remove()
            selected_rectangles.clear()
            plt.draw()

        elif key == "m":
            medianAPPROX = not medianAPPROX
            plt.suptitle(filename + '\n' + medianApporx_str(medianAPPROX))
            plt.draw()

    # connect events
    fig.canvas.mpl_connect("button_press_event", onClick)
    fig.canvas.mpl_connect("pick_event", onPick)
    fig.canvas.mpl_connect("key_press_event", onType)
    rs = RectangleSelector(ax[0], on_select, useblit=True, button=[3], minspanx=5, minspany=5)

    # initial draw
    refreshPlot(wave, flux)

    plt.show()

    # after window closed: attempt to return last continuum and normalized array if present
    continuum = None
    # try to find continuum in ax[0] artists (if plotted)
    for artist in ax[0].get_children():
        if getattr(artist, "get_label", lambda: "")() == "continuum":
            try:
                cdata = artist.get_data()
                if len(cdata[1]) == len(wave):
                    continuum = np.asarray(cdata[1])
            except Exception:
                continuum = None
            break

    # fallback: if template exists, rebuild continuum from it
    if continuum is None and os.path.exists(TEMPLATE_FILE):
        try:
            pc = np.genfromtxt(TEMPLATE_FILE)
            if pc.ndim == 1 and pc.size == 2:
                pc = pc.reshape((1, 2))
            if pc.shape[0] >= 3:
                spline = splrep(pc[:, 0], pc[:, 1], k=2)
                continuum = splev(wave, spline)
        except Exception:
            continuum = None

    normalized = None if continuum is None else flux / continuum
    return wave, normalized, continuum

def normalize_with_template(wave, flux, template_file):
    """
    Normalize a spectrum using saved continuum-point wavelengths from a template file.
    
    The template file should have two columns: wavelength, continuum_value.
    For each template wavelength, we extract the median flux in a 0.25 Å window
    from the current spectrum, then fit a spline through those points.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Template file not found: {template_file}")

    template = np.genfromtxt(template_file)
    
    # handle single-point template
    if template.ndim == 1 and template.size == 2:
        template = template.reshape((1, 2))

    # Use only the continuum-point wavelengths from template
    x = template[:, 0]
    y = []

    for xi in x:
        window = np.abs(wave - xi) <= 0.25
        if np.any(window):
            y.append(np.median(flux[window]))
        else:
            y.append(np.nan)

    y = np.asarray(y)
    good = np.isfinite(y)
    x = x[good]
    y = y[good]

    if len(x) < 3:
        raise ValueError(f"Not enough valid template points (need ≥3, got {len(x)})")

    # fit spline through template points
    spline = splrep(x, y, k=2)
    continuum = splev(wave, spline)

    normalized = flux / continuum

    return wave, normalized, continuum

if __name__ == "__main__":
    """
    Run interactive normalisation on all .fits files in a directory (or a single file).
    Usage:
      python normalize.py /path/to/fits_dir
    or
      python normalize.py /path/to/file.fits
    """
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.getcwd()  # default: current directory

    fits_list = []
    if os.path.isdir(path):
        fits_list = sorted(glob.glob(os.path.join(path, "*.fits")))
        if not fits_list:
            print("No FITS files found in", path)
            sys.exit(1)
    elif os.path.isfile(path) and path.lower().endswith(".fits"):
        fits_list = [path]
    else:
        print("Provide a directory with .fits files or a single .fits file as argument.")
        sys.exit(1)

    # output dirs
    OUTPUT_DIR = os.path.join(path, "normalised")
    IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
    TEMPLATE_DIR = os.path.join(OUTPUT_DIR, "templates")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

    for fits_file in fits_list:
        print(f"\nOpening {os.path.basename(fits_file)}")
        try:
            wave, flux, mjd = load_fits_spectrum(fits_file)
        except Exception as e:
            print(f"Failed to load {fits_file}: {e}")
            continue

        # call the interactive normaliser (original UI function)
        w_out, normalized, continuum = normalize_spectrum_interactive(
            wave,
            flux,
            filename=os.path.basename(fits_file),
            output_dir=OUTPUT_DIR,
            template_dir=TEMPLATE_DIR,
            image_dir=IMAGE_DIR
        )

        # optionally save normalized result automatically after window closes
        if normalized is not None:
            stem = os.path.splitext(os.path.basename(fits_file))[0]
            outtxt = os.path.join(OUTPUT_DIR, stem + "_norm.txt")
            np.savetxt(outtxt, np.column_stack([w_out, normalized]), header="wavelength normalized_flux")
            print("Saved normalized spectrum to", outtxt)
        else:
            print("No normalized spectrum produced for", fits_file)

    print("\nAll done.")
