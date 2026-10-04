import sys, glob, os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.interpolate import splrep, splev
import input_output as inout


class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self, fig, ax1, ax2, w, f, fname, knot_width):
        bounds = w[0], w[-1]
        self.raw_plot = ax1.plot(w, f, 'black', picker=5, zorder=1)
        ax1.set_xlim([bounds[0], bounds[1]])
        ax1.set_yscale('log')
        ax2.axhline(1, color='turquoise', zorder=999)
        ax2.set_ylim([0, 1.2])

        self.lastind = 0
        self.vertical_x_cen = []

        self.filename = fname
        self.wavelengths = w
        self.fluxes = f
        self.ax1 = ax1
        self.ax2 = ax2
        self.fig = fig

        self.w_min = bounds[0]
        self.w_max = bounds[1]
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]
        self.spline_range = np.arange(self.w_min, self.w_max+1, 1)
        self.knot_width = knot_width
        self.knot_half_width = self.knot_width/2
        self.flux_knots = []

    def onpress(self, event):
        '''
        click - select knot point
        d - delete knot point closest to cursor
        enter - fit and display spline
        n - normalize
        w - write knots file
        r - read in knots file
        '''
        if self.lastind is None:
            return
        if event.key not in ('n', 'd', 'r', 'w', 'enter'):
            return

        elif event.key == 'w':
            knotfname = (outfolder +
                         os.path.basename(self.filename).split('.fits')[0] +
                         '_knots.txt')
            np.savetxt(knotfname, self.vertical_x_cen)
            print('Writing new knots file to ' + knotfname)
            plt.close(self.fig)

        elif event.key == 'enter':
            self.fit_spline()

        elif event.key == 'r':
            try:  # reads in knots if available
                knots_file = glob.glob('*/norm/*knots.txt')[0]
                print("Reading in knots for a different star: " + knots_file)
                x = np.loadtxt(knots_file)
                self.vertical_x_cen = list(x)
                self.vertical_x_cen.sort()
                self.flux_knots = self.determine_spline_pairs(self.wavelengths,
                                                              self.fluxes)
            except Exception:
                print('Could not find knots file.')
                pass

        elif event.key == 'd':
            try:
                xe = event.xdata
                dif = abs(self.vertical_x_cen - xe)
                dif_ind = list(dif).index(min(dif))
                del self.vertical_x_cen[dif_ind]
                del self.flux_knots[dif_ind]
                self.flux_knots = [f for _, f in
                                   sorted(zip(self.vertical_x_cen,
                                              self.flux_knots))]
                self.vertical_x_cen.sort()
            except Exception:
                pass

        self.update()

    def onpick(self, event):
        if self.ax1.get_navigate_mode() is None:
            xe = event.xdata
            if self.w_min < xe < self.w_max:
                self.vertical_x_cen.append(xe)
            self.vertical_x_cen = [i for i in self.vertical_x_cen if
                                   i is not None]
            wavelength = self.wavelengths
            flux = self.fluxes
            inds = [i for i in range(len(wavelength)) if xe -
                    self.knot_half_width <=
                    wavelength[i] <= xe + self.knot_half_width]
            self.flux_knots.append(np.median(flux[inds]))
            self.flux_knots = [f for _, f in sorted(zip(self.vertical_x_cen,
                                                        self.flux_knots))]
            self.vertical_x_cen.sort()
            self.update()

    def update(self):
        if self.lastind is None:
            return
        try:
            self.knots.remove()
        except Exception:
            pass
        try:
            for i in self.knots:
                i.remove()
        except Exception:
            pass
        try:
            for i in self.knot_pairs:
                self.ax1.lines.remove(i)
        except Exception:
            pass

        self.knots = []

        for i in self.vertical_x_cen:
            self.knot = self.ax1.axvspan(i-self.knot_half_width,
                                         i + self.knot_half_width, alpha=0.3,
                                         color='C1')
            self.knots.append(self.knot)

        self.knot_pairs = self.ax1.plot(self.vertical_x_cen, self.flux_knots,
                                        'x', c='C1')
        self.fig.canvas.draw()

    def determine_spline_pairs(self, wavelength, flux):
        final_prespline_fluxes = []
        for cen in self.vertical_x_cen:
            inds = [i for i in range(len(wavelength)) if cen -
                    self.knot_half_width <=
                    wavelength[i] <= cen + self.knot_half_width]
            final_prespline_fluxes.append(np.median(flux[inds]))
        return final_prespline_fluxes

    def fit_spline(self):
        try:
            for i in self.spline_plot:
                self.ax1.lines.remove(i)
            for i in self.norm_plot:
                self.ax2.lines.remove(i)
        except Exception:
            pass

        wave_knots = self.spline_range
        spline_solution = splrep(self.vertical_x_cen, self.flux_knots, k=2)
        self.splinnormalization_fit_splinee_solution = spline_solution
        spline = splev(wave_knots, spline_solution)

        self.spline_plot = self.ax1.plot(wave_knots, spline, 'darkorange')

        spline_norm = splev(self.wavelengths, spline_solution)
        norm_flux = self.fluxes/spline_norm

        self.norm_plot = self.ax2.plot(self.wavelengths, norm_flux, 'k')


###############################################################################
def fit_spline(vertical_x_cen, wavelength, flux):
    flux_knots = []
    for cen in vertical_x_cen:
        inds = [i for i in range(len(wavelength)) if cen -
                knot_half_width <=
                wavelength[i] <= cen + knot_half_width]
        flux_knots.append(np.median(flux[inds]))

    # fit the spline through the knots
    spline_solution = splrep(vertical_x_cen, flux_knots, k=2)

    # normalize the flux
    spline_norm = splev(wave, spline_solution)
    norm_flux = flux/spline_norm

    return flux_knots, spline_norm, norm_flux


def make_diagnostic_plot(wave, flux, norm_flux, x_cen, flux_knots, spl,
                         outfilename):
    # make overall figure in A4 format
    fig_s = plt.figure(figsize=(8.27, 11.69), dpi=100, constrained_layout=True)

    # outer and inner grid
    outer = GridSpec(4, 1, figure=fig_s, wspace=0.2, hspace=0.2)

    # wavelength ranges
    ranges = [[4000, 4500], [4700, 5200], [6300, 6800], [8500, 9000]]
    snr_ranges = [[4430, 5070, 6750, 8810], [4450, 5090, 6770, 8830]]

    snrs = inout.estimate_snr(wave, norm_flux, snr_ranges[0], snr_ranges[1])
    snrs_return = []
    for i in range(len(ranges)):
        # measure the snr

        # mask arrays for plotting
        mask = (wave >= ranges[i][0]) & (wave <= ranges[i][1])
        k_mask = ((np.array(x_cen) >= ranges[i][0]) &
                  (np.array(x_cen) <= ranges[i][1]))

        inn = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], hspace=0.0)

        # axis 1: flux and normalized flux
        axi_1 = plt.Subplot(fig_s, inn[0])
        fig_s.add_subplot(axi_1)
        axi_1.plot(wave[mask], flux[mask], color='k')
        axi_1.plot(wave[mask], spl[mask], color='darkorange', alpha=0.7, lw=2)
        axi_1.plot(np.array(x_cen)[k_mask], np.array(flux_knots)[k_mask],
                   'x', c='darkorange')
        axi_1.set_ylabel('Flux')
        # axi_1.set_yscale('log')

        # axis 2: normalization
        axi_2 = plt.Subplot(fig_s, inn[1], sharex=axi_1)
        fig_s.add_subplot(axi_2)
        axi_2.plot(wave[mask], norm_flux[mask], color='k')
        axi_2.axhline(1, color='turquoise', alpha=0.7, lw=2)
        axi_2.set_ylabel('Norm. Flux')

        # if no S/N was measured: set it to NaN
        try:
            snr = str(round(snrs[i]))
        except ValueError:
            snr = 'NaN'
        snrs_return.append(snr)

        # plot the snr ranges
        axi_2.axvspan(snr_ranges[0][i], snr_ranges[1][i], color='k', alpha=0.2)
        axi_2.text(snr_ranges[0][i]-60, 1.2, 'S/N = ' + snr, color='k')

        axi_1.set_xlim(ranges[i])
        plt.setp(axi_1.get_xticklabels(), visible=False)
        axi_2.set_ylim(0.8, 1.3)

    axi_2.set_xlabel(r'Wavelength [$\AA$]')

    figname = outfilename.split('.fits')[0] + '.pdf'
    fig_s.savefig(figname)
    plt.close(fig_s)

    return snrs_return
###############################################################################


###############################################################################
# normalize spectra with the same knots, make diagnostic plots
###############################################################################
###############################################################################
# path definitions
if len(sys.argv) < 2:
    print("No input folder given. Exiting ...")
    exit()
infolder = sys.argv[1]

# check that folder has all requirements
if infolder[-1] != '/':
    infolder = infolder + '/'

if not infolder.endswith('/'):
    infolder = infolder + '/'
print("---------------------------------------------------------------------")
print("-                          NORMALIZATION                            -")
print("---------------------------------------------------------------------")

print("Input folder is: " + str(infolder))
outfolder = infolder + 'norm/'
print("Output folder is: " + str(outfolder))

# if output folder does not exist yet: create it
if not os.path.exists(outfolder):
    print('Making output directory: ' + outfolder)
    os.makedirs(outfolder)

knot_width = 3.0
knot_half_width = knot_width / 2

###############################################################################
# Files to normalize
# all files that are not normalized yet
norm_files = []
for f in glob.glob(infolder + '/*.fits'):
    norm_name = os.path.basename(f).split('.fits')[0] + '_norm.fits'
    if not os.path.isfile(outfolder + norm_name):
        norm_files.append(f)

if len(norm_files) == 0:
    print("No files to normalize. Check input folder " + infolder)
    exit()

print("---------------------------------------------------------------------")

###############################################################################

###############################################################################
# KNOTS
if len(glob.glob(outfolder + "*_knots.txt")) == 0:
    # if no knots file exists determine knots and save knots file
    print('No knots-file exists.... ')

    # make figure to select the knot points
    fig = plt.figure(1)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)

    ff = norm_files[0]
    wave, flux = inout.read_file(ff)
    print(f"\n=== DIAGNOSTIC INFO ===")
    print(f"First file: {ff}")
    print(f"Wave range: {wave[0]:.6f} to {wave[-1]:.6f}")
    print(f"Wave shape: {wave.shape}")
    print(f"Flux range: {flux.min():.6e} to {flux.max():.6e}")
    print(f"First 5 wavelengths: {wave[:5]}")
    print(f"First 5 fluxes: {flux[:5]}")
    print(f"Flux type: {type(flux)}")
    print(f"======================\n")
    browser = PointBrowser(fig, ax1, ax2, wave, flux, ff, knot_width)

    fig.canvas.mpl_connect('button_press_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)

    plt.show()


knot_file = glob.glob(outfolder + "*_knots.txt")[0]
print('Loading knots from file ' + knot_file)
print("---------------------------------------------------------------------")

wave_knots = np.array(np.loadtxt(knot_file))

# write an overview file with important information
outfnames, mjds, exptimes, snrs_b, snrs_g, snrs_r = [], [], [], [], [], []

logfilename = os.path.join(outfolder, 'norm_overview.txt')
logfile = open(logfilename, 'w')
logfile.write('normf_name,mjd,exptime,snr_b,snr_g,snr_r' + '\n')

###############################################################################
# NORMALIZATION
print('----------------------------------------------------------------------')
print('Number of spectra to normalize: ' + str(len(norm_files)))
for n, n_file in enumerate(norm_files):
    print('Normalizing spectrum ' + str(n) + '/' + str(len(norm_files)))

    filename = os.path.basename(n_file)
    wave, flux = inout.read_file(n_file)
    header = fits.header

    # check that all knots are within the defined wl range:
    k_mask = (wave_knots >= wave[0]) & (wave_knots <= wave[-1])
    w_knots = wave_knots[k_mask]
    f_knots, spline, norm_flux = fit_spline(w_knots, wave, flux)

    outfilename = (outfolder + filename.split('.fits')[0] + '_norm.fits')
    snrs = make_diagnostic_plot(wave, flux, norm_flux, w_knots, f_knots,
                                spline, outfilename)

    header = fits.getheader(n_file)
    if 'MJD-OBS' in header:
        mjd = header['MJD-OBS']
    else:
        try:
            mjd = header['JD-OBS'] - 2400000.5
        except:
            print("Fits header does not contain MJD-OBS or JD-OBS")

    line = (outfilename + ',' +
            str(mjd) + ',' +
            str(header['EXPTIME']) + ',' +
            snrs[0] + ',' + snrs[1] + ',' + snrs[2] + '\n')
    logfile.write(line)

    inout.write_any_spec(n_file, norm_flux, outfilename, wave=wave)
    np.savetxt(outfilename.split('_norm.fits')[0] + '_knots.txt', w_knots)

    print('Saved:  ' + outfilename.split('/')[-1])

print('Logfile written to ' + logfilename)
logfile.close()

print('----------------------------------------------------------------------')
print('Done with the normalization.')
print('----------------------------------------------------------------------')
