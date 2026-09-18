import numpy as np
import astropy.io.fits as fits
import sys
from astropy.table import Table
import matplotlib.pyplot as plt

def read_file(infile):
    """
    read-in spectra and models in different formats
    format from file extension (.fits, .ascii, ...) or keywords in fits header
    """

    # get filename extension
    ext = str(infile.split('.')[-1])

    # depending on extension, use different read function
    if (ext == 'fits'):  # standard fits
        wave, flux = read_fits(infile)

    elif (ext == 'mt'):  # FEROS data where fits files are called mt
        wave, flux = read_fits(infile)

    elif (ext == 'dat' or ext == 'ascii' or ext == 'txt' or ext == 'nspec' or ext == 'fl'):
        wave, flux = read_ascii(infile)  # plain two-column txt

    elif (ext == 'fit'):  # (old) fit files
        wave, flux = read_fits(infile)

    else:  # none of the above
        try:
            wave, flux = np.loadtxt(infile, unpack=True)
        except:
            print("ERROR: Could not read the input file - unknown extension.")
            sys.exit()

    return wave, flux


def read_fits(infile):
    header = fits.getheader(infile)

    if 'HIERARCH SPECTRUM STAR-ID' in header:
        wave, flux = read_psfSpec(infile)

    elif 'DR_NUM' in header:
        wave, flux = read_flames(infile)

    elif 'INSTRUME' in header:
        ins = header['INSTRUME']

        if (ins == 'UVES'):
            wave, flux = read_UVES(infile)

        elif (ins == 'GIRAFFE'):
            wave, flux = read_UVES(infile)
        
        elif (ins == 'FEROS'):
            wave, flux = read_FEROS(infile)
        
        elif (ins == 'HERMES'):  # <-- ADD THIS
            wave, flux = read_HERMES(infile)

        else:  # i.e. BeSS spectra
            wave, flux = read_psfSpec(infile)

    else:
        wave, flux = read_psfSpec(infile)

    return wave, flux


def read_psfSpec(infile):
    print("%s: Input file is a PSF extracted file." % infile)
    try:
        header = fits.getheader(infile)
        flux = fits.getdata(infile, ext=0)
        err = fits.getdata(infile, ext=1)  # read error but don't use it
        wl0 = header['CRVAL1']
        delt = header['CDELT1']
        pix = header['CRPIX1']
        wave = wl0 - (delt * pix - delt) + np.arange(flux.shape[0]) * delt
    except IndexError:
        header = fits.getheader(infile)
        flux = fits.getdata(infile)
        wl0 = header['CRVAL1']
        delt = header['CDELT1']
        pix = header['CRPIX1']
        wave = wl0 - (delt * pix - delt) + np.arange(flux.shape[0]) * delt

    return wave, flux



def read_MUSE(infile):
    print("%s: Input file is a MUSE file." % infile)
    header = fits.getheader(infile)
    data = fits.getdata(infile)
    wl0 = header['CRVAL1']  # Starting wl at CRPIX1
    delt = header['CDELT1']  # Stepwidth of wl
    pix = header['CRPIX1']  # Reference Pixel
    wave = wl0 - (delt * pix - delt) + np.arange(data.shape[0]) * delt

    flux = data

    return wave, flux


def read_FLAMES_n(infile):
    print("%s: Input file is a FLAMES (norm) file." % infile)
    data = fits.getdata(infile, 1)
    wave = data.field('WAVELENGTH')
    flux = data.field('NORM_SKY_SUB_CR')
    return wave, flux


def read_flames(infile):
    print("%s: Input file is a FLAMES file." % infile)
    data = fits.getdata(infile, 1)
    wave = data.field('WAVELENGTH')
    
    try: 
        flux = data.field('Sci_NORM')
    except KeyError or ValueError:
        flux = data.field('SCI_SKYSUB')

    return wave, flux

def read_FEROS(infile):
    print("%s: Input file is a standard fits (FEROS) file." % infile)
    header = fits.getheader(infile)
    
    if 'CRVAL1' in header: 
        flux = fits.getdata(infile) # flux stored in primary extension
        wl0 = header['CRVAL1']  # Starting wl at CRPIX1
        delt = header['CDELT1']  # Stepwidth of wl
        pix = header['CRPIX1']  # Reference Pixel
    
        # create wavelength array
        wave = wl0 - (delt * pix - delt) + np.arange(flux.shape[0]) * delt

    else:
        #print('Caution: unusual format.')
        data = fits.getdata(infile)
        wave = data.field(0)[0]
        flux = data.field(1)[0]

    return wave, flux

def read_HERMES(infile):
    print("%s: Input file is a HERMES file." % infile)
    header = fits.getheader(infile)
    flux = fits.getdata(infile)
    
    naxis1 = header['NAXIS1']
    crval1 = float(header['CRVAL1'])
    cdelt1 = float(header['CDELT1'])
    crpix1 = float(header.get('CRPIX1', 1.0))
    
    # Log-wavelength calibration: wave = exp(CRVAL1 + (pix - CRPIX1) * CDELT1)
    pix = np.arange(naxis1, dtype=float) + 1.0
    log_wave = crval1 + (pix - crpix1) * cdelt1
    wave = np.exp(log_wave)
    
    return wave, flux

def read_UVES(infile):
    print("%s: Input file is a UVES or GIRAFFE file." % infile)
    header = fits.getheader(infile)
    if 'SPEC_COM' in header:
        table = Table.read(infile, hdu=1)
        wave = table['wave']
        flux = table['flux']
    elif 'CRVAL1' in header:
        crval = header['CRVAL1']
        cdelt = header['CDELT1']
        naxis1 = header['NAXIS1']
        wave = crval + np.arange(0, naxis1) * cdelt
        flux = fits.getdata(infile)
    else:
        print('xx')
        data = fits.getdata(infile)
        wave = data.field(0)[0]
        flux = data.field(1)[0]
        # err = data.field(2)[0]

    if header['CUNIT1'] == 'm':
        wave = wave * 10**10 # convert wavelength to A

    elif wave[0] < 1000:  # wavelength is given in nm
        wave = wave * 10 # convert wavelength to A

    return wave, flux

def read_ascii(infile):
    # any type of ascii file (typically I call them .dat)
    # assumes that first column is wave and second column is flux
    print("%s: Input file is an ascii file." % infile)
    with open(infile) as f:
        if ',' in f.readline():
            wave, flux = np.genfromtxt(infile, delimiter=',').transpose()
        else:

            wave, flux = np.loadtxt(infile, usecols=(0, 1), unpack=True)

    return wave, flux


def write_ascii(wave, flux, outfilename, err=False):
    if err is False:
        np.savetxt(outfilename, np.array([wave, flux]).T)
    else:
        np.savetxt(outfilename, np.array([wave, flux, err]).T)
    print("Data written to %s" % outfilename)


# def write_any_spec(infile, flux, outfile):
#     hdul_infile = fits.open(infile)
#     hdul_new = fits.HDUList()
#     primheader = hdul_infile[0].header.copy()
#     hdul_new.append(fits.PrimaryHDU(data=flux, header=primheader))
#     hdul_new.writeto(outfile)

#     print("Data written to %s" % outfile)

def write_any_spec(infile, flux, outfile, wave=None):

    print(f"Writing fits file for {infile}")
    hdul_infile = fits.open(infile)
    hdul_new = fits.HDUList()
    primheader = hdul_infile[0].header.copy()
    hdul_new.append(fits.PrimaryHDU(data=flux, header=primheader))

    if wave is not None:
        print(f"Wave column is present, writing wave and flux column")
        col_wave = fits.Column(name='WAVELENGTH', array=wave, format='D')
        col_flux = fits.Column(name='FLUX', array=flux, format='D')
        table_hdu = fits.BinTableHDU.from_columns([col_wave, col_flux])
        hdul_new.append(table_hdu)

    hdul_new.writeto(outfile)

    print("Data written to %s" % outfile)



def get_key_from_header(infile, key, ext=0):
    header = fits.getheader(infile, ext=ext)
    info = header[key]
    return info


def estimate_snr(wave, flux, wl_begins=[5240, 6080, 6740],
                 wl_ends=[5320, 6160, 6820], plot_regions=False,
                 verbose=False):
    # if no other wavelenghts are given: calculate SNR in three regions
    # blue (5310 - 5450 A), green (6700 - 6840 A), red (7100 - 7240 A)
    wls, snrs = [], []

    for i in range(len(wl_begins)):
        cut_range = ((wave > wl_begins[i]) & (wave < wl_ends[i]))
        cut_wave, cut_flux = wave[cut_range], flux[cut_range]
        mean_wl = np.mean(cut_wave)

        noise = np.std(cut_flux)  # noise = sqrt(variance) = stddev
        signal = np.median(cut_flux)  # median of signal in the same range
        snr = signal / noise

        wls.append(mean_wl)
        snrs.append(snr)

        if verbose is True:
            print("SNR = %i in wavelength region from %i - %i A." %
                  (snr, wl_begins[i], wl_ends[i]))

        if plot_regions is True:
            fig, ax = plt.subplots()
            ax.plot(cut_wave, cut_flux)
            plt.show()

    return snrs

