import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit

c_kms = 299792.458

def lomb_scargle_period(t, rv, err, fmin=1/2000, fmax=1/2, npp=10):
    freq, power = LombScargle(t, rv, err).autopower(
        minimum_frequency=fmin,
        maximum_frequency=fmax,
        samples_per_peak=npp,
    )
    return 1/freq, power

def sinusoid(t, gamma, K, phi, P):
    """Circular model – P is fixed when fitting."""
    return gamma + K*np.sin(2*np.pi*(t-t0)/P + phi)

def fit_fixed_period(t, rv, err, P):
    """Fit gamma, K, phi for a given period."""
    K0 = 0.5*(rv.max() - rv.min())
    def model(t, gamma, K, phi):
        return gamma + K*np.sin(2*np.pi*(t-t0)/P + phi)

    popt, pcov = curve_fit(
        model, t, rv,
        p0=[np.mean(rv), K0, 0.0],
        sigma=err, absolute_sigma=True,
        bounds=([100, 0, -np.pi], [250, 150, np.pi]),
        maxfev=50000,
    )
    perr = np.sqrt(np.diag(pcov))
    chi2 = np.sum(((rv - model(t, *popt))/err)**2)
    redchi = chi2/(len(rv)-3)
    return popt, perr, redchi

def mass_function(P_days, K_kms, e=0.0):
    """f(M) in solar masses (circular by default)."""
    G = 6.67430e-11          # SI
    M_sun = 1.98847e30
    return (P_days*86400)*(K_kms*1e3)**3*(1-e**2)**1.5/(2*np.pi*G)/M_sun
