import numpy as np
import theano
import pymc3 as pm
import theano.tensor as tt


def multiflaremodel(t, tpeaks, fwhms, ampls):
    t = t.astype('float64')
    t = tt.as_tensor_variable(t)
    multiflare_lc = tt.zeros_like(t)
    flare_lc = tt.zeros_like(t)

    def scan_func(tpeak, fwhm, ampl):
        zeropad_flare_lc = tt.zeros_like(t)
        tcut = (((t - tpeak) / fwhm > -1.) *
                ((t - tpeak) / fwhm < 20.)).nonzero()
        flare_lc = _flaremodel(t[tcut], tpeak, fwhm, ampl)
        zeropad_flare_lc = tt.set_subtensor(zeropad_flare_lc[tcut],  flare_lc)
        return zeropad_flare_lc

    components, updates = theano.scan(fn=scan_func,
                                      sequences=[tpeaks, fwhms, ampls],
                                      )
    multiflare_lc = tt.sum(components, axis=0)

    return multiflare_lc


def _flaremodel(t, tpeak, fwhm, ampl):
    # reuses some code from AltaiPony and Apaloosa
    t = tt.as_tensor_variable(t)
    flare_lc = tt.zeros_like(t)
    flare_lc = tt.where((t <= tpeak) * ((t - tpeak) / fwhm > -1.),
                        _before_flare(t, tpeak, fwhm, ampl),
                        flare_lc
                        )
    flare_lc = tt.where((t > tpeak) * ((t - tpeak) / fwhm < 20.),
                        _after_flare(t, tpeak, fwhm, ampl),
                        flare_lc
                        )
    return flare_lc


def _before_flare(t, tpeak, fwhm, ampl):
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    fout = ((_fr[0] + _fr[1] * ((t - tpeak) / fwhm) +
             _fr[2] * ((t - tpeak) / fwhm)**2. +
             _fr[3] * ((t - tpeak) / fwhm)**3. +
             _fr[4] * ((t - tpeak) / fwhm)**4.) *
            ampl)
    return fout


def _after_flare(t, tpeak, fwhm, ampl):
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]
    fout = ((_fd[0] * tt.exp(((t - tpeak) / fwhm) * _fd[1]) +
             _fd[2] * tt.exp(((t - tpeak) / fwhm) * _fd[3])) *
            ampl)
    return fout


def multiflare(x, tpeaks, fwhms, ampls):

    xx = tt.dvector('xx')
    tpeaksx = tt.dvector('tpeaksx')
    fwhmsx = tt.dvector('fwhmsx')
    amplsx = tt.dvector('amplsx')
    multiflare_function = theano.function([xx, tpeaksx, fwhmsx, amplsx],
                                          multiflaremodel(xx, tpeaksx, fwhmsx, amplsx))
    return multiflare_function(x, tpeaks, fwhms, ampls)


# reference implementation in numpy
def multiflaremodelnp(t, tpeaks, fwhms, ampls):
    multiflare_lc = np.zeros_like(t)
    npeaks = tpeaks.shape[0]
    for i in range(npeaks):
        flare_lc = _flaremodelnp(t, tpeaks[i], fwhms[i], ampls[i])
        multiflare_lc = multiflare_lc + flare_lc
    return multiflare_lc


def _flaremodelnp(t, tpeak, fwhm, ampl, oversample=10):
    # reuses some code from AltaiPony and Apaloosa
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]
    flare_lc = np.zeros_like(t)
    flare_lc = np.where((t <= tpeak) * ((t - tpeak) / fwhm > -1.),
                        (_fr[0] +
                         _fr[1] * ((t - tpeak) / fwhm) +
                         _fr[2] * ((t - tpeak) / fwhm)**2. +
                         _fr[3] * ((t - tpeak) / fwhm)**3. +
                         _fr[4] * ((t - tpeak) / fwhm)**4.) *
                        ampl,
                        flare_lc
                        )
    flare_lc = np.where((t > tpeak) * ((t - tpeak) / fwhm < 20.),
                        (_fd[0] * np.exp(((t - tpeak) / fwhm) * _fd[1]) +
                         _fd[2] * np.exp(((t - tpeak) / fwhm) * _fd[3])) *
                        ampl,
                        flare_lc
                        )
    return flare_lc
