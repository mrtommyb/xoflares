import numpy as np
import theano
import theano.tensor as tt


# class FlareLightCurve(object):

#     def __init__(self):
#         pass

#     def get_light_curve(self, time=None, tpeaks=None,
#                         fwhms=None, ampls=None,
#                         oversample=7,
#                         texp=None):
#         self.time = time
#         self.tpeaks = tpeaks
#         self.fwhms = fwhms

#         if texp is not None:
#             # taking this oversample code from
#             # https://github.com/dfm/exoplanet
#             texp = tt.as_tensor_variable(texp)
#             oversample = int(oversample)
#             oversample += 1 - oversample % 2
#             stencil = np.ones(oversample)

#             # Construct the exposure time integration stencil
#             dt = np.linspace(-0.5, 0.5, 2*oversample+1)[1:-1:2]
#             stencil /= np.sum(stencil)

#             dt = texp * dt
#             tgrid = tt.shape_padright(time) + dt

#             lc = multiflaremodel(tgrid)

def get_light_curve(time, tpeaks, fwhms, ampls, texp=None, oversample=7):
    time = time.astype('float64')
    time = tt.as_tensor_variable(time)

    if texp is None:
        tgrid = time
    if texp is not None:
        # taking this oversample code from
        # https://github.com/dfm/exoplanet
        # and https://github.com/lkreidberg/batman
        oversample = int(oversample)
        oversample += 1 - oversample % 2
        dt = np.linspace(-texp / 2., texp / 2.,
                         oversample)
        tgrid = tt.shape_padright(time) + dt

    multiflare_lc = multiflaremodel(tgrid, tpeaks, fwhms, ampls)

    if texp is not None:
        multiflare_lc = tt.mean(tt.reshape(multiflare_lc, (-1, oversample)),
                                axis=1)

    return multiflare_lc


def multiflaremodel(time, tpeaks, fwhms, ampls):
    time = time.astype('float64')
    time = tt.as_tensor_variable(time)
    multiflare_lc = tt.zeros_like(time)

    def scan_func(tpeak, fwhm, ampl):
        zeropad_flare_lc = tt.zeros_like(time)
        tcut = (((time - tpeak) / fwhm > -1.) *
                ((time - tpeak) / fwhm < 20.)).nonzero()
        flare_lc = _flaremodel(time[tcut], tpeak, fwhm, ampl)
        zeropad_flare_lc = tt.set_subtensor(zeropad_flare_lc[tcut], flare_lc)
        return zeropad_flare_lc

    components, updates = theano.scan(fn=scan_func,
                                      sequences=[tpeaks, fwhms, ampls],
                                      )
    multiflare_lc = tt.sum(components, axis=0)

    return multiflare_lc


def _flaremodel(time, tpeak, fwhm, ampl):
    # reuses some code from AltaiPony and Apaloosa
    time = tt.as_tensor_variable(time)
    flare_lc = tt.zeros_like(time)
    flare_lc = tt.where((time <= tpeak) * ((time - tpeak) / fwhm > -1.),
                        _before_flare(time, tpeak, fwhm, ampl),
                        flare_lc
                        )
    flare_lc = tt.where((time > tpeak) * ((time - tpeak) / fwhm < 20.),
                        _after_flare(time, tpeak, fwhm, ampl),
                        flare_lc
                        )
    return flare_lc


def _before_flare(time, tpeak, fwhm, ampl):
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    fout = ((_fr[0] + _fr[1] * ((time - tpeak) / fwhm) +
             _fr[2] * ((time - tpeak) / fwhm)**2. +
             _fr[3] * ((time - tpeak) / fwhm)**3. +
             _fr[4] * ((time - tpeak) / fwhm)**4.) *
            ampl)
    return fout


def _after_flare(time, tpeak, fwhm, ampl):
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]
    fout = ((_fd[0] * tt.exp(((time - tpeak) / fwhm) * _fd[1]) +
             _fd[2] * tt.exp(((time - tpeak) / fwhm) * _fd[3])) *
            ampl)
    return fout


def multiflare(time, tpeaks, fwhms, ampls):

    timex = tt.dvector('timex')
    tpeaksx = tt.dvector('tpeaksx')
    fwhmsx = tt.dvector('fwhmsx')
    amplsx = tt.dvector('amplsx')
    multiflare_function = theano.function([timex, tpeaksx, fwhmsx, amplsx],
                                          multiflaremodel(timex,
                                                          tpeaksx, fwhmsx,
                                                          amplsx))
    return multiflare_function(time, tpeaks, fwhms, ampls)


# reference implementation in numpy
def get_light_curvenp(time, tpeaks, fwhms, ampls, texp=None, oversample=7):
    time = np.asarray(time, dtype=float)

    if texp is None:
        tgrid = time
    if texp is not None:
        # taking this oversample code from
        # https://github.com/dfm/exoplanet
        # and https://github.com/lkreidberg/batman
        texp = float(texp)
        oversample = int(oversample)
        oversample += 1 - oversample % 2
        dt = np.linspace(-texp / 2., texp / 2.,
                         oversample)
        tgrid = (dt + time.reshape(time.size, 1)).flatten()

    multiflare_lc = multiflaremodelnp(tgrid, tpeaks, fwhms, ampls)

    if texp is not None:
        multiflare_lc = np.mean(
            multiflare_lc.reshape(-1, oversample),
            axis=1)

    return multiflare_lc


def multiflaremodelnp(time, tpeaks, fwhms, ampls):
    time = np.asarray(time, dtype=float)
    tpeaks = np.asarray(tpeaks, dtype=float)
    fwhms = np.asarray(fwhms, dtype=float)
    ampls = np.asarray(ampls, dtype=float)
    multiflare_lc = np.zeros_like(time)
    npeaks = tpeaks.shape[0]
    for i in range(npeaks):
        flare_lc = _flaremodelnp(time, tpeaks[i], fwhms[i], ampls[i])
        multiflare_lc = multiflare_lc + flare_lc
    return multiflare_lc


def _flaremodelnp(time, tpeak, fwhm, ampl):
    # reuses some code from AltaiPony and Apaloosa
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]
    flare_lc = np.zeros_like(time)
    flare_lc = np.where((time <= tpeak) * ((time - tpeak) / fwhm > -1.),
                        (_fr[0] +
                         _fr[1] * ((time - tpeak) / fwhm) +
                         _fr[2] * ((time - tpeak) / fwhm)**2. +
                         _fr[3] * ((time - tpeak) / fwhm)**3. +
                         _fr[4] * ((time - tpeak) / fwhm)**4.) *
                        ampl,
                        flare_lc
                        )
    flare_lc = np.where((time > tpeak) * ((time - tpeak) / fwhm < 20.),
                        (_fd[0] * np.exp(((time - tpeak) / fwhm) * _fd[1]) +
                         _fd[2] * np.exp(((time - tpeak) / fwhm) * _fd[3])) *
                        ampl,
                        flare_lc
                        )
    return flare_lc
