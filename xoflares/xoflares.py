import numpy as np
import theano as aesara
import theano.tensor as tt
from scipy import integrate

# theano.config.scan.allow_gc = True

# class FlareLightCurve(object):
#     "at some point soon I'll make this a class"

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
    time = time.astype("float64")
    time = tt.as_tensor_variable(time)

    if texp is None:
        tgrid = time
    if texp is not None:
        # taking this oversample code from
        # https://github.com/dfm/exoplanet
        # and https://github.com/lkreidberg/batman
        oversample = int(oversample)
        oversample += 1 - oversample % 2
        dt = np.linspace(-texp / 2.0, texp / 2.0, oversample)
        tgrid = tt.shape_padright(time) + dt

    multiflare_lc = multiflaremodel(tgrid, tpeaks, fwhms, ampls)

    if texp is not None:
        multiflare_lc = tt.mean(
            tt.reshape(multiflare_lc, (-1, oversample)), axis=1
        )
    return multiflare_lc


def multiflaremodel(time, tpeaks, fwhms, ampls):
    # time = time.astype("float64")
    # time = tt.as_tensor_variable(time)
    multiflare_lc = tt.zeros_like(time)

    def scan_func(tpeak, fwhm, ampl, sum_to_date, time):
        zeropad_flare_lc = tt.zeros_like(time)
        tcut = (
            ((time - tpeak) / fwhm > -1.0) * ((time - tpeak) / fwhm < 20.0)
        ).nonzero()
        flare_lc = _flaremodel(time[tcut], tpeak, fwhm, ampl)
        zeropad_flare_lc = tt.set_subtensor(zeropad_flare_lc[tcut], flare_lc)
        return zeropad_flare_lc + sum_to_date

    components, updates = aesara.scan(
        fn=scan_func, sequences=[tpeaks, fwhms, ampls],
        non_sequences=time, outputs_info=tt.zeros_like(time),
    )
    multiflare_lc = components[-1]

    return multiflare_lc


# def multiflaremodel(time, tpeaks, fwhms, ampls):
#     time = time.astype("float64")
#     time = tt.as_tensor_variable(time)
#     multiflare_lc = tt.zeros_like(time)

#     def scan_func(tpeak, fwhm, ampl, time):
#         zeropad_flare_lc = tt.zeros_like(time)
#         tcut = (
#             ((time - tpeak) / fwhm > -1.0) * ((time - tpeak) / fwhm < 20.0)
#         ).nonzero()
#         flare_lc = _flaremodel(time[tcut], tpeak, fwhm, ampl)
#         zeropad_flare_lc = tt.set_subtensor(zeropad_flare_lc[tcut], flare_lc)
#         return zeropad_flare_lc

#     components, updates = theano.scan(
#         fn=scan_func, sequences=[tpeaks, fwhms, ampls],
#         non_sequences=time, outputs_info=None,
#     )
#     multiflare_lc = tt.sum(components, axis=0)
#     return multiflare_lc


def _flaremodel(time, tpeak, fwhm, ampl):
    # reuses some code from AltaiPony and Apaloosa
    time = tt.as_tensor_variable(time)
    flare_lc = tt.zeros_like(time)
    flare_lc = tt.switch(
        (time <= tpeak) * ((time - tpeak) / fwhm > -1.0),
        _before_flare(time, tpeak, fwhm, ampl),
        flare_lc,
    )
    flare_lc = tt.switch(
        (time > tpeak) * ((time - tpeak) / fwhm < 20.0),
        _after_flare(time, tpeak, fwhm, ampl),
        flare_lc,
    )
    return flare_lc


def _before_flare(time, tpeak, fwhm, ampl):
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    fout = (
        _fr[0]
        + _fr[1] * ((time - tpeak) / fwhm)
        + _fr[2] * ((time - tpeak) / fwhm) ** 2.0
        + _fr[3] * ((time - tpeak) / fwhm) ** 3.0
        + _fr[4] * ((time - tpeak) / fwhm) ** 4.0
    ) * ampl
    return fout


def _after_flare(time, tpeak, fwhm, ampl):
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]
    fout = (
        _fd[0] * tt.exp(((time - tpeak) / fwhm) * _fd[1])
        + _fd[2] * tt.exp(((time - tpeak) / fwhm) * _fd[3])
    ) * ampl
    return fout


def multiflare(time, tpeaks, fwhms, ampls):
    multiflare_function = aesara.function(
        [],
        multiflaremodel(tt.as_tensor_variable(time),
            tt.as_tensor_variable(tpeaks), tt.as_tensor_variable(fwhms),
            tt.as_tensor_variable(ampls)),
    )
    return multiflare_function()

    # timex = tt.dvector("timex")
    # tpeaksx = tt.dvector("tpeaksx")
    # fwhmsx = tt.dvector("fwhmsx")
    # amplsx = tt.dvector("amplsx")
    # multiflare_function = theano.function(
    #     [timex, tpeaksx, fwhmsx, amplsx],
    #     multiflaremodel(timex, tpeaksx, fwhmsx, amplsx),
    # )
    # return multiflare_function(time, tpeaks, fwhms, ampls)


def eval_get_light_curve(time, tpeaks, fwhms, ampls, texp=None, oversample=7):
    # timex = tt.dvector("timex")
    # tpeaksx = tt.dvector("tpeaksx")
    # fwhmsx = tt.dvector("fwhmsx")
    # amplsx = tt.dvector("amplsx")
    # multiflare_function = theano.function(
    #     [timex, tpeaksx, fwhmsx, amplsx],
    #     get_light_curve(timex, tpeaksx, fwhmsx, amplsx, texp, oversample),
    # )
    # return multiflare_function(time, tpeaks, fwhms, ampls)
    multiflare_function = aesara.function(
        [],
        get_light_curve(tt.as_tensor_variable(time),
            tt.as_tensor_variable(tpeaks), tt.as_tensor_variable(fwhms),
            tt.as_tensor_variable(ampls), texp, oversample)
    )
    return multiflare_function()


def _flareintegral(fwhm, ampl):
    t0, t1, t2 = -1, 0, 20
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    def get_int_before(x):
        integral = (
            _fr[0] * x
            + (_fr[1] * x ** 2 / 2)
            + (_fr[2] * x ** 3 / 3)
            + (_fr[3] * x ** 4 / 4)
            + (_fr[4] * x ** 5 / 5)
        )
        return integral

    def get_int_after(x):
        integral = (_fd[0] / _fd[1] * tt.exp(_fd[1] * x)) + (
            _fd[2] / _fd[3] * tt.exp(_fd[3] * x)
        )
        return integral

    before = get_int_before(t1) - get_int_before(t0)
    after = get_int_after(t2) - get_int_after(t1)
    return (before + after) * ampl * fwhm


def multiflareintegral(fwhms, ampls):
    components, updates = aesara.scan(
        fn=_flareintegral, sequences=[fwhms, ampls]
    )
    return components


def eval_multiflareintegral(fwhms, ampls):
    fwhmsx = tt.dvector("fwhmsx")
    amplsx = tt.dvector("amplsx")
    multintegral_function = aesara.function(
        [fwhmsx, amplsx], multiflareintegral(fwhmsx, amplsx)
    )
    return multintegral_function(fwhms, ampls)


# reference implementation in numpy
def get_light_curvenp(time, tpeaks, fwhms, ampls, texp=None, oversample=7):
    time = np.asarray(time, dtype=float)

    tpeaks = np.atleast_1d(tpeaks)
    fwhms = np.atleast_1d(fwhms)
    ampls = np.atleast_1d(ampls)

    if texp is None:
        tgrid = time
    if texp is not None:
        # taking this oversample code from
        # https://github.com/dfm/exoplanet
        # and https://github.com/lkreidberg/batman
        texp = float(texp)
        oversample = int(oversample)
        oversample += 1 - oversample % 2
        dt = np.linspace(-texp / 2.0, texp / 2.0, oversample)
        tgrid = (dt + time.reshape(time.size, 1)).flatten()

    multiflare_lc = multiflaremodelnp(tgrid, tpeaks, fwhms, ampls)

    if texp is not None:
        multiflare_lc = np.mean(multiflare_lc.reshape(-1, oversample), axis=1)

    return multiflare_lc


def multiflaremodelnp(time, tpeaks, fwhms, ampls):
    time = np.asarray(time, dtype=float)
    tpeaks = np.atleast_1d(tpeaks)
    fwhms = np.atleast_1d(fwhms)
    ampls = np.atleast_1d(ampls)
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
    flare_lc = np.where(
        (time <= tpeak) * ((time - tpeak) / fwhm > -1.0),
        (
            _fr[0]
            + _fr[1] * ((time - tpeak) / fwhm)
            + _fr[2] * ((time - tpeak) / fwhm) ** 2.0
            + _fr[3] * ((time - tpeak) / fwhm) ** 3.0
            + _fr[4] * ((time - tpeak) / fwhm) ** 4.0
        )
        * ampl,
        flare_lc,
    )
    flare_lc = np.where(
        (time > tpeak) * ((time - tpeak) / fwhm < 20.0),
        (
            _fd[0] * np.exp(((time - tpeak) / fwhm) * _fd[1])
            + _fd[2] * np.exp(((time - tpeak) / fwhm) * _fd[3])
        )
        * ampl,
        flare_lc,
    )
    return flare_lc


def get_flare_integral_numerical(
    time, tpeak, fwhm, ampl, texp=None, oversample=7
):
    """
    used in testing flare integral
    """
    feval = get_light_curvenp(time, tpeak, fwhm, ampl, texp, oversample)
    tstart, tend = time[feval > 0][0], time[feval > 0][-1]
    integral = integrate.quad(
        get_light_curvenp,
        tstart,
        tend,
        points=tpeak,
        args=(tpeak, fwhm, ampl, texp, oversample),
    )
    return integral


def _flareintegralnp(fwhm, ampl):
    t0, t1, t2 = -1, 0, 20
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    def get_int_before(x):
        integral = (
            _fr[0] * x
            + (_fr[1] * x ** 2 / 2)
            + (_fr[2] * x ** 3 / 3)
            + (_fr[3] * x ** 4 / 4)
            + (_fr[4] * x ** 5 / 5)
        )
        return integral

    def get_int_after(x):
        integral = (_fd[0] / _fd[1] * np.exp(_fd[1] * x)) + (
            _fd[2] / _fd[3] * np.exp(_fd[3] * x)
        )
        return integral

    before = get_int_before(t1) - get_int_before(t0)
    after = get_int_after(t2) - get_int_after(t1)
    return (before + after) * ampl * fwhm


def multiflareintegralnp(fwhms, ampls):
    fwhms = np.atleast_1d(fwhms)
    ampls = np.atleast_1d(ampls)
    npeaks = fwhms.shape[0]
    multiintegral = np.zeros_like(fwhms)
    for i in range(npeaks):
        multiintegral[i] = _flareintegralnp(fwhms[i], ampls[i])
    return multiintegral

