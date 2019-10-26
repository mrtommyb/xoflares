import numpy as np


def test_imports():
    import xoflares
    from xoflares import multiflaremodel
    from xoflares import multiflare
    from xoflares import multiflaremodelnp
    from xoflares import multiflareintegral
    from xoflares import multiflareintegralnp


def test_calc():
    from xoflares import multiflare
    from xoflares import multiflaremodelnp

    x = np.arange(0, 10, 2 / 1440)
    tpeaks = [1, 2, 3, 4]
    fwhms = [0.05, 0.05, 0.05, 0.05]
    ampls = [1, 2, 3, 4]
    flare_lc = multiflare(x, tpeaks, fwhms, ampls)
    flare_lc_np = multiflaremodelnp(x, tpeaks, fwhms, ampls)
    assert np.shape(x)[0] == np.shape(flare_lc)[0]
    assert np.sum(np.abs(flare_lc - flare_lc_np)) < 1.0e-12


def test_calc_with_oversample():
    from xoflares import get_light_curvenp
    from xoflares import eval_get_light_curve

    x = np.arange(0, 10, 2 / 1440)
    tpeaks = np.array([1, 2, 3, 4])
    fwhms = np.array([0.5, 0.5, 0.5, 2])
    ampls = np.array([1, 2, 3, 4])
    flare_lc = eval_get_light_curve(
        x, tpeaks, fwhms, ampls, texp=1, oversample=11
    )
    flare_lc_np = get_light_curvenp(
        x, tpeaks, fwhms, ampls, texp=1, oversample=11
    )
    assert np.shape(x)[0] == np.shape(flare_lc)[0]
    assert np.sum(np.abs(flare_lc - flare_lc_np)) < 1.0e-12


def test_integrals():
    from xoflares import eval_multiflareintegral
    from xoflares import multiflareintegralnp

    fwhms = np.array([0.5, 0.5, 0.5, 2])
    ampls = np.array([1, 2, 3, 4])

    integrals = eval_multiflareintegral(fwhms, ampls)
    integrals_np = multiflareintegralnp(fwhms, ampls)

    assert np.shape(fwhms)[0] == np.shape(integrals)[0]
    assert np.shape(fwhms)[0] == np.shape(integrals_np)[0]
    assert np.sum(np.abs(integrals - integrals_np)) < 1.0e-12

def test_integrals_numerical():
    from xoflares.xoflares import get_flare_integral_numerical
    from xoflares import multiflareintegralnp

    x = np.arange(0, 100, 2 / 1440)
    tpeaks = np.array([10, 20, 30, 40])
    fwhms = np.array([0.01, 0.01, 0.01, 1])
    ampls = np.array([1, 2, 3, 4])
    integrals = np.zeros_like(fwhms)
    for i in range(len(tpeaks)):
        integrals[i] = get_flare_integral_numerical(x, tpeaks[i], fwhms[i], ampls[i])[0]

    integrals_np = multiflareintegralnp(fwhms, ampls)

    assert np.shape(fwhms)[0] == np.shape(integrals)[0]
    assert np.shape(fwhms)[0] == np.shape(integrals_np)[0]
    assert np.sum(np.abs(integrals - integrals_np)) < 1.0e-4