import numpy as np


def test_imports():
    import xoflares
    from xoflares import multiflaremodel
    from xoflares import multiflare
    from xoflares import multiflaremodelnp


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
    assert np.sum(np.abs(flare_lc - flare_lc_np)) < 1.E-12


def test_calc_with_oversample():
    from xoflares import get_light_curvenp
    from xoflares import eval_get_light_curve
    x = np.arange(0, 10, 2 / 1440)
    tpeaks = [1, 2, 3, 4]
    fwhms = [0.5, 0.5, 0.5, 2]
    ampls = [1, 2, 3, 4]
    flare_lc = eval_get_light_curve(x, tpeaks, fwhms, ampls, texp=1,
                                    oversample=11)
    flare_lc_np = get_light_curvenp(x, tpeaks, fwhms, ampls, texp=1,
                                    oversample=11)
    assert np.shape(x)[0] == np.shape(flare_lc)[0]
    assert np.sum(np.abs(flare_lc - flare_lc_np)) < 1.E-12
