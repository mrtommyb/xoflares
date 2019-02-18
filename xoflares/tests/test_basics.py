import numpy as np
import theano
import pymc3 as pm
import theano.tensor as tt

import xoflares

def test_imports():
   import xoflares
   from xoflares import multiflaremodel
   from xoflares import multiflare
   from xoflares import multiflaremodelnp

def test_calc():
    x = np.arange(0, 10, 2/1440)
    tpeaks = [1,2,3,4]
    fwhms = [0.05, 0.05, 0.05, 0.05]
    ampls = [1, 2, 3, 4]
    flare_lc = multiflare(x, tpeaks, fwhms, ampls)
    assert np.shape(x)[0] == np.shape(flare_lc)[0]

    flare_lc_np = multiflaremodelnp(x, tpeaks, fwhms, ampls)
    assert np.abs(flare_lc-flare_lc_np) < 1.E-12
