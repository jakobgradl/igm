#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.interpolate import interp1d, RectBivariateSpline
import xarray as xr

from igm.modules.utils import complete_data


def params(parser):
    pass


def initialize(params, state):

    # path = "02-fwd/geology-tcal-1990-2000-nodivfluxcost-notopgcont-constslid.nc"
    path = "02-fwd/output-ela2850-acc0035-abl0045.nc"

    with xr.open_dataset(path) as ds:
        topg = ds["topg"].isel(time=-1)
        slidingco = ds["slidingco"].isel(time=-1)
        usurf = ds["usurf"].isel(time=-1)
        # lsurf = ds["lsurf"].isel(time=-1)
        thk = ds["thk"].isel(time=-1)
        uvelsurf = ds["uvelsurf"].isel(time=-1)
        vvelsurf = ds["vvelsurf"].isel(time=-1)

    # increase sliding at the front

    # increase = -0.01

    # slidingco_y = xr.ones_like(slidingco) * np.arange(245).reshape(245,1)
    # slidingco_anom = xr.zeros_like(slidingco)
    # slidingco_anom[80:100,:] += increase * (100. - slidingco_y[80:100,:]) / 20.
    # slidingco_anom[:80,:] += increase

    # slidingco_new = slidingco + slidingco_anom
    # state.slidingco = tf.Variable(slidingco_new.astype("float32"), trainable=False)

    # for tres1 steady state
    state.slidingco = tf.Variable(slidingco.astype("float32"), trainable=False)

    state.topg = tf.Variable(topg.astype("float32"), trainable=False)
    state.usurf = tf.Variable(usurf.astype("float32"), trainable=False)
    # state.lsurf = tf.Variable(lsurf.astype("float32"), trainable=False)
    state.thk = tf.Variable(thk.astype("float32"), trainable=False)
    state.uvelsurf = tf.Variable(uvelsurf.astype("float32"), trainable=False)
    state.vvelsurf = tf.Variable(vvelsurf.astype("float32"), trainable=False)


def update(params, state):
    pass

def finalize(params, state):
    pass