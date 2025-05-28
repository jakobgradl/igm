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

    path = "02-fwd/geology-tcal-1990-2000-nodivfluxcost-notopgcont-constslid.nc"

    with xr.open_dataset(path) as ds:
        topg = ds["topg"].isel(time=0)
        slidingco = ds["slidingco"].isel(time=0)

    state.topg = state.usurf = state.lsurf = tf.Variable(topg.astype("float32"), trainable=False)
    state.slidingco = tf.Variable(slidingco.astype("float32"), trainable=False)
    state.thk = tf.zeros_like(state.topg)


def update(params, state):
    pass

def finalize(params, state):
    pass