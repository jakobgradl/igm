#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from scipy.interpolate import interp1d, RectBivariateSpline
import xarray as xr

from igm.modules.utils import complete_data


def params(parser):
    pass


def initialize(params, state):

# grid
    res = 10000.0 # native res of BedMachine: 500.0 m
    # BedMachine-v3 on EPSG:3031
    # max extent of Wilkes+Aurora on BedMachine native grid
    x_min = 683000
    x_max = 2569500
    y_min = -2134000
    y_max = -106500

    state.x = tf.constant(tf.range(x_min, x_max+1, res))
    state.y = tf.constant(tf.range(y_max, y_min-1, -res))
    state.dx = res
    
    state.X, state.Y = tf.meshgrid(state.x, state.y) 
    state.dX = tf.ones_like(state.X) * state.dx
    

# geometry
    bm3WA = xr.open_dataset('~/pvol/measures_data/BedMachine-v3-WilkesAurora-grounded.nc')
    uvel = xr.open_dataset('~/pvol/measures_data/antarctic_ice_vel_phase_map_v01.nc')

    uvelsurfobs = uvel['VX'].interp(x=state.x, y=state.y)
    vvelsurfobs = uvel['VY'].interp(x=state.x, y=state.y)

    usurf = tf.ones_like(state.X)
    topg = tf.ones_like(state.X)
    thk = tf.ones_like(state.X)

    usurf = bm3WA['surface'].sel(x=state.x, y=state.y)
    topg = bm3WA['bed'].sel(x=state.x, y=state.y)
    thk = bm3WA['thickness'].sel(x=state.x, y=state.y)
    icemask = bm3WA['mask'].sel(x=state.x, y=state.y) # BM3 icemask: ocean (0); ice-free land (1); grounded ice (2); floating ice (3); Lake Vostok (4)
    icemask = icemask - 1
    icemask = np.where(np.isnan(usurf), -2, icemask)

    lsurf = np.where(icemask == 1, topg, usurf-thk)
    thk = usurf-lsurf

    state.usurf = tf.Variable(usurf.astype("float32"), trainable=False)
    state.topg = tf.Variable(topg.astype("float32"), trainable=False)
    state.thk = tf.Variable(thk.astype("float32"), trainable=False)
    state.lsurf = tf.Variable(lsurf.astype("float32"), trainable=False)
    state.icemaskobs = tf.Variable(icemask.astype("float32"), trainable=False)

    state.usurfobs = state.usurf    

    state.uvelsurfobs = tf.Variable(uvelsurfobs.astype("float32"), trainable=False)
    state.vvelsurfobs = tf.Variable(vvelsurfobs.astype("float32"), trainable=False)

# parameters

    state.thkobs = tf.ones_like(state.thk)

    

def update(params, state):
    pass

def finalize(params, state):
    pass