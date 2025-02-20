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
    # native res of BedMachine: 500.0 m
    res = 5000.0 
    # max extent of Wilkes+Aurora on BedMachine native grid (EPSG:3031)
    x_min = 683000.0
    x_max = 2569500.0
    y_min = -2134000.0
    y_max = -106500.0


    state.x = tf.constant(tf.range(x_min, x_max+1, res))
    state.y = tf.constant(tf.range(y_max, y_min-1, -res))
    state.dx = res
    
    state.X, state.Y = tf.meshgrid(state.x, state.y) 
    state.dX = tf.ones_like(state.X) * state.dx
    

# geometry
    bm3 = xr.open_dataset('~/pvol/measures_data/BedMachineAntarctica-v3.nc')
    # bm3 = xr.open_dataset('~/pvol/measures_data/BedMachine-v3-WilkesAurora-grounded.nc')
    uvel = xr.open_dataset('~/pvol/measures_data/antarctic_ice_vel_phase_map_v01.nc')

    uvelsurfobs = uvel['VX'].interp(x=state.x, y=state.y)
    vvelsurfobs = uvel['VY'].interp(x=state.x, y=state.y)

    usurf = tf.ones_like(state.X)
    topg = tf.ones_like(state.X)
    thk = tf.ones_like(state.X)

    usurf = bm3['surface'].sel(x=state.x, y=state.y)
    topg = bm3['bed'].sel(x=state.x, y=state.y)
    thk = bm3['thickness'].sel(x=state.x, y=state.y)
    icemask = bm3['mask'].sel(x=state.x, y=state.y) # BM3 icemask: ocean (0); ice-free land (1); grounded ice (2); floating ice (3); Lake Vostok (4)
    # icemask = np.where(np.isnan(icemask), -1, icemask)
    icemask = icemask - 1  # that's the IGM convention
    # icemask = np.where(np.isnan(usurf), -2, icemask)

    lsurf = np.where(icemask == 1, topg, usurf-thk)
    # thk = usurf-lsurf

    state.usurf = tf.Variable(usurf.astype("float32"), trainable=False)
    state.topg = tf.Variable(topg.astype("float32"), trainable=False)
    state.thk = tf.Variable(thk.astype("float32"), trainable=False)
    state.lsurf = tf.Variable(lsurf.astype("float32"), trainable=False)
    state.icemaskobs = state.icemask = tf.Variable(icemask.astype("float32"), trainable=False)

    # state.usurfobs = state.usurf

# parameters

    state.thkobs = tf.ones_like(state.thk) * state.thk
    state.usurfobs = tf.ones_like(state.usurf) * state.usurf

    state.uvelsurfobs = tf.Variable(uvelsurfobs.astype("float32"), trainable=False)
    state.vvelsurfobs = tf.Variable(vvelsurfobs.astype("float32"), trainable=False)



def update(params, state):
    pass

def finalize(params, state):
    pass