#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from scipy.interpolate import interp1d, RectBivariateSpline
import xarray as xr
import sys

from igm.modules.utils import complete_data


def params(parser):
    parser.add_argument(
        "--RGIv6_basins",
        type=list,
        default=[
            "RGI60-08.02928",
            "RGI60-08.02973",
            "RGI60-08.02930",
            "RGI60-08.02972",
            "RGI60-08.02940"
            ],
        help="Individuall RGI-v7 glaciers of Nordre Folgefonna",
    )


def initialize(params, state):

    sys.path.append('/home/ubuntu/igm/igm/Jakob/NorskeBreer')
    from load_nve_data import load_nve_data as lnd
    from load_nve_data import load_aster_data as lad

    glacierID = 'NFF' # Nordre Folgefonna

    usurfobs, thkobs = lnd(params,state,glacierID)
    ave_cumul_dS_tcal = lad(params)

    tsteps = [len(params.tcal_times)]

    state.usurfobs_tcal = tf.Variable(
        tf.ones(tsteps + tf.shape(state.X).numpy().tolist()) * np.nan,
        trainable=False
        )
    state.usurfobs_tcal[11].assign(usurfobs)

    state.thkobs_tcal = tf.Variable(
        tf.ones(tsteps + tf.shape(state.X).numpy().tolist()) * np.nan,
        trainable=False
        )
    state.thkobs_tcal[11].assign(thkobs)


    state.thkobs = tf.Variable(thkobs.astype("float32"), trainable=False)
    state.ave_cumul_dSobs_tcal = tf.Variable( ave_cumul_dS_tcal[:len(params.tcal_times)].astype("float32"), trainable=False )
    # state.ave_cumul_dSobs_tcal = ave_cumul_dSobs_tcal[:len(params.tcal_times)]

    

def update(params, state):
    pass

def finalize(params, state):
    pass