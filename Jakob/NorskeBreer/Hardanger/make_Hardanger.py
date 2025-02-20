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
        "--RGI-basins",
        type=list,
        default=[
            "RGI2000-v7.0-G-08-02896",
            "RGI2000-v7.0-G-08-02895",
            "RGI2000-v7.0-G-08-02897",
            "RGI2000-v7.0-G-08-02893",
            "RGI2000-v7.0-G-08-02892",
            "RGI2000-v7.0-G-08-02898",
            "RGI2000-v7.0-G-08-02899",
            "RGI2000-v7.0-G-08-02894",
            "RGI2000-v7.0-G-08-03091",
            "RGI2000-v7.0-G-08-02888",
            "RGI2000-v7.0-G-08-03092",
            "RGI2000-v7.0-G-08-02889",
            "RGI2000-v7.0-G-08-03093",
            "RGI2000-v7.0-G-08-02887"
            ],
        help="Individuall RGI-v7 glaciers of Harangerjokulen",
    )


def initialize(params, state):

    sys.path.append('/home/ubuntu/igm/igm/Jakob/NorskeBreer')
    from load_nve_data import load_nve_data as lnd

    glacierID = 'HAJ' # Hardangerjokulen

    usurfobs, thkobs = lnd(params,state,glacierID)

    state.usurfobs = tf.Variable(usurfobs.astype("float32"), trainable=False)
    state.thkobs = tf.Variable(thkobs.astype("float32"), trainable=False)

    

def update(params, state):
    pass

def finalize(params, state):
    pass