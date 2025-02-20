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
    pass


def initialize(params, state):

    sys.path.append('/home/ubuntu/igm/igm/Jakob/NorskeBreer')
    from load_nve_data import load_nve_data as lnd

    glacierID = 'NFF' # Nordre Folgefonna

    usurfobs, thkobs = lnd(params,state,glacierID)

    state.usurfobs = tf.Variable(usurfobs.astype("float32"), trainable=False)
    state.thkobs = tf.Variable(thkobs.astype("float32"), trainable=False)

    

def update(params, state):
    pass

def finalize(params, state):
    pass