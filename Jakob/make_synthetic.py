#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.interpolate import interp1d, RectBivariateSpline

from igm.modules.utils import complete_data


def params(parser):
    pass


def initialize(params, state):

    # this is inspired by the synthetic domain found in Ranganathan et al., 2020

    # grid
    res = 100.0 # spatial resolution O(2.5km)

    state.x = tf.constant(tf.range(0.0, 200001.0, res))  # make x-axis, lenght 200 km
    state.y = tf.constant(tf.range(-10000.0, 10001.0, res))  # make y-axis, lenght 20 km
    # state.dx = res * tf.ones_like(state.x)
    state.dx = res
    
    state.X, state.Y = tf.meshgrid(state.x, state.y) 
    state.dX = tf.ones_like(state.X) * state.dx
    

    # geometry
    state.topg = (-100.0 - (state.X/1000))  # define the bedrock topography; in m but X in km
    state.thk = 10.0 * tf.ones_like(state.X) # 10m thick initial slab
    state.lsurf = tf.maximum(state.topg,-0.9*state.thk)
    state.usurf = state.lsurf + state.thk
    

    # parameters

    c = np.array([[0.,0.,40.],
                     [70.,7.,100.],
                     [-70.,7.,5.],
                     [70.,-7.,10.],
                     [-40.,-7.,-5.],
                     [20.,8.,100.],
                     [-70.,3.,20.],
                     [60.,-7.,10.],
                     [-50.,3.,70.],
                     [-30.,3.,20.],
                     [20.,-7.,10.],
                     [-10.,-9.,70]])
    c = c * 1000

    slidingco = 1 + np.sum( c[:,2] * np.exp( -1/250000 * ( np.square(state.X - c[:,0]) + np.square(state.X - c[:,1])) ) )

    slidingco = 1
    for i in np.arange(c.shape[0]):
        cpar = c[i,2] * np.exp( 
            -1/250000 * ( np.square(state.X - c[i,0]) + np.square(state.X - c[i,1]) ) 
            )
        slidingco += cpar


    # Flow rate field from Ranganathan et al., 2020
    arrhenius = 1.6729e-16 * tf.ones_like(state.X) # MPa^-3 a^-1
    state.arrhenius = arrhenius * (1 - 0.5 * np.cos((2 * np.pi * state.Y) / (20000) ))



    state.smb = 0.5 * tf.ones_like(state.X)
    state.arrhenius = (1.0/3.1536) * tf.ones_like(state.X) # 10^-25 [Pa^-3 s^-1] in [MPa^-3 a^-1]    
    state.slidingco = 10.0/(31536000.0**(1/3)) * tf.ones_like(state.X) # 10^7 [Pa m^-1/3 s^1/3] in [MPa m^-1/3 a^1/3], ca 0.031652


    # complete_data(state)


def update(params, state):
    pass

def finalize(params, state):
    pass