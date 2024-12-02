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

    # setup is inspired by the synthetic domain found in Ranganathan et al., 2020

# grid
    res = 50.0 # spatial resolution 100m

    state.x = tf.constant(tf.range(0.0, 200001.0, res))  # make x-axis, lenght 200 km
    # state.x = tf.constant(tf.range(0.0, 110001.0, res))  # make x-axis, lenght 110 km
    state.y = tf.constant(tf.range(-10000.0, 10001.0, res))  # make y-axis, width 20 km
    # state.dx = res * tf.ones_like(state.x)
    state.dx = res
    
    state.X, state.Y = tf.meshgrid(state.x, state.y) 
    state.dX = tf.ones_like(state.X) * state.dx
    

# geometry
    # # Loafpan bed
    # topg = -900.0 * np.ones(state.X.shape) # usurf == 0.1*thk @ GL, i.e., GL at E-border of domain
    # topg[:,0] = topg[0,:] = topg[-1,:] = 3000.0
    # topg[1:-2,1] = topg[1,1:] = topg[-2,1:] = 3000.0 - (3900.0 / 3.0)
    # topg[2:-3,2] = topg[2,2:] = topg[-3,2:] = 3000.0 - 2*(3900.0 / 3)
    # state.topg = tf.Variable(topg.astype("float32"))

    # flat bed
    topg = np.ones(state.X.shape) * -900.0
    state.topg = tf.Variable(topg.astype("float32"))

    # state.usurf = tf.ones_like(state.X) * ( state.X / 1000.0) * (-5.0) + 1000.0 
    # 1000m upstream, decreasing to 450m downstream, inspired by Denman surface slope from Young et al., 2015
    # state.usurf = 400 * tf.math.log((state.X * (-1e-3)) + 201.25)
    # 2100m upstream, decreasing logarithmically to 90m at GL, inspired by Denman surface slope
    usurf = 400 * np.log((state.X.numpy() * (-0.5e-3)) + 101.25) # logSurf
    # usurf = (state.X.numpy()/1000.0) * (-9.5) + 2000.0 # linSurf
    usurf = np.where(usurf > topg, usurf, topg)#+1.0)
    state.usurf = tf.Variable(usurf.astype("float32"))
    
    state.lsurf = state.topg * tf.ones_like(state.topg)
    state.thk = state.usurf - state.lsurf
    

# parameters

    smb = 0.2 * np.ones(state.X.shape)
    state.smb = tf.where(state.topg > -800.0, 0.0, smb)

    slidingco = np.ones(state.X.shape) * 100.0
    slidingco[0:10] = slidingco[-11:-1] = 0.0
    state.slidingco = tf.Variable(slidingco.astype("float32"))

    # c_ranga = False

    # # the sticky spots field from Ranganathan et al., 2020
    # # they use this unit for c: m a^-1 kPa^-3
    # # I think that means iflo_new_friction_param=false
    # # conversion: 1 m a^-1 kPa^-3 == 10^9 m a^-1 MPa^-3, i.e., need to times input by 10^-9

    # if c_ranga: 
    #     c = np.array([[0.,0.,40.],
    #                     [70.,7.,100.],
    #                     [-70.,7.,5.],
    #                     [70.,-7.,10.],
    #                     [-40.,-7.,-5.],
    #                     [20.,8.,100.],
    #                     [-70.,3.,20.],
    #                     [60.,-7.,10.],
    #                     [-50.,3.,70.],
    #                     [-30.,3.,20.],
    #                     [20.,-7.,10.],
    #                     [-10.,-9.,70]])
    #     # convert km to m
    #     c[:,0] *= 1000
    #     c[:,1] *= 1000

    #     slidingco = 1 + np.sum( c[:,2] * np.exp( -1/250000 * ( np.square(state.X - c[:,0]) + np.square(state.X - c[:,1])) ) )

    #     slidingco = 1
    #     for i in np.arange(c.shape[0]):
    #         cpar = c[i,2] * np.exp( 
    #             -1/250000 * ( np.square(state.X - c[i,0]) + np.square(state.Y - c[i,1]) ) 
    #             )
    #         slidingco += cpar   
        
    #     # convert kPa to MPa
    #     slidingco *= 1e-9

    # else:
    #     state.slidingco = 10.0 * tf.ones_like(state.X) #* 1e-9
    #     # state.slidingco = tf.zeros_like(state.X)


    # Flow rate field inspired by Ranganathan et al., 2020
    arrhenius = 40 * tf.ones_like(state.X)
    arrhenius = arrhenius * (1 - 0.8 * np.cos((2 * np.pi * state.Y) / (20000) )) # A between 9 in centre and 71 in margins



    ###### TODO:
    # ask in Discord about how to set boundary conditions
    # Still need to do something about input flux


    # complete_data(state)


def update(params, state):
    pass

def finalize(params, state):
    pass