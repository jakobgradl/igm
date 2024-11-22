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
    state.topg = -900.0 * tf.zeros_like(state.X) # usurf == -lsurf @ GL
    # state.usurf = tf.ones_like(state.X) * ( state.X / 1000.0) * (-5.0) + 1000.0 
    # 1000m upstream, decreasing to 450m downstream, inspired by Denman surface slope from Young et al., 2015
    # state.usurf = 400 * tf.math.log((state.X * (-1e-3)) + 201.25)
    # 2100m upstream, decreasing logarithmically to 90m at GL, inspired by Denman surface slope
    # state.usurf = 400 * tf.math.log((state.X * (-0.5e-3)) + 101.25) # logSurf
    state.usurf = (state.X/1000.0) * (-9.5) + 2000.0 # linSurf
    state.lsurf = state.topg * tf.ones_like(state.topg)
    state.thk = state.usurf - state.lsurf
    

# parameters

    state.smb = 0.2 * tf.ones_like(state.X)

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


    # # Flow rate field from Ranganathan et al., 2020
    # # define as 2d, vertical extrusion in utils.py at initialisation
    # arrhenius = 1.6729e-7 * tf.ones_like(state.X) # kPa^-3 a^-1
    # arrhenius = arrhenius * (1 + 0.5 * np.cos((2 * np.pi * state.Y) / (20000) ))  # frate_reverse
    # # arrhenius *= (1 + 0.5 * np.cos((2 * np.pi * state.Y) / (2*20000) ))               # new_frate
    # state.arrhenius = arrhenius * 1e9 # kPa to MPa



    ###### TO DO:
    # prescribe surface accumulation (0.2 m/a)
    # ask in Discord about how to set boundary conditions
    # if I need to flip frate, then probs something is not right
    # would make sense that the shear margins have higher flow rate/lower viscosity
    # for slidingco: c is not equal c, check the exact implementation and equations to figure out appropriate values
    # make a transient run to evolve the geometry/velocity balance with prescribed mass influx
    # use simple surface geometry (linear)


# remaining issues

    # Still need to do something about lateral no-slip
    # and input flux
    # output/GL boundary is given by hydrostatic pressure. Maybe activate CF?? but CF only applies where lsurf < 0


    # complete_data(state)


def update(params, state):
    pass

def finalize(params, state):
    pass