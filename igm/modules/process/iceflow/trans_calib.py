#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, copy
import matplotlib.pyplot as plt 
import datetime, time
import math
import tensorflow as tf
from scipy import stats 
import xarray as xr

from igm.modules.utils import * 
from .energy_iceflow import *
from .utils import *
from .emulate import *
from .optimize_outputs import *
from .optimize_params_cook import *
 
def trans_calib(params, state):

    load_tcal_obs_data(params, state)

    ###### PERFORM CHECKS PRIOR OPTIMIZATIONS

    # from scipy.ndimage import gaussian_filter
    # state.usurfobs = tf.Variable(gaussian_filter(state.usurfobs.numpy(), 3, mode="reflect"))
    # state.usurf    = tf.Variable(gaussian_filter(state.usurf.numpy(), 3, mode="reflect"))

    params.tcal_control = params.tcal_control_trans + params.tcal_control_const

    # make sure this condition is satisfied
    assert ("usurf" in params.tcal_cost) == ("usurf" in params.tcal_control_trans)

    if "topg" in params.tcal_control:
        assert "topg" in params.tcal_control_const
    else:
        assert hasattr(state, "topgobs_tcal")

    # make sure that there are lease some profiles in thkobs
    if "thk" in params.tcal_cost:
        assert hasattr(state, "thkobs_tcal")
        if tf.reduce_all(tf.math.is_nan(state.thkobs_tcal)):
            print("\n    WARNING: No thickness observation data available; removing thk from tcal_cost    \n")
            params.tcal_cost.remove("thk")



    ###### add a time dimension to all transient variables 

    state.dt_tcal = tf.cast( tf.Variable(params.tcal_times[1:]) - tf.Variable(params.tcal_times[:-1]), dtype=tf.float32 )
    state.dt_tcal = tf.reshape(state.dt_tcal, (len(params.tcal_times)-1,1,1)) # dims(t,y,x)=(nt-1,1,1)
    
    # emulator input and output variables:
    tsteps = [len(params.tcal_times)]
    params.iflo_fieldin_tcal = [s + "_tcal" for s in params.iflo_fieldin]

    for var1, var2 in zip(params.iflo_fieldin, params.iflo_fieldin_tcal):
        vars(state)[var2] = tf.Variable(
            tf.ones(tsteps + tf.shape(vars(state)[var1]).numpy().tolist()) * vars(state)[var1],
            trainable=False
        )
        ## (tsteps + tf.shape(vars(state)[var1]).numpy().tolist()) = [time] + [z,y,x] = [t,z,y,x]
        # vars(state)[var2] *= vars(state)[var1]

    for var1, var2 in zip(["U","V"], ["U_tcal","V_tcal"]):
        vars(state)[var2] = tf.Variable(
            tf.zeros(tsteps + tf.shape(vars(state)[var1]).numpy().tolist()) * vars(state)[var1],
            trainable=False
        )

    # transient control parameters:
    opti_control_tcal = [s + "_tcal" for s in params.tcal_control_trans if not hasattr(state, s+"_tcal")] # s not in params.iflo_fieldin]
    for var1, var2 in zip(params.tcal_control_trans, opti_control_tcal):
        vars(state)[var2] = tf.Variable(
            tf.ones(tsteps + tf.shape(vars(state)[var1]).numpy().tolist()) * vars(state)[var1],
            trainable=False
        )

    # cost parameters:
    opti_cost_tcal = [s + "_tcal" for s in params.tcal_cost if s not in params.tcal_control] # change that to if not hasattr(state, s+"_tcal")
    for var1, var2 in zip(params.tcal_cost, opti_cost_tcal):
        vars(state)[var2] = tf.Variable(
            tf.ones(tsteps + tf.shape(state.thk_tcal).numpy().tolist()),
            trainable=False
        )
        # vars(state)[var2] *= vars(state)[var1]

    # constant control parameters (these will get the time dimension after initializing the GradientTape):
    opti_control_tcal_const = [s + "_tcal" for s in params.tcal_control_const]
    for var1, var2 in zip(params.tcal_control_const, opti_control_tcal_const):
        vars(state)[var2] = tf.Variable(
            tf.ones([1] + tf.shape(vars(state)[var1]).numpy().tolist()) * vars(state)[var1],
            trainable=False
        )

    # # other variables that are required
    # vars_remaining = ["uvelbase", "vvelbase", "ubar", "vbar", "uvelsurf", "vvlesurf"]
    # vars_remaining = [s + "_tcal" for s in vars_remaining]
    # for var in vars_remaining:
    #     vars(state)[var] = tf.ones(tsteps + tf.shape(state.thk_tcal).numpy().tolist())

    # dirty fixes for my project
    # need this for both icemask and divfluxcfz cost
    state.icemaskobs_tcal = tf.where(state.thkobs_tcal == 0., 0., 1.)
    # state.icemaskobs_tcal = tf.where(state.topg_tcal < (state.usurf_tcal - state.thk_tcal - 1.), 2., state.icemaskobs_tcal)
    state.icemask_tcal = state.icemaskobs_tcal


    ###### PREPARE DATA PRIOR OPTIMIZATIONS
 
    if "topg" not in params.tcal_control:
        state.topg_tcal = state.topgobs_tcal
    else:
        if hasattr(state, "thkinit"):
            state.thk_tcal = tf.ones_like(state.thk_tcal) * state.thkinit 
        else:
            state.thk_tcal = state.thk_tcal * 0.0

        if params.tcal_init_zero_thk:
            # state.thk_tcal = state.thk_tcal * 0.0
            state.topg_tcal = tf.expand_dims(
                tf.math.reduce_min(state.usurfobs_tcal, axis=0),
                axis=0
            )

    if params.tcal_init_const_sl:
        state.slidingco_tcal = tf.ones_like(state.slidingco_tcal) * 0.045

    # just checking:
    # state.topg_tcal = tf.where(tf.math.reduce_min(state.icemask_tcal, axis=0) > 0.5, state.topg_tcal - 100.0, state.topg_tcal)
    
        
    # this is a density matrix that will be used to weight the cost function
    # TODO: need to adjust density matrix to 4d
    if hasattr(state, "thkobs_tcal"):
        if params.tcal_uniformize_thkobs:
            state.dens_thkobs_tcal = create_density_matrix(state.thkobs_tcal, kernel_size=5)
            state.dens_thkobs_tcal = tf.where(tf.math.is_nan(state.thkobs_tcal),0.0,state.dens_thkobs_tcal)
            state.dens_thkobs_tcal = tf.where(state.dens_thkobs_tcal>0, 1.0/state.dens_thkobs_tcal_tcal, 0.0)
            state.dens_thkobs_tcal = tf.math.division( # divide every time slice in state.dens_thkobs with its spatial average
                state.dens_thkobs_tcal, 
                tf.reshape(
                    tf.reduce_mean(tf.reduce_mean(state.dens_thkobs_tcal[state.dens_thkobs_tcal>0], axis=1), axis=1), # spatial average of every time slice
                    [state.dens_thkobs_tcal.shape[0],1,1] # reshape to be broadcastable with state.dens_thkobs
                    )
                )
            # tf.reduce_mean(tf.reduce_mean(x)) => mean of every time slice
        else:
            state.dens_thkobs_tcal = tf.ones_like(state.thkobs_tcal)
        
    # force zero slidingco in the floating areas
    if "slidingco" not in params.tcal_control_const:
        state.slidingco_tcal = tf.Variable(tf.where( state.icemaskobs_tcal == 2, 0.0, state.slidingco_tcal))
    
    # # this is generally not active in optimize
    # # this will infer values for slidingco and convexity weight based on the ice velocity and an empirical relationship from test glaciers with thickness profiles
    # if params.tcal_infer_params:
    #     #Because OGGM will index icemask from 0
    #     dummy = infer_params_cook(state, params)
    #     if tf.reduce_max(state.icemask).numpy() < 1:
    #         return
    
    if (int(tf.__version__.split(".")[1]) <= 10) | (int(tf.__version__.split(".")[1]) >= 16) :
        optimizer = tf.keras.optimizers.Adam(learning_rate=params.tcal_step_size)
        opti_retrain = tf.keras.optimizers.Adam(
            learning_rate=params.iflo_retrain_emulator_lr
        )
    else:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=params.tcal_step_size)
        opti_retrain = tf.keras.optimizers.legacy.Adam(
            learning_rate=params.iflo_retrain_emulator_lr
        )

    state.tcomp_optimize = []

    # this thing is outdated with using iflo_new_friction_param default as we use scaling of one.
    sc = {}
    sc["topg_tcal"] = params.tcal_scaling_topg
    # sc["thk_tcal"] = params.tcal_scaling_thk
    sc["usurf_tcal"] = params.tcal_scaling_usurf
    sc["slidingco_tcal"] = params.tcal_scaling_slidingco
    sc["arrhenius_tcal"] = params.tcal_scaling_arrhenius
    
    Ny, Nx = state.thk.shape

    for f in [s + "_tcal" for s in params.tcal_control]:
        vars()[f] = tf.Variable(vars(state)[f] / sc[f])

    # main loop
    for i in range(params.tcal_nbitmax):
        with tf.GradientTape() as t, tf.GradientTape() as tc, tf.GradientTape() as s:
            state.tcomp_optimize.append(time.time())
            
            if params.tcal_step_size_decay < 1:
                optimizer.lr = params.tcal_step_size * (params.tcal_step_size_decay ** (i / 100))

            # is necessary to remember all operation to derive the gradients w.r.t. control variables
            for f in [s + "_tcal" for s in params.tcal_control_trans]: # if s not in params.tcal_control_const]:
                t.watch(vars()[f])
            for f in [s + "_tcal" for s in params.tcal_control_const]:
                tc.watch(vars()[f])

            # # non-transient control parameters
            # # assign the time-average mean to every timestep
            # for var in [s + "_tcal" for s in params.tcal_control_const]:
            #     v = tf.reduce_mean(vars(state)[var], axis=0)
            #     vars(state)[var] = tf.ones_like(vars(state)[var]) * v
                
            for f in [s + "_tcal" for s in params.tcal_control_trans]:
                vars(state)[f] = vars()[f] * sc[f]

            # expand constant control parameters to match the time dimension
            for f in [s + "_tcal" for s in params.tcal_control_const]:
                vars(state)[f] = tf.ones_like(state.usurf_tcal) * vars()[f] *sc[f] 

            # if "thk" not in params.tcal_control:
            # state.thk_tcal = state.usurf_tcal - ( tf.ones_like(state.usurf_tcal) * state.topg_tcal )
            state.thk_tcal = state.usurf_tcal - state.topg_tcal

            # state.topg_tcal = state.usurf_tcal - state.thk_tcal

            fields = [vars(state)[f] for f in params.iflo_fieldin_tcal]
            X = [None] * len(params.tcal_times)
            Y = [None] * len(params.tcal_times)

            U_tcal = {}
            V_tcal = {}

            for iter in range(len(params.tcal_times)):
                fieldin = [var[iter] for var in fields]

                X[iter] = fieldin_to_X(params, fieldin)

                # evalutae th ice flow emulator                
                if params.iflo_multiple_window_size==0:
                    Y[iter] = state.iceflow_model(X[iter])
                else:
                    Y[iter] = state.iceflow_model(tf.pad(X, state.PAD, "CONSTANT"))[:, :Ny, :Nx, :]

                U, V = Y_to_UV(params, Y[iter])

                U_tcal[iter] = U[0]
                V_tcal[iter] = V[0]

            U_tcal = tf.stack(list(U_tcal.values()), axis=0)
            V_tcal = tf.stack(list(V_tcal.values()), axis=0)
            
            # this is strange, but it having state.U instead of U, slidingco is not more optimized ....
            state.uvelbase_tcal = U_tcal[:, 0, :, :]
            state.vvelbase_tcal = V_tcal[:, 0, :, :]
            state.ubar_tcal = tf.reduce_sum(U_tcal * state.vert_weight, axis=1)
            state.vbar_tcal = tf.reduce_sum(V_tcal * state.vert_weight, axis=1)
            state.uvelsurf_tcal = U_tcal[:, -1, :, :]
            state.vvelsurf_tcal = V_tcal[:, -1, :, :]

            # U_tcal = tf.Variable(tf.ones_like(state.U_tcal), trainable=False)
            # V_tcal = tf.Variable(tf.ones_like(state.V_tcal), trainable=False)

            # for iter in range(len(params.tcal_times)):
            #     fieldin = [var[iter] for var in fields]

            #     X[iter] = fieldin_to_X(params, fieldin)

            #     # evaluate th ice flow emulator                
            #     if params.iflo_multiple_window_size==0:
            #         Y[iter] = state.iceflow_model(X[iter])
            #     else:
            #         Y[iter] = state.iceflow_model(tf.pad(X, state.PAD, "CONSTANT"))[:, :Ny, :Nx, :]

            #     U, V = Y_to_UV(params, Y[iter])

            #     U_tcal[iter].assign(U[0])
            #     V_tcal[iter].assign(V[0])
            
            # # this is strange, but it having state.U instead of U, slidingco is not more optimized ....
            # state.uvelbase_tcal = U_tcal[:, 0, :, :]
            # state.vvelbase_tcal = V_tcal[:, 0, :, :]
            # state.ubar_tcal = tf.reduce_sum(U_tcal * state.vert_weight, axis=1)
            # state.vbar_tcal = tf.reduce_sum(V_tcal * state.vert_weight, axis=1)
            # state.uvelsurf_tcal = U_tcal[:, -1, :, :]
            # state.vvelsurf_tcal = V_tcal[:, -1, :, :]

            state.divflux_tcal = compute_divflux_tcal(state.ubar_tcal, state.vbar_tcal, state.thk_tcal, state.dx, state.dx, method=params.tcal_divflux_method)
 
            if not params.tcal_smooth_anisotropy_factor == 1:
                _compute_flow_direction_for_anisotropic_smoothing(state)

            cost = {} 
                 
            # misfit between surface velocity
            if "velsurf" in params.tcal_cost:
                cost["velsurf"] = misfit_velsurf(params,state)

            # misfit between ice thickness profiles
            if "thk" in params.tcal_cost:
                cost["thk"] = misfit_thk(params, state)

            # misfit between divergence of flux
            if ("divfluxfcz" in params.tcal_cost):
                cost["divflux"] = cost_divfluxfcz(params, state, i)
            # elif ("divfluxobs" in params.tcal_cost):
            #     cost["divflux"] = cost_divfluxobs(params, state, i)
 
            # misfit between top ice surfaces
            if "usurf" in params.tcal_cost:
                cost["usurf"] = misfit_usurf(params, state) 

            # misfit between topg and (usurf-thk)
            # if "topg" in params.tcal_control:
            #     cost["topg"] = tf.math.reduce_mean( (state.topg_tcal - (state.usurf_tcal - state.thk_tcal)) ** 2)

            # dynamical connection between the timesteps
            # if "dSdt" in params.tcal_cost:
            if len(params.tcal_times) > 1:
                cost["dSdt"] = misfit_dSdt(params, state)

            # force zero thikness outisde the mask
            if "icemask" in params.tcal_cost:
                cost["icemask"] = 10**10 * tf.math.reduce_mean( tf.where(state.icemaskobs_tcal > 0.5, 0.0, state.thk_tcal**2) )

            # Here one enforces non-negative ice thickness, and possibly zero-thickness in user-defined ice-free areas.
            if "topg" in params.tcal_control:
                # cost["thk_positive"] = 10**10 * tf.math.reduce_mean( tf.where(state.thk_tcal >= 0, 0.0, state.thk_tcal**2) )
                cost["thk_positive"] = 10**10 * tf.math.reduce_mean( 
                    tf.where( state.topg_tcal <= state.usurf_tcal, 0., (state.usurf_tcal-state.topg_tcal)**2 ) 
                    )
    
            # if params.tcal_infer_params:
            #     cost["volume"] = cost_vol(params, state)
    
            # Here one adds a regularization terms for the bed toporgraphy to the cost function
            if "topg" in params.tcal_control:
                cost["thk_regu"] = regu_thk(params, state)

            # Here one adds a regularization terms for slidingco to the cost function
            if "slidingco" in params.tcal_control:
                cost["slid_regu"] = regu_slidingco(params, state)

            # Here one adds a regularization terms for arrhenius to the cost function
            if "arrhenius" in params.tcal_control:
                cost["arrh_regu"] = regu_arrhenius(params, state) 
  
            cost_total = tf.reduce_sum(tf.convert_to_tensor(list(cost.values())))

            # Here one allow retraining of the ice flow emaultor
            cost["glen"] = tf.reduce_mean(tf.zeros(3))
            if params.tcal_retrain_iceflow_model: # and (i > (params.tcal_nbitmax / 3)):
                for iter in range(len(params.tcal_times)):
                    
                    C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(params, X[iter], Y[iter])

                    # cost["glen"] += 10**3 * (tf.reduce_mean(C_shear) + tf.reduce_mean(C_slid) + tf.reduce_mean(C_grav)  + tf.reduce_mean(C_float))**2
                    cost["glen"] += tf.reduce_mean(C_shear) + tf.reduce_mean(C_slid) + tf.reduce_mean(C_grav)  + tf.reduce_mean(C_float)

                cost["glen"] /= len(params.tcal_times)  
                grads = s.gradient(cost["glen"], state.iceflow_model.trainable_variables)

                opti_retrain.apply_gradients(
                    zip(grads, state.iceflow_model.trainable_variables)
                )

            print_costs(params, state, cost, i)

            #################

            var_trans_to_opti = [ ]
            var_const_to_opti = []
            for f in [s + "_tcal" for s in params.tcal_control_trans]: # if s not in params.tcal_control_const]:
                var_trans_to_opti.append(vars()[f])
            for f in [s + "_tcal" for s in params.tcal_control_const]:
                var_const_to_opti.append(vars()[f])

            # Compute gradient of COST w.r.t. X
            # cost_total = tf.reduce_sum(tf.convert_to_tensor(list(cost.values())))
            grads = tf.Variable(t.gradient(cost_total, var_trans_to_opti))
            grads_const = tf.Variable(tc.gradient(cost_total, var_const_to_opti))

            # this serve to restict the optimization of controls to the mask
            # TODO: need to fix grads_const in the sole_mask case
            if params.sole_mask: # !!! THIS DOESN'T WORK ATM, use sole_mask = False
                for ii in range(grads.shape[0]):
                    if not "slidingco" == params.tcal_control_trans[ii]:
                        grads[ii].assign(tf.where((state.icemaskobs_tcal > 0.5), grads[ii], 0))
                        grads_const[ii].assign(tf.where(tf.math.reduce_min(state.icemaskobs_tcal, axis=0) > 0.5, grads_const[ii], 0.))
                    else:
                        grads[ii].assign(tf.where((state.icemaskobs_tcal == 1.), grads[ii], 0))
                        grads_const[ii].assign(tf.where(tf.math.reduce_min(state.icemaskobs_tcal, axis=0) == 1., grads_const[ii], 0.))
            else:
                for ii in range(grads.shape[0]):
                    if not "slidingco" == params.tcal_control_trans[ii]:
                        grads[ii].assign(tf.where((state.icemaskobs_tcal > 0.5), grads[ii], 0.))
                for ii in range(grads_const.shape[0]):
                    if not "slidingco" == params.tcal_control_const[ii]:
                        grads_const[ii].assign(tf.where(
                            tf.expand_dims(tf.math.reduce_min(state.icemaskobs_tcal, axis=0),axis=0) > 0.5,
                            # tf.expand_dims(tf.math.reduce_mean(state.icemaskobs_tcal, axis=0),axis=0) > 0., 
                            grads_const[ii], 
                            0.))
                        
            if i % params.tcal_output_freq == 0:
                for f in range(len(var_trans_to_opti)):
                    print([s + "_tcal" for s in params.tcal_control_trans][f], tf.math.reduce_mean(grads[f]).numpy())
                for f in range(len(var_const_to_opti)):
                    print([s + "_tcal" for s in params.tcal_control_const][f], tf.math.reduce_mean(grads_const[f]).numpy())

            # One step of descent -> this will update input variable X
            optimizer.apply_gradients(
                zip([grads[i] for i in range(grads.shape[0])], var_trans_to_opti)
            )
            optimizer.apply_gradients(
                zip([grads_const[i] for i in range(grads_const.shape[0])], var_const_to_opti)
            )

            ###################

            # get back optimized variables in the pool of state.variables
            if "topg" in params.tcal_control:
                state.usurf_tcal = tf.where(state.icemaskobs_tcal > 0.5, state.usurf_tcal, state.topg_tcal)
            # if "topg" in params.tcal_control_const:
            #     state.topg_tcal = tf.expand_dims(tf.where(
            #         tf.math.reduce_min(state.icemaskobs_tcal, axis=0) == 0.0, 
            #         tf.math.reduce_min(state.usurfobs_tcal, axis=0),
            #         state.topg_tcal[0]
            #         ), axis=0)
                
                # state.thk = tf.where(state.thk < 0.01, 0, state.thk)
                # thk_mask = tf.where(state.icemaskobs_tcal > 0.5, True, False)
                # thk_update = tf.where(thk_mask, state.thk_tcal, 0.)
                # state.thk_tcal.assign(thk_update)

            # state.divflux_tcal = compute_divflux_tcal(
            #     state.ubar_tcal, state.vbar_tcal, state.thk_tcal, state.dx, state.dx, method=params.tcal_divflux_method
            # )

            #state.divflux = tf.where(ACT, state.divflux, 0.0)

            _compute_rms_std_optimization(state, i)

            state.tcomp_optimize[-1] -= time.time()
            state.tcomp_optimize[-1] *= -1

            # if i % params.tcal_output_freq == 0:
            #     if params.tcal_plot2d:
            #         update_plot_inversion(params, state, i)
                # if params.tcal_save_iterat_in_ncdf:
                #     update_ncdf_optimize(params, state, i)

            # stopping criterion: stop if the cost no longer decrease
            # if i>params.tcal_nbitmin:
            #     cost = [c[0] for c in costs]
            #     if np.mean(cost[-10:])>np.mean(cost[-20:-10]):
            #         break;

	# for final iteration
    i = params.tcal_nbitmax

    print_costs(params, state, cost, i)

    # if i % params.tcal_output_freq == 0:
    #     if params.tcal_plot2d:
    #         update_plot_inversion(params, state, i)
    #     if params.tcal_save_iterat_in_ncdf:
    #         update_ncdf_optimize(params, state, i)

#    for f in params.tcal_control:
#        vars(state)[f] = vars()[f] * sc[f]

    # now that the ice thickness is optimized, we can fix the bed once for all! (ONLY FOR GROUNDED ICE)
    # state.topg = state.usurf - state.thk

    if not params.tcal_save_result_in_ncdf=="":
        output_ncdf_tcal_final(params, state)

    plot_cost_functions() 

    plt.close("all")

    save_rms_std(params, state)

    # Flag so we can check if initialize was already called
    state.optimize_initializer_called = True
 
####################################

def misfit_velsurf(params,state):

    velsurf = tf.stack([state.uvelsurf_tcal,  state.vvelsurf_tcal], axis=-1)
    velsurfobs = tf.stack([state.uvelsurfobs_tcal, state.vvelsurfobs_tcal], axis=-1)

    REL = tf.expand_dims( (tf.norm(velsurfobs,axis=-1) >= params.tcal_velsurfobs_thr ) , axis=-1)

    ACT = ~tf.math.is_nan(velsurfobs) 

    cost = 0.5 * tf.reduce_mean(
        ( (velsurfobs[ACT & REL] - velsurf[ACT & REL]) / params.tcal_velsurfobs_std  )** 2
    )

    if params.tcal_include_low_speed_term:

        # This terms penalize the cost function when the velocity is low
        # Reference : Inversion of basal friction in Antarctica using exact and incompleteadjoints of a higher-order model
        # M. Morlighem, H. Seroussi, E. Larour, and E. Rignot, JGR, 2013
        cost += 0.5 * 100 * tf.reduce_mean(
            tf.math.log( (tf.norm(velsurf[ACT],axis=-1)+1) / (tf.norm(velsurfobs[ACT],axis=-1)+1) )** 2
        )

    return cost

def misfit_thk(params,state):

    ACT = ~tf.math.is_nan(state.thkobs_tcal)

    return 0.5 * tf.reduce_mean( state.dens_thkobs_tcal[ACT] * 
        ((state.thkobs_tcal[ACT] - state.thk_tcal[ACT]) / params.tcal_thkobs_std) ** 2
    )



def cost_divfluxfcz(params,state,i):

    # divflux_tcal = compute_divflux_tcal(
    #     state.ubar_tcal, state.vbar_tcal, state.thk_tcal, state.dx, state.dx, method=params.tcal_divflux_method
    # )
 
    ACT = state.icemaskobs_tcal > 0.5
    if i % 10 == 0:
        # his does not need to be comptued any iteration as this is expensive
        state.res = stats.linregress(
            state.usurf_tcal[ACT], state.divflux_tcal[ACT]
        )  # this is a linear regression (usually that's enough)
    # or you may go for polynomial fit (more gl, but may leads to errors)
    #  weights = np.polyfit(state.usurf[ACT],divflux[ACT], 2)
    divfluxtar = tf.where(
        ACT, state.res.intercept + state.res.slope * state.usurf_tcal, 0.0
    )
#   divfluxtar = tf.where(ACT, np.poly1d(weights)(state.usurf) , 0.0 )
    
    # ACT = state.icemaskobs_tcal > 0.5
    COST_D = 0.5 * tf.reduce_mean(
        ((divfluxtar[ACT] - state.divflux_tcal[ACT]) / params.tcal_divfluxobs_std) ** 2
    )

    if params.tcal_force_zero_sum_divflux:
            # ACT = state.icemaskobs_tcal > 0.5
            COST_D += 0.5 * 1000 * tf.reduce_mean(state.divflux_tcal[ACT] / params.tcal_divfluxobs_std) ** 2

    return COST_D
 
# def cost_divfluxobs(params,state,i):

#     divflux = compute_divflux_tcal(
#         state.ubar_tcal, state.vbar_tcal, state.thk_tcal, state.dx, state.dx, method=params.tcal_divflux_method
#     )
 
#     divfluxtar = state.divfluxobs
#     ACT = ~tf.math.is_nan(divfluxtar)
#     COST_D = 0.5 * tf.reduce_mean(
#         ((divfluxtar[ACT] - divflux[ACT]) / params.tcal_divfluxobs_std) ** 2
#     )
 
#     dddx = (divflux[:, 1:] - divflux[:, :-1])/state.dx
#     dddy = (divflux[1:, :] - divflux[:-1, :])/state.dx
#     COST_D += (params.tcal_regu_param_div) * 0.5 * ( tf.reduce_mean(dddx**2) + tf.reduce_mean(dddy**2) )

#     if params.tcal_force_zero_sum_divflux:
#         ACT = state.icemaskobs > 0.5
#         COST_D += 0.5 * 1000 * tf.reduce_mean(divflux[ACT] / params.tcal_divfluxobs_std) ** 2

#     return COST_D

def misfit_usurf(params,state):

    ACT = state.icemaskobs_tcal > 0.5

    return 0.5 * tf.reduce_mean(
        (
            (state.usurf_tcal[ACT] - state.usurfobs_tcal[ACT])
            / params.tcal_usurfobs_std
        )
        ** 2
    )

# @tf.function()
def misfit_dSdt(params,state):

    state.divflux_tcal_slopelim = compute_divflux_slope_limiter_tcal(
            state.ubar_tcal, state.vbar_tcal, state.thk_tcal, state.dx, state.dx, state.dt_tcal, slope_type=params.tcal_thk_slope_type
        ) # (nt-1,ny,nx)

    masksmb = ~tf.math.is_nan(state.smbobs_tcal[:-1])
    maskice = state.icemaskobs_tcal[:-1] > 0.5
    masksmbice = np.logical_and(masksmb,maskice)
    
    maskthk1 = state.usurf_tcal[1:] > 0.
    maskthk2 = state.usurf_tcal[:-1] > 0.
    maskthk = np.logical_and(maskthk1,maskthk2)

    ACT = np.logical_and(maskthk,masksmbice)

    return 0.5 * tf.reduce_mean(
        (
            (state.usurf_tcal[1:] - state.usurf_tcal[:-1])[ACT] - state.dt_tcal * (state.smbobs_tcal[:-1][ACT] - state.divflux_tcal_slopelim[ACT])
        )
    ) ** 2

    # make sure the produced velocity field is consistent with the enforced surface change
    # dSdt is constrained via misfit_usurf
    # make sure that divflux can also produce dSdt, given a prescribed/observed smb
    # this is a constraint on the velocity field via self-consistency

    # if only dSdt data is available and no usurf(t)
    # may be better to convert the dSdtobs into a usurfobs to still constrain the usurf data misfit
    # keep dSdt misfit as outlined above to ensure self-consistency
    

# def cost_vol(params,state):

#     ACT = state.icemaskobs_tcal > 0.5
    
#     num_basins = int(tf.reduce_max(state.icemaskobs).numpy())
#     ModVols = tf.experimental.numpy.copy(state.icemaskobs)
    
#     for j in range(1,num_basins+1):
#         ModVols = tf.where(ModVols==j,(tf.reduce_sum(tf.where(state.icemask==j,state.thk,0.0))*state.dx**2)/1e9,ModVols)

#     cost = 0.5 * tf.reduce_mean(
#            ( (state.volumes[ACT] - ModVols[ACT]) / state.volume_weights[ACT]  )** 2
#     )
#     return cost

def regu_thk(params,state):

    areaicemask_tcal = tf.reduce_sum(tf.reduce_sum(tf.where(state.icemask_tcal>0.5,1.0,0.0),axis=1),axis=1)*state.dx**2

    # here we had factor 8*np.pi*0.04, which is equal to 1
    # if params.tcal_infer_params:
    #     gamma = tf.zeros_like(state.thk)
    #     gamma = state.convexity_weights * areaicemask_tcal**(params.tcal_convexity_power-2.0)
    # else:
    gamma = params.tcal_convexity_weight * areaicemask_tcal**(params.tcal_convexity_power-2.0)

    if params.tcal_to_regularize == 'topg':
        # field = state.usurf - state.thk
        field = state.topg_tcal
    elif params.tcal_to_regularize == 'thk':
        field = state.thk_tcal

    if params.tcal_smooth_anisotropy_factor == 1:
        dbdx = (field[:, :, 1:] - field[:, :, :-1])/state.dx
        dbdy = (field[:, 1:, :] - field[:, :-1, :])/state.dx

        if params.sole_mask:
            dbdx = tf.where( (state.icemaskobs_tcal[:, :, 1:] > 0.5) & (state.icemaskobs_tcal[:, :, :-1] > 0.5) , dbdx, 0.0)
            dbdy = tf.where( (state.icemaskobs_tcal[:, 1:, :] > 0.5) & (state.icemaskobs_tcal[:, :-1, :] > 0.5) , dbdy, 0.0)

        # if params.fix_opti_normalization_issue:
        REGU_H = (params.tcal_regu_param_thk) * 0.5 * (
            tf.math.reduce_mean(tf.math.reduce_mean(dbdx**2,axis=1),axis=1) + tf.math.reduce_mean(tf.math.reduce_mean(dbdy**2,axis=1),axis=1)
            - gamma * tf.math.reduce_mean(tf.math.reduce_mean(state.thk_tcal,axis=1),axis=1)
        )
        # else:
        #     REGU_H = (params.tcal_regu_param_thk) * (
        #         tf.nn.l2_loss(dbdx) + tf.nn.l2_loss(dbdy)
        #         - gamma * tf.math.reduce_sum(tf.math.reduce_sum(state.thk,axis=1),axis=1)
        #     )
    else:
        dbdx = (field[:, :, 1:] - field[:, :, :-1])/state.dx
        dbdx = (dbdx[:, 1:, :] + dbdx[:, :-1, :]) / 2.0
        dbdy = (field[:, 1:, :] - field[:, :-1, :])/state.dx
        dbdy = (dbdy[:, :, 1:] + dbdy[:, :, :-1]) / 2.0

        if params.sole_mask:
            MASK = (state.icemaskobs_tcal[:, 1:, 1:] > 0.5) & (state.icemaskobs_tcal[:, 1:, :-1] > 0.5) & (state.icemaskobs_tcal[:, :-1, 1:] > 0.5) & (state.icemaskobs_tcal[:, :-1, :-1] > 0.5)
            dbdx = tf.where( MASK, dbdx, 0.0)
            dbdy = tf.where( MASK, dbdy, 0.0)
 
        # if params.fix_opti_normalization_issue:
        REGU_H = (params.tcal_regu_param_thk) * 0.5 * (
            (1.0/np.sqrt(params.tcal_smooth_anisotropy_factor))
            * tf.math.reduce_mean(tf.math.reduce_mean((dbdx * state.flowdirx_tcal + dbdy * state.flowdiry_tcal)**2, axis=1),axis=1)
            + np.sqrt(params.tcal_smooth_anisotropy_factor)
            * tf.math.reduce_mean(tf.math.reduce_mean((dbdx * state.flowdiry_tcal - dbdy * state.flowdirx_tcal)**2, axis=1), axis=1)
            - gamma * tf.math.reduce_mean(tf.math.reduce_mean(state.thk_tcal, axis=1), axis=1)
        )
        # else:
        #     REGU_H = (params.tcal_regu_param_thk) * (
        #         (1.0/np.sqrt(params.tcal_smooth_anisotropy_factor))
        #         * tf.nn.l2_loss((dbdx * state.flowdirx + dbdy * state.flowdiry))
        #         + np.sqrt(params.tcal_smooth_anisotropy_factor)
        #         * tf.nn.l2_loss((dbdx * state.flowdiry - dbdy * state.flowdirx))
        #         - tf.math.reduce_sum(gamma*state.thk)
        #     )

    return tf.math.reduce_sum(REGU_H)

def regu_slidingco(params,state):

#    if not hasattr(state, "flowdirx"):
    dadx = (state.slidingco_tcal[:, :, 1:] - state.slidingco_tcal[:, :, :-1])/state.dx
    dady = (state.slidingco_tcal[:, 1:, :] - state.slidingco_tcal[:, :-1, :])/state.dx

    if params.sole_mask:                
        dadx = tf.where( (state.icemaskobs_tcal[:, :, 1:] == 1) & (state.icemaskobs_tcal[:, :, :-1] == 1) , dadx, 0.0)
        dady = tf.where( (state.icemaskobs_tcal[:, 1:, :] == 1) & (state.icemaskobs_tcal[:, :-1, :] == 1) , dady, 0.0)

    if params.tcal_smooth_anisotropy_factor_sl == 1:
        # if params.fix_opti_normalization_issue:
        REGU_S = (params.tcal_regu_param_slidingco) * 0.5 * (
            tf.math.reduce_mean(dadx**2) + tf.math.reduce_mean(dady**2)
        )
        # else:
        #     REGU_S = (params.tcal_regu_param_slidingco) * (
        #         tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
        #     )
    else:
        dadx = (state.slidingco_tcal[:, :, 1:] - state.slidingco_tcal[:, :, :-1])/state.dx
        dadx = (dadx[:, 1:, :] + dadx[:, :-1, :]) / 2.0
        dady = (state.slidingco_tcal[:, 1:, :] - state.slidingco_tcal[:, :-1, :])/state.dx
        dady = (dady[:, :, 1:] + dady[:, :, :-1]) / 2.0
 
        if params.sole_mask:
            MASK = (state.icemaskobs_tcal[:, 1:, 1:] > 0.5) & (state.icemaskobs_tcal[:, 1:, :-1] > 0.5) & (state.icemaskobs_tcal[:, :-1, 1:] > 0.5) & (state.icemaskobs_tcal[:, :-1, :-1] > 0.5)
            dadx = tf.where( MASK, dadx, 0.0)
            dady = tf.where( MASK, dady, 0.0)
 
        # if params.fix_opti_normalization_issue:
        REGU_S = (params.tcal_regu_param_slidingco) * 0.5 * (
            (1.0/np.sqrt(params.tcal_smooth_anisotropy_factor_sl))
            * tf.math.reduce_mean((dadx * state.flowdirx_tcal + dady * state.flowdiry_tcal)**2)
            + np.sqrt(params.tcal_smooth_anisotropy_factor_sl)
            * tf.math.reduce_mean((dadx * state.flowdiry_tcal - dady * state.flowdirx_tcal)**2)
        )
        # else:
        #     REGU_S = (params.tcal_regu_param_slidingco) * (
        #         (1.0/np.sqrt(params.tcal_smooth_anisotropy_factor_sl))
        #         * tf.nn.l2_loss((dadx * state.flowdirx + dady * state.flowdiry))
        #         + np.sqrt(params.tcal_smooth_anisotropy_factor_sl)
        #         * tf.nn.l2_loss((dadx * state.flowdiry - dady * state.flowdirx)) )
 
    REGU_S = REGU_S + 10**10 * tf.math.reduce_mean( tf.where(state.slidingco_tcal >= 0, 0.0, state.slidingco_tcal**2) ) 
    # this last line serve to enforce non-negative slidingco
 
    return REGU_S

def regu_arrhenius(params,state):

#    if not hasattr(state, "flowdirx"):
    dadx = (state.arrhenius_tcal[:, :, 1:] - state.arrhenius_tcal[:, :, :-1])/state.dx
    dady = (state.arrhenius_tcal[:, 1:, :] - state.arrhenius_tcal[:, :-1, :])/state.dx

    if params.sole_mask:                
        dadx = tf.where( (state.icemaskobs_tcal[:, :, 1:] == 1) & (state.icemaskobs_tcal[:, :, :-1] == 1) , dadx, 0.0)
        dady = tf.where( (state.icemaskobs_tcal[:, 1:, :] == 1) & (state.icemaskobs_tcal[:, :-1, :] == 1) , dady, 0.0)
    
    # if params.fix_opti_normalization_issue:
    REGU_A = (params.tcal_regu_param_arrhenius) * 0.5 * (
        tf.math.reduce_mean(dadx**2) + tf.math.reduce_mean(dady**2)
    )
    # else:
    #     REGU_A = (params.tcal_regu_param_arrhenius) * (
    #         tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
    #     )

    REGU_A = REGU_A + 10**10 * tf.math.reduce_mean( tf.where(state.arrhenius >= 0, 0.0, state.arrhenius**2) ) 
    # this last line serve to enforce non-negative arrhenius 
        
    return REGU_A


@tf.function()
def compute_divflux_tcal(u, v, h, dx, dy, method='upwind'):
    """
    upwind computation of the divergence of the flux : d(u h)/dx + d(v h)/dy
    First, u and v are computed on the staggered grid (i.e. cell edges)
    Second, one extend h horizontally by a cell layer on any bords (assuming same value)
    Third, one compute the flux on the staggered grid slecting upwind quantities
    Last, computing the divergence on the staggered grid yields values def on the original grid
    """

    if method == 'upwind':

        ## Compute u and v on the staggered grid
        u = tf.concat(
            [u[:, :, 0:1], 0.5 * (u[:, :, :-1] + u[:, :, 1:]), u[:, :, -1:]], axis=2
        )  # has shape (nt,ny,nx+1)
        v = tf.concat(
            [v[:, 0:1, :], 0.5 * (v[:, :-1, :] + v[:, 1:, :]), v[:, -1:, :]], axis=1
        )  # has shape (nt,ny+1,nx)

        # Extend h with constant value at the domain boundaries
        Hx = tf.pad(h, [[0, 0], [0, 0], [1, 1]], "CONSTANT")  # has shape (nt,ny,nx+2)
        Hy = tf.pad(h, [[0, 0], [1, 1], [0, 0]], "CONSTANT")  # has shape (nt,ny+2,nx)

        ## Compute fluxes by selcting the upwind quantities
        Qx = u * tf.where(u > 0, Hx[:, :, :-1], Hx[:, :, 1:])  # has shape (nt,ny,nx+1)
        Qy = v * tf.where(v > 0, Hy[:, :-1, :], Hy[:, 1:, :])  # has shape (nt,ny+1,nx)

    elif method == 'centered':

        Qx = u * h  
        Qy = v * h  
        
        Qx = tf.concat(
            [Qx[:, :, 0:1], 0.5 * (Qx[:, :, :-1] + Qx[:, :, 1:]), Qx[:, :, -1:]], axis=2
        )  # has shape (nt,ny,nx+1) 
        
        Qy = tf.concat(
            [Qy[:, 0:1, :], 0.5 * (Qy[:, :-1, :] + Qy[:, 1:, :]), Qy[:, -1:, :]], axis=1
        )  # has shape (nt,ny+1,nx)
        
        ## Computation of the divergence, final shape is (nt,ny,nx)
    return (Qx[:, :, 1:] - Qx[:, :, :-1]) / dx + (Qy[:, 1:, :] - Qy[:, :-1, :]) / dy

##################################

def print_costs(params, state, cost, i):

    vol = ( np.sum(state.thk_tcal[-1]) * (state.dx**2) / 10**9 ).numpy()
    # mean_slidingco = tf.math.reduce_mean(state.slidingco[state.icemaskobs > 0.5])

    f = open('costs.dat','a')

    def bound(x):
        return min(x, 9999999)

    keys = list(cost.keys()) 
    if i == 0:
        L = [f"{key:>8}" for key in ["it","vol"]] + [f"{key:>12}" for key in keys]
        print("Costs:     " + "   ".join(L))
        print("   ".join([f"{key:>12}" for key in keys]),file=f)
        os.system("echo rm costs.dat >> clean.sh")

    if i % params.tcal_output_freq == 0:
        L = [datetime.datetime.now().strftime("%H:%M:%S"),f"{i:0>{8}}",f"{vol:>8.4f}"] \
          + [f"{bound(cost[key].numpy()):>12.4f}" for key in keys]
        print("   ".join(L))

    print("   ".join([f"{bound(cost[key].numpy()):>12.4f}" for key in keys]),file=f)

def save_rms_std(params, state):

    np.savetxt(
        "rms_std.dat",
        np.stack(
            [
                state.rmsthk,
                state.stdthk,
                state.rmsvel,
                state.stdvel,
                state.rmsdiv,
                state.stddiv,
                state.rmsusurf,
                state.stdusurf,
            ],
            axis=-1,
        ),
        fmt="%.10f",
        header="        rmsthk      stdthk       rmsvel       stdvel       rmsdiv       stddiv       rmsusurf       stdusurf",
    )

    os.system(
        "echo rm " + "rms_std.dat" + " >> clean.sh"
    )

def create_density_matrix(data, kernel_size):
    # Convert data to binary mask (1 for valid data, 0 for NaN)
    binary_mask = tf.where(tf.math.is_nan(data), tf.zeros_like(data), tf.ones_like(data))

    # Create a kernel for convolution (all ones)
    kernel = tf.ones((kernel_size, kernel_size, 1, 1), dtype=binary_mask.dtype)

    for i in range(data.shape[0]):
        # Apply convolution to count valid data points in the neighborhood
        density_slice = tf.nn.conv2d(tf.expand_dims(tf.expand_dims(binary_mask[i], 0), -1), 
                            kernel, strides=[1, 1, 1, 1], padding='SAME')

        # Remove the extra dimensions added for convolution
        density_slice = tf.squeeze(density_slice)

        density_slice = tf.expand_dims(density_slice, axis=0)
        if i==0:
            density = density_slice
        else:
            density = tf.concat([density, density_slice], axis=0)

    return density

def _compute_rms_std_optimization(state, i):
    I = state.icemaskobs_tcal > 0.5

    if i == 0:
        state.rmsthk = []
        state.stdthk = []
        state.rmsvel = []
        state.stdvel = []
        state.rmsusurf = []
        state.stdusurf = []
        state.rmsdiv = []
        state.stddiv = []

    if hasattr(state, "thkobs_tcal"):
        ACT = ~tf.math.is_nan(state.thkobs_tcal)
        if np.sum(ACT) == 0:
            state.rmsthk.append(0)
            state.stdthk.append(0)
        else:
            state.rmsthk.append(np.nanmean(state.thk_tcal[ACT] - state.thkobs_tcal[ACT]))
            state.stdthk.append(np.nanstd(state.thk_tcal[ACT] - state.thkobs_tcal[ACT]))

    else:
        state.rmsthk.append(0)
        state.stdthk.append(0)

    if hasattr(state, "uvelsurfobs_tcal"):
        velsurf_mag = getmag(state.uvelsurf_tcal, state.vvelsurf_tcal).numpy()
        velsurfobs_mag = getmag(state.uvelsurfobs_tcal, state.vvelsurfobs_tcal).numpy()
        ACT = ~np.isnan(velsurfobs_mag)

        state.rmsvel.append(
            np.mean(velsurf_mag[(I & ACT).numpy()] - velsurfobs_mag[(I & ACT).numpy()])
        )
        state.stdvel.append(
            np.std(velsurf_mag[(I & ACT).numpy()] - velsurfobs_mag[(I & ACT).numpy()])
        )
    else:
        state.rmsvel.append(0)
        state.stdvel.append(0)

    if hasattr(state, "divfluxobs_tcal"):
        state.rmsdiv.append(np.mean(state.divfluxobs_tcal[I] - state.divflux_tcal[I]))
        state.stddiv.append(np.std(state.divfluxobs_tcal[I] - state.divflux_tcal[I]))
    else:
        state.rmsdiv.append(0)
        state.stddiv.append(0)

    if hasattr(state, "usurfobs_tcal"):
        state.rmsusurf.append(np.mean(state.usurf_tcal[I] - state.usurfobs_tcal[I]))
        state.stdusurf.append(np.std(state.usurf_tcal[I] - state.usurfobs_tcal[I]))
    else:
        state.rmsusurf.append(0)
        state.stdusurf.append(0)


def _compute_flow_direction_for_anisotropic_smoothing(state):
    uvelsurf = tf.where(tf.math.is_nan(state.uvelsurf_tcal), 0.0, state.uvelsurf_tcal)
    vvelsurf = tf.where(tf.math.is_nan(state.vvelsurf_tcal), 0.0, state.vvelsurf_tcal)

    state.flowdirx_tcal = (
        uvelsurf[:, 1:, 1:] + uvelsurf[:, :-1, 1:] + uvelsurf[:, 1:, :-1] + uvelsurf[:, :-1, :-1]
    ) / 4.0
    state.flowdiry_tcal = (
        vvelsurf[:, 1:, 1:] + vvelsurf[:, :-1, 1:] + vvelsurf[:, 1:, :-1] + vvelsurf[:, :-1, :-1]
    ) / 4.0

    from scipy.ndimage import gaussian_filter

    state.flowdirx_tcal = gaussian_filter(state.flowdirx_tcal, 3, mode="constant", axes=(1,2))
    state.flowdiry_tcal = gaussian_filter(state.flowdiry_tcal, 3, mode="constant", axes=(1,2))

    # Same as gaussian filter above but for tensorflow is (NOT TESTED)
    # import tensorflow_addons as tfa
    # state.flowdirx = ( tfa.image.gaussian_filter2d( state.flowdirx , sigma=3, filter_shape=100, padding="CONSTANT") )

    state.flowdirx_tcal /= getmag(state.flowdirx_tcal, state.flowdiry_tcal) # modified getmag() in utils.py to work with _tcal variables
    state.flowdiry_tcal /= getmag(state.flowdirx_tcal, state.flowdiry_tcal)

    state.flowdirx_tcal = tf.where(tf.math.is_nan(state.flowdirx_tcal), 0.0, state.flowdirx_tcal)
    state.flowdiry_tcal = tf.where(tf.math.is_nan(state.flowdiry_tcal), 0.0, state.flowdiry_tcal)
    
    # state.flowdirx = tf.zeros_like(state.flowdirx)
    # state.flowdiry = tf.ones_like(state.flowdiry)

    # this is to plot the observed flow directions
    # fig, axs = plt.subplots(1, 1, figsize=(8,16))
    # plt.quiver(state.flowdirx,state.flowdiry)
    # axs.axis("equal")

def minmod(a, b):
    return tf.where( (tf.abs(a)<tf.abs(b))&(a*b>0.0), a, tf.where((tf.abs(a)>tf.abs(b))&(a*b>0.0),b,0))
    
def maxmod(a, b):
    return tf.where( (tf.abs(a)<tf.abs(b))&(a*b>0.0), b, tf.where((tf.abs(a)>tf.abs(b))&(a*b>0.0),a,0))

@tf.function()
def compute_divflux_slope_limiter_tcal(u, v, h, dx, dy, dt, slope_type):
    """
    upwind computation of the divergence of the flux : d(u h)/dx + d(v h)/dy
    propose a slope limiter for the upwind scheme with 3 options : godunov, minmod, superbee
    
    References :
    - Numerical Methods for Engineers, Leif Rune Hellevik, book
      https://folk.ntnu.no/leifh/teaching/tkt4140/._main074.html
    
    - hydro_examples github page, Michael Zingale, Ian Hawke
     collection of simple python codes that demonstrate some basic techniques used in hydrodynamics codes.
     https://github.com/python-hydro/hydro_examples
    """
    
    u = tf.concat( [u[:-1,:, 0:1], 0.5 * (u[:-1,:, :-1] + u[:-1,:, 1:]), u[:-1,:, -1:]], 2 )  # has shape (nt-1,ny,nx+1)
    v = tf.concat( [v[:-1,0:1, :], 0.5 * (v[:-1,:-1, :] + v[:-1,1:, :]), v[:-1,-1:, :]], 1 )  # has shape (nt-1,ny+1,nx)

    Hx = tf.pad(h[:-1], [[0,0],[0,0],[2,2]], 'CONSTANT') # (nt-1,ny,nx+4)
    Hy = tf.pad(h[:-1], [[0,0],[2,2],[0,0]], 'CONSTANT') # (nt-1,ny+4,nx)
    
    sigpx = (Hx[:,:,2:]-Hx[:,:,1:-1])/dx    # (nt-1,ny,nx+2)
    sigmx = (Hx[:,:,1:-1]-Hx[:,:,:-2])/dx   # (nt-1,ny,nx+2) 

    sigpy = (Hy[:,2:,:] -Hy[:,1:-1,:])/dy   # (nt-1,ny+2,nx)
    sigmy = (Hy[:,1:-1,:]-Hy[:,:-2,:])/dy   # (nt-1,ny+2,nx) 

    if slope_type == "godunov":
 
        slopex = tf.zeros_like(sigpx)  
        slopey = tf.zeros_like(sigpy)  
        
    elif slope_type == "minmod":
 
        slopex  = minmod(sigmx,sigpx) 
        slopey  = minmod(sigmy,sigpy)

    elif slope_type == "superbee":

        sig1x  = minmod( sigpx , 2.0*sigmx )
        sig2x  = minmod( sigmx , 2.0*sigpx )
        slopex = maxmod( sig1x, sig2x)

        sig1y  = minmod( sigpy , 2.0*sigmy )
        sig2y  = minmod( sigmy , 2.0*sigpy )
        slopey = maxmod( sig1y, sig2y)

    w   = Hx[:,:,1:-2] + 0.5*dx*(1.0 - u*dt/dx)*slopex[:,:,:-1]      #  (nt-1,ny,nx+1)      
    e   = Hx[:,:,2:-1] - 0.5*dx*(1.0 + u*dt/dx)*slopex[:,:,1:]       #  (nt-1,ny,nx+1)    
    
    s   = Hy[:,1:-2,:] + 0.5*dy*(1.0 - v*dt/dy)*slopey[:,:-1,:]      #  (nt-1,ny+1,nx)      
    n   = Hy[:,2:-1,:] - 0.5*dy*(1.0 + v*dt/dy)*slopey[:,1:,:]       #  (nt-1,ny+1,nx)    
     
    Qx = u * tf.where(u > 0, w, e)  #  (nt-1,ny,nx+1)   
    Qy = v * tf.where(v > 0, s, n)  #  (nt-1,ny+1,nx)   
     
    return (Qx[:, :, 1:] - Qx[:, :, :-1]) / dx + (Qy[:, 1:, :] - Qy[:, :-1, :]) / dy  # (nt-1,ny,nx)



def load_tcal_obs_data(params, state):

    # load state variables with load_ncdf.py
    # here, load multiple _obs variables that I can dynamically pass to optimize()

    with xr.open_dataset(params.tcal_input_file) as ds:

        obs_list = params.tcal_obs_list
        tcal_list = [s + "obs_tcal" for s in obs_list]

        for var in obs_list:
            vars()[var] = ds[var].sel(time=params.tcal_times)
            vars()[var] = np.where(vars()[var] > 10**35, np.nan, vars()[var])

        for obsvar, tcalvar in zip(obs_list, tcal_list):
            vars(state)[tcalvar] = tf.Variable(vars()[obsvar], dtype=tf.float32, trainable=False)




def output_ncdf_tcal_final(params, state):
    """
    Write final geology after optimizing
    """
    # if params.opti_save_iterat_in_ncdf==False:
    if "velbase_mag" in params.tcal_vars_to_save:
        state.velbase_mag_tcal = getmag(state.uvelbase_tcal, state.vvelbase_tcal)

    if "velsurf_mag" in params.tcal_vars_to_save:
        state.velsurf_mag_tcal = getmag(state.uvelsurf_tcal, state.vvelsurf_tcal)

    if "velsurfobs_mag" in params.tcal_vars_to_save:
        state.velsurfobs_mag_tcal = getmag(state.uvelsurfobs_tcal, state.vvelsurfobs_tcal)

    if "velsurfdiff_mag" in params.tcal_vars_to_save:
        vsurf = getmag(state.uvelsurf_tcal, state.vvelsurf_tcal)
        vsurfobs = getmag(state.uvelsurfobs_tcal, state.vvelsurfobs_tcal)
        state.velsurfdiff_mag_tcal = vsurf - vsurfobs
    
    if "sliding_ratio" in params.tcal_vars_to_save:
        state.sliding_ratio_tcal = tf.where(state.velsurf_mag_tcal > 10, state.velbase_mag_tcal / state.velsurf_mag_tcal, np.nan)

    if "thkdiff" in params.tcal_vars_to_save:
        state.thkdiff_tcal = tf.where(tf.math.is_nan(state.thkobs_tcal), np.nan, state.thk_tcal - state.thkobs_tcal)

    if "topgdiff" in params.tcal_vars_to_save:
        state.topgdiff_tcal = tf.where(tf.math.is_nan(state.topgobs_tcal), np.nan, state.topg_tcal - state.topgobs_tcal[0])

    if "difffluxdiff" in params.tcal_vars_to_save:
        ACT = ~tf.math.is_nan(state.divfluxobs_tcal)
        state.difffluxdiff_tcal = state.divflux_tcal[ACT] - state.divfluxobs_tcal[ACT]

    if "usurfdiff" in params.tcal_vars_to_save:
        state.usurfdiff_tcal = state.usurf_tcal - state.usurfobs_tcal

    nc = Dataset(
        params.tcal_save_result_in_ncdf,
        "w",
        format="NETCDF4",
    )

    nc.createDimension("time", None)
    E = nc.createVariable("time", np.dtype("float32").char, ("time",))
    E.units = "yr"
    E.long_name = "time"
    E.axis = "T"
    E[:] = [float(t) for t in params.tcal_times]

    nc.createDimension("y", len(state.y))
    E = nc.createVariable("y", np.dtype("float32").char, ("y",))
    E.units = "m"
    E.long_name = "y"
    E.axis = "Y"
    E[:] = state.y.numpy()

    nc.createDimension("x", len(state.x))
    E = nc.createVariable("x", np.dtype("float32").char, ("x",))
    E.units = "m"
    E.long_name = "x"
    E.axis = "X"
    E[:] = state.x.numpy()

    # for v in [var+"_tcal" for var in params.tcal_vars_to_save]:
    #     if hasattr(state, v):
    #         E = nc.createVariable(v-"_tcal", np.dtype("float32").char, ("time", "y", "x"))
    #         E.standard_name = v-"_tcal"
    #         E[:] = vars(state)[v]

    for v in params.tcal_vars_to_save:
        var = v + "_tcal"
        if hasattr(state, var):
            E = nc.createVariable(v, np.dtype("float32").char, ("time", "y", "x"))
            E.standard_name = v
            E[:] = vars(state)[var]

    nc.close()

    os.system(
        "echo rm "
        + params.opti_save_result_in_ncdf
        + " >> clean.sh"
    )