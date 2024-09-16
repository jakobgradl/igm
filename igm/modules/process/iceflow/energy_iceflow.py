import numpy as np 
import tensorflow as tf 
from igm.modules.utils import *
# import igm.modules.process.vert_flow.vert_flow as vf


@tf.function(experimental_relax_shapes=True)
def _compute_gradient_stag(s, dX, dY):
    """
    compute spatial gradient, outcome on stagerred grid
    """

    E = 2.0 * (s[:, :, 1:] - s[:, :, :-1]) / (dX[:, :, 1:] + dX[:, :, :-1])
    diffx = 0.5 * (E[:, 1:, :] + E[:, :-1, :])

    EE = 2.0 * (s[:, 1:, :] - s[:, :-1, :]) / (dY[:, 1:, :] + dY[:, :-1, :])
    diffy = 0.5 * (EE[:, :, 1:] + EE[:, :, :-1])

    return diffx, diffy


@tf.function(experimental_relax_shapes=True)
def _compute_strainrate_Glen_tf(U, V, thk, slidingco, dX, ddz, sloptopgx, sloptopgy, thr):
    # Compute horinzontal derivatives
    dUdx = (U[:, :, :, 1:] - U[:, :, :, :-1]) / dX[0, 0, 0]
    dVdx = (V[:, :, :, 1:] - V[:, :, :, :-1]) / dX[0, 0, 0]
    dUdy = (U[:, :, 1:, :] - U[:, :, :-1, :]) / dX[0, 0, 0]
    dVdy = (V[:, :, 1:, :] - V[:, :, :-1, :]) / dX[0, 0, 0]

    # Homgenize sizes in the horizontal plan on the stagerred grid
    dUdx = (dUdx[:, :, :-1, :] + dUdx[:, :, 1:, :]) / 2
    dVdx = (dVdx[:, :, :-1, :] + dVdx[:, :, 1:, :]) / 2
    dUdy = (dUdy[:, :, :, :-1] + dUdy[:, :, :, 1:]) / 2
    dVdy = (dVdy[:, :, :, :-1] + dVdy[:, :, :, 1:]) / 2

    # homgenize sizes in the vertical plan on the stagerred grid
    if U.shape[1] > 1:
        dUdx = (dUdx[:, :-1, :, :] + dUdx[:, 1:, :, :]) / 2
        dVdx = (dVdx[:, :-1, :, :] + dVdx[:, 1:, :, :]) / 2
        dUdy = (dUdy[:, :-1, :, :] + dUdy[:, 1:, :, :]) / 2
        dVdy = (dVdy[:, :-1, :, :] + dVdy[:, 1:, :, :]) / 2

    # compute the horizontal average, these quantitites will be used for vertical derivatives
    Um = (U[:, :, 1:, 1:] + U[:, :, 1:, :-1] + U[:, :, :-1, 1:] + U[:, :, :-1, :-1]) / 4
    Vm = (V[:, :, 1:, 1:] + V[:, :, 1:, :-1] + V[:, :, :-1, 1:] + V[:, :, :-1, :-1]) / 4

    if U.shape[1] > 1:
        # vertical derivative if there is at least two layears
        dUdz = (Um[:, 1:, :, :] - Um[:, :-1, :, :]) / tf.maximum(ddz, thr)
        dVdz = (Vm[:, 1:, :, :] - Vm[:, :-1, :, :]) / tf.maximum(ddz, thr)
        slc = tf.expand_dims(_stag4(slidingco), axis=1)
        dUdz = tf.where(slc > 0, dUdz, 0.01 * dUdz)
        dVdz = tf.where(slc > 0, dVdz, 0.01 * dVdz)
    else:
        # zero otherwise
        dUdz = 0.0
        dVdz = 0.0

    # This correct for the change of coordinate z -> z - b
    dUdx = dUdx - dUdz * sloptopgx
    dUdy = dUdy - dUdz * sloptopgy
    dVdx = dVdx - dVdz * sloptopgx
    dVdy = dVdy - dVdz * sloptopgy

    Exx = dUdx
    Eyy = dVdy
    Ezz = -dUdx - dVdy
    Exy = 0.5 * dVdx + 0.5 * dUdy
    Exz = 0.5 * dUdz
    Eyz = 0.5 * dVdz
    
    srx = 0.5 * ( Exx**2 + Exy**2 + Exy**2 + Eyy**2 + Ezz**2 )
    srz = 0.5 * ( Exz**2 + Eyz**2 + Exz**2 + Eyz**2 )

    return srx, srz


def _stag2(B):
    return (B[:, 1:] + B[:, :1]) / 2


def _stag4(B):
    return (B[:, 1:, 1:] + B[:, 1:, :-1] + B[:, :-1, 1:] + B[:, :-1, :-1]) / 4


def _stag4b(B):
    return (
        B[:, :, 1:, 1:] + B[:, :, 1:, :-1] + B[:, :, :-1, 1:] + B[:, :, :-1, :-1]
    ) / 4


def _stag8(B):
    return (
        B[:, 1:, 1:, 1:]
        + B[:, 1:, 1:, :-1]
        + B[:, 1:, :-1, 1:]
        + B[:, 1:, :-1, :-1]
        + B[:, :-1, 1:, 1:]
        + B[:, :-1, 1:, :-1]
        + B[:, :-1, :-1, 1:]
        + B[:, :-1, :-1, :-1]
    ) / 8


def iceflow_energy(params, U, V, fieldin): #*args):
    # if args.size == 1:
    #     fieldin = args[0]
    #     W = 'False'
    # elif args.size == 2:
    #     W = args[0]
    #     fieldin = args[1]
    thk, usurf, arrhenius, slidingco, dX, topg = fieldin

    return _iceflow_energy(
        U,
        V, 
        # W,
        thk,
        usurf,
        arrhenius,
        slidingco,
        dX,
        topg,
        params.iflo_Nz,
        params.iflo_vert_spacing,
        params.iflo_exp_glen,
        params.iflo_exp_weertman,
        params.iflo_regu_glen,
        params.iflo_regu_weertman,
        params.iflo_thr_ice_thk,
        params.iflo_ice_density,
        params.iflo_gravity_cst,
        params.iflo_new_friction_param,
        params.iflo_cf_cond,
        params.iflo_cf_eswn,
        params.iflo_div_cond,
        params.iflo_div_eswn,
        params.iflo_regu,
        params.iflo_min_sr,
        params.iflo_max_sr,
        params.iflo_force_negative_gravitational_energy
    )


@tf.function(experimental_relax_shapes=True)
def _iceflow_energy(
    U,
    V,
    # W,
    thk,
    usurf,
    arrhenius,
    slidingco,
    dX,
    topg,
    Nz,
    vert_spacing,
    exp_glen,
    exp_weertman,
    regu_glen,
    regu_weertman,
    thr_ice_thk,
    ice_density,
    gravity_cst,
    new_friction_param,
    iflo_cf_cond,
    iflo_cf_eswn,
    iflo_div_cond,
    iflo_div_eswn,
    iflo_regu,
    min_sr,
    max_sr,
    iflo_force_negative_gravitational_energy
    
):
    # warning, the energy is here normalized dividing by int_Omega

    COND = (
        (thk[:, 1:, 1:] > 0)
        & (thk[:, 1:, :-1] > 0)
        & (thk[:, :-1, 1:] > 0)
        & (thk[:, :-1, :-1] > 0)
    )
    COND = tf.expand_dims(COND, axis=1)

    # Vertical discretization
    if Nz > 1:
        zeta = np.arange(Nz) / (Nz - 1)  # formerly ...
        #zeta = tf.range(Nz, dtype=tf.float32) / (Nz - 1)
        temp = (zeta / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta)
        temd = temp[1:] - temp[:-1]
        dz = tf.stack([_stag4(thk) * z for z in temd], axis=1)  # formerly ..
        #dz = (tf.expand_dims(tf.expand_dims(temd,axis=-1),axis=-1)*tf.expand_dims(_stag4(thk),axis=0))
    else:
        dz = tf.expand_dims(_stag4(thk), axis=0)

    # B has Unit Mpa y^(1/n)
    B = 2.0 * arrhenius ** (-1.0 / exp_glen)

    if new_friction_param:
        C = 1.0 * slidingco  # C has unit Mpa y^m m^(-m)
    else:
        if exp_weertman == 1:
            # C has unit Mpa y^m m^(-m)
            C = 1.0 * slidingco
        else:
            C = (slidingco + 10 ** (-12)) ** -(1.0 / exp_weertman)

    p = 1.0 + 1.0 / exp_glen
    s = 1.0 + 1.0 / exp_weertman

    sloptopgx, sloptopgy = _compute_gradient_stag(usurf - thk, dX, dX)
    sloptopgx = tf.expand_dims(sloptopgx, axis=1)
    sloptopgy = tf.expand_dims(sloptopgy, axis=1)

    # TODO : sloptopgx, sloptopgy must be the elevaion of layers! not the bedrock, this probably has very little effects.

    # sr has unit y^(-1)
    srx, srz = _compute_strainrate_Glen_tf(
        U, V, thk, C, dX, dz, sloptopgx, sloptopgy, thr=thr_ice_thk
    )
    
    sr = srx + srz

    sr = tf.where(COND, sr, 0.0)
    
    srcapped = tf.clip_by_value(sr, min_sr**2, max_sr**2)

    srcapped = tf.where(COND, srcapped, 0.0)
 

    # C_shear is unit  Mpa y^(1/n) y^(-1-1/n) * m = Mpa m/y
    if len(B.shape) == 3:
        C_shear = _stag4(B) * tf.reduce_sum(dz * ((srcapped + regu_glen**2) ** ((p-2) / 2)) * sr, axis=1 ) / p
    else:
        C_shear = tf.reduce_sum( _stag8(B) * dz * ((srcapped + regu_glen**2) ** ((p-2) / 2)) * sr, axis=1 ) / p
        
    if iflo_regu > 0:
        
        srx = tf.where(COND, srx, 0.0)
 
        if len(B.shape) == 3:
            C_shear_2 = _stag4(B) * tf.reduce_sum(dz * ((srx + regu_glen**2) ** (p / 2)), axis=1 ) / p
        else:
            C_shear_2 = tf.reduce_sum( _stag8(B) * dz * ((srx + regu_glen**2) ** (p / 2)), axis=1 ) / p 

        C_shear = C_shear + iflo_regu*C_shear_2
        
    lsurf = usurf - thk

    sloptopgx, sloptopgy = _compute_gradient_stag(lsurf, dX, dX)

    # C_slid is unit Mpa y^m m^(-m) * m^(1+m) * y^(-1-m)  = Mpa  m/y
    N = (
        _stag4(U[:, 0, :, :] ** 2 + V[:, 0, :, :] ** 2)
        + regu_weertman**2
        + (_stag4(U[:, 0, :, :]) * sloptopgx + _stag4(V[:, 0, :, :]) * sloptopgy) ** 2
    )
    C_slid = _stag4(C) * N ** (s / 2) / s

    slopsurfx, slopsurfy = _compute_gradient_stag(usurf, dX, dX)
    slopsurfx = tf.expand_dims(slopsurfx, axis=1)
    slopsurfy = tf.expand_dims(slopsurfy, axis=1)

    # if not iflo_cf_cond:
    if Nz > 1:
        uds = _stag8(U) * slopsurfx + _stag8(V) * slopsurfy
    else:
        uds = _stag4b(U) * slopsurfx + _stag4b(V) * slopsurfy  

    if iflo_force_negative_gravitational_energy:
        uds = tf.minimum(uds, 0.0) # force non-postiveness

    uds = tf.where(COND, uds, 0.0)

    # C_slid is unit Mpa m^-1 m/y m = Mpa m/y
    C_grav = (
        ice_density
        * gravity_cst
        * 10 ** (-6)
        * tf.reduce_sum(dz * uds, axis=1)
    )

    # elif iflo_cf_cond: ### THIS PRODUCES REASONABLE ENERGY VALUES
    #     # floating = tf.where(lsurf > topg, 1.0, 0.0)
    #     # grounded = tf.where(lsurf > topg, 0.0, 1.0)

    #     # floating = tf.expand_dims(_stag4(floating), axis=1)
    #     # grounded = tf.expand_dims(_stag4(grounded), axis=1)

    #     usurf = tf.expand_dims(_stag4(usurf), axis=1)
    #     lsurf = tf.expand_dims(_stag4(lsurf), axis=1)
    #     topg = tf.expand_dims(_stag4(topg), axis=1)
    #     # thk = tf.expand_dims(_stag4(thk), axis=1)

    #     C_grav_float = tf.zeros_like(_stag8(U))
    #     C_grav_ground = tf.zeros_like(_stag8(U))

    #     # grounded ice
    #     if Nz > 1:
    #         uds = tf.where((topg-lsurf)>1, 0.0, _stag8(U) * slopsurfx + _stag8(V) * slopsurfy)
    #         # (topg-lsurf)>1 works better than lsurf>topg, probably a float-precision issue
    #     else:
    #         uds = tf.where((topg-lsurf)>1, 0.0, _stag4b(U) * slopsurfx + _stag4b(V) * slopsurfy)

    #     if iflo_force_negative_gravitational_energy:
    #         uds = tf.minimum(uds, 0.0) # force non-postiveness

    #     uds = tf.where(COND, uds, 0.0)

    #     # C_slid is unit Mpa m^-1 m/y m = Mpa m/y
    #     C_grav_ground = (
    #         ice_density
    #         * gravity_cst
    #         * 10 ** (-6)
    #         * tf.reduce_sum(dz * uds, axis=1)
    #         )

    #     # floating ice
    #     # Cuffey&Patterson, 2010, Eq. 8.110
    #     if Nz > 1:
    #         uds = tf.where((topg-lsurf)>1, _stag8(U) * usurf + _stag8(V) * usurf, 0.0)
    #     else:
    #         uds = tf.where((topg-lsurf)>1, _stag4b(U) * usurf + _stag4b(V) * usurf, 0.0)

    #     if iflo_force_negative_gravitational_energy:
    #         uds = tf.minimum(uds, 0.0) # force non-postiveness

    #     uds = tf.where(COND, uds, 0.0)

    #     C_grav_float = (
    #         0.5
    #         * ice_density
    #         * gravity_cst
    #         * 10 ** (-6)
    #         * tf.reduce_sum(dz * uds, axis=1)
    #         )
        
    #     C_grav = C_grav_float + C_grav_ground






    # elif iflo_cf_cond:
    #     # floating = tf.where(lsurf > topg, 1.0, 0.0)
    #     # grounded = tf.where(lsurf > topg, 0.0, 1.0)

    #     # floating = tf.expand_dims(_stag4(floating), axis=1)
    #     # grounded = tf.expand_dims(_stag4(grounded), axis=1)

    #     usurf = tf.expand_dims(_stag4(usurf), axis=1)
    #     lsurf = tf.expand_dims(_stag4(lsurf), axis=1)
    #     topg = tf.expand_dims(_stag4(topg), axis=1)
    #     thk = tf.expand_dims(_stag4(thk), axis=1)

    #     C_grav_float = tf.zeros_like(_stag8(U))
    #     C_grav_ground = tf.zeros_like(_stag8(U))

    #     # grounded ice
    #     if Nz > 1:
    #         uds = tf.where(lsurf<0, 0.0, _stag8(U) * slopsurfx + _stag8(V) * slopsurfy)
    #         # (topg-lsurf)>1 works better than lsurf>topg, probably a float-precision issue
    #     else:
    #         uds = tf.where(lsurf<0, 0.0, _stag4b(U) * slopsurfx + _stag4b(V) * slopsurfy)

    #     if iflo_force_negative_gravitational_energy:
    #         uds = tf.minimum(uds, 0.0) # force non-postiveness

    #     uds = tf.where(COND, uds, 0.0)

    #     # C_slid is unit Mpa m^-1 m/y m = Mpa m/y
    #     C_grav_ground = (
    #         ice_density
    #         * gravity_cst
    #         * 10 ** (-6)
    #         * tf.reduce_sum(dz * uds, axis=1)
    #         )

    #     # floating ice
    #     # Cuffey&Patterson, 2010, Eq. 8.110
    #     if Nz > 1:
    #         P = tf.where(lsurf<0, 0.5 * 10 ** (-6) * 9.81 * 910 * ( thk**2 - (1000/910)*lsurf**2 ), 0.0)  / dX[:, 0, 0]
    #     else:
    #         P = tf.where(lsurf<0, 0.5 * 10 ** (-6) * 9.81 * 910 * ( thk**2 - (1000/910)*lsurf**2 ), 0.0)  / dX[:, 0, 0]

    #     # if iflo_force_negative_gravitational_energy:
    #     #     uds = tf.minimum(uds, 0.0) # force non-postiveness

    #     # uds = tf.where(COND, uds, 0.0)

    #     C_grav_float = (
    #         # 0.5
    #         # * ice_density
    #         # * gravity_cst
    #         # * 10 ** (-6)
    #         # * 
    #         P * (
    #         tf.reduce_sum(dz * _stag8(U), axis=1)
    #         - tf.reduce_sum(dz * _stag8(V), axis=1)
    #         ))

    #     C_grav = C_grav_float + C_grav_ground
    




    # if activae this applies the stress condition along the calving front
    if iflo_cf_cond:

        ################################################################
        
        lsurf = usurf - thk

        density_ratio = 910/1000

        shelf = tf.where(lsurf<0)
        len_shelf = tf.reduce_sum(dX[:,0,shelf[0,1]:])
        thk_shelf = tf.ones_like(thk) * thk[0,-1]

        #P = tf.where(lsurf==(-density_ratio*thk), 0.5 * ice_density * gravity_cst * thk * usurf , 0.0)  / dX[:, 0, 0]
        #shelf = tf.where(lsurf==(-density_ratio*thk), 1.0, 0.0)
        
        
    #   Check formula (17) in [Jouvet and Graeser 2012], Unit is Mpa 

        # Eq 17, ii)
        P =tf.where(lsurf<0, 0.5 * 10 ** (-6) * 9.81 * 910 * ( thk**2 - (1000/910)*lsurf**2 ) , 0.0)  / dX[:, 0, 0]
        # P =tf.where(lsurf<0, 0.5 * 10 ** (-6) * 9.81 * 910 * ( thk**2 - (1000/910)*lsurf**2 ) , 0.0)  / len_shelf

        # Eq 17, iii) 
        # P =tf.where(lsurf==(-density_ratio*thk), 0.5 * 10 ** (-6) * 9.81 * 910 * thk_shelf**2 * (1 - density_ratio) , 0.0) / len_shelf
        
        if len(iflo_cf_eswn) == 0:
            thkext = tf.pad(thk,[[0,0],[1,1],[1,1]],"CONSTANT",constant_values=1)
            lsurfext = tf.pad(lsurf,[[0,0],[1,1],[1,1]],"CONSTANT",constant_values=1)
        else:
            thkext = thk
            thkext = tf.pad(thkext,[[0,0],[1,0],[0,0]],"CONSTANT",constant_values=1.0*('S' not in iflo_cf_eswn))
            thkext = tf.pad(thkext,[[0,0],[0,1],[0,0]],"CONSTANT",constant_values=1.0*('N' not in iflo_cf_eswn))
            thkext = tf.pad(thkext,[[0,0],[0,0],[1,0]],"CONSTANT",constant_values=1.0*('W' not in iflo_cf_eswn))
            thkext = tf.pad(thkext,[[0,0],[0,0],[0,1]],"CONSTANT",constant_values=1.0*('E' not in iflo_cf_eswn)) 
            lsurfext = lsurf
            lsurfext = tf.pad(lsurfext,[[0,0],[1,0],[0,0]],"CONSTANT",constant_values=1.0*('S' not in iflo_cf_eswn))
            lsurfext = tf.pad(lsurfext,[[0,0],[0,1],[0,0]],"CONSTANT",constant_values=1.0*('N' not in iflo_cf_eswn))
            lsurfext = tf.pad(lsurfext,[[0,0],[0,0],[1,0]],"CONSTANT",constant_values=1.0*('W' not in iflo_cf_eswn))
            lsurfext = tf.pad(lsurfext,[[0,0],[0,0],[0,1]],"CONSTANT",constant_values=1.0*('E' not in iflo_cf_eswn)) 
        
        # this permits to locate the calving front in a cell in the 4 directions
        CF_W = tf.where((lsurf<0)&(thk>0)&(thkext[:,1:-1,:-2]==0)&(lsurfext[:,1:-1,:-2]<=0),1.0,0.0)
        CF_E = tf.where((lsurf<0)&(thk>0)&(thkext[:,1:-1,2:]==0)&(lsurfext[:,1:-1,2:]<=0),1.0,0.0) 
        CF_S = tf.where((lsurf<0)&(thk>0)&(thkext[:,:-2,1:-1]==0)&(lsurfext[:,:-2,1:-1]<=0),1.0,0.0)
        CF_N = tf.where((lsurf<0)&(thk>0)&(thkext[:,2:,1:-1]==0)&(lsurfext[:,2:,1:-1]<=0),1.0,0.0)

        # CF_E = tf.where((lsurf<0)&(thk>0), 1.0, 0.0)
 
        if Nz > 1:
            # Blatter-Pattyn
            weight = tf.stack([tf.ones_like(thk) * z for z in temd], axis=1) # dimensionless, 
            C_float = (
                  P * tf.reduce_sum(weight * _stag2(U), axis=1) * CF_W
                - P * tf.reduce_sum(weight * _stag2(U), axis=1) * CF_E #* dX[:, 0, 0]
                + P * tf.reduce_sum(weight * _stag2(V), axis=1) * CF_S 
                - P * tf.reduce_sum(weight * _stag2(V), axis=1) * CF_N 
            )
            # C_float = (P * tf.reduce_sum(weight * _stag2(U), axis=1) + P * tf.reduce_sum(weight * _stag2(V), axis=1))
        else:
            # SSA
            C_float = ( P * U * CF_W - P * U * CF_E  + P * V * CF_S - P * V * CF_N )  

        ###########################################################

        # ddz = tf.stack([thk * z for z in temd], axis=1) 

        # zzz = tf.expand_dims(lsurf, axis=1) + tf.math.cumsum(ddz, axis=1)

        # f = 10 ** (-6) * ( 910 * 9.81 * (tf.expand_dims(usurf, axis=1) - zzz) + 1000 * 9.81 * tf.minimum(0.0, zzz) )  # Mpa m^(-1) 

        # C_float = (
        #       tf.reduce_sum(ddz * f * _stag2(U), axis=1) * CF_W
        #     - tf.reduce_sum(ddz * f * _stag2(U), axis=1) * CF_E 
        #     + tf.reduce_sum(ddz * f * _stag2(V), axis=1) * CF_S 
        #     - tf.reduce_sum(ddz * f * _stag2(V), axis=1) * CF_N 
        # )   # Mpa m / y
        
    #     ##########################################################

    #     # f = 10 ** (-6) * ( 910 * 9.81 * thk + 1000 * 9.81 * tf.minimum(0.0, lsurf) ) # Mpa 

 
    #     # sloptopgx, sloptopgy = compute_gradient_tf(lsurf[0], dX[0, 0, 0], dX[0, 0, 0])
    #     # slopn = (sloptopgx**2 + sloptopgy**2 + 1.e-10 )**0.5
    #     # nx = tf.expand_dims(sloptopgx/slopn,0)
    #     # ny = tf.expand_dims(sloptopgy/slopn,0)
            
    #     # C_float_2 = - tf.where( (thk>0)&(slidingco==0), - f * (U[:,0] * nx + V[:,0] * ny), 0.0 ) # Mpa m/y

    #     # #C_float is unit  Mpa m * (m/y) / m + Mpa m / y = Mpa m / y    
    #     # C_float = C_float + C_float_2 
        
    else:
        C_float = tf.zeros_like(C_shear)



    # Regularization of the velocity gradient through H1 semi norm
    # sqrt( integral( abs(grad(u))^2 )dx )
    if Nz > 1:
        gradu = (U[:,:,:,:-1]-U[:,:,:,1:])/dX[:,0,0]
        gradv = (V[:,:,:-1,:]-V[:,:,1:,:])/dX[:,0,0]
        grad2u = (gradu[:,:,:,:-1]-gradu[:,:,:,1:])/dX[:,0,0]
        grad2v = (gradv[:,:,:,:-1]-gradv[:,:,:,1:])/dX[:,0,0]
        C_h1 = tf.sqrt(tf.reduce_sum(tf.square(grad2u))) + tf.sqrt(tf.reduce_sum(tf.square(grad2v)))
    else:
        grad2u = _stag4b(U) + _stag4b(V)
        C_h1 = tf.sqrt(tf.reduce_sum(tf.square(_stag4b(grad2u))))


    # # Regularization of surface velocity change for training stability
    # # new surface velocities should be within 10% of old velocities
    # Vel_current = tf.sqrt(tf.square(U_old)+tf.square(V_old))
    # Vel_new = tf.sqrt(tf.square(U[:,0,:,:])+tf.square(V[:,0,:,:]))
    # Vel_diff = (Vel_new - Vel_current) / Vel_current
    # Vel_diff = tf.sqrt(tf.square(Vel_diff))
    # if tf.reduce_max(Vel_diff) > 0.1:
    #     C_stabil = 0.1 * Vel_diff
    # else:
    #     C_stabil = 0.0

    # Boundary conditions

    # ice divide and no slip BCs
    # residue = tf.sqrt(tf.reduce_sum(U[:,:,:,0])**2 + tf.reduce_sum(V[:,:,0,:])**2 + tf.reduce_sum(V[:,:,-1,:])**2) * 10**(-6)

    # Calving front BC, force zero vertical velocity
    # W = _compute_vertical_velocity_incompressibility(Nz, vert_spacing, topg, U[0,:,:,:], V[0,:,:,:], thk, dX)
    # res_cf = tf.sqrt(tf.reduce_sum(W[:,:,-1])**2) * 10**(-6)

    # COST_energy = (tf.reduce_mean(C_shear) 
    #                + tf.reduce_mean(C_slid) 
    #                + tf.reduce_mean(C_grav+(residue/(thk.shape[1]*thk.shape[2]))) 
    #                + tf.reduce_mean(C_float+(res_cf/thk.shape[1])))

    # return C_shear, C_slid, C_grav+(residue/(thk.shape[1]*thk.shape[2])), C_float+(res_cf/thk.shape[1]), C_h1
    return C_shear, C_slid, C_grav, C_float, C_h1
    # return COST_energy, C_h1#, C_stabil






# @tf.function(experimental_relax_shapes=True)
def iceflow_energy_XY(state, params, X, Y):
    if params.iflo_emulate_vert_vel:
        U, V, W = Y_to_UVW(params, Y)
    else:
        U, V = Y_to_UV(params, Y)

    # if params.iflo_div_cond: # or params.iflo_cf_cond:
    #     U, V = _impose_BCs(state, params, U, V)

    fieldin = X_to_fieldin(params, X)

    if params.iflo_emulate_vert_vel:
        return iceflow_energy(params, U, V, W, fieldin)
    else:
        return iceflow_energy(params, U, V, fieldin)
    


def _compute_vertical_velocity_incompressibility(Nz, vert_space, topg, U, V, thk, dX):
    # Compute horinzontal derivatives
    dUdx = (U[:, :, 2:] - U[:, :, :-2]) / (2 * dX[0, 0, 0])
    dVdy = (V[:, 2:, :] - V[:, :-2, :]) / (2 * dX[0, 0, 0])

    dUdx = tf.pad(dUdx, [[0, 0], [0, 0], [1, 1]], "CONSTANT")
    dVdy = tf.pad(dVdy, [[0, 0], [1, 1], [0, 0]], "CONSTANT")

    dUdx = (dUdx[1:] + dUdx[:-1]) / 2  # compute between the layers
    dVdy = (dVdy[1:] + dVdy[:-1]) / 2  # compute between the layers

    # get dVdz from impcrompressibility condition
    dVdz = -dUdx - dVdy

    # get the basal vertical velocities
    sloptopgx, sloptopgy = compute_gradient_tf(topg, dX[0, 0, 0], dX[0, 0, 0])
    wvelbase = U[0] * sloptopgx + V[0] * sloptopgy

    # get the vertical thickness layers
    zeta = np.arange(Nz) / (Nz - 1)
    temp = (zeta / vert_space) * (
        1.0 + (vert_space - 1.0) * zeta
    )
    temd = temp[1:] - temp[:-1]
    dz = tf.stack([thk * z for z in temd], axis=0)

    W = []
    W.append(wvelbase)
    for l in range(dVdz.shape[0]):
        W.append(W[-1] + dVdz[l] * dz[l])
    W = tf.stack(W)

    return W



def _impose_BCs(state, params, U, V):
    # exact boundary conditions
    # Barschkis, 2023; Sukumar & Srivastava, 2021
    # Prediction_with_BCs = G + phi * Prediction_without_BCs

    # let's start simple: ice divide at W-boundary with U=0
    
    # dist = tf.zeros_like(state.X)
    # for i in np.arange(state.X.shape[0]):
    #     dist_new = np.square(state.X - state.X[i,0])
    #     dist += dist_new
    # dist = np.sqrt(dist)

    if params.iflo_div_cond:
        # ice divide boundary, i.e., 
        BCvel = params.iflo_div_vel # default 10e-3 (1 mm/yr)
        G_u = tf.zeros_like(state.X) + BCvel
        exp = 1
        extent = np.max(state.X)
        phi = state.X * ((-100.0 / ( state.X + (100.0)**(1/exp) )**exp) + 1.0) # for a boundary in the West

    nU = tf.Variable(tf.ones_like(U)*U)
    # G = G_cf + G_u
    for i in np.arange(nU.shape[0]):
        for j in np.arange(nU.shape[1]):
            nU[i,j,:,:].assign(phi * nU[i,j,:,:])
            nU[i,j,:,:].assign(nU[i,j,:,:] + G_u)
    # V = G_v + phi * V

    if params.iflo_fslip_cond:
        # free slip boundary, i.e., no flow perpendicular to boundary
        G_v = 0.0 * tf.ones_like(state.X)
        exp = 1
        phi = state.Y * ((-100.0 / ( state.Y + (100.0)**(1/exp) )**exp) + 1.0) # for a boundary in the North and South

    nV = tf.Variable(tf.ones_like(V)*V)
    # G = G_cf + G_u
    for i in np.arange(nV.shape[0]):
        for j in np.arange(nV.shape[1]):
            nV[i,j,:,:].assign(phi * nV[i,j,:,:])
            nV[i,j,:,:].assign(nV[i,j,:,:] + G_v)
    # V = G_v + phi * V
 
    return nU, V


def Y_to_UV(params, Y):
    N = params.iflo_Nz

    U = tf.experimental.numpy.moveaxis(Y[:, :, :, :N], [-1], [1])
    V = tf.experimental.numpy.moveaxis(Y[:, :, :, N:], [-1], [1])

    return U, V

def Y_to_UVW(params, Y):
    N = params.iflo_Nz

    U = tf.experimental.numpy.moveaxis(Y[:, :, :, :N], [-1], [1])
    V = tf.experimental.numpy.moveaxis(Y[:, :, :, N:-N], [-1], [1])
    W = tf.experimental.numpy.moveaxis(Y[:, :, :, -N:], [-1], [1])

    return U, V, W


def UV_to_Y(params, U, V):
    UU = tf.experimental.numpy.moveaxis(U, [0], [-1])
    VV = tf.experimental.numpy.moveaxis(V, [0], [-1])
    RR = tf.expand_dims(
        tf.concat(
            [UU, VV],
            axis=-1,
        ),
        axis=0,
    )

    return RR


def fieldin_to_X(params, fieldin):
    X = []

    fieldin_dim = [0, 0, 1 * (params.iflo_dim_arrhenius == 3), 0, 0, 0]

    for f, s in zip(fieldin, fieldin_dim):
        if s == 0:   
            X.append(tf.expand_dims(f, axis=-1))
        else:
            X.append(tf.experimental.numpy.moveaxis(f, [0], [-1]))

    return tf.expand_dims(tf.concat(X, axis=-1), axis=0)


def X_to_fieldin(params, X):
    i = 0

    fieldin_dim = [0, 0, 1 * (params.iflo_dim_arrhenius == 3), 0, 0, 0]

    fieldin = []

    # for f, s in zip(params.iflo_fieldin, fieldin_dim):
    for f, s in zip(np.arange(6), fieldin_dim):
        if s == 0:
            fieldin.append(X[:, :, :, i])
            i += 1
        else:
            fieldin.append(
                tf.experimental.numpy.moveaxis(
                    X[:, :, :, i : i + params.iflo_Nz], [-1], [1]
                )
            )
            i += params.iflo_Nz

    return fieldin
