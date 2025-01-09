from igm.modules.utils import *

def params_tcal(parser):

    parser.add_argument(
        "--tcal_input_file",
        type=str,
        default="input.nc",
        help="NetCDF input data file",
    )
    parser.add_argument(
        "--tcal_times",
        type=list,
        default=[2099, 2100],
        help="List of timesteps (type integer) used in transient calibration",
    )
    parser.add_argument(
        "--tcal_obs_list",
        type=list,
        default=["uvelsurf", "vvelsurf", "thk", "usurf", "smb"],
        help="List of observed variables used in transient calibration",
    )

    parser.add_argument(
        "--tcal_init_zero_thk",
        type=str2bool,
        default="False",
        help="Initialize the optimization with zero ice thickness",
    )
    parser.add_argument(
        "--tcal_init_const_sl",
        type=str2bool, 
        default="False",
        help="Initialize the optimization with spatially constant slidingco"
    )

    parser.add_argument(
        "--tcal_control_trans",
        type=list,
        default=["topg", "thk", "usurf", "slidingco", "arrhenius"],
        help="List of control parameters"
    )
    parser.add_argument(
        "--tcal_control_const", # ALL const parameters also need to be in tcal_control
        type=list,
        default=["topg"], # for other paramters you need to add a cost function in trans_calib.py
        help="Are any of the control parameters CONSTANT over time?"
    )
    parser.add_argument(
        "--tcal_cost",
        type=list,
        default=["usurf", "velsurf", "thk", "icemask"],
        help="List of cost parameters"
    )
    parser.add_argument(
        "--tcal_step_size",
        type=float,
        default=0.001,
        help="Step size for optimization"
    )
    parser.add_argument(
        "--tcal_step_size_decay",
        type=float,
        default=0.9,
        help="Step size decay for optimization"
    )
    parser.add_argument(
        "--tcal_nbitmin",
        type=int,
        default=50,
        help="Min iterations for the optimization",
    )
    parser.add_argument(
        "--tcal-nbitmax",
        type=int,
        default=1000,
        help="Maximum number of iterations in the transient optimization"
    )
    parser.add_argument(
        "--tcal_output_freq",
        type=int,
        default=50,
        help="Frequency of the output for the optimization",
    )
    parser.add_argument(
        "--tcal_retrain_iceflow_model",
        type=str2bool,
        default=True,
        help="Retrain the iceflow model simulatounously ?",
    )

    # parser.add_argument(
    #     "--tcal_scaling_thk",
    #     type=float,
    #     default=2.0,
    #     help="Scaling factor for the ice thickness in the optimization, serve to adjust step-size of each controls relative to each other",
    # )
    # parser.add_argument(
    #     "--tcal_scaling_slidingco",
    #     type=float,
    #     default=0.0001,
    #     help="Scaling factor for the slidingco in the optimization, serve to adjust step-size of each controls relative to each other",
    # )
    # parser.add_argument(
    #     "--tcal_scaling_arrhenius",
    #     type=float,
    #     default=0.1,
    #     help="Scaling factor for the Arrhenius in the optimization, serve to adjust step-size of each controls relative to each other",
    # )
    parser.add_argument(
        "--tcal_uniformize_thkobs",
        type=str2bool,
        default=True,
        help="uniformize the density of thkobs",
    )

    parser.add_argument(
        "--tcal_regu_param_thk",
        type=float,
        default=10.0,
        help="Regularization weight for the ice thickness in the optimization",
    )
    parser.add_argument(
        "--tcal_regu_param_slidingco",
        type=float,
        default=1,
        help="Regularization weight for the slidingco field in the optimization",
    )
    parser.add_argument(
        "--tcal_regu_param_arrhenius",
        type=float,
        default=10.0,
        help="Regularization weight for the arrhenius field in the optimization",
    )
    parser.add_argument(
        "--tcal_regu_param_div",
        type=float,
        default=1,
        help="Regularization weight for the divrgence field in the optimization",
    )

    parser.add_argument(
        "--tcal_thk_slope_type",
        type=str,
        default="superbee",
        help="Type of slope limiter for the dhdt cost function (godunov or superbee)",
    )

    parser.add_argument(
        "--tcal_smooth_anisotropy_factor",
        type=float,
        default=0.2,
        help="Smooth anisotropy factor for the ice thickness regularization in the optimization",
    )
    parser.add_argument(
        "--tcal_smooth_anisotropy_factor_sl",
        type=float,
        default=1.0,
        help="Smooth anisotropy factor for the slidingco regularization in the optimization",
    )
    parser.add_argument(
        "--tcal_convexity_weight",
        type=float,
        default=0.002,
        help="Convexity weight for the ice thickness regularization in the optimization",
    )
    parser.add_argument(
        "--tcal_convexity_power",
        type=float,
        default=1.3,
        help="Power b in the area-volume scaling V ~ a * A^b taking fom 'An estimate of global glacier volume', A. Grinste, TC, 2013",
    )

    parser.add_argument(
        "--tcal_usurfobs_std",
        type=float,
        default=2.0,
        help="Confidence/STD of the top ice surface as input data for the optimization",
    )
    parser.add_argument(
        "--tcal_velsurfobs_std",
        type=float,
        default=1.0,
        help="Confidence/STD of the surface ice velocities as input data for the optimization (if 0, velsurfobs_std field must be given)",
    )
    parser.add_argument(
        "--tcal_thkobs_std",
        type=float,
        default=3.0,
        help="Confidence/STD of the ice thickness profiles (unless given)",
    )
    parser.add_argument(
        "--tcal_divfluxobs_std",
        type=float,
        default=1.0,
        help="Confidence/STD of the flux divergence as input data for the optimization (if 0, divfluxobs_std field must be given)",
    )
    parser.add_argument(
        "--tcal_dSdtobs_std",
        type=float,
        default=1.0,
        help="Confidence/STD of the surface elevation change"
    )
    parser.add_argument(
        "--tcal_vol_std",
        type=float,
        default=1000.0,
        help="Confidence/STD of the volume estimates from volume-area scaling",
    )
    parser.add_argument(
        "--tcal_velsurfobs_thr",
        type=float,
        default=0.0,
        help="Threshold for the surface ice velocities as input data for the optimization, anything below this value will be ignored",
    )

    parser.add_argument(
        "--tcal_divflux_method",
        type=str,
        default="upwind",
        help="Compute the divergence of the flux using the upwind or centered method",
    )
    parser.add_argument(
        "--tcal_force_zero_sum_divflux",
        type=str2bool,
        default="False",
        help="Add a penalty to the cost function to force the sum of the divergence of the flux to be zero",
    )
    parser.add_argument(
        "--tcal_scaling_thk",
        type=float,
        default=2.0,
        help="Scaling factor for the ice thickness in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--tcal_scaling_topg",
        type=float,
        default=0.5,
        help="Scaling factor for the bed topography in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--tcal_scaling_usurf",
        type=float,
        default=0.5,
        help="Scaling factor for the ice thickness in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--tcal_scaling_slidingco",
        type=float,
        default=0.0001,
        help="Scaling factor for the slidingco in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--tcal_scaling_arrhenius",
        type=float,
        default=0.1,
        help="Scaling factor for the Arrhenius in the optimization, serve to adjust step-size of each controls relative to each other",
    )

    parser.add_argument(
       "--tcal_to_regularize",
       type=str,
       default='topg',
       help="Field to regularize : topg or thk",
    )
    parser.add_argument(
       "--tcal_include_low_speed_term",
       type=str2bool,
       default=False,
       help="opti_include_low_speed_term",
    ) 

    parser.add_argument(
        "--tcal_vars_to_save",
        type=list,
        default=[
            "usurf",
            "thk",
            "slidingco",
            "velsurf_mag",
            "velsurfobs_mag",
            "divflux",
            "icemask",
        ],
        help="List of variables to be recorded in the ncdef file",
    )
    parser.add_argument(
        "--tcal_save_result_in_ncdf",
        type=str,
        default="geology-tcal.nc",
        help="Geology input file",
    )
    parser.add_argument(
        "--tcal_save_iterat_in_ncdf",
        type=str2bool,
        default=False,
        help="write_ncdf_optimize",
    )






    parser.add_argument(
        "--tcal_plot2d_live",
        type=str2bool,
        default=False,
        help="plot2d_live_inversion",
    )
    parser.add_argument(
        "--tcal_plot2d",
        type=str2bool,
        default=False,
        help="plot 2d inversion",
    )
    parser.add_argument(
        "--tcal_editor_plot2d",
        type=str,
        default="vs",
        help="optimized for VS code (vs) or spyder (sp) for live plot",
    )

    parser.add_argument(
        "--tcal_infer_params",
        type=str2bool,
        default=False,
        help="infer slidingco and convexity weight from velocity observations",
    )
    parser.add_argument(
        "--tcal_tidewater_glacier",
        type=str2bool,
        default=False,
        help="Is the glacier you're trying to infer parameters for a tidewater type?",
    )
    # parser.add_argument(
    #     "--fix_opti_normalization_issue",
    #     type=str2bool,
    #     default=False,
    #     help="formerly, the oce was mixing reduce_mean and l2_loss leadinf to dependence to the resolution of the grid",
    # )
    