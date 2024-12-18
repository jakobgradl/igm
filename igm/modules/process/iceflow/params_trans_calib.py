from igm.modules.utils import *

def params_tcal(parser):

    parser.add_argument(
        "--tcal_times",
        type=list,
        default=[2099, 2100],
        help="List of timesteps (type integer) used in transient calibration",
    )
    parser.add_argument(
        "--tcal_control_const",
        type=list,
        default=["topg", "slidingco", "arrhenius"],
        help="List of control parameters that are CONSTANT over time"
    )
    parser.add_argument(
        "--tcal_control_trans",
        type=list,
        default=["usurf"],
        help="List of control parameters that are NOT CONSTANT over time"
    )
    parser.add_argument(
        "--tcal_cost_const",
        type=list,
        default=["thk", "icemask"],
        help="List of cost parameters for CONSTANT control variables"
    )
    parser.add_argument(
        "--tcal_cost_trans",
        type=list,
        default=["usurf", "velsurf"],
        help="List of cost parameters for TRANSIENT control variables"
    )
    parser.add_argument(
        "--tcal_input_file",
        type=str,
        default="input.nc",
        help="NetCDF input data file",
    )
    parser.add_argument(
        "--tcal_step_size",
        type=float,
        default=1.0,
        help="Step size for optimization"
    )
    parser.add_argument(
        "--tcal_step_size_decay",
        type=float,
        default=0.9,
        help="Step size decay for optimization"
    )
    parser.add_argument(
        "--tcal-nbitmax",
        type=int,
        default=1000,
        help="Maximum number of iterations in the transient optimization"
    )
    parser.add_argument(
        "--tcal_scaling_topg",
        type=float,
        default=1.0,
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