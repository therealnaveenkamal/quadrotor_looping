import argparse
from quadrotor_looping.controller import controller
from quadrotor_looping.controller_sqp import sqp_call
from quadrotor_looping.quadrotor import simulate, animate_robot
from qpsolvers import available_solvers
import numpy as np

def main(args):
    if(not args.sqp):
        print("Starting MPC")
        x_init = np.array([float(i) for i in args.x_init])  # Initial state of the quadrotor
        horizon_length = args.horizon_length  # Simulation horizon

        # Simulate quadrotor dynamics
        t, state, u = simulate(
            z0=x_init,
            controller=controller,
            horizon_length=horizon_length,
            disturbance=args.disturbance,
        )

        # Animate the simulation results
        animate_robot(state, u)
    else:
        print("Starting SQP")
        state, u = sqp_call(x=args.x_init, t=0, N=args.horizon_length)
        animate_robot(state.T, u.T)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadrotor Simulation")
    parser.add_argument(
        "--sqp",
        action="store_true",
        help="Add this argument to use SQP parameter",
    )
    parser.add_argument(
        "--x_init",
        nargs=6,
        type=float,
        default=[0, 0, 0, 0, 0, 0],
        help="Initial state of the quadrotor as a list of 6 floats (e.g., --x_init 0 0 0 0 0 0)",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=100,
        help="Simulation horizon length (number of time steps).",
    )
    parser.add_argument(
        "--disturbance",
        action="store_true",
        help="Enable disturbances in the simulation.",
    )
    args = parser.parse_args()
    main(args)
