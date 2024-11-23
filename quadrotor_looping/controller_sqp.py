import numpy as np
import quadrotor_looping.quadrotor as quadrotor
from quadrotor_looping.sqp_solver import sqp_problem
from quadrotor_looping.trajectory import compute_desired_states

# Initialize data structures for plotting
plot_x = []
plot_u1 = []
plot_u2 = []
plot_t = []

def sqp_call(x, t, N):
    """
    Compute control inputs for the quadrotor using SQP.
    """
    xbar_dims = N*(quadrotor.DIM_STATE + quadrotor.DIM_CONTROL)
    horizon_length = N
    xbar = np.zeros(xbar_dims)
    xbar[:6] = x
    print(f"Time step: {t:.2f}")

    xk = sqp_problem(
        x_initial=xbar,
        desired_states=compute_desired_states(N=N, horizon_length=horizon_length),
        N=horizon_length,
        x_init=x,
        timestep=t,
        verbosity=True,
        max_iterations=100,
        convergence_tolerance=0.001,
    )

    x = []
    u = []
    for i in range(horizon_length):
        x.append(xk[8*i:8*i+6])
        u.append(xk[8*i+6:8*i+8])

    print(f"Execution Successful")

    return np.array(x), np.array(u)
