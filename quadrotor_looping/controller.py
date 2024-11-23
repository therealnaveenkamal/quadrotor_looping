import numpy as np
from quadrotor_looping.sqp_solver import sqp_problem
from quadrotor_looping.trajectory import compute_desired_states

# Initialize lists for plotting
plot_x = []
plot_u1 = []
plot_u2 = []
plot_t = []

n = 50 # THIS IS THE MINI HORIZON TO SOLVE SQP PROBLEM IN MPC

def controller(x, t, N):
    """
    Compute control inputs for the quadrotor using SQP.
    """
    xbar_dims = n*8
    horizon_length = xbar_dims // 8
    xbar = np.zeros(xbar_dims)
    xbar[:6] = x
    print(f"Time step: {t:.2f}")

    xk = sqp_problem(
        x_initial=xbar,
        desired_states=compute_desired_states(N=N, horizon_length=horizon_length),
        N=horizon_length,
        x_init=x,
        timestep=t,
        verbosity=False,
        max_iterations=10,
        convergence_tolerance=0.01,
    )

    u1, u2 = xk[6], xk[7]
    print(f"Control Inputs - u1: {u1:.4f}, u2: {u2:.4f}")

    plot_t.append(t)
    plot_x.append(x)
    plot_u1.append(u1)
    plot_u2.append(u2)

    return np.array([u1, u2])
