import numpy as np
from quadrotor_looping.trajectory import compute_desired_states
from quadrotor_looping.utils import compute_A, compute_b, compute_inequality_G, compute_inequality_g, compute_hessian_cost, compute_gradient_cost, perform_line_search, compute_constraint_violation, compute_inequality_violation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import IPython
import math

import quadrotor_looping.quadrotor as quadrotor
from qpsolvers import Problem, solve_problem
from scipy.sparse import csr_matrix
import pdb
import numpy as np

def sqp_problem(x_initial, desired_states, N, max_iterations=100, convergence_tolerance=1e-5, x_init=np.zeros(quadrotor.DIM_STATE), verbosity=True, timestep=0):
    """
    Solves a Sequential Quadratic Programming (SQP) problem using quadratic programming solvers.
    
    Parameters:
        x_initial (np.ndarray): Initial guess for state and control variables.
        desired_states (np.ndarray): Array of desired states over the planning horizon.
        N (int): Number of time steps in the planning horizon.
        max_iterations (int): Maximum number of iterations for the SQP algorithm.
        convergence_tolerance (float): Tolerance for constraint violation convergence.
        x_init (np.ndarray): Current State
        verbosity (bool): If True, prints detailed logs during the optimization process.
        timestep (int): Compute controls from current timestep to the goal

    Returns:
        np.ndarray: Optimal state and control trajectory.
        np.ndarray: Updated Lagrange multipliers.
    """
    # Initialization
    x_current = x_initial
    f_best = np.inf
    c_best = np.inf

    # Set bounds for state and control variables
    num_vars = (quadrotor.DIM_STATE + quadrotor.DIM_CONTROL) * N  # 6 state variables + 2 control variables per time step

    # Main SQP iteration loop
    for iteration in range(max_iterations):
        if verbosity:
            print(f"\n=== Iteration {iteration + 1} ===")

        # Compute necessary matrices and gradients
        equality_G = compute_A(x_current, N)
        equality_g = compute_b(x_current, N, x_init)
        inequality_G = compute_inequality_G(N)
        inequality_g = compute_inequality_g(x_current, N)
        hessian_H = compute_hessian_cost(x_current, N)
        gradient_f = compute_gradient_cost(desired_states, x_current, N, timestep=timestep)

        if verbosity:
            print("Computed gradients, Hessians, and constraints.")

        # Formulate and solve the quadratic programming problem
        problem = Problem(hessian_H, gradient_f, inequality_G, inequality_g, equality_G, -equality_g)
        solution = solve_problem(problem=problem, solver="cvxopt")

        if solution.is_optimal(1e-8):
            if verbosity:
                print("Solution is optimal.")
        else:
            if verbosity:
                print("Solution is NOT optimal. Proceeding with current best guess.")

        # Update decision variables using linear search
        step_direction = solution.x
        x_current, f_best, c_best, alpha = perform_line_search(
            x_current, x_init, desired_states, timestep, step_direction, f_best, c_best, rho=0.5, alpha=1, N=N
        )

        if verbosity:
            print(f"Step size: {alpha:.5f}")
            print(f"Updated cost: {f_best:.5f}, Constraint violation: {c_best:.5f}")

        # Check constraint violations
        total_violation = compute_constraint_violation(x_current, N, x_init=x_init) + compute_inequality_violation(x_current, N)
        if verbosity:
            print(f"Total constraint violation: {total_violation:.8f}")

        if total_violation < convergence_tolerance:
            if verbosity:
                print(f"Converged! Total constraint violation: {total_violation:.8f}")
            break

    return x_current
