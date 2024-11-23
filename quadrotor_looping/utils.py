import quadrotor_looping.quadrotor as quadrotor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import IPython
import math

from qpsolvers import Problem, solve_problem
from scipy.sparse import csr_matrix
import pdb

def compute_A(x_bar, N):
    """
    Computes the first-order linear approximation of the quadrotor system dynamics.

    Parameters:
        x_bar (numpy.ndarray): The current state trajectory of size (8 * N, 1),
                               where N is the number of time steps.
        N (int): The number of time steps.

    Returns:
        numpy.ndarray: The constraint matrix A of size (6 * N, 8 * N), representing the
                       first-order approximation of the system dynamics.
    """
    # Quadrotor parameters
    g = quadrotor.GRAVITY_CONSTANT  # Gravitational constant
    dt = quadrotor.DT              # Time step duration
    r = quadrotor.LENGTH           # Distance from center to propellers
    m = quadrotor.MASS             # Mass of the quadrotor
    I = quadrotor.INERTIA          # Moment of inertia of the quadrotor

    # Initialize the constraint matrix
    A = np.zeros((6 * N, 8 * N))

    # Initialize the first row of the matrix (boundary condition)
    A[0, 0] = 1  # p_x(0)
    A[1, 1] = 1  # v_x(0)
    A[2, 2] = 1  # p_y(0)
    A[3, 3] = 1  # v_y(0)
    A[4, 4] = 1  # theta(0)
    A[5, 5] = 1  # omega(0)

    # Compute the dynamics for each time step
    for i in range(1, N):
        z = 8 * (i - 1)  # State offset for time step i

        # Extract necessary variables from the current state
        theta = x_bar[z + 4].item()  # Orientation
        u1 = x_bar[z + 6].item()    # Control input u1
        u2 = x_bar[z + 7].item()    # Control input u2

        # Precompute common terms for clarity
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        u_sum = u1 + u2

        # Dynamics coefficients
        a = (dt * cos_theta * u_sum) / m
        b = (dt * sin_theta) / m
        c = (dt * cos_theta) / m
        d = (dt * sin_theta * u_sum) / m
        angular_effect = (dt * r) / I

        # Populate rows in the matrix for this time step
        A[6 * i, 8 * (i - 1):8 * (i - 1) + 9] = [1, dt, 0, 0, 0, 0, 0, 0, -1]
        A[6 * i + 1, 8 * (i - 1):8 * (i - 1) + 10] = [0, 1, 0, 0, -a, 0, -b, -b, 0, -1]
        A[6 * i + 2, 8 * (i - 1):8 * (i - 1) + 11] = [0, 0, 1, dt, 0, 0, 0, 0, 0, 0, -1]
        A[6 * i + 3, 8 * (i - 1):8 * (i - 1) + 12] = [0, 0, 0, 1, -d, 0, c, c, 0, 0, 0, -1]
        A[6 * i + 4, 8 * (i - 1):8 * (i - 1) + 13] = [0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0, 0, -1]
        A[6 * i + 5, 8 * (i - 1):8 * (i - 1) + 14] = [0, 0, 0, 0, 0, 1, angular_effect, -angular_effect, 0, 0, 0, 0, 0, -1]

    return A


def compute_b(x_bar, N, x_init):
    """
    Computes the constraints of the quadrotor system dynamics.

    Parameters:
        x_bar (numpy.ndarray): The current state trajectory of size (8 * N, 1),
                               where N is the number of time steps.
        N (int): The number of time steps.
        x_init (numpy.ndarray): The current state

    Returns:
        numpy.ndarray: The constraints vector A of size (6 * N,), representing
                       the deviation from the system dynamics.
    """
    # Quadrotor parameters
    g = quadrotor.GRAVITY_CONSTANT  # Gravitational constant
    dt = quadrotor.DT              # Time step duration
    r = quadrotor.LENGTH           # Distance from center to propellers
    m = quadrotor.MASS             # Mass of the quadrotor
    I = quadrotor.INERTIA          # Moment of inertia of the quadrotor

    # Initialize the constraints vector
    A = np.zeros((6 * N,))

    # Set initial constraints to zero (boundary conditions)
    A[:6] =  x_init-x_bar[:6]

    # Compute constraints for each time step
    for i in range(1, N):
        z = 8 * (i - 1)  # State offset for time step i

        # Position and velocity constraints
        A[6 * i] = x_bar[z] + dt * x_bar[z + 1] - x_bar[z + 8]  # px constraint
        A[6 * i + 1] = (
            x_bar[z + 1]
            - (dt * math.sin(x_bar[z + 4]) * x_bar[z + 6]) / m
            - (dt * math.sin(x_bar[z + 4]) * x_bar[z + 7]) / m
            - x_bar[z + 9]
        )  # vx constraint

        A[6 * i + 2] = x_bar[z + 2] + dt * x_bar[z + 3] - x_bar[z + 10]  # py constraint
        A[6 * i + 3] = (
            x_bar[z + 3]
            - dt * g
            + (dt * math.cos(x_bar[z + 4]) * x_bar[z + 6]) / m
            + (dt * math.cos(x_bar[z + 4]) * x_bar[z + 7]) / m
            - x_bar[z + 11]
        )  # vy constraint

        # Orientation and angular velocity constraints
        A[6 * i + 4] = x_bar[z + 4] + dt * x_bar[z + 5] - x_bar[z + 12]  # theta constraint
        A[6 * i + 5] = (
            x_bar[z + 5]
            + (dt * r * x_bar[z + 6]) / I
            - (dt * r * x_bar[z + 7]) / I
            - x_bar[z + 13]
        )  # omega constraint

    return A


def compute_inequality_G(N):
    """
    Computes the inequality matrix G for control bounds in a convex optimization problem.

    Parameters:
        N (int): The number of time steps.

    Returns:
        numpy.ndarray: A matrix of size (4 * N, 8 * N) representing inequality constraints.
    """
    # Initialize the G matrix with zeros
    G = np.zeros((4 * N, 8 * N))  # 4 rows per time step for 2 bounds on each control input

    # Populate G to enforce bounds on u1 and u2
    for i in range(N):
        G[4 * i][8 * i + 6] = 1    # +1 for u1 upper bound
        G[4 * i + 1][8 * i + 6] = -1  # -1 for u1 lower bound
        G[4 * i + 2][8 * i + 7] = 1    # +1 for u2 upper bound
        G[4 * i + 3][8 * i + 7] = -1  # -1 for u2 lower bound

    return G

def compute_inequality_g(x_bar, N):
    """
    Computes the inequality vector g for control bounds in a convex optimization problem.

    Parameters:
        x_bar (numpy.ndarray): The current state trajectory of size (8 * N, 1),
                               where each state contains the control inputs u1 and u2.
        N (int): The number of time steps.

    Returns:
        numpy.ndarray: A vector of size (4 * N,) representing inequality constraints.
    """
    # Initialize the g vector with zeros
    g = np.zeros((4 * N,))  # 4 constraints per time step (2 bounds each for u1 and u2)

    # Populate g to enforce the bounds 0 <= u1, u2 <= 10
    z = 0
    for i in range(N):
        g[4 * i]     = 10 - x_bar[z + 6].item()  # u1 upper bound
        g[4 * i + 1] = x_bar[z + 6].item()       # u1 lower bound
        g[4 * i + 2] = 10 - x_bar[z + 7].item()  # u2 upper bound
        g[4 * i + 3] = x_bar[z + 7].item()       # u2 lower bound
        z += 8  # Move to the next state in x_bar

    return g


def compute_gradient_cost(x_desired, x_current, N, timestep):
    """
    Computes the gradient of the cost function: P * x + g.

    Parameters:
        x_desired (numpy.ndarray): Desired state trajectory of size (N, state_dim).
        x_current (numpy.ndarray): Current state trajectory of size (8 * N, 1).
        N (int): Number of time steps.
        timestep (int): Current timestep

    Returns:
        numpy.ndarray: The gradient vector of the cost function.
    """
    hessian = compute_hessian_cost(x_current, N)
    Q = np.diag(get_Q())
    gradient_list = []

    for i in range(timestep, timestep + N):
        temp_gradient = -1 * (x_desired[i].T @ Q)
        temp_gradient = np.append(temp_gradient, [0, 0])  # Append zeros for control terms
        gradient_list.append(temp_gradient)

    g_vector = np.concatenate(gradient_list)
    return hessian @ x_current + g_vector


def compute_hessian_cost(x_current, N):
    """
    Computes the Hessian matrix of the cost function: P.

    Parameters:
        x_current (numpy.ndarray): Current state trajectory of size (8 * N, 1).
        N (int): Number of time steps.

    Returns:
        numpy.ndarray: The Hessian matrix of size (8 * N, 8 * N).
    """
    Q = get_Q()
    R = get_R()
    hessian = np.zeros((x_current.shape[0], x_current.shape[0]))

    for i in range(N):
        hessian[8 * i + 0, 8 * i + 0] = Q[0]
        hessian[8 * i + 1, 8 * i + 1] = Q[1]
        hessian[8 * i + 2, 8 * i + 2] = Q[2]
        hessian[8 * i + 3, 8 * i + 3] = Q[3]
        hessian[8 * i + 4, 8 * i + 4] = Q[4]
        hessian[8 * i + 5, 8 * i + 5] = Q[5]
        hessian[8 * i + 6, 8 * i + 6] = R[0]
        hessian[8 * i + 7, 8 * i + 7] = R[1]

    return hessian


def compute_cost_function(x_desired, x_current, N, timestep):
    """
    Computes the cost function: (1/2) * x.T * P * x + g.T * x.

    Parameters:
        x_desired (numpy.ndarray): Desired state trajectory of size (N, state_dim).
        x_current (numpy.ndarray): Current state trajectory of size (8 * N, 1).
        N (int): Number of time steps.
        timestep (int): Current timestep

    Returns:
        float: The cost function value.
    """
    hessian = compute_hessian_cost(x_current, N)
    Q = np.diag(get_Q())
    gradient_list = []

    for i in range(timestep, timestep+N):
        temp_gradient = -1 * (x_desired[i].T @ Q)
        temp_gradient = np.append(temp_gradient, [0, 0])  # Append zeros for control terms
        gradient_list.append(temp_gradient)

    g_vector = np.concatenate(gradient_list)
    return 0.5 * (x_current.T @ hessian @ x_current) + g_vector.T @ x_current



def compute_constraint_violation(x_current, N, x_init):
    """
    Computes the total violation of equality constraints g(x) = 0.

    Parameters:
        x_current (numpy.ndarray): Current state trajectory of size (8 * N, 1).
        N (int): Number of time steps.
        x_init (numpy.ndarray): Current state.

    Returns:
        float: Total violation of equality constraints.
    """
    constraints = compute_b(x_current, N, x_init)
    return np.sum(np.abs(constraints))


def compute_inequality_violation(x_current, N):
    """
    Computes the total violation of inequality bounds (0 <= u1, u2 <= 10).

    Parameters:
        x_current (numpy.ndarray): Current state trajectory of size (8 * N, 1).
        N (int): Number of time steps.

    Returns:
        float: Total violation of inequality bounds.
    """
    violation = 0

    for i in range(N):
        u1 = x_current[8 * i + 6]
        u2 = x_current[8 * i + 7]

        if u1 > 10:
            violation += abs(u1 - 10)
        elif u1 < 0:
            violation += abs(u1)

        if u2 > 10:
            violation += abs(u2 - 10)
        elif u2 < 0:
            violation += abs(u2)

    return violation


def perform_line_search(x_current, x_init, desired_states, timestep, search_direction, cost_best, constraint_best, 
                        rho=0.5, alpha=1.0, N=100):
    """
    Performs line search to determine step size based on cost improvement or constraint satisfaction.

    Parameters:
        x_current (numpy.ndarray): Current state trajectory.
        x_init (numpy.ndarray): Current state
        desired_states (np.ndarray): Array of desired states over the planning horizon.
        timestep (int): Current timestep
        search_direction (numpy.ndarray): Search direction vector.
        cost_best (float): Best cost function value.
        constraint_best (float): Best constraint violation value.
        rho (float): Reduction factor for step size.
        alpha (float): Initial step size.
        N (int): Number of time steps.

    Returns:
        tuple: Updated state, new cost, new constraint violation, and step size.
    """
    while True:
        x_new = x_current + alpha * search_direction
        cost_new = compute_cost_function(desired_states, x_new, N, timestep)
        constraint_new = (compute_constraint_violation(x_new, N, x_init=x_init) + 
                          compute_inequality_violation(x_new, N))

        if cost_new < cost_best or constraint_new < constraint_best:
            return x_new, cost_new, constraint_new, alpha
        else:
            alpha *= rho
            if alpha < 1e-10:
                print("Step size alpha is too small.")
                break

    return x_current, cost_best, constraint_best, alpha


def get_Q():
  Q = np.array([50, 1, 50, 1, 50.35, 1])
  return Q

def get_R():
  R = np.array([1, 1])
  return R

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
