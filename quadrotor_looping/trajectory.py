import numpy as np

def compute_desired_states(N=100, horizon_length=50):
    """
    Computes desired state trajectories for looping motion.

    Parameters:
        N (int): Number of time steps.
        horizon_length (int): Mini-Horizon SQP problems in Model Predictive Control

    Returns:
        numpy.ndarray: Array of desired states of size (N, state_dim).
    """
    desired_states = []

    for i in range(N+horizon_length+1):
        if i <= N/5:
            desired_states.append([0, 0, 0, 0, (2*np.pi/N)*i, 0])
        elif N/5 < i <= 2*(N/5):
            desired_states.append([2, 0, 1, 0, (2*np.pi/N)*i, 0])
        elif 2*(N/5) < i <= 3*(N/5):
            desired_states.append([0, 0, 2, 0, (2*np.pi/N)*i, 0])
        elif 3*(N/5) < i <= 4*(N/5):
            desired_states.append([-1.5, 0, 1, 0, (2*np.pi/N)*i, 0])
        else:
            desired_states.append([0, 0, 0, 0, (2*np.pi/N)*i, 0])

    return np.array(desired_states)
