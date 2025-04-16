import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
mu = 398600  # Earth's gravitational parameter, km^3/s^2
J2 = 0.00108263
Re = 6378.1  # Earth's radius in km
g0 = 9.80665  # m/s^2

# === Helper Functions ===

def compute_perturbation_matrix(x):
    p, f, g, h, k, L = x
    q = 1 + f * np.cos(L) + g * np.sin(L)
    s = np.sqrt(1 + h**2 + k**2)
    B = (1 / q) * np.sqrt(p / mu) * np.array([
        [0, 2 * p, 0],
        [q * np.sin(L), (q + 1) * np.cos(L) + f, -g * (h * np.sin(L) - k * np.cos(L))],
        [-q * np.cos(L), (q + 1) * np.sin(L) + g, f * (h * np.sin(L) - k * np.cos(L))],
        [0, 0, s * np.cos(L) / 2],
        [0, 0, s * np.sin(L) / 2],
        [0, 0, (h * np.sin(L) - k * np.cos(L))]
    ])
    return B

def compute_J2_acceleration(x):
    p, f, g, h, k, L = x
    q = 1 + f * np.cos(L) + g * np.sin(L)
    r = p / q
    C1 = mu * J2 * Re**2 / r**4
    C2 = h * np.sin(L) - k * np.cos(L)
    C3 = (1 + h**2 + k**2)**2

    ar = -3 * C1 / 2 * (1 - 12 * C2**2 / C3)
    at = -12 * C1 * C2 * (h * np.cos(L) + k * np.sin(L)) / C3
    an = -6 * C1 * C2 * (1 - h**2 - k**2) / C3
    return np.array([ar, at, an])

def compute_A_vector(x):
    p, f, g, h, k, L = x
    q = 1 + f * np.cos(L) + g * np.sin(L)
    return np.array([0, 0, 0, 0, 0, np.sqrt(mu * p) * (q / p)**2])

def sigmoid_throttle(l_vec, x_vec):
    """
    Sigmoid throttle function
    
    Args:
    S: Switch function value
    rho: Continuation parameter
    
    Returns:
    Throttle value between 0 and 1
    """
    B=compute_perturbation_matrix(x_vec)
    S= np.linalg.norm(B.T@l_vec)
    return 0.5 * (1 + np.sign(S))

def eom_mee_with_perturbations(x ,l_vec, u_vec, thrust_mag, mass, isp):
    B = compute_perturbation_matrix(x)
    aJ2 = compute_J2_acceleration(x)
    A = compute_A_vector(x)
    c = isp * g0 / 1000  # km/s
    delta = sigmoid_throttle[L_vec,x]
    a_thrust = (thrust_mag * delta / mass) * u_vec
    dx = A + B @ aJ2 + B @ a_thrust
    dm = -thrust_mag * delta / c
    return dx, dm
