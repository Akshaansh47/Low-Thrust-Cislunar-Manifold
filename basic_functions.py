import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify

# Constants
mu = 398600  # Earth's gravitational parameter, km^3/s^2
J2 = 0.00108263
Re = 6378.1  # Earth's radius in km
g0 = 9.80665  # m/s^2

# === Helper Functions ===

def derive_costate_equations():
    # Define symbolic variables
    p, f, g, h, k, L, m = sp.symbols('p f g h k L m')
    l_p, l_f, l_g, l_h, l_k, l_L, l_m = sp.symbols('l_p l_f l_g l_h l_k l_L l_m')
    u_r, u_t, u_n = sp.symbols('u_r u_t u_n')
    a_r, a_t, a_n = sp.symbols('a_r a_t a_n')
    delta, T, c, mu = sp.symbols('delta T c mu')    

    # State and costate vectors
    x = sp.Matrix([p, f, g, h, k, L])
    lam = sp.Matrix([l_p, l_f, l_g, l_h, l_k, l_L])
    u = sp.Matrix([u_r, u_t, u_n])
    a = sp.Matrix([a_r, a_t, a_n])
    

    q = 1 + f * sp.cos(L) + g * sp.sin(L)
    s = sp.sqrt(1 + h**2 + k**2)

    A = sp.Matrix([0, 0, 0, 0, 0, sp.sqrt(mu * p) * (q / p)**2])    

    B = (1 / q) * sp.sqrt(p / mu) * sp.array([
        [0, 2 * p, 0],
        [q * sp.sin(L), (q + 1) * sp.cos(L) + f, -g * (h * sp.sin(L) - k * sp.cos(L))],
        [-q * sp.cos(L), (q + 1) * sp.sin(L) + g, f * (h * sp.sin(L) - k * sp.cos(L))],
        [0, 0, s * sp.cos(L) / 2],
        [0, 0, s * sp.sin(L) / 2],
        [0, 0, (h * sp.sin(L) - k * sp.cos(L))]
    ])

    H = 1 + lam.dot(A + B @ a + delta * (T / m) * B @ u ) - l_m * delta * (T / c)

    # Costate equations: λ̇ = -∂H/∂x
    costate_odes = -H.jacobian(x)

    #Add costate equation for l_m: l̇_m = -∂H/∂m
    l_m_dot = -sp.diff(H, m)
    
    # Combine all costate equations
    full_costate_odes = sp.Matrix([costate_odes, l_m_dot])


    return x, lam, m, l_m, u, a, full_costate_odes


def get_costate_dynamics_func():
    x, lam, m, l_m, u, a, costate_odes = derive_costate_equations()
    
    # Flatten input variables
    state_vars = x
    costate_vars = lam
    control_vars = u
    acc_var = a
    mass = m
    l_mass = l_m
    delta = sp.Symbol('delta')
    mu = sp.Symbol('mu')  

    # Create a numerical function
    ode_func = lambdify((state_vars, mass, costate_vars, l_m, control_vars, acc_var, delta, mu), costate_odes, 'numpy')
    return ode_func




def compute_perturbation_matrix(x , mu):
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

def compute_J2_acceleration(x , mu):
    p, f, g, h, k, L = x
    q = 1 + f * np.cos(L) + g * np.sin(L)
    r_vec , v_vec = mee_to_rv(p, f, g, h, k, L, mu)
    r = np.linalg.norm(r_vec)
    C1 = mu * J2 * Re**2 / r**4
    C2 = h * np.sin(L) - k * np.cos(L)
    C3 = (1 + h**2 + k**2)**2

    ar = -3 * C1 / 2 * (1 - 12 * C2**2 / C3)
    at = -12 * C1 * C2 * (h * np.cos(L) + k * np.sin(L)) / C3
    an = -6 * C1 * C2 * (1 - h**2 - k**2) / C3
    return np.array([ar, at, an])

def compute_A_vector(x , mu):
    p, f, g, h, k, L = x
    q = 1 + f * np.cos(L) + g * np.sin(L)
    return np.array([0, 0, 0, 0, 0, np.sqrt(mu * p) * (q / p)**2])

def switch_ustar(l_vec, x_vec):
    """
    Sigmoid throttle function
    
    Args:
    S: Switch function value
    rho: Continuation parameter
    
    Returns:
    Throttle value between 0 and 1
    """
    B=compute_perturbation_matrix(x_vec , mu)
    norm= np.linalg.norm(B.T@l_vec)

    #U_star
    u_star=-(B.T@l_vec)/norm
    #switch function
    switch=norm-1
    return u_star,switch

def mee_to_rv(p, f, g, h, k, L, mu):
    """
    Converts Modified Equinoctial Elements to position and velocity vectors.
    
    Parameters:
    - p: semi-latus rectum
    - f, g, h, k: equinoctial elements
    - L: true longitude (in radians)
    - mu: gravitational parameter (default is for Earth in km^3/s^2)
    
    Returns:
    - r_vec: position vector (3,)
    - v_vec: velocity vector (3,)
    """
    cosL = np.cos(L)
    sinL = np.sin(L)

    # Intermediate quantities
    alpha_sq = h**2 - k**2
    s_sq = 1 + h**2 + k**2
    w = 1 + f * cosL + g * sinL
    r = p / w
    sqrt_mu_over_p = np.sqrt(mu / p)
    
    # Position vector r
    r_x = r / s_sq * (cosL + alpha_sq * cosL + 2 * h * k * sinL)
    r_y = r / s_sq * (sinL - alpha_sq * sinL + 2 * h * k * cosL)
    r_z = 2 * r / s_sq * (h * sinL - k * cosL)
    r_vec = np.array([r_x, r_y, r_z])
    
    # Velocity vector v
    v_x = -sqrt_mu_over_p / s_sq * (sinL + alpha_sq * sinL - 2 * h * k * cosL + g - 2 * f * h * k + alpha_sq * g)
    v_y = -sqrt_mu_over_p / s_sq * (-cosL + alpha_sq * cosL + 2 * h * k * sinL - f + 2 * g * h * k + alpha_sq * f)
    v_z = 2 * sqrt_mu_over_p / s_sq * (h * cosL + k * sinL + f * h + g * k)
    v_vec = np.array([v_x, v_y, v_z])
    
    return r_vec, v_vec


def rotation_mat(r , v):
    i_r = r/(np.linalg.norm(r))
    h = np.cross(r , v)
    i_t = (np.cross(h, r)) / (np.linalg.norm(np.cross(h, r)))
    i_n = (h) / (np.linalg.norm(h))
    Q = np.vstack([i_r, i_t, i_n])

    return Q


def third_body(r_target,r_perturbingbody,mu):
    
    rel_r = r_target - r_perturbingbody
    rel_r_norm = np.linalg.norm(rel_r)
    q = r_target.T @ (r_target - 2 * r_perturbingbody) / (r_perturbingbody.T @ r_perturbingbody)
    F = q * (3 + 3 * q + q**2) / (1 + np.sqrt(1 + q)**3)
    t = -(mu / (rel_r_norm**3)) * (r_target + F * r_perturbingbody)
    return t


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
    B = compute_perturbation_matrix(x , mu)
    aJ2 = compute_J2_acceleration(x , mu)
    A = compute_A_vector(x , mu)
    c = isp * g0 / 1000  # km/s
    delta = sigmoid_throttle[l_vec,x]
    a_thrust = (thrust_mag * delta / mass) * u_vec
    dx = A + B @ aJ2 + B @ a_thrust
    dm = -thrust_mag * delta / c
    return dx, dm


