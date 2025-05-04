import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify
from skyfield.api import load
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


# === Constants ===
mu = 398600.435436096 # Earth's gravitational parameter, km^3/s^2
mu_sun = 132712440041.939  # Sun's gravitational parameter, km^3/s^2
mu_moon = 4902.8000661638  # Moon's gravitational parameter, km^3/s^2

thrust_mag = 0.235/1000  # Thrust magnitude in km/s^2
isp = 4155  # Specific impulse in seconds

# Earth's J2 coefficient (oblateness factor)
J2 = 0.00108263
Re = 6378.1366  # Earth's radius in km
g0 = 9.80665  # Standard gravity acceleration in m/s^2
c = isp * g0/1000 # Exhaust velocity in km/s (specific impulse * gravity)


# === Skyfield setup for celestial body positions ===
ts = load.timescale()
eph = load('de440s.bsp')  # NASA JPL ephemeris file for planetary positions
earth = eph['Earth']
sun = eph['Sun']
moon = eph['Moon']
ts = load.timescale()


# === Helper Functions ===

def Hamiltonian(x_vec, lam, l_m, m, a, T, c, mu, jd):
    """
    Calculate the Hamiltonian value for the optimal control problem

    Args:
        x_vec: Modified equinoctial elements state vector
        lam: Costates for modified equinoctial elements
        l_m: Costate for mass
        m: Current mass
        a: Perturbation acceleration vector
        T: Thrust magnitude
        c: Exhaust velocity
        mu: Gravitational parameter
        jd: Julian date for third body positions

    Returns:
        H: Hamiltonian value
    """
    r_target, v_target = mee_to_rv(x_vec, mu)

    # Get positions of celestial bodies
    r_perturbingbody_s = sun.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_m = moon.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_e = earth.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_srt = - r_perturbingbody_e + r_perturbingbody_s  # Sun position relative to Earth
    r_perturbingbody_mrt = r_perturbingbody_m - r_perturbingbody_e    # Moon position relative to Earth

    m_to_sun = r_perturbingbody_s - r_perturbingbody_m  # Vector from Moon to Sun

    r_perturbingbody_srtn = np.linalg.norm(r_perturbingbody_srt)  # Distance from Earth to Sun
    r_perturbingbody_mrtn = np.linalg.norm(m_to_sun)              # Distance from Moon to Sun

    unitvec_e_s = r_perturbingbody_srt/r_perturbingbody_srtn  # Unit vector from Earth to Sun
    unitvec_m_s = m_to_sun/r_perturbingbody_mrtn              # Unit vector from Moon to Sun


    # === Shadow Detection Logic ===
    # Calculate parallel and tangential components to determine if spacecraft is in shadow
    r_parallelm = np.dot((r_target-r_perturbingbody_mrt), unitvec_m_s)*unitvec_m_s
    r_parallele = np.dot(r_target, unitvec_e_s)*unitvec_e_s

    r_tangentialm = (r_target-r_perturbingbody_mrt)-r_parallelm  # Tangential component relative to Moon
    r_tangentiale = r_target-r_parallele                         # Tangential component relative to Earth

    # Check if spacecraft is in shadow of Earth or Moon
    if(np.dot((r_target-r_perturbingbody_mrt), unitvec_m_s) < 0 or np.dot(r_target, unitvec_e_s) < 0):
        if(np.linalg.norm(r_tangentiale) < 6378.1366 or np.linalg.norm(r_tangentialm) < 1737.4):
            # Spacecraft is in shadow, no thrust possible (e.g., for solar electric propulsion)
            delta = 0
        else:
            # Not in shadow, calculate throttle level
            delta = sigmoid_throttle(lam, x_vec, mu, l_m, m, c)
    else:
        # Not in shadow, calculate throttle level
        delta = sigmoid_throttle(lam, x_vec, mu, l_m, m, c)

    # Get optimal control direction
    u = switch_ustar(lam, x_vec)

    # Get dynamics matrices
    A = compute_A_vector(x_vec, mu)
    B = compute_perturbation_matrix(x_vec, mu)

    # Calculate Hamiltonian: H = 1 + λᵀ(A + Ba + δ(T/m)Bu) - λₘδ(T/c)
    H = 1 + np.dot(lam, (A + B @ a + delta * (T / m) * B @ u)) - l_m * delta * (T / c)
    return H

def derive_costate_equations():
    """
    Symbolic derivation of costate differential equations using sympy

    Returns:
        Symbolic variables and costate equations for computation
    """
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

    # Modified equinoctial elements intermediate terms
    q = 1 + f * sp.cos(L) + g * sp.sin(L)
    s = sp.sqrt(1 + h**2 + k**2)

    # Unperturbed dynamics (Keplerian motion)
    A = sp.Matrix([0, 0, 0, 0, 0, sp.sqrt(mu * p) * (q / p)**2])

    # Control influence matrix for Modified Equinoctial Elements
    B = (1 / q) * sp.sqrt(p / mu) * sp.Matrix([
        [0, 2 * p, 0],
        [q * sp.sin(L), (q + 1) * sp.cos(L) + f, -g * (h * sp.sin(L) - k * sp.cos(L))],
        [-q * sp.cos(L), (q + 1) * sp.sin(L) + g, f * (h * sp.sin(L) - k * sp.cos(L))],
        [0, 0, s * sp.cos(L) / 2],
        [0, 0, s * sp.sin(L) / 2],
        [0, 0, (h * sp.sin(L) - k * sp.cos(L))]
    ])

    # Hamiltonian definition for optimal control problem
    # H = 1 + λᵀ(A + Ba + δ(T/m)Bu) - λₘδ(T/c)
    H = 1 + lam.dot(A + B @ a + delta * (T / m) * B @ u) - l_m * delta * (T / c)

    # Costate equations: λ̇ = -∂H/∂x
    costate_odes = -sp.Matrix([sp.diff(H, var) for var in x])

    # Add costate equation for l_m: l̇_m = -∂H/∂m
    l_m_dot = -sp.diff(H, m)

    # Combine all costate equations
    full_costate_odes = sp.Matrix([costate_odes, l_m_dot])

    return x, lam, m, l_m, u, a, T, delta, c, mu, full_costate_odes

def get_costate_dynamics_func():
    """
    Convert symbolically derived costate equations to numerical functions

    Returns:
        Numerical function for costate dynamics
    """
    x, lam, m, l_m, u, a, T, delta, c, mu, costate_odes = derive_costate_equations()

    # Flatten input variables
    state_vars = x
    costate_vars = lam
    control_vars = u
    acc_var = a
    mass = m
    l_mass = l_m
    Thrust = T

    # Create a numerical function using sympy's lambdify
    ode_func = lambdify((state_vars, mass, costate_vars, l_mass, control_vars, acc_var, delta, mu, T, c), costate_odes, 'numpy')
    return ode_func


def compute_perturbation_matrix(x, mu):
    """
    Calculate the control influence matrix for Modified Equinoctial Elements

    Args:
        x: Modified equinoctial elements state
        mu: Gravitational parameter

    Returns:
        B: Control influence matrix
    """
    p, f, g, h, k, L = x
    q = 1 + f * np.cos(L) + g * np.sin(L)
    s = np.sqrt(1 + h**2 + k**2)

    # Control influence matrix (maps RTN accelerations to MEE state derivatives)
    B = (1 / q) * np.sqrt(p / mu) * np.array([
        [0, 2 * p, 0],
        [q * np.sin(L), (q + 1) * np.cos(L) + f, -g * (h * np.sin(L) - k * np.cos(L))],
        [-q * np.cos(L), (q + 1) * np.sin(L) + g, f * (h * np.sin(L) - k * np.cos(L))],
        [0, 0, s * np.cos(L) / 2],
        [0, 0, s * np.sin(L) / 2],
        [0, 0, (h * np.sin(L) - k * np.cos(L))]
    ])
    return B

def compute_J2_acceleration(x, mu):
    """
    Calculate J2 perturbation acceleration in RTN frame

    Args:
        x: Modified equinoctial elements state
        mu: Gravitational parameter

    Returns:
        J2 acceleration vector in RTN frame
    """
    p, f, g, h, k, L = x
    r_vec, v_vec = mee_to_rv(x, mu)
    r = np.linalg.norm(r_vec)
    C1 = (mu * J2 * Re**2) / r**4
    C2 = h * np.sin(L) - k * np.cos(L)
    C3 = (1 + h**2 + k**2)**2

    # J2 accelerations in radial, tangential, normal directions
    ar = (-3 * C1 / 2) * (1 - (12 * C2**2 / C3))
    at = -12 * C1 * C2 * (h * np.cos(L) + k * np.sin(L)) / C3
    an = -6 * C1 * C2 * (1 - h**2 - k**2) / C3
    return np.array([ar, at, an])

def compute_A_vector(x, mu):
    """
    Calculate the unperturbed dynamics vector for Modified Equinoctial Elements

    Args:
        x: Modified equinoctial elements state
        mu: Gravitational parameter

    Returns:
        A: Unperturbed dynamics vector
    """
    p, f, g, h, k, L = x
    q = 1 + f * np.cos(L) + g * np.sin(L)
    # Only true longitude rate is non-zero in unperturbed motion
    return np.array([0, 0, 0, 0, 0, np.sqrt(mu * p) * (q / p)**2])

def switch_ustar(l_vec, x_vec):
    """
    Calculate optimal control direction based on primer vector theory

    Args:
        l_vec: Costate vector
        x_vec: State vector

    Returns:
        Optimal control direction (unit vector)
    """
    B = compute_perturbation_matrix(x_vec, mu)
    norm = np.linalg.norm(B.T @ l_vec)

    # Optimal control direction is along the negative of the primer vector
    u_star = -(B.T @ l_vec)/norm

    return u_star

def mee_to_rv(x, mu):
    """
    Converts Modified Equinoctial Elements to position and velocity vectors.

    Parameters:
        x: Modified equinoctial elements [p, f, g, h, k, L]
            - p: semi-latus rectum
            - f, g: eccentricity vector components
            - h, k: inclination and ascending node parameters
            - L: true longitude (in radians)
        mu: gravitational parameter

    Returns:
        r_vec: position vector (3,)
        v_vec: velocity vector (3,)
    """
    p, f, g, h, k, L = x
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

def rotation_mat(r, v):
    """
    Create rotation matrix from inertial to RTN (Radial-Tangential-Normal) frame

    Args:
        r: Position vector
        v: Velocity vector

    Returns:
        Q: Rotation matrix from inertial to RTN
    """
    i_r = r/(np.linalg.norm(r))  # Radial unit vector
    h = np.cross(r, v)           # Angular momentum vector
    i_t = (np.cross(h, r)) / (np.linalg.norm(np.cross(h, r)))  # Tangential unit vector
    i_n = (h) / (np.linalg.norm(h))  # Normal unit vector
    Q = np.vstack([i_r, i_t, i_n])

    return Q

def third_body(r_target, r_perturbingbody, mu):
    """
    Calculate third-body perturbation acceleration

    Args:
        r_target: Position vector of spacecraft
        r_perturbingbody: Position vector of perturbing body (Moon or Sun)
        mu: Gravitational parameter of perturbing body

    Returns:
        Perturbation acceleration vector
    """
    rel_r = r_target - r_perturbingbody
    rel_r_norm = np.linalg.norm(rel_r)
    q = r_target.T @ (r_target - 2 * r_perturbingbody) / (r_perturbingbody.T @ r_perturbingbody)
    F = q * (3 + 3 * q + q**2) / (1 + np.sqrt(1 + q)**3)
    t = -(mu / (rel_r_norm**3)) * (r_target + F * r_perturbingbody)
    return t

def sigmoid_throttle(l_vec, x_vec, mu, l_m, m, c):
    """
    Calculate throttle level using switching function from optimal control theory

    Args:
        l_vec: Costate vector
        x_vec: State vector
        mu: Gravitational parameter
        l_m: Mass costate
        m: Current mass
        c: Exhaust velocity

    Returns:
        Throttle value (0 or 1)
    """
    B = compute_perturbation_matrix(x_vec, mu)
    # Switching function S = |B'λ| + λₘ(m/c)
    S = np.linalg.norm(B.T @ l_vec) + l_m * (m/c)

    # Bang-bang control (on/off) based on sign of switching function
    throttle = 0.5 * (1 + np.sign(S))

    return throttle

def eom_mee_with_perturbations(x, l_vec, u_vec, thrust_mag, m, isp, jd, mu, l_m):
    """
    Equations of motion for Modified Equinoctial Elements with all perturbations

    Args:
        x: State vector (MEE)
        l_vec: Costate vector
        u_vec: Control direction vector
        thrust_mag: Thrust magnitude
        m: Current mass
        isp: Specific impulse
        jd: Julian date for third body positions
        mu: Earth's gravitational parameter
        l_m: Mass costate

    Returns:
        dx: State derivatives
        dm: Mass derivative
        a: Total perturbation acceleration
    """
    B = compute_perturbation_matrix(x, mu)
    aJ2 = compute_J2_acceleration(x, mu)

    r_target, v_target = mee_to_rv(x, mu)

    # Get positions of celestial bodies
    r_perturbingbody_s = sun.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_m = moon.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_e = earth.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_srt = - r_perturbingbody_e + r_perturbingbody_s
    r_perturbingbody_mrt = r_perturbingbody_m - r_perturbingbody_e

    m_to_sun = r_perturbingbody_s - r_perturbingbody_m

    r_perturbingbody_srtn = np.linalg.norm(r_perturbingbody_srt)
    r_perturbingbody_mrtn = np.linalg.norm(m_to_sun)

    unitvec_e_s = r_perturbingbody_srt/r_perturbingbody_srtn
    unitvec_m_s = m_to_sun/r_perturbingbody_mrtn

    # === Shadow Detection Logic ===
    # Calculate parallel and tangential components to determine if spacecraft is in shadow
    r_parallelm = np.dot((r_target-r_perturbingbody_mrt), unitvec_m_s)*unitvec_m_s
    r_parallele = np.dot(r_target, unitvec_e_s)*unitvec_e_s

    r_tangentialm = (r_target-r_perturbingbody_mrt)-r_parallelm
    r_tangentiale = r_target-r_parallele

    # Check if spacecraft is in shadow of Earth or Moon
    if(np.dot((r_target-r_perturbingbody_mrt), unitvec_m_s) < 0 or np.dot(r_target, unitvec_e_s) < 0):
        if(np.linalg.norm(r_tangentiale) < 6378.1366 or np.linalg.norm(r_tangentialm) < 1737.4):
            # Spacecraft is in shadow, no thrust
            delta = 0
        else:
            # Not in shadow, calculate throttle level
            delta = sigmoid_throttle(l_vec, x, mu, l_m, m, c)
    else:
        # Not in shadow, calculate throttle level
        delta = sigmoid_throttle(l_vec, x, mu, l_m, m, c)

    # Calculate third-body perturbations
    t_m = third_body(r_target, r_perturbingbody_mrt, mu_moon)  # Moon perturbation
    t_s = third_body(r_target, r_perturbingbody_srt, mu_sun)   # Sun perturbation
    t = t_m + t_s  # Combined third-body perturbation

    # Rotate perturbations to RTN frame
    Q = rotation_mat(r_target, v_target)
    a_3b = Q.T @ t
    a = aJ2 + a_3b  # Total perturbation acceleration

    # Calculate state dynamics
    A = compute_A_vector(x, mu)
    a_thrust = (thrust_mag * delta / m) * u_vec  # Thrust acceleration
    dx = A + B @ a + B @ a_thrust  # Total state derivative

    # Mass rate of change
    dm = -thrust_mag * delta / c

    return dx, dm, a

def convert_cartesian_to_modified_equinoctial(state_cartesian, mu):
    """
    Convert Cartesian state (position and velocity) to Modified Equinoctial Elements

    Args:
        state_cartesian: Cartesian state vectors [r, v]
        mu: Gravitational parameter

    Returns:
        Modified equinoctial elements [p, f, g, h, k, L]
    """
    position = state_cartesian[0:3, :]
    velocity = state_cartesian[3:6, :]

    radius = np.linalg.norm(position, axis=0)
    h_vec = np.cross(position.T, velocity.T).T  # Angular momentum vector
    h_mag = np.linalg.norm(h_vec, axis=0)
    radial_velocity = np.sum(position * velocity, axis=0) / radius

    # Unit vectors
    r_hat = position / radius
    v_hat = (radius * velocity - radial_velocity * position) / h_mag
    h_hat = h_vec / h_mag

    # Eccentricity vector
    ecc_vec = np.cross(velocity.T, h_vec.T).T / mu - r_hat

    # Semi-latus rectum (p)
    p = h_mag ** 2 / mu

    # Retrograde factor elements (h and k)
    denom = 1 + h_hat[2, :]
    k = h_hat[0, :] / denom
    h = -h_hat[1, :] / denom

    # Equinoctial frame unit vectors
    denom_eq = 1 + k**2 + h**2
    f_hat = np.vstack([
        1 - k**2 + h**2,
        2 * k * h,
        -2 * k
    ]) / denom_eq

    g_hat = np.vstack([
        2 * k * h,
        1 + k**2 - h**2,
        2 * h
    ]) / denom_eq

    # Eccentricity elements (f and g)
    f_elem = np.sum(ecc_vec * f_hat, axis=0)
    g_elem = np.sum(ecc_vec * g_hat, axis=0)

    # True longitude (L)
    cosl = r_hat[0, :] + v_hat[1, :]
    sinl = r_hat[1, :] - v_hat[0, :]
    L = np.arctan2(sinl, cosl)

    return np.vstack([p, f_elem, g_elem, h, k, L])


def dynamics(t, vec, ode_func):
    """
    Combined state and costate dynamics for solving the optimal control problem

    Args:
        t: Time
        vec: Combined state and costate vector
        ode_func: Function for costate dynamics

    Returns:
        Derivatives of state and costate vectors
    """
    # Extract state and costate variables
    p, f, g, h, k, l, m = vec[:7]  # State
    l_p, l_f, l_g, l_h, l_k, l_l, l_m = vec[7:]  # Costates

    # Calculate Julian date from simulation time
    part_of_day = t / 86400
    jd = 2459783.4788888893 + part_of_day

    l_vec = np.array([l_p, l_f, l_g, l_h, l_k, l_l])
    x_vec = np.array([p, f, g, h, k, l])

    # Get optimal control direction
    u_opt = switch_ustar(l_vec, x_vec)

    r_target, v_target = mee_to_rv(x_vec, mu)

    # Get positions of celestial bodies
    r_perturbingbody_s = sun.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_m = moon.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_e = earth.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_srt = - r_perturbingbody_e + r_perturbingbody_s
    r_perturbingbody_mrt = r_perturbingbody_m - r_perturbingbody_e

    m_to_sun = r_perturbingbody_s - r_perturbingbody_m

    r_perturbingbody_srtn = np.linalg.norm(r_perturbingbody_srt)
    r_perturbingbody_mrtn = np.linalg.norm(m_to_sun)

    unitvec_e_s = r_perturbingbody_srt/r_perturbingbody_srtn
    unitvec_m_s = m_to_sun/r_perturbingbody_mrtn

    # === Shadow Detection Logic ===
    r_parallelm = np.dot((r_target-r_perturbingbody_mrt), unitvec_m_s)*unitvec_m_s
    r_parallele = np.dot(r_target, unitvec_e_s)*unitvec_e_s

    r_tangentialm = (r_target-r_perturbingbody_mrt)-r_parallelm
    r_tangentiale = r_target-r_parallele

    # Check if spacecraft is in shadow
    if(np.dot((r_target-r_perturbingbody_mrt), unitvec_m_s) < 0 or np.dot(r_target, unitvec_e_s) < 0):
        if(np.linalg.norm(r_tangentiale) < 6378.1366 or np.linalg.norm(r_tangentialm) < 1737.4):
            # In shadow, no thrust
            delta = 0
        else:
            # Not in shadow
            delta = sigmoid_throttle(l_vec, x_vec, mu, l_m, m, c)
    else:
        # Not in shadow
        delta = sigmoid_throttle(l_vec, x_vec, mu, l_m, m, c)

    # Calculate state derivatives
    dx, dm, acc = eom_mee_with_perturbations(x_vec, l_vec, u_opt, thrust_mag, m, isp, jd, mu, l_m)

    dot_p, dot_f, dot_g, dot_h, dot_k, dot_l = dx
    dot_m = dm

    # Calculate costate derivatives
    dlam = ode_func(x_vec, m, l_vec, l_m, u_opt, acc, delta, mu, thrust_mag, c)
    dlam = np.array(dlam).flatten()
    dot_l_p, dot_l_f, dot_l_g, dot_l_h, dot_l_k, dot_l_l, dot_l_m = dlam

    # Return combined state and costate derivatives
    return [dot_p, dot_f, dot_g, dot_h, dot_k, dot_l, dot_m, dot_l_p, dot_l_f, dot_l_g, dot_l_h, dot_l_k, dot_l_l, dot_l_m]


def single_shooting(para):
    #initial state
    p , f , g , h , k , l ,m = 42164,0,0,0,0,0,500
    # unpacking the guess
    l_p0, l_f0, l_g0, l_h0, l_k0, l_l0 = para[:6]
    l_m0 = para[6]
    t_seg = para[7]*86400 #converting time in sec
    #Final State
    p_f , f_f , g_f , h_f , k_f , l_f = 187889.321482990,0.387690133540649,-0.0243836062752043,0.0765422736430671,-0.0275226715429439, 26.2238381898900

    # Initial state for first segment
    initial_state_seg = [ p , f , g , h , k , l , m,
                          l_p0, l_f0, l_g0, l_h0, l_k0, l_l0, l_m0]

    #Making Equations for Costates
    ode_func = get_costate_dynamics_func()

    #Time segments
    t_span1 = np.linspace(-t_seg, 0, 1000)

    #solve
    sol = solve_ivp(lambda t, y: dynamics(t, y, ode_func),
                [-t_seg, 0],
                initial_state_seg,
                method='RK45',  # Better for stiff problems
                t_eval=t_span1,
                rtol=1e-8,
                atol=1e-8)


    #unpacking data at final time
    x_vec = [sol.y[0,-1], sol.y[1,-1], sol.y[2,-1], sol.y[3,-1], sol.y[4,-1], sol.y[5,-1] ]
    l_vec = [sol.y[7,-1], sol.y[8,-1], sol.y[9,-1], sol.y[10,-1], sol.y[11,-1], sol.y[12,-1] ]
    l_m = sol.y[13,-1]
    m = sol.y[6,-1]


    jd= 2459843.479016204
    #J2 acc
    aJ2 = compute_J2_acceleration(x_vec , mu)


    r_target , v_target = mee_to_rv(x_vec, mu)

    # Get positions of celestial bodies
    r_perturbingbody_s = sun.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_m = moon.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_e = earth.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_srt = - r_perturbingbody_e + r_perturbingbody_s
    r_perturbingbody_mrt = r_perturbingbody_m - r_perturbingbody_e

    # Calculate third-body perturbations
    t_m = third_body(r_target,r_perturbingbody_mrt,mu_moon) #moon
    t_s = third_body(r_target,r_perturbingbody_srt,mu_sun) #sun
    # Combined third-body perturbation
    t= t_m +t_s

    # Rotate perturbations to RTN frame
    Q = rotation_mat(r_target , v_target)
    a_3b = Q.T @ t

    #total acc
    acc = aJ2 + a_3b

    #Hamiltonian
    Haml = Hamiltonian(x_vec, l_vec, l_m, m, acc, thrust_mag, c, mu,jd)

    #constraints
    constraints = [
        # final state  constraints (6 equations)
        (sol.y[0,-1] - p_f)/Re,
        (sol.y[1,-1] - f_f),
        (sol.y[2,-1] - g_f),
        (sol.y[3,-1] - h_f),
        (sol.y[4,-1] - k_f),
        (((sol.y[5,-1] - l_f) + np.pi)% (2 * np.pi) - np.pi),
        #hamiltonian = 0
        Haml,

        #lamda_m(tf) = 0
        sol.y[13,-1]

    ]

    return constraints

#initial state
initialState = np.array([-70.7266912380230,-563.081632198072,-103.175601217641,-739.274612104689,365.891880171321,-13.5870464342646, 7000 , 55.83]);

import time
start_time = time.time()
# Solve using fsolve
solution, info, ier, msg = fsolve(single_shooting, initialState, full_output=True)

end_time = time.time()
print("Runtime:", end_time - start_time, "seconds")

if ier == 1:
    print("Converged successfully!")
else:
    print(f"Did not converge. Message: {msg}")

def dynamics_1(t, vec, ode_func):
    """
    Computes state and costate derivatives, and returns:
    - derivatives
    - throttle (delta)
    - eclipse status (is_in_shadow)

    Parameters:
        t : float
            Time in seconds
        vec : ndarray
            Full state + costate vector (14,)
        ode_func : callable
            Function for computing costate derivatives

    Returns:
        derivatives : list
            Time derivatives of state and costates
        delta : float
            Throttle value [0 to 1]
        is_in_shadow : bool
            True if spacecraft is in Earth's or Moon's shadow
    """
    # Unpack state and costate
    p, f, g, h, k, l, m = vec[:7]
    l_p, l_f, l_g, l_h, l_k, l_l, l_m = vec[7:]

    # Compute Julian Date
    part_of_day = t / 86400
    jd = 2459783.4788888893 + part_of_day

    # State and costate vectors
    x_vec = np.array([p, f, g, h, k, l])
    l_vec = np.array([l_p, l_f, l_g, l_h, l_k, l_l])

    # Optimal thrust direction
    u_opt = switch_ustar(l_vec, x_vec)

    # Position and velocity vectors
    r_target, v_target = mee_to_rv(x_vec, mu)

    # Third body positions (inertial)
    r_perturbingbody_s = sun.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_m = moon.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_e = earth.at(ts.ut1_jd(jd)).position.km

    # Relative position vectors
    r_perturbingbody_srt = r_perturbingbody_s - r_perturbingbody_e
    r_perturbingbody_mrt = r_perturbingbody_m - r_perturbingbody_e
    m_to_sun = r_perturbingbody_s - r_perturbingbody_m

    # Normalize directions
    unitvec_e_s = r_perturbingbody_srt / np.linalg.norm(r_perturbingbody_srt)
    unitvec_m_s = m_to_sun / np.linalg.norm(m_to_sun)

    # Project position onto shadow axes
    r_parallelm = np.dot((r_target - r_perturbingbody_mrt), unitvec_m_s) * unitvec_m_s
    r_parallele = np.dot(r_target, unitvec_e_s) * unitvec_e_s
    r_tangentialm = (r_target - r_perturbingbody_mrt) - r_parallelm
    r_tangentiale = r_target - r_parallele

    is_in_shadow = False

    # Eclipse condition: geometric and umbral check
    if (np.dot((r_target - r_perturbingbody_mrt), unitvec_m_s) < 0 or np.dot(r_target, unitvec_e_s) < 0):
        if (np.linalg.norm(r_tangentiale) < 6378.1366 or np.linalg.norm(r_tangentialm) < 1737.4):
            delta = 0
            is_in_shadow = True
        else:
            delta = sigmoid_throttle(l_vec, x_vec, mu, l_m, c, m)
    else:
        delta = sigmoid_throttle(l_vec, x_vec, mu, l_m, c, m)

    # Compute full state dynamics
    dx, dm, acc = eom_mee_with_perturbations(x_vec, l_vec, u_opt, thrust_mag, m, isp, jd, mu, l_m)
    dot_p, dot_f, dot_g, dot_h, dot_k, dot_l = dx
    dot_m = dm

    # Costate dynamics
    dlam = ode_func(x_vec, m, l_vec, l_m, u_opt, acc, delta, mu, thrust_mag, c)
    dlam = np.array(dlam).flatten()
    dot_l_p, dot_l_f, dot_l_g, dot_l_h, dot_l_k, dot_l_l, dot_l_m = dlam

    # Return everything
    derivatives = [
        dot_p, dot_f, dot_g, dot_h, dot_k, dot_l, dot_m,
        dot_l_p, dot_l_f, dot_l_g, dot_l_h, dot_l_k, dot_l_l, dot_l_m
    ]
    return derivatives, delta, is_in_shadow


def single_shooting_1(para):
    """
    Performs a single shooting integration and returns full trajectory with delta tracking.

    Parameters:
        para : list or ndarray (8,)
            Initial costate values + segment duration [l_p0, ..., l_l0, l_m0, duration_days]

    Returns:
        x_vec : list of ndarrays
            Time histories of [p, f, g, h, k, L]
        delta_values : list
            Throttle (delta) values at each time step
        shadow_flags : list
            Eclipse boolean flags at each time step
        sol.t : ndarray
            Time vector (s)
        l_m : ndarray
            Lambda mass time history
        m : ndarray
            Mass time history
        l_vec : list of ndarrays
            Time histories of costates [l_p, ..., l_l]
    """
    # Initial state values
    p, f, g, h, k, l, m = 42164, 0, 0, 0, 0, 0, 500

    # Unpack parameters
    l_p0, l_f0, l_g0, l_h0, l_k0, l_l0 = para[:6]
    l_m0 = para[6]
    t_seg = para[7] * 86400  # Convert days to seconds

    # Target final state is not used here — just forward integration
    initial_state_seg = [p, f, g, h, k, l, m,
                         l_p0, l_f0, l_g0, l_h0, l_k0, l_l0, l_m0]

    # Costate dynamics function
    ode_func = get_costate_dynamics_func()

    # Lists to track delta and eclipse flags
    delta_values = []
    shadow_flags = []

    # Wrapper to record delta and eclipse
    def dynamics_wrapper(t, y):
        derivatives, delta, is_in_shadow = dynamics_1(t, y, ode_func)
        delta_values.append(delta)
        shadow_flags.append(is_in_shadow)
        return derivatives

    # Time vector (backward integration)
    t_span1 = np.linspace(-t_seg, 0, 3752)

    # Run integration
    sol = solve_ivp(
        dynamics_wrapper,
        [-t_seg, 0],
        initial_state_seg,
        method='RK45',
        t_eval=t_span1,
        rtol=1e-8,
        atol=1e-8
    )

    # Extract time histories
    x_vec = [sol.y[0, :], sol.y[1, :], sol.y[2, :], sol.y[3, :], sol.y[4, :], sol.y[5, :]]
    l_vec = [sol.y[7, :], sol.y[8, :], sol.y[9, :], sol.y[10, :], sol.y[11, :], sol.y[12, :]]
    m = sol.y[6, :]
    l_m = sol.y[13, :]

    return x_vec, delta_values, shadow_flags, sol.t, l_m, m, l_vec

# Run single shooting to get trajectory, throttle usage, and other state history
SU, delta_values, shadow_flags, trajectory_times, l_m, m, CSU = single_shooting_1(solution)

# Print final mass profile
print(m)

# Compute relative time from start (in seconds)
times = trajectory_times - trajectory_times[0]

# Load reference target state (final boundary condition)
p_f, f_f, g_f, h_f, k_f, l_f = 187889.321482990, 0.387690133540649, -0.0243836062752043, \
                               0.0765422736430671, -0.0275226715429439, 26.2238381898900

# Convert MEE states to Cartesian for plotting
r, v = mee_to_rv(SU, mu)

# Load reference trajectory from file
from scipy.io import loadmat
data = loadmat('manifold_data.mat')
ECi = data['manifold_eci']
states = ECi[97][1]
x1 = states[86:, 0]
y1 = states[86:, 1]
z1 = states[86:, 2]

# Plot 3D trajectory comparison between manifold and optimized trajectory
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x1/Re, y1/Re, z1/Re, label='Manifold Trajectory', color='blue')
x2 = r[0, :]
y2 = r[1, :]
z2 = r[2, :]
ax.plot3D(x2/Re, y2/Re, z2/Re, label='Optimal Trajectory', color='red')

# Mark final point of optimized trajectory with a hollow circle
ax.scatter(x2[-1]/Re, y2[-1]/Re, z2[-1]/Re, marker='o', s=200, facecolor='none', edgecolor='green', linewidths=3, label='Boundary condition')

# Set labels and save figure
ax.set_xlabel('X (Earth radii)')
ax.set_ylabel('Y (Earth radii)')
ax.set_zlabel('Z (Earth radii)')
ax.legend()
plt.tight_layout()
plt.savefig('trajectory_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot throttle (delta) vs time
plt.figure(figsize=(14, 6))
plt.plot(times / 86400, delta_values)
plt.xlabel('Time (days)')
plt.ylabel('Delta Value')
plt.title('Delta Value Over Time')
plt.grid(True)
plt.savefig('delta_values_over_time_with_lm.png', dpi=300)
plt.show()

# Plot costate λ_m vs time
plt.figure(figsize=(14, 6))
plt.plot(times / 86400, l_m)
plt.xlabel('Time (days)')
plt.ylabel(r'$\lambda_m$')
plt.title(r'$\lambda_m$ vs time')
plt.grid(True)
plt.savefig('lm.png', dpi=300)
plt.show()

# Plot p vs time with final boundary condition marked
plt.figure(figsize=(14, 6))
plt.plot(times / 86400, SU[0] / Re, label='p', color='blue')
plt.plot(times[-1] / 86400, p_f / Re, marker='.', markersize=20, color='red', label='Boundary condition')
plt.xlabel('Time (days)')
plt.ylabel('p (Earth radii)')
plt.title('Element p vs Time')
plt.grid(True)
plt.savefig('p_vs_time.png', dpi=300)
plt.legend()
plt.show()

# Plot L vs time with final value
plt.figure(figsize=(14, 6))
plt.plot(times / 86400, ((SU[5] + np.pi) % (2 * np.pi) - np.pi), label='L', color='blue')
plt.plot(times[-1] / 86400, ((l_f + np.pi) % (2 * np.pi) - np.pi), marker='.', markersize=20, color='red', label='Boundary condition')
plt.xlabel('Time (days)')
plt.ylabel('L (in Radian)')
plt.title('Element L vs Time')
plt.grid(True)
plt.savefig('L_vs_time.png', dpi=300)
plt.legend()
plt.show()

# Plot f, g, h, k vs time and compare with boundary values
plt.figure(figsize=(14, 6))
labels = ['f', 'g', 'h', 'k']
labelbc = ['f Boundary condition', 'g Boundary condition', 'h Boundary condition', 'k Boundary condition']
colors = ['orange', 'green', 'red', 'purple','black']
final = [0.387690133540649, -0.0243836062752043, 0.0765422736430671, -0.0275226715429439]

# Loop over each element and plot
for i in range(1, 5):
    plt.plot(times / 86400, SU[i], label=labels[i - 1], color=colors[i - 1])
    plt.plot(times[-1] / 86400, final[i - 1], marker='.', markersize=20, label=labelbc[i - 1], color=colors[i])

plt.xlabel('Time (days)')
plt.ylabel('State Values')
plt.title('Elements f, g, h, k vs Time')
plt.legend()
plt.grid(True)
plt.savefig('fg_hkL_vs_time.png', dpi=300)
plt.legend()
plt.show()

# Initialize list to store Hamiltonian values at each time step
HAM = []

# Loop through each time step in the trajectory
for i in range(len(trajectory_times)):
    # Compute Julian date for current time step
    jd = 2459843.479016204 - trajectory_times[i] / 86400

    # Extract state and costate vectors at time i
    x_vec = [SU[0][i], SU[1][i], SU[2][i], SU[3][i], SU[4][i], SU[5][i]]
    l_vec = [CSU[0][i], CSU[1][i], CSU[2][i], CSU[3][i], CSU[4][i], CSU[5][i]]

    # Compute J2 perturbation in RTN frame
    aJ2 = compute_J2_acceleration(x_vec, mu)

    # Convert MEE to position and velocity vectors
    r_target, v_target = mee_to_rv(x_vec, mu)

    # Get positions of Sun, Moon, and Earth in inertial frame
    r_perturbingbody_s = sun.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_m = moon.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_e = earth.at(ts.ut1_jd(jd)).position.km

    # Compute Earth-relative positions for third-body calculations
    r_perturbingbody_srt = r_perturbingbody_s - r_perturbingbody_e
    r_perturbingbody_mrt = r_perturbingbody_m - r_perturbingbody_e

    # Compute third-body accelerations (Moon and Sun)
    t_m = third_body(r_target, r_perturbingbody_mrt, mu_moon)
    t_s = third_body(r_target, r_perturbingbody_srt, mu_sun)
    t = t_m + t_s

    # Transform third-body acceleration to RTN frame
    Q = rotation_mat(r_target, v_target)
    a_3b = Q.T @ t

    # Total perturbation acceleration in RTN frame
    acc = aJ2 + a_3b

    # Compute Hamiltonian at current time step and append to list
    HAM.append(Hamiltonian(x_vec, l_vec, l_m[i], m[i], acc, thrust_mag, c, mu, jd))

# Plot Hamiltonian vs time
plt.figure(figsize=(14, 6))
plt.plot(times / 86400, HAM, label='p', color='blue')
plt.xlabel('Time (days)')
plt.ylabel('Hamiltonian')
plt.title('Hamiltonian vs Time')
plt.grid(True)
plt.savefig('Hamiltonian_vs_time.png', dpi=300)
plt.show()
