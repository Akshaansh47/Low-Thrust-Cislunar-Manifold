import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify
from skyfield.api import load
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve




# Constants
mu = 398600.435436096 # Earth's gravitational parameter, km^3/s^2
mu_sun = 132712440041.939  # Sun's gravitational parameter, km^3/s^2
mu_moon = 4902.8000661638  # Moon's gravitational parameter, km^3/s^2

thrust_mag = 0.235/1000
isp = 4155


J2 = 0.00108263
Re = 6378.1366  # Earth's radius in km
g0 = 9.80665  # m/s^2
c = isp * g0/1000 # Exhaust velocity


#This is the skyfield implementation
ts = load.timescale()
eph = load('de440s.bsp')
earth = eph['Earth']
sun = eph['Sun']
moon = eph['Moon']
ts = load.timescale()


# === Helper Functions ===

def Hamiltonian(x_vec, lam , m , a, T , c, mu):
    delta = sigmoid_throttle(lam, x_vec, mu)
    u = switch_ustar(lam, x_vec)
    A =compute_A_vector(x_vec, mu)
    B = compute_perturbation_matrix(x_vec, mu)
    H = 1 + np.dot(lam,(A + B @ a + delta * (T / m) * B @ u )) 
    return H

def derive_costate_equations():
    # Define symbolic variables
    p, f, g, h, k, L, m = sp.symbols('p f g h k L m')
    l_p, l_f, l_g, l_h, l_k, l_L = sp.symbols('l_p l_f l_g l_h l_k l_L')
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

    B = (1 / q) * sp.sqrt(p / mu) * sp.Matrix([
        [0, 2 * p, 0],
        [q * sp.sin(L), (q + 1) * sp.cos(L) + f, -g * (h * sp.sin(L) - k * sp.cos(L))],
        [-q * sp.cos(L), (q + 1) * sp.sin(L) + g, f * (h * sp.sin(L) - k * sp.cos(L))],
        [0, 0, s * sp.cos(L) / 2],
        [0, 0, s * sp.sin(L) / 2],
        [0, 0, (h * sp.sin(L) - k * sp.cos(L))]
    ])
    
    H = 1 + lam.dot(A + B @ a + delta * (T / m) * B @ u) 

    # Costate equations: λ̇ = -∂H/∂x
    full_costate_odes = -sp.Matrix([sp.diff(H, var) for var in x])

    return x, lam, m, u, a, T, delta, c, mu, full_costate_odes

def get_costate_dynamics_func():
    x, lam, m, u, a, T, delta, c, mu, costate_odes = derive_costate_equations()
    
    # Flatten input variables
    state_vars = x
    costate_vars = lam
    control_vars = u
    acc_var = a
    mass = m
    Thrust = T 

    # Create a numerical function
    ode_func = lambdify((state_vars, mass, costate_vars, control_vars, acc_var, delta, mu, T, c), costate_odes, 'numpy')
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
    r_vec , v_vec = mee_to_rv(x, mu)
    r = np.linalg.norm(r_vec)
    C1 = (mu * J2 * Re**2) / r**4
    C2 = h * np.sin(L) - k * np.cos(L)
    C3 = (1 + h**2 + k**2)**2

    ar = (-3 * C1 / 2) * (1 - (12 * C2**2 / C3))
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

    return u_star 

def mee_to_rv( x , mu):
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

def sigmoid_throttle(l_vec, x_vec, mu):
    """
    Sigmoid throttle function
    
    Args:
    S: Switch function value
    rho: Continuation parameter
    
    Returns:
    Throttle value between 0 and 1
    """
    B=compute_perturbation_matrix(x_vec,mu)
    S= np.linalg.norm(B.T@l_vec)
    throttle = 0.5 * (1 + np.sign(S))
    if throttle!=1:
        print(throttle)
    return throttle

def eom_mee_with_perturbations(x ,l_vec, u_vec, thrust_mag, mass, isp , jd, mu):
    B = compute_perturbation_matrix(x , mu)
    aJ2 = compute_J2_acceleration(x , mu)

    r_target , v_target = mee_to_rv(x, mu)

    r_perturbingbody_s = sun.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_m = moon.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_e = earth.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_srt = - r_perturbingbody_e + r_perturbingbody_s
    r_perturbingbody_mrt = r_perturbingbody_m - r_perturbingbody_e

    t_m = third_body(r_target,r_perturbingbody_mrt,mu_moon)
    t_s = third_body(r_target,r_perturbingbody_srt,mu_sun)
    t= t_m +t_s

    Q = rotation_mat(r_target , v_target)
    a_3b = Q.T @ t
    a = aJ2 + a_3b
    A = compute_A_vector(x , mu)
    delta = sigmoid_throttle(l_vec,x,mu)
    a_thrust = (thrust_mag * delta / mass) * u_vec
    dx = A + B @ a + B @ a_thrust
    dm = -thrust_mag * delta / c
    return dx, dm , a

def convert_cartesian_to_modified_equinoctial(state_cartesian, mu):
    position = state_cartesian[0:3, :]
    velocity = state_cartesian[3:6, :]

    radius = np.linalg.norm(position, axis=0)
    h_vec = np.cross(position.T, velocity.T).T
    h_mag = np.linalg.norm(h_vec, axis=0)
    radial_velocity = np.sum(position * velocity, axis=0) / radius


    r_hat = position / radius
    v_hat = (radius * velocity - radial_velocity * position) / h_mag
    h_hat = h_vec / h_mag

    # Eccentricity vector
    ecc_vec = np.cross(velocity.T, h_vec.T).T / mu - r_hat

    # p element
    p = h_mag ** 2 / mu

    # h and k elements
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

    # f and g elements
    f_elem = np.sum(ecc_vec * f_hat, axis=0)
    g_elem = np.sum(ecc_vec * g_hat, axis=0)

    # L element
    cosl = r_hat[0, :] + v_hat[1, :]
    sinl = r_hat[1, :] - v_hat[0, :]
    L = np.arctan2(sinl, cosl)

    return np.vstack([p, f_elem, g_elem, h, k, L])


def dynamics(t, vec , ode_func):
    p, f, g, h, k, l, m = vec[:7]
    l_p, l_f, l_g, l_h, l_k, l_l = vec[7:13]

    part_of_day = t / 86400
    jd = 2459783.4788888893 + part_of_day
    
    #print(t)
    
    l_vec = np.array([l_p, l_f, l_g, l_h, l_k, l_l])
    x_vec = np.array([p, f, g, h, k, l])

    u_opt = switch_ustar(l_vec, x_vec)
    delta = sigmoid_throttle(l_vec, x_vec, mu)

    dx , dm , acc = eom_mee_with_perturbations(x_vec ,l_vec, u_opt, thrust_mag , m , isp  , jd , mu)

    dot_p, dot_f, dot_g, dot_h, dot_k, dot_l=dx
    dot_m = dm
    
    dlam = ode_func(x_vec, m , l_vec  , u_opt , acc , delta ,mu, thrust_mag, c)

    dlam = np.array(dlam).flatten()

    dot_l_p, dot_l_f, dot_l_g, dot_l_h, dot_l_k, dot_l_l = dlam

    return [dot_p, dot_f, dot_g, dot_h, dot_k, dot_l, dot_m, dot_l_p, dot_l_f, dot_l_g, dot_l_h, dot_l_k, dot_l_l]



def single_shooting(para):
    p , f , g , h , k , l ,m = 42164,0,0,0,0,0,500
    #l_p0, l_f0, l_g0, l_h0, l_k0, l_l0, l_m0 = para[7:14]
    #t_seg = para[14]
    l_p0, l_f0, l_g0, l_h0, l_k0, l_l0 = para[:6]
    t_seg = para[6]*86400
    p_f , f_f , g_f , h_f , k_f , l_f = 187889.321482990,0.387690133540649,-0.0243836062752043,0.0765422736430671,-0.0275226715429439, 26.2238381898900

    #print()

    # Initial state for first segment
    initial_state_seg = [ p , f , g , h , k , l , m, 
                          l_p0, l_f0, l_g0, l_h0, l_k0, l_l0]
    
    ode_func = get_costate_dynamics_func()
    
    t_span1 = np.linspace(-t_seg, 0, 1000)
    
    sol = solve_ivp(lambda t, y: dynamics(t, y, ode_func), 
                [-t_seg, 0], 
                initial_state_seg, 
                method='RK45',  # Better for stiff problems
                t_eval=t_span1, 
                rtol=1e-8, 
                atol=1e-8)
    #sol = solve_ivp(lambda t, y: dynamics(t, y, ode_func), [t_start, t_start+t_seg], 
    #                 initial_state_seg, method='RK45', t_eval=t_span1, rtol = 1e-10, atol = 1e-10)
    
    
    
    
    x_vec = [sol.y[0,-1], sol.y[1,-1], sol.y[2,-1], sol.y[3,-1], sol.y[4,-1], sol.y[5,-1]]
    l_vec = [sol.y[7,-1], sol.y[8,-1], sol.y[9,-1], sol.y[10,-1], sol.y[11,-1], sol.y[12,-1] ]
    m = sol.y[6,-1]
    
    #for acc
    jd= 2459843.479016204 
    aJ2 = compute_J2_acceleration(x_vec , mu)

    r_target , v_target = mee_to_rv(x_vec, mu)

    r_perturbingbody_s = sun.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_m = moon.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_e = earth.at(ts.ut1_jd(jd)).position.km
    r_perturbingbody_srt = - r_perturbingbody_e + r_perturbingbody_s
    r_perturbingbody_mrt = r_perturbingbody_m - r_perturbingbody_e

    t_m = third_body(r_target,r_perturbingbody_mrt,mu_moon)
    t_s = third_body(r_target,r_perturbingbody_srt,mu_sun)
    t= t_m +t_s

    Q = rotation_mat(r_target , v_target)
    a_3b = Q.T @ t
    acc = aJ2 + a_3b

    Haml = Hamiltonian(x_vec, l_vec, m, acc, thrust_mag, c, mu)

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

    ]

    print(constraints)

    

    return constraints



initialState = np.array([-28.7266912380230,-563.081632198072,-103.175601217641,-739.274612104689,365.891880171321,-13.5870464342646, 52]);
#finalState = [190971.941319164	0.380248267433124	-0.00612969611643008	0.0753952145926258	-0.0234393284840843	33.8032581970921	469.889665727792	8.60974909187551e-06	-0.000122196663579075	-0.000139539451433293	-0.000241305514225556	0.000109110517158978	0.000123171020423738]'; % Manifold injection state

#initialState = [42164	0	0	0	0	0	500	-28.7266912380230	-563.081632198072	-103.175601217641	-739.274612104689	365.891880171321	-13.5870464342646]';
#finalState = [187889.321482990	0.387690133540649	-0.0243836062752043	0.0765422736430671	-0.0275226715429439	26.2238381898900	475.847816187510	-1.43830989874229e-05	-0.000885399387670010	-6.10659359790114e-05	-0.000292009585449994	0.000367324738558634	2.03851418831806e-05]';
#Traj segments




# Solve using fsolve
solution, info, ier, msg = fsolve(single_shooting, initialState, full_output=True)

if ier == 1:
    print("Converged successfully!")
else:
    print(f"Did not converge. Message: {msg}")




def single_shooting_1(para):
    p , f , g , h , k , l ,m = 42164,0,0,0,0,0,500
    #l_p0, l_f0, l_g0, l_h0, l_k0, l_l0, l_m0 = para[7:14]
    #t_seg = para[14]
    l_p0, l_f0, l_g0, l_h0, l_k0, l_l0 = para[:6]
    t_seg = para[6]*86400
    p_f , f_f , g_f , h_f , k_f , l_f = 187889.321482990,0.387690133540649,-0.0243836062752043,0.0765422736430671,-0.0275226715429439, 26.2238381898900

    #print()

    # Initial state for first segment
    initial_state_seg = [ p , f , g , h , k , l , m, 
                          l_p0, l_f0, l_g0, l_h0, l_k0, l_l0]
    
    ode_func = get_costate_dynamics_func()
    
    t_span1 = np.linspace(0, t_seg, 1000)
    
    sol = solve_ivp(lambda t, y: dynamics(t, y, ode_func), 
                [0, t_seg], 
                initial_state_seg, 
                method='RK45',  # Better for stiff problems
                t_eval=t_span1, 
                rtol=1e-8, 
                atol=1e-8)
    #sol = solve_ivp(lambda t, y: dynamics(t, y, ode_func), [t_start, t_start+t_seg], 
    #                 initial_state_seg, method='RK45', t_eval=t_span1, rtol = 1e-10, atol = 1e-10)
    
    
    
    
    x_vec = [sol.y[0,:], sol.y[1,:], sol.y[2,:], sol.y[3,:], sol.y[4,:], sol.y[5,:] ]
    l_vec = [sol.y[7,-1], sol.y[8,-1], sol.y[9,-1], sol.y[10,-1], sol.y[11,-1], sol.y[12,-1] ]
    m = sol.y[6,-1]
    

    print(m)

    return x_vec



SU=single_shooting_1(solution)



r,v=mee_to_rv(SU,mu)
#print(r)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example: SU has shape (3, N)
# SU = np.array([[x1, x2, x3, ...], [y1, y2, y3, ...], [z1, z2, z3, ...]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, z
x = r[0, :]
y = r[1, :]
z = r[2, :]

# Plot
ax.plot3D(x, y, z, label='Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()



from scipy.io import loadmat

data = loadmat('manifold_data.mat')
ECi=data['manifold_eci']
states = ECi[97][1]
# Extract x, y, z positions
x = states[:, 0]
y = states[:, 1]
z = states[:, 2]

# Plot in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Trajectory')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Trajectory')
ax.legend()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load data
data = loadmat('manifold_data.mat')
ECi = data['manifold_eci']

# First trajectory from the manifold data
states = ECi[97][1]
x1 = states[:, 0]
y1 = states[:, 1]
z1 = states[:, 2]


# Plot both
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot first trajectory (from manifold data)
ax.plot3D(x1, y1, z1, label='Manifold Trajectory', color='blue')

# Plot second trajectory (from r)
x2 = r[0, :]
y2 = r[1, :]
z2 = r[2, :]
ax.plot3D(x2, y2, z2, label='Second Trajectory', color='red')

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Save the figure before showing it
plt.savefig('trajectory_plot.png', dpi=300, bbox_inches='tight')

plt.show()




