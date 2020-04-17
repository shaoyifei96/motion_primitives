import numpy as np
from scipy.linalg import expm
import scipy.integrate as integrate
import matplotlib.pyplot as plt


control_space_q = 3  # which derivative of position is the control space
num_dims = 3  # Dimension of the configuration space
n = (control_space_q)*num_dims
start_pt = np.ones((n, 1))*.05

max_u = 1  # max control input
num_u_set = 11  # Number of MPs to consider at a given time
max_dt = 1  # Max time horizon of MP
num_dts = 100  # Number of time horizons to consider between 0 and max_dt


def A_and_B_matrices(n):
    A = np.zeros((n, n))
    B = np.zeros((n, num_dims))
    for p in range(1, control_space_q):
        A[(p-1)*num_dims: (p-1)*num_dims+num_dims, p*num_dims: p*num_dims+num_dims] = np.eye(num_dims)
    B[-num_dims:, :] = np.eye(num_dims)
    return A, B


A, B = A_and_B_matrices(n)


def quad_dynamics_integral(sigma):
    return expm(A*(dt-sigma))@B


def quad_dynamics(start_pt, u, dt):
    return expm(A*dt)@start_pt + integrate.quad_vec(quad_dynamics_integral, 0, dt)[0]@u


single_u_set = np.linspace(-max_u, max_u, num_u_set)
dt_set = np.linspace(0, max_dt, num_dts)
integral_set = np.empty((num_dts, n, num_dims))
expm_A_set = np.empty((num_dts, n, n))
for i, dt in enumerate(dt_set):
    integral_set[i, :, :] = integrate.quad_vec(quad_dynamics_integral, 0, dt)[0]
    expm_A_set[i, :, :] = expm(A*dt)
u_grid = np.meshgrid(*[single_u_set for i in range(num_dims)])
u_set = np.dstack(([x.flatten() for x in u_grid]))[0].T

first_terms = expm_A_set@start_pt  # indexed by dt index num_dt x n x 1
first_terms_repeated = np.repeat(first_terms, num_u_set**num_dims, 2)  # num_dt x n x num_u_set**num_dims
second_terms = integral_set@u_set  # num_dt x n x num_u_set^num_dims

total = first_terms_repeated + second_terms
total = np.transpose(total, (0, 2, 1))  # Order so that it's dt,du,result
plt.plot(total[:, :, 0], total[:, :, 1], marker='.', color='k', linestyle='none')

border = [np.amin(total, axis=(0, 1)), np.amax(total, axis=(0, 1))]  # todo maybe dont make a rectangle
plt.plot(border[0][0], border[0][1], 'ro')
plt.plot(border[1][0], border[1][1], 'ro')
plt.plot(start_pt[0], start_pt[1], 'og')
plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.show()

# num_mps = 5
# for i in range(num_mps):
#     pass
