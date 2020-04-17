import numpy as np
from scipy.linalg import expm
import scipy.integrate as integrate
import matplotlib.pyplot as plt

control_space_q = 3  # which derivative of position is the control space
max_u = 1  # max control input
num_dims = 3  # Dimension of the configuration space

n = (control_space_q)*num_dims
A = np.zeros((n, n))
B = np.zeros((n, num_dims))
for p in range(1, control_space_q):
    A[(p-1)*num_dims: (p-1)*num_dims+num_dims, p*num_dims: p*num_dims+num_dims] = np.eye(num_dims)
B[-num_dims:, :] = np.eye(num_dims)


def quad_dynamics_integral(sigma):
    return expm(A*(dt-sigma))@B


def quad_dynamics(start_pt, u, dt):
    return expm(A*dt)@start_pt + integrate.quad_vec(quad_dynamics_integral, 0, dt)[0]@u


start_pt = np.zeros((n, 1))
u = np.ones((num_dims, 1))*max_u
dt = .5
sample_pt = quad_dynamics(start_pt, u, dt)
plt.plot(start_pt[0], start_pt[1], 'og')
plt.plot(sample_pt[0], sample_pt[1], 'or')
plt.xlabel("Position")
plt.ylabel("Velocity")


num_u_set = 5
single_u_set = np.linspace(-max_u, max_u, num_u_set)
max_dt = 1
num_dts = 10
dt_set = np.linspace(0, max_dt, num_dts)
integral_set = np.empty((num_dts, n, num_dims))
expm_A_set = np.empty((num_dts, n, n))
for i, dt in enumerate(dt_set):
    integral_set[i, :, :] = integrate.quad_vec(quad_dynamics_integral, 0, dt)[0]
    expm_A_set[i, :, :] = expm(A*dt)
u_grid = np.meshgrid(*[single_u_set for i in range(num_dims)])
u_set = np.dstack(([x.flatten() for x in u_grid]))[0].T

print(integral_set.shape)
print(u_set.shape)
second_terms = integral_set@u_set  # num_dt x n x num_u_set^num_dims
first_terms = expm_A_set@start_pt  # indexed by dt index num_dt x n x 1
first_terms_repeated = np.repeat(first_terms, num_u_set**num_dims, 2)
total = first_terms_repeated + second_terms
total = np.transpose(total, (0, 2, 1))
plt.plot(total[:, :, 0], total[:, :, 1], marker='.', color='k', linestyle='none')
# u = np.array([i, j])
# sample_pt = expm(A*t)@start_pt + integrate.quad_vec(quad_dynamics_integral, 0, t)[0]@u

# print(xu, yu)
# plt.plot(xu, yu, marker='.', color='k', linestyle='none')
plt.show()

# num_mps = 5
# for i in range(num_mps):
#     pass
