import numpy as np
from scipy.linalg import expm
import scipy.integrate as integrate
import matplotlib.pyplot as plt

control_space_q = 2  # which derivative of position is the control space
max_u = 1  # max control input
dt = .5
num_dims = 1  # Dimension of the configuration space

n = (control_space_q)*num_dims
A = np.zeros((n, n))
B = np.zeros((n, num_dims))
for p in range(1, control_space_q):
    A[(p-1)*num_dims: (p-1)*num_dims+num_dims, p*num_dims: p*num_dims+num_dims] = np.eye(num_dims)
B[-num_dims:, :] = np.eye(num_dims)


def quad_dynamics(sigma):
    return expm(A*(dt-sigma))@B


start_pt = np.zeros((n, 1))
u = np.ones((num_dims, 1))*max_u
sample_pt = expm(A*dt)@start_pt + integrate.quad_vec(quad_dynamics, 0, dt)[0]@u
print(sample_pt)
plt.plot(start_pt[0], start_pt[1], 'og')
plt.plot(sample_pt[0], sample_pt[1], 'or')
plt.xlabel("Position")
plt.ylabel("Velocity")
plt.show()


# nx, ny = (3, 2)
# x = np.linspace(0, 1, nx)
# y = np.linspace(0, 1, ny)
# xv, yv = np.meshgrid(x, y)

# print(xv, yv)

# num_mps = 5
# for i in range(num_mps):
#     pass
