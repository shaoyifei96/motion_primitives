import numpy as np
from scipy.linalg import expm
import scipy.integrate as integrate
import matplotlib.pyplot as plt


class MotionPrimitive():

    def __init__(self):
        self.control_space_q = 3  # which derivative of position is the control space
        self.num_dims = 3  # Dimension of the configuration space
        self.n = (self.control_space_q)*self.num_dims

        self.max_u = 1  # max control input
        self.num_u_set = 11  # Number of MPs to consider at a given time
        self.max_dt = 1  # Max time horizon of MP
        self.num_dts = 100  # Number of time horizons to consider between 0 and max_dt

        self.A, self.B = self.A_and_B_matrices()

    def create_mps(self):
        start_pt = np.ones((self.n, 1))*.05

        single_u_set = np.linspace(-self.max_u, self.max_u, self.num_u_set)
        dt_set = np.linspace(0, self.max_dt, self.num_dts)
        integral_set = np.empty((self.num_dts, self.n, self.num_dims))
        expm_A_set = np.empty((self.num_dts, self.n, self.n))
        for i, dt in enumerate(dt_set):
            integral_set[i, :, :] = integrate.quad_vec(self.quad_dynamics_integral_wrapper(dt), 0, dt)[0]
            expm_A_set[i, :, :] = expm(self.A*dt)
        u_grid = np.meshgrid(*[single_u_set for i in range(self.num_dims)])
        u_set = np.dstack(([x.flatten() for x in u_grid]))[0].T

        first_terms = expm_A_set@start_pt  # indexed by dt index num_dt x n x 1
        first_terms_repeated = np.repeat(first_terms, self.num_u_set**self.num_dims,
                                         2)  # num_dt x n x num_u_set**num_dims
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

    def A_and_B_matrices(self):
        n = self.n
        num_dims = self.num_dims
        control_space_q = self.control_space_q

        A = np.zeros((n, n))
        B = np.zeros((n, num_dims))
        for p in range(1, control_space_q):
            A[(p-1)*num_dims: (p-1)*num_dims+num_dims, p*num_dims: p*num_dims+num_dims] = np.eye(num_dims)
        B[-num_dims:, :] = np.eye(num_dims)
        return A, B

    def quad_dynamics_integral(self, sigma, dt):
        return expm(self.A*(dt-sigma))@self.B

    def quad_dynamics_integral_wrapper(self, dt):
        def fn(sigma): return self.quad_dynamics_integral(sigma, dt)
        return fn

    def quad_dynamics(self, start_pt, u, dt):
        return expm(self.A*dt)@start_pt + integrate.quad_vec(self.quad_dynamics_integral_wrapper(dt), 0, dt)[0]@u


if __name__ == "__main__":
    mp = MotionPrimitive()
    mp.create_mps()
