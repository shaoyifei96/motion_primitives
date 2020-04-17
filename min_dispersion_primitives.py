import numpy as np
from scipy.linalg import expm
import scipy.integrate as integrate
import matplotlib.pyplot as plt


class MotionPrimitive():

    def __init__(self, control_space_q=3, num_dims=3):
        self.control_space_q = control_space_q  # which derivative of position is the control space
        self.num_dims = num_dims  # Dimension of the configuration space
        self.n = (self.control_space_q)*self.num_dims

        self.max_u = 1  # max control input
        self.num_u_set = 3  # Number of MPs to consider at a given time
        self.max_dt = 1  # Max time horizon of MP
        self.num_dts = 5  # Number of time horizons to consider between 0 and max_dt

        self.A, self.B = self.A_and_B_matrices_quadrotor()

    def compute_all_possible_mps(self, start_pt):

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

        sample_pts = first_terms_repeated + second_terms
        sample_pts = np.transpose(sample_pts, (0, 2, 1))  # Order so that it's dt_index, du_index, state_space_index
        plt.plot(sample_pts[:, :, 0], sample_pts[:, :, 1], marker='.', color='k', linestyle='none')

        plt.plot(start_pt[0], start_pt[1], 'og')
        plt.xlabel("X Position")
        plt.ylabel("Y Position")

        # plt.show()

        return sample_pts

    def compute_min_dispersion_set(self, start_pt, num_output_mps):
        potential_sample_pts = self.compute_all_possible_mps(start_pt)
        # TODO maybe dont make a rectangle in state space; do another polygon?
        border = np.array((np.amin(potential_sample_pts, axis=(0, 1)), np.amax(potential_sample_pts, axis=(0, 1))))
        plt.plot(border[0][0], border[0][1], 'ro')
        plt.plot(border[1][0], border[1][1], 'ro')

        # actual_sample_pts = np.zeros((self.n, 1))
        # comparison_pts = np.hstack((actual_sample_pts, border))
        comparison_pts = np.vstack((border, start_pt.T))
        actual_sample_pts = None
        actual_sample_indices = None
        # comparison_pts = start_pt.T
        for mp_num in range(2):  # num_output_mps
            print("New mp")
            score = np.zeros((potential_sample_pts.shape[0], potential_sample_pts.shape[1]))
            for i in range(comparison_pts.shape[0]):  # TODO vectorize
                # print(potential_sample_pts - comparison_pts[i, :])
                score += np.linalg.norm(potential_sample_pts - comparison_pts[i, :], axis=2)
            if actual_sample_indices is not None:
                print(actual_sample_indices)
                score[actual_sample_indices[:, 0], actual_sample_indices[:, 1]] = -10**10
            dt_index, du_index = np.where(score == np.amax(score))
            result_pt = potential_sample_pts[dt_index[0], du_index[0], :].T
            if actual_sample_pts is None:
                actual_sample_pts = result_pt
                actual_sample_pts = actual_sample_pts.reshape((1, self.n))
                actual_sample_indices = np.array((dt_index[0], du_index[0]))
                actual_sample_indices = actual_sample_indices.reshape((1, 2))
            else:
                actual_sample_pts = np.vstack((actual_sample_pts, result_pt))
                actual_sample_indices = np.vstack((actual_sample_indices, np.array((dt_index[0], du_index[0]))))
            comparison_pts = np.vstack((comparison_pts, result_pt))
        plt.plot(actual_sample_pts[:, 0], actual_sample_pts[:, 1], 'ob')
        plt.show()
        # print(potential_sample_pts.shape)
        # print((np.repeat(potential_sample_pts, comparison_pts.shape[1], 2)).shape)
        # print(potential_sample_pts - border[:, 0])

    def A_and_B_matrices_quadrotor(self):
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
    mp = MotionPrimitive(control_space_q=1, num_dims=2)
    start_pt = np.ones((mp.n, 1))*0.01
    # mp.compute_all_possible_mps(start_pt)
    mp.compute_min_dispersion_set(start_pt, 5)
