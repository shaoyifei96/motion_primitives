#!/usr/bin/python3
import numpy as np
from scipy.linalg import expm
import scipy.integrate as integrate
from scipy.special import factorial
import matplotlib.pyplot as plt
import cProfile
from pycallgraph import PyCallGraph, Config
from pycallgraph.output import GraphvizOutput
import time
import pickle
import sympy as sym
from pathlib import Path


class MotionPrimitive():
    """
    Compute motion primitives for quadrotors over different size state spaces
    """

    def __init__(self, control_space_q=3, num_dims=2, num_u_per_dimension=5, max_state=[1, 1, 1, 1], num_state_deriv_pts=10, plot=False):
        """
        Input:
            control_space_q, derivative of configuration which is the control input.
            num_dims,        dimension of configuration space
            num_u_per_dimension, how many motion primitives per dimension 
            max_state, list of max values of configuration space and its derivatives
            num_state_deriv_pts, if creating a lookup table, how many samples per state per dimension
            plot, boolean of whether to create/show plots
        """

        self.control_space_q = control_space_q  # which derivative of position is the control space
        self.num_dims = num_dims  # Dimension of the configuration space
        self.num_u_per_dimension = num_u_per_dimension
        self.max_state = np.array(max_state)
        self.num_state_deriv_pts = num_state_deriv_pts
        self.plot = plot

        self.n = (self.control_space_q)*self.num_dims  # dimension of state space
        self.num_output_mps = self.num_u_per_dimension**self.num_dims  # number of total motion primitives

        # max control input #TODO should be a vector b/c perhaps different in Z
        self.max_u = self.max_state[self.control_space_q]
        self.num_u_set = 20  # Number of MPs to consider at a given time
        self.min_dt = 0
        self.max_dt = .5  # Max time horizon of MP
        self.num_dts = 10  # Number of time horizons to consider between 0 and max_dt

        self.A, self.B = self.A_and_B_matrices_quadrotor()
        self.quad_dynamics_polynomial = self.quad_dynamics_polynomial_symbolic()

    def compute_all_possible_mps(self, start_pt):
        """
        Compute a sampled reachable set from a start point, up to a max dt
        """
        single_u_set = np.linspace(-self.max_u, self.max_u, self.num_u_set)
        dt_set = np.linspace(self.min_dt, self.max_dt, self.num_dts)
        u_grid = np.meshgrid(*[single_u_set for i in range(self.num_dims)])
        u_set = np.dstack(([x.flatten() for x in u_grid]))[0].T
        sample_pts = np.array(self.quad_dynamics_polynomial(start_pt, u_set[:, :, np.newaxis], dt_set[np.newaxis, :]))
        sample_pts = np.transpose(sample_pts, (2, 1, 0))

        if self.plot:
            if self.num_dims > 1:
                plt.plot(sample_pts[:, :, 0], sample_pts[:, :, 1], marker='.', color='k', linestyle='none')
                plt.plot(start_pt[0], start_pt[1], 'og')
                plt.xlabel("X Position")
                plt.ylabel("Y Position")
            else:
                plt.plot(sample_pts[:, :, 0], np.zeros(sample_pts.shape[0:1]), marker='.', color='k', linestyle='none')
                plt.plot(start_pt[0], 0, 'og')

        return sample_pts, dt_set, u_set

    def uniform_state_set(self, bounds, resolution):
        """
        Return a uniform Cartesian sampling over vector bounds with vector resolution.
        Input:
            bounds, (N, 2) bounds over N dimensions
            resolution, (N,) resolution over N dimensions
        Output:
            pts, (M,N) set of M points sampled in N dimensions
        """
        assert len(bounds) == len(resolution)
        independent = []
        bounds = np.asarray(bounds)
        for (a, b, r) in zip(bounds[:, 0], bounds[:, 1], resolution):
            for _ in range(self.num_dims):
                independent.append(np.arange(a, b, r))
        joint = np.meshgrid(*independent)
        pts = np.stack([j.ravel() for j in joint], axis=-1)
        return pts

    def compute_min_dispersion_points(self, num_output_pts, potential_sample_pts, score, starting_output_sample_index):
        actual_sample_pts = np.zeros((num_output_pts, self.n))
        actual_sample_indices = np.zeros((num_output_pts)).astype(int)
        actual_sample_pts[0, :] = potential_sample_pts[starting_output_sample_index]
        actual_sample_indices[0] = starting_output_sample_index

        # distances of potential sample points to closest chosen output MP node # bottleneck
        min_score = np.ones((score.shape[0], 2))*np.inf
        for mp_num in range(1, num_output_pts):  # start at 1 because we already chose the closest point as a motion primitive
            # distances of potential sample points to closest chosen output MP node
            min_score[:, 0] = np.amin(min_score, axis=1)
            # take the new point with the maximum distance to its closest node
            index = np.argmax(min_score[:, 0])
            result_pt = potential_sample_pts[index, :]
            actual_sample_pts[mp_num, :] = result_pt
            actual_sample_indices[mp_num] = np.array((index))
            score[:, mp_num] = np.linalg.norm((potential_sample_pts - result_pt), axis=1)
            score[index, :] = - np.inf  # give nodes we have already chosen low score
            min_score[index, 0] = - np.inf  # give nodes we have already chosen low score
            min_score[:, 1] = score[:, mp_num]

        return actual_sample_pts, actual_sample_indices

    def compute_min_dispersion_set(self, start_pt):
        """
        Compute a set of num_output_mps primitives (u, dt) which generate a
        minimum state dispersion within the reachable state space after one
        step.
        """
        # TODO add stopping policy?

        score = np.ones((self.num_dts*self.num_u_set**self.num_dims, self.num_output_mps))*np.inf
        potential_sample_pts, dt_set, u_set = self.compute_all_possible_mps(start_pt)
        potential_sample_pts = potential_sample_pts.reshape(
            potential_sample_pts.shape[0]*potential_sample_pts.shape[1], potential_sample_pts.shape[2])
        # Take the closest motion primitive as the first choice (may want to change later)
        first_score = np.linalg.norm(potential_sample_pts-start_pt.T, axis=1)
        closest_pt = np.argmin(first_score)
        score[:, 0] = first_score

        actual_sample_pts, actual_sample_indices = self.compute_min_dispersion_points(
            self.num_output_mps, potential_sample_pts, score, closest_pt)

        actual_sample_indices = np.unravel_index(actual_sample_indices, (self.num_dts, self.num_u_set**self.num_dims))
        # Else compute minimum dispersion points over the whole state space (can be quite slow) (very similar to original Dispertio)
        dts = dt_set[actual_sample_indices[0]]
        us = u_set[:, actual_sample_indices[1]]

        if self.plot:
            if self.num_dims > 1:
                plt.plot(actual_sample_pts[:, 0], actual_sample_pts[:, 1], 'om')
                self.create_evenly_spaced_mps(start_pt, self.max_dt/2.0)
            else:
                plt.plot(actual_sample_pts[:, 0], np.zeros(actual_sample_pts.shape), 'om')

        return np.vstack((dts, us))

    def compute_min_dispersion_space(self, num_output_pts=250, resolution=[0.2, 0.2, 0.2]):
        """
        Using the bounds on the state space, compute a set of minimum dispersion points
        (Similar to original Dispertio paper)
        Can easily make too big of an array with small resolution :(
        """
        # Generate all points
        bounds = np.vstack((-self.max_state[:self.control_space_q], self.max_state[:self.control_space_q])).T
        potential_sample_pts = self.uniform_state_set(bounds, resolution)
        # potential_sample_pts = potential_sample_pts[:, np.newaxis, :]
        # print(potential_sample_pts.shape)
        score = np.ones((potential_sample_pts.shape[0], num_output_pts))*np.inf

        actual_sample_pts, actual_sample_indices = self.compute_min_dispersion_points(num_output_pts,
                                                                                      potential_sample_pts, score, 0)
        if self.plot:
            if self.num_dims > 1:
                plt.plot(actual_sample_pts[:, 0], actual_sample_pts[:, 1], 'om')

    def create_evenly_spaced_mps(self, start_pt, dt):
        """
        Create motion primitives for a start point by taking an even sampling over the 
        input space at a given dt
        i.e. old sikang method
        """
        single_u_set = np.linspace(-self.max_u, self.max_u, self.num_u_per_dimension)
        u_grid = np.meshgrid(*[single_u_set for i in range(self.num_dims)])
        u_set = np.dstack(([x.flatten() for x in u_grid]))[0].T
        sample_pts = np.array(self.quad_dynamics_polynomial(start_pt, u_set, dt))
        if self.plot:
            plt.plot(sample_pts[0, :], sample_pts[1, :], '*c')

        return np.vstack((sample_pts, np.ones((1, self.num_output_mps))*dt, u_set)).T

    def create_state_space_MP_lookup_table(self):
        """
        Uniformly sample the state space, and for each sample independently
        calculate a minimum dispersion set of motion primitives.
        """

        # Numpy nonsense that could be cleaner. Generate start pts at lots of initial conditions of the derivatives.
        y = np.array([np.tile(np.linspace(-i, i, self.num_state_deriv_pts), (self.num_dims, 1))
                      for i in self.max_state[1:self.control_space_q]])  # start at 1 to skip position
        z = np.reshape(y, (y.shape[0]*y.shape[1], y.shape[2]))
        start_pts_grid = np.meshgrid(*z)
        start_pts_set = np.dstack(([x.flatten() for x in start_pts_grid]))[0].T
        start_pts_set = np.vstack((np.zeros_like(start_pts_set[:self.num_dims, :]), start_pts_set))
        prim_list = []
        for start_pt in start_pts_set.T:
            prim_list.append(self.compute_min_dispersion_set(np.reshape(start_pt, (self.n, 1))))
            print(str(len(prim_list)) + '/' + str(start_pts_set.shape[1]))
            if self.plot:
                plt.show()

        self.start_pts = start_pts_set
        self.motion_primitives_list = prim_list
        file_path = Path("pickle/dimension_" + str(self.num_dims) + "/control_space_" +
                         str(self.control_space_q) + '/MotionPrimitive.pkl')
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open('wb') as output:  # TODO add timestamp of something back
            self.plot = False
            self.quad_dynamics_polynomial = None  # pickle has trouble with lambda function
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def A_and_B_matrices_quadrotor(self):
        """
        Generate constant A, B matrices for integrator of order control_space_q
        in configuration dimension num_dims. Linear approximation of quadrotor dynamics 
        that work (because differential flatness or something)
        """

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
        """
        Helper function to integrate A and B matrices to get G(t) term
        """
        return expm(self.A*(dt-sigma))@self.B

    def quad_dynamics_integral_wrapper(self, dt):
        def fn(sigma): return self.quad_dynamics_integral(sigma, dt)
        return fn

    def quad_dynamics(self, start_pt, u, dt):
        """
        Computes the state transition map given a starting state, control u, and 
        time dt. Slow because of integration and exponentiation
        """
        return expm(self.A*dt)@start_pt + integrate.quad_vec(self.quad_dynamics_integral_wrapper(dt), 0, dt)[0]@u

    def quad_dynamics_polynomial_symbolic(self):
        """
        Return a function that computes the state transition map given a
        starting state, control u, and time dt. Fast because just substitution into a polynomial.
        """
        start_pt = sym.Matrix([sym.symbols(f'start_pt{i}') for i in range(self.n)])
        u = sym.Matrix([sym.symbols(f'u{i}') for i in range(self.num_dims)])
        dt = sym.symbols('dt')
        pos = u*dt**self.control_space_q/factorial(self.control_space_q)
        for i in range(self.control_space_q):
            pos += sym.Matrix(start_pt[i*self.num_dims:(i+1)*self.num_dims]) * dt**i/factorial(i)
        x = pos
        for j in range(1, self.control_space_q):
            d = sym.diff(pos, dt, j)
            x = np.vstack((x, d))
        x = x.T[0]
        return sym.lambdify([start_pt, u, dt], x)


def create_many_state_space_lookup_tables(max_control_space):
    """
    Make motion primitive lookup tables for different state/input spaces
    """
    num_u_per_dimension = 3
    max_state = [2, 2, 1, 1, 1]
    num_state_deriv_pts = 7
    plot = False
    moprim_list = [MotionPrimitive(control_space_q, num_dims, num_u_per_dimension,
                                   max_state, num_state_deriv_pts, plot) for control_space_q in range(2, max_control_space) for num_dims in range(2, 3)]
    for moprim in moprim_list:
        print(moprim.control_space_q, moprim.num_dims)
        moprim.create_state_space_MP_lookup_table()


if __name__ == "__main__":
    control_space_q = 3
    num_dims = 2
    num_u_per_dimension = 5
    max_state = [1, .5, 1, 1]
    num_state_deriv_pts = 7
    plot = True
    mp = MotionPrimitive(control_space_q=control_space_q, num_dims=num_dims,
                         num_u_per_dimension=num_u_per_dimension, max_state=max_state, num_state_deriv_pts=num_state_deriv_pts, plot=plot)
    start_pt = np.ones((mp.n))*0.01
    # # mp.compute_all_possible_mps(start_pt)

    # with PyCallGraph(output=GraphvizOutput(), config=Config(max_depth=3)):
    mp.compute_min_dispersion_set(start_pt)
    # mp.compute_min_dispersion_space()
    # mp.create_state_space_MP_lookup_table()

    # # mp.create_evenly_spaced_mps(start_pt, mp.max_dt/2.0)

    # create_many_state_space_lookup_tables(5)

    plt.show()
