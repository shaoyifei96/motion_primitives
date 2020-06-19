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
# from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from motion_primitive import PolynomialMotionPrimitive, JerksMotionPrimitive
# from scipy.integrate import solve_bvp


class MotionPrimitiveGraph():
    """
    Compute motion primitive graphs for quadrotors over different size state spaces
    """

    def __init__(self, control_space_q=3, num_dims=2,  max_state=[1, 1, 1, 1], plot=False):
        """
        Input:
            control_space_q, derivative of configuration which is the control input.
            num_dims,        dimension of configuration space
            max_state, list of max values of position space and its derivatives
            plot, boolean of whether to create/show plots
        """

        self.control_space_q = control_space_q  # which derivative of position is the control space
        self.num_dims = num_dims  # Dimension of the configuration space
        self.max_state = np.array(max_state)
        self.plot = plot
        self.dispersion_distance_fn = self.dispersion_distance_fn_simple_norm  # TODO pass as param/input?
        self.n = (self.control_space_q)*self.num_dims  # dimension of state space

        self.A, self.B = self.A_and_B_matrices_quadrotor()
        self.quad_dynamics_polynomial = self.quad_dynamics_polynomial_symbolic()
        self.x_derivs = PolynomialMotionPrimitive.setup_bvp_meam_620_style(self.control_space_q)
        if self.plot:
            self.fig = plt.figure()
        if self.num_dims == 3:
            ax = self.fig.add_subplot(111, projection='3d')

    def pickle_self(self):
        file_path = Path("pickle/dimension_" + str(self.num_dims) + "/control_space_" +
                         str(self.control_space_q) + '/MotionPrimitive.pkl')
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open('wb') as output:  # TODO add timestamp of something back
            self.plot = False
            self.quad_dynamics_polynomial = None  # pickle has trouble with lambda function
            self.x_derivs = None
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

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
                independent.append(np.arange(a, b+.00001, r))
        joint = np.meshgrid(*independent)
        pts = np.stack([j.ravel() for j in joint], axis=-1)
        return pts

    def dispersion_distance_fn_simple_norm(self, potential_sample_pts, result_pt):
        return np.linalg.norm((potential_sample_pts - result_pt), axis=1)
        # return np.linalg.norm((potential_sample_pts - result_pt)[:,:self.num_dims], axis=1) # position only

    def compute_min_dispersion_points(self, num_output_pts, potential_sample_pts, starting_score, starting_output_sample_index):
        actual_sample_indices = np.zeros((num_output_pts)).astype(int)
        actual_sample_indices[0] = starting_output_sample_index

        # distances of potential sample points to closest chosen output MP node # bottleneck
        min_score = np.ones((starting_score.shape[0], 2))*np.inf
        min_score[:, 0] = np.amin(starting_score, axis=1)
        for mp_num in range(1, num_output_pts):  # start at 1 because we already chose the closest point as a motion primitive
            # print(mp_num)
            # distances of potential sample points to closest chosen output MP node
            min_score[:, 0] = np.amin(min_score, axis=1)
            # take the new point with the maximum distance to its closest node
            index = np.argmax(min_score[:, 0])
            result_pt = potential_sample_pts[index, :]
            actual_sample_indices[mp_num] = np.array((index))
            min_score[index, 0] = - np.inf  # give nodes we have already chosen low score
            min_score[:, 1] = self.dispersion_distance_fn(potential_sample_pts, result_pt)  # new point's score
        actual_sample_pts = potential_sample_pts[actual_sample_indices]
        return actual_sample_pts, actual_sample_indices

    def create_evenly_spaced_mps(self, start_pt, dt, num_u_per_dimension):
        """
        Create motion primitives for a start point by taking an even sampling over the
        input space at a given dt
        i.e. old sikang method
        """
        max_u = self.max_state[self.control_space_q]
        single_u_set = np.linspace(-max_u, max_u, num_u_per_dimension)
        u_grid = np.meshgrid(*[single_u_set for i in range(self.num_dims)])
        u_set = np.dstack(([x.flatten() for x in u_grid]))[0].T
        sample_pts = np.array(self.quad_dynamics_polynomial(start_pt, u_set, dt))
        if self.plot:
            plt.plot(sample_pts[0, :], sample_pts[1, :], '*c')
        return np.vstack((sample_pts, np.ones_like(sample_pts[0])*dt, u_set)).T

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

