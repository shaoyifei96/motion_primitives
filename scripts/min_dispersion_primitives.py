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
from sklearn.neighbors import NearestNeighbors
# from scipy.integrate import solve_bvp


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
            max_state, list of max values of position space and its derivatives
            num_state_deriv_pts, if creating a lookup table, how many samples per state per dimension
            plot, boolean of whether to create/show plots
        """

        self.control_space_q = control_space_q  # which derivative of position is the control space
        self.num_dims = num_dims  # Dimension of the configuration space
        self.num_u_per_dimension = num_u_per_dimension
        self.max_state = np.array(max_state)
        self.num_state_deriv_pts = num_state_deriv_pts
        self.plot = plot
        self.dispersion_distance_fn = self.dispersion_distance_fn_simple_norm  # TODO pass as param/input

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
        self.setup_bvp_meam_620_style()

    def pickle_self(self):
        file_path = Path("pickle/dimension_" + str(self.num_dims) + "/control_space_" +
                         str(self.control_space_q) + '/MotionPrimitive.pkl')
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open('wb') as output:  # TODO add timestamp of something back
            self.plot = False
            self.quad_dynamics_polynomial = None  # pickle has trouble with lambda function
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

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
                independent.append(np.arange(a, b+.00001, r))
        joint = np.meshgrid(*independent)
        pts = np.stack([j.ravel() for j in joint], axis=-1)
        return pts

    def dispersion_distance_fn_simple_norm(self, potential_sample_pts, result_pt):
        return np.linalg.norm((potential_sample_pts - result_pt), axis=1)

    def dispersion_distance_fn_path_length(self, potential_sample_pts, result_pt):
        score = np.zeros(potential_sample_pts.shape[0])
        for i in range(potential_sample_pts.shape[0]):
            polys, t = self.iteratively_solve_bvp_meam_620_style(result_pt, potential_sample_pts[i, :])
            score[i] = t #+ np.linalg.norm(u)*.0001  # tie break w/ u?
        return score

    def compute_min_dispersion_points(self, num_output_pts, potential_sample_pts, starting_score, starting_output_sample_index):
        actual_sample_indices = np.zeros((num_output_pts)).astype(int)
        actual_sample_indices[0] = starting_output_sample_index

        # distances of potential sample points to closest chosen output MP node # bottleneck
        min_score = np.ones((starting_score.shape[0], 2))*np.inf
        min_score[:, 0] = np.amin(starting_score, axis=1)
        for mp_num in range(1, num_output_pts):  # start at 1 because we already chose the closest point as a motion primitive
            print(mp_num)
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

    def compute_min_dispersion_set(self, start_pt):
        """
        Compute a set of num_output_mps primitives (u, dt) which generate a
        minimum state dispersion within the reachable state space after one
        step.
        """
        # TODO add stopping policy?
        self.dispersion_distance_fn = self.dispersion_distance_fn_simple_norm
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
        # TODO not actually dispertio yet because not using steer function/reachable sets
        """
        self.dispersion_distance_fn = self.dispersion_distance_fn_path_length

        bounds = np.vstack((-self.max_state[:self.control_space_q], self.max_state[:self.control_space_q])).T
        potential_sample_pts = self.uniform_state_set(bounds, resolution[:self.control_space_q])
        print(potential_sample_pts.shape)
        score = np.ones((potential_sample_pts.shape[0], num_output_pts))*np.inf
        starting_output_sample_index = 0
        score[:, 0] = self.dispersion_distance_fn_simple_norm(potential_sample_pts, potential_sample_pts[starting_output_sample_index, :])
        actual_sample_pts, actual_sample_indices = self.compute_min_dispersion_points(num_output_pts,
                                                                                      potential_sample_pts, score, starting_output_sample_index)
        print(actual_sample_pts)
        self.reconnect_lattice(actual_sample_pts)
        if self.plot:
            if self.num_dims > 1:
                plt.plot(actual_sample_pts[:, 0], actual_sample_pts[:, 1], 'om')

        return actual_sample_pts

    def reconnect_lattice(self, sample_pts):
        print("reconnect lattice")
        for start_pt in sample_pts:
            for end_pt in sample_pts:
                if (start_pt == end_pt).all():
                    continue
                polys, T = self.iteratively_solve_bvp_meam_620_style(start_pt, end_pt)
                #TODO enforce a max number of connections
                #TODO save and output to pickle
                if self.plot:
                    if ~np.isinf(T):
                        t_list = np.linspace(0, T, 10)
                        x = [np.polyval(polys[0, :], i) for i in t_list]
                        y = [np.polyval(polys[1, :], i) for i in t_list]
                        plt.plot(x, y)
                        
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

    def create_state_space_MP_lookup_table_tree(self):
        """
        Uniformly sample the state space, and for each sample independently
        calculate a minimum dispersion set of motion primitives.
        """

        # Numpy nonsense that could be cleaner. Generate start pts at lots of initial conditions of the derivatives.
        # TODO replace with Jimmy's cleaner uniform_sample function
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
        self.pickle_self()

    # def create_state_space_MP_lookup_table_lattice(self, resolution=[.1, .1, .1]):
    #     lattice_pts = self.compute_min_dispersion_space()
    #     bounds = np.vstack((-self.max_state[:self.control_space_q], self.max_state[:self.control_space_q])).T
    #     # TODO make sure lattice points are included in start pts
    #     start_pts = self.uniform_state_set(bounds, resolution[:self.control_space_q])
    #     nbrs = NearestNeighbors(n_neighbors=self.num_output_mps, algorithm='ball_tree').fit(
    #         lattice_pts)  # check into other metrics and algorithm types (kdtree)
    #     indices = nbrs.kneighbors(start_pts, return_distance=False)
    #     knn_pts = lattice_pts[indices].shape
    #     # self.compute_mps_from_BCs(knn_pts)

    #     # self.pickle_self()

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

    def setup_bvp_meam_620_style(self):
        t = sym.symbols('t')
        self.poly_order = (self.control_space_q-1)*2+1
        x = np.squeeze(sym.Matrix(np.zeros((self.poly_order+1))))
        for i in range(self.poly_order+1):
            x[i] = t**(self.poly_order-i)  # Construct polynomial of the form [T**5,    T**4,   T**3, T**2, T, 1]

        self.x_derivs = []
        for i in range(self.control_space_q+1):
            self.x_derivs.append(sym.lambdify([t], x))
            x = sym.diff(x)  # iterate through all the derivatives
        self.q_factorial = factorial(self.control_space_q)

    def solve_bvp_meam_620_style(self, xi, xf, T):
        """
        Return polynomial coefficients from xi ((n,) array) to xf ((n,) array) in time interval [0,T]
        """

        A = np.zeros((self.poly_order+1, self.poly_order+1))
        for i in range(self.control_space_q):
            x = self.x_derivs[i]  # iterate through all the derivatives
            A[i, :] = x(0)  # x(ti) = xi
            A[self.control_space_q+i, :] = x(T)  # x(tf) = xf
        # u = np.zeros(self.num_dims)
        polys = np.zeros((self.num_dims, self.poly_order+1))
        for i in range(self.num_dims):  # Construct a separate polynomial for each dimension

            # vector of the form [xi,xf,xi_dot,xf_dot,...]
            b = np.hstack((xi[i::self.num_dims], xf[i::self.num_dims]))  # TODO this line kind of slow
            poly = np.linalg.solve(A, b)
            # only care about the first coefficient, which encodes the constant u
            # u[i] = poly[0]*self.q_factorial

            polys[i, :] = poly
        # if self.plot:
        #     t_list = np.linspace(0, T, 100)
        #     x = [np.polyval(polys[0, :], i) for i in t_list]
        #     y = [np.polyval(polys[1, :], i) for i in t_list]
        #     plt.plot(xi[0], xi[1], 'og')
        #     plt.plot(xf[0], xf[1], 'or')
        #     plt.plot(x, y)
        #     x = np.zeros((100, self.n))

            # for i, t in enumerate(t_list):
            #     x[i, :] = self.quad_dynamics_polynomial(xi, u, t)
            # plt.plot(x[:, 0], x[:, 1], 'y')
        return polys

    def iteratively_solve_bvp_meam_620_style(self, start_pt, goal_pt):
        # TODO make parameters
        dt = .2
        max_t = 1
        t = 0
        u_max = np.inf
        polys = None
        while u_max > self.max_state[self.control_space_q]:
            t += dt
            if t > max_t:
                # u = np.ones(self.num_dims)*np.inf
                polys = None
                t = np.inf
                break
            polys = self.solve_bvp_meam_620_style(start_pt, goal_pt, t)
            u_max = max(abs(np.sum(polys*self.x_derivs[-1](t),axis=1))) # TODO this is only u(t), not necessarily max(u) from 0 to t which we would want, use critical points maybe?
            u_max = max(u_max,max(abs(np.sum(polys*self.x_derivs[-1](0),axis=1))))
        # print(u_max,polys,t)
        return polys, t


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
    control_space_q = 2
    num_dims = 2
    num_u_per_dimension = 5
    max_state = [1, 1, 10, 100, 1, 1]
    num_state_deriv_pts = 7
    plot = True
    mp = MotionPrimitive(control_space_q=control_space_q, num_dims=num_dims,
                         num_u_per_dimension=num_u_per_dimension, max_state=max_state, num_state_deriv_pts=num_state_deriv_pts, plot=plot)
    start_pt = np.ones((mp.n))
    # start_pt = np.array([-1., -2., 0, 0.5])
    # print(mp.solve_bvp_meam_620_style(start_pt, start_pt*2, 1))
    # print(mp.iteratively_solve_bvp_meam_620_style(start_pt,start_pt*2))

    with PyCallGraph(output=GraphvizOutput(), config=Config(max_depth=6)):
        mp.compute_min_dispersion_space(num_output_pts=10, resolution=[.5,.5,.75])
    # # mp.compute_all_possible_mps(start_pt)

    # with PyCallGraph(output=GraphvizOutput(), config=Config(max_depth=3)):
    # mp.compute_min_dispersion_set(start_pt)
    # mp.compute_min_dispersion_space()
    # mp.create_state_space_MP_lookup_table_tree()
    # mp.create_state_space_MP_lookup_table_lattice()

    # # mp.create_evenly_spaced_mps(start_pt, mp.max_dt/2.0)

    # create_many_state_space_lookup_tables(5)

    plt.show()
