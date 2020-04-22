#!/usr/bin/python3
import numpy as np
from scipy.linalg import expm
import scipy.integrate as integrate
from scipy.special import factorial
import matplotlib.pyplot as plt
import cProfile
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import time
import pickle
import sympy as sym


class MotionPrimitive():

    def __init__(self, control_space_q=3, num_dims=2, num_u_per_dimension=5, max_state_derivs=[1, 1, 1, 1], num_state_deriv_pts=10, plot=False):
        self.control_space_q = control_space_q  # which derivative of position is the control space
        self.num_dims = num_dims  # Dimension of the configuration space
        self.num_u_per_dimension = num_u_per_dimension
        self.max_state_derivs = max_state_derivs
        self.num_state_deriv_pts = num_state_deriv_pts
        self.plot = plot

        self.n = (self.control_space_q)*self.num_dims
        self.num_output_mps = num_u_per_dimension**num_dims

        self.max_u = 1  # max control input #TODO should be a vector b/c different in Z
        self.num_u_set = 30  # Number of MPs to consider at a given time
        self.min_dt = 0
        self.max_dt = .5  # Max time horizon of MP
        self.num_dts = 60  # Number of time horizons to consider between 0 and max_dt

        self.A, self.B = self.A_and_B_matrices_quadrotor()
        self.quad_dynamics_polynomial = self.quad_dynamics_polynomial_symbolic()
        # self.quad_dynamics_polynomial = self.quad_dynamics_polynomial_symbolic_better_vectorization()

    def compute_all_possible_mps(self, start_pt):

        single_u_set = np.linspace(-self.max_u, self.max_u, self.num_u_set)
        dt_set = np.linspace(self.min_dt, self.max_dt, self.num_dts)
        u_grid = np.meshgrid(*[single_u_set for i in range(self.num_dims)])
        u_set = np.dstack(([x.flatten() for x in u_grid]))[0].T
        sample_pts = None
        for dt_index in range(dt_set.shape[0]):  # TODO vectorize better. Has to be in the symbolic math function (quad_dynamics_polynomial)
            sp = np.array(self.quad_dynamics_polynomial(start_pt, u_set, dt_set[dt_index]))
            if sample_pts is None:
                sample_pts = sp
            else:
                sample_pts = np.dstack((sample_pts, sp))
        # print(self.quad_dynamics_polynomial(start_pt, u_set, dt_set))
        sample_pts = np.transpose(sample_pts, (1, 2, 0))

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

    def compute_min_dispersion_set(self, start_pt):
        potential_sample_pts, dt_set, u_set = self.compute_all_possible_mps(start_pt)
        # TODO maybe dont make a rectangle in state space; do another polygon?
        # border = np.array((np.amin(potential_sample_pts, axis=(0, 1)), np.amax(potential_sample_pts, axis=(0, 1))))
        # print(border.shape)
        # border = np.vstack((border, np.array((border[0], border[1])), np.array((border[1], border[0]))))
        # print(border.shape)
        # plt.plot(border[:, 0], border[:, 1], 'ro')

        # comparison_pts = np.vstack((border, start_pt.T))
        # TODO add stopping policy?
        a = np.linalg.norm(potential_sample_pts-start_pt.T, axis=2)
        closest_pt = np.unravel_index(np.argmin(a, axis=None), a.shape)
        actual_sample_pts = potential_sample_pts[closest_pt]
        actual_sample_pts = actual_sample_pts.reshape((1, self.n))
        actual_sample_indices = np.array(closest_pt)
        actual_sample_indices = actual_sample_indices.reshape((1, 2))
        for mp_num in range(self.num_output_mps):  # num_output_mps
            score = np.zeros((potential_sample_pts.shape[0], potential_sample_pts.shape[1], actual_sample_pts.shape[0]))
            for i in range(actual_sample_pts.shape[0]):  # TODO vectorize
                score[:, :, i] = np.linalg.norm(potential_sample_pts - actual_sample_pts[i, :], axis=2)
            score = np.amin(score, axis=2)
            score[actual_sample_indices[:, 0], actual_sample_indices[:, 1]] = -10**10
            dt_index, du_index = np.where(score == np.amax(score))
            result_pt = potential_sample_pts[dt_index[0], du_index[0], :].T
            actual_sample_pts = np.vstack((actual_sample_pts, result_pt))
            actual_sample_indices = np.vstack((actual_sample_indices, np.array((dt_index[0], du_index[0]))))
        if self.plot:
            if self.num_dims > 1:
                plt.plot(actual_sample_pts[:, 0], actual_sample_pts[:, 1], 'ob')
                self.create_evenly_spaced_mps(start_pt, self.max_dt/2.0)
            else:
                plt.plot(actual_sample_pts[:, 0], np.zeros(actual_sample_pts.shape), 'ob')

        dts = dt_set[actual_sample_indices[:, 0]]
        us = u_set[:, actual_sample_indices[:, 1]]
        return np.vstack((dts, us))

    def create_evenly_spaced_mps(self, start_pt, dt):
        # i.e. old sikang method
        single_u_set = np.linspace(-self.max_u, self.max_u, self.num_u_per_dimension)
        u_grid = np.meshgrid(*[single_u_set for i in range(self.num_dims)])
        u_set = np.dstack(([x.flatten() for x in u_grid]))[0].T
        sample_pts = np.array(self.quad_dynamics_polynomial(start_pt, u_set, dt))
        if self.plot:
            plt.plot(sample_pts[0, :], sample_pts[1, :], 'oy')

        return np.vstack((sample_pts, np.ones((1, self.num_output_mps))*dt, u_set)).T

    def create_state_space_MP_lookup_table(self):
        # Numpy nonsense that could be cleaner. Generate start pts at lots of initial conditions of the derivatives.
        y = np.array([np.tile(np.linspace(-i, i, self.num_state_deriv_pts), (self.num_dims, 1))
                      for i in self.max_state_derivs[:self.control_space_q-1]])
        z = np.reshape(y, (y.shape[0]*y.shape[1], y.shape[2]))
        start_pts_grid = np.meshgrid(*z)
        start_pts_set = np.dstack(([x.flatten() for x in start_pts_grid]))[0].T
        start_pts_set = np.vstack((np.zeros_like(start_pts_set[:num_dims, :]), start_pts_set))

        prim_list = []
        for start_pt in start_pts_set.T:
            prim_list.append(mp.compute_min_dispersion_set(np.reshape(start_pt, (self.n, 1))))
            if self.plot:
                plt.show()

        self.start_pts = start_pts_set
        self.motion_primitives_list = prim_list
        with open('pickle/MotionPrimitive.pkl', 'wb') as output:  # TODO add timestamp of something back
            self.plot = False
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

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

    def quad_dynamics_polynomial_symbolic(self):
        start_pt = sym.Matrix([sym.symbols('start_pt%d' % i) for i in range(self.n)])
        u = sym.Matrix([sym.symbols('u%d' % i) for i in range(self.num_dims)])
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

    def quad_dynamics_polynomial_symbolic_better_vectorization(self):
        # not working
        start_pt = sym.Matrix([sym.symbols('start_pt%d' % i) for i in range(self.n)])
        u = sym.Matrix([sym.symbols('u%d' % i) for i in range(self.num_dims)])
        dt = sym.Matrix([sym.symbols('dt')])
        pos = np.outer(u, dt**(self.control_space_q)) / factorial(self.control_space_q)
        for i in range(self.control_space_q):
            pos += np.outer(sym.Matrix(start_pt[i*self.num_dims:(i+1)*self.num_dims]), dt**i)/factorial(i)
        x = pos
        x = x.reshape((self.num_dims))
        for j in range(1, self.control_space_q):
            d = sym.diff(pos, dt, j)
            d = d.reshape((self.num_dims))
            x = np.hstack((x, d))
        print(x)
        # x = np.flip(x.T)[0]
        # x = x.reshape((self.n, 1))
        return sym.lambdify([start_pt, u, dt], x)


if __name__ == "__main__":
    control_space_q = 3
    num_dims = 2
    num_u_per_dimension = 5
    max_state_derivs = [1, 1, 1, 1]
    num_state_deriv_pts = 5
    plot = True
    mp = MotionPrimitive(control_space_q=control_space_q, num_dims=num_dims,
                         num_u_per_dimension=num_u_per_dimension, max_state_derivs=max_state_derivs, num_state_deriv_pts=num_state_deriv_pts, plot=plot)
    start_pt = np.ones((mp.n))*0.5
    mp.compute_all_possible_mps(start_pt)

    # with PyCallGraph(output=GraphvizOutput()):

    # mp.compute_min_dispersion_set(start_pt)
    # mp.create_evenly_spaced_mps(start_pt, mp.max_dt/2.0)

    # mp.create_state_space_MP_lookup_table()

    print(mp.quad_dynamics_polynomial(start_pt,[1,1],1.5))
    print(mp.quad_dynamics(start_pt,[1,1],1.5))

    plt.show()
