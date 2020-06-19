from py_opt_control import min_time_bvp
import matplotlib.pyplot as plt
import sympy as sym
from scipy.special import factorial
import numpy as np


class MotionPrimitive():
    """
    # WIP
    A motion primitive that defines a trajectory from a over a time T. 
    Put functions that all MPs should have in this base class. 
    If the implementation is specific to the subclass, raise a NotImplementedError
    """

    def __init__(self, start_state, end_state, num_dims, max_state):
        """
        """
        self.start_state = start_state
        self.end_state = end_state
        self.num_dims = num_dims
        self.max_state = max_state
        self.control_space_q = int(start_state.shape[0]/num_dims)

    def get_state(self, t):
        """
        Given a time t, return the state of the motion primitive at that time. Will be specific to the subclass, so we raise an error if the subclass has not implemented it
        """
        raise NotImplementedError

    def plot_from_sampled_states(self, st, sp, sv, sa, sj):
        """
        Plot time vs. position, velocity, acceleration, and jerk (input is already sampled)
        """
        # Plot the state over time.
        fig, axes = plt.subplots(4, 1, sharex=True)
        for i in range(sp.shape[0]):
            for ax, s, l in zip(axes, [sp, sv, sa, sj], ('pos', 'vel', 'acc', 'jerk')):
                if s is not None:
                    ax.plot(st, s[i, :])
                ax.set_ylabel(l)
        axes[3].set_xlabel('time')
        fig.suptitle('Full State over Time')


class PolynomialMotionPrimitive(MotionPrimitive):
    """
    A motion primitive constructed from polynomial coefficients
    """

    def __init__(self, start_state, end_state, num_dims, max_state, x_derivs=None):
        super().__init__(start_state, end_state, num_dims, max_state)  # Run MotionPrimitive's instantiation first
        self.polynomial_constructor(x_derivs)

    def get_state(self, t):
        pass

    def polynomial_constructor(self, x_derivs=None):
        """
        Create the polynomial representation of the motion primitive. 
        Optional input x_derivs is the list of lambda functions evaluating polynomial derivative of state (see setup_bvp_meam_620_style).
        It can be passed in to save on repeated computation.
        """
        if x_derivs is None:
            self.x_derivs = self.setup_bvp_meam_620_style(self.control_space_q)
        self.polys, self.traj_time = self.iteratively_solve_bvp_meam_620_style(
            self.start_state, self.end_state, self.num_dims, self.max_state, self.x_derivs)

    @staticmethod
    def setup_bvp_meam_620_style(control_space_q):
        """
        Create an array of lambda functions that evaluate the derivatives of the univariate polynmial with all 1 coefficient of order (control_space_q)*2-1
        Example for control_space_q = 3 (polynomials of order 5 minimizing jerk)
        x_derivs[0] = lambda t: [t**5, t**4, t**3, t**,2 t, 1]
        x_derivs[1] = lambda t: [5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0]
        x_derivs[2] = lambda t: [20*t**3, 12*t**2, 6*t, 2, 0, 0]
        x_derivs[3] = lambda t: [60*t**2, 24*t, 6, 0, 0, 0]
        Only needs to be computed once for any control_space_q
        These derivatives are used to construct the constraint matrix to solve the 2-point BVP in solve_bvp_meam_620_style
        """
        t = sym.symbols('t')
        poly_order = (control_space_q)*2-1  # why?
        x = np.squeeze(sym.Matrix(np.zeros(poly_order+1)))
        for i in range(poly_order+1):
            x[i] = t**(poly_order-i)  # Construct polynomial of the form [T**5,    T**4,   T**3, T**2, T, 1]

        x_derivs = []
        for i in range(control_space_q+1):
            x_derivs.append(sym.lambdify([t], x))
            x = sym.diff(x)  # iterate through all the derivatives
        return x_derivs

    @staticmethod
    def solve_bvp_meam_620_style(start_state, end_state, num_dims, x_derivs, T):
        """
        Return polynomial coefficients for a trajectory from start_state ((n,) array) to end_state ((n,) array) in time interval [0,T]
        The array of lambda functions created in setup_bvp_meam_620_style and the dimension of the configuration space are also required.
        """
        control_space_q = int(start_state.shape[0]/num_dims)
        poly_order = (control_space_q)*2-1
        A = np.zeros((poly_order+1, poly_order+1))
        for i in range(control_space_q):
            x = x_derivs[i]  # iterate through all the derivatives
            A[i, :] = x(0)  # x(ti) = start_state
            A[control_space_q+i, :] = x(T)  # x(tf) = end_state

        polys = np.zeros((num_dims, poly_order+1))
        b = np.zeros(control_space_q*2)
        for i in range(num_dims):  # Construct a separate polynomial for each dimension

            # vector of the form [start_state,end_state,start_state_dot,end_state_dot,...]
            b[:control_space_q] = start_state[i::num_dims]
            b[control_space_q:] = end_state[i::num_dims]
            poly = np.linalg.solve(A, b)

            polys[i, :] = poly

        return polys

    @staticmethod
    def iteratively_solve_bvp_meam_620_style(start_state, end_states, num_dims, max_state, x_derivs):
        """
        Given a start and goal pt, iterate over solving the BVP until the input constraint is satisfied-ish. TODO: only checking input constraint at start, middle, and end at the moment
        """
        # TODO make parameters
        dt = .2
        max_t = 1
        t = 0
        u_max = np.inf
        polys = None
        control_space_q = int(start_state.shape[0]/num_dims)
        while u_max > max_state[control_space_q]:
            t += dt
            if t > max_t:
                # u = np.ones(self.num_dims)*np.inf
                polys = None
                t = np.inf
                break
            polys = PolynomialMotionPrimitive.solve_bvp_meam_620_style(start_state, end_states, num_dims, x_derivs, t)
            # TODO this is only u(t), not necessarily max(u) from 0 to t which we would want, use critical points maybe?
            u_max = max(abs(np.sum(polys*x_derivs[-1](t), axis=1)))
            u_max = max(u_max, max(abs(np.sum(polys*x_derivs[-1](t/2), axis=1))))
            u_max = max(u_max, max(abs(np.sum(polys*x_derivs[-1](0), axis=1))))
        return polys, t

    def evaluate_polynomial_at_derivative(self, deriv_num, st):
        """
        Use the derivative helper function from setup_bvp_meam_620_style to evaluate all the derivatives of the self.polys polynomial (represented as polynomial coefficients)
        at the st sample times
        Returns a numpy array of size (num_dims x len(st))
        """
        # TODO reuse this into get_state
        # TODO: clean up/document this better
        return np.vstack([np.array([np.polyval(np.pad((self.x_derivs[deriv_num](1)* self.polys[0, :]),((deriv_num),(0)))[:-1], i) for i in st]) for j in range(self.num_dims)])

    def plot(self):
        """
        Generate the sampled state and input trajectories and plot them
        """
        st = np.linspace(0, self.traj_time, 100)
        if not np.isinf(self.traj_time):
            sp = np.vstack([np.array([np.polyval(self.polys[j, :], i) for i in st]) for j in range(self.num_dims)])
            sv = self.evaluate_polynomial_at_derivative(1, st)
            sa = self.evaluate_polynomial_at_derivative(2, st)
            if self.control_space_q >= 3:
                sj = self.evaluate_polynomial_at_derivative(3, st)
            else:
                sj = None
            self.plot_from_sampled_states(st, sp, sv, sa, sj)
        else:
            print("Trajectory was not found")


class JerksMotionPrimitive(MotionPrimitive):
    """
    A motion primitive constructed from a sequence of constant jerks
    """

    def __init__(self, start_state, end_state, num_dims, max_state):
        super().__init__(start_state, end_state, num_dims, max_state)
        assert(self.control_space_q == 3), "This function only works for jerk input space (and maybe acceleration input space one day)"
        self.jerks_constructor()

    def get_state(self, t):
        pass

    def jerks_constructor(self):
        """
        When the constructor is called, solve the min-time two-point BVP given the class parameters
        """
        self.switch_times, self.jerks = self.solve_bvp_min_time(self.start_state, self.end_state, self.num_dims, self.max_state)

    @staticmethod
    def solve_bvp_min_time(start_state, end_state, num_dims, max_state):
        """
        Solve the BVP for time optimal jerk control trajectories as in Beul ICUAS '17 https://github.com/jpaulos/opt_control
        """
        control_space_q = int(start_state.shape[0]/num_dims)
        # start point
        p0, v0, a0 = np.split(start_state, control_space_q)
        # end point
        p1, v1, a1 = np.split(end_state, control_space_q)

        # state and input limits
        v_max, a_max, j_max = max_state[1:1+control_space_q]
        v_min, a_min, j_min = -max_state[1:1+control_space_q]
        # call to optimization library
        (t, j) = min_time_bvp.min_time_bvp(p0, v0, a0, p1, v1, a1, v_min, v_max, a_min,
                                           a_max, j_min, j_max)

        return t, j

    def plot(self):
        """
        Generate the sampled state and input trajectories and plot them
        """
        p0, v0, a0 = np.split(self.start_state, self.control_space_q)
        st, sj, sa, sv, sp = min_time_bvp.sample_min_time_bvp(p0, v0, a0, self.switch_times, self.jerks, dt=0.001)
        self.plot_from_sampled_states(st, sp, sv, sa, sj)


if __name__ == "__main__":
    # mp = PolynomialMotionPrimitive([1, 2, 3, 4, 5])
    _start_state = np.ones((6,))*.1
    _end_state = np.zeros((6,))
    _num_dims = 2
    _max_state = np.ones((6,))*100
    mp = JerksMotionPrimitive(_start_state, _end_state, _num_dims, _max_state)
    mp.plot()

    mp = PolynomialMotionPrimitive(_start_state, _end_state, _num_dims, _max_state)
    mp.plot()

    plt.show()
