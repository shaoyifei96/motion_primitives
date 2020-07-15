from motion_primitives_py.motion_primitive import *
import sympy as sym


class PolynomialMotionPrimitive(MotionPrimitive):
    """
    A motion primitive constructed from polynomial coefficients
    """

    def __init__(self, start_state, end_state, num_dims, max_state, subclass_specific_data={}):
        # Initialize class
        super().__init__(start_state, end_state, num_dims, max_state, subclass_specific_data)
        # TODO: figure out another way to accomplish this - code is duplicated
        if self.subclass_specific_data.get('x_derivs') is None:
            self.x_derivs = self.setup_bvp_meam_620_style(self.control_space_q)
        else:
            self.x_derivs = self.subclass_specific_data.get('x_derivs')

        # Solve boundary value problem
        self.polys, traj_time = self.iteratively_solve_bvp_meam_620_style(
            self.start_state, self.end_state, self.num_dims, self.max_state, self.x_derivs)
        if self.polys is not None:
            self.is_valid = True
            self.cost = traj_time

    @classmethod
    def from_dict(cls, dict, num_dims, max_state):
        """
        Load a polynomial representation of the motion primitive from a dictionary 
        """
        mp = super(PolynomialMotionPrimitive, cls).from_dict(dict, num_dims, max_state)
        if mp:
            mp.polys = np.array(dict["polys"])
            # TODO: figure out another way to accomplish this - code is duplicated
            if mp.subclass_specific_data.get('x_derivs') is None:
                mp.x_derivs = mp.setup_bvp_meam_620_style(mp.control_space_q)
            else:
                mp.x_derivs = mp.subclass_specific_data.get('x_derivs')
        return mp

    def to_dict(self):
        """
        Write important attributes of motion primitive to a dictionary
        """
        dict = super().to_dict()
        if dict:
            dict["polys"] = self.polys.tolist()
        return dict

    def get_state(self, t):
        """
        Evaluate full state of a trajectory at a given time
        Input:
            t, numpy array of times to sample at
        Return:
            state, a numpy array of size (num_dims x control_space_q, len(t))
        """
        return np.vstack([self.evaluate_polynomial_at_derivative(i, [t]) for i in range(self.control_space_q)])

    def get_sampled_states(self):
        # TODO connect w/ get_state
        if self.is_valid:
            st = np.linspace(0, self.cost, 100)
            sp = np.vstack([np.array([np.polyval(self.polys[j, :], i) for i in st]) for j in range(self.num_dims)])
            sv = self.evaluate_polynomial_at_derivative(1, st)
            sa = self.evaluate_polynomial_at_derivative(2, st)
            if self.control_space_q >= 3:
                sj = self.evaluate_polynomial_at_derivative(3, st)
            else:
                sj = None
            return st, sp, sv, sa, sj
        else:
            return None, None, None, None, None

    def evaluate_polynomial_at_derivative(self, deriv_num, st):
        """
        Sample the specified derivative number of the polynomial trajectory at
        the specified times
        Input:
            deriv_num, order of derivative, scalar
            st, numpy array of times to sample
        Output:
            sampled, array of polynomial derivative evaluated at sample times
        """

        sampled = np.vstack([np.array([np.polyval(np.pad((self.x_derivs[deriv_num](1) * self.polys[j, :]),
                                                         ((deriv_num), (0)))[:self.polys.shape[1]], i) for i in st]) for j in range(self.num_dims)])

        return sampled

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
        max_t = 2
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


if __name__ == "__main__":
    # problem parameters
    num_dims = 2
    control_space_q = 3

    # setup problem
    start_state = np.zeros((num_dims * control_space_q,))
    end_state = np.random.rand(num_dims * control_space_q,)
    max_state = np.ones((num_dims * control_space_q,))*100

    # polynomial
    mp2 = PolynomialMotionPrimitive(start_state, end_state, num_dims, max_state)

    # save
    assert(mp2.is_valid)
    dictionary2 = mp2.to_dict()

    # reconstruct
    mp2 = PolynomialMotionPrimitive.from_dict(dictionary2, num_dims, max_state)

    # plot
    st, sp, sv, sa, sj = mp2.get_sampled_states()
    mp2.plot_from_sampled_states(st, sp, sv, sa, sj)
    plt.show()
