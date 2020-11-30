from motion_primitives_py import MotionPrimitive
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.linalg import expm
import scipy.integrate as integrate


class PolynomialMotionPrimitive(MotionPrimitive):
    """
    A motion primitive constructed from polynomial coefficients
    """

    def __init__(self, start_state, end_state, num_dims, max_state,
                 subclass_specific_data={}):
        # Initialize class
        super().__init__(start_state, end_state, num_dims, max_state,
                         subclass_specific_data)
        if "dynamics" not in self.subclass_specific_data:
            self.subclass_specific_data['dynamics'] = self.get_dynamics_polynomials(self.control_space_q)
        # Solve boundary value problem
        self.polys, self.traj_time = self.iteratively_solve_bvp_meam_620_style(
            self.start_state, self.end_state, self.num_dims,
            self.max_state, self.subclass_specific_data['dynamics'], subclass_specific_data.get('iterative_bvp_dt'), subclass_specific_data.get('iterative_bvp_max_t'))
        if self.polys is not None:
            self.is_valid = True
            if self.subclass_specific_data.get('rho') is None:
                self.cost = self.traj_time
            else:
                self.cost = self.traj_time * self.subclass_specific_data['rho']
                st, su = self.get_sampled_input()
                self.cost += np.linalg.norm(np.sum((su)**2 * st, axis=1))

    @classmethod
    def from_dict(cls, dict, num_dims, max_state, subclass_specific_data={}):
        """
        Load a polynomial representation of a motion primitive from a dictionary
        """
        mp = super().from_dict(dict, num_dims, max_state)
        if mp:
            mp.polys = np.array(dict["polys"])
            if "dynamics" in subclass_specific_data:
                mp.subclass_specific_data['dynamics'] = subclass_specific_data['dynamics']
            else:
                mp.subclass_specific_data['dynamics'] = mp.get_dynamics_polynomials(mp.control_space_q)
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
        return np.vstack([self.evaluate_polynomial_at_derivative(i, [t])
                          for i in range(self.control_space_q)])

    def get_sampled_states(self, step_size=0.1):
        # TODO connect w/ get_state
        if self.is_valid:
            st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
            sp = self.evaluate_polynomial_at_derivative(0, st)
            sv = self.evaluate_polynomial_at_derivative(1, st)
            sa = self.evaluate_polynomial_at_derivative(2, st)
            if self.control_space_q >= 3:
                sj = self.evaluate_polynomial_at_derivative(3, st)
            else:
                sj = None
            return st, sp, sv, sa, sj
        else:
            return None, None, None, None, None

    def get_sampled_position(self, step_size=0.1):
        if self.is_valid:
            st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
            sp = self.evaluate_polynomial_at_derivative(0, st)
            return st, sp
        else:
            return None, None

    def get_sampled_input(self, step_size=0.1):
        if self.is_valid:
            st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time / step_size) + 1))
            su = self.evaluate_polynomial_at_derivative(self.control_space_q, st)
            return st, su
        else:
            return None, None

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
        return self.evaluate_polynomial_at_derivative_static(deriv_num, st, self.subclass_specific_data['dynamics'], self.polys, self.num_dims)

    @staticmethod
    def evaluate_polynomial_at_derivative_static(deriv_num, st, dynamics, polys, num_dims):
        """
        Sample the specified derivative number of the polynomial trajectory at
        the specified times
        Input:
            deriv_num, order of derivative, scalar
            st, numpy array of times to sample
        Output:
            sampled, array of polynomial derivative evaluated at sample times
        """
        if deriv_num > -1:
            poly_coeffs = np.array([np.concatenate((np.zeros(deriv_num), dynamics[deriv_num](1) * polys[j, :]))[:polys.shape[1]]
                                    for j in range(num_dims)]).T  # TODO maybe can move to precompute in general and then just multiply by polys
        else:
            poly_coeffs = polys.T
        sampled = np.array([np.dot(dynamics[0](t),poly_coeffs) for t in st]).T
        return sampled

    @staticmethod
    def get_dynamics_polynomials(control_space_q):
        """
        Returns an array of lambda functions that evaluate the derivatives of
        a polynomial of specified order with coefficients all set to 1

        Example for polynomial order 5:
        time_derivatives[0] = lambda t: [t**5, t**4, t**3, t**2, t, 1]
        time_derivatives[1] = lambda t: [5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0]
        time_derivatives[2] = lambda t: [20*t**3, 12*t**2, 6*t, 2, 0, 0]
        time_derivatives[3] = lambda t: [60*t**2, 24*t, 6, 0, 0, 0]

        Input:
            control_space_q, derivative of configuration which is control input
                infer order from this using equation: 2 * control_space_q - 1,
                number of derivatives returned will be control_space_q + 1

        Output:
            time_derivatives, an array of length (control_space_q + 1)
                represents the time derivatives of the specified polynomial with
                the ith element of the array representing the ith derivative
        """
        # construct polynomial of the form [T**5, T**4, T**3, T**2, T, 1]
        order = 2 * control_space_q - 1
        t = sym.symbols('t')
        x = np.squeeze(sym.Matrix([t**(order - i) for i in range(order + 1)]))

        # iterate through relevant derivatives and make function for each
        time_derivatives = []
        for i in range(max(control_space_q + 2, order)):
            time_derivatives.append(sym.lambdify([t], x))
            x = sym.derive_by_array(x, t)
        return time_derivatives

    @staticmethod
    def solve_bvp_meam_620_style(start_state, end_state, num_dims, dynamics, T):
        """
        Return polynomial coefficients for a trajectory from start_state ((n,) array) to end_state ((n,) array) in time interval [0,T]
        The array of lambda functions created in get_dynamics_polynomials and the dimension of the configuration space are also required.
        """
        control_space_q = int(start_state.shape[0]/num_dims)
        poly_order = (control_space_q)*2-1
        A = np.zeros((poly_order+1, poly_order+1))
        for i in range(control_space_q):
            x = dynamics[i]  # iterate through all the derivatives
            A[i, :] = x(0)  # x(ti) = start_state
            A[control_space_q+i, :] = x(T)  # x(tf) = end_state

        polys = np.zeros((num_dims, poly_order+1))
        b = np.zeros(control_space_q*2)
        for i in range(num_dims):  # Construct a separate polynomial for each dimension

            # vector of the form [start_state,end_state,start_state_dot,end_state_dot,...]
            b[: control_space_q] = start_state[i:: num_dims]
            b[control_space_q:] = end_state[i:: num_dims]
            poly = np.linalg.solve(A, b)

            polys[i, :] = poly

        return polys

    @staticmethod
    def iteratively_solve_bvp_meam_620_style(start_state, end_states, num_dims, max_state, dynamics, dt, max_t):
        """
        Given a start and goal pt, iterate over solving the BVP until the input constraint is satisfied-ish.
        """
        def check_max_state_and_input():
            critical_pts = np.zeros(polys.shape[1] + 2)
            critical_pts[:2] = [0, t]
            for k in range(1, control_space_q+1):
                u_max = 0
                for i in range(num_dims):
                    roots = np.roots((polys*dynamics[k + 1](1))[i, :])
                    roots = roots[np.isreal(roots)]
                    critical_pts[2:2+roots.shape[0]] = roots
                    critical_pts[2+roots.shape[0]:] = 0
                    critical_us = PolynomialMotionPrimitive.evaluate_polynomial_at_derivative_static(
                        k, critical_pts, dynamics, polys, num_dims)
                    u_max = max(u_max, np.max(np.abs(critical_us)))
                if u_max > max_state[k]:
                    return False
            return True

        if dt == None:
            dt = .1
        if max_t == None:
            max_t = 10

        t = 0
        polys = None
        control_space_q = int(start_state.shape[0]/num_dims)
        done = False
        while not done:
            t += dt + float(np.random.rand(1)*dt/5.)
            if t > max_t:
                polys = None
                t = np.inf
                break
            polys = PolynomialMotionPrimitive.solve_bvp_meam_620_style(start_state, end_states, num_dims, dynamics, t)
            done = check_max_state_and_input()
        return polys, t

    def A_and_B_matrices_quadrotor(self):
        """
        Generate constant A, B matrices for integrator of order control_space_q
        in configuration dimension num_dims. Linear approximation of quadrotor dynamics
        that work (because differential flatness or something)
        """
        # TODO possibly delete
        num_dims = self.num_dims
        control_space_q = self.control_space_q
        n = num_dims*control_space_q

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
        # TODO possibly delete
        return expm(self.A*(dt-sigma))@self.B

    def quad_dynamics_integral_wrapper(self, dt):
        # TODO possibly delete
        def fn(sigma): return self.quad_dynamics_integral(sigma, dt)
        return fn

    def quad_dynamics(self, start_pt, u, dt):
        """
        Computes the state transition map given a starting state, control u, and
        time dt. Slow because of integration and exponentiation
        """
        # TODO possibly delete
        return expm(self.A*dt)@start_pt + integrate.quad_vec(self.quad_dynamics_integral_wrapper(dt), 0, dt)[0]@u


if __name__ == "__main__":
    # problem parameters
    num_dims = 2
    control_space_q = 3

    # setup problem
    start_state = np.zeros((num_dims * control_space_q,))
    # end_state = np.random.rand(num_dims * control_space_q,)
    end_state = np.ones_like(start_state)
    end_state[0] = 2
    max_state = 1 * np.ones((control_space_q+1,))

    # polynomial
    mp = PolynomialMotionPrimitive(start_state, end_state, num_dims, max_state, {'rho':1})

    # save
    assert(mp.is_valid)
    assert(np.array_equal(mp.end_state, end_state))
    print(mp.cost)
    dictionary = mp.to_dict()

    # reconstruct
    mp = PolynomialMotionPrimitive.from_dict(dictionary, num_dims, max_state)

    # plot
    st, sp, sv, sa, sj = mp.get_sampled_states()
    mp.plot_from_sampled_states(st, sp, sv, sa, sj)
    plt.show()
