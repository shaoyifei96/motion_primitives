from motion_primitives_py import MotionPrimitive, PolynomialMotionPrimitive
import numpy as np
import cvxpy as cvx
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


class OptimizationMotionPrimitive(PolynomialMotionPrimitive):
    """
    Find the optimal motion primitive for an n-integrator LTI system, with cost  $\sum_{t} {(||u||^2 + \rho) * t}$, given state and input constraints.
    """

    def __init__(self, start_state, end_state, num_dims, max_state,
                 subclass_specific_data={}):
        MotionPrimitive.__init__(self, start_state, end_state, num_dims, max_state,
                                 subclass_specific_data)
        self.rho = subclass_specific_data.get('rho', 1)  # multiplier on time in cost function
        self.c_A, self.c_B = OptimizationMotionPrimitive.A_and_B_matrices_quadrotor(self.num_dims, self.control_space_q)
        self.steps = 10  # number of time steps in inner_bvp
        self.x_box = np.repeat(self.max_state[:self.control_space_q], self.num_dims)
        self.u_box = np.repeat(self.max_state[self.control_space_q], self.num_dims)

        self.is_valid = False
        self.outer_bvp()
        self.polynomial_setup(self.poly_coeffs.shape[1]-1)

    @staticmethod
    def A_and_B_matrices_quadrotor(num_dims, control_space_q):
        """
        Generate constant A, B matrices for continuous integrator of order control_space_q
        in configuration dimension num_dims.
        """
        n = num_dims*control_space_q
        A = np.zeros((n, n))
        B = np.zeros((n, num_dims))
        for p in range(1, control_space_q):
            A[(p-1)*num_dims: (p-1)*num_dims+num_dims, p*num_dims: p*num_dims+num_dims] = np.eye(num_dims)
        B[-num_dims:, :] = np.eye(num_dims)
        return A, B

    def outer_bvp(self):
        """
        Given a function inner_bvp that finds an optimal motion primitive given a time allocation, finds the optimal time allocation.
        """
        # exponential search to find *a* feasible dt
        dt_start = 1e-2
        while self.inner_bvp(dt_start) == np.inf:
            dt_start *= 10
        # binary search to find a better minimum starting bound on dt
        begin = dt_start/10
        end = dt_start
        tolerance = 1e-2
        while (end-begin)/2 > tolerance:
            mid = (begin + end)/2
            if self.inner_bvp(mid) == np.inf:
                begin = mid
            else:
                end = mid
        dt_start = end

        # optimization problem with lower bound on dt (since below this it is infeasible). Upper bound is arbitrary
        sol = minimize_scalar(self.inner_bvp, bounds=[dt_start, 2], method='bounded')
        self.optimal_dt = sol.x
        self.is_valid = sol.success
        if not self.is_valid:
            print("Did not find solution to outer BVP")
        else:
            self.cost, self.traj, self.inputs = self.inner_bvp(self.optimal_dt, return_traj=True)
            self.traj_time = self.optimal_dt * self.steps
            time_vec = np.linspace(0, self.optimal_dt*(self.steps), self.traj.shape[0])
            self.poly_coeffs = np.polyfit(time_vec, self.traj[:, :self.num_dims], self.n).T  # TODO what degree polynomial
        return self.is_valid

    def inner_bvp(self, dt, return_traj=False):
        """
        Computes an optimal MP between a start and goal state, under bounding box constraints with a given time interval (dt) allocation.
        Accomplishes this by constraining x(t) and u(t) at discrete steps to obey the input, state, and dynamic constraints.
        Note that it is parameterized by time step size (dt), with a fixed number of time steps set in the constructor.
        """
        if dt <= 0:  # Don't allow negative time (outer_bvp may try to do this)
            return np.inf

        # Transform a continuous to a discrete state-space system, given dt
        sysd = cont2discrete((self.c_A, self.c_B, np.eye(self.n), 0), dt)
        A = sysd[0]
        B = sysd[1]

        cost = 0  # initializing cost
        state_variables = []
        state_constraints = []
        input_variables = []
        input_constraints = []
        dynamic_constraints = []
        R = np.eye(self.num_dims)

        x0_var = cvx.Variable(self.start_state.shape)
        # obey starting condition
        state_constraints.append(x0_var == self.start_state)
        state_variables.append(x0_var)
        for _ in range(self.steps):
            xt = cvx.Variable(self.start_state.shape)
            ut = cvx.Variable(R.shape[-1])

            # make this obey dynamics and box constraints
            dynamic_constraints.append(xt == A @ state_variables[-1] + B @ ut)
            state_constraints += [xt >= -self.x_box, xt <= self.x_box]
            input_constraints += [ut >= -self.u_box, ut <= self.u_box]

            # add these to state variables and input variables so we can extract them later
            state_variables.append(xt)
            input_variables.append(ut)
            cost += cvx.quad_form(ut, R*dt) + self.rho*dt  # $\sum_{t} {(||u||^2 + \rho) * t}$

        # obey terminal condition
        state_constraints.append(state_variables[-1] == self.end_state)

        objective = cvx.Minimize(cost)
        constraints = state_constraints + dynamic_constraints + input_constraints
        prob = cvx.Problem(objective, constraints)
        total_cost = prob.solve()
        # print("Solution is {}".format(prob.status))
        if return_traj:
            try:
                trajectory = np.array([state.value for state in state_variables])
                inputs = np.array([control.value for control in input_variables])
                return total_cost, trajectory, inputs
            except:
                print("No trajectory to return")
                pass
        return total_cost

    def plot_inner_bvp_sweep_t(self):
        plt.figure()
        data = []
        for t in np.arange(0.01, 1, .03):
            cost = self.inner_bvp(t)
            data.append((t, cost))
        data = np.array(data)
        plt.plot(data[:, 0]*(self.steps), data[:, 1], 'bo', label="Feasible inner_bvp solutions")
        infeasible = data[data[:, 1] == np.inf, 0]*self.steps
        plt.plot(infeasible, np.zeros_like(infeasible), 'ro', label="Infeasible inner_bvp at these times")
        plt.xlabel("Trajectory Time")
        plt.ylabel(r"Cost $\sum_{t} {(||u||^2 + \rho) * t}$")
        if self.is_valid:
            plt.plot(self.optimal_dt*self.steps, self.cost, '*m', markersize=10, label="outer_bvp optimum")
        plt.legend()

    def plot_outer_bvp_x(self):
        if self.is_valid:
            fig, ax = plt.subplots(self.n + self.num_dims, sharex=True)
            time_vec = np.linspace(0, self.optimal_dt*(self.steps), self.traj.shape[0])
            for i in range(self.n):
                ax[i].plot(time_vec, self.traj[:, i], label="trajectory")
                ax[i].plot(0, self.start_state[i], 'og', label="goal")
                ax[i].plot(time_vec[-1], self.end_state[i], 'or', label="start")
                ax[i].set_ylabel(fr"$x_{i}$")
                st = np.linspace(0, self.optimal_dt*(self.steps), 100)
                deriv_num = int(np.floor(i/self.num_dims))
                p = self.evaluate_polynomial_at_derivative(deriv_num, st)[i % self.num_dims]
                thresholded = np.maximum(np.minimum(p, self.max_state[deriv_num]), -self.max_state[deriv_num])
                ax[i].plot(st, thresholded)
            time_vec = np.linspace(0, self.optimal_dt*(self.steps-1), self.inputs.shape[0])
            for i in range(self.num_dims):
                ax[self.n+i].plot(time_vec, self.inputs[:, i])
                ax[self.n + i].set_ylabel(fr"$u_{i}$")
                st = np.linspace(0, self.optimal_dt*(self.steps-1), 100)
                p = self.evaluate_polynomial_at_derivative(self.control_space_q, st)[i % self.num_dims]
                thresholded = np.maximum(np.minimum(p, self.max_state[deriv_num]), -self.max_state[deriv_num])
                ax[self.n+i].plot(st, thresholded)

        ax[-1].set_xlabel("Trajectory time [s]")

    def plot_outer_bvp_sweep_rho(self):
        self.rho = .01
        fig, ax = plt.subplots(self.n + self.num_dims)
        for _ in range(5):
            self.rho *= 10
            if self.outer_bvp():
                time_vec = np.linspace(0, self.optimal_dt*(self.steps-1), self.inputs.shape[0])
                for i in range(self.num_dims):
                    ax[i].plot(time_vec, self.inputs[:, i], label=rf"$\rho =$ {self.rho}")
                    ax[i].set_ylabel(fr"$u_{i}$")
                time_vec = np.linspace(0, self.optimal_dt*(self.steps), self.traj.shape[0])
                for i in range(self.n):
                    ax[i + self.num_dims].plot(time_vec, self.traj[:, i], label=rf"$\rho =$ {self.rho}")
                    ax[i + self.num_dims].set_ylabel(fr"$x_{i}$")
        ax[-1].set_xlabel("Trajectory time [s]")
        plt.legend(loc="center right")


if __name__ == "__main__":
    num_dims = 2
    control_space_q = 2
    rho = 1e2

    start_state = np.array([5.3, .1, 2, -1])  # initial state
    end_state = np.zeros(num_dims*control_space_q)  # terminal state
    max_state = [np.inf, 2, 1, 10, 10]
    subclass_specific_data = {'rho': rho}

    print('initial')
    mp = OptimizationMotionPrimitive(start_state, end_state, num_dims, max_state,
                                     subclass_specific_data=subclass_specific_data)
    dictionary = mp.to_dict()
    # reconstruct
    print('reconstruct')
    mp = OptimizationMotionPrimitive.from_dict(dictionary, num_dims, max_state)

    # mp.plot_inner_bvp_sweep_t()
    # mp.plot_outer_bvp_x()
    # q.plot_outer_bvp_sweep_rho()

    # mp.plot()
    plt.show()
