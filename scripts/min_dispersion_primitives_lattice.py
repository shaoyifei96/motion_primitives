from motion_primitive_graph import *
from py_opt_control import min_time_bvp
from motion_primitive import *


class MotionPrimitiveLattice(MotionPrimitiveGraph):
    """
    A class that provides functions to compute a lattice of minimum dispersion points in the state space connected by feasible trajectories
    """

    def dispersion_distance_fn_time(self, potential_sample_pts, start_pt):
        """
        A function that evaluates the cost of a path from start_pt to an array of potential_sample_pts. For the moment the cost is the time of the optimal path.
        """
        score = np.zeros(potential_sample_pts.shape[0])
        for i in range(potential_sample_pts.shape[0]):
            polys, t = self.iteratively_solve_bvp_meam_620_style(start_pt, potential_sample_pts[i, :])
            score[i] = t  # + np.linalg.norm(u)*.0001  # tie break w/ u?
        return score

    def compute_min_dispersion_space(self, num_output_pts=250, resolution=[0.2, 0.2, 0.2]):
        """
        Using the bounds on the state space, compute a set of minimum dispersion points
        (Similar to original Dispertio paper)
        """
        # self.dispersion_distance_fn = self.dispersion_distance_fn_time

        bounds = np.vstack((-self.max_state[:self.control_space_q], self.max_state[:self.control_space_q])).T
        potential_sample_pts = self.uniform_state_set(bounds, resolution[:self.control_space_q])
        print(potential_sample_pts.shape)
        score = np.ones((potential_sample_pts.shape[0], num_output_pts))*np.inf
        starting_output_sample_index = 0
        score[:, 0] = self.dispersion_distance_fn(potential_sample_pts, potential_sample_pts[starting_output_sample_index, :])
        actual_sample_pts, actual_sample_indices = self.compute_min_dispersion_points(num_output_pts,
                                                                                      potential_sample_pts, score, starting_output_sample_index)
        print(actual_sample_pts)
        self.reconnect_lattice(actual_sample_pts)
        if self.plot:
            if self.num_dims == 2:
                plt.plot(actual_sample_pts[:, 0], actual_sample_pts[:, 1], 'om')
            if self.num_dims == 3:
                plt.plot(actual_sample_pts[:, 0], actual_sample_pts[:, 1], actual_sample_pts[:, 2], 'om')

        return actual_sample_pts

    def reconnect_lattice(self, sample_pts):
        """
        Given a set of min dispersion sample points, connect each point to each other via solving BVPs. TODO: limit the number of connections for each point
        """
        print("reconnect lattice")
        # self.start_pts_set = sample_pts
        # self.motion_primitives_list = []
        for start_pt in sample_pts:
            for end_pt in sample_pts:
                if (start_pt == end_pt).all():
                    continue
                mp = PolynomialMotionPrimitive(start_pt, end_pt, self.num_dims, self.max_state)
                # self.motion_primitives_list.append()
                # TODO enforce a max number of connections
                # TODO save and output to pickle
                if self.plot:
                    if ~np.isinf(mp.traj_time):
                        t_list = np.linspace(0, mp.traj_time, 30)
                        x = [np.polyval(mp.polys[0, :], i) for i in t_list]
                        y = [np.polyval(mp.polys[1, :], i) for i in t_list]
                        if self.num_dims == 2:
                            plt.plot(x, y)
                        if self.num_dims == 3:
                            z = [np.polyval(mp.polys[2, :], i) for i in t_list]
                            plt.plot(x, y, z)

    def solve_bvp_meam_620_style(self, xi, xf, T):
        """
        Return polynomial coefficients for a trajectory from xi ((n,) array) to xf ((n,) array) in time interval [0,T]
        """
        # TODO might want to do quadratic program instead to actually enforce constraints. Then probably don't need iteratively solve BVP function
        A = np.zeros((self.poly_order+1, self.poly_order+1))
        for i in range(self.control_space_q):
            x = self.x_derivs[i]  # iterate through all the derivatives
            A[i, :] = x(0)  # x(ti) = xi
            A[self.control_space_q+i, :] = x(T)  # x(tf) = xf

        polys = np.zeros((self.num_dims, self.poly_order+1))
        b = np.zeros(self.control_space_q*2)
        for i in range(self.num_dims):  # Construct a separate polynomial for each dimension

            # vector of the form [xi,xf,xi_dot,xf_dot,...]
            b[:self.control_space_q] = xi[i::self.num_dims]
            b[self.control_space_q:] = xf[i::self.num_dims]
            poly = np.linalg.solve(A, b)

            polys[i, :] = poly
        return polys

    def iteratively_solve_bvp_meam_620_style(self, start_pt, goal_pt):
        """
        Given a start and goal pt, iterate over solving the BVP until the input constraint is satisfied. TODO: only checking input constraint at start and end at the moment
        """
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
            # TODO this is only u(t), not necessarily max(u) from 0 to t which we would want, use critical points maybe?
            u_max = max(abs(np.sum(polys*self.x_derivs[-1](t), axis=1)))
            u_max = max(abs(np.sum(polys*self.x_derivs[-1](t/2), axis=1)))
            u_max = max(u_max, max(abs(np.sum(polys*self.x_derivs[-1](0), axis=1))))
        # print(u_max,polys,t)
        return polys, t


if __name__ == "__main__":
    control_space_q = 2
    num_dims = 2
    max_state = [1, 1, 10, 100, 1, 1]
    plot = True
    mp = MotionPrimitiveLattice(control_space_q=control_space_q, num_dims=num_dims, max_state=max_state, plot=plot)
    start_pt = np.ones((mp.n))
    # start_pt = np.array([-1., -2., 0, 0.5])
    # print(mp.solve_bvp_meam_620_style(start_pt, start_pt*2, 1))
    # print(mp.iteratively_solve_bvp_meam_620_style(start_pt,start_pt*2))

    # with PyCallGraph(output=GraphvizOutput(), config=Config(max_depth=6)):
    mp.compute_min_dispersion_space(num_output_pts=100, resolution=[.2, .2, .2, 1, 1, 1])

    if mp.plot:
        plt.show()
