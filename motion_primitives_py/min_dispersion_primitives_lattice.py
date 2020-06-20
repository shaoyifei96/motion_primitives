from motion_primitive_graph import *


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
            polys, t = PolynomialMotionPrimitive.iteratively_solve_bvp_meam_620_style(
                start_pt, potential_sample_pts[i, :], self.num_dims, self.max_state, self.x_derivs)

            # t, j = JerksMotionPrimitive.solve_bvp_min_time(start_pt,potential_sample_pts[i,:],self.num_dims,self.max_state)
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
                # mp = JerksMotionPrimitive(start_pt, end_pt, self.num_dims, self.max_state)
                # self.motion_primitives_list.append()
                # TODO enforce a max number of connections
                # TODO save and output to pickle
                if self.plot:
                    st, sp, sv, sa, sj = mp.get_sampled_states()
                    if st is not None:
                        if self.num_dims == 2:
                            plt.plot(sp[0, :], sp[1, :])
                        if self.num_dims == 3:
                            plt.plot(sp[0, :], sp[1, :], sp[2, :])


if __name__ == "__main__":
    control_space_q = 3
    num_dims = 2
    max_state = [1, 1, 1, 100, 1, 1]
    plot = True
    mp = MotionPrimitiveLattice(control_space_q=control_space_q, num_dims=num_dims, max_state=max_state, plot=plot)
    # start_pt = np.ones((mp.n))
    # start_pt = np.array([-1., -2., 0, 0.5])
    # print(mp.solve_bvp_meam_620_style(start_pt, start_pt*2, 1))
    # print(mp.iteratively_solve_bvp_meam_620_style(start_pt,start_pt*2))

    # with PyCallGraph(output=GraphvizOutput(), config=Config(max_depth=6)):
    mp.compute_min_dispersion_space(num_output_pts=10, resolution=[.5, .5, .5, 1, 1, 1])

    if mp.plot:
        plt.show()
