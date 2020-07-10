from motion_primitive_graph import *


class MotionPrimitiveLattice(MotionPrimitiveGraph):
    """
    A class that provides functions to compute a lattice of minimum dispersion points in the state space connected by feasible trajectories
    """

    def dispersion_distance_fn_trajectory(self, start_pts, end_pts):
        """
        A function that evaluates the cost of a path from an array of start_pts to an array of end_pts. For the moment the cost is the time of the optimal path.
        """
        assert(start_pts.shape[0] == 1 or end_pts.shape[0] == 1), "Either start_pts or end_pts must be only one point"
        num_pts = np.max([start_pts.shape[0], end_pts.shape[0]])

        score = -np.ones(num_pts)*np.inf
        mp_list = np.empty(num_pts, dtype=object)
        for i in range(start_pts.shape[0]):
            for j in range(end_pts.shape[0]):
                if (start_pts[i, :] == end_pts[j, :]).all():
                    continue
                # mp = JerksMotionPrimitive(start_pts[i, :], end_pts[j, :], self.num_dims, self.max_state)
                mp = PolynomialMotionPrimitive(start_pts[i, :], end_pts[j, :], self.num_dims, self.max_state, {'x_derivs': self.x_derivs})
                # TODO pass motion primitive class type around
                ind = max(i, j)
                mp_list[ind] = mp
                if mp.is_valid:
                    score[ind] = mp.cost  # + np.linalg.norm(u)*.0001  # tie break w/ u?
        return score, mp_list

    def compute_min_dispersion_points(self, num_output_pts, potential_sample_pts):
        # TODO will pick the same point if all MPs fail
        # overloaded from motion_primitive_graph for the moment # TODO maybe unify with original version used in tree
        mp_adjacency_matrix = np.empty((num_output_pts, potential_sample_pts.shape[0]), dtype=object)
        score = np.ones((potential_sample_pts.shape[0], num_output_pts))*np.inf
        score[:, 0], mp_list = self.dispersion_distance_fn(potential_sample_pts, potential_sample_pts[0, :][np.newaxis, :])
        actual_sample_indices = np.zeros((num_output_pts)).astype(int)
        actual_sample_indices[0] = 0
        mp_adjacency_matrix[0, :] = mp_list

        # distances of potential sample points to closest chosen output MP node # bottleneck
        min_score = np.ones((score.shape[0], 2))*np.inf
        min_score[:, 0] = np.amin(score, axis=1)
        for sample_pt_num in range(1, num_output_pts):  # start at 1 because we already chose the closest point as a motion primitive
            print(f"{sample_pt_num}/{num_output_pts}")
            # distances of potential sample points to closest chosen output MP node
            min_score[:, 0] = np.amin(min_score, axis=1)
            # take the new point with the maximum distance to its closest node
            index = np.argmax(min_score[:, 0])
            result_pt = potential_sample_pts[index, :]
            actual_sample_indices[sample_pt_num] = np.array((index))
            min_score[index, 0] = - np.inf  # give nodes we have already chosen low score
            min_score[:, 1], mp_list = self.dispersion_distance_fn(potential_sample_pts, result_pt[np.newaxis, :])  # new point's score
            # min_score[:, 1], mp_list = self.dispersion_distance_fn(result_pt[np.newaxis, :],potential_sample_pts)  # new point's score
            mp_adjacency_matrix[sample_pt_num, :] = mp_list
        actual_sample_pts = potential_sample_pts[actual_sample_indices]
        return actual_sample_pts, actual_sample_indices, mp_adjacency_matrix

    def compute_min_dispersion_space(self, num_output_pts=250, resolution=[0.2, 0.2, 0.2]):
        """
        Using the bounds on the state space, compute a set of minimum dispersion points
        (Similar to original Dispertio paper)
        """
        self.dispersion_distance_fn = self.dispersion_distance_fn_trajectory

        bounds = np.vstack((-self.max_state[:self.control_space_q], self.max_state[:self.control_space_q])).T
        potential_sample_pts = self.uniform_state_set(bounds, resolution[:self.control_space_q], random=True)
        print(potential_sample_pts.shape)
        actual_sample_pts, actual_sample_indices, mp_adjacency_matrix = self.compute_min_dispersion_points(
            num_output_pts, potential_sample_pts)
        if self.plot:
            if self.num_dims == 2:
                plt.plot(actual_sample_pts[:, 0], actual_sample_pts[:, 1], 'om')
            if self.num_dims == 3:
                plt.plot(actual_sample_pts[:, 0], actual_sample_pts[:, 1], actual_sample_pts[:, 2], 'om')
        self.sample_pts = actual_sample_pts  # TODO pass these around functionally instead of parametrically
        return actual_sample_pts, mp_adjacency_matrix #mp_adjacency_matrix[:, actual_sample_indices]

    def connect_lattice(self, sample_pts, adjacency_matrix):
        """
        Given a set of min dispersion sample points, connect each point to each other via solving BVPs. TODO: limit the number of connections for each point
        """
        cost_threshold = np.inf
        for i in range(sample_pts.shape[0]):
            for j in range(sample_pts.shape[0]):
                mp = adjacency_matrix[i, j]
                if mp is not None and mp.is_valid and mp.cost < cost_threshold:
                    self.motion_primitives_list.append(mp)
                    if self.plot:
                        st, sp, sv, sa, sj = mp.get_sampled_states() 
                        if self.num_dims == 2:
                            plt.plot(sp[0, :], sp[1, :])
                            plt.plot(sample_pts[:, 0], sample_pts[:, 1], 'om')
                        elif self.num_dims == 3:
                            plt.plot(sp[0, :], sp[1, :], sp[2, :])
                            plt.plot(sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2], 'om')

    def tile_points(self, pts):
        """
        function that takes in an array of points and returns those points tiled in position space
        """
        bounds = 2* np.array([0, -self.max_state[0], self.max_state[0]])
        tiled_pts = np.array([pts for i in range(3 ** self.num_dims)])
        if self.num_dims == 2:
            offsets = itertools.product(bounds, bounds)
        elif self.num_dims == 3:
            offsets = itertools.product(bounds, bounds, bounds)
        for i, offset in enumerate(offsets):
            tiled_pts[i, :, :self.num_dims] += offset
        return tiled_pts.reshape(len(pts) * 3 ** self.num_dims, 
                                 self.num_dims * self.control_space_q)


if __name__ == "__main__":
    # define parameters
    control_space_q = 2
    num_dims = 2
    max_state = [1, .1, 100, 100, 1, 1]  # .5 m/s max velocity 14 m/s^2 max acceleration
    plot = True

    # build lattice
    mps = MotionPrimitiveLattice(control_space_q=control_space_q, num_dims=num_dims, max_state=max_state, plot=plot)
    # with PyCallGraph(output=GraphvizOutput(), config=Config(max_depth=8)):
    V, E = mps.compute_min_dispersion_space(num_output_pts=50, resolution=[.2, .1, .1, 25, 1, 1])
    mps.connect_lattice(V, E)

    # plot
    if mps.plot:
        plt.show()
