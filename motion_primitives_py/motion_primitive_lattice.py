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
        if self.plot:
            if self.num_dims == 2:
                plt.plot(actual_sample_pts[:, 0], actual_sample_pts[:, 1], 'om')
            if self.num_dims == 3:
                plt.plot(actual_sample_pts[:, 0], actual_sample_pts[:, 1], actual_sample_pts[:, 2], 'om')
        self.sample_pts = actual_sample_pts # TODO pass these around functionally instead of parametrically
        return actual_sample_pts

    def connect_lattice(self, k):
        """
        Given a set of min dispersion sample points, connect each point to each other via solving BVPs. TODO: limit the number of connections for each point
        """
        tiled_pts = self.tile_lattice()
        if self.plot:
            if self.num_dims == 2:
                plt.plot(tiled_pts[:, 0], tiled_pts[:, 1], 'om')
            if self.num_dims == 3:
                plt.plot(tiled_pts[:, 0], tiled_pts[:, 1], tiled_pts[:, 2], 'om')
        self.near_neighbors = k
        # TODO determine which algorithm is best for this application
        # need a better metric to compare distances
        neighbors = NearestNeighbors(n_neighbors=len(self.sample_pts), algorithm='auto').fit(tiled_pts)
        # loop through starting points
        for i in range(len(self.sample_pts)):
            valid_mps = 0

            # loop through neighbors in order of which are best (excludes self)
            for j in neighbors.kneighbors(self.sample_pts)[1][i][1:]:
                # check if max k connections have been acheived
                if valid_mps == k:
                    continue

                # create the motion primitive and check validity
                mp = PolynomialMotionPrimitive(tiled_pts[i], tiled_pts[j], self.num_dims, self.max_state)
                # mp = JerksMotionPrimitive(start_pt, end_pt, self.num_dims, self.max_state)
                st, sp, sv, sa, sj = mp.get_sampled_states()
                if st is None:
                    continue

                # add it to the list of valid mps
                self.motion_primitives_list.append(mp)
                valid_mps += 1

                # handle plotting of the motion primitive
                if self.plot:

                    if self.num_dims == 2:
                        plt.plot(sp[0, :], sp[1, :])
                    if self.num_dims == 3:
                        plt.plot(sp[0, :], sp[1, :], sp[2, :])
                mp.x_derivs = None  # TODO hacky thing to help with pickling. Polynomial MPs are carrying around lambda functions which are hard to pickle
        self.pickle_self()

    def tile_lattice(self):
        bounds = 2* np.array([0, -self.max_state[0], self.max_state[0]])
        tiled_pts = np.array([self.sample_pts for i in range(3 ** self.num_dims)])
        if self.num_dims == 2:
            offsets = itertools.product(bounds, bounds)
        elif self.num_dims == 3:
            offsets = itertools.product(bounds, bounds, bounds)
        for i, offset in enumerate(offsets):
            tiled_pts[i, :, :self.num_dims] += offset
        
        tiled_pts = tiled_pts.reshape(len(self.sample_pts) * 3 ** self.num_dims, self.num_dims * self.control_space_q)
        if self.plot:
            plt.plot(tiled_pts[:,0],tiled_pts[:,1],'ko')
            plt.plot(self.sample_pts[:,0],self.sample_pts[:,1],'mo')
        return tiled_pts


if __name__ == "__main__":
    # define parameters
    control_space_q = 3
    num_dims = 2
    max_state = [1, 1, 1, 100, 1, 1]  # .5 m/s max velocity 14 m/s^2 max acceleration
    plot = True
    min_connections = 5

    # build lattice
    mps = MotionPrimitiveLattice(control_space_q=control_space_q, num_dims=num_dims, max_state=max_state, plot=plot)
    mps.compute_min_dispersion_space(num_output_pts=20, resolution=[.1, 1, 1, 25, 1, 1])
    mps.connect_lattice(min_connections)
    mps.tile_lattice()

    # plot
    if mps.plot:
        plt.show()
