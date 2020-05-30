from motion_primitive import *


class MotionPrimitiveTree(MotionPrimitive):
    """
    """

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

def create_many_state_space_lookup_tables(max_control_space):
    """
    Make motion primitive lookup tables for different state/input spaces
    """
    num_u_per_dimension = 3
    max_state = [2, 2, 1, 1, 1]
    num_state_deriv_pts = 7
    plot = False
    moprim_list = [MotionPrimitiveTree(control_space_q, num_dims, num_u_per_dimension,
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
    mp = MotionPrimitiveTree(control_space_q=control_space_q, num_dims=num_dims,
                         num_u_per_dimension=num_u_per_dimension, max_state=max_state, num_state_deriv_pts=num_state_deriv_pts, plot=plot)
    start_pt = np.ones((mp.n))

    with PyCallGraph(output=GraphvizOutput(), config=Config(max_depth=6)):
        # mp.compute_all_possible_mps(start_pt)
        mp.compute_min_dispersion_set(start_pt)
        # mp.create_state_space_MP_lookup_table_tree()

    # create_many_state_space_lookup_tables(5)

    plt.show()