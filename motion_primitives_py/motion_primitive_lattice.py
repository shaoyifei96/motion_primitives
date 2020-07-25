from motion_primitive_graph import *


class MotionPrimitiveLattice(MotionPrimitiveGraph):
    """
    A class that provides functions to compute a lattice of minimum dispersion
    points in the state space connected by feasible trajectories
    """
    @classmethod
    def load(cls, filename, tiling=True, plot=False):
        """
        create a motion primitive lattice from a given json file
        """
        # read from JSON file
        try:
            with open(filename) as json_file:
                data = json.load(json_file)
                print("Reading lattice from", filename, "...")
        except:
            print("Error reading from", filename)
            return None

        # build motion primitive lattice from data
        mpl = cls(control_space_q=data["control_space_q"],
                  num_dims=data["num_dims"],
                  max_state=data["max_state"],
                  motion_primitive_type=getattr(sys.modules[__name__], data["mp_type"]),
                  tiling=tiling,
                  plot=plot)
        mpl.vertices = np.array(data["vertices"])
        mpl.edges = np.empty((len(mpl.vertices), len(mpl.vertices)), dtype=object)
        for i in range(len(mpl.vertices)):
            for j in range(len(mpl.vertices)):
                mpl.edges[i, j] = mpl.motion_primitive_type.from_dict(
                    data["edges"][i * len(mpl.vertices) + j],
                    mpl.num_dims, mpl.max_state)
        print("Lattice successfully read")
        return mpl

    def save(self, filename):
        """
        save the motion primitive lattice to a JSON file
        """
        # convert the motion primitives to a form that can be written
        mps = []
        for i in range(len(self.vertices)):
            for j in range(len(self.vertices)):
                mp = self.edges[i, j]
                if mp is not None:
                    mps.append(mp.to_dict())
                else:
                    mps.append({})

        # write the JSON file
        with open(filename, "w") as output_file:
            saved_params = {"mp_type": self.motion_primitive_type.__name__,
                            "control_space_q": self.control_space_q,
                            "num_dims": self.num_dims,
                            "max_state": self.max_state.tolist(),
                            "vertices": self.vertices.tolist(),
                            "edges": mps
                            }
            json.dump(saved_params, output_file, indent=4)

    def dispersion_distance_fn_trajectory(self, start_pts, end_pts):
        """
        A function that evaluates the cost of a path from an array of start_pts
        to an array of end_pts. For the moment the cost is the time of the
        optimal path.
        """
        score = np.ones((len(start_pts), len(end_pts))) * -np.inf
        mp_list = np.empty((len(start_pts), len(end_pts)), dtype=object)
        for i in range(start_pts.shape[0]):
            for j in range(end_pts.shape[0]):
                if (start_pts[i, :] == end_pts[j, :]).all():
                    continue
                mp = self.motion_primitive_type(start_pts[i, :], end_pts[j, :],
                                                self.num_dims, self.max_state,
                                                self.mp_subclass_specific_data)
                if mp.is_valid:
                    mp_list[i, j] = mp
                    score[i, j] = mp.cost
        return score, mp_list

    def compute_min_dispersion_points(self, num_output_pts, potential_sample_pts, animate=False):
        # TODO will pick the same point if all MPs fail
        # overloaded from motion_primitive_graph for the moment
        # TODO maybe unify with original version used in tree

        # always take the all zero state as the first actual sample
        potential_sample_pts = np.vstack((np.zeros(self.n), potential_sample_pts))
        index = 0

        # initialize data structures
        if self.tiling:
            tiling_num = 3 ** self.num_dims
        else:
            tiling_num = 1
        mp_adjacency_matrix = np.empty((num_output_pts*tiling_num, len(potential_sample_pts)), dtype=object)
        actual_sample_indices = np.zeros((num_output_pts)).astype(int)
        min_score = np.ones((len(potential_sample_pts), tiling_num + 1)) * np.inf

        # each time through loop add point to the set and update data structures
        print("potential sample points:", len(potential_sample_pts))
        for i in range(num_output_pts):
            print(f"{i + 1}/{num_output_pts}")

            # add index to the list of sample node indices
            actual_sample_indices[i] = np.array((index))

            # update scores of nodes
            min_score[index, 0] = -np.inf  # give node we chose low score
            if self.tiling:
                end_pts = self.tile_points([potential_sample_pts[index, :]])
            else:
                end_pts = potential_sample_pts[index, :][np.newaxis, :]
            min_score[:, 1:], mp_list = self.dispersion_distance_fn(
                potential_sample_pts, end_pts)
            min_score[:, 0] = np.amin(min_score, axis=1)

            # take the new point with the maximum distance to its closest node
            index = np.argmax(min_score[:, 0])
            if min_score[index, 0] == -np.inf:
                print("""ERROR: no new valid trajectories to a point in the
                      sample set. You probably need to increase max state or
                      decrease resolution. Exiting.""")
                raise SystemExit

            # save motion primitives in the adjacency matrix
            mp_adjacency_matrix[i*tiling_num:(i+1)*tiling_num, :] = mp_list.transpose()

            # update dispersion metric
            self.dispersion = max(min_score[:, 0])
            self.dispersion_list.append(self.dispersion)

        # create graph representation to return
        vertices = potential_sample_pts[actual_sample_indices]
        edges = mp_adjacency_matrix[:, actual_sample_indices]

        # create an animation of the dispersion set growth
        if animate:
            self.make_animation_min_dispersion_points(actual_sample_indices,
                                                      mp_adjacency_matrix,
                                                      vertices,
                                                      potential_sample_pts)
        return vertices, edges

    def compute_min_dispersion_space(self, num_output_pts, resolution, animate=False):
        """
        Using the bounds on the state space, compute a set of minimum dispersion
        points (similar to original Dispertio paper) and save the resulting
        graph as a class attribute

        Input:
            num_output_pts, desired number of samples (M) in the set
            resolution, (N,) resolution over N dimensions
        """
        # TODO maybe move this somewhere else
        self.dispersion_distance_fn = self.dispersion_distance_fn_trajectory

        bounds = np.vstack((-self.max_state[:self.control_space_q], self.max_state[:self.control_space_q])).T
        potential_sample_pts = self.uniform_state_set(bounds, resolution[:self.control_space_q], random=False)
        self.vertices, self.edges = self.compute_min_dispersion_points(num_output_pts, potential_sample_pts, animate)
        if self.plot:
            if self.num_dims == 2:
                self.ax.plot(self.vertices[:, 0], self.vertices[:, 1], 'og')
            if self.num_dims == 3:
                self.ax_3d.plot(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], 'og')

    def limit_connections(self, cost_threshold):
        """
        Examine the graph of motion primitives making up the lattice and remove
        edges that have costs greater than a given threshold

        Input:
            cost_threshold, max allowable cost for any edge in returned graph
        """
        for i in range(len(self.edges)):
            for j in range(len(self.vertices)):
                mp = self.edges[i, j]
                if mp is not None and mp.is_valid and mp.cost < cost_threshold:
                    if self.plot:
                        st, sp, sv, sa, sj = mp.get_sampled_states()
                        if self.num_dims == 2:
                            if self.tiling:
                                t = self.tile_points(self.vertices)
                            self.ax.plot(sp[0, :], sp[1, :])
                            self.ax.plot(self.vertices[:, 0], self.vertices[:, 1], 'og')
                            self.ax_3d.plot(sp[0, :], sp[1, :], sv[0, :])
                            self.ax_3d.plot(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], 'og')

                        elif self.num_dims == 3:
                            plt.plot(sp[0, :], sp[1, :], sp[2, :])
                            plt.plot(self.vertices[:, 0], self.vertices[:, 1],
                                     self.vertices[:, 2], 'om')
                else:
                    self.edges[i, j] = None

    def tile_points(self, pts):
        """
        Given a set of points in state space, return the set of points that
        copies the original set into the neighbors of an 8 or 26-connected grid.
        Each cell in this grid will have dimensions corresponding to the
        position bounds of the motion primitive lattice.

        Input:
            pts, (M, N) a set of M points each of N dimension
        Output:
            tiled_pts, (L, N) the tiled set of input points.
                L is 9M or 27M depending on the dimension of the state space
                each row that is a multiple of M corresponds to M[row]
        """
        bounds = 2 * np.array([0, -self.max_state[0], self.max_state[0]])
        tiled_pts = np.array([pts for i in range(3 ** self.num_dims)])
        if self.num_dims == 2:
            offsets = itertools.product(bounds, bounds)
        elif self.num_dims == 3:
            offsets = itertools.product(bounds, bounds, bounds)
        for i, offset in enumerate(offsets):
            tiled_pts[i, :, :self.num_dims] += offset
        return tiled_pts.reshape(len(pts) * 3 ** self.num_dims,
                                 self.n)

    def animation_helper(self, i, costs_mat, colors, sample_inds, adj_mat, vertices, potential_sample_pts):
        print(f"frame {i+1}/{vertices.shape[0]}")
        tiling_num = int(adj_mat.shape[0]/vertices.shape[0])
        closest_sample_pt = np.argmin(costs_mat[:(i+1)*tiling_num, ], axis=0)
        for j in range(adj_mat.shape[1]):
            mp = adj_mat[closest_sample_pt[j], j]
            if mp is not None and mp.is_valid:
                if j in sample_inds[:i+1]:  # Don't plot the trajectories between actual samples
                    sp = np.array([[], []])
                else:
                    _, sp, _, _, _ = mp.get_sampled_states()
                self.lines[0][j].set_data(sp[0, :], sp[1, :])
                self.lines[0][j].set_color(colors[closest_sample_pt[j] % 20])
        if i+1 < sample_inds.shape[0]:
            self.lines[4].set_data(range(i+1), self.dispersion_list[:i+1])
        if self.tiling:
            tiled_vertices = self.tile_points(vertices[:i+1, :])
            self.lines[2].set_data(tiled_vertices[i+1:, 0], tiled_vertices[i+1:, 1])
        self.lines[3].set_data(vertices[:i+1, 0], vertices[:i+1, 1])

        return self.lines

    def make_animation_min_dispersion_points(self, sample_inds, adj_mat, vertices, potential_sample_pts):
        save_animation = True
        if save_animation:
            import matplotlib
            normal_backend = matplotlib.get_backend()
            matplotlib.use("Agg")
        import matplotlib.animation as animation

        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_xlim(-self.max_state[0]*1.2*3, self.max_state[0]*1.2*3)
        ax1.set_ylim(-self.max_state[0]*1.2*3, self.max_state[0]*1.2*3)
        ax1.set_title("Sample Set Evolution")
        ax2.set_xlim(0, sample_inds.shape[0])
        ax2.set_ylim(0, self.dispersion_list[0]*1.2)
        ax2.set_title("Trajectory Length Dispersion")

        traj_lines = []
        for j in range(adj_mat.shape[1]):
            traj_lines.append(ax1.plot([], [], linewidth=.4)[0])
        dense_sample_pt_line, = ax1.plot([], [], 'o', markersize=1, color=('0.8'))
        actual_sample_pt_line, = ax1.plot([], [], 'og')
        tiled_pts_line, = ax1.plot([], [], 'ob')
        dispersion_line, = ax2.plot([], [], 'ok--')
        self.lines = [traj_lines, dense_sample_pt_line, tiled_pts_line, actual_sample_pt_line, dispersion_line]
        # self.lines[1].set_data(potential_sample_pts[:, 0], potential_sample_pts[:, 1])

        costs_mat = np.array([getattr(obj, 'cost', np.inf) if getattr(obj, 'is_valid', False) else np.inf for index,
                              obj in np.ndenumerate(adj_mat)]).reshape(adj_mat.shape)
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        ani = animation.FuncAnimation(
            f, self.animation_helper, vertices.shape[0], interval=3000, fargs=(costs_mat, colors, sample_inds, adj_mat, vertices, potential_sample_pts), repeat=False)

        if save_animation:
            print("Saving animation to disk")
            ani.save('dispersion_algorithm.mp4')
            print("Finished saving animation")
            matplotlib.use(normal_backend)
        else:
            plt.show()


if __name__ == "__main__":
    # define parameters
    control_space_q = 2
    num_dims = 2
    max_state = [2, 2*np.pi, 2*np.pi, 100, 1, 1]
    motion_primitive_type = ReedsSheppMotionPrimitive
    tiling = False
    plot = True
    animate = False

    # motion_primitive_type = PolynomialMotionPrimitive
    # control_space_q = 2
    # num_dims = 2
    # max_state = [2, 1, 100, 100, 1, 1]

    # motion_primitive_type = JerksMotionPrimitive
    # control_space_q = 3
    # num_dims = 2
    # max_state = [1, 1, 1, 100, 1, 1]

    # build lattice
    mpl = MotionPrimitiveLattice(control_space_q, num_dims, max_state, motion_primitive_type, tiling, plot)
    # with PyCallGraph(output=GraphvizOutput(), config=Config(max_depth=8)):
    mpl.compute_min_dispersion_space(num_output_pts=5, resolution=[.3, .3, np.inf, 25, 1, 1], animate=animate)
    # print(mpl.vertices)
    # mpl.limit_connections(np.inf)
    mpl.limit_connections(2*mpl.dispersion)
    # mpl.save("lattice_test.json")
    # mpl = MotionPrimitiveLattice.load("lattice_test.json")

    # plot
    if mpl.plot:
        plt.show()
