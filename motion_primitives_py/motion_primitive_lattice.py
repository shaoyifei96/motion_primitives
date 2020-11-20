from motion_primitives_py import MotionPrimitiveGraph
import motion_primitives_py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
import ujson as json
import sys
from multiprocessing import Pool


class MotionPrimitiveLattice(MotionPrimitiveGraph):
    """
    A class that provides functions to compute a lattice of minimum dispersion
    points in the state space connected by feasible trajectories
    """
    @classmethod
    def load(cls, filename, plot=False):
        """
        create a motion primitive lattice from a given json file
        """
        # read from JSON file
        with open(filename) as json_file:
            data = json.load(json_file)
            print("Reading lattice from", filename, "...")

        # build motion primitive lattice from data
        mpl = cls(control_space_q=data["control_space_q"],
                  num_dims=data["num_dims"],
                  max_state=data["max_state"],
                  motion_primitive_type=getattr(motion_primitives_py, data["mp_type"]),
                  tiling=data["tiling"], plot=plot)
        mpl.dispersion = data["dispersion"]
        mpl.vertices = np.array(data["vertices"])
        mpl.edges = np.empty((len(mpl.vertices)*mpl.num_tiles, len(mpl.vertices)), dtype=object)
        for i in range(len(mpl.edges)):
            for j in range(len(mpl.vertices)):
                mpl.edges[i, j] = mpl.motion_primitive_type.from_dict(
                    data["edges"][i * len(mpl.vertices) + j], mpl.num_dims,
                    mpl.max_state, mpl.mp_subclass_specific_data)
        print("Lattice successfully read")
        return mpl

    def save(self, filename):
        """
        save the motion primitive lattice to a JSON file
        """
        # convert the motion primitives to a form that can be written
        mps = []
        for i in range(len(self.edges)):
            for j in range(len(self.vertices)):
                mp = self.edges[i, j]
                if mp is not None:
                    mps.append(mp.to_dict())
                else:
                    mps.append({})
        # write the JSON file
        with open(filename, "w") as output_file:
            print("Saving lattice to", filename, "...")
            saved_params = {"mp_type": self.motion_primitive_type.__name__,
                            "control_space_q": self.control_space_q,
                            "num_dims": self.num_dims,
                            "tiling": True if self.num_tiles > 1 else False,
                            "max_state": self.max_state.tolist(),
                            "vertices": self.vertices.tolist(),
                            "edges": mps,
                            "dispersion": self.dispersion
                            }
            json.dump(saved_params, output_file, indent=4)
            print("Lattice successfully saved")

    def dispersion_distance_fn_trajectory(self, inputs):
        """
        A function that evaluates the cost of a path from an array of start_pts
        to an array of end_pts. For the moment the cost is the time of the
        optimal path.
        """
        start_pts = inputs[0]
        end_pts = inputs[1]
        mp = self.motion_primitive_type(start_pts, end_pts,
                                        self.num_dims, self.max_state, mp_subclass_specific_data)
        mp.subclass_specific_data['dynamics'] = None  # hacky stuff to avoid pickling lambda functions
        if not mp.is_valid:
            mp.cost = np.nan
        return mp

    def multiprocessing_init(self):
         # hacky stuff to avoid pickling lambda functions
        global mp_subclass_specific_data
        mp_subclass_specific_data = self.mp_subclass_specific_data

    def multiprocessing_dispersion_distance_fn(self, pool, start_pts, end_pts):
        paramlist = list(itertools.product(start_pts, end_pts))
        if 'dynamics' in self.mp_subclass_specific_data:
            self.mp_subclass_specific_data['dynamics'] = None  # hacky stuff to avoid pickling lambda functions
        pool_output = pool.map(self.dispersion_distance_fn, paramlist)
        min_score = np.array([mp.cost for mp in pool_output]).reshape(start_pts.shape[0], end_pts.shape[0])
        mp_list = np.array(pool_output).reshape(start_pts.shape[0], end_pts.shape[0])
        return min_score, mp_list

    def compute_min_dispersion_points(self, num_output_pts, potential_sample_pts, check_backwards_dispersion=False, animate=False):
        # overloaded from motion_primitive_graph for the moment
        # TODO maybe unify with original version used in tree

        # always take the all zero state as the first actual sample
        potential_sample_pts = np.vstack((np.zeros(self.n), potential_sample_pts))
        index = 0

        # initialize data structures
        mp_adjacency_matrix_fwd = np.empty((num_output_pts * self.num_tiles, len(potential_sample_pts)), dtype=object)
        actual_sample_indices = np.zeros((num_output_pts)).astype(int)
        min_score = np.ones((len(potential_sample_pts), 2)) * np.inf
        # create multiprocessing pool to compute MPs
        pool = Pool(initializer=self.multiprocessing_init)

        # each time through loop add point to the set and update data structures
        print("potential sample points:", len(potential_sample_pts))
        for i in range(num_output_pts):
            print(f"MP {i + 1}/{num_output_pts}, Dispersion = {self.dispersion}")

            # add index to the list of sample node indices
            actual_sample_indices[i] = np.array((index))
            print(potential_sample_pts[index])

            # update scores of nodes
            min_score[index, 0] = -np.inf  # give node we chose low score
            if self.num_tiles > 1:
                end_pts = self.tile_points([potential_sample_pts[index, :]])
            else:
                end_pts = potential_sample_pts[index, :][np.newaxis, :]

            min_score_fwd, mp_list_fwd = self.multiprocessing_dispersion_distance_fn(pool, potential_sample_pts, end_pts)
            if check_backwards_dispersion:
                min_score_bwd, mp_list_bwd = self.multiprocessing_dispersion_distance_fn(pool, end_pts, potential_sample_pts)
                min_score[:, 1] = np.maximum(np.nanmin(min_score_fwd, axis=1), np.nanmin(min_score_bwd.T, axis=1))
            else:
                min_score[:, 1] = np.nanmin(min_score_fwd, axis=1)
            min_score[:, 0] = np.minimum(min_score[:, 0], min_score[:, 1])  # faster than amin according to numpy doc
            np.set_printoptions(threshold=sys.maxsize)

            # take the new point with the maximum distance to its closest node
            index = np.argmax(min_score[:, 0])
            if min_score[index, 0] == -np.inf:
                print("""ERROR: no new valid trajectories to a point in the
                      sample set. You probably need to increase max state or
                      decrease resolution. Exiting.""")
                raise SystemExit

            # save motion primitives in the adjacency matrix
            mp_adjacency_matrix_fwd[i * self.num_tiles:(i + 1) * self.num_tiles, :] = mp_list_fwd.T

            # update dispersion metric
            self.dispersion = max(min_score[:, 0])
            self.dispersion_list.append(self.dispersion)

        pool.close()  # end multiprocessing pool

        # create graph representation to return
        vertices = potential_sample_pts[actual_sample_indices]
        edges = mp_adjacency_matrix_fwd[:, actual_sample_indices]

        # create an animation of the dispersion set growth
        if animate:
            self.make_animation_min_dispersion_points(actual_sample_indices,
                                                      mp_adjacency_matrix_fwd,
                                                      vertices,
                                                      potential_sample_pts)
        return vertices, edges

    def compute_min_dispersion_space(self, num_output_pts, resolution, check_backwards_dispersion=False, animate=False):
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

        potential_sample_pts = self.uniform_state_set(
            self.max_state[:self.control_space_q], resolution[:self.control_space_q], random=False)
        self.vertices, self.edges = self.compute_min_dispersion_points(
            num_output_pts, potential_sample_pts, check_backwards_dispersion, animate)
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
        # TODO determine how we want to interface this with saving
        for i in range(len(self.edges)):
            for j in range(len(self.vertices)):
                mp = self.edges[i, j]
                if mp is not None and mp.is_valid and mp.cost < cost_threshold + 1e-5:
                    if self.plot:
                        st, sp, sv, sa, sj = mp.get_sampled_states(.1)
                        if self.num_dims == 2:
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

<<<<<<< Updated upstream
=======
    def plot_config(self, ax=None, plot_mps=False, start_position_override=[0,0,0], style1='og', style2='ob', linewidth=1):
        """
        Plot the graph and motion primitives projected into the 2D or 3D
        configuration space.
        """
        tiled_verts = self.tile_points(self.vertices) + start_position_override

        if ax is None:
            _, ax = plt.subplots(1, 1, subplot_kw={'projection': {2: 'rectilinear', 3: '3d'}[self.num_dims]})
        vertices = self.vertices + start_position_override
        ax.plot(vertices[:, 0], vertices[:, 1], style1, zorder=5)
        if self.num_tiles > 1:
            ax.plot(tiled_verts[:, 0], tiled_verts[:, 1], style2, zorder=4)

        if plot_mps:
            for i in range(len(self.edges)):
                for j in range(len(self.vertices)):
                    mp = self.edges[i, j]
                    if mp != None and mp.is_valid:
                        mp.subclass_specific_data = self.mp_subclass_specific_data
                        mp.plot(position_only=True, ax=ax, start_position_override=mp.start_state + start_position_override, linewidth=linewidth)
                        _, sp = mp.get_sampled_position(.1)
        return ax

>>>>>>> Stashed changes
    def get_neighbor_mps(self, node_index):
        """
        return the indices and costs of nodes that are neighbors of the given
        node index

        Input:
            node_index, index of queried node in list of vertices

        Output:
            neighbors, list of tuples with entries corresponding to neighbors
                and the MotionPrimitive object representing the trajectory
                to get to them respectively
        """
        neighbors = []
        reset_map_index = int(np.floor(node_index / self.num_tiles))
        for i, mp in enumerate(self.edges[:, reset_map_index]):
            if mp is not None and mp.is_valid:
                neighbors.append(mp)
        return neighbors

    def find_mps_to_lattice(self, state):
        """
        Given an arbitrary state, return a list of motion primitives to connect
        it to the lattice

        Input:
            state, point in state space

        Output:
            connections, list of tuples with entries corresponding to node
                indices and the MotionPrimitive object respectively
        """
        # build list of neighbors
        connections = []
        for i, vertex in enumerate(self.vertices):
            mp = self.motion_primitive_type(state, vertex,
                                            self.num_dims, self.max_state,
                                            self.mp_subclass_specific_data)
            if mp.is_valid:
                connections.append((i, mp))
        return connections

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
        return tiled_pts.reshape(len(pts) * 3 ** self.num_dims, self.n)

    def animation_helper(self, i, costs_mat, sample_inds, adj_mat, vertices, potential_sample_pts):
        print(f"frame {i+1}/{vertices.shape[0]}")
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        closest_sample_pt = np.argmin(costs_mat[:(i + 1) * self.num_tiles, ], axis=0)
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
        if self.num_tiles > 1:
            tiled_vertices = self.tile_points(vertices[:i+1, :])
            self.lines[2].set_data(tiled_vertices[i+1:, 0], tiled_vertices[i+1:, 1])
        self.lines[3].set_data(vertices[:i+1, 0], vertices[:i+1, 1])
<<<<<<< Updated upstream

=======
        # for vertex in vertices[:i+1, :]:
        #     circle = plt.Circle(vertex[:self.num_dims], 2*self.dispersion_list[i]*self.max_state[1], color='b', fill=False, zorder=4)
        #     ax1.add_artist(circle)
>>>>>>> Stashed changes
        return self.lines

    def make_animation_min_dispersion_points(self, sample_inds, adj_mat, vertices, potential_sample_pts):
        save_animation = False
        if save_animation:
            import matplotlib
            normal_backend = matplotlib.get_backend()
            matplotlib.use("Agg")

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
        ani = animation.FuncAnimation(
<<<<<<< Updated upstream
            f, self.animation_helper, vertices.shape[0], interval=3000, fargs=(costs_mat, sample_inds, adj_mat, vertices, potential_sample_pts), repeat=False)

=======
            f, self.animation_helper, vertices.shape[0], interval=3000, fargs=(ax1, costs_mat, sample_inds, adj_mat, vertices, potential_sample_pts), repeat=False)
            
>>>>>>> Stashed changes
        if save_animation:
            print("Saving animation to disk")
            ani.save('dispersion_algorithm.mp4')
            frames = ani.new_frame_seq()
            print("Finished saving animation")
            matplotlib.use(normal_backend)
        else:
            plt.show()

<<<<<<< Updated upstream
=======
        # plotting code for paper figure
        for i in range(vertices.shape[0]):
            lines = self.animation_helper(i,ax1, costs_mat, sample_inds, adj_mat, vertices, potential_sample_pts)
            x = np.array(lines, dtype='object').flatten()
            for line in x:
                ax1.add_line(line)

    def compute_dispersion_from_graph(self, vertices, resolution, no_sampling_value=0, colorbar_max=None):
        max_state = self.max_state[:self.control_space_q]
        max_state[0] = max(vertices[:, 0])*.7
        dense_sampling, axis_sampling = self.uniform_state_set(
            max_state, resolution[:self.control_space_q], random=False, no_sampling_value=no_sampling_value)
        pool = Pool(initializer=self.multiprocessing_init)
        self.vertices = None
        self.edges = None
        print(dense_sampling.shape)

        score, adj_mat = self.multiprocessing_dispersion_distance_fn_trajectory(pool, dense_sampling, vertices)
        pool.close()  # end multiprocessing pool
        costs_mat = np.array([getattr(obj, 'cost', np.inf) if getattr(obj, 'is_valid', False) else np.inf for index,
                              obj in np.ndenumerate(adj_mat)]).reshape(adj_mat.shape)
        # print(costs_mat)
        closest_sample_pt = np.argmin(costs_mat, axis=1)
        min_score = np.nanmin(score, axis=1)
        dispersion = np.nanmax(min_score)
        if colorbar_max is None:
            colorbar_max = dispersion
        plt.pcolormesh(axis_sampling[0], axis_sampling[1], np.amin(costs_mat, axis=1).reshape(
            (axis_sampling[0].shape[0], axis_sampling[1].shape[0])), edgecolors='k', shading='gouraud', norm=plt.Normalize(0, colorbar_max))
        plt.colorbar()

        plt.figure()
        colors = plt.cm.viridis(np.linspace(0, 1, 101))
        for j in range(adj_mat.shape[0]):
            mp = adj_mat[j, closest_sample_pt[j]]
            mp.subclass_specific_data = self.mp_subclass_specific_data
            if mp.is_valid:
                mp.plot(position_only=True, color=colors[int(np.floor(mp.cost/colorbar_max*100))])

        plt.scatter(dense_sampling[:, 0], dense_sampling[:, 1], c=min_score)
        plt.plot(vertices[:, 0], vertices[:, 1], '*')

        return dispersion

>>>>>>> Stashed changes

if __name__ == "__main__":
    # %%
    from motion_primitives_py import *
    import numpy as np
    import time
    from pycallgraph import PyCallGraph, Config
    from pycallgraph.output import GraphvizOutput

    tiling = True
    plot = True
    animate = True
    check_backwards_dispersion = True
    mp_subclass_specific_data = {}

    # %%
    # define parameters
    control_space_q = 2
    num_dims = 2
    max_state = [.5, np.pi/4, 2*np.pi, 100, 1, 1]
    motion_primitive_type = ReedsSheppMotionPrimitive
    resolution = [.21, .2]

<<<<<<< Updated upstream
    # # %%
    motion_primitive_type = PolynomialMotionPrimitive
    control_space_q = 2
    num_dims = 2
    max_state = [.51, 3.51, 15, 100, 1, 1]
    mp_subclass_specific_data = {'iterative_bvp_dt': .05, 'iterative_bvp_max_t': 2}
    resolution = [.11, 1.21]

    # %%
=======
    # # # %%
    #motion_primitive_type = PolynomialMotionPrimitive
    #control_space_q = 2
    #num_dims = 2
    #max_state = [5.51, 1.51, 15, 100]
    #mp_subclass_specific_data = {'iterative_bvp_dt': .1, 'iterative_bvp_max_t': 5, 'rho':1}
    
    control_space_q = 2
    num_dims = 2
    max_state = [3.5, 2*np.pi]
    motion_primitive_type = ReedsSheppMotionPrimitive
    num_dense_samples = 100#num_dense_samples = 100
    
    # # # %%
>>>>>>> Stashed changes
    # motion_primitive_type = JerksMotionPrimitive
    # control_space_q = 3
    # num_dims = 2
    # max_state = [.51, 1.51, .51, 100, 1, 1]

    # %%
    # build lattice
    mpl = MotionPrimitiveLattice(control_space_q, num_dims, max_state, motion_primitive_type, tiling, False, mp_subclass_specific_data)
    tic = time.time()
    # with PyCallGraph(output=GraphvizOutput(), config=Config(max_depth=8)):
    mpl.compute_min_dispersion_space(
        num_output_pts=15, resolution=resolution, check_backwards_dispersion=check_backwards_dispersion, animate=animate)
    toc = time.time()
    print(toc-tic)
    print(mpl.vertices)
    mpl.limit_connections(2*mpl.dispersion)
    mpl.save("lattice_test.json")
    # mpl = MotionPrimitiveLattice.load("lattice_test.json", plot)
    # mpl.limit_connections(2*mpl.dispersion)
    # print(mpl.dispersion)
    # print(sum([1 for i in np.nditer(mpl.edges, ['refs_ok']) if i != None])/len(mpl.vertices))

    # %%
    # plot
    plt.show()


# %%
