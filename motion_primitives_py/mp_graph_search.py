#!/usr/bin/python3

from motion_primitives_py.motion_primitive_lattice import *
from motion_primitives_py.occupancy_map import OccupancyMap
from heapq import heappush, heappop, heapify  # Recommended.
from scipy import spatial
from enum import Enum
from copy import deepcopy
# # from pycallgraph import PyCallGraph
# # from pycallgraph.output import GraphvizOutput
# from pathlib import Path


class Node:
    """
    Container for node data. Nodes are sortable by the value of (f, -g).
    """

    def __init__(self, g, h, state, parent, index=None, parent_index=None):
        self.f = g + h          # total-cost
        self.g = g              # cost-to-come
        self.h = h              # heuristic
        self.state = state
        self.parent = parent
        self.index = index
        self.parent_index = parent_index
        self.is_closed = False  # True if node has been closed.

    def __lt__(self, other):
        return (self.f, -self.g) < (other.f, -other.g)

    def __repr__(self):
        return f"Node g={self.g}, h={self.h}, state={self.state}, parent={self.parent}, is_closed={self.is_closed}, index={self.index}, parent_index={self.parent_index}"


class GraphSearch:
    """
    Uses a motion primitive lookup table stored in a pickle file to perform a graph search. Must run min_dispersion_primitives_tree.py to create a pickle file first.
    """

    def __init__(self, motion_primitive_graph, occupancy_map, start_state, goal_state, goal_tolerance, mp_sampling_step_size=0.1, heuristic='euclidean', neighbors='lattice', plot=False):
        # Save arguments as parameters
        self.motion_primitive_graph = motion_primitive_graph
        self.map = occupancy_map
        self.start_state = np.array(start_state)
        self.goal_state = np.array(goal_state)
        self.goal_tolerance = np.array(goal_tolerance)
        self.mp_sampling_step_size = mp_sampling_step_size
        self.plot = plot

        # Dimensionality parameters
        self.num_dims = self.motion_primitive_graph.num_dims
        self.control_space_q = self.motion_primitive_graph.control_space_q
        self.n = self.motion_primitive_graph.n

        # Create comparators for extreme states
        map_size = self.map.extent
        self.min_state_comparator = np.hstack([np.array(map_size[:self.num_dims])] +
                                              [np.repeat(-i, self.num_dims) for i in self.motion_primitive_graph.max_state[:self.control_space_q-1]])
        self.max_state_comparator = np.hstack([np.array(map_size[self.num_dims:])] +
                                              [np.repeat(i, self.num_dims) for i in self.motion_primitive_graph.max_state[:self.control_space_q-1]])

        # Parameter used in min time heuristic from Sikang's paper
        self.rho = 1.0
        self.heuristic_dict = {
            'zero': self.zero_heuristic,
            'euclidean': self.euclidean_distance_heuristic,
            'min_time': self.min_time_heuristic,
            'bvp': self.bvp_heuristic, }
        self.heuristic = self.heuristic_dict[heuristic]

        self.neighbor_dict = {
            # 'min_dispersion' : self.get_neighbors_min_dispersion,
            'evenly_spaced': self.get_neighbors_evenly_spaced,
            'lattice': self.get_neighbors_lattice, }
        self.get_neighbors = self.neighbor_dict[neighbors]

        self.start_position_offset = np.hstack((self.start_state[:self.num_dims], np.zeros_like(self.start_state[self.num_dims:])))
        # self.mp_start_pts_tree = spatial.KDTree(start_state)  # self.motion_primitive_graph.start_pts)

    def is_valid_state(self, state):
        if (state < self.min_state_comparator).any() or (state > self.max_state_comparator).any():
            return False
        return True

    def zero_heuristic(self, state):
        return 0

    def min_time_heuristic(self, state):
        # sikang heuristic 1
        return self.rho * np.linalg.norm(state[0:self.num_dims] - self.goal_state[0:self.num_dims], ord=np.inf)/self.motion_primitive_graph.max_state[1]

    def euclidean_distance_heuristic(self, state):
        return np.linalg.norm(state[:self.num_dims] - self.goal_state[:self.num_dims])

    def bvp_heuristic(self, state):
        cost = self.motion_primitive_graph.motion_primitive_type(state, self.goal_state, self.motion_primitive_graph.num_dims, self.motion_primitive_graph.max_state, self.motion_primitive_graph.mp_subclass_specific_data).cost
        if cost==None:
            return np.inf
        return cost

    def build_path(self, node):
        """
        Build path from start point to goal point using the goal node's parents.
        """
        path = [node.state]
        sampled_path = []
        path_cost = 0
        while node.parent is not None:
            path.append(node.parent)
            mp = [mp for (i, mp) in self.motion_primitive_graph.get_neighbors(node.parent_index)if i == node.index][0]
            st, sp, sv, sa, sj = mp.get_sampled_states()
            sampled_path.append(sp + node.parent[:self.num_dims][:, np.newaxis] - mp.start_state[:self.num_dims][:, np.newaxis])
            path_cost += mp.cost
            node = self.node_dict[node.parent.tobytes()]
        path.append(self.start_state)
        mp = self.start_edges[node.index]
        st, sp, sv, sa, sj = mp.get_sampled_states()
        path_cost += mp.cost

        sampled_path.append(sp + self.start_state[:self.num_dims][:, np.newaxis])

        path.reverse()
        sampled_path.reverse()
        return np.vstack(path).transpose(), np.hstack(sampled_path), path_cost

    def get_neighbors_min_dispersion(self, node):
        # TODO change neighbors to nodes
        start_pt = np.array(node.state)
        closest_start_pt_index = self.mp_start_pts_tree.query(start_pt)[1]
        motion_primitives_list = self.motion_primitive_graph.motion_primitives_list[closest_start_pt_index]
        dt_set = motion_primitives_list[0, :]
        u_set = motion_primitives_list[1:, :]
        neighbors = np.array(self.motion_primitive_graph.quad_dynamics_polynomial(start_pt, u_set, dt_set)).T
        neighbors = np.hstack((neighbors, motion_primitives_list.T))
        return neighbors

    def get_neighbors_evenly_spaced(self, node):
        # TODO change neighbors to nodes
        dt = .5
        num_u_per_dimension = 5  # self.motion_primitive_graph.motion_primitives_list[0].shape[1]
        s = np.reshape(np.array(node.state), (self.n, 1))
        neighbor_states = self.motion_primitive_graph.create_evenly_spaced_mps(s, dt, num_u_per_dimension)
        neighbors = []
        for i, neighbor in enumerate(neighbor_states):
            state = neighbor[:self.n]
            neighbor_node = Node(neighbor[self.n] + node.g, self.heuristic(state), state, node.state)
            neighbors.append(neighbor_node)
        node.is_closed = True
        return neighbors

    def get_neighbors_lattice(self, node):
        neighbors = []
        reset_map_index = int(np.floor(node.index / self.motion_primitive_graph.num_tiles))
        for i, mp in enumerate(self.motion_primitive_graph.edges[:, reset_map_index]):
            if mp is not None and mp.is_valid and self.map.is_mp_collision_free(mp, step_size=self.mp_sampling_step_size, offset=node.state[:self.num_dims]):
                state = deepcopy(node.state)
                state[:self.num_dims] += (mp.end_state - mp.start_state)[:self.num_dims]
                state[self.num_dims:] = mp.end_state[self.num_dims:]
                neighbor_node = Node(mp.cost + node.g, self.heuristic(state), state, node.state, index=i, parent_index=node.index)
                neighbors.append(neighbor_node)
        node.is_closed = True
        return neighbors

    def reset_graph_search(self):
        self.node_dict = {}  # A dict where key is an state and the value is a node in the queue.
        self.queue = []      # A priority queue of nodes as a heapq.
        self.neighbor_list = []
        self.closed_nodes = []

        if not self.map.is_free_and_valid_position(self.start_state[:self.num_dims]):
            print("start invalid")
            self.queue = None
            return
        if not self.map.is_free_and_valid_position(self.goal_state[:self.num_dims]):
            print("goal invalid")
            self.queue = None
            return

        if self.get_neighbors.__name__ == 'get_neighbors_lattice':
            starting_neighbors = self.motion_primitive_graph.find_mps_to_lattice(deepcopy(self.start_state) - self.start_position_offset)
            self.start_edges = []
            for i, mp in starting_neighbors:
                state = mp.end_state + self.start_position_offset
                node = Node(mp.cost, self.heuristic(state), state, None, i, None)
                self.start_edges.append(mp)
                if self.map.is_mp_collision_free(mp, step_size=self.mp_sampling_step_size, offset=self.start_position_offset):
                    heappush(self.queue, node)
                    self.node_dict[node.state.tobytes()] = node
        else:
            node = Node(0, 0, self.start_state, None)
            heappush(self.queue, node)

    def run_graph_search(self):
        self.reset_graph_search()

        # While queue is not empty, pop the next smallest total cost f node
        path = None
        sampled_path = None
        path_cost = None
        nodes_expanded = 0

        while self.queue:
            node = heappop(self.queue)
            # If node has been closed already, skip.
            if node.is_closed:
                continue

            # Otherwise, expand node and for each neighbor...
            nodes_expanded += 1
            self.closed_nodes.append(node)  # for animation/plotting

            # If node is the goal node, return path.
            # TODO separately compare states besides position for goal tolerance
            if np.linalg.norm(node.state[:self.num_dims] - (self.goal_state[:self.num_dims])) < self.goal_tolerance[0]:
                print("Path found")
                path, sampled_path, path_cost = self.build_path(node)
                break

            ###JUST FOR TESTING
            # if (nodes_expanded) > 100:
            #     break

            neighbors = self.get_neighbors(node)
            for neighbor_node in neighbors:
                old_neighbor = self.node_dict.get(neighbor_node.state.tobytes(), None)
                if old_neighbor is None or neighbor_node.g < old_neighbor.g:
                    heappush(self.queue, neighbor_node)
                    self.node_dict[neighbor_node.state.tobytes()] = neighbor_node
                if old_neighbor is not None:
                    old_neighbor.is_closed = True
                self.neighbor_list.append(neighbor_node.state)  # for plotting

        if self.queue is not None:
            print()
            print(f"Nodes in queue at finish: {len(self.queue)}")
            print(f"Closed nodes in queue at finish: {sum(node.is_closed for node in self.queue)}")
            print()
            print(f"Nodes expanded: {nodes_expanded}")
            self.neighbor_list = np.array(self.neighbor_list)
            self.closed_nodes = np.array(self.closed_nodes)

        if path is None:
            print("No path found")
        return path, sampled_path, path_cost

    def animation_helper(self, i, closed_set_states):
        # print(f"frame {i+1}/{len(self.closed_nodes)+10}")
        for k in range(len(self.lines[0])):
            self.lines[0][k].set_data([], [])

        if i >= len(self.closed_nodes):
            i = len(self.closed_nodes)-1
            node = self.closed_nodes[i]
            path, sampled_path, path_cost = self.build_path(node)
            self.lines[0][0].set_data(sampled_path[0, :], sampled_path[1, :])
            self.lines[0][0].set_linewidth(2)
            self.lines[0][0].set_zorder(11)
        else:
            node = self.closed_nodes[i]
            open_list = []
            for j, (_, mp) in enumerate(self.motion_primitive_graph.get_neighbors(node.index)):
                _, sp, _, _, _ = mp.get_sampled_states()
                shifted_sp = sp + node.state[:self.num_dims][:, np.newaxis] - mp.start_state[:self.num_dims][:, np.newaxis]
                open_list.append(shifted_sp[:, -1])
                self.lines[0][j].set_data(shifted_sp[0, :], shifted_sp[1, :])
            if open_list != []:
                self.open_list_states_animation = np.vstack((self.open_list_states_animation, np.array(open_list)))
        self.lines[3].set_data(closed_set_states[:i+1, 0], closed_set_states[:i+1, 1])
        self.lines[4].set_data(self.open_list_states_animation[:, 0], self.open_list_states_animation[:, 1])
        return self.lines

    def make_graph_search_animation(self, save_animation=False):
        plt.close('all')
        if save_animation:
            import matplotlib
            normal_backend = matplotlib.get_backend()
            matplotlib.use("Agg")

        f, ax = plt.subplots(1, 1)
        bounds = (min(self.start_state[0], self.goal_state[0]) - 1,
                  max(self.start_state[0], self.goal_state[0]) + 1,
                  min(self.start_state[1], self.goal_state[1]) - 1,
                  max(self.start_state[1], self.goal_state[1]) + 1)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])

        transparent_map = np.zeros(self.map.voxels.T.shape + (4,))
        transparent_map[self.map.voxels.T > 0, :] = (0, 0, 0, 1)
        plt.imshow(transparent_map, origin='lower', extent=self.map.extent, zorder=2)

        ax.axis('equal')

        mp_lines = []
        for j in range(len(self.motion_primitive_graph.edges)):
            mp_lines.append(ax.plot([], [], linewidth=.4)[0])
        start_line, = ax.plot(self.start_state[0], self.start_state[1], 'og')
        goal_line, = ax.plot(self.goal_state[0], self.goal_state[1], 'or')
        closed_set_line, = ax.plot([], [], 'm*', zorder=3)
        open_set_line, = ax.plot([], [], '.', color=('.8'),  zorder=1)
        circle = plt.Circle(self.goal_state[:self.num_dims], self.goal_tolerance[0], color='b', fill=False)
        circle_patch = ax.add_artist(circle)
        self.lines = [mp_lines, start_line, goal_line, closed_set_line, open_set_line, circle_patch]
        closed_set = np.array([node.state for node in self.closed_nodes])
        self.open_list_states_animation = self.start_state[:self.num_dims]
        ani = animation.FuncAnimation(f, self.animation_helper, len(self.closed_nodes)+10,
                                      interval=100, fargs=(closed_set,), repeat=False)

        if save_animation:
            print("Saving animation to disk")
            ani.save('graph_search.mp4')
            print("Finished saving animation")
            matplotlib.use(normal_backend)
        else:
            plt.show(block=False)
            plt.pause((len(self.closed_nodes)+10)/10)


if __name__ == "__main__":
    mpl = MotionPrimitiveLattice.load("lattice_test.json")
    print(mpl.vertices)
    print(mpl.dispersion)

    # mpl = MotionPrimitiveLattice(2, 2, mpl.max_state, PolynomialMotionPrimitive)

    plt.close('all')
    # print(mpl.dispersion)
    # print(sum([1 for i in np.nditer(mpl.edges, ['refs_ok']) if i != None])/len(mpl.vertices))

    # mpl.limit_connections(2*mpl.dispersion)
    # print(sum([1 for i in np.nditer(mpl.edges, ['refs_ok']) if i != None])/len(mpl.vertices))

    start_state = np.zeros((mpl.n))
    goal_state = np.zeros_like(start_state)

    # occ_map = OccupancyMap.fromVoxelMapBag('/home/laura/mpl_ws/src/mpl_ros/mpl_test_node/maps/simple/simple.bag', 0)
    # occ_map = OccupancyMap.fromVoxelMapBag('test2d.bag', 0)
    # start_state[0:2] = [2, -14]
    # goal_state[0:2] = [4, 2]

    resolution = 1
    origin = [0, 0]
    dims = [10, 100]
    data = np.zeros(dims)
    data[5:10, 10:15] = 100
    data = data.flatten('F')
    occ_map = OccupancyMap(resolution, origin, dims, data)
    start_state[0:3] = [8, 2, 0]
    goal_state[0:3] = [1, 90, 0]

    # occ_map = OccupancyMap.fromVoxelMapBag('trees_dispersion_0.6_1.bag', 0)
    # start_state[0:2] = [10, 6]
    # goal_state[0:2] = [70, 6]

    # occ_map.plot()
    # plt.plot(start_state[0], start_state[1], 'og')
    # plt.plot(goal_state[0], goal_state[1], 'or')
    # plt.show()

    goal_tolerance = np.ones_like(start_state)
    plot = True
    gs = GraphSearch(mpl, occ_map, start_state, goal_state, goal_tolerance, plot=plot, heuristic='euclidean', neighbors='lattice')
    # with PyCallGraph(output=GraphvizOutput(), config=Config(max_depth=8)):
    import time
    tic = time.time()
    path, sampled_path, path_cost = gs.run_graph_search()
    toc = time.time()
    print(f"Planning time: {toc - tic}s")

    if gs.queue is not None:
        gs.make_graph_search_animation(True)

    plt.figure()
    gs.map.plot()
    plt.plot(gs.start_state[0], gs.start_state[1], 'og')
    plt.plot(gs.goal_state[0], gs.goal_state[1], 'or')
    # # plt.plot(gs.neighbor_list[:, 0], gs.neighbor_list[:, 1], 'k.')
    # # plt.plot(gs.expanded_nodes_list[:, 0], gs.expanded_nodes_list[:, 1], 'k.')

    if sampled_path is not None:
        print(path.T)
        plt.plot(sampled_path[0, :], sampled_path[1, :])
        print(f'cost: {path_cost}')
        plt.plot(path[0, :], path[1, :], '*m')
        # mp = mpl.motion_primitive_type(start_state, goal_state, mpl.num_dims, mpl.max_state)
        # st, sp, sv, sa, sj = mp.get_sampled_states()
        # print(f'optimal path cost: {mp.cost}')
        # plt.plot(sp[0, :], sp[1, :])

    plt.show()
