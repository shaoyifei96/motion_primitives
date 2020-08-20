#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop, heapify
from copy import deepcopy
import matplotlib.animation as animation
from motion_primitives_py import MotionPrimitiveLattice, MotionPrimitiveTree


class Node:
    """
    Container for node data. Nodes are sortable by the value of (f, -g).
    """

    def __init__(self, g, h, state, parent, mp, index=None, parent_index=None, graph_depth=0):
        self.f = g + h          # total-cost
        self.g = g              # cost-to-come
        self.h = h              # heuristic
        self.state = state
        self.parent = parent
        self.mp = mp
        self.index = index
        self.parent_index = parent_index
        self.graph_depth = graph_depth
        self.is_closed = False  # True if node has been closed.

    def __lt__(self, other):
        return (self.f, -self.g) < (other.f, -other.g)

    def __repr__(self):
        return f"Node g={self.g}, h={self.h}, state={self.state}, parent={self.parent}, is_closed={self.is_closed}, index={self.index}, parent_index={self.parent_index}"


class GraphSearch:
    """
    Uses a motion primitive lookup table stored in a pickle file to perform a graph search. Must run min_dispersion_primitives_tree.py to create a pickle file first.
    """

    def __init__(self, motion_primitive_graph, occupancy_map, start_state, goal_state, goal_tolerance, mp_sampling_step_size=0.1, heuristic='euclidean'):
        # Save arguments as parameters
        self.motion_primitive_graph = motion_primitive_graph
        self.map = occupancy_map
        self.start_state = np.array(start_state)
        self.goal_state = np.array(goal_state)
        self.goal_tolerance = np.array(goal_tolerance)
        self.mp_sampling_step_size = mp_sampling_step_size

        # Dimensionality parameters
        self.num_dims = self.motion_primitive_graph.num_dims
        self.control_space_q = self.motion_primitive_graph.control_space_q
        self.n = self.motion_primitive_graph.n

        # Create comparators for extreme states
        map_size = self.map.extent

        # Parameter used in min time heuristic from Sikang's paper
        self.rho = 1.0
        self.heuristic_dict = {
            'zero': self.zero_heuristic,
            'euclidean': self.euclidean_distance_heuristic,
            'min_time': self.min_time_heuristic,
            'bvp': self.bvp_heuristic, }
        self.heuristic = self.heuristic_dict[heuristic]

        self.start_position_offset = np.hstack((self.start_state[:self.num_dims], np.zeros_like(self.start_state[self.num_dims:])))
        # self.mp_start_pts_tree = spatial.KDTree(start_state)  # self.motion_primitive_graph.start_pts)

        if type(self.motion_primitive_graph) is MotionPrimitiveTree:
            self.dt = .6
            self.num_u_per_dimension = 20
            self.num_mps = self.num_u_per_dimension**self.num_dims
            self.get_neighbor_nodes = self.get_neighbor_nodes_evenly_spaced
        elif type(self.motion_primitive_graph) is MotionPrimitiveLattice:
            self.num_mps = len(self.motion_primitive_graph.edges)
            self.get_neighbor_nodes = self.get_neighbor_nodes_lattice

    def zero_heuristic(self, state):
        return 0

    def min_time_heuristic(self, state):
        # sikang heuristic 1
        return self.rho * np.linalg.norm(state[:self.num_dims] - self.goal_state[:self.num_dims], ord=np.inf)/self.motion_primitive_graph.max_state[1]

    def euclidean_distance_heuristic(self, state):
        return np.linalg.norm(state[:self.num_dims] - self.goal_state[:self.num_dims])

    def bvp_heuristic(self, state):
        cost = self.motion_primitive_graph.motion_primitive_type(
            state, self.goal_state, self.motion_primitive_graph.num_dims, self.motion_primitive_graph.max_state, self.motion_primitive_graph.mp_subclass_specific_data).cost
        if cost == None:
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
            mp = node.mp
            _, sp = mp.get_sampled_position()
            sampled_path.append(sp + node.parent[:self.num_dims][:, np.newaxis] - mp.start_state[:self.num_dims][:, np.newaxis])
            path_cost += mp.cost
            node = self.node_dict[node.parent.tobytes()]
        path.append(self.start_state)

        path.reverse()
        sampled_path.reverse()
        return np.vstack(path).transpose(), np.hstack(sampled_path), path_cost

    def plot_path(self, path, sampled_path, path_cost):

        fig, ax = plt.subplots()
        ax.plot(self.start_state[0], self.start_state[1], 'og', zorder=5)
        ax.plot(self.goal_state[0], self.goal_state[1], 'or', zorder=5)

        if sampled_path is not None:
            print(path.T)
            ax.plot(sampled_path[0, :], sampled_path[1, :], zorder=4)
            print(f'cost: {path_cost}')
            ax.plot(path[0, :], path[1, :], 'co', zorder=4)
        ax.add_patch(plt.Circle(self.goal_state[:self.num_dims], self.goal_tolerance[0], color='b', fill=False, zorder=5))
        closed_nodes_states = np.array([node.state for node in self.closed_nodes]).T
        ax.plot(closed_nodes_states[0, :], closed_nodes_states[1, :], 'm*', zorder=3)
        neighbor_nodes_states = np.array([node.state for node in self.neighbor_nodes]).T
        ax.plot(neighbor_nodes_states[0, :], neighbor_nodes_states[1, :], '.', color=('.8'), zorder=2)
        self.map.plot(ax=ax)

    def get_neighbor_nodes_evenly_spaced(self, node):
        neighbor_mps = self.motion_primitive_graph.get_neighbor_mps(node.state, self.dt, self.num_u_per_dimension)
        neighbors = []
        for i, mp in enumerate(neighbor_mps):
            if self.map.is_mp_collision_free(mp, step_size=self.mp_sampling_step_size):
                state = mp.end_state
                neighbor_node = Node(mp.cost + node.g, self.heuristic(state), state, node.state, mp, graph_depth=node.graph_depth+1)
                neighbors.append(neighbor_node)
        node.is_closed = True
        return neighbors

    def get_neighbor_nodes_lattice(self, node):
        neighbors = []
        reset_map_index = int(np.floor(node.index / self.motion_primitive_graph.num_tiles))
        for i, mp in enumerate(self.motion_primitive_graph.edges[:, reset_map_index]):
            if mp is not None and self.map.is_mp_collision_free(mp, step_size=self.mp_sampling_step_size, offset=node.state[:self.num_dims]):
                state = deepcopy(node.state)
                state[:self.num_dims] += (mp.end_state - mp.start_state)[:self.num_dims]
                state[self.num_dims:] = mp.end_state[self.num_dims:]
                neighbor_node = Node(mp.cost + node.g, self.heuristic(state), state, node.state, mp,
                                     index=i, parent_index=node.index, graph_depth=node.graph_depth+1)
                neighbors.append(neighbor_node)
        node.is_closed = True
        return neighbors

    def reset_graph_search(self):
        self.node_dict = {}  # A dict where key is an state and the value is a node in the queue.
        self.queue = []      # A priority queue of nodes as a heapq.
        self.neighbor_nodes = []
        self.closed_nodes = []

        if not self.map.is_free_and_valid_position(self.start_state[:self.num_dims]):
            print("start invalid")
            self.queue = None
            return
        if not self.map.is_free_and_valid_position(self.goal_state[:self.num_dims]):
            print("goal invalid")
            self.queue = None
            return

        if type(self.motion_primitive_graph) is MotionPrimitiveLattice:
            starting_neighbors = self.motion_primitive_graph.find_mps_to_lattice(deepcopy(self.start_state) - self.start_position_offset)
            self.start_edges = []
            for i, mp in starting_neighbors:
                state = mp.end_state + self.start_position_offset
                node = Node(mp.cost, self.heuristic(state), state, None, mp, index=i, parent_index=None, graph_depth=0)
                self.start_edges.append(mp)
                if self.map.is_mp_collision_free(mp, step_size=self.mp_sampling_step_size, offset=self.start_position_offset):
                    heappush(self.queue, node)
                    self.node_dict[node.state.tobytes()] = node
        else:
            node = Node(0, self.heuristic(self.start_state), self.start_state, None, None, graph_depth=0)
            self.node_dict[node.state.tobytes()] = node
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
            if self.n == 3:  # Hack for ReedShepp
                state = np.zeros(self.n+1)
                state[:self.n] = node.state - self.goal_state
                norm = np.linalg.norm(state.reshape(self.control_space_q, self.num_dims), axis=1)
            else:
                norm = np.linalg.norm((node.state - self.goal_state).reshape(self.control_space_q, self.num_dims), axis=1)
            if (norm < self.goal_tolerance[:self.control_space_q]).all():
                print("Path found")
                path, sampled_path, path_cost = self.build_path(node)
                break

            # JUST FOR TESTING
            # if (nodes_expanded) > 50:
            #     break
            # if node.graph_depth > 5:
            #     break
            # if len(self.queue) > 30000:
            #     break

            neighbors = self.get_neighbor_nodes(node)
            for neighbor_node in neighbors:
                old_neighbor = self.node_dict.get(neighbor_node.state.tobytes(), None)
                if old_neighbor is None or neighbor_node.g < old_neighbor.g:
                    heappush(self.queue, neighbor_node)
                    self.node_dict[neighbor_node.state.tobytes()] = neighbor_node
                if old_neighbor is not None:
                    old_neighbor.is_closed = True
                self.neighbor_nodes.append(neighbor_node)  # for plotting

        if self.queue is not None:
            print()
            print(f"Nodes in queue at finish: {len(self.queue)}")
            print(f"Closed nodes in queue at finish: {sum(node.is_closed for node in self.queue)}")
            print()
            print(f"Nodes expanded: {nodes_expanded}")
            self.neighbor_nodes = np.array(self.neighbor_nodes)
            self.closed_nodes = np.array(self.closed_nodes)

        if path is None:
            print("No path found")
        return path, sampled_path, path_cost

    def animation_helper(self, i, closed_set_states):
        print(f"frame {i+1}/{len(self.closed_nodes)+10}")
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

            if type(self.motion_primitive_graph) is MotionPrimitiveLattice:
                iterator = enumerate(self.motion_primitive_graph.get_neighbor_mps(node.index))
            elif type(self.motion_primitive_graph) is MotionPrimitiveTree:
                iterator = enumerate(self.motion_primitive_graph.get_neighbor_mps(node.state, self.dt, self.num_u_per_dimension))
            for j, mp in iterator:
                _, sp = mp.get_sampled_position()
                shifted_sp = sp + node.state[:self.num_dims][:, np.newaxis] - mp.start_state[:self.num_dims][:, np.newaxis]
                open_list.append(shifted_sp[:, -1])
                self.lines[0][j].set_data(shifted_sp[0, :], shifted_sp[1, :])
            if open_list != []:
                self.open_list_states_animation = np.vstack((self.open_list_states_animation, np.array(open_list)))
        self.lines[3].set_data(closed_set_states[0, :i+1, ], closed_set_states[1, :i+1])
        self.lines[4].set_data(self.open_list_states_animation[:, 0], self.open_list_states_animation[:, 1])
        return self.lines

    def make_graph_search_animation(self, save_animation=False):
        plt.close('all')
        if self.queue is None:
            return
        if save_animation:
            import matplotlib
            normal_backend = matplotlib.get_backend()
            matplotlib.use("Agg")

        f, ax = plt.subplots(1, 1)
        self.map.plot(ax=ax)
        ax.axis('equal')

        mp_lines = []
        for j in range(self.num_mps):
            mp_lines.append(ax.plot([], [], linewidth=.4)[0])
        start_line, = ax.plot(self.start_state[0], self.start_state[1], 'og', zorder=4)
        goal_line, = ax.plot(self.goal_state[0], self.goal_state[1], 'or', zorder=4)
        closed_set_line, = ax.plot([], [], 'm*', zorder=3)
        open_set_line, = ax.plot([], [], '.', color=('.8'),  zorder=2)
        circle = plt.Circle(self.goal_state[:self.num_dims], self.goal_tolerance[0], color='b', fill=False, zorder=4)
        circle_patch = ax.add_artist(circle)
        self.lines = [mp_lines, start_line, goal_line, closed_set_line, open_set_line, circle_patch]
        closed_set = np.array([node.state for node in self.closed_nodes]).T
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
    from motion_primitives_py import *
    import time
    from pycallgraph import PyCallGraph, Config
    from pycallgraph.output import GraphvizOutput

    mpl = MotionPrimitiveLattice.load("lattice_test.json")
    print(mpl.dispersion)
    print(sum([1 for i in np.nditer(mpl.edges, ['refs_ok']) if i != None])/len(mpl.vertices))
    print(mpl.max_state)

    start_state = np.zeros((6))
    goal_state = np.zeros_like(start_state)

    resolution = .2
    origin = [0, 0]
    dims = [20, 20]
    data = np.zeros(dims)
    data[5:10, 4:6] = 100
    data[0:5, 11:13] = 100
    data = data.flatten('F')
    occ_map = OccupancyMap(resolution, origin, dims, data)
    start_state[0:3] = np.array([2, 1, 0])*resolution
    goal_state[0:3] = np.array([3, 15, 0])*resolution

    # occ_map = OccupancyMap.fromVoxelMapBag('trees_dispersion_1.1.bag', 0)
    # start_state[0:2] = [10, 6]
    # goal_state[0:2] = [22, 6]

    goal_tolerance = np.ones_like(start_state)*occ_map.resolution*3

    print("Motion Primitive Tree")
    mpt = MotionPrimitiveTree(mpl.control_space_q, mpl.num_dims,  mpl.max_state, InputsMotionPrimitive, plot=False)
    mpt.max_state[3] = 10
    gs = GraphSearch(mpt, occ_map, start_state[:mpl.n], goal_state[:mpl.n], goal_tolerance,
                     heuristic='euclidean', mp_sampling_step_size=occ_map.resolution/mpl.max_state[1])
    # with PyCallGraph(output=GraphvizOutput(output_file='tree.png'), config=Config(max_depth=15)):
    #     path, sampled_path, path_cost = gs.run_graph_search()
    tic = time.time()
    path, sampled_path, path_cost = gs.run_graph_search()
    toc = time.time()
    gs.plot_path(path, sampled_path, path_cost)
    print(f"Planning time: {toc - tic}s")
    # gs.make_graph_search_animation(True)

    print("Motion Primitive Lattice")
    mpl.plot = False
    gs = GraphSearch(mpl, occ_map, start_state[:mpl.n], goal_state[:mpl.n], goal_tolerance,
                     heuristic='min_time', mp_sampling_step_size=occ_map.resolution/mpl.max_state[1]/5)
    # with PyCallGraph(output=GraphvizOutput(output_file='lattice.png'), config=Config(max_depth=15)):
    #     path, sampled_path, path_cost = gs.run_graph_search()
    tic = time.time()
    path, sampled_path, path_cost = gs.run_graph_search()
    toc = time.time()
    gs.plot_path(path, sampled_path, path_cost)
    print(f"Planning time: {toc - tic}s")
    # gs.make_graph_search_animation(True)

    plt.show()
