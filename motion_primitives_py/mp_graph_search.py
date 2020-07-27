#!/usr/bin/python3

from motion_primitive_lattice import *
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

    def __init__(self, g, h, u, dt, state, parent, index=None, parent_index=None):
        self.f = g + h          # total-cost
        self.g = g              # cost-to-come
        self.h = h              # heuristic
        self.u = u
        self.dt = dt
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

    def __init__(self, motion_primitive_graph, start_state, goal_state, goal_tolerance, map_size=[-1, -1, -1, 1, 1, 1], plot=False):
        self.motion_primitive_graph = motion_primitive_graph
        self.start_state = np.array(start_state)
        self.goal_state = np.array(goal_state)
        self.goal_tolerance = np.array(goal_tolerance)
        self.map_size = np.array(map_size)
        self.plot = plot

        self.num_dims = self.motion_primitive_graph.num_dims
        self.control_space_q = self.motion_primitive_graph.control_space_q
        self.n = self.motion_primitive_graph.n

        self.min_state_comparator = np.hstack([np.array(map_size[:self.num_dims])] +
                                              [np.repeat(-i, self.num_dims) for i in self.motion_primitive_graph.max_state[:self.control_space_q-1]])
        self.max_state_comparator = np.hstack([np.array(map_size[self.num_dims:])] +
                                              [np.repeat(i, self.num_dims) for i in self.motion_primitive_graph.max_state[:self.control_space_q-1]])

        self.rho = 0.0

        # TODO I don't really think this is optimal stylistically/pythonically
        class HeuristicType(Enum):
            ZERO = self.zero_heuristic
            EUCLIDEAN = self.euclidean_distance_heuristic
            MIN_TIME = self.min_time_heuristic
        self.heuristic_type = HeuristicType
        self.heuristic = HeuristicType.EUCLIDEAN

        class NeighborType(Enum):
            MIN_DISPERSION = self.get_neighbors_min_dispersion
            EVENLY_SPACED = self.get_neighbors_evenly_spaced
            LATTICE = self.get_neighbors_lattice

        self.neighbor_type = NeighborType
        self.get_neighbors = NeighborType.MIN_DISPERSION
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
        return self.rho * np.linalg.norm(state[0:self.num_dims] - self.goal_state[0:self.num_dims], ord=np.inf)/self.motion_primitive_graph.max_state_derivs[0]

    def euclidean_distance_heuristic(self, state):
        return np.linalg.norm(state[0:self.num_dims] - self.goal_state[0:self.num_dims])

    def build_path(self, node):
        """
        Build path from start point to goal point using the goal node's parents.
        """
        path = [node.state + self.start_position_offset]
        sampled_path = []
        while node.parent_index is not None:
            path.append(node.parent + self.start_position_offset)
            node = self.node_dict[node.parent_index]
        path.append(self.start_state)
        st, sp, sv, sa, sj = self.start_edges[node.index].get_sampled_states()
        sampled_path.append(sp + self.start_state[:self.num_dims][:, np.newaxis])
        path.reverse()
        sampled_path.reverse()
        return np.array(path), np.squeeze(np.array(sampled_path))

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
        neighbors = self.motion_primitive_graph.create_evenly_spaced_mps(s, dt, num_u_per_dimension)
        return neighbors

    def get_neighbors_lattice(self, node):
        neighbors = []
        for i, mp in self.motion_primitive_graph.get_neighbors(node.index):
            state = mpl.vertices[i]
            g = mp.cost
            dt = mp.cost
            h = gs.heuristic(state)
            parent = self.start_state
            index = i
            parent_index = None
            node = Node(g, h, None, dt, state, parent, index, parent_index)
            neighbors.append(node)
        return neighbors

    def reset_graph_search(self):
        self.node_dict = {}  # A dict where key is an state and the value is a node in the queue.
        self.queue = []      # A priority queue of nodes as a heapq.
        self.neighbor_list = []
        self.expanded_nodes_list = []

        if self.neighbor_type.LATTICE == self.get_neighbors:
            starting_neighbors = self.motion_primitive_graph.find_mps_to_lattice(deepcopy(self.start_state) - self.start_position_offset)
            self.start_edges = []
            for i, mp in starting_neighbors:
                state = mpl.vertices[i]
                g = mp.cost
                dt = mp.cost
                h = gs.heuristic(state)
                parent = self.start_state
                index = i
                parent_index = None
                node = Node(g, h, None, dt, state, parent, index, parent_index)
                heappush(self.queue, node)
                self.start_edges.append(mp)
        else:
            node = Node(0, 0, 0, 0, self.start_state, None)
            heappush(self.queue, node)

    def run_graph_search(self, neighbor_method="min_dispersion"):
        self.reset_graph_search()

        # # While queue is not empty, pop the next smallest total cost f node.
        path = None
        sampled_path = None
        nodes_expanded = 0
        # print(self.queue)
        while self.queue:
            node = heappop(self.queue)
            # If node has been closed already, skip.
            if node.is_closed:
                continue
            node.is_closed = True
            # If node is the goal node, return path.
            # TODO separately compare derivative for goal tolerance
            if np.linalg.norm(node.state[:self.num_dims] - (self.goal_state[:self.num_dims]-self.start_position_offset[:self.num_dims])) < self.goal_tolerance[0]:
                path, sampled_path = self.build_path(node)
                break

            # Otherwise, expand node and for each neighbor...
            nodes_expanded += 1

            # JUST FOR TESTING
            # if (nodes_expanded) > 100:
            #     break

            neighbors = self.get_neighbors(node)
            for neighbor_node in neighbors:
                # If the neighbor is valid, calculate a new cost-to-come g.
                # if self.is_valid_state(neighbor_node.state):
                neighbor_node.g = node.g + neighbor_node.g

                old_neighbor = self.node_dict.get(neighbor_node.index, None)
                if old_neighbor is None or neighbor_node.g < old_neighbor.g:
                    heappush(self.queue, neighbor_node)
                    self.node_dict[neighbor_node.index] = neighbor_node
                    plt.plot(neighbor_node.state[0]+self.start_position_offset[0],
                             neighbor_node.state[1]+self.start_position_offset[1], '.')

        print()
        print(f"Nodes in queue at finish: {len(self.queue)}")
        print(f"Closed nodes in queue at finish: {sum(node.is_closed for node in self.queue)}")
        print()
        print(f"Nodes expanded: {nodes_expanded}")

        if path is None:
            print("No path found")
        return path, sampled_path

    # def plot_path(self, path, poly, fig=None, axs=None):
    #     if fig is None:
    #         fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    #         ax = axs[0][0]
    #     else:
    #         ax = axs[1][0]
    #     ax.set_aspect('equal', 'box')
    #     ax.set_title((self.get_neighbors.__name__, self.heuristic.__name__))
    #     ax.plot(self.start_state[0], self.start_state[1], 'og')
    #     ax.plot(self.goal_state[0], self.goal_state[1], 'or')
    #     circle = plt.Circle(self.goal_state[:self.num_dims], self.goal_tolerance[0], color='b', fill=False)
    #     ax.add_artist(circle)
    #     positions = path[:, :self.num_dims]
    #     poly_positions = poly[:, :self.num_dims]
    #     ax.plot(positions[:, 0], positions[:, 1], 'o')
    #     ax.plot(poly_positions[:, 0], poly_positions[:, 1], '-')
    #     ax.tick_params(reset=True)
    #     return fig, axs

    # def plot_all_nodes(self, fig=None, axs=None):
    #     if fig is None:
    #         fig = plt.gcf()
    #         ax = axs[0][1]
    #     else:
    #         ax = axs[1][1]
    #     ax.set_aspect('equal', 'box')
    #     ax.plot(self.start_state[0], self.start_state[1], 'og')
    #     ax.plot(self.goal_state[0], self.goal_state[1], 'or')
    #     circle = plt.Circle(self.goal_state[:self.num_dims], self.goal_tolerance[0], color='b', fill=False)
    #     ax.add_artist(circle)
    #     n = np.array(self.neighbor_list)
    #     ax.plot(n[:, 0], n[:, 1], 'k.')
    #     m = np.array(self.expanded_nodes_list)
    #     ax.plot(m[:, 0], m[:, 1], 'c.')
    #     ax.tick_params(reset=True)


if __name__ == "__main__":
    mpl = MotionPrimitiveLattice.load("lattice_test.json")

    map_size = [-2, -2, 2, 2]
    start_state = np.ones((mpl.n))*10.9
    goal_state = np.ones_like(start_state)*9.2
    plt.plot(mpl.vertices[:, 0] + start_state[0], mpl.vertices[:, 1] + start_state[1], '*k')

    # goal_state[0] = .7
    # goal_state[1] = 1.8
    goal_tolerance = np.ones_like(start_state)*.5
    plot = True
    gs = GraphSearch(mpl, start_state, goal_state, goal_tolerance, map_size, plot)
    gs.get_neighbors = gs.neighbor_type.LATTICE
    path, sampled_path = gs.run_graph_search()
    plt.plot(gs.start_state[0], gs.start_state[1], 'og')
    plt.plot(gs.goal_state[0], gs.goal_state[1], 'or')

    if sampled_path is not None:
        plt.plot(sampled_path[0, :], sampled_path[1, :])
    plt.show()
