"""
Implements Octile heuristic and priority queue based on lexicographical sort of (f, -g).
"""

from heapq import heappush, heappop, heapify  # Recommended.
import numpy as np
import itertools
import copy
from min_dispersion_primitives import MotionPrimitive
import pickle
from scipy import spatial
import matplotlib.pyplot as plt
from enum import Enum


class Node:
    """
    Container for node data. Nodes are sortable by the value of (f, -g).
    """

    def __init__(self, g, h, state, parent):
        self.f = g + h          # total-cost
        self.g = g              # cost-to-come
        self.h = h              # heuristic
        self.state = state      # (i,j,k)
        self.parent = parent    # (i,j,k)
        self.is_closed = False  # True if node has been closed.

    def __lt__(self, other):
        return (self.f, -self.g) < (other.f, -other.g)

    def __repr__(self):
        return f"Node g={self.g}, h={self.h}, state={self.state}, parent={self.parent}, is_closed={self.is_closed}"


class GraphSearch:

    def __init__(self, motion_primitive, start_state, goal_state, goal_tolerance, map_size=[-1, -1, -1, 1, 1, 1], plot=False, heuristic_type=1, neighbor_type=1):
        self.motion_primitive = motion_primitive
        self.start_state = np.array(start_state)
        self.goal_state = np.array(goal_state)
        # print(self.start_state.T)
        # print(self.goal_state.T)
        self.goal_tolerance = np.array(goal_tolerance)
        self.map_size = np.array(map_size)
        self.plot = plot
        self.heuristic_type = heuristic_type
        self.neighbor_type = neighbor_type

        self.num_dims = self.motion_primitive.num_dims
        self.control_space_q = self.motion_primitive.control_space_q
        self.n = self.motion_primitive.n

        self.min_state_comparator = np.hstack([np.array(map_size[:self.num_dims])] +
                                              [np.repeat(-i, self.num_dims) for i in self.motion_primitive.max_state_derivs[:self.control_space_q-1]])
        self.max_state_comparator = np.hstack([np.array(map_size[self.num_dims:])] +
                                              [np.repeat(i, self.num_dims) for i in self.motion_primitive.max_state_derivs[:self.control_space_q-1]])

        self.node_dict = {}  # A dict where key is an state and the value is a node in the queue.
        self.queue = []      # A priority queue of nodes as a heapq.

        # Choose a heuristic function based on input arguments.
        # self.heuristic = self.zero_heuristic
        self.rho = .1
        # self.heuristic = self.min_time_heuristic
        # self.heuristic = self.euclidean_distance_heuristic

        class HeuristicType(Enum):
            ZERO = self.zero_heuristic
            EUCLIDEAN = self.euclidean_distance_heuristic
            MIN_TIME = self.min_time_heuristic
        self.heuristic_type = HeuristicType
        self.heuristic = HeuristicType.EUCLIDEAN

        class NeighborType(Enum):
            MIN_DISPERSION = self.min_dipsersion_neighbors
            EVENLY_SPACED = self.evenly_spaced_neighbors

        self.neighbor_type = NeighborType
        self.get_neighbors = NeighborType.MIN_DISPERSION

        self.mp_start_pts_tree = spatial.KDTree(self.motion_primitive.start_pts.T)

        if self.plot:
            fig = plt.figure()
            self.ax = fig.add_subplot(111)
            self.ax.plot(start_state[0], start_state[1], 'og')
            self.ax.plot(goal_state[0], goal_state[1], 'or')
            circle = plt.Circle(goal_state[:self.num_dims], self.goal_tolerance[0], color='b', fill=False)
            self.ax.add_artist(circle)
            plt.ion()
            plt.show()

    def is_valid_state(self, state):
        if (state < self.min_state_comparator).any() or (state > self.max_state_comparator).any():
            return False
        return True

    def zero_heuristic(self, state):
        return 0

    def min_time_heuristic(self, state):
        # sikang heuristic 1
        return self.rho * np.linalg.norm(state[0:self.num_dims] - self.goal_state[0:self.num_dims], ord=np.inf)/self.motion_primitive.max_state_derivs[0]

    def euclidean_distance_heuristic(self, state):
        return np.linalg.norm(state[0:self.num_dims] - self.goal_state[0:self.num_dims])

    def update_node_cost_to_come(self, state, g, parent):
        """
        Update a node with new cost-to-come g and parent.
        """
        state = tuple(state.tolist())
        old = self.node_dict.get(state, None)
        if old is not None:
            old.is_closed = True
            new = Node(g, old.h, state, parent)
        else:
            h = self.heuristic(state)
            new = Node(g, h, state, parent)
        self.node_dict[state] = new
        heappush(self.queue, new)

    def build_path(self, node):
        """
        Build path from start point to goal point using the goal node's parents.
        """
        # Build list of node centers.
        path = [node.state]
        while node.parent:
            path.append(node.parent)
            node = self.node_dict[node.parent]
        path.reverse()
        return np.array(path)

    def min_dipsersion_neighbors(self, node):
        start_pt = np.array(node.state)
        closest_start_pt_index = self.mp_start_pts_tree.query(start_pt)[1]
        motion_primitives_list = self.motion_primitive.motion_primitives_list[closest_start_pt_index]
        neighbors = []
        for column in motion_primitives_list.T:
            neighbors.append(np.concatenate(
                (self.motion_primitive.quad_dynamics(start_pt, column[1:], column[0]), column)))

        return np.array(neighbors)

    def evenly_spaced_neighbors(self, node):
        dt = .2
        s = np.reshape(np.array(node.state), (self.n, 1))
        neighbors = self.motion_primitive.create_evenly_spaced_mps(s, dt)
        return neighbors

    def run_graph_search(self, neighbor_method="min_dispersion"):
        # # Initialize priority queue with start index.
        self.update_node_cost_to_come(self.start_state, 0, None)

        # # While queue is not empty, pop the next smallest total cost f node.
        path = None
        nodes_expanded = 0
        while self.queue:
            node = heappop(self.queue)
            if self.plot:
                self.ax.plot(node.state[0], node.state[1], 'bo')

            # If node has been closed already, skip.
            if node.is_closed:
                continue

            # If node is the goal node, return path.
            # TODO separately compare derivative for goal tolerance
            if np.linalg.norm(node.state[:self.num_dims] - self.goal_state[:self.num_dims]) < self.goal_tolerance[0]:
                path = self.build_path(node)
                break

            # Otherwise, expand node and for each neighbor...
            nodes_expanded += 1

            neighbors = self.get_neighbors(node)

            for neighbor in neighbors:
                # print(neighbor)
                # If the neighbor is valid, calculate a new cost-to-come g.
                neighbor_state = neighbor[:self.n]
                if self.is_valid_state(neighbor_state):
                    dt = neighbor[self.n]
                    u = neighbor[-self.num_dims:]
                    g = node.g + dt*(self.rho + (np.linalg.norm(u))**2)

                    old_neighbor = self.node_dict.get(tuple(neighbor_state.tolist()), None)
                    if old_neighbor is None or g < old_neighbor.g:
                        self.update_node_cost_to_come(neighbor_state, g, parent=node.state)
                        if self.plot:
                            self.ax.plot(neighbor_state[0], neighbor_state[1], 'ko')
                            plt.pause(.001)

            #         neighbor = find_node.get(neighbor_index, None)
            #         # If the cost-to-come g is better than previous, update the node cost and parent.
            #         if neighbor is None or g < neighbor.g:
            #             update_node_cost_to_come(neighbor_index, g, parent=node.index)

        print()
        print(f"Nodes in queue at finish: {len(self.queue)}")
        print(f"Closed nodes in queue at finish: {sum(node.is_closed for node in self.queue)}")
        print()
        print(f"Nodes expanded: {nodes_expanded}")

        return path

    def plot_path(self, path):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(start_state[0], start_state[1], 'og')
        ax.plot(goal_state[0], goal_state[1], 'or')
        circle = plt.Circle(goal_state[:self.num_dims], self.goal_tolerance[0], color='b', fill=False)
        ax.add_artist(circle)
        positions = path[:, :self.num_dims]
        ax.plot(positions[:, 0], positions[:, 1], 'o')
        plt.show()


if __name__ == "__main__":

    with open('pickle/MotionPrimitive.pkl', 'rb') as input:
        mp = pickle.load(input)

    start_state = np.zeros((mp.n))
    goal_state = np.ones_like(start_state)
    goal_tolerance = np.ones_like(start_state)*.1
    map_size = [-1, -1, 1, 1]
    plot = True
    gs = GraphSearch(mp, start_state, goal_state, goal_tolerance, map_size, plot)
    gs.heuristic = gs.heuristic_type.EUCLIDEAN
    gs.get_neighbors = gs.neighbor_type.MIN_DISPERSION

    path = gs.run_graph_search()
    plt.ioff()
    gs.plot_path(path)
