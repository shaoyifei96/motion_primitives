"""
Implements Octile heuristic and priority queue based on lexicographical sort of (f, -g).
"""

from heapq import heappush, heappop, heapify  # Recommended.
import numpy as np
import itertools
import copy

from flightsim.world import World

from .occupancy_map import OccupancyMap  # Recommended.


def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      select heuristic
                        False:       Dijkstra
                        True:        A* with Euclidean heuristic
                        'euclidean': A* with Euclidean heuristic
                        'octile':    A* with Octile heuristic
    Output:
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)

    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    goal_center_point = tuple(occ_map.index_to_metric_center(goal_index))

    class Node:
        """
        Container for node data. Nodes are sortable by the value of (f, -g).
        """

        def __init__(self, g, h, index, parent):
            self.f = g + h          # total-cost
            self.g = g              # cost-to-come
            self.h = h              # heuristic
            self.index = index      # (i,j,k)
            self.parent = parent    # (i,j,k)
            self.is_closed = False  # True if node has been closed.

        def __lt__(self, other):
            return (self.f, -self.g) < (other.f, -other.g)

        def __repr__(self):
            return f"Node g={self.g}, h={self.h}, index={self.index}, parent={self.parent}, is_closed={self.is_closed}"

    find_node = {}  # A dict where key is an index and the value is a node in the queue.
    Q = []         # A priority queue of nodes as a heapq.

    # Build lookup table of step distances.
    step_distance_table = np.zeros((3, 3, 3))
    for direction in itertools.product((-1, 0, 1), repeat=3):
        direction = np.array(direction)
        index = tuple(direction + 1)
        step_distance_table[index] = np.linalg.norm(direction * resolution)

    def get_step_distance(direction):
        """
        Return the step distance for an index direction (i,j,k).
        """
        index = tuple(np.array(direction)+1)
        return step_distance_table[index]

    def zero_heuristic(index):
        return 0

    def euclidean_heuristic(index):
        delta = occ_map.index_to_metric_center(index) - goal_center_point
        return np.linalg.norm(delta)

    def octile_heuristic(index):
        """
        Heuristic using the octile distance from node index to goal index. The
        octile distance is the Manhattan distance extended to diagonals, or the
        distance on the graph through an empty map.
        Implementation follows Jay Lanzafane, "Real-Time, 3D Path Planning for UAVs."
        """
        delta = np.abs(np.array(goal_index) - np.array(index))
        index = np.argsort(delta)
        delta = delta[index]

        dir_3 = [1, 1, 1]
        len_3 = get_step_distance(dir_3)
        dir_2 = np.array([0, 1, 1])[index]
        len_2 = get_step_distance(dir_2)
        dir_1 = np.array([0, 0, 1])[index]
        len_1 = get_step_distance(dir_1)
        return delta[0]*len_3 + (delta[1]-delta[0])*len_2 + (delta[2]-delta[1])*len_1

    def update_node_cost_to_come(index, g, parent):
        """
        Update a node with new cost-to-come g and parent.
        """
        old = find_node.get(index, None)
        if old is not None:
            old.is_closed = True
            new = Node(g, old.h, index, parent)
        else:
            h = heuristic(index)
            new = Node(g, h, index, parent)
        find_node[index] = new
        heappush(Q, new)

    def build_path(node):
        """
        Build path from start point to goal point using the goal node's parents.
        """
        # Build list of node centers.
        path = [occ_map.index_to_metric_center(node.index)]
        while node.parent:
            path.append(occ_map.index_to_metric_center(node.parent))
            node = find_node[node.parent]
        path.reverse()
        # Add true start and goal point.
        path.insert(0, start)
        path.append(goal)
        return np.array(path)

    # Choose a heuristic function based on input arguments.
    if astar == False:
        heuristic = zero_heuristic
    elif astar == True:
        heuristic = octile_heuristic
    elif astar == 'octile':
        heuristic = octile_heuristic
    elif astar == 'euclidean':
        heuristic = euclidean_heuristic

    # Initialize priority queue with start index.
    update_node_cost_to_come(start_index, 0, None)

    # While queue is not empty, pop the next smallest total cost f node.
    path = None
    nodes_expanded = 0
    while Q:
        node = heappop(Q)
        # If node has been closed already, skip.
        if node.is_closed:
            continue

        # If node is the goal node, return path.
        if node.index == goal_index:
            path = build_path(node)
            break

        # Otherwise, expand node and for each neighbor...
        nodes_expanded += 1
        for direction in itertools.product((-1, 0, 1), repeat=3):
            neighbor_index = (node.index[0]+direction[0], node.index[1]+direction[1], node.index[2]+direction[2])
            # If the neighbor is valid, calculate a new cost-to-come g.
            if occ_map.is_valid_index(neighbor_index) and not occ_map.is_occupied_index(neighbor_index):
                g = node.g + get_step_distance(direction)
                neighbor = find_node.get(neighbor_index, None)
                # If the cost-to-come g is better than previous, update the node cost and parent.
                if neighbor is None or g < neighbor.g:
                    update_node_cost_to_come(neighbor_index, g, parent=node.index)

    print()
    print(f"Nodes in queue at finish: {len(Q)}")
    print(f"Closed nodes in queue at finish: {sum(node.is_closed for node in Q)}")
    print()
    print(f"Nodes expanded: {nodes_expanded}")

    return path
