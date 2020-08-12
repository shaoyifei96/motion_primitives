import unittest
import numpy as np
import matplotlib.pyplot as plt
from motion_primitives_py.occupancy_map import OccupancyMap
from motion_primitives_py.motion_primitive_lattice import MotionPrimitiveLattice
from motion_primitives_py.mp_graph_search import GraphSearch
from motion_primitives_py.reeds_shepp_motion_primitive \
    import ReedsSheppMotionPrimitive


class TestGraphSearch(unittest.TestCase):
    def setUp(self):        
        # setup occupancy map
        resolution = 1
        origin = [0, 0]
        dims = [10, 20]
        data = np.zeros(dims)
        data[5:10, 10:15] = 100
        data = data.flatten('F')
        self.om = OccupancyMap(resolution, origin, dims, data)

        # create motion primitive lattice
        control_space_q = 2
        num_dims = 2
        num_output_pts = 5
        max_state = [1, 2*np.pi, 2*np.pi, 100, 1, 1]
        resolution = [.2, .2, np.inf, 25, 1, 1]
        self.mpl = MotionPrimitiveLattice(control_space_q, num_dims, max_state, 
                                          ReedsSheppMotionPrimitive)
        self.mpl.compute_min_dispersion_space(num_output_pts, resolution)
        self.mpl.limit_connections(2 * self.mpl.dispersion)

        # define parameters for a graph search
        self.start_state = [8, 2, 0]
        self.goal_state = [8, 18, 0]
        self.goal_tol = np.ones_like(self.goal_state) * self.mpl.dispersion

    def test_search(self):
        plt.plot(self.start_state[0], self.start_state[1], 'og')
        plt.plot(self.goal_state[0], self.goal_state[1], 'or')
        gs = GraphSearch(self.mpl, self.om, self.start_state, self.goal_state, 
                         self.goal_tol)
        gs.get_neighbors = gs.neighbor_type.LATTICE
        gs.heuristic = gs.heuristic_type.EUCLIDEAN
        path, sampled_path, path_cost = gs.run_graph_search()
        assert(gs.queue)

if __name__ == '__main__':
    unittest.main()
