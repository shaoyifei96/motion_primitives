import unittest
import numpy as np
import matplotlib.pyplot as plt
from motion_primitives_py.occupancy_map import OccupancyMap
from motion_primitives_py.motion_primitive_lattice import MotionPrimitiveLattice
from motion_primitives_py.mp_graph_search import GraphSearch
from motion_primitives_py.reeds_shepp_motion_primitive \
    import ReedsSheppMotionPrimitive


class TestGraphSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):        
        # setup occupancy map
        resolution = 1
        origin = [0, 0]
        dims = [10, 20]
        data = np.zeros(dims)
        data[5:10, 10:15] = 100
        data = data.flatten('F')
        cls.om = OccupancyMap(resolution, origin, dims, data)

        # create motion primitive lattice
        control_space_q = 2
        num_dims = 2
        num_output_pts = 5
        max_state = [1, 2*np.pi, 2*np.pi, 100, 1, 1]
        resolution = [.2, .2, np.inf, 25, 1, 1]
        cls.mpl = MotionPrimitiveLattice(control_space_q, num_dims, max_state, 
                                          ReedsSheppMotionPrimitive)
        cls.mpl.compute_min_dispersion_space(num_output_pts, resolution)
        cls.mpl.limit_connections(2 * cls.mpl.dispersion)

        # define parameters for a graph search
        cls.start_state = [8, 2, 0]
        cls.goal_state = [8, 18, 0]
        cls.goal_tol = np.ones_like(cls.goal_state) * cls.mpl.dispersion

    def test_search(self):
        gs = GraphSearch(self.mpl, self.om, self.start_state, self.goal_state, 
                         self.goal_tol, heuristic='euclidean', neighbors='lattice')
        path, sampled_path, path_cost = gs.run_graph_search()
        assert(gs.queue)
        gs.make_graph_search_animation()


if __name__ == '__main__':
    unittest.main()
