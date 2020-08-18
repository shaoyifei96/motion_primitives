import unittest
import numpy as np
import matplotlib.pyplot as plt
from motion_primitives_py import *


class TestOccupancyMap(unittest.TestCase):
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

        # define some test points
        cls.occupied_valid_pos = np.array([6, 11])
        cls.invalid_pos = np.array([11, 11])
        cls.unoccupied_valid_pos = np.array([1, 1])

        # define some test motion primitives
        cls.bad_mp = PolynomialMotionPrimitive([7, 7, 0, 0], [7, 18, 0, 0],
                                               len(cls.om.dims),
                                               [100, 100, 100, 100])
        cls.good_mp2 = PolynomialMotionPrimitive([11, 15, 0, 0], [7, 18, 0, 0],
                                               len(cls.om.dims),
                                               [100, 100, 100, 100])
        cls.good_mp = PolynomialMotionPrimitive([3, 7, 0, 0], [3, 18, 0, 0],
                                                len(cls.om.dims),
                                                [100, 100, 100, 100])

    def test_plot(self):
        self.om.plot()
        self.good_mp.plot(True)
        self.good_mp2.plot(True)
        self.bad_mp.plot(True)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    def test_is_free_and_valid_position(self):
        assert(not self.om.is_free_and_valid_position(self.occupied_valid_pos))
        assert(not self.om.is_free_and_valid_position(self.invalid_pos))
        assert(self.om.is_free_and_valid_position(self.unoccupied_valid_pos))

    def test_is_valid_position(self):
        assert(self.om.is_valid_position(self.occupied_valid_pos))
        assert(not self.om.is_valid_position(self.invalid_pos))
        assert(self.om.is_valid_position(self.unoccupied_valid_pos))

    def test_is_mp_collision_free(self):
        assert(self.om.is_mp_collision_free(self.bad_mp))
        assert(self.om.is_mp_collision_free(self.good_mp))
        assert(self.om.is_mp_collision_free(self.good_mp2))


if __name__ == '__main__':
    unittest.main()
