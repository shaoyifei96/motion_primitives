import unittest
import motion_primitive_test
import numpy as np
from motion_primitives_py.motion_primitive import JerksMotionPrimitive, PolynomialMotionPrimitive


class TestMotionPrimitive(unittest.TestCase):
    def setUp(self):
        self.start_state = np.array([0, 2, 0, 0, 0, 0])
        self.end_state = np.array([3, 1, 1, 0, 0, 0])
        self.num_dims = 2
        self.max_state = 100 * np.ones((6,))
        self.polynomial_mp = PolynomialMotionPrimitive(self.start_state, 
                                                       self.end_state, 
                                                       self.num_dims, 
                                                       self.max_state)
        self.jerk_mp = JerksMotionPrimitive(self.start_state, self.end_state, 
                                            self.num_dims, self.max_state)
    
    def test_polynomial_initial_state(self):
        s0 = self.polynomial_mp.get_state(np.array([0]))
        self.assertTrue(((np.squeeze(s0) - self.start_state) < 10e-5).all())

    def test_polynomial_final_state(self):
        tf = self.polynomial_mp.traj_time
        sf = self.polynomial_mp.get_state(np.array([tf]))
        self.assertTrue(((np.squeeze(tf) - self.start_state) < 10e-5).all())

    def test_jerks_initial_state(self):
        s0 = self.jerk_mp.get_state(np.array([0]))
        self.assertTrue(((np.squeeze(s0) - self.start_state) < 10e-5).all())

    def test_jerks_final_state(self):
        tf = self.jerk_mp.traj_time
        sf = self.jerk_mp.get_state(np.array([tf]))
        self.assertTrue(((np.squeeze(tf) - self.start_state) < 10e-5).all())

if __name__ == '__main__':
    unittest.main()
