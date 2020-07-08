import unittest
import motion_primitive_test
import numpy as np
from motion_primitives_py.motion_primitive import JerksMotionPrimitive, PolynomialMotionPrimitive


class TestMotionPrimitive(unittest.TestCase):
    def setUp(self):
        self.start_state = np.random.rand(6,)
        self.end_state = np.random.rand(6,)
        self.num_dims = 2
        self.max_state = 100 * np.ones((6,))
    
    def test_polynomial_initial_state(self):
        mp = PolynomialMotionPrimitive(self.start_state, self.end_state, 
                                       self.num_dims, self.max_state)
        s0 = mp.get_state(np.array([0]))
        self.assertTrue(((np.squeeze(s0) - self.start_state) < 10e-5).all())

    def test_jerks_initial_state(self):
        mp = JerksMotionPrimitive(self.start_state, self.end_state, 
                                  self.num_dims, self.max_state)
        s0 = mp.get_state(np.array([0]))
        self.assertTrue(((np.squeeze(s0) - self.start_state) < 10e-5).all())

if __name__ == '__main__':
    unittest.main()
