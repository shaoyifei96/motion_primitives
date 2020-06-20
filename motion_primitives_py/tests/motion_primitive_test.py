import unittest
import motion_primitive_test
import numpy as np
from motion_primitives_py.motion_primitive import JerksMotionPrimitive, PolynomialMotionPrimitive


class TestMotionPrimitive(unittest.TestCase):

    def test_polynomial_get_state(self):

        _start_state = np.random.rand(6,)
        _end_state = np.random.rand(6,)
        _num_dims = 2
        _max_state = np.ones((6,))*100
        mp = PolynomialMotionPrimitive(_start_state, _end_state, _num_dims, _max_state)
        s0 = mp.get_state(0)
        self.assertTrue(((np.squeeze(s0)-_start_state) < 10e-5).all())


if __name__ == '__main__':
    unittest.main()
