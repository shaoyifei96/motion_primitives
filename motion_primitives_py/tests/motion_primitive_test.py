import unittest
import motion_primitive_test
import numpy as np
from motion_primitives_py.jerks_motion_primitive import JerksMotionPrimitive
from motion_primitives_py.polynomial_motion_primitive import PolynomialMotionPrimitive

class TestMotionPrimitive(unittest.TestCase):
    def setUp(self):
        self.start_state = np.array([0, 0, 0, 0, 0, 0])
        self.end_state = np.array([1, 1, 0, 0, 0, 0])
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
        tf = self.polynomial_mp.cost
        sf = self.polynomial_mp.get_state(np.array([tf]))
        self.assertTrue(((np.squeeze(sf) - self.end_state) < 10e-5).all())

    def test_polynomial_save_load(self):
        d = self.polynomial_mp.to_dict()
        mp2 = PolynomialMotionPrimitive.from_dict(d, self.num_dims, self.max_state)
        st1, sp1, sv1, sa1, sj1 = self.polynomial_mp.get_sampled_states()
        st2, sp2, sv2, sa2, sj2 = mp2.get_sampled_states()
        self.assertTrue((abs(st1 - st2) < 10e-5).all())
        self.assertTrue((abs(sp1 - sp2) < 10e-5).all())
        self.assertTrue((abs(sv1 - sv2) < 10e-5).all())
        self.assertTrue((abs(sa1 - sa2) < 10e-5).all())
        self.assertTrue((abs(sj1 - sj2) < 10e-5).all())

    def test_jerks_initial_state(self):
        s0 = self.jerk_mp.get_state(np.array([0]))
        self.assertTrue(((np.squeeze(s0) - self.start_state) < 10e-5).all())

    def test_jerks_final_state(self):
        tf = self.jerk_mp.cost
        sf = self.jerk_mp.get_state(np.array([tf]))
        self.assertTrue(((np.squeeze(sf) - self.end_state) < 10e-5).all())

    def test_jerks_save_load(self):
        d = self.jerk_mp.to_dict()
        mp2 = JerksMotionPrimitive.from_dict(d, self.num_dims, self.max_state)
        st1, sp1, sv1, sa1, sj1 = self.jerk_mp.get_sampled_states()
        st2, sp2, sv2, sa2, sj2 = mp2.get_sampled_states()
        self.assertTrue((abs(st1 - st2) < 10e-5).all())
        self.assertTrue((abs(sp1 - sp2) < 10e-5).all())
        self.assertTrue((abs(sv1 - sv2) < 10e-5).all())
        self.assertTrue((abs(sa1 - sa2) < 10e-5).all())
        self.assertTrue((abs(sj1 - sj2) < 10e-5).all())


if __name__ == '__main__':
    unittest.main()
