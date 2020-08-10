import unittest
import numpy as np
from motion_primitives_py.jerks_motion_primitive import JerksMotionPrimitive
from motion_primitives_py.polynomial_motion_primitive import PolynomialMotionPrimitive
from motion_primitives_py.reeds_shepp_motion_primitive import ReedsSheppMotionPrimitive
from motion_primitives_py.motion_primitive_lattice import MotionPrimitiveLattice
from motion_primitives_py.occupancy_map import OccupancyMap
import tempfile
import os


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

    def test_lattice_save_load(self):
        control_space_q = 2
        num_dims = 2
        max_state = [2, 2*np.pi, 2*np.pi, 100, 1, 1]
        motion_primitive_type = ReedsSheppMotionPrimitive
        tiling = True
        mpl1 = MotionPrimitiveLattice(control_space_q, num_dims, max_state, motion_primitive_type, tiling)
        mpl1.compute_min_dispersion_space(num_output_pts=2, resolution=[.5, .5, np.inf, 25, 1, 1])
        with tempfile.TemporaryDirectory() as td:
            f_name = os.path.join(td, 'temp')
            with open(f_name, 'w') as tf:
                mpl1.save(tf.name)
            with open(f_name) as tf:
                mpl2 = MotionPrimitiveLattice.load(tf.name)
        assert((mpl1.vertices == mpl2.vertices).all())
        assert((mpl1.edges == mpl2.edges).all())
        assert(mpl1.control_space_q == mpl2.control_space_q)
        assert(mpl1.num_dims == mpl2.num_dims)
        assert((mpl1.max_state == mpl2.max_state).all())
        assert(mpl1.motion_primitive_type == mpl2.motion_primitive_type)
        assert(mpl1.num_tiles == mpl2.num_tiles)

    def test_occ_map(self):
        import matplotlib.pyplot as plt

        resolution = 1
        origin = [0, 0]
        dims = [10, 20]
        data = np.zeros(dims)
        data[5:10, 10:15] = 100
        data = data.flatten('F')
        om = OccupancyMap(resolution, origin, dims, data)
        om.plot()
        plt.show()

        occupied_valid_position = np.array([6, 6])
        invalid_position = np.array([11, 11])
        unoccupied_valid_position = np.array([1, 1])

        assert(om.is_free_and_valid_position(occupied_valid_position) == False)
        assert(om.is_valid_position(occupied_valid_position) == True)

        assert(om.is_free_and_valid_position(invalid_position) == False)
        assert(om.is_valid_position(invalid_position) == False)

        assert(om.is_free_and_valid_position(unoccupied_valid_position) == True)
        assert(om.is_valid_position(unoccupied_valid_position) == True)

        mp = PolynomialMotionPrimitive([7, 7, 0, 0], [7, 18, 0, 0], len(om.dims), [100, 100, 100, 100])
        assert(not om.is_mp_collision_free(mp))

        om.plot()
        mp.plot(position_only=True)
        plt.show()


if __name__ == '__main__':
    unittest.main()
