#!/usr/bin/env python3

from motion_primitives_py import MotionPrimitive
import numpy as np
import matplotlib.pyplot as plt
from planning_ros_msgs.msg import SplineTrajectory, Spline, Polynomial
from copy import copy
from mav_traj_gen import *
import time


class ETHMotionPrimitive(MotionPrimitive):

    def __init__(self, start_state, end_state, num_dims, max_state, subclass_specific_data={}):
        super().__init__(start_state, end_state, num_dims, max_state, subclass_specific_data)
        self.is_valid = False
        self.traj_time = 0
        self.cost = np.inf

        self.calculate_trajectory()

        if self.is_valid:
            if self.subclass_specific_data.get('rho') is None:
                self.cost = self.traj_time
            else:
                self.cost = self.traj_time * self.subclass_specific_data['rho']
                st, su = self.get_sampled_input()
                self.cost += np.linalg.norm(np.sum((su)**2 * st, axis=1))

    def calculate_trajectory(self):
        dimension = self.num_dims

        derivative_to_optimize = derivative_order.SNAP

        start = Vertex(dimension)
        end = Vertex(dimension)

        start.addConstraint(derivative_order.POSITION, self.start_state[:self.num_dims])
        start.addConstraint(derivative_order.VELOCITY, self.start_state[self.num_dims:self.num_dims*2])
        end.addConstraint(derivative_order.POSITION, self.end_state[:self.num_dims])
        end.addConstraint(derivative_order.VELOCITY, self.end_state[self.num_dims:self.num_dims*2])
        if self.control_space_q > 2:
            start.addConstraint(derivative_order.ACCELERATION, self.start_state[self.num_dims*2:self.num_dims*3])
            end.addConstraint(derivative_order.ACCELERATION, self.end_state[self.num_dims*2:self.num_dims*3])

        vertices = [start, end]
        max_v = self.max_state[1]
        max_a = self.max_state[2]
        segment_times = estimateSegmentTimes(vertices, max_v, max_a)
        if segment_times[0] <= 0:
            return None
        parameters = NonlinearOptimizationParameters()

        opt = PolynomialOptimizationNonLinear(dimension, parameters)
        opt.setupFromVertices(vertices, segment_times, derivative_to_optimize)

        opt.addMaximumMagnitudeConstraint(derivative_order.VELOCITY, max_v)
        opt.addMaximumMagnitudeConstraint(derivative_order.ACCELERATION, max_a)

        result_code = opt.optimize()
        if result_code > 0:
            trajectory = Trajectory()
            opt.getTrajectory(trajectory)
            self.traj_time = trajectory.get_segment_times()[0]
            seg = trajectory.get_segments()[0]
            self.is_valid = True
            self.poly_coeffs = np.array([seg.getPolynomialsRef()[i].getCoefficients(0) for i in range(self.num_dims)])
            return seg
        return None

    def get_state(self, t, seg=None):
        if seg is None:
            seg = self.calculate_trajectory()
        if seg is not None:
            state = np.zeros(self.n)
            for i in range(self.control_space_q):
                state[self.num_dims*i:self.num_dims*(i+1)] = seg.evaluate(t, i)
            return state

    def get_sampled_states(self, step_size=0.1):
        """
        Return an array consisting of sample times and a sampling of the trajectory for plotting 
        Will be specific to the subclass, so we raise an error if the subclass has not implemented it
        """
        seg = self.calculate_trajectory()
        st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
        sampled_array = np.zeros((1+self.n, st.shape[0]))
        sampled_array[0, :] = st
        for i, t in enumerate(st):
            sampled_array[1:, i] = self.get_state(t, seg)
        return sampled_array

    def get_sampled_position(self, step_size=0.1):
        seg = self.calculate_trajectory()
        st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
        sp = np.zeros((1+self.num_dims, st.shape[0]))
        for i, t in enumerate(st):
            sp[1:, i] = seg.evaluate(t, 0)
        return st, sp

    def get_input(self, t):
        seg = self.calculate_trajectory()
        return seg.evaluate(t, self.control_space_q)

    def get_sampled_input(self, step_size=0.1):
        seg = self.calculate_trajectory()
        st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
        su = np.zeros((1+self.num_dims, st.shape[0]))
        for i, t in enumerate(st):
            su[1:, i] = seg.evaluate(t, self.control_space_q)
        return st, su


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from copy import deepcopy

    num_dims = 2
    control_space_q = 3

    start_state = np.zeros((num_dims * control_space_q,))
    end_state = np.ones((num_dims * control_space_q,))
    # end_state = np.random.rand(num_dims * control_space_q,)*2
    start_state[0] = 10
    start_state[1] = -5
    max_state = 2 * np.ones((control_space_q+1))

    mp = ETHMotionPrimitive(start_state, end_state, num_dims, max_state)

    mp.plot(position_only=False)
    # plt.plot(start_state[0], start_state[1], 'og')
    # plt.plot(end_state[0], end_state[1], 'or')
    plt.show()
