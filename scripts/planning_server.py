#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import OccupancyGrid #, Odometry
import numpy as np
import matplotlib.pyplot as plt
from motion_primitives_py import OccupancyMap, MotionPrimitiveLattice, GraphSearch
import rospkg
from copy import deepcopy
from action_planner.msg import PlanTwoPointAction, PlanTwoPointResult
from planning_ros_msgs.msg import Trajectory, Primitive, VoxelMap, SplineTrajectory, Spline, Polynomial
import actionlib

import time


class PlanningServer():
    def __init__(self, direct_run=False):
        rospack = rospkg.RosPack()
        self.mpl = MotionPrimitiveLattice.load(rospack.get_path(
            'motion_primitives') + "/motion_primitives_py/data_for_run/lattices/dispersion100.json")
        self.occ_map = None
        self.map_frame = None

        if direct_run:
            rospy.Subscriber("/quadrotor/mapper/local_voxel_map", VoxelMap, self.voxel_map_callback, queue_size=1)
            self.traj_vis_pub = rospy.Publisher("/quadrotor/laura_traj", Trajectory, queue_size=10)
            self.traj_opt_pub = rospy.Publisher("/quadrotor/laura_traj_traj_opt", SplineTrajectory, queue_size=10)
            self.action_server = actionlib.SimpleActionServer(
                "/quadrotor/local_plan_server/plan_local_trajectory", PlanTwoPointAction, execute_cb=self.execute_cb, auto_start=False)
        else:
            rospy.Subscriber("voxel_map", VoxelMap, self.voxel_map_callback, queue_size=1)
            self.traj_vis_pub = rospy.Publisher("laura_traj", Trajectory, queue_size=10)
            self.traj_opt_pub = rospy.Publisher("laura_traj_traj_opt", SplineTrajectory, queue_size=10)
            self.action_server = actionlib.SimpleActionServer(
                rospy.get_name() + "/plan_local_trajectory", PlanTwoPointAction, execute_cb=self.execute_cb, auto_start=False)

        self.action_server.start()  # recommended auto_start=False http://docs.ros.org/en/noetic/api/actionlib/html/classactionlib_1_1simple__action__server_1_1SimpleActionServer.html

    def execute_cb(self, msg):

        start_time = time.time()
        rospy.loginfo("Dispersion motion primitive action server goal received.")
        if self.occ_map is None:
            rospy.logerr("Missing voxel map for motion primitive planner, aborting action server.")
            self.action_server.set_aborted()
            return

        start_state = [msg.p_init.position.x, msg.p_init.position.y, msg.v_init.linear.x, msg.v_init.linear.y]
        goal_state = [msg.p_final.position.x, msg.p_final.position.y, msg.v_final.linear.x, msg.v_final.linear.y]
        gs = GraphSearch(self.mpl, self.occ_map, start_state, goal_state, heuristic='min_time',
                         goal_tolerance=[5, 1000, 1000], mp_sampling_step_size=.1) # tolerance is position, velocity, acceleration, jrk (only check up to control_space_q dim)
        rospy.loginfo("Starting graph search.")
        start_search = time.time()
        gs.run_graph_search()
        end_search = time.time()
        print('[Timer:] graph search time:', end_search-start_search)

        if gs.succeeded is False:
            rospy.logerr("Graph search failed, aborting action server.")
            self.action_server.set_aborted()
            return
        rospy.loginfo("Graph search succeeded.")

        result = PlanTwoPointResult()
        result.epoch = msg.epoch
        result.execution_time = msg.execution_time
        result.traj = SplineTrajectory()
        result.traj.header.stamp = rospy.Time.now()
        result.traj.header.frame_id = self.map_frame # same ref frame as map
        result.success = True


        traj_for_pub = Trajectory()
        traj_for_pub.header.stamp = rospy.Time.now()
        traj_for_pub.header.frame_id = self.map_frame # same ref frame as map

        additional_traj_dim = 0

        start_traj_assemble = time.time()
        for i in range(self.mpl.num_dims):
            spline = Spline()
            for mp in deepcopy(gs.mp_list):
                poly = Polynomial()
                poly.degree = mp.polys.shape[1] - 1
                poly.basis = 0
                poly.dt = mp.traj_time
                poly.coeffs = np.flip(mp.polys[i, :])  # need to reverse the order due to different conventions in spline polynomial and mp.polys
                spline.segs.append(poly)

            spline.segments = len(gs.mp_list)
            spline.t_total = np.sum([mp.traj_time for mp in gs.mp_list])
            result.traj.data.append(spline)

        # add z-dim
        if self.mpl.num_dims < 3:
            spline = Spline()
            for mp in deepcopy(gs.mp_list):
                poly = Polynomial()
                poly.degree = mp.polys.shape[1] - 1
                poly.basis = 0
                poly.dt = mp.traj_time
                # set poly with 0 coeffs for z direction
                poly.coeffs = np.zeros(mp.polys.shape[1])
                # set the z position to be start z position
                poly.coeffs[0] = msg.p_init.position.z
                spline.segs.append(poly)

            spline.segments = len(gs.mp_list)
            spline.t_total = np.sum([mp.traj_time for mp in gs.mp_list])
            result.traj.data.append(spline)
            additional_traj_dim = 1

        result.traj.dimensions = self.mpl.num_dims + additional_traj_dim
        self.action_server.set_succeeded(result)
       
       # publish trajectory (traj_opt type) for comparison with original planner 
        self.traj_opt_pub.publish(result.traj)

        # get traj for visualization:
        for mp in deepcopy(gs.mp_list):
            primitive = Primitive()
            primitive.cx = np.zeros(6, )
            primitive.cx[2:] = mp.polys[0, :]
            primitive.cx *= [120, 24, 6, 2, 1, 1]
            primitive.cy = np.zeros(6, )
            primitive.cy[2:] = mp.polys[1, :]
            primitive.cy *= [120, 24, 6, 2, 1, 1]
            primitive.cz = np.zeros(6, )
            primitive.cyaw = np.zeros(6, )
            primitive.cz[-1] = msg.p_init.position.z
            primitive.t = mp.traj_time
            traj_for_pub.primitives.append(primitive)
        # publish trajectory for visualization
        self.traj_vis_pub.publish(traj_for_pub)

        end_traj_assemble = time.time()
        print('[Timer:] traj_assemble time:', end_traj_assemble-start_traj_assemble)
        print('[Timer:] total time:', end_traj_assemble-start_time)

        # ax = self.occ_map.plot()
        # gs.plot_path(ax)
        # plt.show()


    def voxel_map_callback(self, msg):
        if self.occ_map is None:
            print('Map received!')
        self.map_frame = msg.header.frame_id
        self.occ_map = OccupancyMap(msg.resolution, (msg.origin.x, msg.origin.y), (int(msg.dim.x),
                                                                                   int(msg.dim.y), int(msg.dim.z)), msg.data, unknown_is_free=True, force_2d=True)


if __name__ == "__main__":
    rospy.init_node('planning_server')
    PlanningServer(direct_run=True)
    rospy.spin()
