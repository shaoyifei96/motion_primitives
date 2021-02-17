#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import OccupancyGrid #, Odometry
import numpy as np
import matplotlib.pyplot as plt
from motion_primitives_py import OccupancyMap, MotionPrimitiveLattice, GraphSearch
import rospkg
from copy import deepcopy
from traj_opt_msgs.msg import Trajectory, Spline, Polynomial
from action_planner.msg import PlanTwoPointAction, PlanTwoPointResult
from planning_ros_msgs.msg import VoxelMap
import actionlib


class PlanningServer():
    def __init__(self):
        rospack = rospkg.RosPack()
        self.mpl = MotionPrimitiveLattice.load(rospack.get_path(
            'motion_primitives') + "/motion_primitives_py/data/lattices/dispersion100.json")
        rospy.Subscriber("voxel_map", VoxelMap, self.voxel_map_callback, queue_size=1)
        self.traj_pub = rospy.Publisher("laura_traj", Trajectory, queue_size=1)

        # self.sub = rospy.Subscriber("odom", Odometry, self.odom_callback)
        self.occ_map = None
        self.map_frame = None
        # self.odom_state = None
        self.action_server = actionlib.SimpleActionServer(
            rospy.get_name() + "/plan_local_trajectory", PlanTwoPointAction, execute_cb=self.execute_cb, auto_start=False)
        self.action_server.start()  # recommended auto_start=False http://docs.ros.org/en/noetic/api/actionlib/html/classactionlib_1_1simple__action__server_1_1SimpleActionServer.html

    def execute_cb(self, msg):
        rospy.loginfo("Dispersion motion primitive action server goal received.")
        if self.occ_map is None:
            rospy.logerr("Missing voxel map for motion primitive planner, aborting action server.")
            self.action_server.set_aborted()
            return
        # if self.odom_state is None:
        #     rospy.logerr("Missing odometry for motion primitive planner, aborting action server.")
        #     self.action_server.set_aborted()
        #     return
        start_state = [msg.p_init.position.x, msg.p_init.position.y, msg.v_init.linear.x, msg.v_init.linear.y] #self.odom_state
        goal_state = [msg.p_final.position.x, msg.p_final.position.y, msg.v_final.linear.x, msg.v_final.linear.y]
        gs = GraphSearch(self.mpl, self.occ_map, start_state, goal_state, heuristic='min_time',
                         goal_tolerance=[1, 1, 1], mp_sampling_step_size=.1)
        rospy.loginfo("Starting graph search.")
        gs.run_graph_search()
        if gs.succeeded is False:
            rospy.logerr("Graph search failed, aborting action server.")
            self.action_server.set_aborted()
            return
        rospy.loginfo("Graph search succeeded.")

        result = PlanTwoPointResult()
        result.epoch = msg.epoch
        result.execution_time = msg.execution_time
        result.traj = Trajectory()#Trajectory_traj_opt()
        result.traj.header.stamp = rospy.Time.now()
        result.traj.header.frame_id = self.map_frame # same ref frame as map
        result.success = True
        for i in range(self.mpl.num_dims):
            spline = Spline()
            for mp in deepcopy(gs.mp_list):
                poly = Polynomial()
                poly.degree = mp.polys.shape[1] - 1
                poly.basis = 0
                poly.dt = mp.traj_time
                poly.coeffs = mp.polys[i, :]
                spline.segs.append(poly)
            spline.segments = len(gs.mp_list)
            spline.t_total = np.sum([mp.traj_time for mp in gs.mp_list])
            result.traj.data.append(spline)
        result.traj.dimensions = self.mpl.num_dims
        self.traj_pub.publish(result.traj)
        self.action_server.set_succeeded(result)

        # ax = self.occ_map.plot()
        # gs.plot_path(ax)
        # plt.show()

    def voxel_map_callback(self, msg):
        self.map_frame = msg.header.frame_id
        self.occ_map = OccupancyMap(msg.resolution, (msg.origin.x, msg.origin.y), (int(msg.dim.x),
                                                                                   int(msg.dim.y), int(msg.dim.z)), msg.data, unknown_is_free=True, force_2d=True)

    # def odom_callback(self, msg):
    #     self.odom_msg = msg
    #     self.odom_state = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.twist.twist.linear.x, msg.twist.twist.linear.y]


if __name__ == "__main__":
    rospy.init_node('planning_server')
    PlanningServer()
    rospy.spin()
