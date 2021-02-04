#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import matplotlib.pyplot as plt
from motion_primitives_py import OccupancyMap, MotionPrimitiveLattice, GraphSearch
import rospkg
from copy import deepcopy
from autonomy_map_plan_msgs.msg import Trajectory_mpl, Trajectory_traj_opt, Spline, Polynomial
from autonomy_map_plan_msgs.msg import PlanTwoPointAction, PlanTwoPointActionResult, VoxelMap
import actionlib


class PlanningServer():
    def __init__(self):
        rospack = rospkg.RosPack()
        self.mpl = MotionPrimitiveLattice.load(rospack.get_path(
            'motion_primitives') + "/motion_primitives_py/data/lattices/dispersion100.json")
        rospy.Subscriber("voxel_map", VoxelMap, self.voxel_map_callback, queue_size=1)
        self.sub = rospy.Subscriber("odom", Odometry, self.odom_callback)
        self.occ_map = None
        self.odom_state = None
        self.action_server = actionlib.SimpleActionServer(
            rospy.get_name(), PlanTwoPointAction, execute_cb=self.execute_cb, auto_start=False)
        self.action_server.start()  # recommended auto_start=False http://docs.ros.org/en/noetic/api/actionlib/html/classactionlib_1_1simple__action__server_1_1SimpleActionServer.html

    def execute_cb(self, msg):
        rospy.loginfo("Dispersion motion primitive action server goal received.")
        if self.occ_map is None:
            rospy.logerr("Missing voxel map for motion primitive planner, aborting action server.")
            self.action_server.set_aborted()
            return
        if self.odom_state is None:
            rospy.logerr("Missing odometry for motion primitive planner, aborting action server.")
            self.action_server.set_aborted()
            return
        start_state = self.odom_state
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

        result_msg = PlanTwoPointActionResult()
        result_msg.header.stamp = rospy.Time.now()
        result_msg.result.traj = Trajectory_traj_opt()
        result_msg.result.traj.header.stamp = rospy.Time.now()
        result_msg.result.traj.header.frame_id = 'world'  # TODO need to rotate into local coord frame?
        # result.success = True
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
            result_msg.result.traj.data.append(spline)
        result_msg.result.traj.dimensions = self.mpl.num_dims
        self.action_server.set_succeeded(result_msg)

        # ax = self.occ_map.plot()
        # gs.plot_path(ax)
        # plt.show()

    def voxel_map_callback(self, msg):
        self.occ_map = OccupancyMap(msg.resolution, (msg.origin.x, msg.origin.y), (int(msg.dim.x),
                                                                                   int(msg.dim.y), int(msg.dim.z)), msg.data, unknown_is_free=True, force_2d=True)

    def odom_callback(self, msg):
        self.odom_msg = msg
        self.odom_state = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.twist.twist.linear.x, msg.twist.twist.linear.y]


if __name__ == "__main__":
    rospy.init_node('planning_server')
    PlanningServer()
    rospy.spin()
