#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import matplotlib.pyplot as plt
from motion_primitives_py import OccupancyMap, MotionPrimitiveLattice, GraphSearch
import rospkg
from planning_ros_msgs.msg import Trajectory, Primitive
from copy import deepcopy
# def map_callback(msg):
#     print("test")


class PlanningNode():
    def __init__(self):
        # rospy.Subscriber("/ddk/occupied_cells_vis_array", MarkerArray, map_callback)

        rospack = rospkg.RosPack()

        self.mpl = MotionPrimitiveLattice.load(rospack.get_path(
            'motion_primitives') + "/motion_primitives_py/data/lattices/dispersion100.json")
        print(self.mpl.max_state)
        rospy.Subscriber("/ddk/projected_map", OccupancyGrid, self.occ_grid_callback)
        rospy.Subscriber("/ddk/ground_truth/odom", Odometry, self.odom_callback)
        self.traj_pub = rospy.Publisher('/ddk/trackers_manager/mpl_tracker/goal', Trajectory, queue_size=10)

    def odom_callback(self, msg):
        self.odom_msg = msg
        self.odom_state = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.twist.twist.linear.x, msg.twist.twist.linear.y]

    def occ_grid_callback(self, msg):
        # data_reshaped = np.array(msg.data).reshape((msg.info.height, msg.info.width)) > 0
        occ_map = OccupancyMap(msg.info.resolution, (msg.info.origin.position.x,
                                                     msg.info.origin.position.y), (msg.info.width, msg.info.height), msg.data, unknown_is_free=True)
        goal_state = [3, -.2, 0, 0]
        gs = GraphSearch(self.mpl, occ_map, self.odom_state, goal_state, heuristic='min_time',
                         goal_tolerance=[1, 1, 1], mp_sampling_step_size=.01)
        gs.run_graph_search()

        traj = Trajectory()
        traj.header.stamp = rospy.Time.now()
        traj.header.frame_id = 'world'  # TODO need to rotate into local coord frame?
        ax = occ_map.plot()
        print([mp.start_state for mp in gs.mp_list])
        print([mp.end_state for mp in gs.mp_list])
        for mp in deepcopy(gs.mp_list):
            primitive = Primitive()
            primitive.cx = np.zeros(6,)
            primitive.cx[2:] = mp.polys[0, :]
            primitive.cx *= [120, 24, 6, 2, 1, 1]
            primitive.cy = np.zeros(6,)
            primitive.cy[2:] = mp.polys[1, :]
            primitive.cy *= [120, 24, 6, 2, 1, 1]
            primitive.cz = np.zeros(6,)
            primitive.cyaw = np.zeros(6,)
            primitive.cz[-1] = self.odom_msg.pose.pose.position.z
            primitive.t = mp.traj_time
            traj.primitives.append(primitive)
            t = np.arange(0, mp.traj_time, .01)
            mpl_coeffs = np.vstack([t**5/120, t**4/24, t**3/6, t**2/2, t, np.ones_like(t)])
            ax.plot(primitive.cx@mpl_coeffs, primitive.cy@mpl_coeffs)

        self.traj_pub.publish(traj)
        gs.plot_path(ax)
        plt.show()
        rospy.signal_shutdown("")


if __name__ == "__main__":
    rospy.init_node('planning_node', anonymous=True)
    PlanningNode()
    rospy.spin()
