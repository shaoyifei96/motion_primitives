#!/usr/bin/env python
import rospy
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import matplotlib.pyplot as plt
from motion_primitives_py import OccupancyMap, MotionPrimitiveLattice, GraphSearch
import rospkg
from planning_ros_msgs.msg import Trajectory, Primitive

# def map_callback(msg):
#     print("test")


class PlanningNode():
    def __init__(self):
        # rospy.Subscriber("/ddk/occupied_cells_vis_array", MarkerArray, map_callback)

        rospack = rospkg.RosPack()

        self.mpl = MotionPrimitiveLattice.load(rospack.get_path(
            'motion_primitives') + "/motion_primitives_py/data/lattices/dispersion100.json")
        rospy.Subscriber("/ddk/projected_map", OccupancyGrid, self.occ_grid_callback)
        rospy.Subscriber("/ddk/ground_truth/odom", Odometry, self.odom_callback)
        self.traj_pub = rospy.Publisher('/ddk/trackers_manager/mpl_tracker/goal', Trajectory, queue_size=10)

    def odom_callback(self, msg):
        self.odom_state = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.twist.twist.linear.x, msg.twist.twist.linear.y]

    def occ_grid_callback(self, msg):
        # data_reshaped = np.array(msg.data).reshape((msg.info.height, msg.info.width)) > 0
        occ_map = OccupancyMap(msg.info.resolution, (msg.info.origin.position.x,
                                                     msg.info.origin.position.y), (msg.info.width, msg.info.height), msg.data, unknown_is_free=True)
        goal_state = [3, -.2, 0, 0]
        gs = GraphSearch(self.mpl, occ_map, self.odom_state, goal_state, heuristic='min_time', goal_tolerance=[])
        gs.run_graph_search()
        # ax = occ_map.plot()
        # gs.plot_path(path, sampled_path, path_cost, ax)
        # plt.show()
        t = Trajectory()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = 'world'
        for mp in gs.mp_list:
            primitive = Primitive()
            primitive.cx = mp.polys[0, :]
            primitive.cy = mp.polys[1, :]
            primitive.cz = np.zeros_like(mp.polys[0,:])
            primitive.cyaw = np.zeros_like(mp.polys[0,:])
            primitive.cyaw[-1] = 1
            primitive.cz[-1] = 1
            primitive.t = mp.traj_time
            t.primitives.append(primitive)
        self.traj_pub.publish(t)
        print(t)
        rospy.signal_shutdown("")


if __name__ == "__main__":
    rospy.init_node('planning_node', anonymous=True)
    PlanningNode()
    rospy.spin()
