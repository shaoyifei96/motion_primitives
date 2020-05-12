import rospy
from octomap_msgs.msg import Octomap
import octomap


def octomap_callback(msg):
    octree = octomap.OcTree(msg.resolution)
    octomap.OcTree.read(msg.data)


rospy.init_node('octomap_listener', anonymous=True)
rospy.Subscriber("octomap_full", Octomap, octomap_callback)
rospy.spin()
