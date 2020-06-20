import os
import rospy
from octomap_msgs.msg import Octomap
import octomap


def octomap_callback(msg):
    tree = octomap.OcTree(msg.resolution)

def create_OcTree():
    filename = os.path.join(os.path.dirname(__file__), '../data/fr_078_tidyup.bt')
    return octomap._octree_read(bytes(filename, encoding='utf-8'))

tree = create_OcTree()

rospy.init_node('octomap_listener', anonymous=True)
rospy.Subscriber("octomap_full", Octomap, octomap_callback)
rospy.spin()
