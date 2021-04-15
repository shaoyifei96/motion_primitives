#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <boost/foreach.hpp>
#include <sensor_msgs/PointCloud.h>
#include "motion_primitives/graph_search.h"

using namespace motion_primitives;
// num indicates the max number of elements to read, -1 means read till the end
template <class T>
std::vector<T> read_bag(std::string file_name, std::string topic,
                        unsigned int num) {
  rosbag::Bag bag;
  bag.open(file_name, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back(topic);
  rosbag::View view(bag, rosbag::TopicQuery(topics));

  std::vector<T> msgs;
  BOOST_FOREACH (rosbag::MessageInstance const m, view) {
    if (m.instantiate<T>() != NULL) {
      msgs.push_back(*m.instantiate<T>());
      if (msgs.size() > num) break;
    }
  }
  bag.close();
  if (msgs.empty())
    ROS_WARN("Fail to find '%s' in '%s', make sure md5sum are equivalent.",
             topic.c_str(), file_name.c_str());
  else
    ROS_INFO("Get voxel map data!");
  return msgs;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "motion_primitive_graph_search_cpp");
  ros::NodeHandle pnh("~");
  ros::Publisher traj_pub =
      pnh.advertise<planning_ros_msgs::Trajectory>("trajectory", 1, true);
  ros::Publisher map_pub =
      pnh.advertise<planning_ros_msgs::VoxelMap>("voxel_map", 1, true);
  ros::Publisher sg_pub =
      pnh.advertise<sensor_msgs::PointCloud>("start_and_goal", 1, true);

  // Read map from bag file
  std::string map_file, map_topic, graph_file;
  pnh.param("map_file", map_file, std::string("voxel_map"));
  pnh.param("map_topic", map_topic, std::string("voxel_map"));
  planning_ros_msgs::VoxelMap voxel_map =
      read_bag<planning_ros_msgs::VoxelMap>(map_file, map_topic, 0).back();

  pnh.param("graph_file", graph_file, std::string("dispersionopt101.json"));
  std::vector<double> s, g;
  Eigen::Vector4d start,goal;
  pnh.param("start_state", s, std::vector<double>{0,0,0,0});
  pnh.param("goal_state", g, std::vector<double>{0,0,0,0});
  start = Eigen::Map<Eigen::VectorXd>(s.data(), s.size());
  goal = Eigen::Map<Eigen::VectorXd>(g.data(), g.size());

  GraphSearch gs(read_motion_primitive_graph(graph_file), start, goal,
                 voxel_map);
  ROS_INFO("Started planning.");
  ros::Time planner_start_time = ros::Time::now();
  auto path = gs.run_graph_search();
  if (path.size() > 0) {
    planning_ros_msgs::Trajectory traj = gs.path_to_traj_msg(path);
    traj_pub.publish(traj);
  } else {
    ROS_WARN("No trajectory found.");
  }
  ROS_INFO("Finished planning. Planning time %f s",(ros::Time::now()-planner_start_time).toSec());
  map_pub.publish(voxel_map);

  sensor_msgs::PointCloud sg_cloud;
  sg_cloud.header = voxel_map.header;
  geometry_msgs::Point32 pt1, pt2;
  pt1.x = start[0], pt1.y = start[1], pt1.z = start[2];
  pt2.x = goal[0], pt2.y = goal[1], pt2.z = goal[2];
  sg_cloud.points.push_back(pt1), sg_cloud.points.push_back(pt2);
  sg_pub.publish(sg_cloud);

  ros::spin();
}
