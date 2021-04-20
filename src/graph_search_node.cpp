#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <visualization_msgs/MarkerArray.h>

#include <boost/foreach.hpp>

#include "motion_primitives/graph_search.h"
#include "motion_primitives/graph_search2.h"
#include "motion_primitives/utils.h"

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

int main(int argc, char** argv) {
  ros::init(argc, argv, "motion_primitive_graph_search_cpp");
  ros::NodeHandle pnh("~");
  ros::Publisher traj_pub =
      pnh.advertise<planning_ros_msgs::Trajectory>("trajectory", 1, true);
  ros::Publisher traj_pub2 =
      pnh.advertise<planning_ros_msgs::Trajectory>("trajectory2", 1, true);
  ros::Publisher map_pub =
      pnh.advertise<planning_ros_msgs::VoxelMap>("voxel_map", 1, true);
  ros::Publisher sg_pub =
      pnh.advertise<visualization_msgs::MarkerArray>("start_and_goal", 1, true);
  ros::Publisher visited_pub =
      pnh.advertise<visualization_msgs::MarkerArray>("visited", 1, true);

  // Read map from bag file
  std::string map_file, map_topic, graph_file;
  pnh.param("map_file", map_file, std::string("voxel_map"));
  pnh.param("map_topic", map_topic, std::string("voxel_map"));
  auto voxel_map =
      read_bag<planning_ros_msgs::VoxelMap>(map_file, map_topic, 0).back();
  voxel_map.header.stamp = ros::Time::now();
  map_pub.publish(voxel_map);
  ROS_INFO("Publish map");

  pnh.param("graph_file", graph_file, std::string("dispersionopt101.json"));
  std::vector<double> s, g;
  Eigen::Vector4d start, goal;
  pnh.param("start_state", s, std::vector<double>{0, 0, 0, 0});
  pnh.param("goal_state", g, std::vector<double>{0, 0, 0, 0});
  start = Eigen::Map<Eigen::VectorXd>(s.data(), s.size());
  goal = Eigen::Map<Eigen::VectorXd>(g.data(), g.size());
  const auto mp_graph = read_motion_primitive_graph(graph_file);

  visualization_msgs::MarkerArray sg_markers;
  visualization_msgs::Marker start_marker, goal_marker;
  start_marker.header = voxel_map.header;
  start_marker.pose.position.x = start[0],
  start_marker.pose.position.y = start[1],
  start_marker.pose.position.z = start[2];
  start_marker.pose.orientation.w = 1;
  start_marker.color.g = 1;
  start_marker.color.a = 1;
  start_marker.type = 2;
  start_marker.scale.x = start_marker.scale.y = start_marker.scale.z = 0.3;
  goal_marker = start_marker;
  goal_marker.id = 1;
  goal_marker.pose.position.x = goal[0], goal_marker.pose.position.y = goal[1],
  goal_marker.pose.position.z = goal[2];
  goal_marker.color.g = 0;
  goal_marker.color.r = 1;
  sg_markers.markers.push_back(start_marker);
  sg_markers.markers.push_back(goal_marker);
  sg_pub.publish(sg_markers);

  if (true) {
    GraphSearch gs(mp_graph, start, goal, voxel_map);
    ROS_INFO("Started planning gs.");
    const auto start_time = ros::Time::now();
    const auto path = gs.run_graph_search();

    ROS_INFO("Finished planning. Planning time %f s",
             (ros::Time::now() - start_time).toSec());
    ROS_INFO_STREAM("path size: " << path.size());

    if (!path.empty()) {
      const auto traj = path_to_traj_msg(path, voxel_map.header);
      traj_pub.publish(traj);
    } else {
      ROS_WARN("No trajectory found.");
    }
  }

  {
    GraphSearch2 gs2(mp_graph, start, goal, voxel_map);
    ROS_INFO("Started planning gs2.");
    const auto start_time = ros::Time::now();
    const auto path = gs2.Search(start, goal, 0.5, true);
    const auto total_time = (ros::Time::now() - start_time).toSec();

    ROS_INFO("Finished planning. Planning time %f s", total_time);
    ROS_INFO_STREAM("path size: " << path.size());
    for (const auto& [k, v] : gs2.timings) {
      ROS_INFO_STREAM(k << ": " << v << "s, " << (v / total_time * 100) << "%");
    }

    if (!path.empty()) {
      const auto traj = path_to_traj_msg(path, voxel_map.header);
      traj_pub2.publish(traj);

      const auto visited_marray = StatesToMarkerArray(
          gs2.GetVisitedStates(), gs2.spatial_dim(), voxel_map.header);
      visited_pub.publish(visited_marray);
    } else {
      ROS_WARN("No trajectory found.");
    }
  }

  ros::spin();
}
