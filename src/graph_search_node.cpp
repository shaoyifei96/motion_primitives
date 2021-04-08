#include <ros/ros.h>

#include <fstream>
#include <nlohmann/json.hpp>

#include "motion_primitives/graph_search.h"


int main(int argc, char **argv) {
  ros::init(argc, argv, "motion_primitive_graph_search_cpp");
  ros::NodeHandle n;
  ros::Publisher traj_pub = n.advertise<planning_ros_msgs::Trajectory>("trajectory", 10);


  std::string s(
      "/home/laura/dispersion_ws/src/motion_primitives/motion_primitives_py/"
      "data/lattices/opt2/dispersionopt101.json");
  std::ifstream json_file(s);
  nlohmann::json json_data;
  json_file >> json_data;
  auto graph = json_data.get<motion_primitives::MotionPrimitiveGraph>();
  ROS_INFO("Finished loading graph.");
  Eigen::Vector4d start(0, 0, 0, 0);
  Eigen::Vector4d goal(10, 10, 0, 0);
  motion_primitives::GraphSearch gs(graph, start, goal, "map");
  auto path = gs.run_graph_search();
  planning_ros_msgs::Trajectory traj = gs.path_to_traj_msg(path);
  traj_pub.publish(traj);

}
