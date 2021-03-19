#include <ros/ros.h>

#include <fstream>
#include <nlohmann/json.hpp>

#include "motion_primitives/graph_search.h"

namespace motion_primitives {
int main(int argc, char **argv) {
  ros::init(argc, argv, "motion_primitive_graph_search_cpp");
  ros::NodeHandle n;
  std::string s(
      "/home/laura/dispersion_ws/src/motion_primitives/motion_primitives_py/"
      "data/lattices/dispersion110.json");
  std::ifstream json_file(s);
  nlohmann::json json_data;
  json_file >> json_data;
  auto graph = json_data.get<motion_primitives::MotionPrimitiveGraph>();
  motion_primitives::GraphSearch gs(graph);
  // std::cout << gs.graph_.mps_.size() << std::endl;
  // motion_primitives::Node node;
  // gs.get_neighbor_nodes_lattice(node);
}
}  // namespace motion_primitives