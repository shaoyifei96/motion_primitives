
#include "motion_primitives/graph_search.h"

#include "motion_primitive_graph.cpp"  //TODO remove this, there is some CMake problem causing this

namespace motion_primitives {
// the top of the priority queue is the greatest element by default,
// but we want the smallest, so flip the sign
template <int state_dim>
bool operator<(const Node<state_dim> &n1, const Node<state_dim> &n2) {
  return n1.f > n2.f;
}

template <int state_dim>
std::vector<Node<state_dim>> GraphSearch<state_dim>::get_neighbor_nodes_lattice(
    Node<state_dim> node) {
  std::vector<Node<state_dim>> neighbor_nodes;
  int reset_map_index = floor(node.index_ / graph_.num_tiles_);
  for (int i = 0; i < graph_.edges_.rows(); i++) {
    MotionPrimitive<state_dim> mp = graph_.edges_(i, reset_map_index);
    if (mp.initialized_) {
      mp.translate(Eigen::Matrix<float, state_dim, 1>(node.state_.data()));
      // TODO collision check
      // if collision check succeeds:
      Node<state_dim> neighbor_node(
          mp.cost_ + node.g_, 0, mp.end_state_, node.state_, mp, false, i,
          node.index_, node.graph_depth_ + 1);  // TODO add heuristic
      neighbor_nodes.push_back(neighbor_node);
    }
  }
  node.is_closed_ = true;
  return neighbor_nodes;
}
}  // namespace motion_primitives

int main(int argc, char **argv) {
  ros::init(argc, argv, "motion_primitive_graph_search_cpp");
  ros::NodeHandle n;
  std::string s(
      "/home/laura/dispersion_ws/src/motion_primitives/motion_primitives_py/"
      "data/lattices/dispersion180.json");
  std::ifstream json_file(s);
  nlohmann::json json_data;
  json_file >> json_data;
  auto graph = json_data.get<motion_primitives::MotionPrimitiveGraph<4>>();
  motion_primitives::GraphSearch<4> gs(graph);
  motion_primitives::Node<4> node;
  gs.get_neighbor_nodes_lattice(node);
}
