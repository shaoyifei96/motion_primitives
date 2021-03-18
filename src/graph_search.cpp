
#include "motion_primitives/graph_search.h"

// #include "motion_primitive_graph.cpp"  //TODO remove this, there is some
// CMake problem causing this

namespace motion_primitives {
// the top of the priority queue is the greatest element by default,
// but we want the smallest, so flip the sign

std::vector<Node<state_dim>> GraphSearch::get_neighbor_nodes_lattice(
    Node node) {
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
