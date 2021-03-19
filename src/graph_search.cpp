
#include "motion_primitives/graph_search.h"

namespace motion_primitives {

std::vector<Node> GraphSearch::get_neighbor_nodes_lattice(Node node) {
  std::vector<Node> neighbor_nodes;
  int reset_map_index = floor(node.index_ / graph_.num_tiles_);
  // for (int i = 0; i < graph_.edges_.rows(); i++) {
  //   MotionPrimitive mp = graph_.edges_(i, reset_map_index);
  //   mp.translate(node.state_);
  //   // TODO collision check
  //   // if collision check succeeds:
  //   Node neighbor_node(mp.cost_ + node.g_, 0, mp.end_state_, node.state_, mp,
  //                      false, i, node.index_,
  //                      node.graph_depth_ + 1);  // TODO add heuristic
  //   neighbor_nodes.push_back(neighbor_node);
  // }
  // node.is_closed_ = true;
  // return neighbor_nodes;
}
}  // namespace motion_primitives
