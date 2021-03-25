#ifndef MOTION_PRIMITIVES_GRAPH_SEARCH_H
#define MOTION_PRIMITIVES_GRAPH_SEARCH_H

#include <math.h>

#include "motion_primitives/motion_primitive_graph.h"

namespace motion_primitives {

class Node {
 public:
  Node(){};
  Node(float g, float h, const Eigen::VectorXd& state,
       const Eigen::VectorXd& parent_state, const MotionPrimitive& mp,
       bool is_closed = false, int index = -1, int parent_index = -1,
       int graph_depth = 0)
      : g_(g),
        h_(h),
        f_(g + h),
        state_(state),
        parent_state_(parent_state),
        index_(index),
        parent_index_(parent_index),
        is_closed_(is_closed),
        mp_(mp),
        graph_depth_(graph_depth) {}

  friend bool operator<(const Node& s, const Node& rhs);
  friend class GraphSearch;

 private:
  float f_;  // total-cost
  float g_;  // cost-to-come
  float h_;  // heuristic
  Eigen::VectorXd state_;
  Eigen::VectorXd parent_state_;
  MotionPrimitive mp_;
  int index_;
  int parent_index_;
  int graph_depth_;
  bool is_closed_;
};

bool operator<(const Node& n1, const Node& n2) { return n1.f_ < n2.f_; }

class GraphSearch {
private:
  MotionPrimitiveGraph graph_;
public:
  std::vector<Node> get_neighbor_nodes_lattice(Node node);
  GraphSearch(MotionPrimitiveGraph graph) : graph_(graph) {}
};

}  // namespace motion_primitives

#endif