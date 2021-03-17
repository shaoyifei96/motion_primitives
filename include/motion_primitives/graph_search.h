#ifndef MOTION_PRIMITIVES_GRAPH_SEARCH_H
#define MOTION_PRIMITIVES_GRAPH_SEARCH_H

#include <math.h>
#include <ros/ros.h>

#include "motion_primitives/motion_primitive_graph.h"

namespace motion_primitives {
template <int state_dim>
class Node {
 public:
  float f_;  // total-cost
  float g_;  // cost-to-come
  float h_;  // heuristic
  Eigen::Matrix<float, state_dim, 1> state_;
  Eigen::Matrix<float, state_dim, 1> parent_state_;
  MotionPrimitive<state_dim> mp_;
  int index_;
  int parent_index_;
  int graph_depth_;
  bool is_closed_;
  Node(){};
  Node(float g, float h, Eigen::Matrix<float, state_dim, 1> state,
       Eigen::Matrix<float, state_dim, 1> parent_state,
       MotionPrimitive<state_dim> mp, bool is_closed = false, int index = -1,
       int parent_index = -1, int graph_depth = 0)
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
};
template <int state_dim>
bool operator<(const Node<state_dim> &n1, const Node<state_dim> &n2);

template <int state_dim>
class GraphSearch {
 public:
  MotionPrimitiveGraph<state_dim> graph_;
  std::vector<Node<state_dim>> get_neighbor_nodes_lattice(Node<state_dim> node);
  GraphSearch(MotionPrimitiveGraph<state_dim> graph) : graph_(graph) {}
};

}  // namespace motion_primitives

#endif