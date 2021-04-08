#ifndef MOTION_PRIMITIVES_GRAPH_SEARCH_H
#define MOTION_PRIMITIVES_GRAPH_SEARCH_H

#include <planning_ros_msgs/Primitive.h>
#include <planning_ros_msgs/Trajectory.h>
#include <ros/init.h>

#include <queue>

#include "motion_primitives/motion_primitive_graph.h"

namespace Eigen {
bool operator<(Eigen::VectorXd const& a, Eigen::VectorXd const& b) {
  CHECK_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] < b[i]) return true;
    if (a[i] > b[i]) return false;
  }
  return false;
}
}  // namespace Eigen
namespace motion_primitives {

class Node {
  friend std::ostream& operator<<(std::ostream& out, const Node& node);

 private:
  double f_;  // total-cost
  double g_;  // cost-to-come
  double h_;  // heuristic
  Eigen::VectorXd state_;
  Eigen::VectorXd parent_state_;
  int index_;
  int parent_index_;
  friend class GraphSearch;

 public:
  Node() = default;
  Node(double g, double h, const Eigen::VectorXd& state,
       const Eigen::VectorXd& parent_state, int index, int parent_index = -1)
      : g_(g),
        h_(h),
        f_(g + h),
        state_(state),
        parent_state_(parent_state),
        index_(index),
        parent_index_(parent_index) {}

  friend bool operator>(const Node& n1, const Node& n2);
  friend std::ostream& operator<<(std::ostream& os, const MotionPrimitive& m);
};

bool operator>(const Node& n1, const Node& n2) { return n1.f_ > n2.f_; }

class GraphSearch {
 private:
  MotionPrimitiveGraph graph_;
  Eigen::VectorXd start_state_;
  Eigen::VectorXd goal_state_;
  std::string msgs_frame_id_;
  std::vector<Node> get_neighbor_nodes_lattice(const Node& node);
  double heuristic(const Eigen::VectorXd& v);
  std::vector<MotionPrimitive> reconstruct_path(
      const Node& end_node,
      const std::map<Eigen::VectorXd, Node>& shortest_path_history);

 public:
  planning_ros_msgs::Trajectory path_to_traj_msg(
      const std::vector<motion_primitives::MotionPrimitive>& mp_vec);
  std::vector<MotionPrimitive> run_graph_search();
  GraphSearch(const MotionPrimitiveGraph& graph,
              const Eigen::VectorXd& start_state,
              const Eigen::VectorXd& goal_state, std::string msgs_frame_id)
      : graph_(graph),
        start_state_(start_state),
        goal_state_(goal_state),
        msgs_frame_id_(msgs_frame_id) {}
};

}  // namespace motion_primitives

#endif