
#include "motion_primitives/graph_search.h"

namespace motion_primitives {

double GraphSearch::heuristic(const Eigen::VectorXd& v) {
  CHECK_EQ(v.size(), goal_state_.size());
  Eigen::VectorXd x;
  x.resize(graph_.spatial_dim_);
  for (int i = 0; i < graph_.spatial_dim_; i++) {
    x(i) = v(i) - goal_state_(i);
  }
  return graph_.rho_ * x.lpNorm<Eigen::Infinity>() / graph_.max_state_(1);
}

std::vector<Node> GraphSearch::get_neighbor_nodes_lattice(const Node& node) {
  std::vector<Node> neighbor_nodes;
  CHECK_GE(node.index_, 0);
  int reset_map_index = floor(node.index_ / graph_.num_tiles_);
  for (int i = 0; i < graph_.edges_.rows(); i++) {
    if (graph_.edges_(i, reset_map_index) >= 0) {
      MotionPrimitive mp = graph_.mps_[graph_.edges_(i, reset_map_index)];
      mp.translate(node.state_);
      //   // TODO collision check
      //   // if collision check succeeds:
      Node neighbor_node(node.g_ + mp.cost_, heuristic(mp.end_state_),
                         mp.end_state_, node.state_, i, node.index_);
      neighbor_nodes.push_back(neighbor_node);
    }
  }
  return neighbor_nodes;
}

std::vector<MotionPrimitive> GraphSearch::run_graph_search() {
  Node start_node(1e-20, heuristic(start_state_), start_state_,
                  Eigen::VectorXd(1), 0);
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
  std::map<Eigen::VectorXd, Node> shortest_path_history;
  pq.push(start_node);
  shortest_path_history[start_state_] = start_node;
  while (!pq.empty() && ros::ok()) {
    Node current_node = pq.top();
    if ((current_node.state_ - goal_state_).norm() < 2) {
      return reconstruct_path(current_node, shortest_path_history);
      break;
    }
    pq.pop();
    for (auto& neighbor_node : get_neighbor_nodes_lattice(current_node)) {
      auto neighbor_past_g = shortest_path_history[neighbor_node.state_].g_;
      if (neighbor_node.g_ < neighbor_past_g || neighbor_past_g == 0) {
        pq.push(neighbor_node);
        shortest_path_history[neighbor_node.state_] = neighbor_node;
      }
    }
  }
}

std::vector<MotionPrimitive> GraphSearch::reconstruct_path(
    const Node& end_node,
    const std::map<Eigen::VectorXd, Node>& shortest_path_history) {
  // Build path from start point to goal point using the goal node's parents.
  double path_cost = 0;
  Node node = end_node;
  std::vector<MotionPrimitive> path;
  while (true && ros::ok()) {
    if (node.parent_state_.size() > 1) {
      int reset_map_index = floor(node.parent_index_ / graph_.num_tiles_);
      MotionPrimitive mp =
          graph_.mps_[graph_.edges_(node.index_, reset_map_index)];
      path_cost += mp.cost_;
      mp.translate(node.parent_state_);
      path.push_back(mp);

      node = shortest_path_history.at(node.parent_state_);
    } else
      break;
  }
  std::reverse(path.begin(), path.end());
  ROS_INFO_STREAM(path);
  return path;
}

planning_ros_msgs::Trajectory GraphSearch::path_to_traj_msg(
    const std::vector<motion_primitives::MotionPrimitive>& mp_vec) {
  Eigen::ArrayXXd pc_resized(graph_.spatial_dim_, 6);
  Eigen::ArrayXXd coeff_multiplier(pc_resized.rows(), pc_resized.cols());
  planning_ros_msgs::Trajectory trajectory;

  trajectory.header.stamp = ros::Time::now();
  trajectory.header.frame_id = msgs_frame_id_;

  for (int i = 0; i < pc_resized.rows(); i++) {
    coeff_multiplier.row(i) << 120, 24, 6, 2, 1, 1;
  }

  for (auto mp : mp_vec) {
    planning_ros_msgs::Primitive primitive;
    pc_resized.block(0, pc_resized.cols() - mp.poly_coeffs_.cols(),
                     pc_resized.rows(), mp.poly_coeffs_.cols()) =
        mp.poly_coeffs_;

    pc_resized *= coeff_multiplier;
    for (int i = 0; i < pc_resized.cols(); i++) {
      primitive.cx.push_back(pc_resized(0, i));
      primitive.cy.push_back(pc_resized(1, i));
      if (graph_.spatial_dim_ > 2) {
        primitive.cz.push_back(pc_resized(2, i));
      } else {
        primitive.cz.push_back(0.);
      }
      primitive.cyaw.push_back(0.);
    }
    primitive.t = mp.traj_time_;
    trajectory.primitives.push_back(primitive);
  }
  return trajectory;
}

std::ostream& operator<<(std::ostream& os, const Node& node) {
  os << node.state_.transpose() << std::endl;
  return os;
}

}  // namespace motion_primitives
