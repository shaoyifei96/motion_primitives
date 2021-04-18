#include "motion_primitives/graph_search.h"

#include <planning_ros_msgs/Primitive.h>
#include <ros/init.h>

namespace motion_primitives {

namespace {
// Check if two position are within d meters apart
bool position_within(const Eigen::Ref<const Eigen::VectorXd>& p1,
                     const Eigen::Ref<const Eigen::VectorXd>& p2, double d) {
  return (p1 - p2).squaredNorm() < (d * d);
}

}  // namespace

Node::Node(double g, double h, const Eigen::VectorXd& state, int index)
    : cost_to_come_(g),
      heuristic_cost_(h),
      total_cost_(g + h),
      state_(state),
      index_(index) {}

Eigen::Vector3i GraphSearch::get_indices_from_position(
    const Eigen::Vector3d& position) const {
  return floor(((position - map_origin_) / voxel_map_.resolution).array())
      .cast<int>();
}

int GraphSearch::get_linear_indices(const Eigen::Vector3i& indices) const {
  return indices[0] + map_dims_[0] * indices[1] +
         map_dims_[0] * map_dims_[1] * indices[2];
}

bool GraphSearch::is_valid_indices(const Eigen::Vector3i& indices) const {
  // TODO add back unknown_is_free option
  for (int i = 0; i < spatial_dim(); ++i) {
    if (indices[i] < 0 || (map_dims_[i] - indices[i]) <= 0) {
      return false;
    }
  }
  return true;
}

bool GraphSearch::is_free_and_valid_indices(
    const Eigen::Vector3i& indices) const {
  return is_valid_indices(indices) &&
         voxel_map_.data[get_linear_indices(indices)] <= 0;
}

bool GraphSearch::is_free_and_valid_position(Eigen::VectorXd position) const {
  if (position.rows() < 3) {
    position.conservativeResize(3);
    position(2) = 0;
  }
  return is_free_and_valid_indices(get_indices_from_position(position));
}

bool GraphSearch::is_mp_collision_free(const MotionPrimitive& mp,
                                       double step_size) const {
  const auto samples = mp.sample_positions(step_size);
  for (int i = 0; i < samples.rows(); ++i) {
    if (!is_free_and_valid_position(samples.row(i))) {
      return false;
    }
  }
  return true;
}

double GraphSearch::heuristic(const Eigen::VectorXd& v) const {
  CHECK_EQ(v.size(), goal_state_.size());
  Eigen::VectorXd x;
  x.resize(spatial_dim());
  for (int i = 0; i < graph_.spatial_dim_; ++i) {
    x(i) = v(i) - goal_state_(i);
  }
  // TODO [theoretical] needs a lot of improvement. Not admissible, but too slow
  // otherwise with higher velocities.
  return 1.2 * graph_.rho_ * x.lpNorm<Eigen::Infinity>() / graph_.max_state_(1);
}

std::vector<Node> GraphSearch::get_neighbor_nodes_lattice(
    const Node& node) const {
  std::vector<Node> neighbor_nodes;
  // TODO explain reset_map_index
  int reset_map_index = std::floor(node.index_ / graph_.num_tiles_);
  for (int i = 0; i < graph_.edges_.rows(); ++i) {
    if (graph_.edges_(i, reset_map_index) >= 0) {
      MotionPrimitive mp = graph_.get_mp_between_indices(i, reset_map_index);
      mp.translate(node.state_);
      if (is_mp_collision_free(mp)) {
        Node neighbor_node(node.cost_to_come_ + mp.cost_,
                           heuristic(mp.end_state_), mp.end_state_, i);
        neighbor_nodes.push_back(neighbor_node);
      }
    }
  }
  return neighbor_nodes;
}

std::vector<MotionPrimitive> GraphSearch::run_graph_search() const {
  Node start_node(0, heuristic(start_state_), start_state_, 0);
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
  std::unordered_map<Eigen::VectorXd, Node, matrix_hash<Eigen::VectorXd>>
      shortest_path_history;
  pq.push(start_node);

  while (!pq.empty() && ros::ok()) {
    Node current_node = pq.top();
    if (position_within(current_node.state_.head(spatial_dim()),
                        goal_state_.head(spatial_dim()), 0.5)) {
      // TODO parameterize termination conditions, add
      // BVP end condition
      LOG(INFO) << "pq: " << pq.size();
      LOG(INFO) << "hist: " << shortest_path_history.size();
      return reconstruct_path(current_node, shortest_path_history);
    }
    pq.pop();
    for (auto& neighbor_node : get_neighbor_nodes_lattice(current_node)) {
      double neighbor_past_g =
          shortest_path_history[neighbor_node.state_].cost_to_come_;
      if (neighbor_node.cost_to_come_ <
          neighbor_past_g +
              get_mp_between_nodes(current_node, neighbor_node).cost_) {
        pq.push(neighbor_node);
        shortest_path_history[neighbor_node.state_] = current_node;
      }
    }
  }
  return {};
}

GraphSearch::GraphSearch(const MotionPrimitiveGraph& graph,
                         const Eigen::VectorXd& start_state,
                         const Eigen::VectorXd& goal_state,
                         const planning_ros_msgs::VoxelMap& voxel_map)
    : graph_(graph),
      start_state_(start_state),
      goal_state_(goal_state),
      voxel_map_(voxel_map) {
  map_dims_[0] = voxel_map_.dim.x;
  map_dims_[1] = voxel_map_.dim.y;
  map_dims_[2] = voxel_map_.dim.z;
  map_origin_[0] = voxel_map_.origin.x;
  map_origin_[1] = voxel_map_.origin.y;
  map_origin_[2] = voxel_map_.origin.z;
}

std::vector<MotionPrimitive> GraphSearch::reconstruct_path(
    const Node& end_node,
    const std::unordered_map<Eigen::VectorXd, Node,
                             matrix_hash<Eigen::VectorXd>>&
        shortest_path_history) const {
  Node node = end_node;
  Node parent_node;
  std::vector<MotionPrimitive> path;

  if (end_node.cost_to_come_ == 0) {
    ROS_WARN("No trajectory found due to start being too close to the goal.");
    return {};
  }

  while (ros::ok()) {
    parent_node = shortest_path_history.at(node.state_);
    path.push_back(get_mp_between_nodes(parent_node, node));
    if (parent_node.cost_to_come_ == 0) {
      break;
    }
    node = parent_node;
  }

  std::reverse(path.begin(), path.end());
  //  ROS_INFO_STREAM(path);
  ROS_INFO("Optimal trajectory cost %f", end_node.cost_to_come_);
  return path;
}

MotionPrimitive GraphSearch::get_mp_between_nodes(const Node& start_node,
                                                  const Node& end_node) const {
  int reset_map_index = floor(start_node.index_ / graph_.num_tiles_);
  MotionPrimitive mp =
      graph_.get_mp_between_indices(end_node.index_, reset_map_index);
  mp.translate(start_node.state_);
  return mp;
}

std::ostream& operator<<(std::ostream& os, const Node& node) {
  os << node.state_.transpose() << "\n";
  return os;
}

}  // namespace motion_primitives
