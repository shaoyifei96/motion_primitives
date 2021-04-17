//#include <absl/container/flat_hash_map.h>
#include <glog/logging.h>
#include <ros/init.h>  // ok()

#include <boost/container/flat_map.hpp>

#include "motion_primitives/graph_search.h"

namespace motion_primitives {

namespace {

bool state_pos_within(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2,
                      int spatial_dim, double d) {
  return (p1.head(spatial_dim) - p2.head(spatial_dim)).squaredNorm() < (d * d);
}

}  // namespace

struct Node2 {
  int mp_index{0};  // use this to retreive the mp from expanded set
  double motion_cost{std::numeric_limits<double>::infinity()};
  double heuristic_cost{0.0};

  double total_cost() const noexcept { return motion_cost + heuristic_cost; }
};

std::vector<MotionPrimitive> recover_path(
    const boost::container::flat_map<int, Node2>& history,
    const std::vector<MotionPrimitive>& expanded_mps, const Node2& node) {
  std::vector<MotionPrimitive> path_mps;
  Node2 curr_node = node;
  while (ros::ok()) {
    // Stop if we reach the first dummy mp
    if (curr_node.mp_index == 0) {
      break;
    }
    // Get the parent node given the current mp
    path_mps.push_back(expanded_mps.at(curr_node.mp_index));
    curr_node = history.at(curr_node.mp_index);
  }

  std::reverse(path_mps.begin(), path_mps.end());
  ROS_INFO("Optimal trajectory cost %f", node.motion_cost);
  return path_mps;
}

std::vector<MotionPrimitive> GraphSearch::expand_mp(
    const MotionPrimitive& mp) const {
  std::vector<MotionPrimitive> mps;
  mps.reserve(64);

  int reset_map_index = std::floor(mp.id() / graph_.num_tiles_);
  for (int i = 0; i < graph_.edges_.rows(); ++i) {
    // No edge
    if (graph_.edges_(i, reset_map_index) < 0) {
      continue;
    }

    // Get a copy and move into output later
    MotionPrimitive new_mp = graph_.get_mp_between_indices(i, reset_map_index);
    new_mp.translate(mp.end_state());

    // not collision free
    if (!is_mp_collision_free(new_mp)) {
      continue;
    }

    new_mp.id_ = i;
    mps.push_back(std::move(new_mp));
  }

  return mps;
}

std::vector<MotionPrimitive> GraphSearch::search_path(
    const Eigen::VectorXd& start_state, const Eigen::VectorXd& end_state,
    double distance_threshold) const {
  // Early exit if start and end positions are close
  if (state_pos_within(start_state, end_state, spatial_dim(),
                       distance_threshold)) {
    ROS_INFO("start and end too close");
    return {};
  }

  // Start node is reached by a sentinel mp, and has 0 cost
  Node2 start_node;
  start_node.mp_index = 0;
  start_node.motion_cost = 0.0;
  start_node.heuristic_cost = heuristic(start_state);

  // > for min heap
  auto cost_cmp = [](const Node2& n1, const Node2& n2) {
    return n1.total_cost() > n2.total_cost();
  };
  using MinHeap =
      std::priority_queue<Node2, std::vector<Node2>, decltype(cost_cmp)>;

  std::vector<Node2> cont;
  cont.reserve(1024);
  MinHeap pq{cost_cmp, std::move(cont)};
  pq.push(start_node);

  // All expaned primitives
  std::vector<MotionPrimitive> expanded_mps;
  expanded_mps.reserve(1024);

  // Add a sentinel mp
  MotionPrimitive dummy_mp;
  dummy_mp.id_ = 0;
  dummy_mp.cost_ = 0;
  dummy_mp.end_state_ = start_state;
  expanded_mps.push_back(dummy_mp);

  // Shortest path history, stores the parent node of a particular mp (int)
  boost::container::flat_map<int, Node2> history;

  while (!pq.empty() && ros::ok()) {
    Node2 curr_node = pq.top();

    // Check if we are close enough to the end
    // Use at for safety, later change to []
    const auto& curr_mp = expanded_mps.at(curr_node.mp_index);
    if (state_pos_within(curr_mp.end_state(), end_state, spatial_dim(),
                         distance_threshold)) {
      ROS_INFO("Close to goal, stop");
      return recover_path(history, expanded_mps, curr_node);
    }

    pq.pop();
    const auto neighbor_mps = expand_mp(curr_mp);

    for (const auto& next_mp : neighbor_mps) {
      // Add this to the expaned mp set
      expanded_mps.push_back(next_mp);

      // Create a corresponding node
      Node2 next_node;
      next_node.mp_index = expanded_mps.size() - 1;
      next_node.motion_cost = curr_node.motion_cost + next_mp.cost_;
      next_node.heuristic_cost = heuristic(next_mp.end_state());
      const auto last_cost = history[next_node.mp_index].motion_cost;

      // Check if we found a better path
      if (next_node.motion_cost < last_cost) {
        // push to min_heap
        pq.push(next_node);
        history[next_node.mp_index] = curr_node;
      }
    }
  }

  return {};
}

}  // namespace motion_primitives
