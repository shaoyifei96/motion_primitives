//#include <absl/container/flat_hash_map.h>
#include <glog/logging.h>
#include <ros/init.h>  // ok()
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <boost/timer/timer.hpp>

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

  // consider marking this [[nodiscard]]
  double total_cost() const noexcept { return motion_cost + heuristic_cost; }
};

std::vector<MotionPrimitive> recover_path(
    const std::unordered_map<int, Node2>& history,
    const std::vector<MotionPrimitive>& expanded_mps, const Node2& end_node) {
  std::vector<MotionPrimitive> path_mps;
  Node2 curr_node = end_node;
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
  ROS_INFO("Optimal trajectory cost %f", end_node.motion_cost);
  return path_mps;
}

std::vector<MotionPrimitive> GraphSearch::expand_mp_par(
    const MotionPrimitive& mp) const {
  int reset_map_index = std::floor(mp.id() / graph_.num_tiles_);

  using PrivVec = tbb::enumerable_thread_specific<std::vector<MotionPrimitive>>;
  PrivVec priv_mps;

  tbb::parallel_for(tbb::blocked_range<int>(0, graph_.edges_.rows()),
                    [&, this](const tbb::blocked_range<int>& r) {
                      auto& local = priv_mps.local();

                      for (int i = r.begin(); i < r.end(); ++i) {
                        // No edge
                        if (graph_.edges_(i, reset_map_index) < 0) {
                          continue;
                        }

                        // Get a copy and move into output later
                        auto new_mp =
                            graph_.get_mp_between_indices(i, reset_map_index);
                        new_mp.translate(mp.end_state());

                        // not collision free
                        if (!is_mp_collision_free(new_mp)) {
                          continue;
                        }
                        new_mp.id_ = i;
                        local.push_back(std::move(new_mp));
                      }
                    });

  // combine
  std::vector<MotionPrimitive> mps;
  mps.reserve(64);
  for (auto i = priv_mps.begin(); i != priv_mps.end(); ++i) {
    const auto& each = *i;
    mps.insert(mps.end(), each.begin(), each.end());
  }
  return mps;
}

std::vector<MotionPrimitive> GraphSearch::expand_mp(
    const MotionPrimitive& mp) const {
  std::vector<MotionPrimitive> mps;
  mps.reserve(64);

  //  boost::timer::cpu_timer timer;

  int reset_map_index = std::floor(mp.id() / graph_.num_tiles_);

  for (int i = 0; i < graph_.edges_.rows(); ++i) {
    // No edge
    if (graph_.edges_(i, reset_map_index) < 0) {
      continue;
    }

    // Get a copy and move into output later
    //    timer.start();
    MotionPrimitive new_mp = graph_.get_mp_between_indices(i, reset_map_index);
    new_mp.translate(mp.end_state());
    //    timings["expand_get_mp"] += timer.elapsed().wall / 1e9;

    // not collision free
    //    timer.start();
    if (!is_mp_collision_free(new_mp)) {
      continue;
    }
    //    timings["expand_collision"] += timer.elapsed().wall / 1e9;

    new_mp.id_ = i;
    mps.push_back(std::move(new_mp));
  }

  return mps;
}

std::vector<MotionPrimitive> GraphSearch::search_path(
    const Eigen::VectorXd& start_state, const Eigen::VectorXd& end_state,
    double distance_threshold) const {
  expanded_mps_.clear();  // clean up expansion

  // Early exit if start and end positions are close
  if (state_pos_within(start_state, end_state, spatial_dim(),
                       distance_threshold)) {
    return {};
  }

  // Start node is reached by a dummy mp, and has 0 cost
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

  MinHeap pq{cost_cmp};
  pq.push(start_node);

  // Add a sentinel mp (MP does not have a constructor)
  MotionPrimitive dummy_mp;
  dummy_mp.id_ = 0;
  dummy_mp.cost_ = 0;
  dummy_mp.spatial_dim_ = spatial_dim();
  dummy_mp.end_state_ = start_state;
  expanded_mps_.push_back(dummy_mp);

  // Shortest path history, stores the parent node of a particular mp (int)
  std::unordered_map<int, Node2> history;

  // timer
  boost::timer::cpu_timer timer_astar, timer;

  while (!pq.empty() && ros::ok()) {
    Node2 curr_node = pq.top();

    // Check if we are close enough to the end
    // Use at for safety, later change to []
    const auto& curr_mp = expanded_mps_.at(curr_node.mp_index);
    if (state_pos_within(curr_mp.end_state(), end_state, spatial_dim(),
                         distance_threshold)) {
      timings["astar"] = timer_astar.elapsed().wall / 1e9;
      LOG(INFO) << "pq: " << pq.size();
      LOG(INFO) << "hist: " << history.size();
      return recover_path(history, expanded_mps_, curr_node);
    }

    pq.pop();
    timer.start();
    const auto neighbor_mps = expand_mp(curr_mp);
    timings["astar_expand"] += timer.elapsed().wall / 1e9;

    for (const auto& next_mp : neighbor_mps) {
      // Add this to the expaned mp set
      expanded_mps_.push_back(next_mp);

      // Create a corresponding node
      Node2 next_node;
      next_node.mp_index = expanded_mps_.size() - 1;
      next_node.motion_cost = curr_node.motion_cost + next_mp.cost_;
      next_node.heuristic_cost = heuristic(next_mp.end_state());
      //      const auto last_cost = history[next_node.mp_index].motion_cost;

      // Check if we found a better path
      //      if (next_node.motion_cost < last_cost) {
      // push to min_heap
      timer.start();
      pq.push(next_node);
      timings["astar_push"] += timer.elapsed().wall / 1e9;
      history[next_node.mp_index] = curr_node;
      //      }
    }
  }

  return {};
}

}  // namespace motion_primitives
