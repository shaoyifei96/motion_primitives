#include "motion_primitives/graph_search.h"

#include <glog/logging.h>
#include <ros/init.h>  // ok()
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <boost/timer/timer.hpp>

namespace motion_primitives {

namespace {

double Elapsed(const boost::timer::cpu_timer& timer) noexcept {
  return timer.elapsed().wall / 1e9;
}

bool StatePosWithin(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2,
                    int spatial_dim, double d) noexcept {
  return (p1.head(spatial_dim) - p2.head(spatial_dim)).squaredNorm() < (d * d);
}

}  // namespace

GraphSearch::GraphSearch(const MotionPrimitiveGraph& graph,
                         const planning_ros_msgs::VoxelMap& voxel_map)
    : graph_(graph),
      voxel_map_(voxel_map) {
  map_dims_[0] = voxel_map_.dim.x;
  map_dims_[1] = voxel_map_.dim.y;
  map_dims_[2] = voxel_map_.dim.z;
  map_origin_[0] = voxel_map_.origin.x;
  map_origin_[1] = voxel_map_.origin.y;
  map_origin_[2] = voxel_map_.origin.z;
}

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
  for (int i = 0; i < samples.cols(); ++i) {
    if (!is_free_and_valid_position(samples.col(i))) {
      return false;
    }
  }
  return true;
}

std::size_t VectorXdHash::operator()(const Eigen::VectorXd& vd) const noexcept {
  using std::size_t;

  // allow sufficiently close state to map to the same hash value
  const Eigen::VectorXi v = (vd * 100).cast<int>();

  size_t seed = 0;
  for (size_t i = 0; i < static_cast<size_t>(v.size()); ++i) {
    const auto elem = *(v.data() + i);
    seed ^= std::hash<int>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

auto GraphSearch::Expand(const Node& node, const State& goal_state) const
    -> std::vector<Node> {
  std::vector<Node> nodes;
  nodes.reserve(64);

  const int state_index = graph_.NormIndex(node.state_index);

  for (int i = 0; i < graph_.num_tiled_states(); ++i) {
    if (!graph_.HasEdge(i, state_index)) continue;

    auto mp = graph_.get_mp_between_indices(i, state_index);
    mp.translate(node.state);

    // Check if already visited
    if (visited_states_.find(mp.end_state) != visited_states_.cend()) continue;

    // Then check if its collision free
    if (!is_mp_collision_free(mp)) continue;

    // This is a good next node
    Node next_node;
    next_node.state_index = i;
    next_node.state = mp.end_state;
    next_node.motion_cost = node.motion_cost + mp.cost;
    next_node.heuristic_cost = ComputeHeuristic(mp.end_state, goal_state);
    nodes.push_back(next_node);
  }

  return nodes;
}

auto GraphSearch::ExpandPar(const Node& node, const State& goal_state) const
    -> std::vector<Node> {
  const int state_index = graph_.NormIndex(node.state_index);

  using PrivVec = tbb::enumerable_thread_specific<std::vector<Node>>;
  PrivVec priv_nodes;

  tbb::parallel_for(
      tbb::blocked_range<int>(0, graph_.num_tiled_states()),
      [&, this](const tbb::blocked_range<int>& r) {
        auto& local = priv_nodes.local();

        for (int i = r.begin(); i < r.end(); ++i) {
          if (!graph_.HasEdge(i, state_index)) continue;

          auto mp = graph_.get_mp_between_indices(i, state_index);
          mp.translate(node.state);

          // Check if already visited
          if (visited_states_.find(mp.end_state) != visited_states_.end()) {
            continue;
          }

          // Then check if its collision free
          if (!is_mp_collision_free(mp)) continue;

          // This is a good next node
          Node next_node;
          next_node.state_index = i;
          next_node.state = mp.end_state;
          next_node.motion_cost = node.motion_cost + mp.cost;
          next_node.heuristic_cost = ComputeHeuristic(mp.end_state, goal_state);

          local.push_back(std::move(next_node));
        }
      });

  // combine
  std::vector<Node> nodes;
  nodes.reserve(64);
  //  for (auto i = priv_nodes.begin(); i != priv_nodes.end(); ++i) {
  for (const auto& each : priv_nodes) {
    //    const auto& each = *i;
    nodes.insert(nodes.end(), each.begin(), each.end());
  }
  return nodes;
}

MotionPrimitive GraphSearch::GetPrimitiveBetween(const Node& start_node,
                                                 const Node& end_node) const {
  const int start_index = graph_.NormIndex(start_node.state_index);
  auto mp = graph_.get_mp_between_indices(end_node.state_index, start_index);
  mp.translate(start_node.state);
  return mp;
}

std::vector<MotionPrimitive> GraphSearch::RecoverPath(
    const PathHistory& history, const Node& end_node) const {
  std::vector<MotionPrimitive> path_mps;
  Node const* curr_node = &end_node;

  while (ros::ok()) {
    if (curr_node->motion_cost == 0) break;
    Node const* prev_node = &(history.at(curr_node->state).parent_node);
    path_mps.push_back(GetPrimitiveBetween(*prev_node, *curr_node));
    curr_node = prev_node;
  }

  std::reverse(path_mps.begin(), path_mps.end());
  return path_mps;
}

double GraphSearch::ComputeHeuristic(const State& v,
                                     const State& goal_state) const noexcept {
  const Eigen::VectorXd x = (v - goal_state).head(spatial_dim());
  // TODO [theoretical] needs a lot of improvement. Not admissible, but too
  // slow otherwise with higher velocities.
  return 1.1 * graph_.rho() * x.lpNorm<Eigen::Infinity>() /
         graph_.max_state()(1);
}

auto GraphSearch::Search(const Option& option) -> std::vector<MotionPrimitive> {
  // Debug
  //  LOG(INFO) << "adj mat: " << graph_.edges_.rows() << " "
  //            << graph_.edges_.cols() << ", nnz: " << (graph_.edges_ >
  //            0).count();
  //  LOG(INFO) << "mps: " << graph_.mps_.size();
  //  LOG(INFO) << "verts: " << graph_.vertices_.rows() << " "
  //            << graph_.vertices_.cols();

  timings_.clear();
  visited_states_.clear();

  // Early exit if start and end positions are close
  if (StatePosWithin(option.start_state, option.goal_state,
                     graph_.spatial_dim(), option.distance_threshold)) {
    return {};
  }

  Node start_node;
  start_node.state_index = 0;
  start_node.state = option.start_state;
  start_node.motion_cost = 0.0;
  start_node.heuristic_cost =
      ComputeHeuristic(start_node.state, option.goal_state);

  // > for min heap
  auto node_cmp = [](const Node& n1, const Node& n2) {
    return n1.total_cost() > n2.total_cost();
  };
  using MinHeap =
      std::priority_queue<Node, std::vector<Node>, decltype(node_cmp)>;

  MinHeap pq{node_cmp};
  pq.push(start_node);

  // Shortest path history, stores the parent node of a particular mp (int)
  PathHistory history;

  // timer
  boost::timer::cpu_timer timer;

  while (!pq.empty() && ros::ok()) {
    Node curr_node = pq.top();

    // Check if we are close enough to the end
    if (StatePosWithin(curr_node.state, option.goal_state, graph_.spatial_dim(),
                       option.distance_threshold)) {
      LOG(INFO) << "== pq: " << pq.size();
      LOG(INFO) << "== hist: " << history.size();
      LOG(INFO) << "== nodes: " << visited_states_.size();
      return RecoverPath(history, curr_node);
    }

    timer.start();
    pq.pop();
    timings_["astar_pop"] += Elapsed(timer);

    // Due to the imutability of std::priority_queue, we have no way of
    // modifying the priority of an element in the queue. Therefore, when we
    // push the next node into the queue, there might be duplicated nodes with
    // the same state but different costs. This could cause us to expand the
    // same state multiple times.
    // Although this does not affect the correctness of the implementation
    // (since the nodes are correctly sorted), it might be slower to repeatedly
    // expanding visited states. The timiing suggest more than 80% of the time
    // is spent on the Expand(node) call. Thus, we will check here if this state
    // has been visited and skip if it has. This will save around 20%
    // computation.
    if (visited_states_.find(curr_node.state) != visited_states_.cend()) {
      continue;
    }
    // add current state to visited
    visited_states_.insert(curr_node.state);

    timer.start();
    const auto next_nodes = option.parallel_expand
                                ? ExpandPar(curr_node, option.goal_state)
                                : Expand(curr_node, option.goal_state);
    timings_["astar_expand"] += Elapsed(timer);
    for (const auto& next_node : next_nodes) {
      // this is the best cost reaching this state (next_node) so far
      // could be inf if this state has never been visited
      const auto best_cost = history[next_node.state].best_cost;

      // compare reaching next_node from curr_node and mp to best cost
      if (next_node.motion_cost < best_cost) {
        timer.start();
        pq.push(next_node);
        timings_["astar_push"] += Elapsed(timer);
        history[next_node.state] = {curr_node, next_node.motion_cost};
      }
    }
  }

  return {};
}

std::vector<Eigen::VectorXd> GraphSearch::GetVisitedStates() const noexcept {
  return {visited_states_.cbegin(), visited_states_.cend()};
}

}  // namespace motion_primitives
