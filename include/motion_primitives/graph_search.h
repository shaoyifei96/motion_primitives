#ifndef MOTION_PRIMITIVES_GRAPH_SEARCH_H
#define MOTION_PRIMITIVES_GRAPH_SEARCH_H

#include <planning_ros_msgs/Trajectory.h>
#include <planning_ros_msgs/VoxelMap.h>

#include <queue>
#include <unordered_map>

#include "motion_primitives/motion_primitive_graph.h"

// A hash function for Eigen matrix/vector from the internet: 
// https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
template <typename T>
struct matrix_hash : std::unary_function<T, size_t> {
  std::size_t operator()(T const& matrix) const {
    // Note that it is oblivious to the storage order of Eigen matrix (column-
    // or row-major). It will give you the same hash value for two different
    // matrices if they are the transpose of each other in different storage
    // order.
    size_t seed = 0;
    for (size_t i = 0; i < matrix.size(); ++i) {
      auto elem = *(matrix.data() + i);
      seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
    }
    return seed;
  }
};

namespace motion_primitives {

class Node {
 private:
  double total_cost_{std::numeric_limits<double>::max()};
  double cost_to_come_{std::numeric_limits<double>::max()};
  double heuristic_cost_{0};
  Eigen::VectorXd state_;
  int index_{-1};
  friend class GraphSearch;

 public:
  Node() = default;
  Node(double g, double h, const Eigen::VectorXd& state, int index)
      : cost_to_come_(g),
        heuristic_cost_(h),
        total_cost_(g + h),
        state_(state),
        index_(index) {}
  friend bool operator>(const Node& n1, const Node& n2);
  friend std::ostream& operator<<(std::ostream& out, const Node& node);
};

bool operator>(const Node& n1, const Node& n2) {
  return n1.total_cost_ > n2.total_cost_;
}

class GraphSearch {
 private:
  MotionPrimitiveGraph graph_;
  Eigen::VectorXd start_state_;
  Eigen::VectorXd goal_state_;
  Eigen::Vector3i map_dims_;
  Eigen::Vector3d map_origin_;
  planning_ros_msgs::VoxelMap voxel_map_;
  Eigen::Vector3i get_indices_from_position(
      const Eigen::Vector3d& position) const;
  // Converts from vector of indices to single index into
  // planning_ros_msgs::VoxelMap.data
  int get_linear_indices(const Eigen::Vector3i& indices) const;
  bool is_valid_indices(const Eigen::Vector3i& indices) const;
  bool is_free_and_valid_indices(const Eigen::Vector3i& indices) const;
  bool is_free_and_valid_position(Eigen::VectorXd v) const;
  // Samples motion primitive along step_size time steps and checks for
  // collisions
  bool is_mp_collision_free(const MotionPrimitive& mp, double step_size) const;
  // Returns a vector of collision free nodes that are connected to the current
  // node by graph_ edges
  std::vector<Node> get_neighbor_nodes_lattice(const Node& node) const;
  double heuristic(const Eigen::VectorXd& v) const;
  std::vector<MotionPrimitive> reconstruct_path(
      const Node& end_node,
      const std::unordered_map<Eigen::VectorXd, Node,
                               matrix_hash<Eigen::VectorXd>>&
          shortest_path_history) const;
  MotionPrimitive get_mp_between_indices(int i1, int i2) const;
  MotionPrimitive get_mp_between_nodes(const Node& n1, const Node& n2) const;

 public:
  planning_ros_msgs::Trajectory path_to_traj_msg(
      const std::vector<motion_primitives::MotionPrimitive>& mp_vec) const;
  std::vector<MotionPrimitive> run_graph_search() const;
  GraphSearch(const MotionPrimitiveGraph& graph,
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
};

}  // namespace motion_primitives

#endif