#ifndef MOTION_PRIMITIVES_GRAPH_SEARCH_H
#define MOTION_PRIMITIVES_GRAPH_SEARCH_H

#include <planning_ros_msgs/Primitive.h>
#include <planning_ros_msgs/Trajectory.h>
#include <planning_ros_msgs/VoxelMap.h>
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
  Eigen::Vector3i map_dims_;
  Eigen::Vector3d map_origin_;
  planning_ros_msgs::VoxelMap voxel_map_;
  Eigen::Vector3i get_indices_from_position(
      const Eigen::Vector3d& position) const;
  int get_linear_indices(const Eigen::Vector3i& indices) const;
  bool is_valid_indices(const Eigen::Vector3i& indices) const;
  bool is_free_and_valid_indices(const Eigen::Vector3i& indices) const;
  bool is_free_and_valid_position(Eigen::VectorXd v) const;
  bool is_mp_collision_free(const MotionPrimitive& mp, double step_size) const;

  std::vector<Node> get_neighbor_nodes_lattice(const Node& node) const;
  double heuristic(const Eigen::VectorXd& v) const;
  std::vector<MotionPrimitive> reconstruct_path(
      const Node& end_node,
      const std::map<Eigen::VectorXd, Node>& shortest_path_history) const;

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