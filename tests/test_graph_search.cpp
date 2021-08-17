
// Copyright 2021 Laura Jarin-Lipschitz
#include <gtest/gtest.h>
#include <ros/console.h>

#include "motion_primitives/graph_search.h"
using motion_primitives::GraphSearch;
using motion_primitives::read_motion_primitive_graph;
namespace {

TEST(GraphSearchTest, OptimalPath) {
  const auto mp_graph = read_motion_primitive_graph("simple_test.json");
  planning_ros_msgs::VoxelMap voxel_map;
  voxel_map.resolution = 1.;
  voxel_map.dim.x = 20;
  voxel_map.dim.y = 20;
  voxel_map.data.resize(voxel_map.dim.x * voxel_map.dim.y, 0);
  Eigen::Vector2d start(3, 3);
  Eigen::Vector2d goal(5, 5);
  GraphSearch::Option option = {
      .start_state = start,
      .goal_state = goal,
      .distance_threshold = 0.001,
      .parallel_expand = true,
      .heuristic = "min_time",
      .access_graph = false,
      .fixed_z = 0,
      .using_ros = false
  };

  GraphSearch gs(mp_graph, voxel_map, option);
  const auto path = gs.Search();

  float path_cost = 0;
  for (auto seg : path) {
    path_cost += seg->cost_;
    ROS_INFO_STREAM(seg->start_state_);
    ROS_INFO_STREAM(seg->end_state_);
  }
  EXPECT_EQ(path_cost, 2);
}

}  // namespace