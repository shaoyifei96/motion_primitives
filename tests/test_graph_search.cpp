
#include <gtest/gtest.h>

#include "motion_primitives/graph_search.h"

using namespace motion_primitives;
namespace {

TEST(GraphSearchTest, OptimalPath) {
  const auto mp_graph = read_motion_primitive_graph("simple_test.json");
  planning_ros_msgs::VoxelMap voxel_map;
  voxel_map.resolution = 1.;
  voxel_map.dim.x = 20;
  voxel_map.dim.y = 20;
  voxel_map.data.resize(voxel_map.dim.x * voxel_map.dim.y, 0);
  GraphSearch gs(mp_graph, voxel_map);
  Eigen::Vector2d start(3, 3);
  Eigen::Vector2d goal(5, 5);
  const auto path = gs.Search({.start_state = start,
                               .goal_state = goal,
                               .distance_threshold = 0.001,
                               .parallel_expand = true,
                               .using_ros = false});

  float path_cost;
  for (auto seg : path) {
    path_cost += seg.cost;
  }
  EXPECT_EQ(path_cost, 2);
}

}  // namespace