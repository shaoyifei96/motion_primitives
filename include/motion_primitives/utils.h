#pragma once

#include <planning_ros_msgs/Trajectory.h>
#include <visualization_msgs/MarkerArray.h>

#include "motion_primitives/motion_primitive_graph.h"

namespace motion_primitives {

planning_ros_msgs::Trajectory path_to_traj_msg(
    const std::vector<MotionPrimitive>& mps, const std_msgs::Header& header);

auto mps_to_marker_array(const std::vector<MotionPrimitive>& mps,
                         const std_msgs::Header& header, double scale = 0.1,
                         bool draw_traj = false)
    -> visualization_msgs::MarkerArray;

}  // namespace motion_primitives
