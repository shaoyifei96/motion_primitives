#pragma once

#include <planning_ros_msgs/SplineTrajectory.h>
#include <planning_ros_msgs/Trajectory.h>
#include <visualization_msgs/MarkerArray.h>

#include "motion_primitives/motion_primitive_graph.h"

namespace motion_primitives {

planning_ros_msgs::Trajectory path_to_traj_msg(
    const std::vector<MotionPrimitive*>& mps, const std_msgs::Header& header);

planning_ros_msgs::SplineTrajectory path_to_spline_traj_msg(
    const std::vector<MotionPrimitive*>& mps, const std_msgs::Header& header,
    float z_height = 0.0);

planning_ros_msgs::SplineTrajectory path_to_spline_traj_msg(
    const std::vector<RuckigMotionPrimitive*>& mps, const std_msgs::Header& header,
    float z_height = 0.0);

auto StatesToMarkerArray(const std::vector<Eigen::VectorXd>& states,
                         int spatial_dim, const std_msgs::Header& header,
                         double scale = 0.1, bool show_vel = false)
    -> visualization_msgs::MarkerArray;

}  // namespace motion_primitives
