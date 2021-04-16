#pragma once

#include <planning_ros_msgs/Trajectory.h>

#include "motion_primitives/motion_primitive_graph.h"

namespace motion_primitives {

planning_ros_msgs::Trajectory path_to_traj_msg(
    const std::vector<MotionPrimitive>& mps, int spatial_dim,
    const std_msgs::Header& header);

}
