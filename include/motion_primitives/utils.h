// Copyright 2021 Laura Jarin-Lipschitz
#ifndef MOTION_PRIMITIVES_UTILS_H_
#define MOTION_PRIMITIVES_UTILS_H_

#include <kr_planning_msgs/SplineTrajectory.h>
#include <kr_planning_msgs/Trajectory.h>
#include <ros/console.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <visualization_msgs/MarkerArray.h>

#include <boost/math/tools/polynomial.hpp>
#include <memory>
#include <string>
#include <vector>

#include "motion_primitives/motion_primitive_graph.h"

namespace motion_primitives {

typedef boost::math::tools::polynomial<double> Poly;

Poly differentiate(const Poly& p);
Eigen::MatrixXd differentiate(const Eigen::MatrixXd& coeffs);
Eigen::VectorXd evaluate_poly_coeffs(const Eigen::MatrixXd& poly_coeffs, float t);

std::shared_ptr<MotionPrimitive> recover_mp_from_SplineTrajectory(
    const kr_planning_msgs::SplineTrajectory& traj,
    std::shared_ptr<MotionPrimitiveGraph> graph, int seg_num);


Eigen::Vector3d getState(const std::vector<std::shared_ptr<MotionPrimitive>> &traj,
                         double time, int deriv_num, double fixed_z = 0.0);

// Convert from internal representation to ROS Trajectory message
kr_planning_msgs::Trajectory path_to_traj_msg(
    const std::vector<std::shared_ptr<MotionPrimitive>>& mps,
    const std_msgs::Header& header, float z_height = 0.0);

// Convert from internal representation to ROS SplineTrajectory message
kr_planning_msgs::SplineTrajectory path_to_spline_traj_msg(
    const std::vector<std::shared_ptr<MotionPrimitive>>& mps,
    const std_msgs::Header& header, float z_height = 0.0);

// Convert from a list of states to ROS MarkerArray message (used to visualize
// visited states)
auto StatesToMarkerArray(const std::vector<Eigen::VectorXd>& states,
                         int spatial_dim, const std_msgs::Header& header,
                         double scale = 0.1, bool show_vel = false, double fixed_z = 0.0)
    -> visualization_msgs::MarkerArray;

// num indicates the max number of elements to read, -1 means read till the end
template <class T>
std::vector<T> read_bag(std::string file_name, std::string topic,
                        unsigned int num) {
  rosbag::Bag bag;
  bag.open(file_name, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back(topic);
  rosbag::View view(bag, rosbag::TopicQuery(topics));

  std::vector<T> msgs;
  BOOST_FOREACH (rosbag::MessageInstance const m, view) {
    if (m.instantiate<T>() != NULL) {
      msgs.push_back(*m.instantiate<T>());
      if (msgs.size() > num) break;
    }
  }
  bag.close();
  if (msgs.empty())
    ROS_WARN("Fail to find '%s' in '%s', make sure md5sum are equivalent.",
             topic.c_str(), file_name.c_str());
  else
    ROS_INFO("Get voxel map data!");
  return msgs;
}

}  // namespace motion_primitives
#endif  // MOTION_PRIMITIVES_UTILS_H_