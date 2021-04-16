#include "motion_primitives/utils.h"

namespace motion_primitives {

planning_ros_msgs::Trajectory path_to_traj_msg(
    const std::vector<MotionPrimitive>& mps, int spatial_dim_,
    const std_msgs::Header& header) {
  Eigen::ArrayXXd pc_resized(spatial_dim_, 6);
  Eigen::ArrayXXd coeff_multiplier(pc_resized.rows(), pc_resized.cols());
  planning_ros_msgs::Trajectory trajectory;

  trajectory.header = header;
  trajectory.primitives.reserve(mps.size());
  //  trajectory.header.stamp = ros::Time::now();
  //  trajectory.header.frame_id = voxel_map_.header.frame_id;

  for (int i = 0; i < pc_resized.rows(); ++i) {
    // These hardcoded coefficients come from how
    // planning_ros_msgs::Primitive/MPL defines polynomial trajectories
    coeff_multiplier.row(i) << 120, 24, 6, 2, 1, 1;
  }

  for (const auto& mp : mps) {
    planning_ros_msgs::Primitive primitive;
    pc_resized.block(0, pc_resized.cols() - mp.poly_coeffs().cols(),
                     pc_resized.rows(), mp.poly_coeffs().cols()) =
        mp.poly_coeffs();

    pc_resized *= coeff_multiplier;
    for (int i = 0; i < pc_resized.cols(); i++) {
      primitive.cx.push_back(pc_resized(0, i));
      primitive.cy.push_back(pc_resized(1, i));
      if (spatial_dim_ > 2) {
        primitive.cz.push_back(pc_resized(2, i));
      } else {
        primitive.cz.push_back(0.);
      }
      primitive.cyaw.push_back(0.);
    }
    primitive.t = mp.traj_time();
    trajectory.primitives.push_back(primitive);
  }
  return trajectory;
}

}  // namespace motion_primitives
