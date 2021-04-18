#include "motion_primitives/utils.h"

#include <geometry_msgs/Point.h>
#include <ros/console.h>

namespace motion_primitives {

using geometry_msgs::Point;
using planning_ros_msgs::Primitive;
using planning_ros_msgs::Trajectory;
using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

Trajectory path_to_traj_msg(const std::vector<MotionPrimitive>& mps,
                            const std_msgs::Header& header) {
  if (mps.empty()) return {};

  int spatial_dim = mps[0].spatial_dim();
  Eigen::ArrayXXd pc_resized(spatial_dim, 6);
  Eigen::ArrayXXd coeff_multiplier(pc_resized.rows(), pc_resized.cols());
  Trajectory trajectory;

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
    if (mp.poly_coeffs().size() == 0) break;

    Primitive primitive;
    pc_resized.block(0, pc_resized.cols() - mp.poly_coeffs().cols(),
                     pc_resized.rows(), mp.poly_coeffs().cols()) =
        mp.poly_coeffs();

    pc_resized *= coeff_multiplier;
    for (int i = 0; i < pc_resized.cols(); i++) {
      primitive.cx.push_back(pc_resized(0, i));
      primitive.cy.push_back(pc_resized(1, i));
      if (spatial_dim > 2) {
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

MarkerArray StatesToMarkerArray(const std::vector<Eigen::VectorXd>& states,
                                int spatial_dim, const std_msgs::Header& header,
                                double scale) {
  MarkerArray marray;
  marray.markers.reserve(2);

  // end point of each mps, put them in the same marker as a sphere list to
  // speed up rendering
  Marker m_state;
  m_state.id = 0;
  m_state.ns = "state";
  m_state.header = header;
  m_state.color.b = 1.0;
  m_state.color.a = 0.5;
  m_state.type = Marker::SPHERE_LIST;
  m_state.scale.x = m_state.scale.y = m_state.scale.z = scale;
  m_state.pose.orientation.w = 1.0;
  for (const auto& state : states) {
    Point p;
    p.x = state.x();
    p.y = state.y();
    p.z = spatial_dim == 3 ? state.z() : 0;
    m_state.points.push_back(p);
  }
  marray.markers.push_back(std::move(m_state));

  return marray;
}

}  // namespace motion_primitives
