#include "motion_primitives/utils.h"

#include <geometry_msgs/Point.h>
#include <ros/console.h>

namespace motion_primitives {

using geometry_msgs::Point;
using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

planning_ros_msgs::Trajectory path_to_traj_msg(
    const std::vector<MotionPrimitive>& mps, const std_msgs::Header& header) {
  if (mps.empty()) return {};

  int spatial_dim = mps[0].spatial_dim();
  Eigen::ArrayXXd pc_resized(spatial_dim, 6);
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

MarkerArray mps_to_marker_array(const std::vector<MotionPrimitive>& mps,
                                const std_msgs::Header& header, double scale,
                                bool draw_traj) {
  MarkerArray marray;
  marray.markers.reserve(2);

  // end point of each mps, put them in the same marker as a sphere list to
  // speed up rendering
  Marker m_end;
  m_end.id = 0;
  m_end.ns = "node";
  m_end.header = header;
  m_end.color.b = 1.0;
  m_end.color.a = 0.5;
  m_end.type = Marker::SPHERE_LIST;
  m_end.scale.x = m_end.scale.y = m_end.scale.z = scale;
  m_end.pose.orientation.w = 1.0;
  for (const auto& mp : mps) {
    Point p;
    p.x = mp.end_state().x();
    p.y = mp.end_state().y();
    p.z = mp.spatial_dim() == 3 ? mp.end_state().z() : 0;
    m_end.points.push_back(p);
  }
  marray.markers.push_back(std::move(m_end));

  if (draw_traj) {
    Marker m_mps;
    m_mps.header = header;
    m_mps.id = 0;
    m_mps.ns = "edge";
    m_mps.color = m_end.color;
    m_mps.type = Marker::LINE_LIST;
    m_mps.scale.x = scale / 3.0;  // traj should be thinner than end point
    m_mps.pose.orientation.w = 1.0;

    for (const auto& mp : mps) {
      // TODO: we have a dummy mp in the expanded set, maybe need a better way
      // to get rid of it
      if (mp.sampled_positions().size() == 0) {
        continue;
      }

      const auto& points = mp.sampled_positions();
      for (int i = 0; i < points.rows() - 1; ++i) {
        Point p1;
        p1.x = points.row(i).x();
        p1.y = points.row(i).y();
        p1.z = mp.spatial_dim() == 3 ? points.row(i).z() : 0;
        Point p2;
        p2.x = points.row(i + 1).x();
        p2.y = points.row(i + 1).y();
        p1.z = mp.spatial_dim() == 3 ? points.row(i + 1).z() : 0;
        m_mps.points.push_back(p1);
        m_mps.points.push_back(p2);

        // Gradually change alpha from 0 to 1 to indicate start to end of mp
        std_msgs::ColorRGBA c1 = m_mps.color;
        c1.a = (i * 1.0f) / points.rows();
        std_msgs::ColorRGBA c2 = m_mps.color;
        c2.a = (i + 1.0f) / points.rows();
        m_mps.colors.push_back(c1);
        m_mps.colors.push_back(c2);
      }
    }
    marray.markers.push_back(std::move(m_mps));
  }

  return marray;
}

}  // namespace motion_primitives
