// Copyright 2021 Laura Jarin-Lipschitz
#include "motion_primitives/utils.h"

#include <geometry_msgs/Point.h>

namespace motion_primitives {

using geometry_msgs::Point;
using kr_planning_msgs::Polynomial;
using kr_planning_msgs::Primitive;
using kr_planning_msgs::Spline;
using kr_planning_msgs::SplineTrajectory;
using kr_planning_msgs::Trajectory;
using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

Trajectory path_to_traj_msg(
    const std::vector<std::shared_ptr<MotionPrimitive>>& mps,
    const std_msgs::Header& header, float z_height) {
  if (mps.empty()) return {};

  int spatial_dim = mps[0]->spatial_dim_;
  Eigen::ArrayXXd pc_resized(spatial_dim, 6);
  Eigen::ArrayXXd coeff_multiplier(pc_resized.rows(), pc_resized.cols());
  Trajectory trajectory;

  trajectory.header = header;
  trajectory.primitives.reserve(mps.size());
  //  trajectory.header.stamp = ros::Time::now();
  //  trajectory.header.frame_id = voxel_map_.header.frame_id;

  for (int i = 0; i < pc_resized.rows(); ++i) {
    // These hardcoded coefficients come from how
    // kr_planning_msgs::Primitive/MPL defines polynomial trajectories
    coeff_multiplier.row(i) << 120, 24, 6, 2, 1, 1;
  }

  for (const auto& mp : mps) {
    if (mp->poly_coeffs_.size() == 0) break;

    Primitive primitive;
    pc_resized.block(0, pc_resized.cols() - mp->poly_coeffs_.cols(),
                     pc_resized.rows(), mp->poly_coeffs_.cols()) =
        mp->poly_coeffs_;

    pc_resized *= coeff_multiplier;
    for (int i = 0; i < pc_resized.cols(); i++) {
      primitive.cx.push_back(pc_resized(0, i));
      primitive.cy.push_back(pc_resized(1, i));
      if (spatial_dim > 2) {
        primitive.cz.push_back(pc_resized(2, i));
      } else {
        primitive.cz.push_back(0.);
        if (i == pc_resized.cols() - 1) primitive.cz.back() = z_height;
      }
      primitive.cyaw.push_back(0.);
    }
    primitive.t = mp->traj_time_;
    trajectory.primitives.push_back(primitive);
  }
  return trajectory;
}

SplineTrajectory path_to_spline_traj_msg(
    const std::vector<std::shared_ptr<MotionPrimitive>>& mps,
    const std_msgs::Header& header, float z_height) {
  if (mps.empty()) return {};

  SplineTrajectory spline_traj;
  spline_traj.header = header;
  spline_traj.dimensions = 3;

  for (int dim = 0; dim < 3; dim++) {
    Spline spline;
    if (mps[0]->spatial_dim_ != dim) {
      for (const auto& mp : mps) {
        spline = mp->add_to_spline(spline, dim);
      }
    } else {
      spline.segments = spline_traj.data[0].segments;
      spline.t_total = spline_traj.data[0].t_total;
      for (int i = 0; i < spline.segments; i++) {
        Polynomial poly;
        poly.degree = spline_traj.data[0].segs[0].degree;
        for (int j = 0; j < poly.degree + 1; j++) {
          poly.coeffs.push_back(0.);
        }
        poly.coeffs[0] = {z_height};
        poly.dt = spline_traj.data[0].segs[i].dt;
        spline.segs.push_back(poly);
      }
    }
    spline_traj.data.push_back(spline);
  }
  return spline_traj;
}

MarkerArray StatesToMarkerArray(const std::vector<Eigen::VectorXd>& states,
                                int spatial_dim, const std_msgs::Header& header,
                                double scale, bool show_vel, double fixed_z) {
  MarkerArray marray;
  marray.markers.reserve(2);

  // end point of each mps, put them in the same marker as a sphere list to
  // speed up rendering
  Marker m_pos;
  m_pos.id = 0;
  m_pos.ns = "pos";
  m_pos.header = header;
  m_pos.color.b = 1.0;
  m_pos.color.a = 0.5;
  m_pos.type = Marker::SPHERE_LIST;
  m_pos.scale.x = m_pos.scale.y = m_pos.scale.z = scale;
  m_pos.pose.orientation.w = 1.0;
  for (const auto& state : states) {
    Point p;
    p.x = state.x();
    p.y = state.y();
    p.z = spatial_dim == 3 ? state.z() : fixed_z;
    m_pos.points.push_back(p);
  }
  marray.markers.push_back(std::move(m_pos));

  if (show_vel) {
    Marker m_vel;
    m_vel.id = 0;
    m_vel.ns = "vel";
    m_vel.header = header;
    m_vel.color.b = 1.0;
    m_vel.color.a = 0.5;
    m_vel.type = Marker::LINE_LIST;
    m_vel.scale.x = m_pos.scale.y = m_pos.scale.z = scale / 4.0;
    m_vel.pose.orientation.w = 1.0;
    for (const auto& state : states) {
      auto pos = state.head(spatial_dim);
      Point p1;
      p1.x = pos[0];
      p1.y = pos[1];
      p1.z = spatial_dim == 3 ? state[2] : 0;
      Point p2 = p1;
      auto vel = state.tail(spatial_dim) / 4.0;
      p2.x += vel[0];
      p2.y += vel[1];
      p2.z += spatial_dim == 3 ? vel[2] : 0;
      m_vel.points.push_back(p1);
      m_vel.points.push_back(p2);
    }
    marray.markers.push_back(std::move(m_vel));
  }

  return marray;
}

Poly differentiate(const Poly& p) {
  Poly::size_type rows = p.size();
  if (rows <= 1) return Poly(0.0);
  std::vector<double> v;
  for (Poly::size_type i = 1; i < rows; i++) {
    double val = static_cast<double>(i);
    v.push_back(p[i] * val);
  }
  Poly result(v.data(), rows - 2);
  return result;
}

Eigen::MatrixXd differentiate(const Eigen::MatrixXd& coeffs) {
  if (coeffs.cols() <= 1) return Eigen::MatrixXd::Zero(coeffs.rows(), 0);
  Eigen::MatrixXd result(coeffs.rows(), coeffs.cols() - 1);
  for (int dim = 0; dim < coeffs.rows(); dim++) {
    for (int i = 0; i < coeffs.cols() - 1; i++) {
      result(dim, i) =
          (coeffs(dim, i) * static_cast<double>(coeffs.cols() - i - 1));
    }
  }
  return result;
}

std::shared_ptr<MotionPrimitive> recover_mp_from_SplineTrajectory(
    const kr_planning_msgs::SplineTrajectory& traj,
    std::shared_ptr<MotionPrimitiveGraph> graph, int seg_num) {
  Eigen::VectorXd start(graph->state_dim());
  Eigen::VectorXd end = start;
  for (int i = 0; i < graph->spatial_dim(); i++) {
    Poly const poly(traj.data[i].segs[seg_num].coeffs.begin(),
                    traj.data[i].segs[seg_num].coeffs.end());
    end[i] = poly.evaluate(1);
    auto first_deriv = differentiate(poly);
    end[i + graph->spatial_dim()] =
        first_deriv.evaluate(1) * 1. / (traj.data[i].segs[seg_num].dt);
    start[i] = traj.data[i].segs[seg_num].coeffs[0];
    start[i + graph->spatial_dim()] = traj.data[i].segs[seg_num].coeffs[1] *
                                      1. / (traj.data[i].segs[seg_num].dt);
    if (graph->control_space_dim() > 2) {
      start[i + 2 * graph->spatial_dim()] =
          traj.data[i].segs[seg_num].coeffs[2] *
          std::pow(1. / (traj.data[i].segs[seg_num].dt), 2);
      end[i + 2 * graph->spatial_dim()] =
          differentiate(first_deriv).evaluate(1) *
          std::pow(1. / (traj.data[i].segs[seg_num].dt), 2);
    }
  }
  auto mp = graph->createMotionPrimitivePtrFromGraph(start, end);
  // mp->compute(graph_->rho());  // could copy poly_coeffs or do this

  // copy poly_coeffs
  int degree = traj.data[0].segs[seg_num].coeffs.size();
  mp->poly_coeffs_.resize(graph->spatial_dim(), degree);
  for (int i = 0; i < graph->spatial_dim(); i++) {
    for (int j = 0; j < degree; j++) {
      mp->poly_coeffs_(i, degree - j - 1) =
          traj.data[i].segs[seg_num].coeffs[j] /
          (std::pow(traj.data[0].segs[seg_num].dt, j));
    }
  }
  mp->start_index_ = traj.data[0].segs[seg_num].start_index;
  mp->end_index_ = traj.data[0].segs[seg_num].end_index;
  mp->traj_time_ = traj.data[0].segs[seg_num].dt;
  return mp;
}

// TODO(Laura) de-duplicate from motion_primitive_graph.h
Eigen::VectorXd evaluate_poly_coeffs(const Eigen::MatrixXd& poly_coeffs,
                                     float t) {
  Eigen::VectorXd time_multiplier(poly_coeffs.cols());
  // TODO(laura) could replace with boost::polynomial
  for (int i = 0; i < poly_coeffs.cols(); ++i) {
    time_multiplier[poly_coeffs.cols() - i - 1] = std::pow(t, i);
  }
  return poly_coeffs * time_multiplier;
}

Eigen::Vector3d getState(
    const std::vector<std::shared_ptr<MotionPrimitive>>& traj, double time,
    int deriv_num, double fixed_z) {
  for (auto mp : traj) {
    if (time >= mp->traj_time_) {
      time -= mp->traj_time_;
    } else {
      auto coeffs = mp->poly_coeffs_;
      for (int i = 0; i < deriv_num; i++) {
        coeffs = differentiate(coeffs);
      }
      auto result = evaluate_poly_coeffs(coeffs, time);
      result.conservativeResize(3);
      if (mp->spatial_dim_ < 3) {
        result(2) = fixed_z;
      }
      return result;
    }
  }
  return Eigen::Vector3d::Zero();
}

}  // namespace motion_primitives
