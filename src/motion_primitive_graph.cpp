#include "motion_primitives/motion_primitive_graph.h"

#include <planning_ros_msgs/Polynomial.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <ostream>

using planning_ros_msgs::Polynomial;
using planning_ros_msgs::Spline;
namespace motion_primitives {

void MotionPrimitive::translate(const Eigen::VectorXd& new_start) {
  end_state_.head(spatial_dim_) = end_state_.head(spatial_dim_) -
                                  start_state_.head(spatial_dim_) +
                                  new_start.head(spatial_dim_);
  start_state_.head(spatial_dim_) = new_start.head(spatial_dim_);
  poly_coeffs_.col(poly_coeffs_.cols() - 1) = new_start.head(spatial_dim_);
}

void RuckigMotionPrimitive::translate(const Eigen::VectorXd& new_start) {
  MotionPrimitive::translate(new_start);
  calculate_ruckig_traj();
}

Spline MotionPrimitive::add_to_spline(Spline spline, int dim) {
  if (poly_coeffs_.size() == 0) return spline;
  Polynomial poly;
  poly.degree = poly_coeffs_.cols() - 1;
  poly.basis = 0;
  poly.dt = traj_time_;
  spline.t_total += traj_time_;
  Eigen::VectorXd p = poly_coeffs_.row(dim).reverse();
  std::vector<float> pc(p.data(), p.data() + p.rows() * p.cols());
  poly.coeffs = pc;
  spline.segments += 1;
  spline.segs.push_back(poly);
  return spline;
}

Spline RuckigMotionPrimitive::add_to_spline(Spline spline, int dim) {
  auto jerk_time_array = ruckig_traj_.get_jerks_and_times();
  std::tuple<float, float, float> state;
  std::get<0>(state) = start_state_[dim];
  std::get<1>(state) = start_state_[dim + spatial_dim_];
  std::get<2>(state) = start_state_[dim + 2 * spatial_dim_];
  for (int seg = 0; seg < 7; seg++) {
    Polynomial poly;
    poly.degree = 3;
    poly.basis = 0;
    poly.dt = jerk_time_array[dim * 2][seg];
    if (poly.dt == 0) {
      continue;
    }
    float j = jerk_time_array[dim * 2 + 1][seg];
    auto [p, v, a] = state;
    poly.coeffs = {p, v, a / 2, j / 6};
    state = ruckig::Profile::integrate(poly.dt, p, v, a, j);
    spline.segments += 1;
    spline.segs.push_back(poly);
    spline.t_total += poly.dt;
  }
  return spline;
}

Eigen::MatrixXd MotionPrimitive::sample_positions(double step_size) const {
  int num_samples = std::ceil(traj_time_ / step_size) + 1;
  Eigen::VectorXd times =
      Eigen::VectorXd::LinSpaced(num_samples, 0, traj_time_);

  Eigen::MatrixXd result(spatial_dim_, num_samples);

  for (int i = 0; i < times.size(); ++i) {
    result.col(i) = evaluate_primitive(times(i));
  }

  return result;
}

Eigen::VectorXd MotionPrimitive::evaluate_primitive(float t) const {
  Eigen::VectorXd time_multiplier(poly_coeffs_.cols());
  for (int i = 0; i < poly_coeffs_.cols(); ++i) {
    time_multiplier[poly_coeffs_.cols() - i - 1] = std::pow(t, i);
  }
  return poly_coeffs_ * time_multiplier;
}

RuckigMotionPrimitive::RuckigMotionPrimitive(int spatial_dim,
                                             const Eigen::VectorXd& start_state,
                                             const Eigen::VectorXd& end_state,
                                             const Eigen::VectorXd& max_state)
    : MotionPrimitive(spatial_dim, start_state, end_state, max_state) {
  calculate_ruckig_traj();
}

void RuckigMotionPrimitive::calculate_ruckig_traj() {
  ruckig::Ruckig<3> otg{0.001};
  ruckig::InputParameter<3> input;
  ruckig::OutputParameter<3> output;

  for (int dim = 0; dim < spatial_dim_; dim++) {
    input.max_velocity[dim] = max_state_[1];
    input.max_acceleration[dim] = max_state_[2];
    input.max_jerk[dim] = max_state_[3];
    input.current_position[dim] = start_state_(dim);
    input.current_velocity[dim] = start_state_(spatial_dim_ + dim);
    input.current_acceleration[dim] = start_state_(2 * spatial_dim_ + dim);
    input.target_position[dim] = end_state_(dim);
    input.target_velocity[dim] = end_state_(spatial_dim_ + dim);
    input.target_acceleration[dim] = end_state_(2 * spatial_dim_ + dim);
  }
  if (spatial_dim_ == 2) {
    input.current_position[2] = 0;
    input.current_velocity[2] = 0;
    input.current_acceleration[2] = 0;
    input.target_position[2] = 0;
    input.target_velocity[2] = 0;
    input.target_acceleration[2] = 0;
    input.max_velocity[2] = 1e-2;
    input.max_acceleration[2] = 1e-2;
    input.max_jerk[2] = 1e-2;
  }
  auto result = otg.calculate(input, ruckig_traj_);
  // if (result < 0) {
  //   ROS_ERROR("Ruckig error %d", result);  // TODO should do print/more than
  //   print
  // }
  traj_time_ = ruckig_traj_.duration;
  cost_ = traj_time_;
}

Eigen::VectorXd RuckigMotionPrimitive::evaluate_primitive(float t) const {
  std::array<double, 3> position, velocity, acceleration;
  ruckig_traj_.at_time(t, position, velocity, acceleration);
  Eigen::VectorXd state(3 * spatial_dim_);
  for (int dim = 0; dim < spatial_dim_; dim++) {
    state[dim] = position[dim];
    state[spatial_dim_ + dim] = velocity[dim];
    state[2*spatial_dim_ + dim] = acceleration[dim];
  }
  return state;
}

std::ostream& operator<<(std::ostream& os, const MotionPrimitive& m) {
  os << "start state: " << m.start_state_.transpose() << "\n";
  os << "end state: " << m.end_state_.transpose() << "\n";
  os << "cost: " << m.cost_ << "\n";
  return os;
}

std::ostream& operator<<(std::ostream& os, const MotionPrimitiveGraph& mpg) {
  os << "Vertices:\n" << mpg.vertices_ << "\n";
  return os;
}

template <typename T>
std::shared_ptr<MotionPrimitive> createInstance(
    int spatial_dim, const Eigen::VectorXd& start_state,
    const Eigen::VectorXd& end_state, const Eigen::VectorXd& max_state) {
  return std::make_shared<T>(spatial_dim, start_state, end_state, max_state);
}

void from_json(const nlohmann::json& json_data, MotionPrimitiveGraph& graph) {
  std::map<std::string, std::shared_ptr<MotionPrimitive> (*)(
                            int spatial_dim, const Eigen::VectorXd& start_state,
                            const Eigen::VectorXd& end_state,
                            const Eigen::VectorXd& max_state)>
      mp_type_map;

  mp_type_map["RuckigMotionPrimitive"] = &createInstance<RuckigMotionPrimitive>;
  mp_type_map["PolynomialMotionPrimitive"] =
      &createInstance<PolynomialMotionPrimitive>;
  mp_type_map["OptimizationMotionPrimitive"] =
      &createInstance<OptimizationMotionPrimitive>;

  json_data.at("dispersion").get_to(graph.dispersion_);
  json_data.at("tiling").get_to(graph.tiling_);
  json_data.at("num_dims").get_to(graph.spatial_dim_);
  json_data.at("control_space_q").get_to(graph.control_space_dim_);
  json_data.at("rho").get_to(graph.rho_);
  graph.state_dim_ = graph.spatial_dim_ * graph.control_space_dim_;
  graph.num_tiles_ = graph.tiling_ ? pow(3, graph.spatial_dim_) : 1;
  auto s = json_data.at("max_state").get<std::vector<double>>();
  graph.max_state_ = Eigen::Map<Eigen::VectorXd>(s.data(), s.size());
  graph.vertices_.resize(json_data["vertices"].size(),
                         graph.spatial_dim_ * graph.control_space_dim_);
  // Convert from json to std::vector to Eigen::VectorXd, maybe could be
  // improved by implementing get for Eigen::VectorXd. For some reason doing
  // this in one line gives the wrong values.
  for (int i = 0; i < json_data.at("vertices").size(); i++) {
    auto x = json_data.at("vertices")[i].get<std::vector<double>>();
    Eigen::Map<Eigen::VectorXd> eigen_vec(x.data(), x.size());
    graph.vertices_.row(i) = eigen_vec;
  }
  graph.edges_.resize(graph.vertices_.rows() * graph.num_tiles_,
                      graph.vertices_.rows());
  for (int i = 0; i < graph.vertices_.rows() * graph.num_tiles_; i++) {
    for (int j = 0; j < graph.vertices_.rows(); j++) {
      auto edge = json_data.at("edges").at(i * graph.vertices_.rows() + j);
      if (edge.size() > 0) {
        auto s = edge.at("start_state").get<std::vector<double>>();
        Eigen::Map<Eigen::VectorXd> start_state(s.data(), s.size());
        auto e = edge.at("end_state").get<std::vector<double>>();
        Eigen::Map<Eigen::VectorXd> end_state(e.data(), e.size());

        // TODO check poly_coeffs for ruckig_mp
        Eigen::MatrixXd poly_coeffs;
        if (edge.contains("polys")) {
          poly_coeffs.resize(graph.spatial_dim_, edge.at("polys")[0].size());
          for (int k = 0; k < graph.spatial_dim_; k++) {
            auto x = edge.at("polys")[k].get<std::vector<double>>();
            poly_coeffs.row(k) =
                Eigen::Map<Eigen::VectorXd>(x.data(), x.size());
          }
        }
        auto mp = mp_type_map[json_data.at("mp_type")](
            graph.spatial_dim_, start_state, end_state, graph.max_state_);
        mp->populate(edge.at("cost"), edge.at("traj_time"), poly_coeffs);
        graph.mps_.push_back(mp);
        graph.edges_(i, j) = graph.mps_.size() - 1;
      } else {
        graph.edges_(i, j) = -1;  // TODO make constant
      }
    }
  }
}

MotionPrimitiveGraph read_motion_primitive_graph(const std::string& s) {
  std::ifstream json_file(s);
  nlohmann::json json_data;
  json_file >> json_data;
  return json_data.get<motion_primitives::MotionPrimitiveGraph>();
}

}  // namespace motion_primitives
