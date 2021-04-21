#include "motion_primitives/motion_primitive_graph.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <ostream>

namespace motion_primitives {

void MotionPrimitive::translate(const Eigen::VectorXd& new_start) {
  poly_coeffs.col(poly_coeffs.cols() - 1) = new_start.head(spatial_dim);
  end_state.head(spatial_dim) = end_state.head(spatial_dim) -
                                start_state.head(spatial_dim) +
                                new_start.head(spatial_dim);
  start_state.head(spatial_dim) = new_start.head(spatial_dim);
}

Eigen::VectorXd MotionPrimitive::evaluate_polynomial(float t) const {
  Eigen::VectorXd time_multiplier;
  time_multiplier.resize(poly_coeffs.cols());
  for (int i = 0; i < poly_coeffs.cols(); ++i) {
    time_multiplier[poly_coeffs.cols() - i - 1] = std::pow(t, i);
  }
  return poly_coeffs * time_multiplier;
}

Eigen::MatrixXd MotionPrimitive::sample_positions(double step_size) const {
  int num_samples = std::ceil(traj_time / step_size) + 1;
  Eigen::VectorXd times = Eigen::VectorXd::LinSpaced(num_samples, 0, traj_time);

  Eigen::MatrixXd result;
  result.resize(num_samples, spatial_dim);

  for (int i = 0; i < times.size(); ++i) {
    result.row(i) = evaluate_polynomial(times(i));
  }

  return result;
}

std::ostream& operator<<(std::ostream& os, const MotionPrimitive& m) {
  os << "start state: " << m.start_state.transpose() << "\n";
  os << "end state: " << m.end_state.transpose() << "\n";
  os << "cost: " << m.cost << "\n";
  return os;
}

std::ostream& operator<<(std::ostream& os, const MotionPrimitiveGraph& mpg) {
  os << "Vertices:\n" << mpg.vertices_ << "\n";
  return os;
}

void from_json(const nlohmann::json& json_data, MotionPrimitiveGraph& graph) {
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

        Eigen::MatrixXd poly_coeffs(graph.spatial_dim_,
                                    edge.at("polys")[0].size());
        for (int k = 0; k < graph.spatial_dim_; k++) {
          auto x = edge.at("polys")[k].get<std::vector<double>>();
          poly_coeffs.row(k) = Eigen::Map<Eigen::VectorXd>(x.data(), x.size());
        }
        graph.mps_.push_back(MotionPrimitive(
            graph.spatial_dim_, start_state, end_state, edge.at("cost"),
            edge.at("traj_time"), poly_coeffs));
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
