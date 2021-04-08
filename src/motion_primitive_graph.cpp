#include "motion_primitives/motion_primitive_graph.h"

#include <nlohmann/json.hpp>
#include <ostream>
namespace motion_primitives {

void MotionPrimitive::translate(const Eigen::VectorXd& new_start) {
  // self.poly_coeffs[:, -1] = start_pt //TODO also translate poly coeffs
  end_state_.head(spatial_dim_) = end_state_.head(spatial_dim_) -
                                  start_state_.head(spatial_dim_) +
                                  new_start.head(spatial_dim_);
  start_state_.head(spatial_dim_) = new_start.head(spatial_dim_);
}

std::ostream& operator<<(std::ostream& os, const MotionPrimitive& m) {
  os << "start state: " << m.start_state_.transpose()
     << ", end state:" << m.end_state_.transpose() << std::endl;
  return os;
}

std::ostream& operator<<(std::ostream& os, const MotionPrimitiveGraph& mpg) {
  os << "Vertices:" << std::endl;
  os << mpg.vertices_ << std::endl;
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
        graph.edges_(i, j) = -1;
      }
    }
  }
}

}  // namespace motion_primitives
