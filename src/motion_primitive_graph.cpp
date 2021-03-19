#include "motion_primitives/motion_primitive_graph.h"

#include <nlohmann/json.hpp>
#include <ostream>

namespace motion_primitives {

void MotionPrimitive::translate(const Eigen::VectorXd& new_start) {
  // self.poly_coeffs[:, -1] = start_pt
  end_state_.head(spatial_dims_) = end_state_.head(spatial_dims_) -
                                   start_state_.head(spatial_dims_) +
                                   new_start.head(spatial_dims_);
  start_state_.head(spatial_dims_) = new_start.head(spatial_dims_);
}

std::ostream& operator<<(std::ostream& os, const MotionPrimitive& m) {
  os << "start state: " << m.start_state_.transpose()
     << ", end state:" << m.end_state_.transpose() << std::endl;
  return os;
}

void from_json(const nlohmann::json& json_data, MotionPrimitiveGraph& graph) {
  json_data.at("dispersion").get_to(graph.dispersion_);
  json_data.at("tiling").get_to(graph.tiling_);
  json_data.at("num_dims").get_to(graph.spatial_dims_);
  json_data.at("control_space_q").get_to(graph.control_space_dim_);

  graph.num_tiles_ = graph.tiling_ ? pow(3, graph.spatial_dims_) : 1;
  int state_dim = graph.spatial_dims_ * graph.control_space_dim_;
  graph.vertices_.resize(json_data["vertices"].size(), state_dim);
  for (int i = 0; i < json_data["vertices"].size(); i++) {
    Eigen::Map<Eigen::VectorXd> eigen_vec(
        json_data.at("vertices")[i].get<std::vector<double>>().data(),
        state_dim);
    graph.vertices_.row(i) = eigen
_vec;
  }
  graph.edges_.resize(graph.vertices_.rows() * graph.num_tiles_,
                      graph.vertices_.rows());
  for (int i = 0; i < graph.vertices_.rows() * graph.num_tiles_; i++) {
    for (int j = 0; j < graph.vertices_.rows(); j++) {
      auto edge = json_data.at("edges").at(i * graph.vertices_.size() + j);
      if (edge.size() > 0) {
        Eigen::Map<Eigen::VectorXd> start_state(
            edge.at("start_state").get<std::vector<double>>().data(),
            state_dim);
        Eigen::Map<Eigen::VectorXd> end_state(
            edge.at("end_state").get<std::vector<double>>().data(), state_dim);
        graph.mps_.push_back(MotionPrimitive(graph.spatial_dims_, start_state,
                                             end_state, edge.at("cost")));
        graph.edges_(i, j) = graph.mps_.size();
      } else {
        graph.edges_(i, j) = -1;
      }
    }
  }
}

}  // namespace motion_primitives
