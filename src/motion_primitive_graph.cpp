#include "motion_primitives/motion_primitive_graph.h"

#include <ros/ros.h>

namespace motion_primitives {

template <int state_dim>
void MotionPrimitive<state_dim>::translate(
    Eigen::Matrix<float, state_dim, 1> new_start) {
  // self.poly_coeffs[:, -1] = start_pt
  end_state_.head(spatial_dims_) = end_state_.head(spatial_dims_) -
                                   start_state_.head(spatial_dims_) +
                                   new_start.head(spatial_dims_);
  start_state_.head(spatial_dims_) = new_start.head(spatial_dims_);
}

template <int state_dim>
std::ostream& operator<<(std::ostream& os,
                         const MotionPrimitive<state_dim>& m) {
  os << "start state: " << m.start_state_.transpose()
     << ", end state:" << m.end_state_.transpose() << std::endl;
  return os;
}

template <int state_dim>
void from_json(const nlohmann::json& json_data,
               MotionPrimitiveGraph<state_dim>& graph) {
  json_data.at("dispersion").get_to(graph.dispersion_);
  json_data.at("tiling").get_to(graph.tiling_);
  json_data.at("num_dims").get_to(graph.spatial_dims_);

  graph.num_tiles_ = graph.tiling_ ? pow(3, graph.spatial_dims_) : 1;
  graph.vertices_.resize(json_data["vertices"].size(), state_dim);
  for (int i = 0; i < json_data["vertices"].size(); i++) {
    Eigen::Matrix<float, state_dim, 1> eigen_vec(
        json_data.at("vertices")[i].get<std::vector<float>>().data());
    graph.vertices_.row(i) = eigen_vec;
  }
  graph.edges_.resize(graph.vertices_.rows() * graph.num_tiles_,
                      graph.vertices_.rows());
  for (int i = 0; i < graph.vertices_.rows() * graph.num_tiles_; i++) {
    for (int j = 0; j < graph.vertices_.rows(); j++) {
      auto edge = json_data.at("edges").at(i * graph.vertices_.size() + j);
      if (edge.size() > 0) {
        std::vector<float> vec;
        edge.at("start_state").get_to(vec);  // for some reason get won't work,
                                             // using more verbose get_to
        Eigen::Matrix<float, state_dim, 1> start_state(vec.data());
        edge.at("end_state").get_to(vec);
        Eigen::Matrix<float, state_dim, 1> end_state(vec.data());
        // std::cout << i << " " << j <<std::endl;
        graph.edges_(i, j) = MotionPrimitive<state_dim>(
            graph.spatial_dims_, start_state, end_state, edge.at("cost"));
      }
    }
  }
}

}  // namespace motion_primitives
