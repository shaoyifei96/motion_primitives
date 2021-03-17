

#ifndef MOTION_PRIMITIVES_MOTION_PRIMITIVE_GRAPH_H
#define MOTION_PRIMITIVES_MOTION_PRIMITIVE_GRAPH_H

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace motion_primitives {

template <int state_dim>
class MotionPrimitive {
 private:
 public:
  int spatial_dims_;
  Eigen::Matrix<float, state_dim, 1> start_state_;
  Eigen::Matrix<float, state_dim, 1> end_state_;
  float cost_;
  bool initialized_;
  template <int sd>
  friend std::ostream& operator<<(std::ostream& os,
                                  const MotionPrimitive<sd>& m);

  MotionPrimitive(){};
  MotionPrimitive(int spatial_dims,
                  Eigen::Matrix<float, state_dim, 1> start_state,
                  Eigen::Matrix<float, state_dim, 1> end_state, float cost)
      : spatial_dims_(spatial_dims),
        start_state_(start_state),
        end_state_(end_state),
        cost_(cost),
        initialized_(true){};
  void translate(Eigen::Matrix<float, state_dim, 1> new_start);
};

template <int state_dim>
class MotionPrimitiveGraph {
 private:
 public:
  Eigen::Array<MotionPrimitive<state_dim>, Eigen::Dynamic, Eigen::Dynamic>
      edges_;
  Eigen::MatrixXf vertices_;
  float dispersion_;
  int spatial_dims_;
  bool tiling_;
  int num_tiles_;
  MotionPrimitiveGraph(){};
};

template <int state_dim>
void from_json(const nlohmann::json& json_data,
               MotionPrimitiveGraph<state_dim>& graph);

}  // namespace motion_primitives
#endif