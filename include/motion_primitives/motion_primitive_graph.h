#ifndef MOTION_PRIMITIVES_MOTION_PRIMITIVE_GRAPH_H
#define MOTION_PRIMITIVES_MOTION_PRIMITIVE_GRAPH_H

#include <glog/glog.h>

#include <Eigen/Core>
#include <iosfwd>
#include <nlohmann/json_fwd.hpp>

namespace motion_primitives {

class MotionPrimitive {
 private:
 public:
  int id_;
  float cost_;
  int spatial_dims_;
  Eigen::VectorXd start_state_;
  Eigen::VectorXd end_state_;
  // bool initialized_;

  friend std::ostream& operator<<(std::ostream& os,
                                  const MotionPrimitive<state_dim>& m);

  MotionPrimitive() = default;
  MotionPrimitive(int spatial_dims, const Eigen::MatrixXd& start_state,
                  const Eigen::MatrixXd& end_state, float cost)
      : spatial_dims_(spatial_dims),
        start_state_(start_state),
        end_state_(end_state),
        cost_(cost),
        initialized_(true) {
    CHECK_EQ(start_state_.rows(), end_state_.rows());
  };
  void translate(const Eigen::MatrixXd& new_start);
};

class MotionPrimitiveGraph {
 public:
  MotionPrimitiveGraph();

 private:
  Eigen::ArrayXXi edges_;
  Eigen::MatrixXd vertices_;
  std::vector<MotionPrimitive> mps_;
  double dispersion_;
  int spatial_dims_;
  int num_tiles_;
  bool tiling_;
};

void from_json(const nlohmann::json& json_data, MotionPrimitiveGraph& graph);

}  // namespace motion_primitives

#endif