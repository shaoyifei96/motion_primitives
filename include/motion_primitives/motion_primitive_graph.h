#ifndef MOTION_PRIMITIVES_MOTION_PRIMITIVE_GRAPH_H
#define MOTION_PRIMITIVES_MOTION_PRIMITIVE_GRAPH_H

#include <glog/logging.h>

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

  friend std::ostream& operator<<(std::ostream& os, const MotionPrimitive& m);

  MotionPrimitive() = default;
  MotionPrimitive(int spatial_dims, const Eigen::VectorXd& start_state,
                  const Eigen::VectorXd& end_state, float cost)
      : spatial_dims_(spatial_dims),
        start_state_(start_state),
        end_state_(end_state),
        cost_(cost) {
    CHECK_EQ(start_state_.rows(), end_state_.rows());
  };
  void translate(const Eigen::VectorXd& new_start);
};

class MotionPrimitiveGraph {
 public:
  MotionPrimitiveGraph() = default;
  friend class GraphSearch;
  friend void from_json(const nlohmann::json& json_data,
                        MotionPrimitiveGraph& graph);
  friend std::ostream &operator<<(std::ostream &out, const MotionPrimitiveGraph& graph);

 private:
  Eigen::ArrayXXi edges_;
  Eigen::MatrixXd vertices_;
  std::vector<MotionPrimitive> mps_;
  double dispersion_;
  int spatial_dims_;
  int control_space_dim_;
  int num_tiles_;
  bool tiling_;
};
std::ostream &operator<<(std::ostream &out, const MotionPrimitiveGraph& graph);

void from_json(const nlohmann::json& json_data, MotionPrimitiveGraph& graph);

}  // namespace motion_primitives

#endif