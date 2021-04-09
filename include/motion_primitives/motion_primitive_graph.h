#ifndef MOTION_PRIMITIVES_MOTION_PRIMITIVE_GRAPH_H
#define MOTION_PRIMITIVES_MOTION_PRIMITIVE_GRAPH_H

#include <glog/logging.h>

#include <Eigen/Core>
#include <iosfwd>
#include <nlohmann/json_fwd.hpp>

namespace motion_primitives {

class MotionPrimitive {
  friend std::ostream& operator<<(std::ostream& os, const MotionPrimitive& m);
  friend class GraphSearch;

 private:
  int id_;
  double cost_;
  double traj_time_;
  int spatial_dim_;
  Eigen::VectorXd start_state_;
  Eigen::VectorXd end_state_;
  Eigen::MatrixXd poly_coeffs_;

 public:
  MotionPrimitive() = default;
  MotionPrimitive(int spatial_dims, const Eigen::VectorXd& start_state,
                  const Eigen::VectorXd& end_state, double cost,
                  double traj_time, const Eigen::MatrixXd& poly_coeffs)
      : spatial_dim_(spatial_dims),
        start_state_(start_state),
        end_state_(end_state),
        cost_(cost),
        traj_time_(traj_time),
        poly_coeffs_(poly_coeffs) {
    CHECK_EQ(start_state_.rows(), end_state_.rows());
  };
  void translate(const Eigen::VectorXd& new_start);
  Eigen::VectorXd evaluate_polynomial(float t) const;
  Eigen::MatrixXd get_sampled_position(double step_size) const;
};

class MotionPrimitiveGraph {
  friend class GraphSearch;
  friend void from_json(const nlohmann::json& json_data,
                        MotionPrimitiveGraph& graph);
  friend std::ostream& operator<<(std::ostream& out,
                                  const MotionPrimitiveGraph& graph);

 private:
  Eigen::ArrayXXi edges_;
  Eigen::MatrixXd vertices_;
  std::vector<MotionPrimitive> mps_;
  Eigen::VectorXd max_state_;
  double dispersion_;
  double rho_;
  int spatial_dim_;
  int control_space_dim_;
  int state_dim_;
  int num_tiles_;
  bool tiling_;

 public:
  MotionPrimitiveGraph() = default;
};

void from_json(const nlohmann::json& json_data, MotionPrimitiveGraph& graph);
MotionPrimitiveGraph read_motion_primitive_graph(std::string s);

template <typename T>
std::ostream& operator<<(std::ostream& output, std::vector<T> const& values) {
  for (auto const& value : values) {
    output << value << std::endl;
  }
  return output;
}

}  // namespace motion_primitives

#endif