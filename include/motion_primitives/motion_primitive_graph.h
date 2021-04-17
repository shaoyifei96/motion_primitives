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
  mutable Eigen::MatrixXd sampled_positions_;
  // Moves the motion primitive to a new position by modifying it's start, end,
  // and polynomial coefficients
  void translate(const Eigen::VectorXd& new_start);
  // Evaluates a polynomial motion primitive at a time t and returns a vector of
  // size spatial_dim_. TODO: make work with evaluating velocities,
  // accelerations, etc., right now it only works for position
  Eigen::VectorXd evaluate_polynomial(float t) const;

 public:
  MotionPrimitive() = default;
  MotionPrimitive(int spatial_dims, const Eigen::VectorXd& start_state,
                  const Eigen::VectorXd& end_state, double cost,
                  double traj_time, const Eigen::MatrixXd& poly_coeffs)
      : cost_(cost),
        traj_time_(traj_time),
        spatial_dim_(spatial_dims),
        start_state_(start_state),
        end_state_(end_state),
        poly_coeffs_(poly_coeffs) {
    CHECK_EQ(start_state_.rows(), end_state_.rows());
  };

  // Samples a motion primitive's position at regular temporal intervals
  // step_size apart.
  // Each row is a position
  Eigen::MatrixXd sample_positions(double step_size = 0.1) const;

  int id() const noexcept { return id_; }
  double traj_time() const noexcept { return traj_time_; }
  int spatial_dim() const noexcept { return spatial_dim_; }
  const Eigen::VectorXd& end_state() const noexcept { return end_state_; }
  const Eigen::MatrixXd& poly_coeffs() const noexcept { return poly_coeffs_; }
  const Eigen::MatrixXd& sampled_positions() const noexcept {
    return sampled_positions_;
  }
};

class MotionPrimitiveGraph {
  friend class GraphSearch;
  friend void from_json(const nlohmann::json& json_data,
                        MotionPrimitiveGraph& graph);
  friend std::ostream& operator<<(std::ostream& out,
                                  const MotionPrimitiveGraph& graph);

 public:
  MotionPrimitive get_mp_between_indices(int i, int j) const {
    return mps_[edges_(i, j)];
  }

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
};

// Overrides a function from nlohmann::json to convert a json file into a
// MotionPrimitiveGraph object.
void from_json(const nlohmann::json& json_data, MotionPrimitiveGraph& graph);
// Creates the intermediate json objects to convert from a file location to a
// MotionPrimitiveGraph.
MotionPrimitiveGraph read_motion_primitive_graph(const std::string& s);

template <typename T>
std::ostream& operator<<(std::ostream& output, std::vector<T> const& values) {
  for (auto const& value : values) {
    output << value << "\n";
  }
  return output;
}

}  // namespace motion_primitives

#endif
