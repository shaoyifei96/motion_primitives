#ifndef MOTION_PRIMITIVES_MOTION_PRIMITIVE_GRAPH_H
#define MOTION_PRIMITIVES_MOTION_PRIMITIVE_GRAPH_H

#include <Eigen/Core>
#include <iosfwd>
#include <nlohmann/json_fwd.hpp>
#include <ruckig/ruckig.hpp>

namespace motion_primitives {

class MotionPrimitive {
 public:
  MotionPrimitive() = default;
  MotionPrimitive(int spatial_dim, const Eigen::VectorXd& start_state,
                  const Eigen::VectorXd& end_state, Eigen::VectorXd max_state)
      : spatial_dim_(spatial_dim),
        start_state_(start_state),
        end_state_(end_state),
        max_state_(max_state){};

  friend std::ostream& operator<<(std::ostream& os, const MotionPrimitive& m);

  int spatial_dim_;
  Eigen::VectorXd start_state_;
  Eigen::VectorXd end_state_;
  Eigen::VectorXd max_state_;

  double cost_;
  double traj_time_;
  Eigen::MatrixXd poly_coeffs_;

  // Evaluates a motion primitive at a time t and returns a state vector
  Eigen::VectorXd evaluate_primitive(float t) const;

  // Moves the motion primitive to a new position by modifying it's start, end,
  // and polynomial coefficients
  void translate(const Eigen::VectorXd& new_start);

  // Samples a motion primitive's position at regular temporal intervals
  // step_size apart.
  // Each row is a position
  Eigen::MatrixXd sample_positions(double step_size = 0.1) const;

  void populate(double cost, double traj_time,
                const Eigen::MatrixXd& poly_coeffs) {
    cost_ = cost;
    traj_time_ = traj_time;
    poly_coeffs_ = poly_coeffs;
  }

};

class RuckigMotionPrimitive : public MotionPrimitive {
  using MotionPrimitive::MotionPrimitive;

 public:
  void compute_ruckig();
  Eigen::VectorXd evaluate_primitive(float t) const;
};

class MotionPrimitiveGraph {
  friend class GraphSearch;
  friend void from_json(const nlohmann::json& json_data,
                        MotionPrimitiveGraph& graph);
  friend std::ostream& operator<<(std::ostream& out,
                                  const MotionPrimitiveGraph& graph);

 public:
  MotionPrimitive get_mp_between_indices(int i, int j) const noexcept {
    return mps_[edges_(i, j)];
  }

  double rho() const noexcept { return rho_; }
  int spatial_dim() const noexcept { return spatial_dim_; }
  int state_dim() const noexcept { return state_dim_; }
  int num_tiled_states() const noexcept { return edges_.rows(); }
  const auto& max_state() const noexcept { return max_state_; }

  bool HasEdge(int i, int j) const noexcept { return edges_(i, j) >= 0; }
  int NormIndex(int i) const noexcept { return std::floor(i / num_tiles_); }

 private:
  std::vector<MotionPrimitive> mps_;
  Eigen::ArrayXXi edges_;
  Eigen::MatrixXd vertices_;
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
