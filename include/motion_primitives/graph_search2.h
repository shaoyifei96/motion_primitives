#pragma once

#include <unordered_set>

#include "motion_primitives/graph_search.h"

namespace motion_primitives {

struct Node2 {
  static constexpr auto kInfCost = std::numeric_limits<double>::infinity();
  int state_index{0};
  Eigen::VectorXd state;
  double motion_cost{kInfCost};
  double heuristic_cost{0.0};

  double total_cost() const noexcept { return motion_cost + heuristic_cost; }
};

struct NodeCost {
  Node2 node;
  double cost{Node2::kInfCost};
};

struct VectorXdHash : std::unary_function<Eigen::VectorXd, std::size_t> {
  std::size_t operator()(const Eigen::VectorXd& vd) const noexcept;
};

using PathHistory = std::unordered_map<Eigen::VectorXd, NodeCost, VectorXdHash>;
using StateSet = std::unordered_set<Eigen::VectorXd, VectorXdHash>;

class GraphSearch2 : public GraphSearch {
 public:
  using GraphSearch::GraphSearch;

  std::vector<MotionPrimitive> Search(const Eigen::VectorXd& start_state,
                                      const Eigen::VectorXd& end_state,
                                      double distance_threshold,
                                      bool parallel = false) const;

  mutable std::unordered_map<std::string, double> timings;

  std::vector<Eigen::VectorXd> GetVisitedStates() const noexcept;

 private:
  std::vector<MotionPrimitive> RecoverPath(const PathHistory& history,
                                           const Node2& end_node) const;
  std::vector<Node2> Expand(const Node2& node) const;
  std::vector<Node2> ExpandPar(const Node2& node) const;
  MotionPrimitive GetPrimitiveBetween(const Node2& start_node,
                                      const Node2& end_node) const;
  mutable StateSet visited_states_;
};

}  // namespace motion_primitives
