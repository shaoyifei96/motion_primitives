#pragma once

#include <unordered_set>

#include "motion_primitives/graph_search.h"

namespace motion_primitives {

// State is the real node
// Node is a wrapper around state that also carries the cost
struct Node2 {
  static constexpr auto kInfCost = std::numeric_limits<double>::infinity();

  int state_index{0};  // used to retrieve mp from graph
  Eigen::VectorXd state;
  double motion_cost{kInfCost};
  double heuristic_cost{0.0};

  double total_cost() const noexcept { return motion_cost + heuristic_cost; }
};

struct NodeCost {
  Node2 node;
  double cost{Node2::kInfCost};  // best cost so far
};

// First convert VectorXd to VectorXi by some scaling then hash
// This is to avoid potential floating point error causing the same state to
// hash to different values
struct VectorXdHash : std::unary_function<Eigen::VectorXd, std::size_t> {
  std::size_t operator()(const Eigen::VectorXd& vd) const noexcept;
};

class GraphSearch2 : public GraphSearch {
 public:
  // Path history stores the parent node of this state and the best cost so far
  using PathHistory =
      std::unordered_map<Eigen::VectorXd, NodeCost, VectorXdHash>;

  // Base ctor
  using GraphSearch::GraphSearch;

  // Search for a path from start_state to end_state, stops if no path found
  // (returns empty vector) or reach within distance_threshold of start_state
  // parallel == true will expand nodes in parallel (~x2 speedup)
  std::vector<MotionPrimitive> Search(const Eigen::VectorXd& start_state,
                                      const Eigen::VectorXd& end_state,
                                      double distance_threshold,
                                      bool parallel = false) const;

  // internal use only, stores (wall) time spent on different parts
  mutable std::unordered_map<std::string, double> timings;

  std::vector<Eigen::VectorXd> GetVisitedStates() const noexcept;

 private:
  std::vector<MotionPrimitive> RecoverPath(const PathHistory& history,
                                           const Node2& end_node) const;
  std::vector<Node2> Expand(const Node2& node) const;
  std::vector<Node2> ExpandPar(const Node2& node) const;
  MotionPrimitive GetPrimitiveBetween(const Node2& start_node,
                                      const Node2& end_node) const;
  using StateSet = std::unordered_set<Eigen::VectorXd, VectorXdHash>;
  mutable StateSet visited_states_;
};

}  // namespace motion_primitives
