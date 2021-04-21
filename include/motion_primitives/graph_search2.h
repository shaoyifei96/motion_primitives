#pragma once

#include <unordered_set>

#include "motion_primitives/graph_search.h"

namespace motion_primitives {

// First convert VectorXd to VectorXi by some scaling then hash
// This is to avoid potential floating point error causing the same state to
// hash to different values
// NOTE: Ideally this should be part of the implementation, we put in the public
// namespace so that we can test it
struct VectorXdHash : std::unary_function<Eigen::VectorXd, std::size_t> {
  std::size_t operator()(const Eigen::VectorXd& vd) const noexcept;
};

class GraphSearch2 : public GraphSearch {
 public:
  // Base ctor (TODO: remove this)
  using GraphSearch::GraphSearch;

  using State = Eigen::VectorXd;

  struct Option {
    State start_state;
    State goal_state;
    double distance_threshold;
    bool parallel_expand{false};
  };

  // Search for a path from start_state to end_state, stops if no path found
  // (returns empty vector) or reach within distance_threshold of start_state
  // parallel == true will expand nodes in parallel (~x2 speedup)
  std::vector<MotionPrimitive> Search(const Option& option);

  std::vector<Eigen::VectorXd> GetVisitedStates() const noexcept;
  const auto& timings() const noexcept { return timings_; }

 private:
  // State is the real node
  // Node is a wrapper around state that also carries the cost info
  struct Node2 {
    static constexpr auto kInfCost = std::numeric_limits<double>::infinity();

    int state_index{0};  // used to retrieve mp from graph
    State state;
    double motion_cost{kInfCost};
    double heuristic_cost{0.0};

    double total_cost() const noexcept { return motion_cost + heuristic_cost; }
  };

  // The state is the key of PathHistory and will not be stored here
  struct StateInfo {
    Node2 parent_node;                  // parent node of this state
    double best_cost{Node2::kInfCost};  // best cost reaching this state so far
  };

  // Path history stores the parent node of this state and the best cost so far
  using PathHistory = std::unordered_map<State, StateInfo, VectorXdHash>;
  std::vector<MotionPrimitive> RecoverPath(const PathHistory& history,
                                           const Node2& end_node) const;

  double ComputeHeuristic(const State& state,
                          const State& goal_state) const noexcept;

  // Stores all visited states
  std::vector<Node2> Expand(const Node2& node, const State& goal_state) const;
  std::vector<Node2> ExpandPar(const Node2& node,
                               const State& goal_state) const;
  // Helper function
  // oid ExpandSingle(int index1, int index2) const;

  MotionPrimitive GetPrimitiveBetween(const Node2& start_node,
                                      const Node2& end_node) const;

  using StateSet = std::unordered_set<State, VectorXdHash>;
  mutable StateSet visited_states_;
  // internal use only, stores (wall) time spent on different parts
  mutable std::unordered_map<std::string, double> timings_;
};

}  // namespace motion_primitives
