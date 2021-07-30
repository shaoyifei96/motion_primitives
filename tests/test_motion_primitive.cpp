#include <gtest/gtest.h>

#include "motion_primitives/motion_primitive_graph.h"
using namespace motion_primitives;

namespace {

TEST(MotionPrimitiveTest, TranslateTest) {
  Eigen::VectorXd start_state(4), end_state(4), max_state(4);
  start_state << 0, 0, 2, 2;
  end_state << 1, 1, 1, 1;
  max_state << 3, 3, 3, 3;

  RuckigMotionPrimitive rmp(2, start_state, end_state, max_state);
  Eigen::VectorXd new_start(2);
  new_start << 4,4;
  Eigen::Vector4d new_start_state(4, 4, 2, 2);
  Eigen::Vector4d new_end_state(5, 5, 1, 1);
  rmp.translate(new_start);
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(rmp.start_state_[i], new_start_state[i]);
    EXPECT_EQ(rmp.end_state_[i], new_end_state[i]);
  }
}

}  // namespace