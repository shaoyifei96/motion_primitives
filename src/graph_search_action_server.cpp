#include <actionlib/server/simple_action_server.h>
#include <motion_primitives/graph_search.h>
#include <motion_primitives/utils.h>
#include <planning_ros_msgs/PlanTwoPointAction.h>
#include <planning_ros_msgs/SplineTrajectory.h>
#include <planning_ros_msgs/Trajectory.h>
#include <planning_ros_msgs/VoxelMap.h>
#include <ros/ros.h>

using namespace motion_primitives;
class PlanningServer {
 protected:
  ros::NodeHandle pnh_;
  actionlib::SimpleActionServer<planning_ros_msgs::PlanTwoPointAction> as_;
  ros::Publisher traj_vis_pub_;
  ros::Publisher spline_traj_pub_;
  planning_ros_msgs::VoxelMap voxel_map_;
  motion_primitives::MotionPrimitiveGraph graph_;
  ros::Subscriber map_sub_;

 public:
  PlanningServer(const ros::NodeHandle& nh)
      : pnh_(nh), as_(nh, "plan_local_trajectory", false) {
    std::string graph_file;
    pnh_.param("graph_file", graph_file, std::string("dispersionopt101.json"));
    ROS_INFO("Reading graph file %s", graph_file.c_str());
    graph_ = read_motion_primitive_graph(graph_file);
    traj_vis_pub_ = pnh_.advertise<planning_ros_msgs::Trajectory>(
        "viz_trajectory", 1, true);
    spline_traj_pub_ = pnh_.advertise<planning_ros_msgs::SplineTrajectory>(
        "spline_trajectory", 1, true);
    map_sub_ =
        pnh_.subscribe("voxel_map", 1, &PlanningServer::voxelMapCB, this);
    as_.registerGoalCallback(boost::bind(&PlanningServer::executeCB, this));
    as_.start();
  }

  ~PlanningServer(void) {}

  void executeCB() {
    const planning_ros_msgs::PlanTwoPointGoal::ConstPtr& msg =
        as_.acceptNewGoal();
    if (voxel_map_.resolution == 0.0) {
      ROS_ERROR(
          "Missing voxel map for motion primitive planner, aborting action "
          "server.");
      as_.setAborted();
      return;
    }
    GraphSearch gs(graph_, voxel_map_);
    Eigen::VectorXd start, goal;
    start.resize(graph_.state_dim());
    goal.resize(graph_.state_dim());
    // TODO fix for other size states
    if (graph_.spatial_dim() == 2) {
      start(0) = msg->p_init.position.x;
      start(1) = msg->p_init.position.y;
      start(2) = msg->v_init.linear.x;
      start(3) = msg->v_init.linear.y;
      goal(0) = msg->p_final.position.x;
      goal(1) = msg->p_final.position.y;
      goal(2) = msg->v_final.linear.x;
      goal(3) = msg->v_final.linear.y;
    } else {
      start(0) = msg->p_init.position.x;
      start(1) = msg->p_init.position.y;
      start(2) = msg->p_init.position.z;
      start(3) = msg->v_init.linear.x;
      start(4) = msg->v_init.linear.y;
      start(5) = msg->v_init.linear.z;
      goal(0) = msg->p_final.position.x;
      goal(1) = msg->p_final.position.y;
      goal(2) = msg->p_final.position.z;
      goal(3) = msg->v_final.linear.x;
      goal(4) = msg->v_final.linear.y;
      goal(5) = msg->v_final.linear.z;
    }

    const auto path = gs.Search({.start_state = start,
                                 .goal_state = goal,
                                 .distance_threshold = 0.5,
                                 .parallel_expand = true});
    if (path.empty()) {
      ROS_ERROR("Graph search failed, aborting action server.");
      as_.setAborted();
      return;
    }

    ROS_INFO("Graph search succeeded.");
    planning_ros_msgs::PlanTwoPointResult result;
    result.epoch = msg->epoch;
    result.execution_time = msg->execution_time;
    result.traj = planning_ros_msgs::SplineTrajectory();
    result.traj.header.stamp = ros::Time::now();
    result.traj.header.frame_id = voxel_map_.header.frame_id;
    result.success = true;
    result.traj = path_to_spline_traj_msg(path, result.traj.header,
                                          msg->p_init.position.z);
    spline_traj_pub_.publish(result.traj);
    traj_vis_pub_.publish(path_to_traj_msg(path, result.traj.header));
    as_.setSucceeded(result);
  }
  void voxelMapCB(const planning_ros_msgs::VoxelMap::ConstPtr& msg) {
    voxel_map_ = *msg;
  }
};

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "motion_primitives_action_server");
  ros::NodeHandle pnh("~");
  PlanningServer ps(pnh);
  ros::spin();
}
