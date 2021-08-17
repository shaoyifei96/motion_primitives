// Copyright 2021 Laura Jarin-Lipschitz
#include <actionlib/server/simple_action_server.h>
#include <motion_primitives/graph_search.h>
#include <motion_primitives/utils.h>
#include <planning_ros_msgs/PlanTwoPointAction.h>
#include <planning_ros_msgs/SplineTrajectory.h>
#include <planning_ros_msgs/Trajectory.h>
#include <planning_ros_msgs/VoxelMap.h>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

namespace motion_primitives {
class PlanningServer {
 protected:
  ros::NodeHandle pnh_;
  actionlib::SimpleActionServer<planning_ros_msgs::PlanTwoPointAction> as_;
  ros::Publisher traj_vis_pub_;
  ros::Publisher spline_traj_pub_;
  ros::Publisher sg_pub_;
  planning_ros_msgs::VoxelMap voxel_map_;
  motion_primitives::MotionPrimitiveGraph graph_;
  ros::Subscriber map_sub_;

 public:
  explicit PlanningServer(const ros::NodeHandle& nh)
      : pnh_(nh), as_(nh, "plan_local_trajectory", false) {
    std::string graph_file;
    pnh_.param("graph_file", graph_file, std::string("dispersionopt101.json"));
    ROS_INFO("Reading graph file %s", graph_file.c_str());
    graph_ = read_motion_primitive_graph(graph_file);
    traj_vis_pub_ =
        pnh_.advertise<planning_ros_msgs::Trajectory>("trajectory", 1, true);
    spline_traj_pub_ = pnh_.advertise<planning_ros_msgs::SplineTrajectory>(
        "spline_trajectory", 1, true);
    map_sub_ =
        pnh_.subscribe("voxel_map", 1, &PlanningServer::voxelMapCB, this);
    sg_pub_ = pnh_.advertise<visualization_msgs::MarkerArray>("start_and_goal",
                                                              1, true);

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
    auto start_and_goal = populateStartGoal(msg);
    const auto& [start, goal] = start_and_goal;

    ROS_INFO_STREAM("Planner start: " << start.transpose());
    ROS_INFO_STREAM("Planner goal: " << goal.transpose());

    GraphSearch::Option options = {.start_state = start,
                                   .goal_state = goal,
                                   .distance_threshold = 0.5,
                                   .parallel_expand = true,
                                   .heuristic = "min_time",
                                   .access_graph = true};
    if (graph_.spatial_dim() == 2) {
      options.fixed_z = msg->p_init.position.z;
    }
    publishStartAndGoal(start_and_goal, options.fixed_z);
    GraphSearch gs(graph_, voxel_map_, options);
    const auto path = gs.Search();
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

  std::array<double, 3> pointMsgToArray(const geometry_msgs::Point& point) {
    return {point.x, point.y, point.z};
  }

  std::array<double, 3> pointMsgToArray(const geometry_msgs::Vector3& point) {
    return {point.x, point.y, point.z};
  }
  std::array<Eigen::VectorXd, 2> populateStartGoal(
      const planning_ros_msgs::PlanTwoPointGoal::ConstPtr& msg) {
    Eigen::VectorXd start, goal;
    start.resize(graph_.state_dim());
    goal.resize(graph_.state_dim());
    auto p_init = pointMsgToArray(msg->p_init.position);
    auto v_init = pointMsgToArray(msg->v_init.linear);
    auto a_init = pointMsgToArray(msg->a_init.linear);
    auto p_final = pointMsgToArray(msg->p_final.position);
    auto v_final = pointMsgToArray(msg->v_final.linear);
    auto a_final = pointMsgToArray(msg->a_final.linear);

    int spatial_dim = graph_.spatial_dim();
    for (int dim = 0; dim < spatial_dim; dim++) {
      start[dim] = p_init[dim];
      goal[dim] = p_final[dim];
      start[spatial_dim + dim] = v_init[dim];
      goal[spatial_dim + dim] = v_final[dim];
      if (graph_.control_space_dim() > 2) {
        start[2 * spatial_dim + dim] = a_init[dim];
        goal[2 * spatial_dim + dim] = a_final[dim];
      }
    }

    std::array<Eigen::VectorXd, 2> start_and_goal{start, goal};
    return start_and_goal;
  }

  void publishStartAndGoal(const std::array<Eigen::VectorXd, 2>& start_and_goal,
                           double fixed_z) {
    visualization_msgs::MarkerArray sg_markers;
    visualization_msgs::Marker start_marker, goal_marker;
    start_marker.header = voxel_map_.header;
    start_marker.pose.position.x = start_and_goal[0][0],
    start_marker.pose.position.y = start_and_goal[0][1];
    start_marker.pose.orientation.w = 1;
    start_marker.color.g = 1;
    start_marker.color.a = 1;
    start_marker.type = 2;
    start_marker.scale.x = start_marker.scale.y = start_marker.scale.z = 0.3;
    goal_marker = start_marker;
    goal_marker.id = 1;
    goal_marker.pose.position.x = start_and_goal[1][0],
    goal_marker.pose.position.y = start_and_goal[1][1];

    if (graph_.spatial_dim() == 2) {
      start_marker.pose.position.z = fixed_z;
      goal_marker.pose.position.z = fixed_z;
    } else {
      start_marker.pose.position.z = start_and_goal[0][2];
      goal_marker.pose.position.z = start_and_goal[1][2];
    }
    goal_marker.color.g = 0;
    goal_marker.color.r = 1;
    sg_markers.markers.push_back(start_marker);
    sg_markers.markers.push_back(goal_marker);
    sg_pub_.publish(sg_markers);
  }
};
}  // namespace motion_primitives

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "motion_primitives_action_server");
  ros::NodeHandle pnh("~");
  motion_primitives::PlanningServer ps(pnh);
  ros::spin();
}
