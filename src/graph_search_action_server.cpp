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

// TODO(laura) add clearFootprint
namespace motion_primitives {
class PlanningServer {
 protected:
  ros::NodeHandle pnh_;
  actionlib::SimpleActionServer<planning_ros_msgs::PlanTwoPointAction> as_;
  ros::Publisher traj_vis_pub_;
  ros::Publisher spline_traj_pub_;
  ros::Publisher viz_traj_pub_;
  ros::Publisher sg_pub_;
  planning_ros_msgs::VoxelMap voxel_map_;
  planning_ros_msgs::SplineTrajectory tracker_traj_;
  motion_primitives::MotionPrimitiveGraph graph_;
  ros::Subscriber map_sub_;
  ros::Subscriber traj_feedback_sub_;
  ros::Subscriber tracker_traj_sub_;
  int seg_num_{-1};

 public:
  explicit PlanningServer(const ros::NodeHandle& nh)
      : pnh_(nh), as_(nh, "plan_local_trajectory", false) {
    std::string graph_file;
    pnh_.param("graph_file", graph_file, std::string("dispersionopt101.json"));
    ROS_INFO("Reading graph file %s", graph_file.c_str());
    graph_ = read_motion_primitive_graph(graph_file);
    // traj_vis_pub_ =
    //     pnh_.advertise<planning_ros_msgs::Trajectory>("traj", 1, true);
    spline_traj_pub_ = pnh_.advertise<planning_ros_msgs::SplineTrajectory>(
        "trajectory", 1, true);
    viz_traj_pub_ =
        pnh_.advertise<planning_ros_msgs::Trajectory>("traj", 1, true);
    map_sub_ =
        pnh_.subscribe("voxel_map", 1, &PlanningServer::voxelMapCB, this);
    tracker_traj_sub_ =
        pnh_.subscribe("tracker_traj", 1, &PlanningServer::trajTrackerCB, this);
    sg_pub_ = pnh_.advertise<visualization_msgs::MarkerArray>("start_and_goal",
                                                              1, true);

    as_.registerGoalCallback(boost::bind(&PlanningServer::executeCB, this));
    as_.start();
  }

  ~PlanningServer(void) {}

  bool is_outside_map(const Eigen::Vector3i& pn, const Eigen::Vector3i& dim) {
    return pn(0) < 0 || pn(0) >= dim(0) || pn(1) < 0 || pn(1) >= dim(1) ||
           pn(2) < 0 || pn(2) >= dim(2);
  }

  void clear_footprint(planning_ros_msgs::VoxelMap& local_map,
                       const Eigen::Vector3f& start) {
    // TODO (laura) copy-pasted from local_plan_server
    // Clear robot footprint
    // TODO (YUEZHAN): pass robot radius as param
    // TODO (YUEZHAN): fix val_free;
    int8_t val_free = 0;
    ROS_WARN_ONCE("Value free is set as %d", val_free);
    double robot_r = 0.5;
    int robot_r_n = std::ceil(robot_r / local_map.resolution);

    std::vector<Eigen::Vector3i> clear_ns;
    for (int nx = -robot_r_n; nx <= robot_r_n; nx++) {
      for (int ny = -robot_r_n; ny <= robot_r_n; ny++) {
        for (int nz = -robot_r_n; nz <= robot_r_n; nz++) {
          clear_ns.push_back(Eigen::Vector3i(nx, ny, nz));
        }
      }
    }

    auto origin_x = local_map.origin.x;
    auto origin_y = local_map.origin.y;
    auto origin_z = local_map.origin.z;
    Eigen::Vector3i dim = Eigen::Vector3i::Zero();
    dim(0) = local_map.dim.x;
    dim(1) = local_map.dim.y;
    dim(2) = local_map.dim.z;
    auto res = local_map.resolution;
    const Eigen::Vector3i pn =
        Eigen::Vector3i(std::round((start(0) - origin_x) / res),
                        std::round((start(1) - origin_y) / res),
                        std::round((start(2) - origin_z) / res));

    for (const auto& n : clear_ns) {
      Eigen::Vector3i pnn = pn + n;
      int idx_tmp = pnn(0) + pnn(1) * dim(0) + pnn(2) * dim(0) * dim(1);
      if (!is_outside_map(pnn, dim) && local_map.data[idx_tmp] != val_free) {
        local_map.data[idx_tmp] = val_free;
        // ROS_ERROR("clearing!!! idx %d", idx_tmp);
      }
    }
  }

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
    std::array<Eigen::VectorXd, 2> start_and_goal = populateStartGoal(msg);
    auto [start, goal] = start_and_goal;

    ROS_INFO_STREAM("Planner start: " << start.transpose());
    ROS_INFO_STREAM("Planner goal: " << goal.transpose());

    const planning_ros_msgs::SplineTrajectory last_traj = msg->last_traj;
    double eval_time = msg->eval_time;

    int planner_start_index = 0;
    bool compute_first_mp = last_traj.data.size() > 0;
    int seg_num = -1;
    double mp_time;
    double poly_start_time = 0;
    std::shared_ptr<MotionPrimitive> mp;
    if (compute_first_mp) {
      // Figure out which segment of the last trajectory we will be at at the
      // requested eval time
      for (int i = 0; i < last_traj.data[0].segments; i++) {
        auto seg = last_traj.data[0].segs[i];
        poly_start_time += seg.dt;
        if (poly_start_time > eval_time) {
          poly_start_time -= seg.dt;
          seg_num = i;
          break;
        }
      }
      // todo(laura) what to do if eval time is past the end of the traj
      if (seg_num == -1) seg_num = last_traj.data[0].segments - 1;
      ROS_INFO_STREAM("Seg num " << seg_num);
      ROS_INFO_STREAM("Poly start time " << poly_start_time);
      ROS_INFO_STREAM("Eval time " << eval_time);

      // Planner will start at the end of the segment we are evaluating at. Will
      // add a custom first primtive after the planner runs
      planner_start_index = last_traj.data[0].segs[seg_num].end_index;

      // Recover Motion Primitive from the graph using the indices
      mp = graph_.get_mp_between_indices(
          last_traj.data[0].segs[seg_num].end_index,
          last_traj.data[0].segs[seg_num].start_index);
      Eigen::VectorXd seg_start(graph_.spatial_dim());
      for (int i = 0; i < graph_.spatial_dim(); i++) {
        seg_start(i) = last_traj.data[i].segs[seg_num].coeffs[0];
      }
      // MP from graph is not in correct translation, it must be moved to the
      // time 0 point of the segment (retrieved using the first
      // polynomial coefficient). We also want the absolute position of the
      // planner to start from the end of the segment we are on.
      mp->translate(seg_start);
      mp_time = mp->traj_time_;
      start = mp->end_state_;
      start_and_goal[0] = start;
      ROS_INFO_STREAM("planner uncropped start" << start.transpose());

    } else {
      ROS_WARN("Unable to compute first MP, starting planner from rest.");
    }

    double tol_pos;
    pnh_.param("tol_pos", tol_pos, 0.5);

    GraphSearch::Option options = {.start_state = start,
                                   .goal_state = goal,
                                   .distance_threshold = tol_pos,
                                   .parallel_expand = true,
                                   .heuristic = "min_time",
                                   .access_graph = false,
                                   .start_index = planner_start_index};
    if (graph_.spatial_dim() == 2) options.fixed_z = msg->p_init.position.z;

    publishStartAndGoal(start_and_goal, options.fixed_z);
    Eigen::Vector3f map_start;
    map_start(0) = start(0);
    map_start(1) = start(1);
    map_start(2) = msg->p_init.position.z;
    clear_footprint(voxel_map_, map_start);

    GraphSearch gs(graph_, voxel_map_, options);

    auto [path, nodes] = gs.Search();

    if (path.empty()) {
      ROS_ERROR("Graph search failed, aborting action server.");
      as_.setAborted();
      return;
    }
    if (compute_first_mp) {
      // To our planned trajectory, we add a cropped motion primitive that is in
      // the middle of the last_traj segment that we are evaluating at

      Eigen::VectorXd cropped_start(graph_.state_dim());
      Eigen::MatrixXd cropped_poly_coeffs;

      double shift_time = 0;
      if (seg_num == 0)
        shift_time = (mp_time - last_traj.data[0].segs[0].dt) + eval_time;
      else
        shift_time = eval_time - poly_start_time;
      cropped_poly_coeffs =
          GraphSearch::shift_polynomial(mp->poly_coeffs_, shift_time);

      for (int i = 0; i < graph_.spatial_dim(); i++) {
        cropped_start(i) =
            cropped_poly_coeffs(i, cropped_poly_coeffs.cols() - 1);
        cropped_start(graph_.spatial_dim() + i) =
            cropped_poly_coeffs(i, cropped_poly_coeffs.cols() - 2);
        if (graph_.control_space_dim() > 2) {
          cropped_start(2 * graph_.spatial_dim() + i) =
              cropped_poly_coeffs(i, cropped_poly_coeffs.cols() - 3);
        }
      }
      ROS_INFO_STREAM("Cropped Start " << cropped_start.transpose());

      auto first_mp = graph_.createMotionPrimitivePtrFromGraph(
          cropped_start, path[0]->start_state_);

      double new_seg_time = mp_time - shift_time;
      first_mp->populate(0, new_seg_time, cropped_poly_coeffs,
                         last_traj.data[0].segs[seg_num].start_index,
                         last_traj.data[0].segs[seg_num].end_index);
      path.insert(path.begin(), first_mp);
      // path[0] = first_mp;
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
    auto viz_traj_msg = path_to_traj_msg(path, result.traj.header, msg->p_init.position.z);
    viz_traj_pub_.publish(viz_traj_msg);
    as_.setSucceeded(result);
  }

  void voxelMapCB(const planning_ros_msgs::VoxelMap::ConstPtr& msg) {
    voxel_map_ = *msg;
  }

  void trajTrackerCB(const planning_ros_msgs::SplineTrajectory::ConstPtr& msg) {
    tracker_traj_ = *msg;
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
    start_marker.scale.x = start_marker.scale.y = start_marker.scale.z = 1;
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
