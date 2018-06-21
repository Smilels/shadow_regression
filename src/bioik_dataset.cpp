#include <ros/ros.h>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include <moveit_msgs/MoveGroupActionResult.h>
#include <moveit_msgs/RobotState.h>
#include <tf/transform_broadcaster.h>
#include <pluginlib/class_loader.h>
#include <moveit/kinematics_base/kinematics_base.h>
#include <sensor_msgs/PointCloud.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

#include <vector>
#include <string>
#include <iostream>
#include <functional>
#include <memory>
#include <sstream>

#include <boost/program_options.hpp>

#include <tf_conversions/tf_eigen.h>

#include <getopt.h>

#include <kdl/frames.hpp>
#include <tf_conversions/tf_kdl.h>

#include <bio_ik/bio_ik.h>

#include <signal.h>
#include <errno.h>

void vprint() {
    std::cerr << std::endl;
}

template<class T>
void vprint(const T& a) {
    std::cerr << a << std::endl;
}

template<class T, class... AA>
void vprint(const T& a, AA... aa) {
    std::cerr << a << " ";
    vprint(aa...);
};

#define LOG(...) vprint(__VA_ARGS__)
#define LOG_VAR(v) LOG(#v, (v));


int main(int argc, char** argv)
{
    ros::init(argc, argv, "bio_ik_test", 1);
    ros::NodeHandle node_handle;
    ros::AsyncSpinner spinner(1);
    spinner.start();
    std::string group_name = "right_hand";
    moveit::planning_interface::MoveGroup move_group(group_name);

    auto robot_model = move_group.getCurrentState()->getRobotModel();
    auto joint_model_group = robot_model->getJointModelGroup(group_name);

    moveit::core::RobotState robot_state(robot_model);
    // build a sequency of goals to track
    std::vector<std::vector<std::unique_ptr<bio_ik::Goal>>> goal_trajectory;

    // track goals using bio ik
    std::vector<robot_state::RobotState> states;
    bio_ik::BioIKKinematicsQueryOptions ik_options;
    ik_options.replace = true;
    ik_options.return_approximate_solution = true;

    auto* th_goal = new bio_ik::PositionGoal();
    auto* ff_goal = new bio_ik::PositionGoal();
    auto* mf_goal = new bio_ik::PositionGoal();
    auto* rf_goal = new bio_ik::PositionGoal();
    auto* lf_goal = new bio_ik::PositionGoal();
    th_goal->setLinkName("rh_thtip");
    ff_goal->setLinkName("rh_fftip");
    mf_goal->setLinkName("rh_mftip");
    rg_goal->setLinkName("rh_rftip");
    lf_goal->setLinkName("rh_lftip");
    ik_options.goals.emplace_back(rh_goal);
    ik_options.goals.emplace_back(ff_goal);
    ik_options.goals.emplace_back(mf_goal);
    ik_options.goals.emplace_back(rf_goal);
    ik_options.goals.emplace_back(lf_goal);

    th_goal->setPosition(center + dl + dg);
    gg_goal->setPosition(center + dl - dg);
    mf_goal->setPosition(center + dr + dg);
    rf_goal->setPosition(center + dr - dg);
    lf_goal->setPosition(center + dr - dg);

    std::string reference_frame = "rh_wrist"

    double timeout = 0.2;
    double starttime = ros::Time::now().toSec();
    robot_state.setFromIK(
                      joint_model_group,           // active PR2 joints
                      EigenSTL::vector_Affine3d(), // no explicit poses here
                      reference_frame,  // no end effector links here
                      0, timeout,                      // take values from YAML file
                      moveit::core::GroupStateValidityCallbackFn(),
                      ik_options       // five fingertip position goals
                    );
    double time = ros::Time::now().toSec() - starttime;

    LOG(i, time, ik_options.solution_fitness);
    publish(robot_state);
    ros::shutdown();

    return 0;
}
