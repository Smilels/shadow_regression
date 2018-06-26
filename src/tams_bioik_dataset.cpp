#include <ros/ros.h>
#include <tf/transform_listener.h>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include <moveit_msgs/RobotState.h>
#include <moveit/kinematics_base/kinematics_base.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/conversions.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>

#include <bio_ik/bio_ik.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


int main(int argc, char** argv)
{
    ros::init(argc, argv, "bio_ik_human_robot_mapping", 1);
    ros::NodeHandle node_handle;
    ros::AsyncSpinner spinner(1);
    spinner.start();

    std::string group_name = "right_hand";
    moveit::planning_interface::MoveGroupInterface mgi(group_name);
    std::string current_frame = mgi.getPoseReferenceFrame();
    mgi.setGoalTolerance(0.01);
    mgi.setPlanningTime(10);
    auto robot_model = mgi.getCurrentState()->getRobotModel();
    auto joint_model_group = robot_model->getJointModelGroup(group_name);
    moveit::core::RobotState robot_state(robot_model);

    // track goals using bio ik
    std::vector<robot_state::RobotState> states;
    bio_ik::BioIKKinematicsQueryOptions ik_options;
    ik_options.replace = true;
    ik_options.return_approximate_solution = true;
    double timeout = 0.2;

    auto* th_goal = new bio_ik::PositionGoal();
    auto* ff_goal = new bio_ik::PositionGoal();
    auto* mf_goal = new bio_ik::PositionGoal();
    auto* rf_goal = new bio_ik::PositionGoal();
    auto* lf_goal = new bio_ik::PositionGoal();
    auto* th_middle_goal = new bio_ik::PositionGoal();
    auto* ff_middle_goal = new bio_ik::PositionGoal();
    auto* mf_middle_goal = new bio_ik::PositionGoal();
    auto* rf_middle_goal = new bio_ik::PositionGoal();
    auto* lf_middle_goal = new bio_ik::PositionGoal();

    th_goal->setLinkName("rh_thtip");
    ff_goal->setLinkName("rh_fftip");
    mf_goal->setLinkName("rh_mftip");
    rf_goal->setLinkName("rh_rftip");
    lf_goal->setLinkName("rh_lftip");
    th_middle_goal->setLinkName("rh_thmiddle");
    ff_middle_goal->setLinkName("rh_ffmiddle");
    mf_middle_goal->setLinkName("rh_mfmiddle");
    rf_middle_goal->setLinkName("rh_rfmiddle");
    lf_middle_goal->setLinkName("rh_lfmiddle");

    th_goal->setWeight(1);
    ff_goal->setWeight(1);
    mf_goal->setWeight(1);
    rf_goal->setWeight(1);
    lf_goal->setWeight(1);
    th_middle_goal->setWeight(0.2);
    ff_middle_goal->setWeight(0.2);
    mf_middle_goal->setWeight(0.2);
    rf_middle_goal->setWeight(0.2);
    lf_middle_goal->setWeight(0.2);


    tf::Vector3 th_position;
    tf::Vector3 ff_position;
    tf::Vector3 mf_position;
    tf::Vector3 rf_position;
    tf::Vector3 lf_position;
    tf::Vector3 th_middle_position;
    tf::Vector3 ff_middle_position;
    tf::Vector3 mf_middle_position;
    tf::Vector3 rf_middle_position;
    tf::Vector3 lf_middle_position;

    std::ifstream mapfile("/home/sli/pr2_ws/src/shadow_regression/data/trainning/human_robot_mapdata_pip_tams.csv");
    std::string line, item;
    int i=0;
    while(std::getline(mapfile, line)){
        mgi.setNamedTarget("open");
        mgi.move();
        ros::Duration(1).sleep();
        std::istringstream myline(line);
        std::vector<double> csvItem;
        i++;
        while(std::getline(myline, item, ','))
        {
            if (item[0]=='i')
            {
                std::cout<< item <<std::endl;
                cv::Mat depth_image;
                cv::Mat hand_shape;
                // depth_image = cv::imread("/home/sli/shadow_ws/imitation/src/shadow_regression/data/trainning/" + item, cv::IMREAD_ANYDEPTH); // Read the file
                // cv::Mat dispImage;
                // cv::normalize(depth_image, dispImage, 0, 1, cv::NORM_MINMAX, CV_32F);
                // cv::imshow("depth_image", dispImage);

                hand_shape = cv::imread("/home/sli/pr2_ws/src/shadow_regression/data/tams_handshape/" + std::to_string(i)+ ".png"); // Read the file
                cv::resize(hand_shape, hand_shape, cv::Size(640, 480));
                cv::imshow("shape", hand_shape);
                cv::waitKey(5); // Wait for a keystroke in the window
                continue;
            }
            csvItem.push_back(std::stof(item));
            // std::cout<< csvItem.back()<<std::endl;
        }

        // the constant transform.getorigin between /world and /rh_wrist
        tf::Vector3 transform_world_wrist(0.0, -0.01, 0.213+0.04);//0.034
        th_position = tf::Vector3(csvItem[0], csvItem[1], csvItem[2]) + transform_world_wrist;
        ff_position = tf::Vector3(csvItem[3], csvItem[4], csvItem[5]) + transform_world_wrist;
        mf_position = tf::Vector3(csvItem[6], csvItem[7], csvItem[8]) + transform_world_wrist;
        rf_position = tf::Vector3(csvItem[9], csvItem[10], csvItem[11]) + transform_world_wrist;
        lf_position = tf::Vector3(csvItem[12], csvItem[13], csvItem[14]) + transform_world_wrist;

        th_middle_position = tf::Vector3(csvItem[15], csvItem[16], csvItem[17]) + transform_world_wrist;
        ff_middle_position = tf::Vector3(csvItem[18], csvItem[19], csvItem[20]) + transform_world_wrist;
        mf_middle_position = tf::Vector3(csvItem[21], csvItem[22], csvItem[23]) + transform_world_wrist;
        rf_middle_position = tf::Vector3(csvItem[24], csvItem[25], csvItem[26]) + transform_world_wrist;
        lf_middle_position = tf::Vector3(csvItem[27], csvItem[28], csvItem[29]) + transform_world_wrist;

        th_goal->setPosition(th_position);
        ff_goal->setPosition(ff_position);
        mf_goal->setPosition(mf_position);
        rf_goal->setPosition(rf_position);
        lf_goal->setPosition(lf_position);
        th_middle_goal->setPosition(th_middle_position);
        ff_middle_goal->setPosition(ff_middle_position);
        mf_middle_goal->setPosition(mf_middle_position);
        rf_middle_goal->setPosition(rf_middle_position);
        lf_middle_goal->setPosition(lf_middle_position);

        ik_options.goals.emplace_back(th_goal);
        ik_options.goals.emplace_back(ff_goal);
        ik_options.goals.emplace_back(mf_goal);
        ik_options.goals.emplace_back(rf_goal);
        ik_options.goals.emplace_back(lf_goal);
        ik_options.goals.emplace_back(th_middle_goal);
        ik_options.goals.emplace_back(ff_middle_goal);
        ik_options.goals.emplace_back(mf_middle_goal);
        ik_options.goals.emplace_back(rf_middle_goal);
        ik_options.goals.emplace_back(lf_middle_goal);

        // set ik solver
        bool found_ik =robot_state.setFromIK(
                          joint_model_group,           // active Shadow joints
                          EigenSTL::vector_Affine3d(), // no explicit poses here
                          std::vector<std::string>(),  // rh_wrist
                          0, timeout,                      // take values from YAML file
                          moveit::core::GroupStateValidityCallbackFn(),
                          ik_options       // five fingertip position goals
                        );

        // move to the solution position
        std::vector<double> joint_values;
        if (found_ik)
        {
            robot_state.copyJointGroupPositions(joint_model_group, joint_values);
            // std::cout<< joint_values[1] << joint_values[2] << joint_values[3] <<std::endl;
            mgi.setJointValueTarget(joint_values);
            if (!static_cast<bool>(mgi.move()))
                    ROS_WARN_STREAM("Failed to execute state");
            ros::Duration(7).sleep();
        }
        else
        {
            ROS_INFO("Did not find IK solution");
        }
        // cv::destroyAllWindows();
        // cv::waitKey(1);
    }
    ros::shutdown();
    return 0;
}
