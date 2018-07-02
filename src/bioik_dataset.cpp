#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

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

bool take_photo = false;
bool take_rgb = false;
int image_count = 0;

void depth_Callback(const sensor_msgs::Image::ConstPtr &image_data)
{
  if (take_photo)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      //cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
      if (image_data->encoding == "32FC1")
      {
         cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::TYPE_32FC1);
         take_photo = false;
         cv::Mat image = cv_ptr->image;
    	 image.convertTo(image, CV_16UC1, 1000);
    	 cv::imwrite("/home/robot/workspace/shadow_hand/imitation/src/shadow_regression/data/depth_shadow/" + std::to_string(image_count ) + ".png", image);
      }
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
  }
}

void rgb_Callback(const sensor_msgs::Image::ConstPtr &image_data)
{
  if (take_rgb)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      //cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
      if (image_data->encoding == "rgb8")
      {
         cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
         take_rgb = false;
         cv::Mat image = cv_ptr->image;
    	 cv::imwrite("/home/robot/workspace/shadow_hand/imitation/src/shadow_regression/data/rgb_shadow/" + std::to_string(image_count ) + ".png", image);
      }
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
  }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "bio_ik_human_robot_mapping", 1);
    ros::NodeHandle n;
    ros::AsyncSpinner spinner(1);
    spinner.start();
    ros::Subscriber sub_depth = n.subscribe("/camera/depth/image_raw", 1, depth_Callback);
    ros::Subscriber sub_rgb = n.subscribe("/camera/rgb/image_raw", 1, rgb_Callback);

    std::string group_name = "right_hand";
    moveit::planning_interface::MoveGroupInterface mgi(group_name);
    std::string current_frame = mgi.getPoseReferenceFrame();
    mgi.setGoalTolerance(0.01);
    mgi.setPlanningTime(10);
    auto robot_model = mgi.getCurrentState()->getRobotModel();
    auto joint_model_group = robot_model->getJointModelGroup(group_name);
    moveit::core::RobotState robot_state(robot_model);

    double timeout = 0.2;

    std::ifstream mapfile("/home/robot/workspace/shadow_hand/imitation/src/shadow_regression/data/training/human_robot_mapdata_whole.csv");
    std::string line, item;
    while(std::getline(mapfile, line)){
        // mgi.setNamedTarget("open");
        // mgi.move();

        // track goals using bio ik
        bio_ik::BioIKKinematicsQueryOptions ik_options;
        ik_options.replace = true;
        ik_options.return_approximate_solution = true;

        auto* th_goal = new bio_ik::PositionGoal();
        auto* ff_goal = new bio_ik::PositionGoal();
        auto* mf_goal = new bio_ik::PositionGoal();
        auto* rf_goal = new bio_ik::PositionGoal();
        auto* lf_goal = new bio_ik::PositionGoal();
        auto* th_pip_goal = new bio_ik::PositionGoal();
        auto* ff_pip_goal = new bio_ik::PositionGoal();
        auto* mf_pip_goal = new bio_ik::PositionGoal();
        auto* rf_pip_goal = new bio_ik::PositionGoal();
        auto* lf_pip_goal = new bio_ik::PositionGoal();
        auto* th_dip_goal = new bio_ik::PositionGoal();
        auto* ff_dip_goal = new bio_ik::PositionGoal();
        auto* mf_dip_goal = new bio_ik::PositionGoal();
        auto* rf_dip_goal = new bio_ik::PositionGoal();
        auto* lf_dip_goal = new bio_ik::PositionGoal();

        th_goal->setLinkName("rh_thtip");
        ff_goal->setLinkName("rh_fftip");
        mf_goal->setLinkName("rh_mftip");
        rf_goal->setLinkName("rh_rftip");
        lf_goal->setLinkName("rh_lftip");
        th_pip_goal->setLinkName("rh_thmiddle");
        ff_pip_goal->setLinkName("rh_ffmiddle");
        mf_pip_goal->setLinkName("rh_mfmiddle");
        rf_pip_goal->setLinkName("rh_rfmiddle");
        lf_pip_goal->setLinkName("rh_lfmiddle");
        th_dip_goal->setLinkName("rh_thdistal");
        ff_dip_goal->setLinkName("rh_ffdistal");
        mf_dip_goal->setLinkName("rh_mfdistal");
        rf_dip_goal->setLinkName("rh_rfdistal");
        lf_dip_goal->setLinkName("rh_lfdistal");

        th_goal->setWeight(1);
        ff_goal->setWeight(1);
        mf_goal->setWeight(1);
        rf_goal->setWeight(1);
        lf_goal->setWeight(1);
        th_pip_goal->setWeight(0.2);
        ff_pip_goal->setWeight(0.2);
        mf_pip_goal->setWeight(0.2);
        rf_pip_goal->setWeight(0.2);
        lf_pip_goal->setWeight(0.2);
        th_dip_goal->setWeight(0.2);
        ff_dip_goal->setWeight(0.1);
        mf_dip_goal->setWeight(0.1);
        rf_dip_goal->setWeight(0.1);
        lf_dip_goal->setWeight(0.1);

        tf::Vector3 th_position;
        tf::Vector3 ff_position;
        tf::Vector3 mf_position;
        tf::Vector3 rf_position;
        tf::Vector3 lf_position;
        tf::Vector3 th_pip_position;
        tf::Vector3 ff_pip_position;
        tf::Vector3 mf_pip_position;
        tf::Vector3 rf_pip_position;
        tf::Vector3 lf_pip_position;
        tf::Vector3 th_dip_position;
        tf::Vector3 ff_dip_position;
        tf::Vector3 mf_dip_position;
        tf::Vector3 rf_dip_position;
        tf::Vector3 lf_dip_position;

        std::istringstream myline(line);
        std::vector<double> csvItem;
        image_count++;
        while(std::getline(myline, item, ','))
        {
            if (item[0]=='i')
            {
                std::cout<< item <<std::endl;
                // cv::Mat depth_image;
                // cv::Mat hand_shape;
                // depth_image = cv::imread("/home/sli/shadow_ws/imitation/src/shadow_regression/data/trainning/" + item, cv::IMREAD_ANYDEPTH); // Read the file
                // cv::Mat dispImage;
                // cv::normalize(depth_image, dispImage, 0, 1, cv::NORM_MINMAX, CV_32F);
                // cv::imshow("depth_image", dispImage);

                // hand_shape = cv::imread("/home/sli/shadow_ws/imitation/src/shadow_regression/data/handshape/" + std::to_string(i)+ ".png"); // Read the file
                // cv::resize(hand_shape, hand_shape, cv::Size(640, 480));
                // cv::imshow("shape", hand_shape);
                // cv::waitKey(3); // Wait for a keystroke in the window
                continue;
            }
            csvItem.push_back(std::stof(item));
            // std::cout<< csvItem.back()<<std::endl;
        }
       // std::cout<< "print the position value pass to bioik: "<<std::endl;
       // std::cout<< csvItem[0]<< csvItem[1]<< csvItem[2] <<std::endl;
       // std::cout<< csvItem[3]<< csvItem[4]<< csvItem[5] <<std::endl;
       // std::cout<< csvItem[6]<<csvItem[7]<< csvItem[8] <<std::endl;
       // std::cout<< csvItem[9]<< csvItem[10]<< csvItem[11] <<std::endl;
       // std::cout<< csvItem[42]<< csvItem[43]<< csvItem[44] <<std::endl;

        // the constant transform.getorigin between /world and /rh_wrist
        tf::Vector3 transform_world_wrist(0.0, -0.01, 0.213+0.05);//0.034
        th_position = tf::Vector3(csvItem[0], csvItem[1], csvItem[2]) + transform_world_wrist;
        ff_position = tf::Vector3(csvItem[3], csvItem[4], csvItem[5]) + transform_world_wrist;
        mf_position = tf::Vector3(csvItem[6], csvItem[7], csvItem[8]) + transform_world_wrist;
        rf_position = tf::Vector3(csvItem[9], csvItem[10], csvItem[11]) + transform_world_wrist;
        lf_position = tf::Vector3(csvItem[12], csvItem[13], csvItem[14]) + transform_world_wrist;

        th_pip_position = tf::Vector3(csvItem[15], csvItem[16], csvItem[17]) + transform_world_wrist;
        ff_pip_position = tf::Vector3(csvItem[18], csvItem[19], csvItem[20]) + transform_world_wrist;
        mf_pip_position = tf::Vector3(csvItem[21], csvItem[22], csvItem[23]) + transform_world_wrist;
        rf_pip_position = tf::Vector3(csvItem[24], csvItem[25], csvItem[26]) + transform_world_wrist;
        lf_pip_position = tf::Vector3(csvItem[27], csvItem[28], csvItem[29]) + transform_world_wrist;

        th_dip_position = tf::Vector3(csvItem[30], csvItem[31], csvItem[32]) + transform_world_wrist;
        ff_dip_position = tf::Vector3(csvItem[33], csvItem[34], csvItem[35]) + transform_world_wrist;
        mf_dip_position = tf::Vector3(csvItem[36], csvItem[37], csvItem[38]) + transform_world_wrist;
        rf_dip_position = tf::Vector3(csvItem[39], csvItem[40], csvItem[41]) + transform_world_wrist;
        lf_dip_position = tf::Vector3(csvItem[42], csvItem[43], csvItem[44]) + transform_world_wrist;

        th_goal->setPosition(th_position);
        ff_goal->setPosition(ff_position);
        mf_goal->setPosition(mf_position);
        rf_goal->setPosition(rf_position);
        lf_goal->setPosition(lf_position);
        th_pip_goal->setPosition(th_pip_position);
        ff_pip_goal->setPosition(ff_pip_position);
        mf_pip_goal->setPosition(mf_pip_position);
        rf_pip_goal->setPosition(rf_pip_position);
        lf_pip_goal->setPosition(lf_pip_position);
        th_dip_goal->setPosition(th_dip_position);
        ff_dip_goal->setPosition(ff_dip_position);
        mf_dip_goal->setPosition(mf_dip_position);
        rf_dip_goal->setPosition(rf_dip_position);
        lf_dip_goal->setPosition(lf_dip_position);

        ik_options.goals.emplace_back(th_goal);
        ik_options.goals.emplace_back(ff_goal);
        ik_options.goals.emplace_back(mf_goal);
        ik_options.goals.emplace_back(rf_goal);
        ik_options.goals.emplace_back(lf_goal);
        ik_options.goals.emplace_back(th_pip_goal);
        ik_options.goals.emplace_back(ff_pip_goal);
        ik_options.goals.emplace_back(mf_pip_goal);
        ik_options.goals.emplace_back(rf_pip_goal);
        ik_options.goals.emplace_back(lf_pip_goal);
        ik_options.goals.emplace_back(th_dip_goal);
        ik_options.goals.emplace_back(ff_dip_goal);
        ik_options.goals.emplace_back(mf_dip_goal);
        ik_options.goals.emplace_back(rf_dip_goal);
        ik_options.goals.emplace_back(lf_dip_goal);

        robot_state = *mgi.getCurrentState();
        // set ik solver
        bool found_ik =robot_state.setFromIK(
                          joint_model_group,           // active Shadow joints
                          EigenSTL::vector_Affine3d(), // no explicit poses here
                          std::vector<std::string>(),  // rh_wrist
                          0, timeout,                      // take values from YAML file
                          moveit::core::GroupStateValidityCallbackFn(),
                          ik_options       // five fingertip position goals
                        );

        // can not move robot when taking photoes.
        while (take_rgb || take_photo)
        {
            std::cout << "taking photoes, please wait" << std::endl;
        }

        // move to the solution position
        std::vector<double> joint_values;
        moveit::planning_interface::MoveGroupInterface::Plan shadow_plan;
        if (found_ik)
        {
            robot_state.copyJointGroupPositions(joint_model_group, joint_values);
            // std::cout<< joint_values[1] << joint_values[2] << joint_values[3] <<joint_values[4]<<joint_values[0]<<std::endl;
            mgi.setJointValueTarget(joint_values);
            if (!(static_cast<bool>(mgi.plan(shadow_plan))))
            {
                std::cout<< "Failed to plan pose '" << image_count << std::endl;
                continue;
            }

            if(!(static_cast<bool>(mgi.execute(shadow_plan))))
            {
                std::cout << "Failed to execute pose '" << image_count<< std::endl;
                continue;
            }
            else
                std::cout << " moved to " << image_count << std::endl;

            ros::Duration(1).sleep();
            take_photo = true;
            take_rgb = true;
            ROS_WARN_STREAM("take photo now");
        }
        else
            ROS_INFO("Did not find IK solution");
        // cv::destroyAllWindows();
        // cv::waitKey(1);
    }
    ros::shutdown();
    return 0;
}
