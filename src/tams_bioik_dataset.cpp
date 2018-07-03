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
#include "collision_free_goal.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

bool take_photo = false;
bool take_rgb = false;
int image_count = 0;

// tf::Vector3 PointTransform(tf::StampedTransform transform, tf::Vector3 source_position, std::string source_frame, std::string target_frame)
// {
//   tf::Stamped<tf::Point> stamped_in(source_position, ros::Time::now(), source_frame);
//   tf::Vector3 target_position;
//
//   target_position = transform * stamped_in;
//   target_position.setZ(target_position.z() + 0.04);
//
//   return target_position;
// }
//
// tf::Vector3 VectorTransform(tf::StampedTransform transform, tf::Vector3 source_direction, std::string source_frame, std::string target_frame)
// {
//   tf::Vector3 target_direction;
//   tf::Vector3 end = source_direction;
//   tf::Vector3 origin = tf::Vector3(0,0,0);
//   target_direction = (transform * end) - (transform * origin);
//
//   return target_direction.normalized();
// }

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
    ros::NodeHandle node_handle;
    ros::AsyncSpinner spinner(1);
    spinner.start();
    tf::TransformListener tf_listener;

    std::string group_name = "right_hand"; // right_arm_and_hand
    moveit::planning_interface::MoveGroupInterface mgi(group_name);
    std::string base_frame = mgi.getPoseReferenceFrame();
    std::cout<< "base_frame: " << base_frame <<std::endl;
    mgi.setGoalTolerance(0.01);
    mgi.setPlanningTime(10);
    auto robot_model = mgi.getCurrentState()->getRobotModel();
    auto joint_model_group = robot_model->getJointModelGroup(group_name);
    moveit::core::RobotState robot_state(robot_model);

    ROS_WARN_STREAM("please move the hand to 'open' pose");
    mgi.setNamedTarget("open");
    mgi.move();

    double timeout = 0.2;
    int attempts = 3;
    std::vector<std::string> MapPositionlinks {
      "rh_thtip",
      "rh_fftip",
      "rh_mftip",
      "rh_rftip",
      "rh_lftip",
      "rh_thmiddle",
      "rh_ffmiddle",
      "rh_mfmiddle",
      "rh_rfmiddle",
      "rh_lfmiddle"
    };
    std::vector<std::string> MapDirectionlinks1 {
      "rh_thproximal",
      "rh_ffproximal",
      "rh_mfproximal",
      "rh_rfproximal",
      "rh_lfproximal",
    };
    std::vector<std::string> MapDirectionlinks2 {
      "rh_thmiddle",
      // "rh_ffmiddle",
      // "rh_mfmiddle",
      // "rh_rfmiddle",
      // "rh_lfmiddle"
    };
    std::vector <float> MapPositionweights {1,1,1,1,1,0.2,0.2,0.2,0.2,0.2};
    std::vector <float> MapDirectionweights1{0.2,0.2,0.2,0.2,0.2};
    std::vector <float> MapDirectionweights2{0.2};

    std::ifstream mapfile("/home/sli/pr2_shadow_ws/src/shadow_regression/data/trainning/human_robot_mapdata_pip_tams.csv");
    std::string line, item;
    while(std::getline(mapfile, line))
    {
        bio_ik::BioIKKinematicsQueryOptions ik_options;
        ik_options.replace = true;
        ik_options.return_approximate_solution = true;

        std::istringstream myline(line);
        std::vector<double> csvItem;
        image_count++;
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

                // hand_shape = cv::imread("/home/sli/pr2_shadow_ws/src/shadow_regression/data/tams_handshape/" + std::to_string(image_count)+ ".png"); // Read the file
                // cv::resize(hand_shape, hand_shape, cv::Size(640, 480));
                // cv::imshow("shape", hand_shape);
                // cv::waitKey(5); // Wait for a keystroke in the window
                continue;
            }
            csvItem.push_back(std::stof(item));
            // std::cout<< csvItem.back()<<std::endl;
        }

        for (int j = 0; j< MapPositionlinks.size(); j++)
        {
            int t = j * 3;
            tf::Vector3 position = tf::Vector3(csvItem[t], csvItem[t+1], csvItem[t+2]);

            // transform position from current rh_wrist into base_frame
            tf::Stamped<tf::Point> stamped_in(position, ros::Time::now(), "rh_wrist");
            tf::Stamped<tf::Vector3> stamped_out;
            tf_listener.waitForTransform(base_frame, "rh_wrist", ros::Time::now(), ros::Duration(5.0));
            tf_listener.transformPoint(base_frame, stamped_in, stamped_out);
            tf::Vector3 Mapposition = stamped_out;
            Mapposition.setZ(Mapposition.z() + 0.04);

            ik_options.goals.emplace_back(new bio_ik::PositionGoal(MapPositionlinks[j], Mapposition, MapPositionweights[j]));
        }

        for (int j = 0; j< MapDirectionlinks1.size(); j++)
        {
            int t = 30 + j * 3;
            tf::Vector3 proximal_direction = (tf::Vector3(csvItem[t], csvItem[t+1], csvItem[t+2])).normalized();

            // transform position from current rh_wrist into base_frame
            tf::Stamped<tf::Point> stamped_in(proximal_direction, ros::Time::now(), "rh_wrist");
            tf::Stamped<tf::Vector3> stamped_out;
            tf_listener.waitForTransform(base_frame, "rh_wrist", ros::Time::now(), ros::Duration(5.0));
            tf_listener.transformVector(base_frame, stamped_in, stamped_out);
            tf::Vector3 Mapdirection = stamped_out;

            ik_options.goals.emplace_back(new bio_ik::DirectionGoal(MapDirectionlinks1[j], tf::Vector3(0,0,1), Mapdirection.normalized(), MapDirectionweights1[j]));
        }

        // for (int j = 0; j< MapDirectionlinks2.size(); j++)
        // {
        //   int t = 45 + j*3;
        //   tf::Vector3 dummy_direction = (tf::Vector3(csvItem[t], csvItem[t+1], csvItem[t+2])).normalized();
        //
        //   // transform position from current rh_wrist into base_frame
        //   tf::Stamped<tf::Point> stamped_in(dummy_direction, ros::Time::now(), "rh_wrist");
        //   tf::Stamped<tf::Vector3> stamped_out;
        //   tf_listener.waitForTransform(base_frame, "rh_wrist", ros::Time::now(), ros::Duration(5.0));
        //   tf_listener.transformVector(base_frame, stamped_in, stamped_out);
        //   tf::Vector3 Mapdirection = stamped_out;
        //
        //   ik_options.goals.emplace_back(new bio_ik::DirectionGoal(MapDirectionlinks1[j], tf::Vector3(0,0,1), Mapdirection.normalized(), MapDirectionweights1[j]));
        // }

        // special design for the position shadow can not learn
        // self_collision_free goal
        // float collision_weight = 1;
        // ik_options.goals.emplace_back(new Collision_freeGoal(collision_weight));

        // set ik solver
        bool found_ik =robot_state.setFromIK(
                          joint_model_group,           // active Shadow joints
                          EigenSTL::vector_Affine3d(), // no explicit poses here
                          std::vector<std::string>(),
                          attempts, timeout,
                          moveit::core::GroupStateValidityCallbackFn(),
                          ik_options
                        );

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
