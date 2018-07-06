#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
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
#include "collision_free_goal.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

bool take_photo = false;
bool take_rgb = false;
std::string item;
std::string depth_img_path_;
std::string rgb_img_path_;
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
                 cv::Mat image = cv_ptr->image;
            	   image.convertTo(image, CV_16UC1, 1000);
            	   cv::imwrite(depth_img_path_ + item, image);
                 take_photo = false;
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
            // cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
            if (image_data->encoding == "rgb8")
            {
                 cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
            	   cv::imwrite(rgb_img_path_ + item, cv_ptr->image);
                 take_rgb = false;
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
    ros::init(argc, argv, "tams_bio_ik_human_robot_mapping", 1);
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    ros::AsyncSpinner spinner(1);
    spinner.start();
    tf::TransformListener tf_listener;

    std::string mapfile_;
    pnh.getParam("mapfile", mapfile_);
    pnh.getParam("depth_img_path", depth_img_path_);
    pnh.getParam("rgb_img_path", rgb_img_path_);
    std::cout<<mapfile_<<std::endl;
    std::cout<<depth_img_path_<<std::endl;
    std::cout<<rgb_img_path_<<std::endl;

    // ros::Subscriber sub_depth = n.subscribe("/camera/depth/image_raw", 1, depth_Callback);
    ros::Subscriber sub_rgb = nh.subscribe("/camera/rgb/image_raw", 1, rgb_Callback);

    std::string group_name = "right_hand"; // right_arm_and_hand
    moveit::planning_interface::MoveGroupInterface mgi(group_name);
    std::string base_frame = mgi.getPoseReferenceFrame();

    mgi.setGoalTolerance(0.01);
    mgi.setPlanningTime(10);
    auto robot_model = mgi.getCurrentState()->getRobotModel();
    auto joint_model_group = robot_model->getJointModelGroup(group_name);
    moveit::core::RobotState robot_state(robot_model);

    ROS_WARN_STREAM("move to 'open' pose");
    mgi.setNamedTarget("open");
    mgi.move();

    std::vector<std::string> failed_images;

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
    std::vector <float> MapDirectionweights1{0.1,0.1,0.1,0.1,0.1};
    std::vector <float> MapDirectionweights2{0.1};

    std::ifstream mapfile(mapfile_);
    std::string line,items;
    while(std::getline(mapfile, line))
    {
        bio_ik::BioIKKinematicsQueryOptions ik_options;
        ik_options.replace = true;
        ik_options.return_approximate_solution = true;

        std::istringstream myline(line);
        std::vector<double> csvItem;

        while(std::getline(myline, items, ','))
        {
            // item = items.substr(0,items.size()-4);
            if (items[0]=='i')
            {
                item = items;
                std::cout<< item <<std::endl;
                // cv::Mat depth_image;
                // cv::Mat hand_shape;
                // depth_image = cv::imread("/home/sli/shadow_ws/imitation/src/shadow_regression/data/trainning/" + item, cv::IMREAD_ANYDEPTH); // Read the file
                // cv::Mat dispImage;
                // cv::normalize(depth_image  , dispImage, 0, 1, cv::NORM_MINMAX, CV_32F);
                // cv::imshow("depth_image", dispImage);

                // hand_shape = cv::imread("/home/sli/pr2_shadow_ws/src/shadow_regression/data/tams_handshape/" + item); // Read the file
                // cv::resize(hand_shape, hand_shape, cv::Size(640, 480));
                // cv::imshow("shape", hand_shape);
                // cv::waitKey(5); // Wait for a keystroke in the window
                continue;
            }
            csvItem.push_back(std::stof(items));
            // // std::cout<< csvItem.back()<<std::endl;
        }

        std::vector<std::unique_ptr<bio_ik::PositionGoal>> Positiongoals;
        for (int j = 0; j< MapPositionlinks.size(); j++)
        {
            int t = j * 3;
            tf::Vector3 position = tf::Vector3(csvItem[t], csvItem[t+1], csvItem[t+2]);
            // std::cout<< position.x()<< position.y()<< position.z()<<std::endl;

            // transform position from current rh_wrist into base_frame
            tf::Stamped<tf::Point> stamped_in(position, ros::Time::now(), "rh_wrist");
            tf::Stamped<tf::Vector3> stamped_out;
            tf_listener.waitForTransform(base_frame, "rh_wrist", ros::Time::now(), ros::Duration(5.0));
            tf_listener.transformPoint(base_frame, stamped_in, stamped_out);
            tf::Vector3 Mapposition = stamped_out;
            Mapposition.setZ(Mapposition.z() + 0.04);
            // std::cout << Mapposition.x() << Mapposition.y() << Mapposition.z() << std::endl;

            ik_options.goals.emplace_back(new bio_ik::PositionGoal(MapPositionlinks[j], Mapposition, MapPositionweights[j]));
        }

        // for (int j = 0; j< MapDirectionlinks1.size(); j++)
        // {
        //     int t = 30 + j * 3;
        //     tf::Vector3 proximal_direction = (tf::Vector3(csvItem[t], csvItem[t+1], csvItem[t+2])).normalized();
        //
        //     // transform position from current rh_wrist into base_frame
        //     tf::Stamped<tf::Point> stamped_in(proximal_direction, ros::Time::now(), "rh_wrist");
        //     tf::Stamped<tf::Vector3> stamped_out;
        //     tf_listener.waitForTransform(base_frame, "rh_wrist", ros::Time::now(), ros::Duration(5.0));
        //     tf_listener.transformVector(base_frame, stamped_in, stamped_out);
        //     tf::Vector3 Mapdirection = stamped_out;
        //
        //     ik_options.goals.emplace_back(new bio_ik::DirectionGoal(MapDirectionlinks1[j], tf::Vector3(0,0,1), Mapdirection.normalized(), MapDirectionweights1[j]));
        // }
        //
        // for (int j = 0; j< MapDirectionlinks2.size(); j++)
        // {
        //     int t = 45 + j*3;
        //     tf::Vector3 dummy_direction = (tf::Vector3(csvItem[t], csvItem[t+1], csvItem[t+2])).normalized();
        //
        //     // transform position from current rh_wrist into base_frame
        //     tf::Stamped<tf::Point> stamped_in(dummy_direction, ros::Time::now(), "rh_wrist");
        //     tf::Stamped<tf::Vector3> stamped_out;
        //     tf_listener.waitForTransform(base_frame, "rh_wrist", ros::Time::now(), ros::Duration(5.0));
        //     tf_listener.transformVector(base_frame, stamped_in, stamped_out);
        //     tf::Vector3 Mapdirection = stamped_out;
        //
        //     ik_options.goals.emplace_back(new bio_ik::DirectionGoal(MapDirectionlinks2[j], tf::Vector3(0,0,1), Mapdirection.normalized(), MapDirectionweights2[j]));
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
            // set two wrist joints always zero
            // joint_values[0] = 0.0;
            // joint_values[1] = 0.0;
            mgi.setJointValueTarget(joint_values);
            if (!(static_cast<bool>(mgi.plan(shadow_plan))))
            {
                std::cout<< "Failed to plan pose " << item << std::endl;
                failed_images.push_back(item);
                continue;
            }

            if(!(static_cast<bool>(mgi.execute(shadow_plan))))
            {
                std::cout << "Failed to execute pose " << item << std::endl;
                failed_images.push_back(item);
                continue;
            }

            std::cout << "Moved to " << item <<". Take photo now" << std::endl;
            ros::Duration(1).sleep();
            // take_photo = true;
            take_rgb = true;

            // can not move robot and change image_count when taking photoes.
            while (take_rgb || take_photo)
                ros::Duration(0.1).sleep();
        }
        else
        {
            std::cout << "Did not find IK solution" << std::endl;
            failed_images.push_back(item);
        }
        for (int j = 0; j <ik_options.goals.size();j++)
            ik_options.goals[j].reset();
        // cv::destroyAllWindows();
        // cv::waitKey(1);
    }

    // save failed images num in order to check and run collision_free goal
    if (failed_images.size() == 0)
        std::cout << "No failed poses" << std::endl;
    else
    {
        std::ofstream outFile;
        outFile.open("/home/sli/pr2_shadow_ws/src/shadow_regression/data/trainning/failed_images.csv",std::ios::app);
        for( auto& t : failed_images )
    		    outFile << t << std::endl;
    }

    ros::shutdown();
    return 0;
}
