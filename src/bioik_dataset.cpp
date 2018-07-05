#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>

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
#include <sys/stat.h>

#include <bio_ik/bio_ik.h>

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
            	   cv::imwrite(depth_img_path_ + item , image);
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
            //cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
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
    ros::init(argc, argv, "bio_ik_human_robot_mapping", 1);
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    ros::AsyncSpinner spinner(1);
    spinner.start();
    tf::TransformListener tf_listener;

    std::string mapfile_;
    std::string handshape_;
    std::string jointsfile_;
    std::string end_effectorfile_;
    pnh.getParam("mapfile", mapfile_);
    pnh.getParam("depth_img_path", depth_img_path_);
    pnh.getParam("rgb_img_path", rgb_img_path_);
    pnh.getParam("handshape", handshape_);
    pnh.getParam("jointsfile", jointsfile_);
    pnh.getParam("end_effectorfile", end_effectorfile_);

    ros::Subscriber sub_depth = nh.subscribe("/camera/depth/image_raw", 1, depth_Callback);
    ros::Subscriber sub_rgb = nh.subscribe("/camera/rgb/image_raw", 1, rgb_Callback);

    std::string group_name = "right_hand";
    moveit::planning_interface::MoveGroupInterface mgi(group_name);
    moveit::planning_interface::MoveGroupInterface mgi_th("rh_thumb");
    moveit::planning_interface::MoveGroupInterface mgi_ff("rh_first_finger");
    moveit::planning_interface::MoveGroupInterface mgi_mf("rh_middle_finger");
    moveit::planning_interface::MoveGroupInterface mgi_rf("rh_ring_finger");
    moveit::planning_interface::MoveGroupInterface mgi_lf("rh_little_finger");

    std::string base_frame = mgi.getPoseReferenceFrame();
    mgi.setGoalTolerance(0.01);
    mgi.setPlanningTime(10);
    auto robot_model = mgi.getCurrentState()->getRobotModel();
    auto joint_model_group = robot_model->getJointModelGroup(group_name);
    moveit::core::RobotState robot_state(robot_model);

    ROS_WARN_STREAM("move to 'open' pose");
    mgi.setNamedTarget("open");
    mgi.move();

    // mgi_th.setPositionTarget(0.0598176, -0.0707044, 0.352295);
    // mgi_ff.setPositionTarget(0.0399553, -0.0865925, 0.387136);
    // mgi_th.move();
    // mgi_ff.move();

    std::vector<std::string> failed_images;

    double timeout = 0.2;

    std::ifstream mapfile(mapfile_);
    std::string line, items;
    while(std::getline(mapfile, line)){
        // track goals using bio ik
        bio_ik::BioIKKinematicsQueryOptions ik_options;
        ik_options.replace = true;
        ik_options.return_approximate_solution = true;

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
          "rh_lfmiddle",
          "rh_thdistal",
          "rh_ffdistal",
          "rh_mfdistal",
          "rh_rfdistal",
          "rh_lfdistal"
        };
        std::vector <float> MapPositionweights {1,1,1,1,1,0.2,0.2,0.2,0.2,0.2,0.2,0.1,0.1,0.1,0.1};

        std::istringstream myline(line);
        std::vector<double> csvItem;
        while(std::getline(myline, items, ','))
        {
            if (items[0]=='i')
            {
                item = items;
                std::cout<< item <<std::endl;
                // cv::Mat depth_image;
                cv::Mat hand_shape;
                // depth_image = cv::imread("/home/sli/shadow_ws/imitation/src/shadow_regression/data/trainning/" + item, cv::IMREAD_ANYDEPTH); // Read the file
                // cv::Mat dispImage;
                // cv::normalize(depth_image, dispImage, 0, 1, cv::NORM_MINMAX, CV_32F);
                // cv::imshow("depth_image", dispImage);

                // hand_shape = cv::imread(handshape_ + item); // Read the file
                // cv::resize(hand_shape, hand_shape, cv::Size(640, 480));
                // cv::imshow("shape", hand_shape);
                // cv::waitKey(3); // Wait for a keystroke in the window
                continue;
            }
            csvItem.push_back(std::stof(items));
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
            Mapposition.setZ(Mapposition.z() + 0.05);

            ik_options.goals.emplace_back(new bio_ik::PositionGoal(MapPositionlinks[j], Mapposition, MapPositionweights[j]));
        }

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

        // move to the solution position
        std::vector<double> joint_values;
        moveit::planning_interface::MoveGroupInterface::Plan shadow_plan;
        if (found_ik)
        {
            robot_state.copyJointGroupPositions(joint_model_group, joint_values);
            // set the angle of two wrist joint zero
            joint_values[0] = 0;
            joint_values[1] = 0;
            mgi.setJointValueTarget(joint_values);
            if (!(static_cast<bool>(mgi.plan(shadow_plan))))
            {
                std::cout<< "Failed to plan pose " << item << std::endl;
                failed_images.push_back(item);
                continue;
            }

            if(!(static_cast<bool>(mgi.execute(shadow_plan))))
            {
                std::cout << "Failed to execute pose " << item<< std::endl;
                failed_images.push_back(item);
                continue;
            }

            std::cout << "Moved to " << item <<". Take photo now" << std::endl;
            // ros::Duration(1).sleep();
            take_photo = true;
            take_rgb = true;

            // can not move robot when taking photoes.
            while (take_rgb || take_photo)
                ros::Duration(0.1).sleep();

            // save joint angles and end_effector pose
            geometry_msgs::PoseStamped end_effector_pose_th = mgi_th.getCurrentPose();
            geometry_msgs::PoseStamped end_effector_wrist_pose_th;
            tf_listener.waitForTransform("rh_wrist", base_frame, ros::Time::now(), ros::Duration(5.0));
            tf_listener.transformPose("rh_wrist", end_effector_pose_th, end_effector_wrist_pose_th);

            geometry_msgs::PoseStamped end_effector_pose_ff = mgi_ff.getCurrentPose();
            geometry_msgs::PoseStamped end_effector_wrist_pose_ff;
            tf_listener.waitForTransform("rh_wrist", base_frame, ros::Time::now(), ros::Duration(5.0));
            tf_listener.transformPose("rh_wrist", end_effector_pose_ff, end_effector_wrist_pose_ff);

            geometry_msgs::PoseStamped end_effector_pose_mf = mgi_mf.getCurrentPose();
            geometry_msgs::PoseStamped end_effector_wrist_pose_mf;
            tf_listener.waitForTransform("rh_wrist", base_frame, ros::Time::now(), ros::Duration(5.0));
            tf_listener.transformPose("rh_wrist", end_effector_pose_mf, end_effector_wrist_pose_mf);

            geometry_msgs::PoseStamped end_effector_pose_rf = mgi_rf.getCurrentPose();
            geometry_msgs::PoseStamped end_effector_wrist_pose_rf;
            tf_listener.waitForTransform("rh_wrist", base_frame, ros::Time::now(), ros::Duration(5.0));
            tf_listener.transformPose("rh_wrist", end_effector_pose_rf, end_effector_wrist_pose_rf);

            geometry_msgs::PoseStamped end_effector_pose_lf = mgi_lf.getCurrentPose();
            geometry_msgs::PoseStamped end_effector_wrist_pose_lf;
            tf_listener.waitForTransform("rh_wrist", base_frame, ros::Time::now(), ros::Duration(5.0));
            tf_listener.transformPose("rh_wrist", end_effector_pose_lf, end_effector_wrist_pose_lf);

            std::ofstream joints_file;
            joints_file.open(jointsfile_,std::ios::app);
            joints_file << item << ',' << std::to_string( joint_values[0]) << ',' << std::to_string( joint_values[1]) <<','
            << std::to_string( joint_values[2]) <<',' << std::to_string( joint_values[3]) <<',' << std::to_string( joint_values[4]) <<','
            << std::to_string( joint_values[5]) <<',' << std::to_string( joint_values[6]) <<',' << std::to_string( joint_values[7]) <<','
            << std::to_string( joint_values[8]) <<',' << std::to_string( joint_values[9]) <<',' << std::to_string( joint_values[10]) <<','
            << std::to_string( joint_values[11]) <<',' << std::to_string( joint_values[12]) <<',' << std::to_string( joint_values[13]) <<','
            << std::to_string( joint_values[14]) <<',' << std::to_string( joint_values[15]) <<',' << std::to_string( joint_values[16]) <<','
            << std::to_string( joint_values[17]) <<',' << std::to_string( joint_values[18]) <<',' << std::to_string( joint_values[19]) <<','
            << std::to_string( joint_values[20]) <<',' << std::to_string( joint_values[21]) <<',' << std::to_string( joint_values[22]) <<','
            << std::to_string( joint_values[23]) << std::endl;

            std::ofstream end_effector_file;
            end_effector_file.open(end_effectorfile_,std::ios::app);
            end_effector_file << item << ',' << std::to_string( end_effector_pose_th.pose.position.x ) << ',' << std::to_string( end_effector_pose_th.pose.position.y ) <<','<< std::to_string( end_effector_pose_th.pose.position.z ) <<','
            << std::to_string( end_effector_pose_th.pose.orientation.x) <<',' << std::to_string( end_effector_pose_th.pose.orientation.y) <<','
            << std::to_string( end_effector_pose_th.pose.orientation.z) <<',' << std::to_string( end_effector_pose_th.pose.orientation.w) <<','
            << std::to_string( end_effector_pose_ff.pose.position.x ) << ',' << std::to_string( end_effector_pose_ff.pose.position.y ) <<','<< std::to_string( end_effector_pose_ff.pose.position.z ) <<','
            << std::to_string( end_effector_pose_ff.pose.orientation.x) <<',' << std::to_string( end_effector_pose_ff.pose.orientation.y) <<','
            << std::to_string( end_effector_pose_ff.pose.orientation.z) <<',' << std::to_string( end_effector_pose_ff.pose.orientation.w) <<','
            << std::to_string( end_effector_pose_mf.pose.position.x ) << ',' << std::to_string( end_effector_pose_mf.pose.position.y ) <<',' << std::to_string( end_effector_pose_mf.pose.position.z ) <<','
            << std::to_string( end_effector_pose_mf.pose.orientation.x) <<',' << std::to_string( end_effector_pose_mf.pose.orientation.y) <<','
            << std::to_string( end_effector_pose_mf.pose.orientation.z) <<',' << std::to_string( end_effector_pose_mf.pose.orientation.w) <<','
            << std::to_string( end_effector_pose_rf.pose.position.x ) << ',' << std::to_string( end_effector_pose_rf.pose.position.y ) <<',' << std::to_string( end_effector_pose_rf.pose.position.z ) <<','
            << std::to_string( end_effector_pose_rf.pose.orientation.x) <<',' << std::to_string( end_effector_pose_rf.pose.orientation.y) <<','
            << std::to_string( end_effector_pose_rf.pose.orientation.z) <<',' << std::to_string( end_effector_pose_rf.pose.orientation.w) <<','
            << std::to_string( end_effector_pose_lf.pose.position.x ) << ',' << std::to_string( end_effector_pose_lf.pose.position.y ) <<','<< std::to_string( end_effector_pose_lf.pose.position.z ) <<','
            << std::to_string( end_effector_pose_lf.pose.orientation.x) <<',' << std::to_string( end_effector_pose_lf.pose.orientation.y) <<','
            << std::to_string( end_effector_pose_lf.pose.orientation.z) <<',' << std::to_string( end_effector_pose_lf.pose.orientation.w) << std::endl;
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
        outFile.open("/home/robot/workspace/shadow_hand/imitation/src/shadow_regression/data/trainning/failed_images.csv",std::ios::app);
        for( auto& t : failed_images )
    		    outFile << t << std::endl;
    }

    ros::shutdown();
    return 0;
}
