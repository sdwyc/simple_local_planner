#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>
#include <vector>
#include <utility>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>

#include <sensor_msgs/PointCloud2.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_pcl/grid_map_pcl.hpp>
#include "grid_map_ros/GridMapRosConverter.hpp"

using namespace std;
using namespace grid_map;

const double PI = 3.1415926;

struct Point{
  float x;
  float y;
  float yaw; // degree
};

struct score{
  float dis; // distance to waypoint for last path point
  float feasible; // describe terrian elevation
  float v; // linear velocity for arrive waypoint fastly
};

double goalX;
double goalY;
bool DriveStart = false;
float stopDisThre = 0.2;
double odomTime;
float vehicleRoll = 0, vehiclePitch = 0, vehicleYaw = 0;
float vehicleRollRec = 0, vehiclePitchRec = 0, vehicleYawRec = 0;
float vehicleX = 0, vehicleY = 0, vehicleZ = 0;
float vehicleXRec = 0, vehicleYRec = 0, vehicleZRec = 0;
double vehicleLength = 0.6;
double vehicleWidth = 0.6;
float minVelocity = -0.5, maxVelocity = 0.5;
float minRotRate = -0.5, maxRotRate = 0.5;
float v_step = 0.05, w_step = 0.05;
float w_dis = 10, w_feasible = 0, w_v = 0;
nav_msgs::Path path;


vector<pair<float, float>> VelSpace;
vector<nav_msgs::Path> PathSpace;
int trajNum = (maxVelocity - minVelocity)/v_step * (maxRotRate - minRotRate)/w_step;


grid_map::GridMap elevation_map;
bool receivedGridMap = false;
ros::Publisher* pubVelSpace_;
ros::Publisher* pubPath_;
ros::Publisher* pubGoal_;

float simTime = 2.0; // The whole prediction time
float deltT = 0.1;  // Time interval

void TransFormFromBase2Map(const float &base_x, const float &base_y, const float &theta,
                           float &map_x, float &map_y ){
  float theta_1 = atan2(base_y, base_x);
  float radius = sqrt(base_x*base_x + base_y*base_y);
  map_x = vehicleX + radius * sin(theta+theta_1);
  map_y = vehicleY + radius * cos(theta+theta_1);
}

void visualization_goalref(float x, float y){
  visualization_msgs::Marker goal_marker;
  goal_marker.points.clear();
  goal_marker.header.frame_id = "/base_link";
  goal_marker.header.stamp = ros::Time::now();
  goal_marker.action = goal_marker.ADD;
  goal_marker.scale.x = 0.5;
  goal_marker.scale.y = 0.5;
  goal_marker.scale.z = 0.5;
  goal_marker.type = goal_marker.CUBE;
  goal_marker.color.a = 1.0;
  goal_marker.color.r = 1.0;
  goal_marker.color.g = 0.0;
  goal_marker.color.b = 0.0;
  geometry_msgs::Point goal;
  goal.x = x;
  goal.y = y;
  goal.z = 0.0;
  goal_marker.pose.position = goal;
  goal_marker.pose.orientation.w = 1.0;
  pubGoal_->publish(goal_marker);
}

// Sample velocity space, output: vel space,number: (v_upper-v_low)*(w_upper*w_low)
void VelSpaceSample(float v_upper, float v_low, float w_upper, float w_low){
  VelSpace.clear();
  float v_step = 0.05, w_step = 0.05;
  for(float i=v_low; i<=v_upper; i+=v_step){
    for(float j=w_low; j<=w_upper; j+=w_step){
      // std::cout << "i,j = " << i << ", " << j <<std::endl;
      pair<float, float> vel_pair = make_pair(i,j);
      VelSpace.push_back(vel_pair);
    }
  }
}
// predict next point pose under current pose, w is in rad, yaw is in degree
Point trajectoryPrediction(float curr_x, float curr_y,float curr_yaw, float v, float w){
  Point next_point;
  next_point.x = curr_x + deltT * v * cos(curr_yaw);
  next_point.y = curr_y + deltT * v * sin(curr_yaw);
  next_point.yaw = curr_yaw + deltT * w;
  if(next_point.yaw > PI/180.0) next_point.yaw -= PI/360.0;
  if(next_point.yaw < PI/-180.0) next_point.yaw += PI/360.0;
  return next_point;
}

float costFunc(float distance, float feasible, float vel){
  return -w_dis*distance - w_feasible*feasible + w_v*vel;
}

void odometryHandler(const nav_msgs::Odometry::ConstPtr& odom)
{
  odomTime = odom->header.stamp.toSec();

  double roll, pitch, yaw;
  geometry_msgs::Quaternion geoQuat = odom->pose.pose.orientation;
  tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

  vehicleRoll = roll;
  vehiclePitch = pitch;
  vehicleYaw = yaw;
  vehicleX = odom->pose.pose.position.x;
  vehicleY = odom->pose.pose.position.y;
  vehicleZ = odom->pose.pose.position.z;
}


void goalHandler(const geometry_msgs::PointStamped::ConstPtr& goal)
{
  goalX = goal->point.x;
  goalY = goal->point.y;
  DriveStart = true;
}

void gridMapHandler(const grid_map_msgs::GridMapConstPtr &msg)
{
	grid_map::GridMapRosConverter::fromMessage(*msg, elevation_map);
	receivedGridMap = true;
}
//visualize whole sampled velocity space
void visualization_velocity(){
  // if(pubVelSpace_->getNumSubscribers()!=0){
    visualization_msgs::MarkerArray velSpace_vis;
    visualization_msgs::Marker single_path;
    // Intialize Marker
    single_path.header.frame_id = "/base_link";
    single_path.header.stamp = ros::Time::now();
    single_path.action = single_path.ADD;
    single_path.scale.x = 0.02;
    single_path.scale.y = 0.02;
    single_path.scale.z = 0.02;
    single_path.type = single_path.LINE_STRIP;
    single_path.color.a = 1.0;
    single_path.color.r = 0.0;
    single_path.color.g = 0.0;
    single_path.color.b = 1.0;

    if(PathSpace.size()!=0){
      float best_score = -INFINITY;
      int bestID;
      for(int trajID = 0; trajID<PathSpace.size(); trajID++){
        single_path.points.clear();
        auto pointer = PathSpace[trajID].poses.begin();
        // three score component
        float distance = INFINITY; 
        float feasible = 0;
        float line_vel = minVelocity + trajID*v_step;
        while(pointer != PathSpace[trajID].poses.end()){
          // Scoring every path
          if(pointer == PathSpace[trajID].poses.end()-1){
            // Transform Goal point from map to base_link
            float goalXRef = cos(vehicleYaw)*(goalX - vehicleX) + sin(vehicleYaw)*(goalY - vehicleY);
            float goalYRef = -sin(vehicleYaw)*(goalX - vehicleX) + cos(vehicleYaw)*(goalY - vehicleY);
            // visualization_goalref(goalXRef, goalYRef);
            distance = sqrt((goalXRef-pointer->pose.position.x)*(goalXRef-pointer->pose.position.x)
                          +((goalYRef-pointer->pose.position.y)*(goalYRef-pointer->pose.position.y)));
          }
          if(pointer != PathSpace[trajID].poses.begin()){
            float dis_to_last = sqrt(((pointer-1)->pose.position.x-pointer->pose.position.x)*((pointer-1)->pose.position.x-pointer->pose.position.x)
                          +(((pointer-1)->pose.position.y-pointer->pose.position.y)*((pointer-1)->pose.position.y-pointer->pose.position.y)));

            // Transform point from base_link to map
            float Map_curr_p_X, Map_curr_p_Y;
            float Map_last_p_X, Map_last_p_Y;            
            TransFormFromBase2Map(pointer->pose.position.x, pointer->pose.position.y, vehicleYaw-vehicleYawRec,
                                  Map_curr_p_X, Map_curr_p_Y);
            TransFormFromBase2Map((pointer-1)->pose.position.x, (pointer-1)->pose.position.y, vehicleYaw-vehicleYawRec,
                                  Map_last_p_X, Map_last_p_X);            
            // Position curr_p{Map_curr_p_X, Map_curr_p_Y};
            // Position last_p{Map_last_p_X, Map_last_p_Y};
            // float last_height = isnan(elevation_map.atPosition("elevation", last_p)) ? 0 : elevation_map.atPosition("elevation", last_p);
            // float curr_height = isnan(elevation_map.atPosition("elevation", curr_p)) ? 0 :elevation_map.atPosition("elevation", last_p);
            // float deltH = fabs(curr_height - last_height);
            // float deltF = deltH/dis_to_last;
            // feasible += deltF;          
          }
          // for visualization
          single_path.points.push_back(pointer->pose.position);
          pointer++;
        }
        float score = costFunc(distance, feasible, line_vel);
        if(score > best_score){
          best_score = score;
          bestID = trajID;
        }
        // std::cout << "path pose size: " << single_path.points.size() << std::endl;
        single_path.id = trajID;
        velSpace_vis.markers.push_back(single_path);
      }
      path.header.frame_id = "base_link";
      path.header.stamp = ros::Time::now();
      path.poses = PathSpace[bestID].poses;
      pubPath_->publish(path);
    }
    // std::cout << "11111111111111111:  " << velSpace_vis.markers.size() << endl;
    pubVelSpace_->publish(velSpace_vis);
  // }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "localPlanner");
  ros::NodeHandle nh;
  ros::NodeHandle nhPrivate = ros::NodeHandle("~");

  nhPrivate.getParam("vehicleLength", vehicleLength);
  nhPrivate.getParam("vehicleWidth", vehicleWidth);
  path.header.frame_id = "/base_link";
  // nhPrivate.getParam("goalX", goalX);
  // nhPrivate.getParam("goalY", goalY);

  ros::Subscriber subOdometry = nh.subscribe<nav_msgs::Odometry>
                                ("/odom", 5, odometryHandler);

  ros::Subscriber subGoal = nh.subscribe<geometry_msgs::PointStamped> ("/way_point", 5, goalHandler);

	ros::Subscriber subGridmap = nh.subscribe<grid_map_msgs::GridMap>("/elevation_mapping/elevation_map", 5, gridMapHandler);

  ros::Publisher pubPath = nh.advertise<nav_msgs::Path> ("/path", 5);
  pubPath_ = &pubPath;

  ros::Publisher pubVelSpace = nh.advertise<visualization_msgs::MarkerArray> ("/velocity_space", 5);
  pubVelSpace_ = &pubVelSpace;
  
  ros::Publisher pubGoal = nh.advertise<visualization_msgs::Marker> ("/local_goal_vis", 5);
  pubGoal_ = &pubGoal;

  printf ("\nInitialization complete.\n\n");

  ros::Rate rate(10);
  bool status = ros::ok();
  while (status) {
    vehicleXRec = vehicleX;
    vehicleYRec = vehicleY;
    vehicleZRec = vehicleZ;
    vehicleRollRec = vehicleRoll;
    vehiclePitchRec = vehiclePitch;
    vehicleYawRec = vehicleYaw;
    ros::spinOnce();
    float disToGoal = sqrt((goalX-vehicleX)*(goalX-vehicleX) + (goalY-vehicleY)*(goalY-vehicleY));
    if(DriveStart && receivedGridMap){
      if(disToGoal >= stopDisThre){
        VelSpaceSample(maxVelocity, minVelocity, maxRotRate, minRotRate);
        // std::cout << "velocity sample space size: " << VelSpace.size() << std::endl; //! 60 is correct
        PathSpace.clear();
        for(int Rot=0; Rot<VelSpace.size(); Rot++){
          float v = VelSpace[Rot].first;
          float w = VelSpace[Rot].second;

          // Point current_pose{vehicleX, vehicleY, vehicleYaw};
          Point current_pose{0.0, 0.0, 0.0};
          float currTime = 0;
          nav_msgs::Path sample_path;
          while(currTime < simTime){
            // sample_path.poses.clear();
            geometry_msgs::PoseStamped pos;
            pos.pose.position.x = current_pose.x;
            pos.pose.position.y = current_pose.y;
            pos.pose.position.z = 0.0;
            auto q = tf::createQuaternionFromRPY(vehicleRoll, vehiclePitch, current_pose.yaw);
            q = q.normalize();
            
            pos.pose.orientation.w = q.getW();
            pos.pose.orientation.x = q.getX();
            pos.pose.orientation.y = q.getY();
            pos.pose.orientation.z = q.getZ();
            sample_path.poses.push_back(pos);
            current_pose = trajectoryPrediction(current_pose.x, current_pose.y, current_pose.yaw, v, w);
            // std::cout << "traj prediction: " << current_pose.x << ", " << current_pose.y << std::endl;
            // std::cout << "currTime" << currTime << ", " << "simTime: " << simTime << std::endl;
            // std::cout << "sample_path pose size: " << sample_path.poses.size() << std::endl;

            currTime += deltT;
          }
          PathSpace.push_back(sample_path);
        }
        visualization_velocity();
      }
      else{
        // Arrived Goal
        DriveStart = false;
      }
    }
    status = ros::ok();
    rate.sleep();
  }

  return 0;
}
