#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <sstream>
#include <cmath>

#include <wall_following_assignment/pid.h>

ros::Publisher g_cmd_pub;
double desired_distance_from_wall  = 1.0; // in meters
double forward_speed = 1.0;              // in meters / sec
float prev_distance_error = 0.0;          // previous distance error
float distance_error = 0.0;
ros::Time Start_time;                        // time point when the callback function was called

void laser_scan_callback(const sensor_msgs::LaserScan::ConstPtr& msg) {
  geometry_msgs::Twist cmd;
  cmd.linear.x = forward_speed;  // forward speed is fixed

  float dt = (ros::Time::now() - Start_time).toSec();  // Get dt

  // Populate this command based on the distance to the closest
  // object in laser scan. I.e. compute the cross-track error
  // as mentioned in the PID slides.
  float distance_from_wall = *std::min_element(msg->ranges.begin(), msg->ranges.end()); // find smallest distance from wall

  // Initiate distance error
  if ( distance_from_wall > desired_distance_from_wall)
  {
      distance_error = desired_distance_from_wall  - distance_from_wall; // calculate error
  }
  else
  {
      if(distance_from_wall < 0.5 * desired_distance_from_wall)
          distance_error = 0.15 * (desired_distance_from_wall  - distance_from_wall);
      distance_error = 0.5 * (desired_distance_from_wall  - distance_from_wall);
  }
  

  // Initiate gain variables (P, I, D)
  float K_p = 5.0;     // PID p gain
  float K_i = 0.04;    // PID i gain
  float K_d = 0.0004;   // PID d gain

  // Initiate P, I, D terms
  //float P_term = 0.0;    // initiate the variable P_term
  //float I_term = 0.0;    // initiate the variable I_term
  //float dError = 0.0;    // initiate the variable D error
  //float D_term = 0.0;    // initiate the variable D_term

  // You can populate the command based on either of the following two methods:
  // (1) using only the distance to the closest wall
  // (2) using the distance to the closest wall and the orientation of the wall
  //
  // If you select option 2, you might want to use cascading PID control.

  // calculate P, I, D terms
  P_term = K_p * distance_error;
  I_term += K_i * distance_error * dt;

  dError = distance_error - prev_distance_error;
  D_term = K_d * (dError / dt);


  // Calculate output
  // cmd.angular.z = ???
  float output_z = (P_term + I_term + D_term);
  output_z *= -1.0;

  cmd.angular.z = output_z;

  Start_time = ros::Time::now();
  prev_distance_error = distance_error;

  // publish the command
  g_cmd_pub.publish(cmd);
}


int main(int argc, char **argv) {
  ros::init(argc, argv, "wall_follower_node");
  ros::NodeHandle nh("~");

  // Getting params before setting up the topic subscribers
  // otherwise the callback might get executed with default
  // wall following parameters
  nh.getParam("forward_speed", forward_speed);
  nh.getParam("desired_distance_from_wall", desired_distance_from_wall );
  forward_speed = 0.2;

  // todo: set up the command publisher to publish at topic '/husky_1/cmd_vel'
  // using geometry_msgs::Twist messages
  g_cmd_pub = nh.advertise<geometry_msgs::Twist>("/husky_1/cmd_vel", 1);

  // todo: set up the laser scan subscriber
  // this will set up a callback function that gets executed
  // upon each spinOnce() call, as long as a laser scan
  // message has been published in the meantime by another node
  ros::Subscriber laser_sub = nh.subscribe("/husky_1/scan", 1, laser_scan_callback);

  ros::Rate rate(50);
  Start_time = ros::Time::now(); // initialize start time
  // this will return false on ctrl-c or when you call ros::shutdown()
  while (ros::ok()) {
    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}

