/*
 * File:   AIC.h
 * Author: Corrado Pezzato, TU Delft, DCSC
 *
 * Created on April 14th, 2020
 *
 * Class to perform active inference control of the 7DOF Franka Emika Panda robot.
 *
 * This class takes care of everything, it subscribes to the topics containing
 * the sensory input and it perfoms free-energy minimization using gradient descent
 * updating the beliefs about the rosbot's states (i.e. joint values) and computing
 * the control actions. The control is in joint space.
 *
 */

#ifndef AIC_H
#define AIC_H
#define _USE_MATH_DEFINES

#include "ros/ros.h"
#include <vector>
#include <cmath>
#include <math.h>
#include <sensor_msgs/JointState.h>
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"
#include "geometry_msgs/PoseStamped.h"
#include "gazebo_msgs/LinkStates.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <random>

// Class AIC to hanle the subscribers and the publishers for the active inference controller
class AIC
{
public:
  // Constructor and destructor
  AIC(int whichRobot);
  ~AIC();

  // Callback to handle the proprioceptive sensory data from the topic /joint_states published at 1kHz
  void jointStatesCallback(const sensor_msgs::JointState::ConstPtr& msg);
  void end_effStatesCallback(const gazebo_msgs::LinkStates::ConstPtr& msg);
  // Method to set the necessary variables once the constructor is called
  void initVariables();
  // Main method which minimises the free-energy using gradient descent
  void minimiseF(int j, int d);
  // Calculate and send the torque commands to compute actions and further minimise the free-energy
  void computeActions();
  // Support method to control the program flow. Data ready returns one when the encoders has been read
  int dataReady();
  // Set desired position
  void setGoal(std::vector<double> desiredPos);

  void noise();
  void save_err();
  // get methods for sensory prediction errors
  std_msgs::Float64MultiArray getSPE();

private:

  // Variances associated with the active inference controller and the confidence relative to sensory input and beliefs
  double var_q, var_qdot, var_mu, var_muprime;
  // Precision matrices, diagonal matrices with the inverce of the variance
  Eigen::Matrix<double, 7, 7> SigmaP_yq0, SigmaP_yq1, SigmaP_mu, SigmaP_muprime;
  // Beliefs about the states and their derivatives mu, mu', mu'', column vectors of 7 elements
  Eigen::Matrix<double, 7, 1> mu, mu_p, mu_pp, mu_dot, mu_dot_p, mu_dot_pp, jointPos, jointVel;
  Eigen::Matrix<double, 7, 1>  err_mu, q_noise;
  // Desired robot's states, column vector of 7 elements
  Eigen::Matrix<double, 7, 1> mu_d;
  Eigen::Matrix<double, 4, 1> end_eff_d, end_eff_d_x, end_eff_d_y, end_eff_d_z;


  // Control actions,  column vector of 7 elements
  Eigen::Matrix<double, 7, 1> u;
  // Learning rates and integration step for the AIC
  double k_mu, k_a, h;
  // End Effectors
  double end_eff_x, end_eff_y, end_eff_z, end_eff, err_end_eff;
  // Sensory prediction errors
  double SPEq, SPEdq, SPEmu_p, SPEmu_pp;
  // Support variable to control the flow of the script
  int dataReceived;
  // ROS related Variables, node handle
  ros::NodeHandle nh;
  // Publishers for joint torques to the topics /panda_joint*_controller/command, and the free-energy
  ros::Publisher tauPub1, tauPub2, tauPub3, tauPub4, tauPub5, tauPub6, tauPub7, IFE_pub;
  // Subscriber for proprioceptive sensors (i.e. from joint_states) and camera (i.e. aruco_single/pose)
  ros::Subscriber sensorSub, end_effSub;
  // Support variables to contain the torques for the joints
  std_msgs::Float64 tau1, tau2, tau3, tau4, tau5, tau6, tau7, F;
  // Values for direct kinematics computation using DH parameters
  Eigen::Matrix<double, 7, 1> DH_a, DH_d, DH_alpha;
  Eigen::Matrix<double, 4, 4> DH_T, DH_A, T;
  Eigen::Matrix<double, 3, 1> eePosition;
  // Definition of variables in order to publish the beliefs about the states and the sensory prediction errors
  std_msgs::Float64MultiArray AIC_mu, AIC_mu_p, AIC_mu_pp, SPE;
  // Publishers for beliefs
  ros::Publisher beliefs_mu_pub, beliefs_mu_p_pub, beliefs_mu_pp_pub, SPE_pub;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution_q0{0, 0.5};
  std::normal_distribution<double> distribution_q1{0, 0.5};
  std::normal_distribution<double> distribution_q2{0, 0.5};
  std::normal_distribution<double> distribution_q3{0, 0.5};
  std::normal_distribution<double> distribution_q4{0, 0.5};
  std::normal_distribution<double> distribution_q5{0, 0.5};
  std::normal_distribution<double> distribution_q6{0, 0.5};


};

#endif
