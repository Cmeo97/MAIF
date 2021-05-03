#include "AIC.h"
#include <random>
#include <iostream>
#include <cmath>
// Constant for class AIC constructor to define which robot to control
const int robot = 1;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "AIC_controller_babbling");
  // Variables to regulate the flow (Force to read once every 1ms the sensors)
  int count = 0;
  int cycles = 0;
  // Variable for desired position, set here the goal for the Panda for each joint
  std::vector<double> desiredPos(7);
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution_desired_q0{-1.2, 1.2};
  std::uniform_real_distribution<double> distribution_desired_q1{-1, 1};
  std::uniform_real_distribution<double> distribution_desired_q2{-0.5, 0.5};
  std::uniform_real_distribution<double> distribution_desired_q3{-1, -0.05};
  std::uniform_real_distribution<double> distribution_desired_q4{-1, 1};
  std::uniform_real_distribution<double> distribution_desired_q5{0.0, 1};
  std::uniform_real_distribution<double> distribution_desired_q6{1.0, 1.0};

  desiredPos[0] = 0.0;
  desiredPos[1] = 0.0;
  desiredPos[2] = 0.0;
  desiredPos[3] = 0.0;
  desiredPos[4] = 0.0;
  desiredPos[5] = 0.0;
  desiredPos[6] = 0.0;


  // Object of the class AIC which will take care of everything
  AIC AIC_controller(robot);
  // Set desired position in the AIC class
  AIC_controller.setGoal(desiredPos);
  int d = 0;
  // Main loop
  ros::Rate rate(1000);
  while (ros::ok()){
    // Manage all the callbacks and so read sensors
    ros::spinOnce();

    // Skip only first cycle to allow reading the sensory input first
    if ((count!=0)&&(AIC_controller.dataReady()==1)&&(cycles<45000)){
      AIC_controller.minimiseF(cycles, d);
      cycles ++;
      if (cycles == 6000){
        d = 1;
        desiredPos[0] = distribution_desired_q0(generator);
        desiredPos[1] = distribution_desired_q1(generator);
        desiredPos[2] = distribution_desired_q2(generator);
        desiredPos[3] = distribution_desired_q3(generator);
        desiredPos[4] = distribution_desired_q4(generator);
        desiredPos[5] = distribution_desired_q5(generator);
        desiredPos[6] = distribution_desired_q6(generator);
        AIC_controller.setGoal(desiredPos);
        cycles = 0;
      }
    }
    else
      count ++;

    rate.sleep();
  }
  return 0;
}
