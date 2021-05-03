/*
 * File:   AIC.cpp
 * Author: Corrado Pezzato, TU Delft, DCSC
 *
 * Created on April 14th, 2019
 *
 * Class to perform active inference control of the 7DOF Franka Emika Panda robot.
 * Definition of the methods contained in AIC.h
 *
 */

#include "AIC.h"

  // Constructor which takes as argument the publishers and initialises the private ones in the class
  AIC::AIC(int whichRobot){

      // Initialize publishers on the topics /robot1/panda_joint*_controller/command for the joint efforts
      tauPub1 = nh.advertise<std_msgs::Float64>("/robot1/panda_joint1_controller/command", 20);
      tauPub2 = nh.advertise<std_msgs::Float64>("/robot1/panda_joint2_controller/command", 20);
      tauPub3 = nh.advertise<std_msgs::Float64>("/robot1/panda_joint3_controller/command", 20);
      tauPub4 = nh.advertise<std_msgs::Float64>("/robot1/panda_joint4_controller/command", 20);
      tauPub5 = nh.advertise<std_msgs::Float64>("/robot1/panda_joint5_controller/command", 20);
      tauPub6 = nh.advertise<std_msgs::Float64>("/robot1/panda_joint6_controller/command", 20);
      tauPub7 = nh.advertise<std_msgs::Float64>("/robot1/panda_joint7_controller/command", 20);
      sensorSub = nh.subscribe("/robot1/joint_states", 1, &AIC::jointStatesCallback, this);
      end_effSub = nh.subscribe("/gazebo/link_states", 1, &AIC::end_effStatesCallback, this);
      // Publisher for the free-energy and sensory prediction errors
      IFE_pub = nh.advertise<std_msgs::Float64>("panda_free_energy", 10);
      SPE_pub = nh.advertise<std_msgs::Float64MultiArray>("panda_SPE", 10);

      // Publishers for beliefs
      beliefs_mu_pub = nh.advertise<std_msgs::Float64MultiArray>("beliefs_mu", 10);
      beliefs_mu_p_pub = nh.advertise<std_msgs::Float64MultiArray>("beliefs_mu_p", 10);
      beliefs_mu_pp_pub = nh.advertise<std_msgs::Float64MultiArray>("beliefs_mu_pp", 10);

      // Publishers for beliefs
      beliefs_mu_pub = nh.advertise<std_msgs::Float64MultiArray>("beliefs_mu", 10);
      beliefs_mu_p_pub = nh.advertise<std_msgs::Float64MultiArray>("beliefs_mu_p", 10);
      beliefs_mu_pp_pub = nh.advertise<std_msgs::Float64MultiArray>("beliefs_mu_pp", 10);
       
    // Initialize the variables for thr AIC
    AIC::initVariables();
  }
  AIC::~AIC(){}

  void   AIC::jointStatesCallback(const sensor_msgs::JointState::ConstPtr& msg)
  {
    // Save joint values
    for( int i = 0; i < 7; i++ ) {
      jointPos(i) = msg->position[i];
      //jointPos(i) = jointPos(i) + q_noise(i);
      jointVel(i) = msg->velocity[i];
    }
    // If this is the first time we read the joint states then we set the current beliefs
    if (dataReceived == 0){
      // Track the fact that the encoders published
      dataReceived = 1;
      // The first time we retrieve the position we define the initial beliefs about the states
      mu = jointPos;
      mu_p = jointVel;
    }
  }


void   AIC::end_effStatesCallback(const gazebo_msgs::LinkStates::ConstPtr& msg)
  {
    // Save joint values
    end_eff_x = msg->pose[9].position.x;
    end_eff_y = msg->pose[9].position.y;
    end_eff_z = msg->pose[9].position.z;

    end_eff = sqrt(pow(end_eff_x,2) + pow(end_eff_y,2) + pow(end_eff_z,2));

  }

  void   AIC::initVariables(){

    // Support variable
    dataReceived = 0;

    // Variances associated with the beliefs and the sensory inputs
    var_mu = 5.0;
    var_muprime = 10.0;
    var_q = 1;
    var_qdot = 1;

    // Learning rates for the gradient descent (found that a ratio of 60 works good)
    k_mu = 11.67;
    k_a = 700;

    // Precision matrices (first set them to zero then populate the diagonal)
    SigmaP_yq0 = Eigen::Matrix<double, 7, 7>::Zero();
    SigmaP_yq1 = Eigen::Matrix<double, 7, 7>::Zero();
    SigmaP_mu = Eigen::Matrix<double, 7, 7>::Zero();
    SigmaP_muprime = Eigen::Matrix<double, 7, 7>::Zero();

    for( int i = 0; i < SigmaP_yq0.rows(); i = i + 1 ) {
      SigmaP_yq0(i,i) = 1/var_q;
      SigmaP_yq1(i,i) = 1/var_qdot;
      SigmaP_mu(i,i) = 1/var_mu;
      SigmaP_muprime(i,i) = 1/var_muprime;
    }

    // Initialize control actions
    u << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    // Initialize prior beliefs about the second ordet derivatives of the states of the robot
    mu_pp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    end_eff_d << 1.05342339042937, 0.683566227681276, 1.01240873890366, 0.920884331281194;
    end_eff_d_x << 0.114635239379695, 0.328372326019533, 0.559577132421758, 0.379710561368265;
    end_eff_d_y << 0.00209295101959208, 0.511476097394308, 0.003013607548644, -0.591409215177049;
    end_eff_d_z << 1.03170767445082, 0.312485466168603, 0.84239241467989, 0.594437411655472;




    // Integration step
    h = 0.001;

    // Resize Float64MultiArray messages
    AIC_mu.data.resize(7);
    AIC_mu_p.data.resize(7);
    AIC_mu_pp.data.resize(7);
    SPE.data.resize(2);
  }

  void AIC::noise(){

    q_noise(0) = distribution_q0(generator);
    q_noise(1) = distribution_q1(generator);
    q_noise(2) = distribution_q2(generator);
    q_noise(3) = distribution_q3(generator);
    q_noise(4) = distribution_q4(generator);
    q_noise(5) = distribution_q5(generator);
    q_noise(6) = distribution_q6(generator);
    std::ofstream fileWrite("noise.txt", std::ios::app );


  }
  void AIC::minimiseF(int j, int d){


    // Compute single sensory prediction errors
    SPEq = (jointPos.transpose()-mu.transpose())*SigmaP_yq0*(jointPos-mu);
    SPEdq = (jointVel.transpose()-mu_p.transpose())*SigmaP_yq1*(jointVel-mu_p);
    SPEmu_p = (mu_p.transpose()+mu.transpose()-mu_d.transpose())*SigmaP_mu*(mu_p+mu-mu_d);
    SPEmu_pp = (mu_pp.transpose()+mu_p.transpose())*SigmaP_muprime*(mu_pp+mu_p);

    // Free-energy as a sum of squared values (i.e. sum the SPE)
    F.data = SPEq + SPEdq + SPEmu_p + SPEmu_pp;

    // Free-energy minimization using gradient descent and beliefs update
    mu_dot = mu_p - k_mu*(-SigmaP_yq0*(jointPos-mu)+SigmaP_mu*(mu_p+mu-mu_d));
    mu_dot_p = mu_pp - k_mu*(-SigmaP_yq1*(jointVel-mu_p)+SigmaP_mu*(mu_p+mu-mu_d)+SigmaP_muprime*(mu_pp+mu_p));
    mu_dot_pp = - k_mu*(SigmaP_muprime*(mu_pp+mu_p));

    // Belifs update
    mu = mu + h*mu_dot;             // Belief about the position
    mu_p = mu_p + h*mu_dot_p;       // Belief about motion of mu
    mu_pp = mu_pp + h*mu_dot_pp;    // Belief about motion of mu'
    //AIC::noise();
    // Publish beliefs as Float64MultiArray
    for (int i=0;i<7;i++){
       AIC_mu.data[i] = mu(i);
       AIC_mu_p.data[i] = mu_p(i);
       AIC_mu_pp.data[i] = mu_pp(i);
    }
    // Define SPE message
    SPE.data[0] = SPEq;
    SPE.data[1] = SPEdq;

    // Calculate and send control actions
    AIC::computeActions();

    // Publish free-energy
    IFE_pub.publish(F);

    // Sensory prediction error publisher
    SPE_pub.publish(SPE);

    // Publish beliefs
    beliefs_mu_pub.publish(AIC_mu);
    beliefs_mu_p_pub.publish(AIC_mu_p);
    beliefs_mu_pp_pub.publish(AIC_mu_pp);

    //err_end_eff  = abs(end_eff_d_x[d] - end_eff_x)  + abs(end_eff_d_y[d] - end_eff_y)  + abs(end_eff_d_z[d] - end_eff_z);

    //err_end_eff = end_eff - end_eff_d(d);
    //std::ofstream fileWrite("err_end_eff_noise.txt", std::ios::app);
    //fileWrite << err_end_eff;
    //fileWrite << "\n";
    //fileWrite.close(); 
  }

  void   AIC::computeActions(){
    // Compute control actions through gradient descent of F
    u = u-h*k_a*(SigmaP_yq1*(jointVel-mu_p)+SigmaP_yq0*(jointPos-mu));

    // Set the toques from u and publish
    tau1.data = u(0); tau2.data = u(1); tau3.data = u(2); tau4.data = u(3);
    tau5.data = u(4); tau6.data = u(5); tau7.data = u(6);
    // Publishing
    tauPub1.publish(tau1); tauPub2.publish(tau2); tauPub3.publish(tau3);
    tauPub4.publish(tau4); tauPub5.publish(tau5); tauPub6.publish(tau6);
    tauPub7.publish(tau7);
  }

  int AIC::dataReady(){
    // Method to control if the joint states have been received already,
    // used in the main function
    if(dataReceived==1)
      return 1;
    else
      return 0;
  }

  void AIC::setGoal(std::vector<double> desiredPos){
    for(int i=0; i<desiredPos.size(); i++){
      mu_d(i) = desiredPos[i];
    }
  }

  std_msgs::Float64MultiArray  AIC::getSPE(){
    return(SPE);
  }
