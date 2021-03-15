#!/usr/bin/env python2.7
import rospy
from std_msgs import Float64MultiArray, MultiArrayDimension, Float64
from geometry_msgs import PoseStamped 
from sensor_msgs import JointState, Image 
import numpy as np
import scipy.io
import math
import sys
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <stdlib.h>



class AIC_agent:

    # Number of joints used
    N_JOINTS = 7

    # Image parameters
    IMG_WIDTH = 256
    IMG_HEIGHT = 256

  def __init__(self, Robot):
        
        self.jointStatesCallback(self,msg);
        self.initVariables(self);
        self.minimiseF_Prop(self);
        self.dF_callback(self, dF_dmu_vis);
        self.dF_callback_attr(self, dF_dmu_attr_p);
        self.minimiseF_PixelAI(self);
        self.computeActions(self);
        self.dataReady(self);
        self.setGoal(self, desiredPos);
        self.getSPE(self);


        self.SigmaP_yq0 = np.zeros((self.N_JOINTS,self.N_JOINTS)
        self.SigmaP_yq1 = np.zeros((self.N_JOINTS,self.N_JOINTS)
        self.SigmaP_mu = np.zeros((self.N_JOINTS,self.N_JOINTS)
        self.SigmaP_muprime = np.zeros((self.N_JOINTS,self.N_JOINTS)
        self.mu = np.zeros((1,self.N_JOINTS)
        self.mu_p = np.zeros((1,self.N_JOINTS)
        self.mu_pp = np.zeros((1,self.N_JOINTS)
        self.mu_dot = np.zeros((1,self.N_JOINTS)
        self.mu_dot_p = np.zeros((1,self.N_JOINTS)
        self.mu_dot_pp = np.zeros((1,self.N_JOINTS)
        self.jointPos = np.zeros((1,self.N_JOINTS)
        self.jointVel = np.zeros((1,self.N_JOINTS)
        self.mu_d = np.zeros((1,self.N_JOINTS)
        self.a = np.zeros((1,self.N_JOINTS)
        self.a_dot = np.zeros((1,self.N_JOINTS)
        self.s_p = np.zeros((1,self.N_JOINTS)


        self.AIC_mu = FloatMultiArray()
        self.AIC_mu_p = FloatMultiArray()
        self.AIC_mu_pp = FloatMultiArray()
        self.SPE = FloatMultiArray()
        self.dF_dg_mu_vis = FloatMultiArray()
        self.dF_dmu_attr = FloatMultiArray()


        self.datareceived = 0  

        self.var_mu = 5.0
        self.var_muprime = 10.0  
        self.var_q = 1
        self.var_qdot = 1

        # Learning rates for the gradient descent (found that a ratio of 60 works good)
        self.k_mu = 11.67
        self.k_a = 700
        #Attractor gain
        self.beta = 1

        for i in range(N_JOINTS):
            SigmaP_yq0[i, i] = 1/var_q
            SigmaP_yq1[i, i] = 1/var_qdot
            SigmaP_mu[i, i] = 1/var_mu
            SigmaP_muprime[i, i] = 1/var_muprime

        self.h = 0.001
        
        #Publisher
        self.tauPub1 = rospy.Publisher("/robot1/panda_joint1_controller/command", Float64, queue_size = 20)
        self.tauPub2 = rospy.Publisher("/robot1/panda_joint2_controller/command", Float64, queue_size = 20)
        self.tauPub3 = rospy.Publisher("/robot1/panda_joint3_controller/command", Float64, queue_size = 20)
        self.tauPub4 = rospy.Publisher("/robot1/panda_joint4_controller/command", Float64, queue_size = 20)
        self.tauPub5 = rospy.Publisher("/robot1/panda_joint5_controller/command", Float64, queue_size = 20)
        self.tauPub6 = rospy.Publisher("/robot1/panda_joint6_controller/command", Float64, queue_size = 20)
        self.tauPub7 = rospy.Publisher("/robot1/panda_joint7_controller/command", Float64, queue_size = 20)

        self.IFE_pub = rospy.Publisher("panda_free_energy", Float64, queue_size = 10)
        self.SPE_pub = rospy.Publisher("panda_SPE", Float64, queue_size = 10)
        

        self.beliefs_mu_pub = rospy.Publisher("beliefs_mu",Float64,queue_size = 20)
        self.beliefs_mu_p_pub = rospy.Publisher("beliefs_mu_p",Float64,queue_size = 20)
        self.beliefs_mu_pp_pub = rospy.Publisher("beliefs_mu_pp",Float64,queue_size = 20)

        #Subscribers
        sensorSub = rospy.Subscriber("/robot1/joint_states", JointState, self.jointStatesCallback);
        #sub = rospy.Subscriber("/perception/pub_dF_vis",, dF_callback);
        #sub_ = rospy.Subscriber("perception/pub_dF_attr",1, AIC::dF_callback_attr);



  def jointStatesCallback(self, msg):
      
        for i in range(N_JOINTS):
        
            self.jointPos[i] = msg.position[i];
            self.jointVel[i] = msg.velocity[i];

        if (self.dataReceived == 0):

           self.dataReceived = 1
           self.mu = self.jointpos
           self.mu_p = self.jointVel;


  def minimiseF_prop(self):
    
        # Compute single sensory prediction errors
        self.SPEq = (np.transpose(self.jointPos)-np.transpose(self.mu))*self.SigmaP_yq0*(self.jointPos-self.mu);
        self.SPEdq = (np.transpose(self.jointVel)-np.transpose(self.mu_p))*self.SigmaP_yq1*(self.jointVel-self.mu_p);
        self.SPEmu_p = (np.transpose(self.mu_p)+np.transpose(self.mu)-np.transpose(self.mu_d))*self.SigmaP_mu*(self.mu_p+self.mu-self.mu_d);
        self.SPEmu_pp = (np.transpose(self.mu_pp)+np.transpose(self.mu_p))*self.SigmaP_muprime*(self.mu_pp+self.mu_p);
        
        # Free-energy as a sum of squared values (i.e. sum the SPE)
        self.F_Prop.data = self.SPEq + self.SPEdq + self.SPEmu_p + self.SPEmu_pp;


       #Free-energy minimization using gradient descent and beliefs update
       self.mu_dot = self.mu_p - self.k_mu*(-self.SigmaP_yq0*(self.jointPos-self.mu)+self.SigmaP_mu*(self.mu_p+self.mu-self.mu_d));
   
       self.mu_dot_p = self.mu_pp - self.k_mu*(-self.SigmaP_yq1*(self.jointVel-self.mu_p)+self.SigmaP_mu*(self.mu_p+self.mu-self.mu_d)+self.SigmaP_muprime*(self.mu_pp+self.mu_p));
       self.mu_dot_pp = - self.k_mu*(self.SigmaP_muprime*(self.mu_pp+self.mu_p));

       #Belifs update
       self.mu = self.mu + self.h*self.mu_dot;             # Belief about the position
       self.mu_p = self.mu_p + self.h*self.mu_dot_p;       # Belief about motion of mu
       self.mu_pp = self.mu_pp + self.h*self.mu_dot_pp;    # Belief about motion of mu'

       for i in range(N_JOINTS):
            self.AIC_mu.data[i] = self.mu[i];
            self.AIC_mu_p.data[i] = self.mu_p[i];
            self.AIC_mu_pp.data[i] = self.mu_pp[i];

       self.SPE.data[0] = self.SPEq;
       self.SPE.data[1] = self.SPEdq;


       self.computeActions()


       self.IFE_pub.publish(self.F_Prop)

       # Sensory prediction error publisher
       self.SPE_pub.publish(self.SPE)

       # Publish beliefs
       self.beliefs_mu_pub.publish(self.AIC_mu)
       self.beliefs_mu_p_pub.publish(self.AIC_mu_p)
       self.beliefs_mu_pp_pub.publish(self.AIC_mu_pp)


   def computeActions(self):

      self.a = self.a-self.h*self.k_a*(self.SigmaP_yq1*(self.jointVel-self.mu_p)+self.SigmaP_yq0*(self.jointPos-self.mu));

      # Set the toques from u and publish
      self.tau1.data = self.a(0) 
      self.tau2.data = self.a(1) 
      self.tau3.data = self.a(2) 
      self.tau4.data = self.a(3)
      self.tau5.data = self.a(4)
      self.tau6.data = self.a(5)
      self.tau7.data = self.a(6)
      # Publishing
      self.tauPub1.publish(self.tau1)
      self.tauPub2.publish(self.tau2)
      self.tauPub3.publish(self.tau3)
      self.tauPub4.publish(self.tau4)
      self.tauPub5.publish(self.tau5)
      self.tauPub6.publish(self.tau6)
      self.tauPub7.publish(self.tau7)


   def dataReady(self):

       if (dataReceived ==1):
            return 1
       else:
            return 0


   def setGoal(self, desiredPos):

       for i in range(N_JOINTS):
           self.mu_d[i] = desiredPos[i]

   

       

      
     



           

       
      

        
     



 
 
 
 
        

        

