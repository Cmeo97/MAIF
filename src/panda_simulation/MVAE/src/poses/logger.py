#! /usr/bin/python2.7
# rospy for the subscriber
import rospy
import os
import numpy as np
# ROS Image message
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import sys
import time

# Instantiate CvBridge
bridge = CvBridge()


class logger:

    def __init__(self):
        self.count_im = 0
        self.count_joints = 0
        self.Im_flag = False
        self.mu_flag = True
        self.Tau = np.zeros((1,7), float)
        self.X = np.zeros((1,7), float)
        self.X_dot = np.zeros((1,7), float)
        self.joint_counter = 0
        self.sim_counter = 0

    def joint_states_callback(self,msg):
           

           if (self.joint_counter == 46):
             
              x = msg.position
              x_dot = msg.velocity 
              tau = msg.effort
             
              self.X[0,:] = x[:-1]
              
              f_x = open( 'JointStates_q.txt','a')
              f_x.write('[')
              f_x.close
              for i in range(7):
                 f_x = open( 'JointStates_q.txt','a')
                 f_x.write(str(np.round(self.X[0,i], 10)) + ' ' )
                 f_x.close
              f_x = open( 'JointStates_q.txt','a')
              f_x.write(']')
              f_x.close
              
              self.Tau[0,:] = tau[:-1]
              
              f_tau = open( 'JointStates_tau.txt','a')
              f_tau.write('[')
              f_tau.close
              for i in range(7):
                 f_tau = open( 'JointStates_tau.txt','a')
                 f_tau.write(str(np.round(self.Tau[0,i], 10)) + ' ' )
                 f_tau.close
              f_tau = open( 'JointStates_tau.txt','a')
              f_tau.write(']')
              f_tau.close
             
              self.X_dot[0,:] = x_dot[:-1]
               
              f_x_dot = open( 'JointStates_q_dot.txt','a')
              f_x_dot.write('[')
              f_x_dot.close
              for i in range(7):
                 f_x_dot = open( 'JointStates_q_dot.txt','a')
                 f_x_dot.write(str(np.round(self.X_dot[0,i], 10)) + ' ' )
                 f_x_dot.close
              f_x_dot = open( 'JointStates_q_dot.txt','a')
              f_x_dot.write(']')
              f_x_dot.close

              
              print('joints:' + str(self.count_joints))
              self.sim_counter += 1
              self.count_joints +=1
              self.joint_counter = 0
              self.Im_flag = True
           self.joint_counter +=1
          


    def open(self):
         f_x = open( 'JointStates_q.txt','a')
         f_x.write('[')
         f_x.close
         f_x_dot = open( 'JointStates_q_dot.txt','a')
         f_x_dot.write('[')
         f_x_dot.close
         f_tau = open( 'JointStates_tau.txt','a')
         f_tau.write('[')
         f_tau.close

    def close(self):
         f_x = open( 'JointStates_q.txt','a')
         f_x.write(']')
         f_x.close
         f_x_dot = open( 'JointStates_q_dot.txt','a')
         f_x_dot.write(']')
         f_x_dot.close
         f_tau = open( 'JointStates_tau.txt','a')
         f_tau.write(']')
         f_tau.close
         

    def image_callback(self,msg):
     
        if (self.Im_flag):
            
            try:
                
                image = bridge.imgmsg_to_cv2(msg, "mono8")
            except CvBridgeError:
                print("error")
            else:
              
                filename = 'camera_image_' + str(self.count_joints - 1) + '.jpeg' 
                cv2.imwrite(filename, image)
                #rospy.sleep(1)
                
                print('Im:' + str(self.count_joints -1))
                self.Im_flag = False
        if (self.sim_counter > 500000):
                  self.close()
                  rospy.signal_shutdown('Finished!')
                  sys.exit()
                   
 
def main():
      np.set_printoptions(precision=4)
      rospy.init_node('logger', disable_signals = True)
      np.set_printoptions(threshold=sys.maxsize)
      model = logger()
      # Define your image topic
      image_topic = "/camera/color/image_raw"
      joints_states_topic = "/robot1/joint_states"
      rospy.Subscriber(joints_states_topic, JointState, model.joint_states_callback) 
      rospy.Subscriber(image_topic, Image, model.image_callback)  
      rospy.spin()    
         
   

if __name__ == '__main__':
      main()
