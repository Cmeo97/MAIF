#!/usr/bin/env python2.7
import rospy
import sys
import os
import numpy as np
from std_msgs.msg import Float64MultiArray,MultiArrayDimension, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

bridge = CvBridge()


def image_callback(msg):
    
  
    try:
        # Convert your ROS Image message to OpenCV2
        cv_image = bridge.imgmsg_to_cv2(msg, "mono8")
    except CvBridgeError:
        print("error")
    else:
       
        pub_camera =rospy.Publisher("perception/Image", Image, queue_size=10)
        pub_camera.publish(bridge.cv2_to_imgmsg(cv_image, "mono8"))
        
        
        
    

def main():
    rospy.init_node('camera')

    image_topic = "/camera/color/image_raw"
    rospy.Subscriber(image_topic, Image, image_callback)
    
    while not rospy.is_shutdown():
        continue

if __name__ == "__main__":
   main()       
   
