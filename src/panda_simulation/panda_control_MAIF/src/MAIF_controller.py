#!/usr/bin/env python2.7

import argparse
from AIC_full import *
import gc
import GPUtil
import time
import subprocess
from multiprocessing import Process

robot = 1
def main():

    rospy.init_node('AIC_controller')
    AIC = AIC_agent()
    #Desired Poses
    desiredPos = np.zeros(7, float)
    desiredPos1 = np.zeros(7, float)
    desiredPos1[0] = 1
    desiredPos1[1] = 0.5
    desiredPos1[2] = 0.0
    desiredPos1[3] = -2
    desiredPos1[4] = 0.0
    desiredPos1[5] = 2.5
    desiredPos1[6] = 0.0

    desiredPos2 = np.zeros(7, float)
    desiredPos2[0] = 0.0
    desiredPos2[1] = 0.2
    desiredPos2[2] = 0.0
    desiredPos2[3] = -1.0
    desiredPos2[4] = 0.0
    desiredPos2[5] = 1.2
    desiredPos2[6] = 0.0

    desiredPos3 = np.zeros(7, float)
    desiredPos3[0] = -1
    desiredPos3[1] = 0.5
    desiredPos3[2] = 0.0
    desiredPos3[3] = -1.2
    desiredPos3[4] = 0.0
    desiredPos3[5] = 1.6
    desiredPos3[6] = 0

    im_path = 'starting_pose.jpeg'
    AIC.setGoal(desiredPos, im_path)

    #AIC.end_effector_selection(0)
    rate = rospy.Rate(120)
    count = 1
    while not rospy.is_shutdown():

        if count == 1:
            AIC.get_latent_action()
            for i in range(300):
                p1 = Process(target=AIC.minimiseF())

        if count > 0:
            p2 = Process(target=AIC.minimiseF())

        count += 1
        if count == 400:
            im_path = 'goal_pose_1.jpeg'
            AIC.setGoal(desiredPos1, im_path)
            AIC.end_effector_selection(1)

        if count == 800:
            im_path = 'goal_pose_2.jpeg'
            AIC.setGoal(desiredPos2, im_path)
            AIC.end_effector_selection(2)

        if count == 1200:
            im_path = 'goal_pose_3.jpeg'
            AIC.setGoal(desiredPos3, im_path)
            AIC.end_effector_selection(3)

        if count == 1600:
            im_path = 'goal_pose_2.jpeg'
            AIC.setGoal(desiredPos2, im_path)
            AIC.end_effector_selection(2)

        if count == 2000:
            im_path = 'goal_pose_1.jpeg'
            AIC.setGoal(desiredPos1, im_path)
            AIC.end_effector_selection(1)

        if count == 2400:
            im_path = 'goal_pose_2.jpeg'
            AIC.setGoal(desiredPos + 0.1, im_path)
            AIC.end_effector_selection(2)

        if count > 2999:
            sys.exit()

    rate.sleep()


if __name__ == "__main__":
    main()
