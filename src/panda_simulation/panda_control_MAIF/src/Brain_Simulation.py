#!/usr/bin/env python2.7
from AIC_full import *
import sys

robot = 1


def main():

    rospy.init_node('AIC_perception')

    AIC = AIC_agent()
    rate = rospy.Rate(1500)
    
    desiredPos = np.zeros(7, float)
    desiredPos[0] = 0
    desiredPos[1] = 0
    desiredPos[2] = 0
    desiredPos[3] = 0
    desiredPos[4] = 0
    desiredPos[5] = 0
    desiredPos[6] = 0

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
    count = 1

    im_path = 'starting_pose.jpeg'
    AIC.setGoal(desiredPos, im_path)
    AIC.get_latent_perception()
    while not rospy.is_shutdown():
        AIC.perception(count)
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
            AIC.setGoal(desiredPos, im_path)
            AIC.end_effector_selection(2)
        count += 1
        if count > 2999:
            AIC.plot_simulation_perception()
            sys.exit()

    rate.sleep()


if __name__ == "__main__":

   main()  
