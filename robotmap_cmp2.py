'''
move shadow to specified pose which from my cmp2 network
'''
from __future__ import print_function
import moveit_commander
import rospy
import math
def clip(x, maxv=None, minv=None):
    if maxv is not None and x > maxv:
        x = maxv
    if minv is not None and x < minv:
        x = minv
    return x

def main():
    # moveit_commander.roscpp_initialize()
    rospy.init_node('shadow_human_dataset')
    mgi_tf = moveit_commander.MoveGroupCommander("right_hand")
    feature =[0.000000,0.000000,0.349066,1.232986,0.037221,0.787171,0.000000,-0.128351,0.000000,0.000000,0.000000,-0.286982,0.493627,0.001749,0.000911,0.238188,0.662593,0.566765,0.707058,0.096420,0.813588,0.046945,0.147854,0.028486]
    # print(feature)
    # feature[0] = clip(feature[0], 1.57, 0)
    # feature[1] = clip(feature[1], 1.57, 0)
    # feature[2] = clip(feature[2], 1.57, 0)
    # feature[4] = clip(feature[4], 1.57, 0)
    # feature[5] = clip(feature[5], 1.57, 0)
    # feature[6] = clip(feature[6], 1.57, 0)
    # feature[9] = clip(feature[9], 1.57, 0)
    # feature[10] = clip(feature[10], 1.57, 0)
    # feature[11] = clip(feature[11], 1.57, 0)
    # feature[13] = clip(feature[13], 1.57, 0)
    # feature[14] = clip(feature[14], 1.57, 0)
    # feature[15] = clip(feature[15], 1.57, 0)
    #
    # feature[3] = clip(feature[3], 0.349, -0.349)
    # feature[7] = clip(feature[7], 0.349, -0.349)
    # feature[12] = clip(feature[12], 0.349, -0.349)
    # feature[16] = clip(feature[16], 0.349, -0.349)
    # feature[8] = clip(feature[8], 0.785, 0)
    # feature[18] = clip(feature[18], 0.524, -0.524)
    # feature[19] = clip(feature[19], 0.209, -0.209)
    # feature[20] = clip(feature[20],  1.222, 0)
    # feature[21] = clip(feature[21], 1.047, -1.047)
    # print(feature)
    # moveit_feature = []
    # moveit_feature.append(0)
    # moveit_feature.append(0)
    # moveit_feature.append(feature[3])
    # moveit_feature.append(feature[2])
    # moveit_feature.append(feature[1])
    # moveit_feature.append(feature[0])
    # moveit_feature.append(feature[8])
    # moveit_feature.append(feature[7])
    # moveit_feature.append(feature[6])
    # moveit_feature.append(feature[5])
    # moveit_feature.append(feature[4])
    # moveit_feature.append(feature[12])
    # moveit_feature.append(feature[11])
    # moveit_feature.append(feature[10])
    # moveit_feature.append(feature[9])
    # moveit_feature.append(feature[16])
    # moveit_feature.append(feature[15])
    # moveit_feature.append(feature[14])
    # moveit_feature.append(feature[13])
    #
    # moveit_feature.append(feature[21])
    # moveit_feature.append(feature[20])
    # moveit_feature.append(feature[19])
    # moveit_feature.append(feature[18])
    # moveit_feature.append(feature[17])
    # print(moveit_feature)

    mgi_tf.set_joint_value_target(feature)
    mgi_tf.go()
    rospy.sleep(2)

if __name__ == '__main__':
    main()
