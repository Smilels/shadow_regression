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
    feature = [0.1763,  1.4695,  0.4058, -0.0090,  0.4160,  1.5671,  1.2508,
               -0.1084,  0.0697,  0.2343,  1.5655,  1.3627,  0.0585,  0.2738,
               1.5672,  1.2333, -0.0249,  0.8397,  0.4157,  0.0598,  1.1375,
                0.1357]
    print(feature)
    feature[0] = clip(feature[0], 1.57, 0)
    feature[1] = clip(feature[1], 1.57, 0)
    feature[2] = clip(feature[2], 1.57, 0)
    feature[4] = clip(feature[4], 1.57, 0)
    feature[5] = clip(feature[5], 1.57, 0)
    feature[6] = clip(feature[6], 1.57, 0)
    feature[9] = clip(feature[9], 1.57, 0)
    feature[10] = clip(feature[10], 1.57, 0)
    feature[11] = clip(feature[11], 1.57, 0)
    feature[13] = clip(feature[13], 1.57, 0)
    feature[14] = clip(feature[14], 1.57, 0)
    feature[15] = clip(feature[15], 1.57, 0)

    feature[3] = clip(feature[3], 0.349, -0.349)
    feature[7] = clip(feature[7], 0.349, -0.349)
    feature[12] = clip(feature[12], 0.349, -0.349)
    feature[16] = clip(feature[16], 0.349, -0.349)
    feature[8] = clip(feature[8], 0.785, 0)
    feature[18] = clip(feature[18], 0.524, -0.524)
    feature[19] = clip(feature[19], 0.209, -0.209)
    feature[20] = clip(feature[20],  1.222, 0)
    feature[21] = clip(feature[21], 1.047, -1.047)
    print(feature)
    moveit_feature = []
    moveit_feature.append(0)
    moveit_feature.append(0)
    moveit_feature.append(feature[3])
    moveit_feature.append(feature[2])
    moveit_feature.append(feature[1])
    moveit_feature.append(feature[0])
    moveit_feature.append(feature[8])
    moveit_feature.append(feature[7])
    moveit_feature.append(feature[6])
    moveit_feature.append(feature[5])
    moveit_feature.append(feature[4])
    moveit_feature.append(feature[12])
    moveit_feature.append(feature[11])
    moveit_feature.append(feature[10])
    moveit_feature.append(feature[9])
    moveit_feature.append(feature[16])
    moveit_feature.append(feature[15])
    moveit_feature.append(feature[14])
    moveit_feature.append(feature[13])

    moveit_feature.append(feature[21])
    moveit_feature.append(feature[20])
    moveit_feature.append(feature[19])
    moveit_feature.append(feature[18])
    moveit_feature.append(feature[17])
    print(moveit_feature)

    mgi_tf.set_joint_value_target(moveit_feature)
    mgi_tf.go()
    rospy.sleep(2)

if __name__ == '__main__':
    main()
