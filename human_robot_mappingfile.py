import numpy as np
import math
import csv
from mayavi import mlab
import cv2

class Map_Loader(object):
    def __init__(self, base_path= "./data/"):
        # load data
        self.base_path = base_path
        DataFile = open(base_path + "test.txt", "r")

        lines = DataFile.read().splitlines()
        self.framelist = [ln.split(' ')[0].replace("\t", "") for ln in lines]
        label_source = [ln.split('\t')[1:] for ln in lines]
        self.label = []
        for ln in label_source:
            ll = ln[0:63]
            self.label.append([float(l.replace(" ", "")) for l in ll])

        self.label = np.array(self.label)
        DataFile.close()
        self.shadow = self.shadow_model()

    def map(self, start):
        rh_palm, rh_pip_mcp, rh_dip_pip, rh_tip_dip = self.shadow_model()
        # the joint order is
        # [Wrist, TMCP, IMCP, MMCP, RMCP, PMCP, TPIP,
        # TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP]
        # for index in range(start, start + batch_size):
        keypoints = self.label[start]
        frame = self.framelist[start]
        keypoints = keypoints.reshape(21, 3)

        tf_palm = keypoints[1] - keypoints[0]
        ff_palm = keypoints[2] - keypoints[0]
        mf_palm = keypoints[3] - keypoints[0]
        rf_palm = keypoints[4] - keypoints[0]
        lf_palm = keypoints[5] - keypoints[0]
        palm = np.array([tf_palm, ff_palm, mf_palm, rf_palm, lf_palm])

        # local wrist frame build
        wrist_z = np.mean(palm[2:4], axis=0)
        wrist_z /= np.linalg.norm(wrist_z)
        wrist_y = np.cross(rf_palm, mf_palm)
        wrist_y /= np.linalg.norm(wrist_y)
        wrist_x = np.cross(wrist_y, wrist_z)
        if np.linalg.norm(wrist_x) != 0:
            wrist_x /= np.linalg.norm(wrist_x)

        local_frame = np.vstack([wrist_x,wrist_y,wrist_z])
        local_points = np.dot((keypoints - keypoints[0]), local_frame.T)

        local_palm = np.array([local_points[1], local_points[2], local_points[3], local_points[4], local_points[5]])
        hh_palm = np.linalg.norm(local_palm, axis=1)

        tf_pip_mcp = local_points[6] - local_points[1]
        tf_dip_pip = local_points[7] - local_points[6]
        tf_tip_dip = local_points[8] - local_points[7]

        ff_pip_mcp = local_points[9] - local_points[2]
        ff_dip_pip = local_points[10] - local_points[9]
        ff_tip_dip = local_points[11] - local_points[10]

        mf_pip_mcp = local_points[12] - local_points[3]
        mf_dip_pip = local_points[13] - local_points[12]
        mf_tip_dip = local_points[14] - local_points[13]

        rf_pip_mcp = local_points[15] - local_points[4]
        rf_dip_pip = local_points[16] - local_points[15]
        rf_tip_dip = local_points[17] - local_points[16]

        lf_pip_mcp = local_points[18] - local_points[5]
        lf_dip_pip = local_points[19] - local_points[18]
        lf_tip_dip = local_points[20] - local_points[19]

        pip_mcp = np.array([tf_pip_mcp, ff_pip_mcp, mf_pip_mcp, rf_pip_mcp, lf_pip_mcp])
        dip_pip = np.array([tf_dip_pip, ff_dip_pip, mf_dip_pip, rf_dip_pip, lf_dip_pip])
        tip_dip = np.array([tf_tip_dip, ff_tip_dip, mf_tip_dip, rf_tip_dip, lf_tip_dip])
        hh_pip_mcp = np.linalg.norm(pip_mcp, axis=1)
        hh_dip_pip = np.linalg.norm(dip_pip, axis=1)
        hh_tip_dip = np.linalg.norm(tip_dip, axis=1)

        # hh_len = hh_palm + hh_pip_mcp + hh_dip_pip + hh_tip_dip

        coe_palm = rh_palm / hh_palm
        rh_wrist_mcp_key = np.multiply(coe_palm.reshape(-1, 1), local_palm)
        rh_wrist_mcp_key[0][2] = rh_wrist_mcp_key[0][2] + 29

        coe_pip_mcp = rh_pip_mcp / hh_pip_mcp
        rh_pip_mcp_key = np.multiply(coe_pip_mcp.reshape(-1, 1), pip_mcp) + rh_wrist_mcp_key
        coe_dip_pip = rh_dip_pip / hh_dip_pip
        rh_dip_pip_key = np.multiply(coe_dip_pip.reshape(-1, 1), dip_pip) + rh_pip_mcp_key
        coe_tip_dip = rh_tip_dip / hh_tip_dip
        rh_tip_dip_key = np.multiply(coe_tip_dip.reshape(-1, 1), tip_dip) + rh_dip_pip_key

        shadow_points = np.vstack([np.array([0, 0, 0]), rh_wrist_mcp_key,
                                    rh_pip_mcp_key[0], rh_dip_pip_key[0], rh_tip_dip_key[0],
                                    rh_pip_mcp_key[1], rh_dip_pip_key[1], rh_tip_dip_key[1],
                                    rh_pip_mcp_key[2], rh_dip_pip_key[2], rh_tip_dip_key[2],
                                    rh_pip_mcp_key[3], rh_dip_pip_key[3], rh_tip_dip_key[3],
                                    rh_pip_mcp_key[4], rh_dip_pip_key[4], rh_tip_dip_key[4]])

        tip_keys = rh_tip_dip_key/1000
        pip_keys = rh_pip_mcp_key/1000
        dip_keys = rh_dip_pip_key/1000
        mcp_keys = rh_wrist_mcp_key/1000
        # from IPython import embed;embed()
        return tip_keys, pip_keys, dip_keys, mcp_keys, frame, local_points, shadow_points

    def shadow_model(self):
        # shadow hand length
        rh_tf_palm = 34
        rh_ff_palm = math.sqrt(math.pow(95, 2) + math.pow(33, 2))
        rh_mf_palm = math.sqrt(math.pow(99, 2) + math.pow(11, 2))
        rh_rf_palm = math.sqrt(math.pow(95, 2) + math.pow(11, 2))
        rh_lf_palm = math.sqrt(math.pow(86.6, 2) + math.pow(33, 2))
        rh_palm = np.array([rh_tf_palm, rh_ff_palm, rh_mf_palm, rh_rf_palm, rh_lf_palm])

        rh_tf_pip_mcp = 38
        rh_tf_dip_pip = 32
        rh_tf_tip_dip = 27.5

        rh_ff_pip_mcp = 45
        rh_ff_dip_pip = 25
        rh_ff_tip_dip = 26

        rh_mf_pip_mcp = 45
        rh_mf_dip_pip = 25
        rh_mf_tip_dip = 26

        rh_rf_pip_mcp = 45
        rh_rf_dip_pip = 25
        rh_rf_tip_dip = 26

        rh_lf_pip_mcp = 45
        rh_lf_dip_pip = 25
        rh_lf_tip_dip = 26

        rh_pip_mcp = np.array([rh_tf_pip_mcp, rh_ff_pip_mcp, rh_mf_pip_mcp, rh_rf_pip_mcp, rh_lf_pip_mcp])
        rh_dip_pip = np.array([rh_tf_dip_pip, rh_ff_dip_pip, rh_mf_dip_pip, rh_rf_dip_pip, rh_lf_dip_pip])
        rh_tip_dip = np.array([rh_tf_tip_dip, rh_ff_tip_dip, rh_mf_tip_dip, rh_rf_tip_dip, rh_lf_tip_dip])

        # rh_len = rh_palm + rh_pip_mcp + rh_dip_pip + rh_tip_dip
        return [rh_palm, rh_pip_mcp, rh_dip_pip ,rh_tip_dip]


def show_line(un1, un2, color='g', scale_factor=1):
    if color == 'b':
        color_f = (0.8, 0, 0.9)
    elif color == 'r':
        color_f = (0.3, 0.2,0.7)
    elif color == 'p':
        color_f = (0.1, 1, 0.8)
    elif color == 'y':
        color_f = (0.5, 1, 1)
    elif color == 'g':
        color_f = (1, 1, 0)
    elif isinstance(color, tuple):
        color_f = color
    else:
        color_f = (1, 1, 1)
    mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=color_f, tube_radius=scale_factor)


def show_points(point, color='b', scale_factor=5):
    if color == 'b':
        color_f = (0, 0, 1)
    elif color == 'r':
        color_f = (1, 0, 0)
    elif color == 'g':
        color_f = (0, 1, 0)
    else:
        color_f = (1, 1, 1)
    if point.size == 3:  # vis for only one point
        mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
    else:  # vis for multiple points
        mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)


def show_hand(points,type='human'):
    show_points(points)
    if type=="human":
        show_line(points[0], points[1], color='r')
        show_line(points[1], points[6], color='r')
        show_line(points[7], points[6], color='r')
        show_line(points[7], points[8], color='r')

        show_line(points[0], points[2], color='y')
        show_line(points[9], points[2], color='y')
        show_line(points[9], points[10], color='y')
        show_line(points[11], points[10], color='y')

        show_line(points[0], points[3], color='g')
        show_line(points[12], points[3], color='g')
        show_line(points[12], points[13], color='g')
        show_line(points[14], points[13], color='g')

        show_line(points[0], points[4], color='b')
        show_line(points[15], points[4], color='b')
        show_line(points[15], points[16], color='b')
        show_line(points[17], points[16], color='b')

        show_line(points[0], points[5], color='p')
        show_line(points[18], points[5], color='p')
        show_line(points[18], points[19], color='p')
        show_line(points[20], points[19], color='p')
    else:
        show_line(points[0], points[1], color='r')
        show_line(points[1], points[6], color='r')
        show_line(points[7], points[6], color='r')
        show_line(points[7], points[8], color='r')

        show_line(points[0], points[2], color='y')
        show_line(points[9], points[2], color='y')
        show_line(points[9], points[10], color='y')
        show_line(points[11], points[10], color='y')

        show_line(points[0], points[3], color='g')
        show_line(points[12], points[3], color='g')
        show_line(points[12], points[13], color='g')
        show_line(points[14], points[13], color='g')

        show_line(points[0], points[4], color='b')
        show_line(points[15], points[4], color='b')
        show_line(points[15], points[16], color='b')
        show_line(points[17], points[16], color='b')

        show_line(points[0], points[5], color='p')
        show_line(points[18], points[5], color='p')
        show_line(points[18], points[19], color='p')
        show_line(points[20], points[19], color='p')
    tf_palm = points[1] - points[0]
    ff_palm = points[2] - points[0]
    mf_palm = points[3] - points[0]
    rf_palm = points[4] - points[0]
    lf_palm = points[5] - points[0]
    # palm = np.array([tf_palm, ff_palm, mf_palm, rf_palm, lf_palm])
    palm = np.array([ff_palm, mf_palm, rf_palm, lf_palm])

    # local wrist frame build
    wrist_z = np.mean(palm, axis=0)
    wrist_z /= np.linalg.norm(wrist_z)
    wrist_y = np.cross(lf_palm, rf_palm)
    wrist_y /= np.linalg.norm(wrist_y)
    wrist_x = np.cross(wrist_y, wrist_z)
    if np.linalg.norm(wrist_x) != 0:
        wrist_x /= np.linalg.norm(wrist_x)

    mlab.quiver3d(points[0][0], points[0][1], points[0][2], wrist_x[0], wrist_x[1], wrist_x[2],
                  scale_factor=50, line_width=0.5, color=(1, 0, 0), mode='arrow')
    mlab.quiver3d(points[0][0], points[0][1], points[0][2], wrist_y[0], wrist_y[1], wrist_y[2],
                  scale_factor=50, line_width=0.5, color=(0, 1, 0), mode='arrow')
    mlab.quiver3d(points[0][0], points[0][1], points[0][2], wrist_z[0], wrist_z[1], wrist_z[2],
                  scale_factor=50, line_width=0.5, color=(0, 0, 1), mode='arrow')
    # from IPython import embed;embed()
    # mlab.show()

if __name__ == '__main__':
    batch_size = 1
    base_path= "./data/trainning/"
    map_loader = Map_Loader(base_path)
    csvSum = open(base_path + "human_robot_mapdata_pip2.csv", "w")
    writer = csv.writer(csvSum)
    print(len(map_loader.framelist))
    for i in range(0, len(map_loader.framelist)):
        tip_keys, pip_keys, dip_keys, mcp_keys, frame, local_points, shadow_points = map_loader.map(i)
        # save key
        result = [frame, tip_keys[0][0], tip_keys[0][1], tip_keys[0][2], tip_keys[1][0], tip_keys[1][1], tip_keys[1][2],
        tip_keys[2][0], tip_keys[2][1], tip_keys[2][2], tip_keys[3][0], tip_keys[3][1], tip_keys[3][2],
        tip_keys[4][0], tip_keys[4][1], tip_keys[4][2], pip_keys[0][0], pip_keys[0][1], pip_keys[0][2],
        pip_keys[1][0], pip_keys[1][1], pip_keys[1][2], pip_keys[2][0], pip_keys[2][1], pip_keys[2][2],
        pip_keys[3][0], pip_keys[3][1], pip_keys[3][2], pip_keys[4][0], pip_keys[4][1], pip_keys[4][2],
        dip_keys[0][0], dip_keys[0][1], dip_keys[0][2], dip_keys[1][0], dip_keys[1][1], dip_keys[1][2],
        dip_keys[2][0], dip_keys[2][1], dip_keys[2][2], dip_keys[3][0], dip_keys[3][1], dip_keys[3][2],
        dip_keys[4][0], dip_keys[4][1], dip_keys[4][2], mcp_keys[0][0], mcp_keys[0][1], mcp_keys[0][2],
        mcp_keys[1][0], mcp_keys[1][1], mcp_keys[1][2], mcp_keys[2][0], mcp_keys[2][1], mcp_keys[2][2],
        mcp_keys[3][0], mcp_keys[3][1], mcp_keys[3][2], mcp_keys[4][0], mcp_keys[4][1], mcp_keys[4][2]]
        writer.writerow(result)
        # img = cv2.imread(base_path + frame, cv2.IMREAD_ANYDEPTH)
        # norm_image = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # cv2.imshow("depth", norm_image)
        # cv2.waitKey(2)
        #
        # mlab.clf
        # mlab.figure(bgcolor=(1,1,1))
        # show_hand(shadow_points, 'shadow')
        # show_hand(local_points, 'human')
        # mlab.savefig(filename= "./data/handshape/" + frame +'.png')
        # mlab.close()
        # mlab.show()
    csvSum.close()
