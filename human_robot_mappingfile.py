import numpy as np
import math
import csv

class Map_Loader(object):
    def __init__(self, base_path= "./data/"):
        # load data
        self.base_path = base_path
        DataFile = open(base_path + "Training_Annotation.txt", "r")

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
        rh_palm, rh_pip_mcp, rh_dip_pip, rh_tip_dip = self.shadow
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

        keys= rh_tip_dip_key/1000
        # from IPython import embed;embed()
        return keys, frame

    def shadow_model(self):
        # shadow hand length
        rh_tf_palm = math.sqrt(math.pow(29, 2) + math.pow(34, 2))
        rh_ff_palm = 95
        rh_mf_palm = 99
        rh_rf_palm = 95
        rh_lf_palm = 86.6
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


if __name__ == '__main__':
    batch_size = 1
    base_path= "./data/"
    map_loader = Map_Loader(base_path)
    csvSum = open(base_path + "human_robot_mapdata.csv", "w")
    writer = csv.writer(csvSum)
    print(len(map_loader.framelist))
    for i in range(0, len(map_loader.framelist)):
        key, frame = map_loader.map(i)
        ## save key
        result = [frame, key[0][0], key[0][1], key[0][2], key[1][0], key[1][1], key[1][2], key[2][0]
        , key[2][1], key[2][2], key[3][0], key[3][1], key[3][2], key[4][0], key[4][1], key[4][2]]
        writer.writerow(result)
    csvSum.close()
