from __future__ import division
import numpy as np
from PIL import Image
import math
import torch.utils.data

# shadow hand length
rh_tf_palm = math.sqrt(math.pow(29,2)+math.pow(34,2))
rh_if_palm = 95
rh_mf_palm = 99
rh_rf_palm = 95
rh_lf_palm = 86.6
rh_palm = np.array([rh_tf_palm, rh_if_palm, rh_mf_palm, rh_rf_palm, rh_rf_palm])

rh_tf_pip_mcp = 38
rh_tf_pip_dip = 32
rh_tf_tip_dip = 27.5

rh_if_pip_mcp = 45
rh_if_pip_dip = 25
rh_if_tip_dip = 26

rh_mf_pip_mcp = 45
rh_mf_pip_dip = 25
rh_mf_tip_dip = 26

rh_rf_pip_mcp = 45
rh_rf_pip_dip = 25
rh_rf_tip_dip = 26

rh_lf_pip_mcp = 45
rh_lf_pip_dip = 25
rh_lf_tip_dip = 26

rh_pip_mcp = np.array([rh_tf_pip_mcp, rh_if_pip_mcp, rh_mf_pip_mcp, rh_rf_pip_mcp, rh_lf_pip_mcp])
rh_pip_dip = np.array([rh_tf_pip_dip, rh_if_pip_dip, rh_mf_pip_dip, rh_rf_pip_dip, rh_lf_pip_dip])
rh_tip_dip = np.array([rh_tf_tip_dip, rh_if_tip_dip, rh_mf_tip_dip, rh_rf_tip_dip, rh_lf_tip_dip])

# rh_tf_len = rh_tf_palm + rh_tf_pip_dip + rh_tf_tip_dip + rh_tf_pip_mcp
# rh_if_len = rh_if_palm + rh_if_pip_dip + rh_if_tip_dip + rh_if_pip_mcp
# rh_mf_len = rh_mf_palm + rh_mf_pip_dip + rh_mf_tip_dip + rh_mf_pip_mcp
# rh_rf_len = rh_rf_palm + rh_rf_pip_dip + rh_rf_tip_dip + rh_rf_pip_mcp
# rh_lf_len = rh_lf_palm + rh_lf_pip_dip + rh_lf_tip_dip + rh_lf_pip_mcp

class HumanHandLoader(torch.utils.data.Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        DataFile = open(self.base_path + "test.txt", "r")

        lines = DataFile.read().splitlines()
        self.framelist = [ln.split(' ')[0].replace("\t", "") for ln in lines]
        label = [ln.split('\t')[1:] for ln in lines]
        self.label = []
        for ln in label:
            ll = ln[0:63]
            self.label.append([float(l.replace(" ", "")) for l in ll])

        self.label = np.array(self.label)
        self.num_data = len(self.framelist)
        DataFile.close()

    def __getitem__(self, index):
        idx = self.framelist[index]
        img = Image.open(self.base_path + str(idx))
        keypoints = self.label[index]

        tf_palm = keypoints[0:3] - keypoints[3:6]
        if_palm = keypoints[0:3] - keypoints[6:9]
        mf_palm = keypoints[0:3] - keypoints[9:12]
        rf_palm = keypoints[0:3] - keypoints[12:15]
        lf_palm = keypoints[0:3] - keypoints[15:18]

        tf_pip_mcp = keypoints[18:21] - keypoints[3:6]
        tf_pip_dip = keypoints[18:21] - keypoints[21:24]
        tf_tip_dip = keypoints[24:27] - keypoints[21:24]

        if_pip_mcp = keypoints[27:30] - keypoints[6:9]
        if_pip_dip = keypoints[27:30] - keypoints[30:33]
        if_tip_dip = keypoints[33:36] - keypoints[30:33]

        mf_pip_mcp = keypoints[36:39] - keypoints[9:12]
        mf_pip_dip = keypoints[36:39] - keypoints[39:42]
        mf_tip_dip = keypoints[42:45] - keypoints[39:42]

        rf_pip_mcp = keypoints[45:48] - keypoints[12:15]
        rf_pip_dip = keypoints[45:48] - keypoints[48:51]
        rf_tip_dip = keypoints[51:54] - keypoints[48:51]

        lf_pip_mcp = keypoints[54:57] - keypoints[15:18]
        lf_pip_dip = keypoints[54:57] - keypoints[57:60]
        lf_tip_dip = keypoints[60:63] - keypoints[57:60]

        palm = np.array([tf_palm, if_palm, mf_palm, rf_palm, lf_palm])
        hh_palm = np.linalg.norm(palm, axis=1)

        pip_mcp = np.array([tf_pip_mcp, if_pip_mcp, mf_pip_mcp, rf_pip_mcp, lf_pip_mcp])
        pip_dip = np.array([tf_pip_dip, if_pip_dip, mf_pip_dip, rf_pip_dip, lf_pip_dip])
        tip_dip = np.array([tf_tip_dip, if_tip_dip, mf_tip_dip, rf_tip_dip, lf_tip_dip])
        hh_pip_mcp = np.linalg.norm(pip_mcp, axis=1)
        hh_pip_dip = np.linalg.norm(pip_dip, axis=1)
        hh_tip_dip = np.linalg.norm(tip_dip, axis=1)

        hh_len = hh_palm + hh_pip_mcp + hh_pip_dip + hh_tip_dip

        coe_palm = rh_palm / hh_palm
        rh_wrist_mcp_key = np.multiply(coe_palm.reshape(-1, 1), palm)
        coe_pip_mcp = rh_pip_mcp / hh_pip_mcp
        rh_pip_mcp_key = np.multiply(coe_pip_mcp.reshape(-1, 1), pip_mcp - palm) + rh_wrist_mcp_key
        coe_pip_dip = rh_pip_dip / hh_pip_dip
        rh_pip_dip_key = np.multiply(coe_pip_dip.reshape(-1, 1), pip_dip - pip_mcp) + rh_pip_mcp_key
        coe_tip_dip = rh_tip_dip / hh_tip_dip
        rh_tip_dip_key = np.multiply(coe_tip_dip.reshape(-1, 1), tip_dip - pip_dip) + rh_pip_dip_key

        return img,rh_tip_dip_key

    def __len__(self):
        return self.num_data

if __name__ == '__main__':
    base_path = "./data/"
    train = HumanHandLoader(base_path)
    train_loader = torch.utils.data.DataLoader(train, batch_size = 16, shuffle=True, num_workers=2)
    img, rh_tip_dip_key = train.__getitem__(4)
    img.show()
