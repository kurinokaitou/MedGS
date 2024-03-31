# In tigre.py, the geometry model of CBCT is defined, 
# and the pre-reconstruction images using SART will be used as prior knowledge initilizing the gaussians
# the SART-reconstructed CT-volume will be preserved untill the training process finish
import os
import cv2
import pickle
import numpy as np
from typing import NamedTuple
from torch.utils.data import Dataset
from PIL import Image
from preprocess.geometry import ConeGeometry
from ct.reconstructor import reconstruct

from utils.graphics_utils import focal2fov

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class TIGREDataset(Dataset):
    """
    TIGRE dataset.
    """
    def __init__(self, path, type="train"):    
        super().__init__()

        with open(path, "rb") as handle:
            data = pickle.load(handle)
        self.path = path
        self.geo = ConeGeometry(data)
        self.type = type
        self.n_samples = data["num"+type.title()]
        self.gt_image = data["image"]
        self.cam_infos = self.buildCamInfos(data)
        # self.reconstructed_volume = reconstruct(data[type]["projections"],
        #                                         data[type]["angles"],
        #                                         self.geo)
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.type == "train":
            projs_valid = (self.projs[index]>0).flatten()
            coords_valid = self.coords[projs_valid]
            select_inds = np.random.choice(coords_valid.shape[0], size=[self.n_rays], replace=False)
            select_coords = coords_valid[select_inds].long()
            rays = self.rays[index, select_coords[:, 0], select_coords[:, 1]]
            projs = self.projs[index, select_coords[:, 0], select_coords[:, 1]]
            out = {
                "projs":projs,
                "rays":rays,
            }
        elif self.type == "val":
            rays = self.rays[index]
            projs = self.projs[index]
            out = {
                "projs":projs,
                "rays":rays,
            }
        return out
    
    def buildCamInfos(self, data):
        cam_infos = []
        # r_t_infos = []
        for idx, proj in enumerate(data[self.type]["projections"]):
            angle = data[self.type]["angles"][idx]
            R, T = self.angle2RotAndTrans(angle)
            fovX = focal2fov(self.geo.DSD, self.geo.sDetector[0])
            fovY = focal2fov(self.geo.DSD, self.geo.sDetector[1])
            proj_uint8 = (cv2.normalize(proj, None, 0, 1, cv2.NORM_MINMAX)  * 255).astype(np.uint8)
            proj_img = Image.fromarray(np.repeat(np.expand_dims(proj_uint8, axis=2), 3, axis=2))
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, 
                                        FovX=fovX, FovY=fovY, 
                                        image=proj_img, image_name=f"projection{idx}", 
                                        image_path=os.path.join(self.path, f"images/projection{idx}.png"),
                                        width=int(self.geo.nDetector[0]),
                                        height=int(self.geo.nDetector[1]))) 
        #     r_t_infos.append({"position": T.tolist(), "rotation": [x.tolist() for x in R]})            
        # if self.type == "train": 
        #     with open("cameras.json",'w') as file:
        #         json.dump(r_t_infos, file)
        #         print("R, T have written to file")
        return cam_infos
        
    
    def angle2RotAndTrans(self, angle):
        trans = np.array([self.geo.DSO * np.cos(angle), 0, self.geo.DSO * np.sin(angle)])
        new_angle = np.pi - angle
        rot = np.array([[np.sin(new_angle), 0, np.cos(new_angle)], # 行主序
                    [0, 1, 0],
                    [np.cos(new_angle), 0, -np.sin(new_angle)]])
        c2w = np.zeros((4, 4))
        c2w[:3, :3] = rot
        c2w[:3, 3] = trans
        c2w[3, 3] = 1.0
        
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].transpose()
        T = w2c[:3, 3]
        
        return R, T
    
    def getNearFar(self, tolerance=0):
        dist1 = np.linalg.norm([self.geo.offOrigin[0] - self.geo.sVoxel[0] / 2, self.geo.offOrigin[1] - self.geo.sVoxel[1] / 2])
        dist2 = np.linalg.norm([self.geo.offOrigin[0] - self.geo.sVoxel[0] / 2, self.geo.offOrigin[1] + self.geo.sVoxel[1] / 2])
        dist3 = np.linalg.norm([self.geo.offOrigin[0] + self.geo.sVoxel[0] / 2, self.geo.offOrigin[1] - self.geo.sVoxel[1] / 2])
        dist4 = np.linalg.norm([self.geo.offOrigin[0] + self.geo.sVoxel[0] / 2, self.geo.offOrigin[1] + self.geo.sVoxel[1] / 2])
        dist_max = np.max([dist1, dist2, dist3, dist4])
        near = np.max([0, self.geo.DSO - dist_max - tolerance])
        far = np.min([self.geo.DSO * 2, self.geo.DSO + dist_max + tolerance])
        return near, far

