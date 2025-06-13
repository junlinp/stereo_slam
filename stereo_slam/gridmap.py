import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from disparity import compute_disparity_sgbm, foundation_stereo_disparity
from features import extract_keypoints, match_keypoints, match_keypoints_original, get_embeddings
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from landmark_tracker import LandmarkTracker
from bundle_adjustment import BundleAdjustment
from rigid3d import Rigid3d
from pose_graph_solver import PoseGraphSolver
import pypose as pp
import stereo_slam_pybind

def test_kitti_yixuan_traj_generate_point_cloud(self):
        FRAME_STEP = 10
        NUM_FRAMES = len(self.kitti_data.load_kitti_poses())
        # NUM_FRAMES = 300

        K, baseline = self.kitti_data.load_intrinsics()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        lines=np.loadtxt("traj.csv",dtype=str, delimiter=",", skiprows=1)

        index = np.array(lines[:,0],dtype=int)
        vio_p_xyz = np.array(lines[:,1:4],dtype=float)
        vio_q_xyzw = np.array(lines[:,4:8],dtype=float)

        load_traj = np.array([[item[0], item[2]] for item in vio_p_xyz])
        plt.figure(figsize=(10, 8))
        plt.plot(load_traj[:, 0], load_traj[:, 1], 'k-', label='Trajectory')
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.title(f'Trajectory (KITTI seq {self.kitti_data.seq_id}, step={FRAME_STEP})')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.savefig(f"kitti_trajectory_{self.kitti_data.seq_id}_yixuan.png")
        plt.close()


        tree = stereo_slam_pybind.OcTree(0.1)
        for i in tqdm(range(index.shape[0]), desc="Processing frames"):
            # print(f"index: {index[i]}")
            # print(f"vio_p_xyz: {vio_p_xyz[i]}")
            # print(f"vio_q_xyzw: {vio_q_xyzw[i]}")
            pose = Rigid3d(translation=vio_p_xyz[i], rotation_quaternion=vio_q_xyzw[i])
            curr_left_img = self.kitti_data.load_gray_image(index[i], left=True)
            curr_right_img = self.kitti_data.load_gray_image(index[i], left=False)
            curr_disparity_map = compute_disparity_sgbm(curr_left_img, curr_right_img)
            color_point_cloud = []  
            for v in range(0, curr_left_img.shape[0], 4):
                for u in range(0, curr_left_img.shape[1], 4):
                    if curr_disparity_map[v, u] > 1:
                        Z = fx * baseline / curr_disparity_map[v, u]
                        if Z > 50:
                            continue
                        X = (u - cx) * Z / fx
                        Y = (v - cy) * Z / fy
                        world_point = pose.to_matrix33() @ np.array([X, Y, Z]) + pose.translation_vector()
                        tree.insert_ray(pose.translation_vector(), world_point)
                        color_point_cloud.append([world_point[0], world_point[1], world_point[2], curr_left_img[v, u], curr_left_img[v, u], curr_left_img[v, u]])
        tree.write_binary("map.bt")