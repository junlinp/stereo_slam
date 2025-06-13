import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from disparity import compute_disparity_sgbm
from features import extract_keypoints, match_keypoints, match_keypoints_original, get_embeddings, segment_sky_mask
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from landmark_tracker import LandmarkTracker
from rigid3d import Rigid3d
import open3d as o3d
from pose_graph_solver import PoseGraphSolver
import pypose as pp
import stereo_slam_pybind
from codetiming import Timer
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
import scipy.ndimage
import torch

# === Draw disparity map with sampled points ===
def draw_disparity_points(img, disparity):
    img = img.copy()
    h, w = disparity.shape
    for y in range(0, h, 10):
        for x in range(0, w, 10):
            d = disparity[y, x]
            if d <= 0:     continue
            color = plt.cm.jet(d / 100.0)[:3] # Normalize and colormap
            color = tuple(int(c * 255) for c in color)
            cv2.circle(img, (x, y), 3, color, -1)
    return img

# === Draw line matches between frames ===
def draw_line_matches(img, kpts0, kpts1):
    img = img.copy()
    for pt0, pt1 in zip(kpts0, kpts1):
        x0, y0 = int(pt0[0]), int(pt0[1])
        x1, y1 = int(pt1[0]), int(pt1[1])
        cv2.line(img, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=1)
        cv2.circle(img, (x0, y0), radius=2, color=(0, 0, 255), thickness=-1)
    return img

def draw_two_view_matches(img0, img1, kpts0, kpts1):
    cv_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0) for i in range(kpts0.shape[0])]
    # convert kpts_prev and kpts_curr to cv2.KeyPoint 
    cv_kpts_prev = [cv2.KeyPoint(x=kpts0[index, 0].item(), y=kpts0[index, 1].item(), size=20) for index in range(kpts0.shape[0])]
    cv_kpts_curr = [cv2.KeyPoint(x=kpts1[index, 0].item(), y=kpts1[index, 1].item(), size=20) for index in range(kpts1.shape[0])]
    output_image = cv2.drawMatches(img0, cv_kpts_prev, img1, cv_kpts_curr, cv_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return output_image
def write_ply(points, filename):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\n")  
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point in tqdm(points, desc="Writing PLY file"):
            f.write("{} {} {} {} {} {}\n".format(point[0], point[1], point[2], point[3], point[4], point[5]))

def plot_trajectory(poses, title, save_path):
    x, z = [], []
    for pose in poses:
            x.append(pose[0, 3])
            z.append(pose[2, 3])
        # save the plot to a file
    plt.figure(figsize=(8, 6))
    plt.plot(x, z, label='Ground Truth', linewidth=2)
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def ComputePose(kpts_prev, kpts_curr, disparity, K, baseline):
    # === Triangulate 3D from disparity and get 2D–3D pairs
    points_3d, points_2d = [], []
    for pt_prev, pt_curr in zip(kpts_prev, kpts_curr):
        u, v = int(pt_curr[0]), int(pt_curr[1])
        if 0 <= v < disparity.shape[0] and 0 <= u < disparity.shape[1]:
            disp = disparity[v, u]
            if disp > 1:
                Z = K[0, 0] * baseline / disp
                X = (pt_curr[0] - K[0, 2]) * Z / K[0, 0]
                Y = (pt_curr[1] - K[1, 2]) * Z / K[1, 1]
                points_3d.append([X, Y, Z])
                points_2d.append(pt_prev)

    if len(points_3d) < 6:
        return False, np.eye(4)

    points_3d = np.array(points_3d, dtype=np.float32)
    points_2d = np.array(points_2d, dtype=np.float32)

    # === Solve PnP
    success, rvec, tvec, _ = cv2.solvePnPRansac(points_3d, points_2d, K, None)
    if not success:
        return False, np.eye(4)
    
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return True, T

def np_cosine_similarity(a: np.ndarray, b: np.ndarray):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_top_similar_embedding(embeddings, index):
    max_simularity = 0
    max_simularity_index = 0
    found = False
    query_embedding = embeddings[index] / np.linalg.norm(embeddings[index])
    for i, embedding in embeddings.items():
        if i > index - 500:
            continue
        reference_embedding = embedding / np.linalg.norm(embedding)
        simularity = np_cosine_similarity(query_embedding[0, :], reference_embedding[0, :])

        if simularity > max_simularity:
            max_simularity = simularity
            max_simularity_index = i
            found = True
    if not found:
        return 0, 0
    return max_simularity, max_simularity_index
    
def depth_to_points_in_global(depth, current_pose:Rigid3d, fx, fy, cx, cy, sky_mask):
    points_in_global = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i, j] > 0 and sky_mask[i, j] == 0:
                X = (j - cx) * depth[i, j] / fx 
                Y = (i - cy) * depth[i, j] / fy
                Z = depth[i, j]
                points_in_global.append(current_pose.to_matrix33() @ (np.array([X, Y, Z])) + current_pose.translation)
    return points_in_global




class PlyOutput:
    def __init__(self, output_path:str, seq_id:str):
        self.output_path = output_path
        self.seq_id = seq_id
        self.frame_ids = []

        self.output_path_base =  os.path.basename(output_path)

    def append_points(self, points_and_colors, seq_id, frame_id):
        if not os.path.exists(os.path.join("/mnt/nas/share-all/junlinp/tmp", self.output_path_base)):
            os.makedirs(os.path.join("/mnt/nas/share-all/junlinp/tmp", self.output_path_base))
        with open(os.path.join("/mnt/nas/share-all/junlinp/tmp", self.output_path_base, f"{self.seq_id}_{frame_id}.ply"), 'a') as f:
            for point_and_color in points_and_colors:
                f.write(f"{point_and_color[0]} {point_and_color[1]} {point_and_color[2]} {point_and_color[3]} {point_and_color[4]} {point_and_color[5]}\n")
        self.frame_ids.append(frame_id)

    def output_ply(self):

        # compute total points
        total_points = 0
        for frame_id in self.frame_ids:
            with open(os.path.join("/mnt/nas/share-all/junlinp/tmp", self.output_path_base, f"{self.seq_id}_{frame_id}.ply"), 'r') as f:
                for line in f:
                    total_points += 1
        # output ply
        with open(self.output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {total_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for frame_id in self.frame_ids:
                with open(os.path.join("/mnt/nas/share-all/junlinp/tmp", self.output_path_base, f"{self.seq_id}_{frame_id}.ply"), 'r') as tmp_f:
                    for line in tmp_f:
                        f.write(line)


def occupancy_grid_to_height_map(occupancy_map:stereo_slam_pybind.OcTree, origin:np.ndarray, height_map_size:np.ndarray, resolution:float, threshold:float):
    # in meters
    X, Y, Z = np.array(height_map_size / resolution, np.int32)

    height_map = np.full((X + 1, Z + 1), np.nan, dtype=np.float32)
    count_map = np.zeros((X + 1, Z + 1), dtype=np.int32)
    query_min = origin - np.array([height_map_size[0] * 0.5, height_map_size[1] * 0.5, height_map_size[2] * 0.5])
    query_max = origin + np.array([height_map_size[0] * 0.5, height_map_size[1] * 0.5, height_map_size[2] * 0.5])

    # x, y, z, occupancy, grid_size
    occupancy_map_points = occupancy_map.search_bounding_box(query_min, query_max)

    height_map_origin = np.array([origin[0] - height_map_size[0] * 0.5, origin[1], origin[2] - height_map_size[2] * 0.5])

    for point in occupancy_map_points:
        x, y, z, occupancy, grid_size = point
        if occupancy > threshold:
            x_idx = int((x - height_map_origin[0]) / resolution)
            z_idx = int((z - height_map_origin[2]) / resolution)
            height_map[x_idx, z_idx] = np.min([height_map[x_idx, z_idx], y]) if not np.isnan(height_map[x_idx, z_idx]) else y
            count_map[x_idx, z_idx] += 1
            # print(f"x_idx: {x_idx}, z_idx: {z_idx}, height_map: {height_map[x_idx, z_idx]}, count_map: {count_map[x_idx, z_idx]}")
    # print(f"height_map: {height_map[count_map > 0].shape}")
    # print(f"height_map: {height_map[count_map > 0]}")
    # print(f"count_map: {count_map[count_map > 0].shape}")
    return height_map, height_map_origin

def max_pooling_height_map(height_map:np.ndarray, kernel_size:int = 5):

    # mask height_map with nan
    height_map_mask = np.isnan(height_map)
    filled = np.copy(height_map) * -1.0
    filled[height_map_mask] = -np.inf

    # max pooling
    height_map_max = scipy.ndimage.maximum_filter(filled, size=kernel_size)

    return height_map_max * -1.0

def generate_trajectory_library(num_samples : int =100, duration=5.0, dt=0.1, acc_std:float = 0.1, omega_std_deg:float = 1.0):
    """
    Generate a library of 2D trajectories under Gaussian-sampled different acceleration and angular velocity.

    Args:
        num_samples: number of trajectories to generate
        duration (float): total time duration (s)
        dt (float): time step (s)
        acc_std (float): standard deviation of acceleration (m/s^2)
        omega_std_deg (float): standard deviation of angular velocity (deg/s)

    Returns:
        trajectories (np.ndarray): (num_samples, 2, num_steps)
        params (list): list of (dv, dtheta) used per trajectory
    """

    times = np.arange(0, duration + dt, dt)
    trajectories = []
    params = []
    acc_samples = np.random.normal(loc=0, scale=acc_std, size=num_samples)
    omega_samples = np.random.normal(loc=0, scale=omega_std_deg * np.pi / 180.0, size=num_samples)

    for dv, dtheta in zip(acc_samples, omega_samples):
        x, y, theta = 0.0, 0.0, 0.0
        v = 10.0
        traj = []
        for _ in times:
            v += dv * dt
            theta += dtheta * dt
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            traj.append([x, y, theta])
        trajectories.append(np.array(traj))
        params.append((dv, dtheta))
    return trajectories, params

def score_trajectories_by_height_map(trajectories:np.ndarray, height_map:np.ndarray, height_map_origin:np.ndarray, resolution:float, threshold:float):
    scores = []
    for traj in trajectories:
        valid_steps = 0
        total_steps = traj.shape[0]
        for i in range(total_steps):
            x_in_global = traj[i, 0]
            y_in_global = traj[i, 1]
            z_in_global = traj[i, 2]
            x_idx = int((x_in_global - height_map_origin[0]) / resolution)
            z_idx = int((z_in_global - height_map_origin[2]) / resolution)
            if 0 <= x_idx < height_map.shape[0] and 0 <= z_idx < height_map.shape[1]:
                if height_map[x_idx, z_idx] > threshold:
                    valid_steps += 1
        scores.append(valid_steps)
    return scores

def scalar_to_rgb(value, vmin=0.0, vmax=1.0, cmap_name='viridis'):
    """
    Maps a scalar value to an RGB color using a matplotlib colormap.

    Args:
        value (float): Scalar value to map, should be in [vmin, vmax].
        vmin (float): Minimum possible value (default 0.0).
        vmax (float): Maximum possible value (default 1.0).
        cmap_name (str): Name of the colormap to use (e.g., 'viridis', 'jet').

    Returns:
        tuple: RGB values in range [0, 255].
    """
    norm_value = np.clip((value - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm_value)
    rgb = tuple((np.array(rgba[:3]) * 255).astype(np.uint8))
    return rgb

def filter_cloud_points_by_sky(cloud_points_in_global:np.ndarray):

    filtered_cloud_points_in_global = []
    for point in cloud_points_in_global:
        if point[1] > -3.0:
            filtered_cloud_points_in_global.append(point)
    return np.array(filtered_cloud_points_in_global)

class KittiData:
    def __init__(self, data_path:str, seq_id:str):
        self.data_path = data_path
        self.seq_id = seq_id
        

    def load_kitti_poses(self):
        poses = []
        with open(os.path.join(self.data_path, f"poses/{self.seq_id}.txt"), 'r') as f:
            for line in f:
                T = np.fromstring(line, sep=' ').reshape(3, 4)
                poses.append(T)
        return poses



    def load_image_pair(self, index):
        left_img = cv2.imread(os.path.join(self.data_path, f"{self.seq_id}/image_0/{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(os.path.join(self.data_path, f"{self.seq_id}/image_1/{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
        return left_img, right_img

    def load_gray_image(self, index, left=True):
        if left:
            img = cv2.imread(os.path.join(self.data_path, f"{self.seq_id}/image_0/{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(os.path.join(self.data_path, f"{self.seq_id}/image_1/{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
        return img

    def load_intrinsics(self):
        with open(os.path.join(self.data_path, f"{self.seq_id}/calib.txt"), 'r') as f:
            for line in f:
                if line.startswith('P0:'):
                    P0 = np.fromstring(line[len('P0:'):], sep=' ').reshape(3, 4)
                elif line.startswith('P1:'):
                    P1 = np.fromstring(line[len('P1:'):], sep=' ').reshape(3, 4)
        # Intrinsics from P0
        fx = P0[0, 0]
        fy = P0[1, 1]
        cx = P0[0, 2]
        cy = P0[1, 2]
        # Baseline from Tx difference between P0 and P1
        Tx_P0 = P0[0, 3] / fx # should be 0.0
        Tx_P1 = P1[0, 3] / fx # actual baseline in meters
        baseline = abs(Tx_P1 - Tx_P0)
        K = np.array([[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]], dtype=np.float32)
        return K, baseline

class KittiUnitTest(unittest.TestCase):
    def setUp(self):
        path = "/mnt/nas/share-all/junlinp/PublicDataSet/KITTI/dataset/sequences"
        self.kitti_data = KittiData(path, "05")

    def test_kitti_data(self):
        poses = self.kitti_data.load_kitti_poses()
        plot_trajectory(poses, f"KITTI Ground Truth Trajectory - Sequence {self.kitti_data.seq_id}", f"kitti_trajectory_{self.kitti_data.seq_id}.png")

    def test_kitti_image_pair(self):
        left_img, right_img = self.kitti_data.load_image_pair(0)
        self.assertIsNotNone(left_img)
        self.assertIsNotNone(right_img)

    def test_kitti_disparity(self):
        left_img, right_img = self.kitti_data.load_image_pair(0)
        disparity = compute_disparity_sgbm(left_img, right_img)
        plt.imshow(disparity, cmap='plasma')
        plt.colorbar(label='Disparity (pixels)')
        plt.savefig(f'kitti_disparity_{self.kitti_data.seq_id}_0.png')
        plt.close()

        disparity_vis =draw_disparity_points(cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR), disparity)
        plt.imshow(cv2.cvtColor(disparity_vis, cv2.COLOR_BGR2RGB))
        plt.savefig(f'kitti_disparity_vis_{self.kitti_data.seq_id}_0.png')
        plt.close()

    def test_kitti_foundation_stereo_disparity(self):
        left_img, right_img = self.kitti_data.load_image_pair(0)
        disparity = foundation_stereo_disparity(cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR))
        plt.imshow(disparity, cmap='plasma')
        plt.colorbar(label='Disparity (pixels)')
        plt.savefig(f'kitti_disparity_foundation_stereo_{self.kitti_data.seq_id}_0.png')
        plt.close()

    def test_kitti_keypoints(self):
        left_img0 = self.kitti_data.load_gray_image(0, left=True)
        left_img1 = self.kitti_data.load_gray_image(1, left=True)

        kpts0 = extract_keypoints(left_img0)
        kpts1 = extract_keypoints(left_img1)

        matches, k0, k1 = match_keypoints(kpts0, kpts1)

        # print(f"matches: {matches}")
        matches_kpts0 = kpts0['keypoints'].squeeze()[matches[..., 0]]
        matches_kpts1 = kpts1['keypoints'].squeeze()[matches[..., 1]]

        # kpts0, kpts1 = match_keypoints_original(left_img0, left_img1)
        #print(f"kpts0: {kpts0.shape}")
        #print(f"kpts1: {kpts1.shape}")
        motion_vis= draw_line_matches(cv2.cvtColor(left_img0, cv2.COLOR_GRAY2BGR), matches_kpts0, matches_kpts1)
        # motion_vis = draw_line_matches(cv2.cvtColor(left_img0, cv2.COLOR_GRAY2BGR), kpts0, kpts1)

        plt.imshow(cv2.cvtColor(motion_vis, cv2.COLOR_BGR2RGB))
        plt.savefig(f'kitti_Left-to-Left Motion (LightGlue)_{self.kitti_data.seq_id}_0.png')
        plt.close()

    def test_kitti_read_intrinsics(self):
        K, baseline = self.kitti_data.load_intrinsics()
        print(f"K: {K}")
        print(f"baseline (m): {baseline}")

    def test_kitti_two_view_poses(self):
        # === Reconstruct 3D from disparity at keypoints in left1 ===
        left_img0 = self.kitti_data.load_gray_image(0, left=True)
        right_img0 = self.kitti_data.load_gray_image(0, left=False)
        left_img1 = self.kitti_data.load_gray_image(1, left=True)
        right_img1 = self.kitti_data.load_gray_image(1, left=False)
        disparity_map = compute_disparity_sgbm(left_img0, right_img0)
        points_3d = []
        points_2d = []
        kpts_0 = extract_keypoints(left_img0)
        kpts_1 = extract_keypoints(left_img1)
        matches, k0, k1 = match_keypoints(kpts_0, kpts_1)

        K, baseline = self.kitti_data.load_intrinsics()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        for pt_0, pt_1 in zip(k0, k1):
            u, v = int(pt_1[0]), int(pt_1[1]) # point in left1
            if 0 <= v < disparity_map.shape[0] and 0 <= u < disparity_map.shape[1]:
                disp = disparity_map[v, u]
                if disp > 1: # valid disparity
                    Z = fx * baseline / disp
                    X = (pt_1[0] - cx) * Z / fx
                    Y = (pt_1[1] - cy) * Z / fy
                    points_3d.append([X, Y, Z])
                    points_2d.append(pt_0) # 2D point in left0
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)
        print(f"Valid 3D-2D correspondences: {len(points_3d)}")
        # === Run PnP ===
        if len(points_3d) >= 6:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=points_3d,
        imagePoints=points_2d,
        cameraMatrix=K,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE
        )
            if success:
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec.ravel()
                print("Estimated camera pose (left1 relative to left0):\n", T)
                print(f"Inliers used: {len(inliers)}")
            else:
                print("PnP failed")
        else:
            print("Not enough correspondences for PnP") 

    def test_kitti_pose_estimation(self):
        FRAME_STEP = 10
        NUM_FRAMES = len(self.kitti_data.load_kitti_poses())

        K, baseline = self.kitti_data.load_intrinsics()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        poses = [np.eye(4)]

        for i in tqdm(range(FRAME_STEP, NUM_FRAMES, FRAME_STEP), desc="Processing frames"):
            prev_left_img = self.kitti_data.load_gray_image(i - FRAME_STEP, left=True)
            prev_right_img = self.kitti_data.load_gray_image(i - FRAME_STEP, left=False)
            curr_left_img = self.kitti_data.load_gray_image(i, left=True)
            curr_right_img = self.kitti_data.load_gray_image(i, left=False)
            curr_disparity_map = compute_disparity_sgbm(curr_left_img, curr_right_img)
            kets_curr = extract_keypoints(curr_left_img)
            kets_prev = extract_keypoints(prev_left_img)
            matches, matches_kpts_prev, matches_kpts_curr = match_keypoints(kets_prev, kets_curr)
            points_3d = []
            points_2d = []
            for pt_prev, pt_curr in zip(matches_kpts_prev, matches_kpts_curr):
                u, v = int(pt_curr[0]), int(pt_curr[1])
                if 0 <= v < curr_disparity_map.shape[0] and 0 <= u < curr_disparity_map.shape[1]:
                    disp = curr_disparity_map[v, u]
                    if disp > 1:
                        Z = fx * baseline / disp
                        X = (pt_curr[0] - cx) * Z / fx  
                        Y = (pt_curr[1] - cy) * Z / fy
                        points_3d.append([X, Y, Z])
                        points_2d.append(pt_prev)
            points_3d = np.array(points_3d, dtype=np.float32)
            points_2d = np.array(points_2d, dtype=np.float32)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d,
            cameraMatrix=K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                print(f"Frame {i}: PnP failed")
                poses.append(poses[-1])
                continue
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.ravel()
            poses.append(poses[-1] @ np.linalg.inv(T))
        # chain pose
        # === Trajectory
        estimated_trajectory = np.array([[ pose[0, 3],-pose[2, 3] ] for pose in poses])
        gt_trajectory = np.array([[ pose[0, 3],pose[2, 3] ] for pose in self.kitti_data.load_kitti_poses()])
        #plot_trajectory(trajectory, f"KITTI Estimated Trajectory - Sequence {self.kitti_data.seq_id}", f"kitti_trajectory_{self.kitti_data.seq_id}_estimated.png")
        # === Plot
        plt.figure(figsize=(8, 6))
        plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], label='Estimated trajectory',linewidth=2, color='red')
        plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], label='Ground truth trajectory',linewidth=2, color='blue')
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.title(f'Trajectory (KITTI seq {self.kitti_data.seq_id}, step={FRAME_STEP})')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.savefig(f"kitti_trajectory_{self.kitti_data.seq_id}_compare.png")
        plt.close()
            
    def test_kitti_test_pose_estimation_foundation_stereo(self):
        FRAME_STEP = 10
        NUM_FRAMES = len(self.kitti_data.load_kitti_poses())

        K, baseline = self.kitti_data.load_intrinsics()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        poses = [np.eye(4)]

        for i in tqdm(range(FRAME_STEP, NUM_FRAMES, FRAME_STEP), desc="Processing frames"):
            prev_left_img = self.kitti_data.load_gray_image(i - FRAME_STEP, left=True)
            prev_right_img = self.kitti_data.load_gray_image(i - FRAME_STEP, left=False)
            curr_left_img = self.kitti_data.load_gray_image(i, left=True)
            curr_right_img = self.kitti_data.load_gray_image(i, left=False)

            curr_disparity_map = foundation_stereo_disparity(cv2.cvtColor(curr_left_img, cv2.COLOR_GRAY2BGR), cv2.cvtColor(curr_right_img, cv2.COLOR_GRAY2BGR), scale=0.25)

            kets_curr = extract_keypoints(curr_left_img)
            kets_prev = extract_keypoints(prev_left_img)
            matches, matches_kpts_prev, matches_kpts_curr = match_keypoints(kets_prev, kets_curr)
            points_3d = []
            points_2d = []
            for pt_prev, pt_curr in zip(matches_kpts_prev, matches_kpts_curr):
                u, v = int(pt_curr[0]), int(pt_curr[1])
                if 0 <= v < curr_disparity_map.shape[0] and 0 <= u < curr_disparity_map.shape[1]:
                    disp = curr_disparity_map[v, u]
                    if disp > 1:
                        Z = fx * baseline / disp
                        X = (pt_curr[0] - cx) * Z / fx  
                        Y = (pt_curr[1] - cy) * Z / fy
                        points_3d.append([X, Y, Z])
                        points_2d.append(pt_prev)
            points_3d = np.array(points_3d, dtype=np.float32)
            points_2d = np.array(points_2d, dtype=np.float32)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d,
            cameraMatrix=K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                print(f"Frame {i}: PnP failed")
                poses.append(poses[-1])
                continue
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.ravel()
            poses.append(poses[-1] @ np.linalg.inv(T))
        # chain pose
        # === Trajectory
        estimated_trajectory = np.array([[ pose[0, 3],-pose[2, 3] ] for pose in poses])
        gt_trajectory = np.array([[ pose[0, 3],pose[2, 3] ] for pose in self.kitti_data.load_kitti_poses()])
        #plot_trajectory(trajectory, f"KITTI Estimated Trajectory - Sequence {self.kitti_data.seq_id}", f"kitti_trajectory_{self.kitti_data.seq_id}_estimated.png")
        # === Plot
        plt.figure(figsize=(8, 6))
        plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], label='Estimated trajectory',linewidth=2, color='red')
        plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], label='Ground truth trajectory',linewidth=2, color='blue')
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.title(f'Trajectory (KITTI seq {self.kitti_data.seq_id}, step={FRAME_STEP})')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.savefig(f"kitti_trajectory_{self.kitti_data.seq_id}_compare_foundation_stereo.png")
        plt.close()

    def test_kitti_simularity(self):

        FRAME_STEP = 10
        NUM_FRAMES = len(self.kitti_data.load_kitti_poses())
        
        gt_poses = self.kitti_data.load_kitti_poses()
        image_embeddings = {}
        for i in tqdm(range(FRAME_STEP, NUM_FRAMES, FRAME_STEP), desc="Processing frames"):
            left_img = self.kitti_data.load_gray_image(i, left=True)
            right_img = self.kitti_data.load_gray_image(i, left=False)
            left_embeddings = get_embeddings(cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR))
            image_embeddings[i] = left_embeddings

        image_max_simularity ={}
        # compute the largest simularity for each frame and only compare to the embedding which timestamp is less then it
        for i in tqdm(range(FRAME_STEP, NUM_FRAMES, FRAME_STEP), desc="Processing simularity"):
            left_img = self.kitti_data.load_gray_image(i, left=True)
            left_embeddings = get_embeddings(cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR))
            max_simularity = 0
            max_simularity_index = 0
            filtered_image_embeddings = {k: v for k, v in image_embeddings.items() if k < i - 50 * FRAME_STEP}
            for j, embedding in filtered_image_embeddings.items():
                simularity = np_cosine_similarity(left_embeddings[0, :], embedding[0, :])
                if simularity > max_simularity:
                    max_simularity = simularity
                    max_simularity_index = j
            image_max_simularity[i] = max_simularity_index

            if not os.path.exists(f"kitti_simularity_{self.kitti_data.seq_id}"):
                os.makedirs(f"kitti_simularity_{self.kitti_data.seq_id}")

            query_pose = gt_poses[i]
            reference_pose = gt_poses[max_simularity_index]

            diff_position = np.linalg.norm(query_pose[0:3, 3] - reference_pose[0:3, 3]) 

            if image_max_simularity[i] > 0.9 and diff_position < 10:
                query_image = left_img
                reference_image = self.kitti_data.load_gray_image(max_simularity_index, left=True)

                gt_poses_2d = np.array([[pose[0, 3], pose[2, 3]] for pose in gt_poses])

                # draw the query image at left and reference image at right
                # title for query index and reference index
                # draw the query_pose_and_reference_pose at the bottom of ax1 and ax2
                # Create a 2x2 grid; use gridspec_kw to span bottom plot across both columns
                fig, axs = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

                # Top-left: query image
                axs[0, 0].imshow(query_image, cmap='gray')
                axs[0, 0].set_title(f"Query Index: {i}")
                axs[0, 0].axis('off')

                # Top-right: reference image
                axs[0, 1].imshow(reference_image, cmap='gray')
                axs[0, 1].set_title(f"Reference Index: {max_simularity_index}")
                axs[0, 1].axis('off')

                # Bottom: pose plot (use entire bottom row)
                ax_pose = plt.subplot2grid((2, 2), (1, 0), colspan=2)
                ax_pose.scatter(gt_poses_2d[:, 0],
                                gt_poses_2d[:, 1],
                                c='blue', s=100, label='Ground Truth')
                ax_pose.scatter(query_pose[0, 3],
                                query_pose[2, 3],
                                c='red', s=100, label='Query')
                ax_pose.scatter(reference_pose[0, 3],
                                reference_pose[2, 3],
                                c='green', s=100, label='Reference')
                # add legend for the scatter points
                ax_pose.legend()

                ax_pose.set_title("Query and Reference Pose")
                ax_pose.set_xlabel("X")
                ax_pose.set_ylabel("Y")
                ax_pose.grid(True)

                # Save and close
                plt.tight_layout()
                plt.savefig(f"kitti_simularity_{self.kitti_data.seq_id}/{i}.png")
                plt.close()

                # match the query_image and reference_image
                kpts_query = extract_keypoints(query_image)
                kpts_reference = extract_keypoints(reference_image)
                matches, kpts0, kpts1 = match_keypoints(kpts_query, kpts_reference)
                simularity_matches_vis = draw_two_view_matches(cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR), cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR), kpts0, kpts1)
                if not os.path.exists(f"kitti_simularity_{self.kitti_data.seq_id}_simularity_matches"):
                    os.makedirs(f"kitti_simularity_{self.kitti_data.seq_id}_simularity_matches")
                cv2.imwrite(f"kitti_simularity_{self.kitti_data.seq_id}_simularity_matches/{i}.png", simularity_matches_vis)

        # draw the simularity with 2d plot with gt_poses where position is x-z plane and the color is the simularity
        gt_poses = np.array([[pose[0, 3], pose[2, 3]] for pose in gt_poses[::FRAME_STEP][:-1]])
        plt.figure(figsize=(8, 6))

        plt.scatter(gt_poses[:, 0], gt_poses[:, 1], c=[v for v in image_max_simularity.values()], cmap='viridis', vmin=0, vmax=1)

        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.title(f'Simularity (KITTI seq {self.kitti_data.seq_id})')
        plt.axis('equal')
        plt.grid()
        plt.colorbar(label='Simularity')
        plt.savefig(f"kitti_simularity_{self.kitti_data.seq_id}.png")
        plt.close()

    def test_kitti_tracker_statistics(self):
        FRAME_STEP = 10
        NUM_FRAMES = len(self.kitti_data.load_kitti_poses())

        K, baseline = self.kitti_data.load_intrinsics()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        poses = { 0 : np.eye(4) }

        landmark_tracker = LandmarkTracker()
        bundle_adjustment = BundleAdjustment()
        bundle_adjustment.update_camera_intrinsics(K)

        for i in tqdm(range(FRAME_STEP, NUM_FRAMES, FRAME_STEP), desc="Processing frames"):
            prev_left_img = self.kitti_data.load_gray_image(i - FRAME_STEP, left=True)
            prev_right_img = self.kitti_data.load_gray_image(i - FRAME_STEP, left=False)
            curr_left_img = self.kitti_data.load_gray_image(i, left=True)
            curr_right_img = self.kitti_data.load_gray_image(i, left=False)

            curr_disparity_map = compute_disparity_sgbm(curr_left_img, curr_right_img)
            kets_curr = extract_keypoints(curr_left_img)
            kets_prev = extract_keypoints(prev_left_img)

            landmark_tracker.add_features(i, kets_curr)
            landmark_tracker.add_features(i - FRAME_STEP, kets_prev)

            matches, matches_kpts_prev, matches_kpts_curr = match_keypoints(kets_prev, kets_curr)
            landmark_tracker.add_matches(i - FRAME_STEP, i, matches)

            points_3d = []
            points_2d = []
            feature_index = []
            for pt_prev, pt_curr, match_indexs in zip(matches_kpts_prev, matches_kpts_curr, matches):
                u, v = int(pt_curr[0]), int(pt_curr[1])
                if 0 <= v < curr_disparity_map.shape[0] and 0 <= u < curr_disparity_map.shape[1]:
                    disp = curr_disparity_map[v, u]
                    if disp > 1:
                        Z = fx * baseline / disp
                        X = (pt_curr[0] - cx) * Z / fx  
                        Y = (pt_curr[1] - cy) * Z / fy
                        points_3d.append([X, Y, Z])
                        points_2d.append(pt_prev)
                        feature_index.append(match_indexs[1])

            points_3d = np.array(points_3d, dtype=np.float32)
            points_2d = np.array(points_2d, dtype=np.float32)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d, points_2d,
                cameraMatrix=K,
                distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )


            if not success:
                print(f"Frame {i}: PnP failed")
                poses[i] = poses[i - FRAME_STEP]
                continue

            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.ravel()
            poses[i] = poses[i - FRAME_STEP] @ np.linalg.inv(T)
            current_pose = poses[i]
            current_pose = Rigid3d.from_matrix(current_pose)
            world_points = current_pose.to_matrix33() @ points_3d.T + current_pose.translation.reshape(-1, 1)
            for index, feature_index in enumerate(feature_index):
                landmark_tracker.assigned_points_3d_if_not_values(i, feature_index, world_points[:, index])

        landmark_tracker_statistics = landmark_tracker.get_statistics()

        # draw the statistics
        plt.figure(figsize=(8, 6))
        plt.plot(landmark_tracker_statistics.keys(), landmark_tracker_statistics.values())
        plt.xlabel('track_length')
        plt.ylabel('number_of_track')
        plt.title(f'Landmark Tracker Statistics (KITTI seq {self.kitti_data.seq_id})')
        plt.savefig(f"kitti_tracker_statistics_{self.kitti_data.seq_id}.png")
        plt.close()

    def test_kitti_pose_pgo(self):
        FRAME_STEP = 10
        NUM_FRAMES = len(self.kitti_data.load_kitti_poses())
        # NUM_FRAMES = 300

        K, baseline = self.kitti_data.load_intrinsics()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        pose_graph_solver = PoseGraphSolver()

        pose_graph_solver.add_node(0, Rigid3d.from_matrix(np.eye(4)).to_se3())

        embeddings = {}

        gt_poses = self.kitti_data.load_kitti_poses()
        last_key_frame_index = 0

        accumulated_T = np.eye(4)

        for i in tqdm(range(FRAME_STEP, NUM_FRAMES, FRAME_STEP), desc="Processing frames"):
            prev_left_img = self.kitti_data.load_gray_image(i - FRAME_STEP, left=True)
            prev_right_img = self.kitti_data.load_gray_image(i - FRAME_STEP, left=False)
            curr_left_img = self.kitti_data.load_gray_image(i, left=True)
            curr_right_img = self.kitti_data.load_gray_image(i, left=False)

            curr_disparity_map = compute_disparity_sgbm(curr_left_img, curr_right_img)
            kets_curr = extract_keypoints(curr_left_img)
            kets_prev = extract_keypoints(prev_left_img)
            matches, matches_kpts_prev, matches_kpts_curr = match_keypoints(kets_prev, kets_curr)
            success, T = ComputePose(matches_kpts_prev, matches_kpts_curr, curr_disparity_map, K, baseline)
            if not success:
                print(f"Frame {i}: PnP failed")
                last_pose : Rigid3d = Rigid3d.from_matrix(pose_graph_solver.get_optimized_pose(i - FRAME_STEP).matrix())
                pose_graph_solver.add_node(i, last_pose.to_se3())
                continue
            T = Rigid3d.from_matrix(T)

            last_pose : Rigid3d = Rigid3d.from_matrix(pose_graph_solver.get_optimized_pose(i - FRAME_STEP).matrix())
            current_pose = Rigid3d.from_matrix(last_pose.matrix44() @ np.linalg.inv(T.matrix44()))
            pose_graph_solver.add_node(i, current_pose.to_se3())

            last_key_frame_pose = pose_graph_solver.get_optimized_pose(last_key_frame_index)
            last_key_frame_pose = Rigid3d.from_matrix(last_key_frame_pose.matrix())
            accumulated_T = accumulated_T @ T.matrix44()
            
            print(f"last_key_frame transform from {last_key_frame_index} to {i}")
            pose_graph_solver.add_edge(last_key_frame_index, i, Rigid3d.from_matrix(accumulated_T).to_se3())
            last_key_frame_index = i
            accumulated_T = np.eye(4)

            # extract embeddings
            left_embeddings = get_embeddings(cv2.cvtColor(curr_left_img, cv2.COLOR_GRAY2BGR))
            embeddings[i] = left_embeddings
            max_simularity, max_simularity_index = find_top_similar_embedding(embeddings, i)

            if max_simularity > 0.90:
                print(f"max_simularity: {max_simularity}, max_simularity_index: {max_simularity_index} to {i}")
                similarity_img = self.kitti_data.load_gray_image(max_simularity_index, left=True)
                kpts_similarity = extract_keypoints(similarity_img)
                _, matches_kpts_similarity, matches_kpts_curr = match_keypoints(kpts_similarity, kets_curr)

                success, T = ComputePose(matches_kpts_similarity, matches_kpts_curr, curr_disparity_map, K, baseline)

                T = Rigid3d.from_matrix(T)
                pose_graph_solver.add_edge(max_simularity_index, i, T.to_se3())

                pose_graph_solver.optimize()

        pose_graph_solver.optimize()
        # {id -> pp.SE3}
        optimized_poses = pose_graph_solver.get_all_optimized_poses()
        # print(f"optimized_poses: {optimized_poses}")
        # print(f"[value.matrix() for key, value in optimized_poses.items()]: {[value.matrix().numpy() for key, value in optimized_poses.items()]}")
        optimized_trajectory = np.array([value.matrix().numpy() for key, value in optimized_poses.items()])
        optimized_trajectory = np.array([[pose[0, 3], -pose[2, 3]] for pose in optimized_trajectory])


        gt_trajectory = np.array([[pose[0, 3], pose[2, 3]] for pose in gt_poses])

        #plot_trajectory(trajectory, f"KITTI Estimated Trajectory - Sequence {self.kitti_data.seq_id}", f"kitti_trajectory_{self.kitti_data.seq_id}_estimated.png")
        # === Plot
        # plt.figure(figsize=(8, 6))
        # plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], label='Estimated trajectory',linewidth=2, color='red')
        plt.plot(optimized_trajectory[:, 0], optimized_trajectory[:, 1], label='Optimized trajectory',linewidth=2, color='green')
        plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], label='Ground truth trajectory',linewidth=2, color='blue')
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.title(f'Trajectory (KITTI seq {self.kitti_data.seq_id}, step={FRAME_STEP})')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.savefig(f"kitti_trajectory_{self.kitti_data.seq_id}_pgo.png")
        plt.close()


    def test_kitti_gt_generate_point_cloud(self):
        FRAME_STEP = 10
        NUM_FRAMES = len(self.kitti_data.load_kitti_poses())
        # NUM_FRAMES = 300

        K, baseline = self.kitti_data.load_intrinsics()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        gt_poses = self.kitti_data.load_kitti_poses()


        ply_output = PlyOutput(f"kitti_gt_point_cloud_{self.kitti_data.seq_id}.ply", self.kitti_data.seq_id)

        for i in tqdm(range(0, NUM_FRAMES, FRAME_STEP), desc="Processing frames"):
            curr_left_img = self.kitti_data.load_gray_image(i, left=True)
            curr_right_img = self.kitti_data.load_gray_image(i, left=False)

            curr_disparity_map = compute_disparity_sgbm(curr_left_img, curr_right_img)

            color_point_cloud = []
            pose = Rigid3d.from_matrix(gt_poses[i])
            for v in range(curr_left_img.shape[0]):
                for u in range(curr_left_img.shape[1]):
                    if curr_disparity_map[v, u] > 1:
                        Z = fx * baseline / curr_disparity_map[v, u]
                        X = (u - cx) * Z / fx
                        Y = (v - cy) * Z / fy
                        world_point = pose.to_matrix33() @ np.array([X, Y, Z]) + pose.translation_vector()
                        color_point_cloud.append([world_point[0], world_point[1], world_point[2], curr_left_img[v, u], curr_left_img[v, u], curr_left_img[v, u]])
            ply_output.append_points(color_point_cloud, self.kitti_data.seq_id, i)

        ply_output.output_ply()

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

        ply_output = PlyOutput(f"kitti_yixuan_point_cloud_{self.kitti_data.seq_id}.ply", self.kitti_data.seq_id)

        tree = stereo_slam_pybind.OcTree(0.1)

        print(f"index: {index.shape}")
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
                        tree.insert_ray(pose.translation_vector(), world_point, False)
                        color_point_cloud.append([world_point[0], world_point[1], world_point[2], curr_left_img[v, u], curr_left_img[v, u], curr_left_img[v, u]])
            ply_output.append_points(color_point_cloud, self.kitti_data.seq_id, index[i])

        tree.write_binary("map.bt")
        ply_output.output_ply()
    
    def test_kitti_yixuan_traj_occupancy_map(self):
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

        ply_output = PlyOutput(f"kitti_yixuan_point_cloud_{self.kitti_data.seq_id}.ply", self.kitti_data.seq_id)

        tree = stereo_slam_pybind.OcTree(0.1)

        print(f"index: {index.shape}")
        os.makedirs("tmp", exist_ok=True)
        for i in tqdm(range(index.shape[0]), desc="Processing frames"):
            # print(f"index: {index[i]}")
            # print(f"vio_p_xyz: {vio_p_xyz[i]}")
            # print(f"vio_q_xyzw: {vio_q_xyzw[i]}")

            pose = Rigid3d(translation=vio_p_xyz[i], rotation_quaternion=vio_q_xyzw[i])

            curr_left_img = self.kitti_data.load_gray_image(index[i], left=True)
            curr_right_img = self.kitti_data.load_gray_image(index[i], left=False)

            curr_disparity_map = compute_disparity_sgbm(curr_left_img, curr_right_img)

            points = []
            for v in range(0, curr_left_img.shape[0], 4):
                    for u in range(0, curr_left_img.shape[1], 4):
                        if curr_disparity_map[v, u] > 1:
                            Z = fx * baseline / curr_disparity_map[v, u]
                            if Z > 50:
                                continue
                            X = (u - cx) * Z / fx
                            Y = (v - cy) * Z / fy
                            world_point = pose.to_matrix33() @ np.array([X, Y, Z]) + pose.translation_vector()
                            points.append(world_point)
            # save the points to a file
            with open(f"tmp/points_{i}.txt", "w") as f:
                for point in points:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")

            with Timer(f"occupancy map",f"map {len(points)} points" +"Elapsed time: {:0.4f} seconds"):
                points = np.array(points)
                tree.insertPointCloudRays( pose.translation_vector(), points, False)
        tree.write_binary("map.bt")

    def test_kitti_occupancy_map(self):
        lines=np.loadtxt("traj.csv",dtype=str, delimiter=",", skiprows=1)
        index = np.array(lines[:,0],dtype=int)
        vio_p_xyz = np.array(lines[:,1:4],dtype=float)
        vio_q_xyzw = np.array(lines[:,4:8],dtype=float)
        tree = stereo_slam_pybind.OcTree(0.1)

        print(f"index: {index.shape}")
        os.makedirs("tmp", exist_ok=True)
        for i in tqdm(range(index.shape[0]), desc="Processing frames"):
            points = np.loadtxt(f"tmp/points_{i}.txt")
            points = np.array(points)
            with Timer(f"occupancy map",f"map {len(points)} points" +"Elapsed time: {:0.4f} seconds"):
                tree.insertPointCloud(vio_p_xyz[i], points, False)
        tree.write_binary("map_insert_point_cloud.bt")

    def test_kitti_occupancy_map_super_ray(self):
        lines=np.loadtxt("traj.csv",dtype=str, delimiter=",", skiprows=1)
        index = np.array(lines[:,0],dtype=int)
        vio_p_xyz = np.array(lines[:,1:4],dtype=float)
        vio_q_xyzw = np.array(lines[:,4:8],dtype=float)
        tree = stereo_slam_pybind.SuperRayOcTree(0.1)

        print(f"index: {index.shape}")
        os.makedirs("tmp", exist_ok=True)
        for i in tqdm(range(index.shape[0]), desc="Processing frames"):
            points = np.loadtxt(f"tmp/points_{i}.txt")
            points = np.array(points)
            with Timer(f"occupancy map",f"map {len(points)} points" +"Elapsed time: {:0.4f} seconds"):
                tree.insertSuperRayCloudRays(vio_p_xyz[i], points, False)
        tree.write_binary("map_insert_super_ray_cloud_rays.bt")


    def test_kitti_octomap(self):
        tree = stereo_slam_pybind.OcTree(0.05)
        tree.insert_point(np.array([1.0, 2.0, 0.5]))
        tree.insert_ray(np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.5]))
        print("Tree size:", tree.size())
        tree.write_binary("map.bt")


    def test_kitti_sam(self):
        from segment_anything import SamPredictor, sam_model_registry
        from segment_anything import SamAutomaticMaskGenerator
        sam = sam_model_registry["vit_b"](checkpoint="/home/junlinp/DMWork/stereo_slam/stereo_slam/sam_vit_b_01ec64.pth")
        img = cv2.cvtColor(self.kitti_data.load_gray_image(0, left=True), cv2.COLOR_GRAY2BGR)
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(img)

        for i, mask in enumerate(masks):
            print(f"mask: {mask}")
            mask_img = img.copy()
            mask_img[mask['segmentation']] = 255
            cv2.imwrite(f"mask_sam_{i}.png", mask_img)

    def test_kitti_rerun(self):
        K, baseline = self.kitti_data.load_intrinsics()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        rr.init("rerun_example_my_data", spawn=True)

        lines=np.loadtxt("traj.csv",dtype=str, delimiter=",", skiprows=1)
        index = np.array(lines[:,0],dtype=int)
        vio_p_xyz = np.array(lines[:,1:4],dtype=float)
        vio_q_xyzw = np.array(lines[:,4:8],dtype=float)

        print(f"index: {index.shape}")
        os.makedirs("tmp", exist_ok=True)
        total_points = None
        for i in tqdm(range(index.shape[0]), desc="Processing frames"):

            curr_left_img = self.kitti_data.load_gray_image(index[i], left=True)
            curr_right_img = self.kitti_data.load_gray_image(index[i], left=False)
            curr_disparity_map = compute_disparity_sgbm(curr_left_img, curr_right_img)

            # convert disparity map to depth map
            depth_map = fx * baseline / curr_disparity_map
            img = cv2.cvtColor(curr_left_img, cv2.COLOR_GRAY2BGR)
            points = np.loadtxt(f"tmp/points_{i}.txt")
            points = np.array(points)
            if total_points is None:
                total_points = points
            else:
                total_points = np.concatenate([total_points, points], axis=0)
            rr.set_time("time", sequence=i)
            rr.log("sensor/points", rr.Points3D(points, radii=0.1))
            #rr.log("point_clouds", rr.Points3D(total_points, radii=0.1))
            current_pose = Rigid3d(translation=vio_p_xyz[i], rotation_quaternion=vio_q_xyzw[i])
            rr.log("world/camera", rr.Transform3D(translation=current_pose.translation, rotation=rr.Quaternion(xyzw=current_pose.rotation_quaternion)))
            rr.log("world/camera", rr.Pinhole(resolution=[1226, 370], image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF))
            rr.log("world/camera", rr.Image(img))
            rr.log("camera/trajectory", rr.LineStrips3D(np.array(vio_p_xyz[:i+1]), colors=np.array([[255, 0, 0]])))

    def test_kitti_convert_point_cloud_into_gridmapping(self):
        import math
        tree = stereo_slam_pybind.OcTree(0.1)
        tree.read_binary("map.bt")
        volume_points = tree.search_bounding_box(np.array([-300, -20, -200]), np.array([300, 2, 400]))

        print(f"volume_points: {volume_points.shape}")

        mask = volume_points[:, 3] > 0.93
        volume_points = volume_points[mask]
        print(f"after occupancy filter volume_points: {volume_points.shape}")

        # rotation_transform = Rigid3d(rotation_quaternion=np.array([math.sqrt(2)/2, 0.0, 0.0, math.sqrt(2)/2]), translation=np.array([0.0, 0.0, 0.0]))
        # volume_points[:, :3] = (rotation_transform.to_matrix33() @ volume_points[:, :3].T).T

        positions = volume_points[:, :3]
        positions_meaned = positions - np.mean(positions, axis=0)

        cov_matrix = np.cov(positions_meaned, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
        sorted_idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sorted_idx]
        eig_vecs = eig_vecs[:, sorted_idx]

        print(f"eig_vals: {eig_vals}")
        print(f"eig_vecs: {eig_vecs}")

        X_pca = np.dot(positions_meaned, eig_vecs)

        volume_points[:, :3] = X_pca

        print(f"volume_points: {volume_points.shape}")
        print(f"volume_points: {volume_points[:, 3]}")
        # position, value, size
        z_axis_mask = volume_points[:, 2] < -5
        volume_points = volume_points[z_axis_mask]
        print(f"after z axis filter volume_points: {volume_points.shape}")


        # convert the volume_points to a gridmap along the z axis, the gridmap is [-500, 500] in x, [-200, 400] in y, resolution is 0.1
        gridmap = np.zeros((6001, 6001))
        for point in volume_points:
            x = int(point[0] / 0.1 + 3000)
            y = int(point[1] / 0.1 + 2000)
            gridmap[x, y] = 1

        print(f"gridmap: {gridmap.shape}")

        # save the gridmap to a file
        np.save("gridmap.npy", gridmap)

        # save the gridmap to a png file
        plt.figure(figsize=(10, 10))
        plt.imshow(gridmap, cmap="gray")
        plt.savefig("gridmap_block.png")
        plt.close()
        
    def test_kitti_all(self):
        FRAME_STEP = 10
        rr.init("rerun_example_my_data", spawn=True)
        NUM_FRAMES = len(self.kitti_data.load_kitti_poses())
        # NUM_FRAMES = 300

        K, baseline = self.kitti_data.load_intrinsics()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        pose_graph_solver = PoseGraphSolver()

        pose_graph_solver.add_node(0, Rigid3d.from_matrix(np.eye(4)).to_se3())

        embeddings = {}

        gt_poses = self.kitti_data.load_kitti_poses()
        last_key_frame_index = 0

        accumulated_T = np.eye(4)
        resolution = 0.2
        occupancy_map = stereo_slam_pybind.OcTree(resolution)
        occupancy_map.setProbHit(0.6)
        occupancy_map.setProbMiss(0.4)
        height_map_size = np.array([100, 20.0, 200])


        for i in tqdm(range(FRAME_STEP, NUM_FRAMES, FRAME_STEP), desc="Processing frames"):
            prev_left_img = self.kitti_data.load_gray_image(i - FRAME_STEP, left=True)
            prev_right_img = self.kitti_data.load_gray_image(i - FRAME_STEP, left=False)
            curr_left_img = self.kitti_data.load_gray_image(i, left=True)
            curr_right_img = self.kitti_data.load_gray_image(i, left=False)

            curr_disparity_map = compute_disparity_sgbm(curr_left_img, curr_right_img)
            kets_curr = extract_keypoints(curr_left_img)
            kets_prev = extract_keypoints(prev_left_img)
            matches, matches_kpts_prev, matches_kpts_curr = match_keypoints(kets_prev, kets_curr)
            success, T = ComputePose(matches_kpts_prev, matches_kpts_curr, curr_disparity_map, K, baseline)
            if not success:
                print(f"Frame {i}: PnP failed")
                last_pose : Rigid3d = Rigid3d.from_matrix(pose_graph_solver.get_optimized_pose(i - FRAME_STEP).matrix())
                pose_graph_solver.add_node(i, last_pose.to_se3())
                continue
            # T from curr to prev
            T = Rigid3d.from_matrix(T)

            last_pose : Rigid3d = Rigid3d.from_matrix(pose_graph_solver.get_optimized_pose(i - FRAME_STEP).matrix())
            current_pose = Rigid3d.from_matrix(last_pose.matrix44() @ T.matrix44())
            pose_graph_solver.add_node(i, current_pose.to_se3())

            last_key_frame_pose = pose_graph_solver.get_optimized_pose(last_key_frame_index)
            last_key_frame_pose = Rigid3d.from_matrix(last_key_frame_pose.matrix())
            accumulated_T = accumulated_T @ np.linalg.inv(T.matrix44())
            
            print(f"last_key_frame transform from {last_key_frame_index} to {i}")
            pose_graph_solver.add_edge(last_key_frame_index, i, Rigid3d.from_matrix(accumulated_T).to_se3())
            last_key_frame_index = i
            accumulated_T = np.eye(4)

            # extract embeddings
            left_embeddings = get_embeddings(cv2.cvtColor(curr_left_img, cv2.COLOR_GRAY2BGR))
            embeddings[i] = left_embeddings
            max_simularity, max_simularity_index = find_top_similar_embedding(embeddings, i)

            if max_simularity > 0.90:
                print(f"max_simularity: {max_simularity}, max_simularity_index: {max_simularity_index} to {i}")
                similarity_img = self.kitti_data.load_gray_image(max_simularity_index, left=True)
                kpts_similarity = extract_keypoints(similarity_img)
                _, matches_kpts_similarity, matches_kpts_curr = match_keypoints(kpts_similarity, kets_curr)

                success, T = ComputePose(matches_kpts_similarity, matches_kpts_curr, curr_disparity_map, K, baseline)

                T = Rigid3d.from_matrix(T)
                pose_graph_solver.add_edge(i, max_simularity_index, T.to_se3())

                pose_graph_solver.optimize()
            sky_mask = segment_sky_mask(curr_left_img)
            # depth to occupancy map
            depth = np.zeros_like(curr_disparity_map)
            depth[curr_disparity_map > 0] = fx * baseline / (curr_disparity_map[curr_disparity_map > 0])
            depth[depth > 50] = 0
            with Timer(f"depth to points in global",f"depth to points in global" +"Elapsed time: {:0.4f} seconds"):
                cloud_points_in_global = depth_to_points_in_global(depth, current_pose, fx, fy, cx, cy, sky_mask)

            # cloud_points_in_global = filter_cloud_points_by_sky(cloud_points_in_global)

            with Timer(f"insert points into occupancy map",f"insert points into occupancy map" +"Elapsed time: {:0.4f} seconds"):
                occupancy_map.insertPointCloudRays(current_pose.translation_vector(), cloud_points_in_global, False)

            height_map, height_map_origin = occupancy_grid_to_height_map(occupancy_map, current_pose.translation_vector(), height_map_size, resolution, 0.93)

            height_map = max_pooling_height_map(height_map)

            trajectories_in_camera, params = generate_trajectory_library(num_samples=20, duration=10.0, dt=0.1, acc_std=0.1, omega_std_deg=1.0)

            trajectory_in_global = []
            for traj in trajectories_in_camera:
                x_in_camera = traj[:, 1]
                z_in_camera = traj[:, 0]
                y_in_camera = np.zeros_like(x_in_camera)
                points_in_camera = np.stack([x_in_camera, y_in_camera, z_in_camera], axis=1)
                points_in_global = ((current_pose.to_matrix33() @ points_in_camera.T).T + current_pose.translation_vector())
                trajectory_in_global.append(points_in_global)
            trajectory_in_global = np.array(trajectory_in_global)

            scores = score_trajectories_by_height_map(trajectories_in_camera, height_map, height_map_origin, resolution, 1.0)

            trajectory_colors = [scalar_to_rgb(score, vmin=0, vmax=np.max(scores)) for score in scores]





            print(f"current_pose.translation_vector(): {current_pose.translation_vector()}")
            print(f"current_pose.rotation_quaternion(): {current_pose.rotation_quaternion}")

            rr.set_time("time", sequence=i)
            rr.log("sensor/points", rr.Points3D(cloud_points_in_global, radii=0.1))
            rr.log("world/camera", rr.Transform3D(translation=current_pose.translation_vector(), rotation=rr.Quaternion(xyzw=current_pose.rotation_quaternion)))
            rr.log("world/camera", rr.Pinhole(resolution=[1226, 370], image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF))
            rr.log("world/camera", rr.Image(cv2.cvtColor(curr_left_img, cv2.COLOR_GRAY2BGR)))
            #rr.log("camera/trajectory", rr.LineStrips3D(np.array(vio_p_xyz[:i+1]), colors=np.array([[255, 0, 0]])))
            rr.log("camera/local_plan", rr.LineStrips3D(trajectory_in_global, colors=np.array(trajectory_colors)))
            rr.log("world/depth", rr.Image(depth))
        
            # height map

            heights_map_centers = []
            heights_map_half_sizes = []
            heights_map_colors = []
            prior_floor_y_value = 2.0
            for x in range(height_map.shape[0]):
                for z in range(height_map.shape[1]):
                    if not np.isnan(height_map[x, z]):
                        heights_map_centers.append(np.array([height_map_origin[0] + x * resolution, (prior_floor_y_value + height_map[x, z]) * 0.5, height_map_origin[2] + z * resolution]))
                        heights_map_half_sizes.append(np.array([resolution, (prior_floor_y_value - height_map[x, z]) * 0.5, resolution]))
                        # map height to color
                        heights_map_colors.append(np.array([np.abs(height_map[x, z]) * 10, 0, 0]))
            heights_map_centers = np.array(heights_map_centers)
            heights_map_half_sizes = np.array(heights_map_half_sizes)
            heights_map_colors = np.array(heights_map_colors)
            print(f"heights_map_centers: {heights_map_centers.shape}")
            print(f"heights_map_half_sizes: {heights_map_half_sizes.shape}")
            print(f"heights_map_colors: {heights_map_colors.shape}")
            rr.log("world/height_map", rr.Boxes3D(centers=heights_map_centers, half_sizes=heights_map_half_sizes, colors=heights_map_colors, fill_mode="solid"))

        pose_graph_solver.optimize()
        # {id -> pp.SE3}
        optimized_poses = pose_graph_solver.get_all_optimized_poses()
        # print(f"optimized_poses: {optimized_poses}")
        # print(f"[value.matrix() for key, value in optimized_poses.items()]: {[value.matrix().numpy() for key, value in optimized_poses.items()]}")
        optimized_trajectory = np.array([value.matrix().numpy() for key, value in optimized_poses.items()])
        optimized_trajectory = np.array([[pose[0, 3], -pose[2, 3]] for pose in optimized_trajectory])


        gt_trajectory = np.array([[pose[0, 3], pose[2, 3]] for pose in gt_poses])

        #plot_trajectory(trajectory, f"KITTI Estimated Trajectory - Sequence {self.kitti_data.seq_id}", f"kitti_trajectory_{self.kitti_data.seq_id}_estimated.png")
        # === Plot
        # plt.figure(figsize=(8, 6))
        # plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], label='Estimated trajectory',linewidth=2, color='red')
        plt.plot(optimized_trajectory[:, 0], optimized_trajectory[:, 1], label='Optimized trajectory',linewidth=2, color='green')
        plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], label='Ground truth trajectory',linewidth=2, color='blue')
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.title(f'Trajectory (KITTI seq {self.kitti_data.seq_id}, step={FRAME_STEP})')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.savefig(f"kitti_trajectory_{self.kitti_data.seq_id}_pgo.png")
        plt.close()

    def test_kitti_segformer(self):
        from transformers import SegformerForSemanticSegmentation
        from transformers import AutoImageProcessor
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        img = cv2.cvtColor(self.kitti_data.load_gray_image(10, left=True), cv2.COLOR_GRAY2BGR)
        inputs = processor(img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"img.shape: {img.shape}")
        predicted_segmentation_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[img.shape[0:2]])[0]
        predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
        print(f"predicted_segmentation_map: {predicted_segmentation_map.shape}")
        print(f"predicted_segmentation_map: {predicted_segmentation_map}")
        print(f"predicted_segmentation_map: {predicted_segmentation_map.max()}")
        print(f"predicted_segmentation_map: {predicted_segmentation_map.min()}")
        ade_palette = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]
        
        color_predicted_segmentation_map = np.zeros((predicted_segmentation_map.shape[0], predicted_segmentation_map.shape[1], 3))
        for i in range(predicted_segmentation_map.shape[0]):
            for j in range(predicted_segmentation_map.shape[1]):
                color_predicted_segmentation_map[i, j] = np.array(ade_palette[predicted_segmentation_map[i, j]], dtype=np.uint8)

        color_predicted_segmentation_map = color_predicted_segmentation_map.astype(np.uint8)
        print(f"color_predicted_segmentation_map: {color_predicted_segmentation_map.shape}")
        print(f"color_predicted_segmentation_map: {color_predicted_segmentation_map}")
        # segment sky mask
        sky_mask = predicted_segmentation_map==2

        segmented_img = img.copy()
        segmented_img[sky_mask] = 0
        # save the segmented_img to a png file
        cv2.imwrite("segmented_img.png", segmented_img)
        # also show the original image in the same figure
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(color_predicted_segmentation_map)
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.savefig("predicted_segmentation_map.png")
        plt.close()
        
        

if __name__ == "__main__":
    unittest.main()