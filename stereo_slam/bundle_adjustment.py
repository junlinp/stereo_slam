
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
import math
from tqdm import tqdm
from rigid3d import Rigid3d

class BundleAdjustment:
    def __init__(self):
        self.camera_intrinsics = None


    def update_camera_intrinsics(self, camera_intrinsics: np.ndarray):
        self.camera_intrinsics = torch.tensor(camera_intrinsics, dtype=torch.float64)
        self.camera_intrinsics.requires_grad = False

    def optimize(self, camera_poses: np.ndarray, landmark_positions: np.ndarray, projection_relations: list, max_iterations: int = 100, lr: float = 0.001):
        if self.camera_intrinsics is None:
            raise ValueError("camera_intrinsics is not set")
        # set the first pose to be constant
        const_camera_pose = torch.tensor(camera_poses[0, :], dtype=torch.float64, requires_grad=False).unsqueeze(0)
        optimizable_camera_poses = torch.tensor(camera_poses[1:, :], dtype=torch.float64, requires_grad=True)

        torch_landmark_positions = torch.tensor(landmark_positions, dtype=torch.float64, requires_grad=True)

        optimizer = torch.optim.Adam([optimizable_camera_poses, torch_landmark_positions], lr=lr)

        data_loader = DataLoader(projection_relations, batch_size=2048, shuffle=False)
        iterator = iter(data_loader)
        for i in range(max_iterations):
            final_loss = 0
            # # check if the iterator is empty
            # try:
            #     batch_projection_relations = next(iterator)
            # except StopIteration:
            #     iterator = iter(data_loader)
            #     batch_projection_relations = next(iterator)

            # gradient accumulation over the whole dataset
            optimizer.zero_grad()
            for batch_projection_relations in tqdm(data_loader, desc="Processing batch"):  
                #print(f"batch_projection_relations: {(batch_projection_relations)}")
                losses = self.forward(const_camera_pose, optimizable_camera_poses, torch_landmark_positions, batch_projection_relations)
                losses.backward()
                final_loss += losses.item()
            optimizer.step()
            print(f"final_loss: {final_loss / len(data_loader)} at iteration {i}")

        print(f"camera_poses: {camera_poses[1, :]}")
        print(f"optimizable_camera_poses: {optimizable_camera_poses.detach().numpy()[0, :]}")
        torch_camera_poses = torch.cat([const_camera_pose, optimizable_camera_poses], dim=0)
        return torch_camera_poses.detach().numpy(), torch_landmark_positions.detach().numpy()


    def forward(self, const_camera_pose: torch.Tensor, optimizable_camera_poses: torch.Tensor, landmark_positions: torch.Tensor, projection_relations: list) -> torch.Tensor:
        # x, y, z, qx, qy, qz, qw
        # camera_pose: (N, 3 + 4)
        # landmark_positions: (N, 3)
        # projection_relations: list of (camera_id, landmark_id, projection_2d)
        loss = 0
        camera_ids, landmark_ids, projection_2ds = projection_relations
        camera_poses = torch.cat([const_camera_pose, optimizable_camera_poses], dim=0)

        for camera_id, landmark_id, projection_2d in zip(camera_ids, landmark_ids, projection_2ds):
            camera_pose = camera_poses[camera_id, :]
            landmark_position = landmark_positions[landmark_id]
            projection_2d_const = torch.tensor(projection_2d, dtype=torch.float64, requires_grad=False)
            projection_2d_pred = self.project(camera_pose, landmark_position)
            # print(f"projection_2d_const: {projection_2d_const}, projection_2d_pred: {projection_2d_pred}")
            loss += torch.sum(projection_2d_const - projection_2d_pred) ** 2
        return loss / len(projection_relations)


    def project(self, camera_pose: torch.Tensor, landmark_position: torch.Tensor) -> torch.Tensor:
        # camera_pose: (3 + 4) pose_in_world
        # landmark_position: (3) landmark_in_world
        # projection_2d: (2)
        # camera_intrinsics: (3, 3)
        # camera_distortion: (4)

        translation = camera_pose[0:3]
        rotation_quaterions = camera_pose[3:7]
        rotation_matrix = self.quaternion_to_rotation_matrix(rotation_quaterions)

        point_in_camera = rotation_matrix.T @ (landmark_position - translation)
        point_in_camera_normalized = point_in_camera  / point_in_camera[2]
        projection_2d = self.camera_intrinsics @ point_in_camera_normalized
        return projection_2d[:2]

    
    def quaternion_to_rotation_matrix(self, quaternion: torch.Tensor) -> torch.Tensor:
        # quaternion: (4)
        # rotation_matrix: (3, 3)
        quaternion = quaternion / quaternion.norm()
        # print(f"quaternion: {quaternion}")
        return quaternion_to_matrix(quaternion).to(torch.float64)



import unittest
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qnorm

def project_point(K, q, t, point3D):
    """Projects a 3D point into a camera using K, quaternion q, and translation t."""
    R = quat2mat(q / np.linalg.norm(q))
    p_cam = R.T @ (point3D - t)
    proj = K @ p_cam
    return proj[:2] / proj[2]

class TestBundleAdjustment(unittest.TestCase):
    def test_project(self):
        # Camera intrinsics
        K = np.array([[800, 0, 320],
                      [0, 800, 240],
                      [0,   0,   1]])

        # Ground-truth camera poses
        q1 = np.array([0, 0, 0, 1])  # identity quaternion
        t1 = np.array([0.0, 0.0, 0.0])

        angle_axis = np.array([0.0, 0.1, 0.0])
        theta = np.linalg.norm(angle_axis)
        axis = angle_axis / theta
        q2 = np.hstack([ axis * np.sin(theta/2), np.cos(theta/2)])  # small Y-axis rotation
        t2 = np.array([0.5, 0.0, 0.1])

        # Generate 5 3D points
        points_3d_true = np.array([
            [0.5, 0.5, 5.0],
            [-0.5, 0.5, 5.2],
            [0.0, -0.5, 4.8],
            [0.2, 0.0, 5.1],
            [-0.3, -0.4, 4.9],
        ])

        n_points = len(points_3d_true)

        # Project points into both cameras
        points_2d = []
        cam_indices = []
        point_indices = []

        for i, point in enumerate(points_3d_true):
            for cam_id, (q, t) in enumerate([(q1, t1), (q2, t2)]):
                proj = project_point(K, q, t, point)
                #proj += np.random.normal(0, 0.5, size=2)  # Add noise
                points_2d.append(proj)
                cam_indices.append(cam_id)
                point_indices.append(i)

        points_3d_init = points_3d_true  + np.random.normal(0, 0.01, size=points_3d_true.shape)

        # Constant (non-learnable) first pose
        pose = np.concatenate([np.concatenate([t1, q1]), np.concatenate([t2, q2])]).reshape(-1, 7)

        landmark_positions = torch.tensor(points_3d_init, requires_grad=True)

        projection_relations = list(zip(cam_indices, point_indices, points_2d))

        bundle_adjustment = BundleAdjustment()
        bundle_adjustment.update_camera_intrinsics(K)

        camera_poses_estimated, landmark_positions_estimated = bundle_adjustment.optimize(pose, landmark_positions, projection_relations, max_iterations=1024, lr=0.001)


        camera_poses_estimated = [Rigid3d(translation=camera_poses_estimated[i, 0:3], rotation_quaternion=camera_poses_estimated[i, 3:7]) for i in range(camera_poses_estimated.shape[0])]
        # compute the loss compute to the ground truth
         
        # loss to t1
        t1_loss = np.linalg.norm(camera_poses_estimated[0].translation_vector() - t1)
        print(f"t1_loss: {t1_loss}")

        # loss to q1
        q1_loss = math.acos(np.clip(camera_poses_estimated[0].rotation_quaternion[3], -1.0, 1.0)) * 2.0
        print(f"q1_loss: {q1_loss}")

        # loss to t2
        t2_loss = np.linalg.norm(camera_poses_estimated[1].translation_vector() - t2)
        print(f"t2_loss: {t2_loss}")

        def quaternion_conjugate(q):
            return torch.tensor([-q[0], -q[1], -q[2], q[3]])

        def quaternion_multiply(q1, q2):
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            return torch.tensor([x, y, z, w])
        # loss to q2
        estimated_q2_conjugate = quaternion_conjugate(camera_poses_estimated[1].rotation_quaternion)
        q2_diff = quaternion_multiply(estimated_q2_conjugate,q2) 
        q2_loss = math.acos(np.clip(torch.abs(q2_diff[3]).item(), -1.0, 1.0)) * 2.0
        print(f"q2_loss: {q2_loss}")

        # loss to landmark positions
        landmark_positions_loss = np.linalg.norm(landmark_positions_estimated - points_3d_true, axis=1).mean()
        print(f"landmark_positions_loss: {landmark_positions_loss}")
        print(f"landmark_positions_estimated: {landmark_positions_estimated}")
        print(f"points_3d_true: {points_3d_true}")
        print(f"landmark_position_init: {points_3d_init}")

if __name__ == "__main__":
    unittest.main()
