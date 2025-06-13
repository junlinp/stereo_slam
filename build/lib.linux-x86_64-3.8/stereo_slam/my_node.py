import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from collections import deque
import numpy as np
import cv2
import torch
#import matplotlib.pyplot as plt
from codetiming import Timer
from scipy.spatial.transform import Rotation as R
import einops
from .landmark_tracker import LandmarkTracker
from .bundle_adjustment import BundleAdjustment
from .rigid3d import Rigid3d

logger = None

# SuperPoint+LightGlue
device = "cuda" if torch.cuda.is_available() else "cpu"



def create_pointcloud2(points: np.ndarray, frame_id="map") -> PointCloud2:
    """Creates a PointCloud2 message from a Nx3 numpy array."""
    assert points.shape[1] == 3, "Points array must be Nx3"

    header = Header()
    header.stamp = rclpy.time.Time().to_msg()
    header.frame_id = frame_id

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    # Flatten the point array to bytes
    data = points.astype(np.float32).tobytes()

    cloud_msg = PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        fields=fields,
        is_bigendian=False,
        point_step=12,
        row_step=12 * points.shape[0],
        is_dense=True,
        data=data
    )
    return cloud_msg









def stereo_rectify(cam0_image, cam1_image, cam0_intrinsics, cam1_intrinsics, cam0_distortion, cam1_distortion,cam0_extrinsics, cam1_extrinsics):
    # extrinsics from cam0 to cam1
    T_cam0_cam1 = cam0_extrinsics * cam1_extrinsics.inverse()
    R = T_cam0_cam1.to_matrix33()
    T = T_cam0_cam1.translation_vector()

    image_size = (cam0_image.shape[1], cam0_image.shape[0])
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=cam0_intrinsics, distCoeffs1=cam0_distortion, cameraMatrix2=cam1_intrinsics, distCoeffs2=cam1_distortion, imageSize=image_size, R=R, T=T)

    map1x, map1y = cv2.initUndistortRectifyMap(cam0_intrinsics, cam0_distortion, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(cam1_intrinsics, cam1_distortion, R2, P2, image_size, cv2.CV_32FC1)

    rectified_cam0_image = cv2.remap(cam0_image, map1x, map1y, cv2.INTER_LINEAR)
    rectified_cam1_image = cv2.remap(cam1_image, map2x, map2y, cv2.INTER_LINEAR)

    return rectified_cam0_image, rectified_cam1_image


class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('Node started!')
        
        # Create queues to store recent images
        self.cam0_queue = deque(maxlen=10)
        self.cam1_queue = deque(maxlen=10)
        
        # Create subscriptions
        self.create_subscription(Image, 'cam0/image_raw', self.image_callback_cam0, 10)
        self.create_subscription(Image, 'cam1/image_raw', self.image_callback_cam1, 10)
        self.create_subscription(CameraInfo, 'cam0/camera_info', self.camera_info_callback_cam0, 10)
        self.create_subscription(CameraInfo, 'cam1/camera_info', self.camera_info_callback_cam1, 10)
        self.create_subscription(PoseStamped, 'cam0_extrinsics', self.extrinsics_callback_cam0, 10)
        self.create_subscription(PoseStamped, 'cam1_extrinsics', self.extrinsics_callback_cam1, 10)
        self.pose_publisher = self.create_publisher(PoseStamped, 'estimated_pose', 10)
        self.simularity_publisher = self.create_publisher(Image, 'simularity', 10)
        self.match_publisher = self.create_publisher(Image, 'match', 10)
        self.point_cloud_publisher = self.create_publisher(PointCloud2, 'point_cloud', 10)

        self.bridge = CvBridge()
        self.cam0_extrinsics = None
        self.cam1_extrinsics = None
        self.cam0_intrinsics = None
        self.cam0_distortion = None
        self.cam1_intrinsics = None
        self.cam1_distortion = None

        self.poses = {}
        self.last_cam0_image = None
        self.last_cam0_timestamp = None
        self.last_cam1_image = None


        self.cam0_embeddings = {}
        self.cam0_gray_images = {}


        self.landmark_tracker = LandmarkTracker()
        self.bundle_adjustment = BundleAdjustment()


    def image_callback_cam0(self, msg):
        self.cam0_queue.append(msg)
        self.try_match_images()
        
    def image_callback_cam1(self, msg):
        self.cam1_queue.append(msg)
        self.try_match_images()

    def extrinsics_callback_cam0(self, msg):

        rotation = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        translation = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.cam0_extrinsics = Rigid3d.from_vector(np.concatenate([translation, rotation.as_quat()]))

    def extrinsics_callback_cam1(self, msg):
        rotation = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        translation = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.cam1_extrinsics = Rigid3d.from_vector(np.concatenate([translation, rotation.as_quat()]))
        
    def try_match_images(self):
        if not self.cam0_queue or not self.cam1_queue:
            return
            
        # Get the most recent images
        cam0_msg = self.cam0_queue[0]
        cam1_msg = self.cam1_queue[0]

        cam0_timestamp = cam0_msg.header.stamp.sec * 1e9 + cam0_msg.header.stamp.nanosec
        cam1_timestamp = cam1_msg.header.stamp.sec * 1e9 + cam1_msg.header.stamp.nanosec
        self.get_logger().info(f'cam0_timestamp: {cam0_timestamp}, cam1_timestamp: {cam1_timestamp}')
        # Compare timestamps
        time_diff = abs(cam0_timestamp - cam1_timestamp)
        
        # If timestamps are close enough (within 0.1 seconds)
        if time_diff < 0.01:
            # Process and publish the matched image pair
            self.process_stereo_pair(cam0_msg, cam1_msg)
            # Remove the processed images from queues
            self.cam0_queue.popleft()
            self.cam1_queue.popleft()
        else:
            if cam0_timestamp > cam1_timestamp:
                self.cam1_queue.popleft()
            else:
                self.cam0_queue.popleft()
            
    def process_stereo_pair(self, cam0_msg, cam1_msg):
        # Here you can implement your stereo image processing
        # For now, we'll just log that we found a match
        self.get_logger().info('Found matching stereo pair!')
        # TODO: Add your stereo processing logic here
        # You might want to convert the images to numpy arrays, process them,
        # and publish the results
        timestamp = int(cam0_msg.header.stamp.sec * 1000000000 + cam0_msg.header.stamp.nanosec)

        if self.cam0_intrinsics is None or self.cam0_distortion is None or self.cam1_intrinsics is None or self.cam1_distortion is None:
            self.get_logger().info("Camera info not available")
            return

        if self.last_cam0_image is None or self.last_cam1_image is None:
            self.last_cam0_image = self.bridge.imgmsg_to_cv2(cam0_msg, "bgr8")
            self.last_cam1_image = self.bridge.imgmsg_to_cv2(cam1_msg, "bgr8")
            last_cam0_image_keypoints = extract_keypoints(self.last_cam0_image)
            self.landmark_tracker.add_features(timestamp, last_cam0_image_keypoints)
            self.last_cam0_timestamp = timestamp
            self.get_logger().info("No images available")
            return

        # get the image from Image message
        cam0_curr_image = self.bridge.imgmsg_to_cv2(cam0_msg, "bgr8")
        cam1_curr_image = self.bridge.imgmsg_to_cv2(cam1_msg, "bgr8")
        self.publish_simularity(timestamp, cam0_curr_image)

        # undistort the images
        cam0_curr_image = self.undistort_image(cam0_curr_image, self.cam0_intrinsics, self.cam0_distortion)
        cam1_curr_image = self.undistort_image(cam1_curr_image, self.cam1_intrinsics, self.cam1_distortion)
        rectified_cam0_image, rectified_cam1_image = stereo_rectify(cam0_curr_image, cam1_curr_image, self.cam0_intrinsics, self.cam1_intrinsics, self.cam0_distortion, self.cam1_distortion, self.cam0_extrinsics, self.cam1_extrinsics)

        disparity = compute_disparity_sgbm(rectified_cam0_image, rectified_cam1_image)

        # extract keypoints 
        features_prev = self.landmark_tracker.get_features(self.last_cam0_timestamp)
        features_curr = extract_keypoints(cam0_curr_image)

        # add keypoints to landmark tracker
        self.landmark_tracker.add_features(timestamp, features_curr)
        # match keypoints
        matches = match_keypoints(features_prev, features_curr)
        kpts_prev = features_prev['keypoints'].squeeze(0)[matches[..., 0]]
        kpts_curr = features_curr['keypoints'].squeeze(0)[matches[..., 1]]
        self.landmark_tracker.add_matches(self.last_cam0_timestamp, timestamp, matches.cpu().numpy())
        valid_landmark_ids = self.landmark_tracker.get_valid_landmark_ids()

        # draw kpts_prev and kpts_curr match
        cv_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0) for i in range(kpts_prev.shape[0])]
        # convert kpts_prev and kpts_curr to cv2.KeyPoint 
        cv_kpts_prev = [cv2.KeyPoint(x=kpts_prev[index, 0].item(), y=kpts_prev[index, 1].item(), size=20) for index in range(kpts_prev.shape[0])]
        cv_kpts_curr = [cv2.KeyPoint(x=kpts_curr[index, 0].item(), y=kpts_curr[index, 1].item(), size=20) for index in range(kpts_curr.shape[0])]

        output_image = cv2.drawMatches(self.last_cam0_image, cv_kpts_prev, cam0_curr_image, cv_kpts_curr, cv_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        image_msg = self.bridge.cv2_to_imgmsg(output_image, "bgr8")
        image_msg.header.stamp.sec = int(timestamp // 1000000000)
        image_msg.header.stamp.nanosec = int(timestamp % 1000000000)
        self.match_publisher.publish(image_msg)

        T_cam0_cam1 = self.cam0_extrinsics * self.cam1_extrinsics.inverse()
        cam0_cam1_rotation = T_cam0_cam1.to_matrix33()
        cam0_cam1_translation = T_cam0_cam1.translation_vector()
        baseline = np.linalg.norm(cam0_cam1_translation)
        points_3d, points_2d = [], []
        points_index = []
        points_index_of_prev = [matches[i, 0].item() for i in range(matches.shape[0])]

        for pt_prev, pt_curr,index in zip(kpts_prev.cpu().numpy(), kpts_curr.cpu().numpy(), points_index_of_prev):
            u, v = int(pt_curr[0]), int(pt_curr[1])
            if 0 <= v < disparity.shape[0] and 0 <= u < disparity.shape[1]:
                disp = disparity[v, u]
                if disp > 1:
                    Z = self.cam0_intrinsics[0, 0] * baseline / disp
                    X = (pt_curr[0] - self.cam0_intrinsics[0, 2]) * Z / self.cam0_intrinsics[0, 0]
                    Y = (pt_curr[1] - self.cam0_intrinsics[1, 2]) * Z / self.cam0_intrinsics[1, 1]
                    points_3d.append([X, Y, Z])
                    points_2d.append(pt_prev)
                    points_index.append(index)



        if len(points_3d) < 6:
            self.get_logger().info(f"Frame {timestamp}: not enough correspondences")
            return
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)

        # cam_prev_pose_in_cam_curr
        success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, self.cam0_intrinsics, None, 
                                                         iterationsCount=100, reprojectionError=8.0, 
                                                         confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            self.get_logger().info(f"Frame {timestamp}: PnP failed")
            return
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        rotation_quaternion = R.from_matrix(rotation_matrix).as_quat()
        translation = tvec.ravel()


        if len(self.poses) == 0:
            self.poses[self.last_cam0_timestamp] = Rigid3d.identity()
        self.poses[timestamp] = self.poses[self.last_cam0_timestamp] * Rigid3d.from_vector(np.concatenate([translation, rotation_quaternion])).inverse()

        for index, point_3d in zip(points_index, points_3d):
            point_in_cam_curr =  np.array(point_3d)
            cam_curr_in_world = self.poses[timestamp]
            point_in_world = cam_curr_in_world.to_matrix33() @ point_in_cam_curr + cam_curr_in_world.translation_vector()
            self.landmark_tracker.assigned_points_3d_if_not_values(self.last_cam0_timestamp, index, np.array(point_in_world))

        self.last_cam0_image = cam0_curr_image
        self.last_cam1_image = cam1_curr_image
        assert timestamp > self.last_cam0_timestamp
        self.last_cam0_timestamp = timestamp

        pose_msg = PoseStamped()
        pose_msg.header.stamp = cam0_msg.header.stamp
        pose_msg.header.frame_id = "world"

        current_pose = self.poses[timestamp]
        current_body_pose = current_pose * self.cam0_extrinsics.inverse()
        current_body_pose_vector = current_body_pose.to_vector()
        pose_msg.pose.position.x = current_body_pose_vector[0]
        pose_msg.pose.position.y = current_body_pose_vector[1]
        pose_msg.pose.position.z = current_body_pose_vector[2]
        pose_msg.pose.orientation.x = current_body_pose_vector[3]
        pose_msg.pose.orientation.y = current_body_pose_vector[4]
        pose_msg.pose.orientation.z = current_body_pose_vector[5]
        pose_msg.pose.orientation.w = current_body_pose_vector[6]
        self.pose_publisher.publish(pose_msg)


        with Timer(text="[bundle_adjustment] Elapsed time: {milliseconds:.0f} ms"):
            timestamp_to_camera_index = {}
            camera_poses = []

            print(f"self.poses: {len(self.poses)}")
            for index, timestamp in enumerate(self.poses.keys()):
                timestamp_to_camera_index[timestamp] = index
                camera_poses.append(self.poses[timestamp].to_vector())
            camera_poses = np.array(camera_poses, dtype=np.float32)
            projection_relations, point_3d_in_world, landmark_id_to_index = self.landmark_tracker.get_projection_relations_and_landmark_position(timestamp_to_camera_index)

            optimized_camera_poses, optimized_point_3d_in_world = self.bundle_adjustment.optimize(camera_poses, point_3d_in_world, projection_relations)

            # update optimized_camera_poses to self.poses
            for index, timestamp in enumerate(self.poses.keys()):
                self.poses[timestamp] = Rigid3d.from_vector(optimized_camera_poses[index])

            # update optimized_point_3d_in_world to self.landmark_tracker
            self.landmark_tracker.update_landmark_positions(optimized_point_3d_in_world, landmark_id_to_index)

        updated_point_3d_in_world =  self.landmark_tracker.get_landmark_positions()

        # publish point_3d_in_world
        msg = create_pointcloud2(updated_point_3d_in_world, frame_id="world")
        self.point_cloud_publisher.publish(msg)

    def undistort_image(self, image, intrinsics, distortion):
        #self.get_logger().info(f"Undistorting image image shape: {image.shape}")
        #self.get_logger().info(f"Undistorting image intrinsics shape: {intrinsics.shape}")
        #self.get_logger().info(f"Undistorting image distortion shape: {distortion.shape}")
        # Convert image to numpy array
        #image_array = #np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        image_array = image
        
        # Apply distortion correction
        undistorted_image = cv2.undistort(image_array, intrinsics, distortion)
        return undistorted_image

    def camera_info_callback_cam0(self, msg):
        self.cam0_info = msg
        self.cam0_intrinsics = np.array(msg.k).reshape(3, 3)
        self.cam0_distortion = np.array(msg.d)
        # self.get_logger().info(f"Cam0 intrinsics: {self.cam0_intrinsics}")
        self.bundle_adjustment.update_camera_intrinsics(self.cam0_intrinsics)

    def camera_info_callback_cam1(self, msg):
        self.cam1_info = msg
        self.cam1_intrinsics = np.array(msg.k).reshape(3, 3)
        self.cam1_distortion = np.array(msg.d)
        # self.get_logger().info(f"Cam1 intrinsics: {self.cam1_intrinsics}")

    def publish_simularity(self,  timestamp, cam0_image):
        cam0_embedding = get_embeddings(cam0_image)
        
        if len(self.cam0_embeddings) == 0:
            self.cam0_embeddings[timestamp] = cam0_embedding
            self.cam0_gray_images[timestamp] = cam0_image
            return

        best_timestamp = None 
        best_simularity = 0
        for timestamp, embedding in self.cam0_embeddings.items():
            simularity = np.dot(cam0_embedding, embedding) / (np.linalg.norm(cam0_embedding) * np.linalg.norm(embedding))
            if best_timestamp == None:
                best_timestamp = timestamp
                best_simularity = simularity
            elif simularity > best_simularity:
                best_timestamp = timestamp
                best_simularity = simularity
        
        # find the image of cam0 at best_timestamp
        cam0_gray_image = self.cam0_gray_images[best_timestamp]

        # draw a image of cv, left is cam0_image, right is cam0_gray_image
        image = np.concatenate((cam0_image, cam0_gray_image), axis=1)
        image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        image_msg.header.stamp.sec = int(timestamp // 1000000000)
        image_msg.header.stamp.nanosec = int(timestamp % 1000000000)
        self.simularity_publisher.publish(image_msg)

        self.cam0_embeddings[timestamp] = cam0_embedding
        self.cam0_gray_images[timestamp] = cam0_image

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
