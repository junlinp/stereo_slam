import rclpy
from rclpy.node import Node
import os
import numpy as np
import yaml
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point, Quaternion


class EurocDataNode(Node):
    def __init__(self, dataset_path):
        super().__init__('euroc_data_node')
        self.get_logger().info('Euroc data node started!')
        self.dataset_path = dataset_path

        # read data from euroc dataset
        self.read_euroc_data()

        # create publishers for left and right images
        self.left_pub = self.create_publisher(Image, 'cam0/image_raw', 10)
        self.right_pub = self.create_publisher(Image, 'cam1/image_raw', 10)

        # create publishers for gt poses
        self.gt_poses_pub = self.create_publisher(PoseStamped, 'gt_poses', 10)
    
        # create publishers for intrinsics and extrinsics
        self.cam0_extrinsics_pub = self.create_publisher(PoseStamped, 'cam0_extrinsics', 10)
        self.cam1_extrinsics_pub = self.create_publisher(PoseStamped, 'cam1_extrinsics', 10)
        self.cam0_info_pub = self.create_publisher(CameraInfo, 'cam0/camera_info', 10)
        self.cam1_info_pub = self.create_publisher(CameraInfo, 'cam1/camera_info', 10)

        self.estimate_pose_sub = self.create_subscription(PoseStamped, 'estimated_pose', self.estimate_pose_callback, 10)
        self.aligned_estimated_pose_pub = self.create_publisher(PoseStamped, 'aligned_estimated_pose', 10)
        
        # create timer for publishing at 20Hz (50ms period)
        self.timer = self.create_timer(0.05, self.publish_data)
        
        # initialize image index
        self.current_idx = 0
        self.timestamps = list(self.data["cam0_image_timestamps"].keys())

        # initialize cv_bridge
        self.bridge = CvBridge()

        self.estimated_poses = {}

        self.world_transform = None

    def estimate_pose_callback(self, msg):
        self.get_logger().info(f"Received estimated pose: {msg}")
        timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec

        pose = np.eye(4)
        pose[:3, :3] = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]).as_matrix()
        pose[:3, 3] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32)
        self.estimated_poses[timestamp] = pose

        if timestamp in self.data["gt_poses"] and self.world_transform is None:
            gt_pose = self.data["gt_poses"][timestamp]
            self.world_transform = gt_pose @ np.linalg.inv(pose)
            print(f"gt_pose : {gt_pose}, transformed_pose : {self.world_transform @ pose}")

        if self.world_transform is not None:
            aligned_pose = self.world_transform @ pose
            aligned_msg = PoseStamped()
            aligned_msg.header.stamp = msg.header.stamp
            aligned_msg.header.frame_id = "world"
            aligned_msg.pose.position.x = aligned_pose[0, 3]
            aligned_msg.pose.position.y = aligned_pose[1, 3]
            aligned_msg.pose.position.z = aligned_pose[2, 3]
            aligned_msg.pose.orientation.x = R.from_matrix(aligned_pose[:3, :3]).as_quat()[0]
            aligned_msg.pose.orientation.y = R.from_matrix(aligned_pose[:3, :3]).as_quat()[1]
            aligned_msg.pose.orientation.z = R.from_matrix(aligned_pose[:3, :3]).as_quat()[2]
            aligned_msg.pose.orientation.w = R.from_matrix(aligned_pose[:3, :3]).as_quat()[3]
            self.aligned_estimated_pose_pub.publish(aligned_msg)


    def read_euroc_data(self):
        self.get_logger().info(f'Reading euroc data from {self.dataset_path}')

        # check if the dataset_path is a directory
        if not os.path.isdir(self.dataset_path):
            raise ValueError(f"The dataset_path {self.dataset_path} is not a directory")

        # check if the dataset_path contains the subfolder mav0
        if "cam0" not in os.listdir(self.dataset_path):
            raise ValueError(f"The dataset_path {self.dataset_path} does not contain the subfolder cam0")

        # check if the dataset_path contains the subfolder mav0
        if "cam1" not in os.listdir(self.dataset_path):
            raise ValueError(f"The dataset_path {self.dataset_path} does not contain the subfolder cam1")
    
        self.data = {
            "cam0_extrinsics": np.eye(4),
            "cam1_extrinsics": np.eye(4),
            "cam0_intrinsics": np.eye(3),
            "cam1_intrinsics": np.eye(3),
            "cam0_distortion": np.zeros(4),
            "cam1_distortion": np.zeros(4),
            "cam0_image_timestamps": {},
            "cam1_image_timestamps": {},
            "gt_poses":{},
        }

        # read cam0 extrinsics and intrinsics from sensor.yaml
        cam0_sensor_yaml_path   = os.path.join(self.dataset_path, "cam0", "sensor.yaml")
        cam1_sensor_yaml_path = os.path.join(self.dataset_path, "cam1", "sensor.yaml")

        # read cam0 extrinsics and intrinsics from sensor.yaml
        with open(cam0_sensor_yaml_path, 'r') as f:
            cam0_sensor_yaml = yaml.safe_load(f)
            # Extract data
            sensor_type = cam0_sensor_yaml.get("sensor_type")
            rate_hz = cam0_sensor_yaml.get("rate_hz")
            resolution = cam0_sensor_yaml.get("resolution")
            camera_model = cam0_sensor_yaml.get("camera_model")
            intrinsics = cam0_sensor_yaml.get("intrinsics")
            distortion_model = cam0_sensor_yaml.get("distortion_model")
            distortion_coefficients = cam0_sensor_yaml.get("distortion_coefficients")

            # Extract and reshape T_BS matrix
            T_BS_data = cam0_sensor_yaml["T_BS"]["data"]
            T_BS = np.array(T_BS_data).reshape((4, 4))

            # Print results
            print("Sensor type:", sensor_type)
            print("Resolution:", resolution)
            print("Camera model:", camera_model)
            print("Intrinsics:", intrinsics)
            print("Distortion model:", distortion_model)
            print("Distortion coefficients:", distortion_coefficients)
            print("T_BS (4x4):\n", T_BS)
            self.data["cam0_extrinsics"] = T_BS
            self.data["cam0_intrinsics"] = np.array([[intrinsics[0], 0, intrinsics[2]],
                                            [0, intrinsics[1], intrinsics[3]],
                                            [0, 0, 1]], dtype=np.float32)
            self.data["cam0_distortion"] = np.array(distortion_coefficients, dtype=np.float32)
    
        # read cam1 extrinsics and intrinsics from sensor.yaml
        with open(cam1_sensor_yaml_path, 'r') as f:
            cam1_sensor_yaml = yaml.safe_load(f)
            # Extract data
            sensor_type = cam1_sensor_yaml.get("sensor_type")
            rate_hz = cam1_sensor_yaml.get("rate_hz")
            resolution = cam1_sensor_yaml.get("resolution")
            camera_model = cam1_sensor_yaml.get("camera_model")
            intrinsics = cam1_sensor_yaml.get("intrinsics") 
            distortion_model = cam1_sensor_yaml.get("distortion_model")
            distortion_coefficients = cam1_sensor_yaml.get("distortion_coefficients")

            # Extract and reshape T_BS matrix
            T_BS_data = cam1_sensor_yaml["T_BS"]["data"]
            T_BS = np.array(T_BS_data).reshape((4, 4))

            # Print results
            print("Sensor type:", sensor_type)
            print("Resolution:", resolution)
            print("Camera model:", camera_model)
            print("Intrinsics:", intrinsics)
            print("Distortion model:", distortion_model)
            print("Distortion coefficients:", distortion_coefficients)
            print("T_BS (4x4):\n", T_BS)

            self.data["cam1_extrinsics"] = T_BS
            self.data["cam1_intrinsics"] = np.array([[intrinsics[0], 0, intrinsics[2]],
                                            [0, intrinsics[1], intrinsics[3]],
                                            [0, 0, 1]], dtype=np.float32)
            self.data["cam1_distortion"] = np.array(distortion_coefficients, dtype=np.float32)
            # read gt poses from state_groundtruth.txt  
        gt_poses_path = os.path.join(self.dataset_path,  "state_groundtruth_estimate0", "data.csv")

        with open(gt_poses_path, 'r') as f:
            # remove the first line
            gt_poses = f.readlines()[1:]
            for line in tqdm(gt_poses):
                line = line.strip()
                if line:
                    timestamp, x, y, z, qw, qx, qy, qz, vx, vy, vz, bwx,bwy,bwz, bax,bay,baz = line.split(",")
                    pose = np.eye(4)
                    pose[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                    pose[:3, 3] = np.array([x, y, z], dtype=np.float32)
                    self.data["gt_poses"][int(timestamp)] = pose

        # sort the gt_poses by timestamp
        self.data["gt_poses"] = dict(sorted(self.data["gt_poses"].items()))


        # read cam0 and cam1 image timestamps from timestamps.txt
        cam0_timestamps_path = os.path.join(self.dataset_path, "cam0", "data.csv")
        cam1_timestamps_path = os.path.join(self.dataset_path, "cam1", "data.csv")

        with open(cam0_timestamps_path, 'r') as f:
            cam0_timestamps_and_image_names = f.readlines()
            # remove the first line
            cam0_timestamps_and_image_names = cam0_timestamps_and_image_names[1:]
            for line in cam0_timestamps_and_image_names:
                line = line.strip()
                if line:
                    timestamp, image_name = line.split(",")
                    self.data["cam0_image_timestamps"][int(timestamp)] = image_name

        with open(cam1_timestamps_path, 'r') as f:
            cam1_timestamps_and_image_names = f.readlines()
            # remove the first line
            cam1_timestamps_and_image_names = cam1_timestamps_and_image_names[1:]
            for line in cam1_timestamps_and_image_names:
                line = line.strip()
                if line:
                    timestamp, image_name = line.split(",")
                    self.data["cam1_image_timestamps"][int(timestamp)] = image_name

        # sort the cam0_image_timestamps and cam1_image_timestamps by timestamp
        self.data["cam0_image_timestamps"] = dict(sorted(self.data["cam0_image_timestamps"].items()))
        self.data["cam1_image_timestamps"] = dict(sorted(self.data["cam1_image_timestamps"].items()))

        # validate the the timestamps between cam0 and cam1 are the same
        for image_timestamp in self.data["cam0_image_timestamps"].keys():
            if image_timestamp not in self.data["cam1_image_timestamps"]:
                raise ValueError(f"The image {image_timestamp} is not in cam1")

    # publish data at 20hz
    def publish_data(self):
        if self.current_idx >= len(self.timestamps):
            self.get_logger().info('Finished publishing all images')
            return

        timestamp = self.timestamps[self.current_idx]
        # Read and publish left image
        left_image_path = os.path.join(self.dataset_path, "cam0", "data", self.data["cam0_image_timestamps"][timestamp])
        left_image = cv2.imread(left_image_path)
        if left_image is not None:
            left_msg = self.bridge.cv2_to_imgmsg(left_image, "bgr8")
            # set stamp to  timestamp
            left_msg.header.stamp.sec = timestamp // 1000000000
            left_msg.header.stamp.nanosec = timestamp % 1000000000
            left_msg.header.frame_id = "cam0"
            self.left_pub.publish(left_msg)

        # Read and publish right image
        right_image_path = os.path.join(self.dataset_path, "cam1", "data", self.data["cam1_image_timestamps"][timestamp])
        right_image = cv2.imread(right_image_path)
        if right_image is not None:
            right_msg = self.bridge.cv2_to_imgmsg(right_image, "bgr8")
            # set stamp to  timestamp
            right_msg.header.stamp.sec = timestamp // 1000000000
            right_msg.header.stamp.nanosec = timestamp % 1000000000
            right_msg.header.frame_id = "cam1"
            self.right_pub.publish(right_msg)

        # Publish ground truth pose if available
        if timestamp in self.data["gt_poses"]:
            gt_pose = self.data["gt_poses"][timestamp]
            gt_msg = PoseStamped()
            # set stamp to  timestamp
            gt_msg.header.stamp.sec = timestamp // 1000000000
            gt_msg.header.stamp.nanosec = timestamp % 1000000000
            gt_msg.header.frame_id = "world"
            
            # Convert rotation matrix to quaternion
            rot = R.from_matrix(gt_pose[:3, :3])
            quat = rot.as_quat()
            
            gt_msg.pose.position = Point(x=float(gt_pose[0, 3]), 
                                       y=float(gt_pose[1, 3]), 
                                       z=float(gt_pose[2, 3]))
            gt_msg.pose.orientation = Quaternion(x=float(quat[0]), 
                                               y=float(quat[1]), 
                                               z=float(quat[2]), 
                                               w=float(quat[3]))
            self.gt_poses_pub.publish(gt_msg)

        # Publish camera parameters (only once)
        if self.current_idx != 0:
            # Publish cam0 extrinsics
            cam0_ext_msg = PoseStamped()
            # set stamp to  timestamp
            cam0_ext_msg.header.stamp.sec = timestamp // 1000000000
            cam0_ext_msg.header.stamp.nanosec = timestamp % 1000000000
            cam0_ext_msg.header.frame_id = "body"
            rot = R.from_matrix(self.data["cam0_extrinsics"][:3, :3])
            quat = rot.as_quat()
            cam0_ext_msg.pose.position = Point(x=float(self.data["cam0_extrinsics"][0, 3]), 
                                             y=float(self.data["cam0_extrinsics"][1, 3]), 
                                             z=float(self.data["cam0_extrinsics"][2, 3]))
            cam0_ext_msg.pose.orientation = Quaternion(x=float(quat[0]), 
                                                     y=float(quat[1]), 
                                                     z=float(quat[2]), 
                                                     w=float(quat[3]))
            self.cam0_extrinsics_pub.publish(cam0_ext_msg)

            # Publish cam1 extrinsics
            cam1_ext_msg = PoseStamped()
            # set stamp to  timestamp
            cam1_ext_msg.header.stamp.sec = timestamp // 1000000000
            cam1_ext_msg.header.stamp.nanosec = timestamp % 1000000000
            cam1_ext_msg.header.frame_id = "body"
            rot = R.from_matrix(self.data["cam1_extrinsics"][:3, :3])
            quat = rot.as_quat()
            cam1_ext_msg.pose.position = Point(x=float(self.data["cam1_extrinsics"][0, 3]), 
                                             y=float(self.data["cam1_extrinsics"][1, 3]), 
                                             z=float(self.data["cam1_extrinsics"][2, 3]))
            cam1_ext_msg.pose.orientation = Quaternion(x=float(quat[0]), 
                                                     y=float(quat[1]), 
                                                     z=float(quat[2]), 
                                                     w=float(quat[3]))
            self.cam1_extrinsics_pub.publish(cam1_ext_msg)

            # Publish cam0 camera info
            cam0_info = CameraInfo()
            # set stamp to  timestamp
            cam0_info.header.stamp.sec = timestamp // 1000000000
            cam0_info.header.stamp.nanosec = timestamp % 1000000000
            cam0_info.header.frame_id = "cam0"
            cam0_info.height = left_image.shape[0]
            cam0_info.width = left_image.shape[1]
            cam0_info.distortion_model = "plumb_bob"
            cam0_info.d = [float(x) for x in self.data["cam0_distortion"]]  # Convert to list of floats
            cam0_info.k = [float(x) for x in self.data["cam0_intrinsics"].flatten()]  # Convert to list of floats
            cam0_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Rectification matrix (identity for now)
            self.cam0_info_pub.publish(cam0_info)

            # Publish cam1 camera info
            cam1_info = CameraInfo()
            # set stamp to  timestamp
            cam1_info.header.stamp.sec = timestamp // 1000000000
            cam1_info.header.stamp.nanosec = timestamp % 1000000000
            cam1_info.header.frame_id = "cam1"
            cam1_info.height = right_image.shape[0]
            cam1_info.width = right_image.shape[1]
            cam1_info.distortion_model = "plumb_bob"
            cam1_info.d = [float(x) for x in self.data["cam1_distortion"]]  # Convert to list of floats
            cam1_info.k = [float(x) for x in self.data["cam1_intrinsics"].flatten()]  # Convert to list of floats
            cam1_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Rectification matrix (identity for now)
            self.cam1_info_pub.publish(cam1_info)

        self.current_idx += 1

def main(args=None):
    rclpy.init(args=args)

    # read data from euroc dataset
    dataset_path = "/home/ros/ros2_ws/src/stereo_slam/V1_01_easy/mav0"
    euroc_data_node = EurocDataNode(dataset_path)

    rclpy.spin(euroc_data_node)
    euroc_data_node.destroy_node()
    rclpy.shutdown()

