import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1 = q1[0]
    y1 = q1[1]
    z1 = q1[2]
    w1 = q1[3]
    x2 = q2[0]
    y2 = q2[1]
    z2 = q2[2]
    w2 = q2[3]

    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2

    return np.array([x, y, z, w])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quaternion_rotation_vector(q: np.ndarray, p:np.ndarray) -> np.ndarray:
    unit_quaternion = q / np.linalg.norm(q)

    p_quaternion = np.concatenate([p, np.zeros(1)])

    q_inverse = quaternion_inverse(unit_quaternion)

    rotated_p_quaternion = quaternion_multiply(quaternion_multiply(  p_quaternion , unit_quaternion), q_inverse)

    return rotated_p_quaternion[:3]


class Rigid3d:
    # qx, qy, qz, qw
    # x, y, z
    def __init__(self, rotation_quaternion, translation):
        self.rotation_quaternion = rotation_quaternion
        self.translation = translation

    def __mul__(self, other):
        return Rigid3d(quaternion_multiply(self.rotation_quaternion, other.rotation_quaternion), self.translation + quaternion_rotation_vector(self.rotation_quaternion, other.translation))

    def transform(self, p: np.ndarray) -> np.ndarray:
        return quaternion_rotation_vector(self.rotation_quaternion, p) + self.translation

    def inverse(self) -> 'Rigid3d':
        rotation_quaternion_inverse = quaternion_inverse(self.rotation_quaternion)
        return Rigid3d(rotation_quaternion_inverse, -quaternion_rotation_vector(rotation_quaternion_inverse, self.translation))

    def to_vector(self) -> np.ndarray:
        return np.concatenate([ self.translation, self.rotation_quaternion])

    def to_matrix33(self) -> np.ndarray:
        return R.from_quat(self.rotation_quaternion).as_matrix()

    def translation_vector(self) -> np.ndarray:
        return self.translation

    @staticmethod
    def from_vector(vector: np.ndarray) -> 'Rigid3d':
        translation = vector[:3]
        rotation_quaternion = vector[3:]
        return Rigid3d(rotation_quaternion, translation)

    @staticmethod
    def identity() -> 'Rigid3d':
        return Rigid3d(np.array([0, 0, 0, 1]), np.array([0, 0, 0]))
        
        