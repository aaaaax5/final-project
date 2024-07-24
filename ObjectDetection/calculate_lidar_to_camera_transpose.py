import numpy as np
import tf_transformations
import cv2

# Helper function to convert a transform to a 4x4 matrix
def transform_to_matrix(transform):
    trans = transform['translation']
    rot = transform['rotation']
    translation_matrix = tf_transformations.translation_matrix((trans['x'], trans['y'], trans['z']))
    rotation_matrix = tf_transformations.quaternion_matrix((rot['x'], rot['y'], rot['z'], rot['w']))
    transform_matrix = np.dot(translation_matrix, rotation_matrix)
    return transform_matrix

# Transforms from your data
T_base_to_front_camera_mount = transform_to_matrix({
    'translation': {'x': 0.045, 'y': 0.0, 'z': 0.085},
    'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
})

T_front_camera_mount_to_camera = transform_to_matrix({
    'translation': {'x': 0.0111, 'y': 0.0, 'z': 0.0193},
    'rotation': {'x': 0.0, 'y': 0.7071067811865475, 'z': 0.0, 'w': 0.7071067811865476}
})

T_base_to_ydlidar_lidar_mount = transform_to_matrix({
    'translation': {'x': -0.01, 'y': 0.0, 'z': 0.085},
    'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
})

T_ydlidar_lidar_mount_to_laser = transform_to_matrix({
    'translation': {'x': 0.0, 'y': 0.0, 'z': 0.04245},
    'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
})

# Compute the full transform from base to camera and base to lidar
T_base_to_camera = np.dot(T_base_to_front_camera_mount, T_front_camera_mount_to_camera)
T_base_to_lidar = np.dot(T_base_to_ydlidar_lidar_mount, T_ydlidar_lidar_mount_to_laser)

# Compute the transform from camera to lidar
T_camera_to_lidar = np.dot(np.linalg.inv(T_base_to_camera), T_base_to_lidar)

# Compute the transform from lidar to camera
T_lidar_to_camera = np.linalg.inv(T_camera_to_lidar)

print(T_lidar_to_camera)

# Extract rotation and translation vectors
rvec, _ = cv2.Rodrigues(T_lidar_to_camera[:3, :3])
tvec = T_lidar_to_camera[:3, 3]

print("rvec:", rvec)
print("tvec:", tvec)