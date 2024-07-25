import rclpy as rp
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import cv2
import socket
import struct
import pickle
from ultralytics import YOLO
import os
from robot_msgs.msg import ObjectPose


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile)
        self.camera_subscription = self.create_subscription(CompressedImage, '/camera', self.camera_callback, 10)
        self.publish_timer = self.create_timer(0.1, self.ObjectDetectionPublisher)
        self.Objectpublisher = self.create_publisher(ObjectPose, '/detected_object', 10)

        self.detector_id = 1
        
        self.model = YOLO('/home/aa/dev_ws/final_project/final_project_package/src/final_project_package/model/detector.pt')
        self.conf = 0.3

        self.image_center_x = 160
        self.image_center_y = 120
        self.image = np.zeros((240, 320, 3), dtype = np.uint8)
        self.video_width = 320
        self.video_height = 240
        self.imaige = np.zeros((self.video_height, self.video_width, 3), np.uint8)

        self.lidar_points = np.array([])

        self.T_lidar_to_camera = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, -0.02315],
            [1, 0, 0, -0.0661],
            [0, 0, 0, 1]
        ])


        self.camera_matrix = np.array([[501.71184364/2, 0, 319.93415777/2],
                                [0, 500.97002316/2, 245.20985799/2],
                                [0, 0, 1]])

        self.dist_coeffs = np.array([0.20046918, -0.5471712, -0.00194516, -0.00144732, 0.42454179])

        self.subscription
        self.camera_subscription



    
    def camera_callback(self, data):
        np_arr = np.frombuffer(data.data, np.uint8)
        self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.video_height = self.image.shape[0]
        self.video_width = self.image.shape[1]
        #print(self.image.shape)

    
    def lidar_callback(self, msg):
        self.last_lidar = msg
        angle_min = self.last_lidar.angle_min
        angle_max = self.last_lidar.angle_max
        angle_increment = self.last_lidar.angle_increment

        self.ranges = np.array(self.last_lidar.ranges)
        self.angles = np.arange(angle_min, angle_max, (angle_max - angle_min)/len(self.ranges))

        self.ranges = self.ranges[70:210]
        self.angles = self.angles[70:210]

        validator = self.ranges > 0.1
        self.angles = self.angles[validator]
        self.ranges = self.ranges[validator]

        x = self.ranges*np.cos(self.angles)
        y = self.ranges*np.sin(self.angles)
        z = np.zeros_like(x)

        self.lidar_points = np.vstack((x, y, z, np.ones_like(x)))
        

    def ObjectDetectionPublisher(self):
        cv_image = self.image

        if len(self.lidar_points):
            lidar_points_camera = self.T_lidar_to_camera @ self.lidar_points

            points_2d, _ = cv2.projectPoints(lidar_points_camera[:3, :].T, np.zeros((3,1), dtype=np.float64), np.zeros((3,1), dtype=np.float64), self.camera_matrix, self.dist_coeffs)

            results = self.model(self.image, conf=self.conf)

            class_list = []
            distance_list = []
            angle_list = []

            start_point = 0
            end_point = self.video_width

            if len(results[0]):
                print(len(results[0]))
                for result in results[0].boxes:
                    try:
                        cls = int(result.cls[0])
                        label = "robot" if cls == 0 else "laborer"

                        x1, y1, x2, y2 = map(int, result.xyxy[0])

                        start_point = x1
                        end_point = x2

                        distances = [abs(point[0][0] - start_point) for point in points_2d]
                        start_point_index = np.argmin(distances)
                        distances = [abs(point[0][0] - end_point) for point in points_2d]
                        end_point_index = np.argmin(distances)
                        closest_index = np.argmin(self.ranges[min(start_point_index, end_point_index):max(start_point_index, end_point_index)])+min(start_point_index, end_point_index)
                        closest_distance = float(self.ranges[closest_index])
                        closest_angle = float(self.angles[closest_index])

                        class_list.append(label)
                        distance_list.append(closest_distance)
                        angle_list.append(closest_angle)

                        cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        cv2.putText(self.image, label, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                    
                    except:
                        print("ERROR!!!!!!!!!!!!!!!!!!")
                        continue
                
            for i, point in enumerate(points_2d):
                x, y = int(point[0][0]), int(point[0][1])
                if 0 <= x < cv_image.shape[1] and 0 <= y < cv_image.shape[0]:
                    cv2.circle(cv_image, (x, y), 3, (0, 255, 0), -1)

            if len(class_list):
                print(class_list)
                print(distance_list)
                print(angle_list)

                Object = ObjectPose()
                Object.detector_id = self.detector_id
                Object.detected = True
                Object.labels = class_list
                Object.ranges = distance_list
                Object.angles = angle_list
                self.Objectpublisher.publish(Object)
            
            else:
                Object = ObjectPose()
                Object.detector_id = self.detector_id
                Object.detected = False
                self.Objectpublisher.publish(Object)

            
            

        cv2.imshow('lidar projection', cv_image)
        cv2.waitKey(1)
        
        
def main(args=None):
    rp.init(args=args)
    scan_node = ObjectDetector()
    rp.spin(scan_node)
    scan_node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()