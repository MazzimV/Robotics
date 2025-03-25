#!/usr/bin/env python
import rospy
import rosbag
import tf
import numpy as np
import math
import cv2
import ultralytics
import os
from sensor_msgs.msg import CompressedImage, PointCloud2, Image, LaserScan
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, PoseWithCovariance, Twist
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from laser_geometry import LaserProjection
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from ultralytics import YOLO
from tf.transformations import quaternion_from_euler


class sign_detection():

    def __init__(self):
        rospy.init_node('sign_detector', anonymous=True)
        # Check if model path exists before loading
        self.model_path = 'src/sign_detection/weights/best.pt'
        if not os.path.exists(self.model_path):
            rospy.logerr(f"Model file not found at {self.model_path}")
            return
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        # Load YOLO model
        self.model = YOLO(self.model_path, task="detect")
        # Subscribe to raw image topic from RGB camera
        self.sub = rospy.Subscriber('/robot_1/depth_cam/rgb/image_raw', Image, self.callback)
        # Publisher for annotated images
        self.pub = rospy.Publisher('/sign_detections', Image, queue_size=10)
        self.pub = rospy.Publisher('/check_cmd_vel', Twist, queue_size=10)


    def callback(self, data):
        
        # Convert ROS Image message to OpenCV image
        try:
        
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # Run YOLO model on image with safe settings
            results = self.model(cv_image)
            high_conf_labels = []
            # Use YOLO to draw bounding boxes with confidence scores
            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf[0].item())  # Extract confidence score
                    if confidence >= 0.80:  # Confidence threshold check
                        label = self.model.names[int(box.cls[0].item())]  # Get label
                        high_conf_labels.append(label)
                annotated_image = result.plot()
                ros_image = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
                cv2.imshow("Original",cv_image)
                cv2.imshow("Sign Detection",annotated_image)
                cv2.waitKey(1)  
                self.pub.publish(ros_image)

            # self.process_signs(high_conf_labels)

            if high_conf_labels:
                rospy.loginfo(f"High-confidence detections: {high_conf_labels}")
            else:
                rospy.loginfo("No detections above 80% confidence.")
        except Exception as e:
            rospy.loginfo(e)

    
    def process_signs(self, detected_signs):

        cmd = Twist()  # Message for velocity control

        if "Stop" in detected_signs:
            rospy.loginfo("STOP sign detected! Stopping the robot.")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0  

        elif "SPEED LIMIT 50" in detected_signs:
            rospy.loginfo("Speed Limit 50 detected! Slowing down.")
            cmd.linear.x = 0.5  # Slow speed (example)

        elif "NO ENTRY" in detected_signs:
            rospy.loginfo("No Entry sign detected! Stopping immediately.")
            cmd.linear.x = 0.0  # Stop robot

        else:
            rospy.loginfo("No special sign detected, continuing at normal speed.")
            cmd.linear.x = 1.0  # Default speed (example)

        # Publish the command to '/check_cmd_vel' topic
        self.cmd_pub.publish(cmd)


if __name__ == '__main__':
    try:
        cv2.destroyAllWindows()
        proc = sign_detection()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
    except Exception as e:
        rospy.logerr(f"Unhandled exception: {e}")