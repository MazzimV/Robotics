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
from std_msgs.msg import String
import roslib

class sign_detection():

    def __init__(self):
        rospy.init_node('sign_detector', anonymous=True)

        self.path = roslib.packages.get_pkg_dir("sign_detection")
        self.model_path = f"{self.path}/weights/best.pt"
        
        self.bridge = CvBridge()

        self.model = YOLO(self.model_path, task="detect")

        self.sub = rospy.Subscriber('/robot_1/depth_cam/rgb/image_raw', Image, self.callback)
        self.pub = rospy.Publisher('/sign_detection', String, queue_size=1)
        self.pub_debug = rospy.Publisher('/sign_detection_debug', Image, queue_size=10)


    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            height, width, _ = cv_image.shape
            
            roi_top = int(height * 0.25)
            roi_bottom = int(height)
            roi_width = int(width)
            roi_left = int(0)
            roi_right = int(roi_left + roi_width)
            image = cv_image[roi_top:roi_bottom, roi_left:roi_right]
    
            results = self.model(image)
            
            maxi = -1

            final_label = None

            score  = -1
            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf[0].item())  # Extract confidence score
                    label = self.model.names[int(box.cls[0].item())]
                    if label == "red_light" or label == "green_light" or label == "yellow_light":
                        if score < confidence:
                            score = confidence
                            final_label = label
                    elif confidence >= 0.80: 

                        x1, y1, x2, y2 = box.xyxy[0] 
                        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                        width  = x2 - x1
                        height = y2 - y1

                        size = height * width
                        if maxi < size:
                            maxi = size
                            final_label = label
                annotated_image = result.plot()
                ros_image = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8") 
                self.pub_debug.publish(ros_image)

            if final_label is not None:
                self.pub.publish(label)
        except Exception as e:
            rospy.loginfo(e)

    detected_signs = ["red_light", "green_light", "yellow_light", "stop", "five", "no_left", "no_limit", "no_right", "park", "right"]

    def process_signs(self, detected_signs):
        cmd = Twist()  

        if "red_light" in detected_signs or "yellow_light" in detected_signs:
            rospy.loginfo("red_light or yellow_light sign detected! Stopping the robot.")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0  

        elif "green_light" in detected_signs:
            rospy.loginfo("green_light detected! Advance.")
            cmd.linear.x = 0.3  

        elif "stop" in detected_signs:
            rospy.loginfo("Stop sign detected! Stopping for 3 seconds.")
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)     
            rospy.sleep(3)  
            rospy.loginfo("Resuming movement after stop sign.")
            cmd.linear.x = 0.3  

        elif "five" in detected_signs:
            rospy.loginfo("Five sign detected! Slowing.")
            cmd.linear.x = 0.1  

        elif "no_limit" in detected_signs:
            rospy.loginfo("No limit sign detected! Accelerating.")
            cmd.linear.x = 0.3  

        elif "no_left" or "right" in detected_signs:
            rospy.loginfo("No left turn or right sign detected! Adjusting direction to the right.")
            if cmd.angular.z <= 0:
                cmd.angular.z = 0.2  

        elif "no_right" in detected_signs:
            rospy.loginfo("No right turn sign detected! Adjusting direction to the left.")
            if cmd.angular.z >= 0:
                cmd.angular.z = -0.2 

        else:
            rospy.loginfo("No special sign detected, continuing at normal speed.")
            cmd.linear.x = 1.0  

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