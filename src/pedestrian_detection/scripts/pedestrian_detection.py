#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import time

class PedestrianDetection:
    def __init__(self):
        rospy.init_node('pedestrian_detection')
        self.bridge = CvBridge()
        
        # Pedestrian lane detection parameters
        self.white_min = np.array([0, 0, 200], dtype=np.uint8)
        self.white_max = np.array([180, 30, 255], dtype=np.uint8)
        
        # Publishers and subscribers
        self.pedestrian_pub = rospy.Publisher('/pedestrian', Bool, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/pedestrian_lane_debug', Image, queue_size=1)
        rospy.Subscriber('/robot_1/depth_cam/rgb/image_raw/compressed', Image, self.image_callback)
        
        rospy.loginfo("Pedestrian lane detector initialized")
    
    def image_callback(self, data):
        """Process camera image for pedestrian lane detection"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            
            # Process image for pedestrian lane detection
            pedestrian_detection, debug_image = self.detect_pedestrian_lane(cv_image)
            
            flag = Bool()
            flag.data = pedestrian_detection
            self.pedestrian_pub.publish(flag)
            
            # Publish debug image
            if debug_image is not None and self.debug_image_pub.get_num_connections() > 0:
                try:
                    self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
                except CvBridgeError as e:
                    rospy.logerr(f"Could not publish debug image: {e}")
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def detect_pedestrian_lane(self, image):
        """Detect pedestrian lane markers in the image"""
        try:
            # Create a copy for visualization
            debug_image = image.copy()
            
            # Get image dimensions
            height, width, _ = image.shape
            
            # Define region of interest
            lroi_top = int(height * 0.8)
            lroi_bottom = int(height)
            lroi_width = int(width * 0.15)
            lroi_left = 0
            lroi_right = int(lroi_left + lroi_width)

            # Extract ROI
            lroi = image[lroi_top:lroi_bottom, lroi_left:lroi_right]

            # Draw ROI on debug image
            cv2.rectangle(debug_image, (lroi_left, lroi_top), (lroi_right, lroi_bottom), (0, 255, 0), 2)
            
            # Convert to HSV for better color detection
            lhsv = cv2.cvtColor(lroi, cv2.COLOR_BGR2HSV)

            # Create mask for pedestrian lane markers
            l_white_mask = cv2.inRange(lhsv, self.white_min, self.white_max)

            # Add text for the detected yellow pixels count
            l_white_pixel_count = np.sum(l_white_mask > 0)
            cv2.putText(debug_image, f"White pixels: {l_white_pixel_count}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if l_white_pixel_count < 600:
                return False, debug_image
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            l_white_mask = cv2.morphologyEx(l_white_mask, cv2.MORPH_OPEN, kernel)
            l_white_mask = cv2.morphologyEx(l_white_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            l_contours, _ = cv2.findContours(l_white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on debug image
            l_contour_image = np.zeros_like(lroi)
            cv2.drawContours(l_contour_image, l_contours, -1, (255, 255, 255), 2)
            debug_image[lroi_top:lroi_bottom, lroi_left:lroi_right] = cv2.addWeighted(
                lroi, 0.7, l_contour_image, 0.3, 0)
            
            # If no contours found, return False
            if not l_contours:
                cv2.putText(debug_image, "No pedestrian lane markers detected", 
                          (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return False, debug_image
            
            # Filter small contours
            l_contours = [c for c in l_contours if cv2.contourArea(c) > 100]
            
            # If no significant contours found, return False
            if not l_contours:
                cv2.putText(debug_image, "No significant pedestrian lane markers", 
                          (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return False, debug_image
                        
            return True, debug_image
            
        except Exception as e:
            rospy.logerr(f"Pedestrian lane detection error: {e}")
            return None, image
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        pedestrian_detection = PedestrianDetection()
        pedestrian_detection.run()
    except rospy.ROSInterruptException:
        pass