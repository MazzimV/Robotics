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
        rospy.Subscriber('/robot_1/depth_cam/rgb/image_raw', Image, self.image_callback)
        
        rospy.loginfo("Pedestrian lane detector initialized")
    
    def image_callback(self, data):
        """Process camera image for pedestrian lane detection"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
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
            roi_top = int(height * 0.6)
            roi_bottom = int(height * 0.8)
            roi_width = int(width * 0.5)
            roi_left = int(width * 0.25)
            roi_right = int(roi_left + roi_width)

            # Extract ROI
            roi = image[roi_top:roi_bottom, roi_left:roi_right]

            # Draw ROI on debug image
            cv2.rectangle(debug_image, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Create mask for pedestrian lane markers
            white_mask = cv2.inRange(hsv, self.white_min, self.white_max)

            # Add text for the detected yellow pixels count
            white_pixel_count = np.sum(white_mask > 0)
            cv2.putText(debug_image, f"White pixels: {white_pixel_count}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if white_pixel_count < 4000:
                return False, debug_image
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on debug image
            contour_image = np.zeros_like(roi)
            cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
            debug_image[roi_top:roi_bottom, roi_left:roi_right] = cv2.addWeighted(
                roi, 0.7, contour_image, 0.3, 0)
            
            # If no contours found, return False
            if not contours:
                cv2.putText(debug_image, "No pedestrian lane markers detected", 
                          (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return False, debug_image
            
            # Filter small contours
            contours = [c for c in contours if cv2.contourArea(c) > 100]
            
            # If no significant contours found, return False
            if not contours:
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