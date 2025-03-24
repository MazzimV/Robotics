#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import time

class RobustLaneFollower:
    def __init__(self):
        rospy.init_node('lane_follower')
        self.bridge = CvBridge()
        
        # Lane detection parameters
        self.yellow_min = np.array([0, 35, 0], dtype=np.uint8)
        self.yellow_max = np.array([30, 255, 255], dtype=np.uint8)
        
        # ROI parameters
        self.roi_top_ratio = rospy.get_param('~roi_top_ratio', 0.5)  # From bottom, fraction of image height
        self.roi_bottom_ratio = rospy.get_param('~roi_bottom_ratio', 0.95)  # From bottom
        self.roi_width_ratio = rospy.get_param('~roi_width_ratio', 0.9)  # Fraction of image width
        
        # Control parameters
        self.max_speed = rospy.get_param('~max_speed', 0.4)  # m/s
        self.min_speed = rospy.get_param('~min_speed', 0.1)  # m/s
        self.max_angle = rospy.get_param('~max_angle', 1.2)  # rad/s
        
        # PID controller parameters
        self.kp = rospy.get_param('~kp', 0.8)  # Proportional gain
        self.ki = rospy.get_param('~ki', 0.01)  # Integral gain
        self.kd = rospy.get_param('~kd', 0.2)  # Derivative gain
        self.integral = 0.0
        self.last_error = 0.0
        self.error_history = []
        self.history_size = 5  # Number of frames to keep in history
        
        # Create a moving average filter
        self.steering_history = []
        self.steering_history_size = 3
        
        # Thresholds for lane detection quality
        self.min_lane_points = 50  # Minimum number of lane points to consider a valid detection
        self.max_steering_change = 0.3  # Maximum change in steering between frames
        
        # Status variables
        self.frame_count = 0
        self.last_good_steering = 0.0
        self.last_process_time = time.time()
        self.fps = 0
        
        # Publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/lane_debug', Image, queue_size=1)
        rospy.Subscriber('/robot_1/depth_cam/rgb/image_raw', Image, self.image_callback)
        
        rospy.loginfo("Lane follower initialized")
    
    def image_callback(self, data):
        """Process camera image for lane following"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Calculate FPS
            current_time = time.time()
            self.fps = 1.0 / (current_time - self.last_process_time)
            self.last_process_time = current_time
            
            # Process image for lane detection
            lane_center, debug_image = self.detect_lane(cv_image)
            
            # Calculate steering command based on lane position
            steering = self.calculate_steering(lane_center, cv_image.shape[1])
            
            # Create and publish command
            cmd = Twist()
            cmd.linear.x = self.max_speed
            cmd.angular.z = steering
            self.cmd_vel_pub.publish(cmd)
            
            # Publish debug image
            if debug_image is not None and self.debug_image_pub.get_num_connections() > 0:
                try:
                    self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
                except CvBridgeError as e:
                    rospy.logerr(f"Could not publish debug image: {e}")
            
            self.frame_count += 1
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def detect_lane(self, image):
        """Detect lane markers in the image"""
        try:
            # Create a copy for visualization
            debug_image = image.copy()
            
            # Get image dimensions
            height, width, _ = image.shape
            
            # Define region of interest
            roi_top = int(height * (1 - self.roi_top_ratio))
            roi_bottom = int(height * self.roi_bottom_ratio)
            roi_width = int(width * self.roi_width_ratio)
            roi_left = int((width - roi_width) / 2)
            roi_right = int(roi_left + roi_width)
            
            # Extract ROI
            roi = image[roi_top:roi_bottom, roi_left:roi_right]

            roi_top_2 = roi_top
            roi_bottom_2 = roi_bottom
            roi_width_2 = int(width * self.roi_width_ratio / 2)
            roi_left_2 = int(width / 2)
            roi_right_2 = int(roi_left_2 + roi_width_2)

            roi_2 = image[roi_top_2:roi_bottom_2, roi_left_2:roi_right_2]

            roi_top_top = int(roi_top_2 / 4)
            roi_bottom_top = roi_top_2
            roi_width_top = int(roi_width / 5)
            roi_left_top = int((width / 2) - (roi_width_top / 2))
            roi_right_top = int((width / 2) + (roi_width_top / 2))

            roi_3 = image[roi_top_top:roi_bottom_top, roi_left_top:roi_right_top]
            
            # Draw ROI on debug image
            cv2.rectangle(debug_image, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)

            cv2.rectangle(debug_image, (roi_left_top, roi_top_top), (roi_right_top, roi_bottom_top), (255, 0, 0), 2)
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            hsv_2 = cv2.cvtColor(roi_2, cv2.COLOR_BGR2HSV)

            hsv_top = cv2.cvtColor(roi_3, cv2.COLOR_BGR2HSV)
            
            # Create mask for yellow lane markers
            yellow_mask = cv2.inRange(hsv, self.yellow_min, self.yellow_max)

            yellow_mask_2 = cv2.inRange(hsv_2, self.yellow_min, self.yellow_max)

            yellow_mask_top = cv2.inRange(hsv_top, self.yellow_min, self.yellow_max)
            
            # Add text for the detected yellow pixels count
            yellow_pixel_count = np.sum(yellow_mask > 0)
            cv2.putText(debug_image, f"Yellow pixels: {yellow_pixel_count}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

            yellow_mask_2 = cv2.morphologyEx(yellow_mask_2, cv2.MORPH_OPEN, kernel)
            yellow_mask_2 = cv2.morphologyEx(yellow_mask_2, cv2.MORPH_CLOSE, kernel)

            yellow_mask_top = cv2.morphologyEx(yellow_mask_top, cv2.MORPH_OPEN, kernel)
            yellow_mask_top = cv2.morphologyEx(yellow_mask_top, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contours_2, _ = cv2.findContours(yellow_mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contours_top, _ = cv2.findContours(yellow_mask_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on debug image
            contour_image = np.zeros_like(roi)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            debug_image[roi_top:roi_bottom, roi_left:roi_right] = cv2.addWeighted(
                roi, 0.7, contour_image, 0.3, 0)
            
            # If no contours found, return None
            if not contours:
                cv2.putText(debug_image, "No lane markers detected", 
                          (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return None, debug_image
            
            if not contours_2:
                if contours_top:
                    return roi_right, debug_image
                return None, debug_image
            
            # Filter small contours
            contours = [c for c in contours if cv2.contourArea(c) > 100]

            contours_2 = [c for c in contours_2 if cv2.contourArea(c) > 100]
            
            # If no significant contours found, return None
            if not contours:
                cv2.putText(debug_image, "No significant lane markers", 
                          (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return None, debug_image
            
            # Find centers of mass for all contours
            centers = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + roi_left
                    cy = int(M["m01"] / M["m00"]) + roi_top
                    centers.append((cx, cy))
                    cv2.circle(debug_image, (cx, cy), 7, (255, 0, 0), -1)

            centers_2 = []
            for contour in contours_2:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + roi_left_2
                    cy = int(M["m01"] / M["m00"]) + roi_top_2
                    centers_2.append((cx, cy))
            
            # If no centers found, return None
            if not centers:
                return None, debug_image
            
            # Calculate average lane center
            lane_center_x = sum(x for x, _ in centers) / len(centers)

            lane_center_left = sum(x for x, _ in centers_2) / len(centers_2)
            
            # Draw lane center line
            cv2.line(debug_image, (int(lane_center_x), roi_top), 
                   (int(lane_center_x), roi_bottom), (0, 0, 255), 2)
            
            # Draw image center line for reference
            cv2.line(debug_image, (width // 2, roi_top), 
                   (width // 2, roi_bottom), (255, 255, 255), 1)
            
            # Calculate error from center
            error = (lane_center_x - width / 2) / (width / 2)  # Normalize to [-1, 1]
            
            # Display error on debug image
            cv2.putText(debug_image, f"Error: {error:.2f}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display FPS
            cv2.putText(debug_image, f"FPS: {self.fps:.1f}", 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
            return lane_center_x, debug_image
            
        except Exception as e:
            rospy.logerr(f"Lane detection error: {e}")
            return None, image
    
    def calculate_steering(self, lane_center, image_width):
        """Calculate steering command using PID controller"""
        if lane_center is None:
            # If lane not detected, use the last known good steering value
            rospy.logwarn_throttle(1.0, "Lane not detected, using last steering value")
            return -self.last_good_steering
        
        # Calculate error (normalized to [-1, 1])
        error = (lane_center - image_width / 2) / (image_width / 2)
        
        # Update error history
        self.error_history.append(error)
        if len(self.error_history) > self.history_size:
            self.error_history.pop(0)
        
        # Use median filtering on the error for robustness
        filtered_error = np.median(self.error_history)
        
        # PID control
        self.integral += filtered_error
        # Anti-windup: limit the integral term
        self.integral = max(-3.0, min(3.0, self.integral))
        
        derivative = filtered_error - self.last_error
        self.last_error = filtered_error
        
        # Calculate steering output
        steering = -(self.kp * filtered_error + self.ki * self.integral + self.kd * derivative)
        
        # Limit the maximum steering change rate for smoother control
        steering_change = steering - self.last_good_steering
        if abs(steering_change) > self.max_steering_change:
            steering = self.last_good_steering + np.sign(steering_change) * self.max_steering_change
        
        # Add to steering history for smoothing
        self.steering_history.append(steering)
        if len(self.steering_history) > self.steering_history_size:
            self.steering_history.pop(0)
        
        # Apply moving average to steering
        smoothed_steering = sum(self.steering_history) / len(self.steering_history)
        
        # Limit steering to max angle
        steering = max(-self.max_angle, min(self.max_angle, smoothed_steering))
        
        # Update last good steering
        self.last_good_steering = steering
        
        return steering
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        lane_follower = RobustLaneFollower()
        lane_follower.run()
    except rospy.ROSInterruptException:
        pass