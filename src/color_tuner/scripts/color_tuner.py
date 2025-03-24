#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import threading
import sys
import tty
import termios

class KeyboardColorTuner:
    def __init__(self):
        rospy.init_node('keyboard_color_tuner')
        self.bridge = CvBridge()
        
        # HSV values to tune
        self.h_min = 0
        self.h_max = 30
        self.s_min = 35
        self.s_max = 255
        self.v_min = 0
        self.v_max = 255
        
        # Control variables
        self.step_size = 5  # How much to change values by
        self.current_param = "h_min"  # Which parameter is currently selected
        self.running = True
        
        # Camera topic
        self.camera_topic = rospy.get_param('~camera_topic', '/robot_1/depth_cam/rgb/image_raw')
        
        # Create publisher
        self.thresh_pub = rospy.Publisher('/color_tuner/thresholded', Image, queue_size=1)
        
        # Subscribe to camera
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        
        # Print instructions
        self.print_instructions()
        
        # Start keyboard input thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_input)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
    
    def print_instructions(self):
        """Print keyboard control instructions"""
        rospy.loginfo("=== Keyboard-Controlled Color Tuner ===")
        rospy.loginfo("Use the following keys to adjust HSV values:")
        rospy.loginfo("  1-6: Select parameter (h_min, h_max, s_min, s_max, v_min, v_max)")
        rospy.loginfo("  + or =: Increase selected parameter")
        rospy.loginfo("  - or _: Decrease selected parameter")
        rospy.loginfo("  [ or {: Decrease step size")
        rospy.loginfo("  ] or }: Increase step size")
        rospy.loginfo("  q: Quit")
        rospy.loginfo("Current parameter: " + self.current_param)
    
    def get_key(self):
        """Get a single keypress from the user"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    def keyboard_input(self):
        """Thread for handling keyboard input"""
        while self.running and not rospy.is_shutdown():
            key = self.get_key()
            
            # Parameter selection
            if key == '1':
                self.current_param = "h_min"
            elif key == '2':
                self.current_param = "h_max"
            elif key == '3':
                self.current_param = "s_min"
            elif key == '4':
                self.current_param = "s_max"
            elif key == '5':
                self.current_param = "v_min"
            elif key == '6':
                self.current_param = "v_max"
            
            # Value adjustment
            elif key in ['+', '=']:
                self.adjust_value(self.step_size)
            elif key in ['-', '_']:
                self.adjust_value(-self.step_size)
            
            # Step size adjustment
            elif key in ['[', '{']:
                self.step_size = max(1, self.step_size - 1)
                rospy.loginfo(f"Step size: {self.step_size}")
            elif key in [']', '}']:
                self.step_size = min(20, self.step_size + 1)
                rospy.loginfo(f"Step size: {self.step_size}")
            
            # Quit
            elif key in ['q', 'Q']:
                self.running = False
                rospy.loginfo("Quitting...")
                rospy.signal_shutdown("User requested shutdown")
            
            # Print current values after any key press
            self.print_values()
    
    def adjust_value(self, amount):
        """Adjust the selected parameter by the given amount"""
        if self.current_param == "h_min":
            self.h_min = max(0, min(179, self.h_min + amount))
        elif self.current_param == "h_max":
            self.h_max = max(0, min(179, self.h_max + amount))
        elif self.current_param == "s_min":
            self.s_min = max(0, min(255, self.s_min + amount))
        elif self.current_param == "s_max":
            self.s_max = max(0, min(255, self.s_max + amount))
        elif self.current_param == "v_min":
            self.v_min = max(0, min(255, self.v_min + amount))
        elif self.current_param == "v_max":
            self.v_max = max(0, min(255, self.v_max + amount))
    
    def print_values(self):
        """Print current HSV values"""
        rospy.loginfo(f"Current: {self.current_param} = {getattr(self, self.current_param)}, Step: {self.step_size}")
        rospy.loginfo(f"H: [{self.h_min}, {self.h_max}], S: [{self.s_min}, {self.s_max}], V: [{self.v_min}, {self.v_max}]")
    
    def image_callback(self, data):
        """Process image and publish results"""
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Convert to HSV
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Create mask
            lower = np.array([self.h_min, self.s_min, self.v_min])
            upper = np.array([self.h_max, self.s_max, self.v_max])
            mask = cv2.inRange(hsv, lower, upper)
            
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Calculate percentage of yellow detected
            yellow_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            percentage = (yellow_pixels * 100.0) / total_pixels
            
            # Log detection percentage periodically
            rospy.loginfo_throttle(2.0, f"Yellow detected: {percentage:.2f}% of image")
            
            # Apply the mask to the original image to see the result
            result = cv2.bitwise_and(cv_image, cv_image, mask=mask)
            
            # Add text with current values
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"H: [{self.h_min}, {self.h_max}], S: [{self.s_min}, {self.s_max}], V: [{self.v_min}, {self.v_max}]"
            cv2.putText(result, text, (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(result, f"Yellow: {percentage:.1f}%", (10, 60), font, 0.7, (255, 255, 255), 2)
            
            # Publish the result
            result_msg = self.bridge.cv2_to_imgmsg(result, "bgr8")
            self.thresh_pub.publish(result_msg)
            
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")
    
    def run(self):
        """Main loop"""
        rospy.loginfo("Color tuner running. Use keyboard to adjust HSV values.")
        rospy.spin()

if __name__ == '__main__':
    try:
        tuner = KeyboardColorTuner()
        tuner.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        # Reset terminal settings on error
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSADRAIN, termios.tcgetattr(fd))