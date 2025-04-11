#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import time
from enum import Enum, auto

class DriveState(Enum):
    """
    Enum representing the possible driving states of the robot.
    """
    LANE_FOLLOWING = auto()
    LANE_ENDING = auto()
    EXECUTING_TURN = auto()


class LaneFollowerNode:
    """
    Main ROS node for lane following.
    """
    def __init__(self):
        """Initialize the lane follower node."""
        rospy.init_node('lane_follower')

        # Main ROI
        self.roi_top_ratio: float = 0.5
        self.roi_bottom_ratio: float = 0.95
        self.roi_width_ratio: float = 0.9
        
        # Look ahead ROI
        self.look_ahead_roi_top_ratio: float = 0.45
        self.look_ahead_roi_bottom_ratio: float = 0.55
        self.look_ahead_width_ratio: float = 0.4

        # Load color parameters
        self.yellow_min = np.array([5, 30, 30], dtype=np.uint8)
        self.yellow_max = np.array([30, 255, 255], dtype=np.uint8)

        # PID controllers
        self.kp=0.6
        self.ki=0.0
        self.kd=0.0
        self.integral = 0.0
        self.last_error = 0.0

        # Speed control
        self.speed=0.3
        self.angular=0.9

        # Turn control
        self.turn_duration=4.0
        self.lane_width_estimate_ratio=0.4
        self.turn_start_time=None
        self.turn_direction='right'

        # Driving state
        self.drive_state = DriveState.LANE_FOLLOWING
        
        # Initialize bridge
        self.bridge = CvBridge()
                
        # Publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/lane_debug', Image, queue_size=1)
        rospy.Subscriber('/robot_1/depth_cam/rgb/image_raw', Image, self.image_callback)

    
    def image_callback(self, data):
        """
        Process camera image for lane following.
        
        Args:
            data: ROS Image message
        """
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Process image for lane detection
            detection_result = self.detect_lane(
                cv_image, 
                self.drive_state, 
            )
            
            # Update driving state
            self.update_state(detection_result)

            # Calculate steering and speed based on current state
            steering = self.calculate_control(
                detection_result,
                self.drive_state,
                self.turn_direction,
                cv_image.shape[1]
            )
            
            # Create and publish command
            cmd = Twist()
            cmd.linear.x = self.speed
            cmd.angular.z = steering
            self.cmd_vel_pub.publish(cmd)
            
            # Publish debug image
            if detection_result.debug_image is not None and self.debug_image_pub.get_num_connections() > 0:
                try:
                    self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(detection_result.debug_image, "bgr8"))
                except CvBridgeError as e:
                    rospy.logerr(f"Could not publish debug image: {e}")
                        
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def detect_lane(self, image, drive_state):
        """
        Main lane detection method that orchestrates the detection pipeline.
        
        Args:
            image: Input camera image
            drive_state: Current drive state
        """
        try:
            # Execute detection pipeline
            processed_data = self.preprocess_image(image)
                
            processed_data = self.create_color_masks(processed_data)
                
            processed_data = self.find_contours(processed_data)
                
            result = self.analyze_lane_position(processed_data)
            
            # Create debug visualization
            debug_image = self.create_debug_visualization(result, processed_data, drive_state)

            result.debug_image = debug_image
            
            return result
            
        except Exception as e:
            rospy.logerr(f"Lane detection error: {e}")
            return {"debug_image": image}

    def preprocess_image(self, image):
        """
        Preprocess the image by defining ROIs and converting to HSV.
        
        Args:
            image: The input camera image
        """
        height, width, _ = image.shape
        
        # Define primary region of interest
        roi_top = int(height * self.roi_top_ratio)
        roi_bottom = int(height * self.roi_bottom_ratio)
        roi_width = int(width * self.roi_width_ratio)
        roi_left = int((width - roi_width) / 2)
        roi_right = int(roi_left + roi_width)
        
        # Define look-ahead region of interest
        look_ahead_top = int(height * self.look_ahead_roi_top_ratio)
        look_ahead_bottom = int(height * self.look_ahead_roi_bottom_ratio)
        look_ahead_width = int(width * self.look_ahead_width_ratio)
        look_ahead_left = int((width - look_ahead_width) / 2)
        look_ahead_right = int(look_ahead_left + look_ahead_width)
        
        # Split ROI for left and right lane detection
        roi_width_pixels = roi_right - roi_left
        roi_mid = roi_left + roi_width_pixels//2

        roi = image[roi_top:roi_bottom, roi_left:roi_right]
        look_ahead_roi = image[look_ahead_top:look_ahead_bottom, look_ahead_left:look_ahead_right]
        roi_left_half = image[roi_top:roi_bottom, roi_left:roi_mid]
        roi_right_half = image[roi_top:roi_bottom, roi_mid:roi_right]
        look_ahead_left_half = image[look_ahead_top:look_ahead_bottom, look_ahead_left:look_ahead_left + look_ahead_width//2]
        look_ahead_right_half = image[look_ahead_top:look_ahead_bottom, look_ahead_left + look_ahead_width//2:look_ahead_right]
        
        # Convert to HSV for better color detection
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_look_ahead = cv2.cvtColor(look_ahead_roi, cv2.COLOR_BGR2HSV)
        hsv_left = cv2.cvtColor(roi_left_half, cv2.COLOR_BGR2HSV)
        hsv_right = cv2.cvtColor(roi_right_half, cv2.COLOR_BGR2HSV)
        hsv_look_ahead_left = cv2.cvtColor(look_ahead_left_half, cv2.COLOR_BGR2HSV)
        hsv_look_ahead_right = cv2.cvtColor(look_ahead_right_half, cv2.COLOR_BGR2HSV)
        
        return {
            'original_image': image,
            'height': height,
            'width': width,
            'roi': roi,
            'look_ahead_roi': look_ahead_roi,
            'roi_left_half': roi_left_half,
            'roi_right_half': roi_right_half,
            'look_ahead_left_half': look_ahead_left_half,
            'look_ahead_right_half': look_ahead_right_half,
            'hsv_roi': hsv_roi,
            'hsv_look_ahead': hsv_look_ahead,
            'hsv_left': hsv_left,
            'hsv_right': hsv_right,
            'hsv_look_ahead_left': hsv_look_ahead_left,
            'hsv_look_ahead_right': hsv_look_ahead_right,
            'roi_top': roi_top,
            'roi_bottom': roi_bottom,
            'roi_left': roi_left,
            'roi_right': roi_right,
            'roi_mid': roi_mid,
            'look_ahead_top': look_ahead_top,
            'look_ahead_bottom': look_ahead_bottom,
            'look_ahead_left': look_ahead_left,
            'look_ahead_right': look_ahead_right
        }
    
    def create_color_masks(self, processed_data):
        """
        Create color masks for lane detection.
        
        Args:
            processed_data: Dictionary with preprocessed image data
        """
            
        # Create masks for yellow lane markers
        yellow_mask = cv2.inRange(
            processed_data['hsv_roi'], 
            self.yellow_min, 
            self.yellow_max
        )
        
        look_ahead_yellow_mask = cv2.inRange(
            processed_data['hsv_look_ahead'], 
            self.yellow_min, 
            self.yellow_max
        )
        
        left_yellow_mask = cv2.inRange(
            processed_data['hsv_left'], 
            self.yellow_min, 
            self.yellow_max
        )
        
        right_yellow_mask = cv2.inRange(
            processed_data['hsv_right'], 
            self.yellow_min, 
            self.yellow_max
        )
        
        left_look_ahead_yellow_mask = cv2.inRange(
            processed_data['hsv_look_ahead_left'], 
            self.yellow_min, 
            self.yellow_max
        )
        
        right_look_ahead_yellow_mask = cv2.inRange(
            processed_data['hsv_look_ahead_right'], 
            self.yellow_min, 
            self.yellow_max
        )
        
        processed_data.update({
            'yellow_mask': yellow_mask,
            'look_ahead_yellow_mask': look_ahead_yellow_mask,
            'left_yellow_mask': left_yellow_mask,
            'right_yellow_mask': right_yellow_mask,
            'left_look_ahead_yellow_mask': left_look_ahead_yellow_mask,
            'right_look_ahead_yellow_mask': right_look_ahead_yellow_mask,
        })
        
        return processed_data
    
    def find_contours(self, processed_data):
        """
        Find contours in the color masks.
        
        Args:
            processed_data: Dictionary with mask data
        """
            
        # Find contours for left and right regions separately
        left_contours, _ = cv2.findContours(
            processed_data['left_yellow_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        right_contours, _ = cv2.findContours(
            processed_data['right_yellow_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Find contours in the main ROI
        contours, _ = cv2.findContours(
            processed_data['yellow_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Find contours in the look-ahead ROI for detecting lane endings
        look_ahead_contours, _ = cv2.findContours(
            processed_data['look_ahead_yellow_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        processed_data.update({
            'left_contours': left_contours,
            'right_contours': right_contours,
            'contours': contours,
            'look_ahead_contours': look_ahead_contours
        })
        
        return processed_data
    
    def analyze_lane_position(self, processed_data):
        """
        Analyze the detected lane markers to find lane position.
        
        Args:
            processed_data: Dictionary with contour data
        """
            
        # Create a copy for visualization
        debug_image = processed_data['original_image'].copy()
        
        # Calculate yellow pixel counts
        left_yellow_pixel_count = np.sum(processed_data['left_yellow_mask'] > 0)
        right_yellow_pixel_count = np.sum(processed_data['right_yellow_mask'] > 0)
        
        # Check for horizontal lines that indicate intersections
        is_lane_ending = self._detect_lane_ending(processed_data, debug_image)
        
        # Find lane center
        lane_center_x = self._calculate_lane_center(
            processed_data, 
            left_yellow_pixel_count, 
            right_yellow_pixel_count,
            debug_image
        )
        
        # Calculate error from center
        error = (lane_center_x - processed_data['width'] / 2) / (processed_data['width'] / 2)
        
        return {
            "debug_image": debug_image,
            "error": error,
            "is_lane_ending": is_lane_ending
        }
    
    def _detect_lane_ending(self, processed_data, debug_image):
        """
        Detect if a lane is ending or if there's an intersection.
        
        Args:
            processed_data: Dictionary with processed image data
            debug_image: Image for visualization
        """
        # Check for horizontal lines that indicate intersections
        horizontal_lines = cv2.HoughLinesP(
            processed_data['look_ahead_yellow_mask'], 
            1, np.pi/180, 50, 
            minLineLength=30, 
            maxLineGap=10
        )

        horizontal_line_detected = False
        if horizontal_lines is not None:
            for line in horizontal_lines:
                x1, y1, x2, y2 = line[0]
                # If line is mostly horizontal (small y difference)
                if abs(y2 - y1) < 10 and abs(x2 - x1) > 20:
                    horizontal_line_detected = True
                    # Draw the detected horizontal line
                    cv2.line(
                        debug_image, 
                        (x1 + processed_data['look_ahead_left'], y1 + processed_data['look_ahead_top']), 
                        (x2 + processed_data['look_ahead_left'], y2 + processed_data['look_ahead_top']), 
                        (0, 255, 255), 2
                    )
        
        return horizontal_line_detected
    
    def _calculate_lane_center(self, 
                              processed_data,
                              debug_image):
        """
        Calculate the center of the lane based on detected contours.
        
        Args:
            processed_data: Dictionary with processed image data
            left_yellow_pixel_count: Number of yellow pixels in left half
            right_yellow_pixel_count: Number of yellow pixels in right half
            debug_image: Image for visualization
        """
        width = processed_data['width']
        
        # Calculate left and right lane positions if contours exist
        left_x = self._get_contour_center_x(
            processed_data['left_contours'], 
            processed_data['roi_left'], 
            processed_data['roi_top'],
            debug_image,
            (255, 0, 0)
        )
        
        right_x = self._get_contour_center_x(
            processed_data['right_contours'], 
            processed_data['roi_mid'], 
            processed_data['roi_top'],
            debug_image,
            (0, 0, 255)
        )
        
        # Estimate lane center based on available data
        lane_width_estimate = int(width * self.lane_width_estimate_ratio)
        
        if left_x is not None and right_x is not None:
            # Both lane boundaries detected - use the midpoint
            lane_center_x = (left_x + right_x) / 2
            cv2.putText(
                debug_image, "BOTH LANES DETECTED", 
                (width//2 - 120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        elif left_x is not None and right_x is None:
            # Only left boundary detected - estimate right boundary
            # Special case for S→1→2 intersection if left lane is strong and right weak
            lane_center_x = left_x + lane_width_estimate
            cv2.putText(
                debug_image, "LEFT ONLY - ADJUST RIGHT", 
                (width//2 - 120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
            )
                
        elif left_x is None and right_x is not None:
            # Only right boundary detected - estimate left boundary
            lane_center_x = right_x - lane_width_estimate
            cv2.putText(
                debug_image, "RIGHT ONLY - ADJUST LEFT", 
                (width//2 - 120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
        
        else:
            # Fallback to traditional center calculation using all contours
            lane_center_x = self._get_contour_center_x(
                processed_data['contours'], 
                processed_data['roi_left'], 
                processed_data['roi_top'],
                debug_image,
                (255, 0, 255)
            )
        
        if lane_center_x is not None:
            # Draw lane center line
            cv2.line(
                debug_image, 
                (int(lane_center_x), processed_data['roi_top']), 
                (int(lane_center_x), processed_data['roi_bottom']), 
                (0, 0, 255), 2
            )
        
        return lane_center_x
    
    def _get_contour_center_x(self, 
                             contours, 
                             x_offset, 
                             y_offset,
                             debug_image,
                             color):
        """
        Calculate the average x-coordinate of contour centers.
        
        Args:
            contours: List of contours
            x_offset: X-coordinate offset
            y_offset: Y-coordinate offset
            debug_image: Optional image for visualization
            color: Color for visualization
        """
        filtered_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        if not filtered_contours:
            return None
            
        centers = []
        for contour in filtered_contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + x_offset
                cy = int(M["m01"] / M["m00"]) + y_offset
                centers.append((cx, cy))
                
                if debug_image is not None:
                    cv2.circle(debug_image, (cx, cy), 7, color, -1)
        
        if not centers:
            return None
            
        return sum(x for x, _ in centers) / len(centers)
    
    def create_debug_visualization(self, 
                                  result, 
                                  processed_data,
                                  drive_state):
        """
        Create visualization for debugging.
        
        Args:
            result: Lane detection result
            processed_data: Dictionary with processed image data
            drive_state: Current drive state
        """
            
        debug_image = result.debug_image
        width = processed_data['width']
        
        # Draw ROIs
        cv2.rectangle(
            debug_image,
            (processed_data['roi_left'], processed_data['roi_top']),
            (processed_data['roi_right'], processed_data['roi_bottom']),
            (0, 255, 0), 2
        )
        
        cv2.rectangle(
            debug_image,
            (processed_data['look_ahead_left'], processed_data['look_ahead_top']),
            (processed_data['look_ahead_right'], processed_data['look_ahead_bottom']),
            (255, 0, 0), 2
        )
        
        cv2.line(
            debug_image,
            (processed_data['roi_mid'], processed_data['roi_top']),
            (processed_data['roi_mid'], processed_data['roi_bottom']),
            (255, 255, 0), 1
        )
        
        # Draw image center line for reference
        cv2.line(
            debug_image,
            (width // 2, processed_data['roi_top']),
            (width // 2, processed_data['roi_bottom']),
            (255, 255, 255), 1
        )
        
        # Show yellow pixel information
        yellow_pixel_count = np.sum(processed_data['yellow_mask'] > 0)
        left_yellow_pixel_count = np.sum(processed_data['left_yellow_mask'] > 0)
        right_yellow_pixel_count = np.sum(processed_data['right_yellow_mask'] > 0)
        
        cv2.putText(
            debug_image, f"Yellow pixels: {yellow_pixel_count}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        
        cv2.putText(
            debug_image, f"Left pixels: {left_yellow_pixel_count}, Right pixels: {right_yellow_pixel_count}", 
            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        
        # Display lane ending warning
        if result.is_lane_ending:
            cv2.putText(
                debug_image, "LANE ENDING", 
                (width//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
        
        # Display error
        if result.error is not None:
            cv2.putText(
                debug_image, f"Error: {result.error:.2f}", 
                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
        
        # Display current state
        cv2.putText(
            debug_image, f"State: {drive_state.name}", 
            (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )
        
        return debug_image
    
    def update_state(self, detection_result):
        """
        Update the driving state based on lane detection results.
        
        Args:
            detection_result: Results from lane detection
        """
        
        # Handle lane ending detection
        if detection_result.is_lane_ending and self.drive_state != DriveState.EXECUTING_TURN:
            self.drive_state = DriveState.EXECUTING_TURN
            self.turn_start_time = time.time()
            
            # Determine turn direction based on which lane is more visible
            if detection_result.left_count > detection_result.right_count + 100:
                self.turn_direction = 'right'  # Turn toward the missing right lane
            elif detection_result.right_count > detection_result.left_count + 100:
                self.turn_direction = 'left'  # Turn toward the missing left lane
        
        # Turn completion check
        elif self.drive_state == DriveState.EXECUTING_TURN:
            if time.time() - self.turn_start_time > self.turn_duration:
                self.drive_state = DriveState.LANE_FOLLOWING

    def calculate_control(self, detection_result, 
                         drive_state, turn_direction, 
                         image_width,):
        """
        Calculate steering and speed based on lane detection and state.
        
        Args:
            detection_result: Results from lane detection
            drive_state: Current drive state
            turn_direction: Direction to turn if in EXECUTING_TURN state
            image_width: Width of the input image            
        """
        if drive_state == DriveState.LANE_FOLLOWING:
            # Normal lane following
            steering = self.calculate_steering(detection_result.lane_center, image_width)
            
            return steering
            
        elif drive_state == DriveState.EXECUTING_TURN:
            
            # Calculate steering based on turn direction, respecting the 3:1 rule
            if turn_direction == 'right':
                steering = -self.angular
            else:  # left turn
                steering = self.angular
                
            return steering
        
        # Default fallback
        return 0.0, self.speed
    
    def calculate_steering(self, lane_center, image_width):
        """
        Calculate steering command using PID controller.
        
        Args:
            lane_center: X-coordinate of lane center, or None if not detected
            image_width: Width of the input image
        """

        # Calculate error (normalized to [-1, 1])
        error = (lane_center - image_width / 2) / (image_width / 2)
                
        # PID control
        self.integral += error
        # Anti-windup: limit the integral term
        self.integral = max(-2.0, min(2.0, self.integral))
        
        derivative = error - self.last_error
        self.last_error = error
        
        # Calculate steering output
        steering = -(self.kp * error + 
                     self.ki * self.integral + 
                     self.kd * derivative)
        
        # Limit steering to max angle
        steering = max(-self.angular, min(self.angular, steering))
        
        return steering
    
    def run(self):
        """Main loop."""
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    try:
        lane_follower = LaneFollowerNode()
        lane_follower.run()
    except rospy.ROSInterruptException:
        pass