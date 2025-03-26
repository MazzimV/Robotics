#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

class DriveState(Enum):
    """
    Enum representing the possible driving states of the robot.
    """
    LANE_FOLLOWING = auto()
    LANE_ENDING = auto()
    EXECUTING_TURN = auto()
    LANE_RECOVERY = auto()  # New state for handling lost lane scenarios


@dataclass
class LaneDetectionResult:
    """
    Data class to store lane detection results.
    """
    lane_center: Optional[float] = None
    lane_quality: float = 0.0
    is_lane_ending: bool = False
    left_count: float = 0.0
    right_count: float = 0.0
    debug_image: Optional[np.ndarray] = None
    error: Optional[float] = None


@dataclass
class ROIConfig:
    """
    Configuration for Region of Interest parameters.
    """
    # Main ROI
    roi_top_ratio: float = 0.5
    roi_bottom_ratio: float = 0.95
    roi_width_ratio: float = 0.9
    
    # Look ahead ROI
    look_ahead_roi_top_ratio: float = 0.45
    look_ahead_roi_bottom_ratio: float = 0.55
    look_ahead_width_ratio: float = 0.4


@dataclass
class ColorConfig:
    """
    Configuration for color detection parameters.
    """
    yellow_min: np.ndarray = np.array([10, 70, 110], dtype=np.uint8)
    yellow_max: np.ndarray = np.array([35, 255, 255], dtype=np.uint8)


@dataclass
class ControlConfig:
    """
    Configuration for control parameters.
    """
    # Speed parameters
    max_speed: float = 0.0
    min_speed: float = 0.0
    
    # Steering parameters
    max_angle: float = 1.2
    max_steering_change: float = 0.3
    
    # Adaptive speed parameters
    enable_adaptive_speed: bool = True
    curve_speed_factor: float = 0.7
    
    # PID parameters
    kp: float = 0.6
    ki: float = 0.0
    kd: float = 0.0
    
    # Filtering parameters
    steering_history_size: int = 3
    error_history_size: int = 5


@dataclass
class LaneConfig:
    """
    Configuration for lane detection parameters.
    """
    min_lane_points: int = 50
    max_lost_frames: int = 10
    turn_duration: float = 3.0
    lane_width_estimate_ratio: float = 0.4  # Lane width as a ratio of image width


class LaneDetector:
    """
    Responsible for detecting lanes in camera images.
    """
    def __init__(self, roi_config: ROIConfig, color_config: ColorConfig, lane_config: LaneConfig):
        """
        Initialize the lane detector with configuration objects.
        
        Args:
            roi_config: Configuration for regions of interest
            color_config: Configuration for color detection
            lane_config: Configuration for lane detection parameters
        """
        self.roi_config = roi_config
        self.color_config = color_config
        self.lane_config = lane_config

    def preprocess_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Preprocess the image by defining ROIs and converting to HSV.
        
        Args:
            image: The input camera image
            
        Returns:
            Dictionary containing preprocessed image components
        """
        height, width, _ = image.shape
        
        # Define primary region of interest
        roi_top = int(height * self.roi_config.roi_top_ratio)
        roi_bottom = int(height * self.roi_config.roi_bottom_ratio)
        roi_width = int(width * self.roi_config.roi_width_ratio)
        roi_left = int((width - roi_width) / 2)
        roi_right = int(roi_left + roi_width)
        
        # Define look-ahead region of interest
        look_ahead_top = int(height * self.roi_config.look_ahead_roi_top_ratio)
        look_ahead_bottom = int(height * self.roi_config.look_ahead_roi_bottom_ratio)
        look_ahead_width = int(width * self.roi_config.look_ahead_width_ratio)
        look_ahead_left = int((width - look_ahead_width) / 2)
        look_ahead_right = int(look_ahead_left + look_ahead_width)
        
        # Split ROI for left and right lane detection
        roi_width_pixels = roi_right - roi_left
        roi_mid = roi_left + roi_width_pixels//2
        
        # Extract ROIs (with safety checks)
        try:
            roi = image[roi_top:roi_bottom, roi_left:roi_right]
            look_ahead_roi = image[look_ahead_top:look_ahead_bottom, look_ahead_left:look_ahead_right]
            roi_left_half = image[roi_top:roi_bottom, roi_left:roi_mid]
            roi_right_half = image[roi_top:roi_bottom, roi_mid:roi_right]
            look_ahead_left_half = image[look_ahead_top:look_ahead_bottom, look_ahead_left:look_ahead_left + look_ahead_width//2]
            look_ahead_right_half = image[look_ahead_top:look_ahead_bottom, look_ahead_left + look_ahead_width//2:look_ahead_right]
        except IndexError as e:
            rospy.logerr(f"ROI indexing error: {e}")
            return {}
            
        # Verify ROIs are not empty
        if roi.size == 0 or look_ahead_roi.size == 0:
            rospy.logerr("ROI is empty! Check image dimensions and ROI parameters.")
            return {}
        
        # Convert to HSV for better color detection
        try:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv_look_ahead = cv2.cvtColor(look_ahead_roi, cv2.COLOR_BGR2HSV)
            hsv_left = cv2.cvtColor(roi_left_half, cv2.COLOR_BGR2HSV)
            hsv_right = cv2.cvtColor(roi_right_half, cv2.COLOR_BGR2HSV)
            hsv_look_ahead_left = cv2.cvtColor(look_ahead_left_half, cv2.COLOR_BGR2HSV)
            hsv_look_ahead_right = cv2.cvtColor(look_ahead_right_half, cv2.COLOR_BGR2HSV)
        except cv2.error as e:
            rospy.logerr(f"OpenCV error in color conversion: {e}")
            return {}
        
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

    def create_color_masks(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create color masks for lane detection.
        
        Args:
            processed_data: Dictionary with preprocessed image data
            
        Returns:
            Dictionary with added color mask data
        """
        if not processed_data:
            return {}
            
        # Create masks for yellow lane markers
        yellow_mask = cv2.inRange(
            processed_data['hsv_roi'], 
            self.color_config.yellow_min, 
            self.color_config.yellow_max
        )
        
        look_ahead_yellow_mask = cv2.inRange(
            processed_data['hsv_look_ahead'], 
            self.color_config.yellow_min, 
            self.color_config.yellow_max
        )
        
        left_yellow_mask = cv2.inRange(
            processed_data['hsv_left'], 
            self.color_config.yellow_min, 
            self.color_config.yellow_max
        )
        
        right_yellow_mask = cv2.inRange(
            processed_data['hsv_right'], 
            self.color_config.yellow_min, 
            self.color_config.yellow_max
        )
        
        left_look_ahead_yellow_mask = cv2.inRange(
            processed_data['hsv_look_ahead_left'], 
            self.color_config.yellow_min, 
            self.color_config.yellow_max
        )
        
        right_look_ahead_yellow_mask = cv2.inRange(
            processed_data['hsv_look_ahead_right'], 
            self.color_config.yellow_min, 
            self.color_config.yellow_max
        )
        
        # Clean up masks
        kernel = np.ones((5, 5), np.uint8)
        processed_data.update({
            'yellow_mask': self._clean_mask(yellow_mask, kernel),
            'look_ahead_yellow_mask': self._clean_mask(look_ahead_yellow_mask, kernel),
            'left_yellow_mask': self._clean_mask(left_yellow_mask, kernel),
            'right_yellow_mask': self._clean_mask(right_yellow_mask, kernel),
            'left_look_ahead_yellow_mask': self._clean_mask(left_look_ahead_yellow_mask, kernel),
            'right_look_ahead_yellow_mask': self._clean_mask(right_look_ahead_yellow_mask, kernel),
            'kernel': kernel
        })
        
        return processed_data
    
    def _clean_mask(self, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up a mask.
        
        Args:
            mask: The mask to clean
            kernel: Kernel for morphological operations
            
        Returns:
            Cleaned mask
        """
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
    
    def find_contours(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find contours in the color masks.
        
        Args:
            processed_data: Dictionary with mask data
            
        Returns:
            Dictionary with added contour data
        """
        if not processed_data:
            return {}
            
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
    
    def analyze_lane_position(self, processed_data: Dict[str, Any]) -> LaneDetectionResult:
        """
        Analyze the detected lane markers to find lane position.
        
        Args:
            processed_data: Dictionary with contour data
            
        Returns:
            LaneDetectionResult object with detection results
        """
        if not processed_data:
            return LaneDetectionResult()
            
        # Create a copy for visualization
        debug_image = processed_data['original_image'].copy()
        
        # Calculate yellow pixel counts
        yellow_pixel_count = np.sum(processed_data['yellow_mask'] > 0)
        look_ahead_yellow_pixel_count = np.sum(processed_data['look_ahead_yellow_mask'] > 0)
        left_yellow_pixel_count = np.sum(processed_data['left_yellow_mask'] > 0)
        right_yellow_pixel_count = np.sum(processed_data['right_yellow_mask'] > 0)
        left_look_ahead_yellow_pixel_count = np.sum(processed_data['left_look_ahead_yellow_mask'] > 0)
        right_look_ahead_yellow_pixel_count = np.sum(processed_data['right_look_ahead_yellow_mask'] > 0)
        
        # Calculate lane quality
        lane_quality = yellow_pixel_count / (processed_data['yellow_mask'].shape[0] * processed_data['yellow_mask'].shape[1])
        look_ahead_lane_quality = look_ahead_yellow_pixel_count / (processed_data['look_ahead_yellow_mask'].shape[0] * processed_data['look_ahead_yellow_mask'].shape[1])
        
        # Check for horizontal lines that indicate intersections
        is_lane_ending = self._detect_lane_ending(processed_data, debug_image)
        
        # Check if contours exist and quality is good
        if (not processed_data['contours'] or 
                yellow_pixel_count < self.lane_config.min_lane_points):
            return LaneDetectionResult(
                lane_center=None,
                lane_quality=lane_quality,
                is_lane_ending=is_lane_ending,
                left_count=left_look_ahead_yellow_pixel_count,
                right_count=right_look_ahead_yellow_pixel_count,
                debug_image=debug_image
            )
        
        # Find lane center
        lane_center_x = self._calculate_lane_center(
            processed_data, 
            left_yellow_pixel_count, 
            right_yellow_pixel_count,
            debug_image
        )
        
        # If no lane center determined, return None
        if lane_center_x is None:
            return LaneDetectionResult(
                lane_center=None,
                lane_quality=lane_quality,
                is_lane_ending=is_lane_ending,
                left_count=left_look_ahead_yellow_pixel_count,
                right_count=right_look_ahead_yellow_pixel_count,
                debug_image=debug_image
            )
        
        # Calculate error from center
        error = (lane_center_x - processed_data['width'] / 2) / (processed_data['width'] / 2)
        
        return LaneDetectionResult(
            lane_center=lane_center_x,
            lane_quality=lane_quality,
            is_lane_ending=is_lane_ending,
            left_count=left_look_ahead_yellow_pixel_count,
            right_count=right_look_ahead_yellow_pixel_count,
            debug_image=debug_image,
            error=error
        )
    
    def _detect_lane_ending(self, processed_data: Dict[str, Any], debug_image: np.ndarray) -> bool:
        """
        Detect if a lane is ending or if there's an intersection.
        
        Args:
            processed_data: Dictionary with processed image data
            debug_image: Image for visualization
            
        Returns:
            True if lane is ending, False otherwise
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
                              processed_data: Dict[str, Any], 
                              left_yellow_pixel_count: int, 
                              right_yellow_pixel_count: int,
                              debug_image: np.ndarray) -> Optional[float]:
        """
        Calculate the center of the lane based on detected contours.
        
        Args:
            processed_data: Dictionary with processed image data
            left_yellow_pixel_count: Number of yellow pixels in left half
            right_yellow_pixel_count: Number of yellow pixels in right half
            debug_image: Image for visualization
            
        Returns:
            X-coordinate of lane center, or None if not found
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
        lane_width_estimate = int(width * self.lane_config.lane_width_estimate_ratio)
        
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
            if left_yellow_pixel_count > 500 and right_yellow_pixel_count < 50:
                lane_center_x = left_x + lane_width_estimate
                cv2.putText(
                    debug_image, "LEFT ONLY - ADJUST RIGHT", 
                    (width//2 - 120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
                )
            else:
                # Normal case - estimate center as half lane width from left
                lane_center_x = left_x + lane_width_estimate/2
                cv2.putText(
                    debug_image, "LEFT ONLY", 
                    (width//2 - 120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                )
        
        elif left_x is None and right_x is not None:
            # Only right boundary detected - estimate left boundary
            lane_center_x = right_x - lane_width_estimate/2
            cv2.putText(
                debug_image, "RIGHT ONLY", 
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
                             contours: List[np.ndarray], 
                             x_offset: int, 
                             y_offset: int,
                             debug_image: Optional[np.ndarray] = None,
                             color: tuple = (0, 255, 0)) -> Optional[float]:
        """
        Calculate the average x-coordinate of contour centers.
        
        Args:
            contours: List of contours
            x_offset: X-coordinate offset
            y_offset: Y-coordinate offset
            debug_image: Optional image for visualization
            color: Color for visualization
            
        Returns:
            Average x-coordinate, or None if no valid contours
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
                                  result: LaneDetectionResult, 
                                  processed_data: Dict[str, Any],
                                  drive_state: DriveState,
                                  fps: float,
                                  lost_frame_count: int) -> np.ndarray:
        """
        Create visualization for debugging.
        
        Args:
            result: Lane detection result
            processed_data: Dictionary with processed image data
            drive_state: Current drive state
            fps: Current frames per second
            lost_frame_count: Number of consecutive frames with lost lane
            
        Returns:
            Debug image with visualizations
        """
        if not processed_data or result.debug_image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        debug_image = result.debug_image
        width, height = processed_data['width'], processed_data['height']
        
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
            debug_image, f"Lane quality: {result.lane_quality:.3f}", 
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        
        look_ahead_lane_quality = np.sum(processed_data['look_ahead_yellow_mask'] > 0) / (
            processed_data['look_ahead_yellow_mask'].shape[0] * 
            processed_data['look_ahead_yellow_mask'].shape[1]
        )
        
        cv2.putText(
            debug_image, f"Look ahead quality: {look_ahead_lane_quality:.3f}", 
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
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
        
        # Display lane lost warning
        if not result.lane_center and lost_frame_count > 0:
            if lost_frame_count > self.lane_config.max_lost_frames:
                cv2.putText(
                    debug_image, "Lane lost for too long!", 
                    (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
            else:
                cv2.putText(
                    debug_image, "No lane markers detected", 
                    (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
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
        
        # Display FPS
        cv2.putText(
            debug_image, f"FPS: {fps:.1f}", 
            (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        
        return debug_image
    
    def detect_lane(self, image: np.ndarray, 
                   drive_state: DriveState, 
                   fps: float, 
                   lost_frame_count: int) -> LaneDetectionResult:
        """
        Main lane detection method that orchestrates the detection pipeline.
        
        Args:
            image: Input camera image
            drive_state: Current drive state
            fps: Current frames per second
            lost_frame_count: Number of consecutive frames with lost lane
            
        Returns:
            LaneDetectionResult with detection results
        """
        try:
            # Execute detection pipeline
            processed_data = self.preprocess_image(image)
            if not processed_data:
                return LaneDetectionResult(debug_image=image)
                
            processed_data = self.create_color_masks(processed_data)
            if not processed_data:
                return LaneDetectionResult(debug_image=image)
                
            processed_data = self.find_contours(processed_data)
            if not processed_data:
                return LaneDetectionResult(debug_image=image)
                
            result = self.analyze_lane_position(processed_data)
            
            # Create debug visualization
            debug_image = self.create_debug_visualization(
                result, processed_data, drive_state, fps, lost_frame_count
            )
            result.debug_image = debug_image
            
            return result
            
        except Exception as e:
            rospy.logerr(f"Lane detection error: {e}")
            return LaneDetectionResult(debug_image=image)


class DriveStateManager:
    """
    Manages the driving state of the robot based on lane detection results.
    """
    def __init__(self, lane_config: LaneConfig):
        """
        Initialize the drive state manager.
        
        Args:
            lane_config: Configuration for lane detection parameters
        """
        self.lane_config = lane_config
        self.drive_state = DriveState.LANE_FOLLOWING
        self.turn_start_time = None
        self.turn_direction = None
        self.lost_frame_count = 0
    
    def update_state(self, detection_result: LaneDetectionResult, current_time: float, last_error: float) -> None:
        """
        Update the driving state based on lane detection results.
        
        Args:
            detection_result: Results from lane detection
            current_time: Current time
            last_error: Last known position error
        """
        # Debug output for lane ending
        if detection_result.is_lane_ending:
            rospy.loginfo(
                f"LANE ENDING DETECTED: left={detection_result.left_count}, right={detection_result.right_count}"
            )
        
        # Update lost frame count
        if detection_result.lane_center is None:
            self.lost_frame_count += 1
            if self.lost_frame_count > self.lane_config.max_lost_frames:
                if self.drive_state != DriveState.LANE_RECOVERY:
                    self.drive_state = DriveState.LANE_RECOVERY
                    rospy.loginfo("Entering lane recovery mode")
        else:
            self.lost_frame_count = 0
            if self.drive_state == DriveState.LANE_RECOVERY:
                self.drive_state = DriveState.LANE_FOLLOWING
                rospy.loginfo("Lane recovered - resuming lane following")
        
        # Handle lane ending detection
        if detection_result.is_lane_ending and self.drive_state != DriveState.EXECUTING_TURN:
            self.drive_state = DriveState.EXECUTING_TURN
            self.turn_start_time = current_time
            
            # Determine turn direction based on which lane is more visible
            if detection_result.left_count > detection_result.right_count + 100:
                self.turn_direction = 'right'  # Turn toward the missing right lane
                rospy.loginfo(
                    f"Left lane stronger ({detection_result.left_count} vs {detection_result.right_count}) - turning RIGHT"
                )
            elif detection_result.right_count > detection_result.left_count + 100:
                self.turn_direction = 'left'  # Turn toward the missing left lane
                rospy.loginfo(
                    f"Right lane stronger ({detection_result.right_count} vs {detection_result.left_count}) - turning LEFT"
                )
            else:
                # If it's ambiguous, use position relative to lane
                if last_error > 0:  # We're to the right of center
                    self.turn_direction = 'right'
                else:  # We're to the left of center
                    self.turn_direction = 'left'
                rospy.loginfo(f"Ambiguous lane detection - turning {self.turn_direction} based on position")
        
        # Turn completion check
        elif self.drive_state == DriveState.EXECUTING_TURN:
            if current_time - self.turn_start_time > self.lane_config.turn_duration:
                self.drive_state = DriveState.LANE_FOLLOWING
                rospy.loginfo("Turn complete - resuming lane following")


class MotionController:
    """
    Controls the robot's motion based on lane detection and state.
    """
    def __init__(self, control_config: ControlConfig):
        """
        Initialize the motion controller.
        
        Args:
            control_config: Configuration for control parameters
        """
        self.config = control_config
        
        # Initialize control state
        self.integral = 0.0
        self.last_error = 0.0
        self.error_history = []
        self.steering_history = []
        self.last_good_steering = 0.0
    
    def calculate_control(self, detection_result: LaneDetectionResult, 
                         drive_state: DriveState, turn_direction: Optional[str], 
                         image_width: int, current_time: float) -> Tuple[float, float]:
        """
        Calculate steering and speed based on lane detection and state.
        
        Args:
            detection_result: Results from lane detection
            drive_state: Current drive state
            turn_direction: Direction to turn if in EXECUTING_TURN state
            image_width: Width of the input image
            current_time: Current time
            
        Returns:
            Tuple of (steering_command, speed_command)
        """
        if drive_state == DriveState.LANE_FOLLOWING:
            # Normal lane following
            steering = self.calculate_steering(detection_result.lane_center, image_width)
            speed = self.calculate_speed(steering)
            
            # Ensure steering respects the 3:1 ratio of angular:linear speed
            max_angular_speed = 3.0 * speed
            steering = max(-max_angular_speed, min(max_angular_speed, steering))
            
            return steering, speed
            
        elif drive_state == DriveState.EXECUTING_TURN:
            # Set a reasonable linear speed for turning
            speed = self.config.min_speed
            
            # Maximum angular speed based on the 3:1 rule
            max_angular_speed = 3.25 * speed
            
            # Calculate steering based on turn direction, respecting the 3:1 rule
            if turn_direction == 'right':
                steering = max(-self.config.max_angle, -max_angular_speed)
            else:  # left turn
                steering = min(self.config.max_angle, max_angular_speed)
                
            return steering, speed
            
        elif drive_state == DriveState.LANE_RECOVERY:
            # Implementation for recovery - e.g., slow down and do a small sweep
            speed = self.config.min_speed * 0.75  # Slower speed during recovery
            
            # Use a time-based oscillation for recovery
            oscillation_period = 4.0  # seconds for a complete sweep
            oscillation_amplitude = self.config.max_angle * 0.5  # reduced amplitude
            
            # Calculate a sweeping motion based on time
            steering = oscillation_amplitude * np.sin(2 * np.pi * current_time / oscillation_period)
            
            return steering, speed
        
        # Default fallback
        return 0.0, self.config.min_speed
    
    def calculate_steering(self, lane_center: Optional[float], image_width: int) -> float:
        """
        Calculate steering command using PID controller.
        
        Args:
            lane_center: X-coordinate of lane center, or None if not detected
            image_width: Width of the input image
            
        Returns:
            Steering command
        """
        if lane_center is None:
            # If lane not detected, use the last known good steering value
            # with a small decay to eventually straighten out if lane remains lost
            decay_factor = 0.95
            steering = self.last_good_steering * decay_factor
            rospy.logwarn_throttle(1.0, "Lane not detected, using decayed steering value")
            return steering
        
        # Calculate error (normalized to [-1, 1])
        error = (lane_center - image_width / 2) / (image_width / 2)
        
        # Update error history
        self.error_history.append(error)
        if len(self.error_history) > self.config.error_history_size:
            self.error_history.pop(0)
        
        # Use median filtering on the error for robustness to outliers
        filtered_error = np.median(self.error_history)
        
        # PID control
        self.integral += filtered_error
        # Anti-windup: limit the integral term
        self.integral = max(-2.0, min(2.0, self.integral))
        
        derivative = filtered_error - self.last_error
        self.last_error = filtered_error
        
        # Calculate steering output
        steering = -(self.config.kp * filtered_error + 
                     self.config.ki * self.integral + 
                     self.config.kd * derivative)
        
        # Limit the maximum steering change rate for smoother control
        steering_change = steering - self.last_good_steering
        if abs(steering_change) > self.config.max_steering_change:
            steering = self.last_good_steering + np.sign(steering_change) * self.config.max_steering_change
        
        # Add to steering history for smoothing
        self.steering_history.append(steering)
        if len(self.steering_history) > self.config.steering_history_size:
            self.steering_history.pop(0)
        
        # Apply moving average to steering
        smoothed_steering = sum(self.steering_history) / len(self.steering_history)
        
        # Limit steering to max angle
        steering = max(-self.config.max_angle, min(self.config.max_angle, smoothed_steering))
        
        # Update last good steering
        self.last_good_steering = steering
        
        return steering
    
    def calculate_speed(self, steering: float) -> float:
        """
        Calculate speed based on steering angle for safer cornering.
        
        Args:
            steering: Steering command
            
        Returns:
            Speed command
        """
        if self.config.enable_adaptive_speed:
            # Reduce speed in curves based on steering angle
            # The sharper the turn (higher abs(steering)), the slower we go
            steering_factor = 1.0 - min(1.0, abs(steering) / self.config.max_angle)
            adaptive_factor = (self.config.curve_speed_factor + 
                              (1.0 - self.config.curve_speed_factor) * steering_factor)
            speed = self.config.max_speed * adaptive_factor
            
            # Ensure we don't go below minimum speed
            speed = max(self.config.min_speed, speed)
            return speed
        else:
            # Fixed speed mode
            return self.config.max_speed


class LaneFollowerNode:
    """
    Main ROS node for lane following.
    """
    def __init__(self):
        """Initialize the lane follower node."""
        rospy.init_node('lane_follower')
        
        # Load parameters
        self._load_parameters()
        
        # Initialize components
        self.lane_detector = LaneDetector(self.roi_config, self.color_config, self.lane_config)
        self.state_manager = DriveStateManager(self.lane_config)
        self.motion_controller = MotionController(self.control_config)
        
        # Initialize bridge
        self.bridge = CvBridge()
        
        # Initialize state variables
        self.frame_count = 0
        self.last_process_time = time.time()
        self.fps = 0
        
        # Publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/lane_debug', Image, queue_size=1)
        rospy.Subscriber('/robot_1/depth_cam/rgb/image_raw', Image, self.image_callback)
        
        rospy.loginfo("Improved lane follower initialized")
    
    def _load_parameters(self):
        """Load parameters from ROS parameter server."""
        # Load ROI parameters
        self.roi_config = ROIConfig(
            roi_top_ratio=rospy.get_param('~roi_top_ratio', 0.5),
            roi_bottom_ratio=rospy.get_param('~roi_bottom_ratio', 0.95),
            roi_width_ratio=rospy.get_param('~roi_width_ratio', 0.9),
            look_ahead_roi_top_ratio=rospy.get_param('~look_ahead_roi_top_ratio', 0.45),
            look_ahead_roi_bottom_ratio=rospy.get_param('~look_ahead_roi_bottom_ratio', 0.55),
            look_ahead_width_ratio=rospy.get_param('~look_ahead_width_ratio', 0.4)
        )
        
        # Load color parameters
        yellow_min = np.array([5, 30, 30], dtype=np.uint8)
        yellow_max = np.array([30, 255, 255], dtype=np.uint8)
        
        self.color_config = ColorConfig(
            yellow_min=yellow_min,
            yellow_max=yellow_max
        )
        
        # Load control parameters
        self.control_config = ControlConfig(
            max_speed=rospy.get_param('~max_speed', 0.2),
            min_speed=rospy.get_param('~min_speed', 0.2),
            max_angle=rospy.get_param('~max_angle', 1.0),
            max_steering_change=rospy.get_param('~max_steering_change', 0.3),
            enable_adaptive_speed=rospy.get_param('~enable_adaptive_speed', True),
            curve_speed_factor=rospy.get_param('~curve_speed_factor', 0.7),
            kp=rospy.get_param('~kp', 0.6),
            ki=rospy.get_param('~ki', 0.0),
            kd=rospy.get_param('~kd', 0.0),
            steering_history_size=3,
            error_history_size=5
        )
        
        # Load lane parameters
        self.lane_config = LaneConfig(
            min_lane_points=rospy.get_param('~min_lane_points', 50),
            max_lost_frames=rospy.get_param('~max_lost_frames', 10),
            turn_duration=rospy.get_param('~turn_duration', 4.0),
            lane_width_estimate_ratio=rospy.get_param('~lane_width_estimate_ratio', 0.4)
        )
    
    def image_callback(self, data):
        """
        Process camera image for lane following.
        
        Args:
            data: ROS Image message
        """
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Verify image is not empty
            if cv_image is None or cv_image.size == 0:
                rospy.logerr("Received empty image from camera!")
                return
                
            # Calculate FPS
            current_time = time.time()
            self.fps = 1.0 / (current_time - self.last_process_time)
            self.last_process_time = current_time
            
            # Process image for lane detection
            detection_result = self.lane_detector.detect_lane(
                cv_image, 
                self.state_manager.drive_state, 
                self.fps, 
                self.state_manager.lost_frame_count
            )
            
            # Update driving state
            self.state_manager.update_state(
                detection_result, 
                current_time, 
                self.motion_controller.last_error
            )

            # Calculate steering and speed based on current state
            steering, speed = self.motion_controller.calculate_control(
                detection_result,
                self.state_manager.drive_state,
                self.state_manager.turn_direction,
                cv_image.shape[1],
                current_time
            )
            
            # Create and publish command
            cmd = Twist()
            cmd.linear.x = speed
            cmd.angular.z = steering
            rospy.loginfo(f"{cmd.linear.x} {cmd.angular.z}")
            self.cmd_vel_pub.publish(cmd)
            
            # Publish debug image
            if detection_result.debug_image is not None and self.debug_image_pub.get_num_connections() > 0:
                try:
                    self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(detection_result.debug_image, "bgr8"))
                except CvBridgeError as e:
                    rospy.logerr(f"Could not publish debug image: {e}")
            
            self.frame_count += 1
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
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