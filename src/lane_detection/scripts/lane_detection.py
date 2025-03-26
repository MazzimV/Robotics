import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class LaneDetection:
    def __init__(self):
        self.bridge = CvBridge()
        # Subscribe to camera feed
        self.image_sub = rospy.Subscriber('/robot_1/depth_cam/rgb/image_raw', Image, self.lane_callback)
        # Publisher for robot movement commands
        self.cmd_pub = rospy.Publisher('/turn', Bool, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/lane_debug', Image, queue_size=1)

    def lane_callback(self, msg):
        """
        Detect if a lane is ending or if there's an intersection.
        
        Args:
            processed_data: Dictionary with processed image data
            debug_image: Image for visualization
            
        Returns:
            True if lane is ending, False otherwise
        """

        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        debug_image = cv_image.copy()

        # Convert to HSV color space for better color segmentation
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Define yellow color range in HSV
        lower_yellow = np.array([10, 70, 110])
        upper_yellow = np.array([35, 255, 255])
        
        # Create mask for yellow
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Apply mask to get just the yellow parts
        yellow_detected = cv2.bitwise_and(cv_image, cv_image, mask=yellow_mask)
        
        height, width, _ = yellow_detected.shape

        # Define regions of interest
        roi_height = height // 10
        roi_width = width // 10
        look_ahead_roi = yellow_detected[4*roi_height:5*roi_height, 3*roi_width:6*roi_width]

        # Check for horizontal lines that indicate intersections
        horizontal_lines = cv2.HoughLinesP(
            yellow_mask, 
            1, np.pi/180, 50, 
            minLineLength=30, 
            maxLineGap=10
        )

        horizontal_line_detected = False
        if horizontal_lines is not None:
            for line in horizontal_lines:
                x1, y1, x2, y2 = line[0]
                # If line is mostly horizontal (small y difference)
                if abs(y2 - y1) < 5 and abs(x2 - x1) > 10:
                    horizontal_line_detected = True
                    # Draw the detected horizontal line
                    cv2.line(
                        debug_image, 
                        (x1 + 3*roi_width, y1 + 4*roi_height), 
                        (x2 + 3*roi_width, y2 + 4*roi_height), 
                        (0, 255, 255), 2
                    )
        
        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
        self.cmd_pub.publish(horizontal_line_detected)

if __name__ == '__main__':
    rospy.init_node('lane_detection')
    detector = LaneDetection()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()