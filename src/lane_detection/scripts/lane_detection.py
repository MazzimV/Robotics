import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist

class LaneDetection:
    def __init__(self):
        self.bridge = CvBridge()
        # Subscribe to camera feed
        self.image_sub = rospy.Subscriber('/robot_1/depth_cam/rgb/image_raw', Image, self.image_callback)
        # Publisher for robot movement commands
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Coefficients for the PID controller
        self.kp = 0.005
        self.ki = 0.0001
        self.kd = 0.001

        # Variables used to keep track of cumulative sums required for PID control
        self.prev_error = 0
        self.integral = 0
        self.prev_time = rospy.Time.now().to_sec()

    def detect_yellow_lanes(self, image):
        # Convert to HSV color space for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define yellow color range in HSV
        lower_yellow = np.array([10, 40, 100])
        upper_yellow = np.array([35, 255, 255])
        
        # Create mask for yellow
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Apply mask to get just the yellow parts
        yellow_detected = cv2.bitwise_and(image, image, mask=yellow_mask)
        
        return yellow_mask, yellow_detected
    
    def process_lanes(self, binary_image):
        height, width = binary_image.shape
        
        # Define regions of interest (bottom half of image)
        roi_height = height // 10
        left_roi = binary_image[5*roi_height:, :width // 2]
        right_roi = binary_image[5*roi_height:, width // 2:]
        # centre_roi = binary_image[6*roi_height:, 2*width // 5:3*width // 5]
        
        # Find lane line positions (centroids of detected yellow)
        left_points = np.argwhere(left_roi > 0)
        right_points = np.argwhere(right_roi > 0)
        # centre_points = np.argwhere(centre_roi > 0)

        rospy.loginfo(f"{len(left_points)}, {left_roi.shape}, {right_roi.shape}, {len(right_points)}")
        
        left_x = 0
        right_x = width
        # if len(centre_points) > 0:
        #     deviation = -20
        #     return deviation
        if len(left_points) > 0:
            left_x = np.mean(left_points[:, 1])
        if len(right_points) > 0:
            right_x = width//2 + np.mean(right_points[:, 1])
        # if len(left_points) > 3 * len(right_points):
        #     deviation = -20
        #     return deviation
        # if len(right_points) > 3 * len(left_points):
        #     deviation = 20
        #     return deviation

        rospy.loginfo(f"{len(left_points)}, {left_points[0]}, {len(right_points)}, {right_x}")
        
        # Calculate center point between lanes
        center_point = (left_x + right_x) // 2
        
        # Calculate deviation from ideal center
        deviation = width//2 - center_point

        rospy.loginfo(f"{deviation}, {center_point}, {width//2}")
        
        return deviation
    
    def control_robot(self, deviation):
        twist = Twist()

        # TODO: incorporate laser scan in lane_detection
        # if laser_scan_navigator.can_move:
        #     self.cmd_pub.publish(twist)
        # else:
            
        # Forward velocity (constant)
        twist.linear.x = 0.2  # Adjust based on your robot

        current_time = rospy.Time.now().to_sec()
        dt = current_time - self.prev_time

        if dt > 0:
            p = self.kp * deviation
            
            # Get a cumulative sum (integral) of the deviation w.r.t. time
            self.integral += deviation * dt
            
            # Limit the integral so that it does not keep growing
            self.integral = max(-75, min(75, self.integral))
            i = self.ki * self.integral

            # Calculate the rate of change of the error
            d = self.kd * (deviation - self.prev_error) / dt

            # Update the previous values
            self.prev_error = deviation
            self.prev_time = current_time
    
            twist.angular.z = max(-(3 * twist.linear.x), min((3 * twist.linear.x), p + i + d))

        rospy.loginfo(str(twist.linear.x) + " " + str(twist.angular.z))

        # Publish command
        self.cmd_pub.publish(twist)

    def image_callback(self, data):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Process image
            yellow_mask, yellow_detected = self.detect_yellow_lanes(cv_image)
            
            # Calculate deviation
            deviation = self.process_lanes(yellow_mask)
            
            # Control robot
            self.control_robot(deviation)
            
            # Visualization
            cv2.imshow("Original", cv_image)
            cv2.imshow("Yellow Detection", yellow_detected)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node('lane_detection')
    detector = LaneDetection()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()