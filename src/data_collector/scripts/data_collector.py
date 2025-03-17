#!/usr/bin/env python
import rospy
import rosbag
import cv2

from sensor_msgs.msg import CompressedImage, PointCloud2, Image, LaserScan
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, PoseWithCovariance, Twist
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from laser_geometry import LaserProjection
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from skimage.metrics import structural_similarity as ssim

class data_collector():

    def __init__(self):
        rospy.init_node('data_collector')
        self.bridge = CvBridge()
        self.previous_image = None

        rospy.Subscriber('/robot_1/depth_cam/rgb/image_raw', Image, self.camera_callback)

    def camera_callback(self, msg):
        try:
            cv_message = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.previous_image is None:
                self.previous_image = cv_message
            else:
                similarity_score = ssim(self.previous_image, cv_message, channel_axis=2)
                if similarity_score <= 0.75:
                    cv2.imwrite(f'./image_data/{rospy.Time.now().to_sec()}.jpg', self.previous_image)
                    self.previous_image = cv_message
        except CvBridgeError as e:
            rospy.logdebug(e)

if __name__ == '__main__':

    proc = data_collector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("shutting down")



