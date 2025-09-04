#!/usr/bin/env python
import rospy
import rosbag
import tf
import numpy as np
import cv2
# import os

from sensor_msgs.msg import CompressedImage, PointCloud2, Image, LaserScan
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, PoseWithCovariance
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from laser_geometry import LaserProjection
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Twist
import ros_numpy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool

LASER_DIST_THRESHOLD = 0.30 
DEPTH_DIST_THRESHOLD = 0.30 

class obj_dist_est():

    def __init__(self):
        rospy.init_node('obj_dist_est', anonymous=True)

        self.cv_bridge = CvBridge()

        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.laser_det = False
        self.depth_det = False
        self.det = False
        
        rospy.Subscriber('/robot_1/scan_raw', LaserScan, self.laser_callback)
        rospy.Subscriber('/robot_1/depth_cam/depth/points', PointCloud2, self.depth_callback)
        self.pub = rospy.Publisher('/object_detected', Bool, queue_size=1)
        rospy.Timer(rospy.Duration(0.1), self.publish)

    def publish(self, event):
        obj_detected = Bool()
        if self.laser_det or self.depth_det:
            obj_detected.data = True
            self.pub.publish(obj_detected)
        else:
            obj_detected.data = False
            self.pub.publish(obj_detected)
    
    def depth_callback(self, msg):
        points_arr = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    
        points = np.zeros((points_arr.shape[0], 3), dtype=np.float32)
        points[:, 0] = points_arr['x']
        points[:, 1] = points_arr['y']
        points[:, 2] = points_arr['z']
        
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        
        if points.size == 0:
            return

        distances = np.linalg.norm(points, axis=1)
        min_dist = np.min(distances)

    def laser_callback(self, msg):
        ranges = msg.ranges
        mini = msg.range_min
        maxi = msg.range_max

        min_dist = float('inf')

        for i in range(0, 10): 
            min_dist = min(ranges[360 - i - 1], min_dist)
            min_dist = min(ranges[i], min_dist)

        if min_dist < LASER_DIST_THRESHOLD:
            self.laser_det = True
        else:
            self.laser_det = False

if __name__ == '__main__':

    proc = obj_dist_est()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("shutting down")