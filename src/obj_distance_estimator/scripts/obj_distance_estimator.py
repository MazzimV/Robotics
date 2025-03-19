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

ROI = 10

# 25 cm
LASER_DIST_THRESHOLD = 0.40 

DEPTH_DIST_THRESHOLD = 0.50 

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


    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)

    def move_robot(self):
        twist = Twist()
        twist.linear.x = 0.2
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)

    def check_fusion(self):
        rospy.loginfo(f"fusion {self.laser_det} {self.depth_det}")
        if self.laser_det or self.depth_det:
            self.det = True
            self.stop_robot()
            return
        self.det = False
        self.move_robot()
    
    def depth_callback(self, msg):
        pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    
        points = np.zeros((pc_array.shape[0], 3), dtype=np.float32)
        points[:, 0] = pc_array['x']
        points[:, 1] = pc_array['y']
        points[:, 2] = pc_array['z']
        
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        
        if points.size == 0:
            return

        distances = np.linalg.norm(points, axis=1)
        min_dist = np.min(distances)
        rospy.loginfo("DEPTH dist: {:.2f} m", min_dist)
        # self.check_fusion()

    def laser_callback(self, msg):
        ranges = msg.ranges
        mini = msg.range_min
        maxi = msg.range_max

        min_dist = float('inf')

        for i in range(0, 30): 
            min_dist = min(ranges[360 - i - 1], min_dist)
            min_dist = min(ranges[i], min_dist)

        if min_dist < LASER_DIST_THRESHOLD:
            self.laser_det = True
        else:
            self.laser_det = False

        rospy.loginfo("LIDAR DIST: %.2f m", min_dist)
        # self.check_fusion()

       

if __name__ == '__main__':

    proc = obj_dist_est()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("shutting down")
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        proc.vel_pub.publish(twist)