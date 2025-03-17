#!/usr/bin/env python
import rospy
import rosbag

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3, Twist

class laser_scan_navigator():

    def __init__(self):
        rospy.init_node('laser_scan_navigator')

        self.bag = rosbag.Bag("laser_scan.bag", 'w')
        
        rospy.Subscriber('/robot_1/scan_raw', LaserScan, self.laser_scan_callback)

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def check_ahead(self, ranges, min, max):
        can_move = False
        min_distance_side = float("inf")
        max_distance_side = 0
        min_distance_front = float("inf")
        max_distance_front = 0
        for i in range(0, 180):
            index = (270 + i) % 360
            if 315 >= index > 270 or 45 < index <= 90:
                if ranges[index] < min_distance_side and min <= ranges[index] <= max:
                    min_distance_side = ranges[index]
                if ranges[index] > max_distance_side and min <= ranges[index] <= max:
                    max_distance_side = ranges[index]
            else:
                if ranges[index] < min_distance_front and min <= ranges[index] <= max:
                    min_distance_front = ranges[index]
                if ranges[index] > max_distance_front and min <= ranges[index] <= max:
                    max_distance_front = ranges[index]
        if float("inf") > min_distance_side > 0.01 and float("inf") > min_distance_front > 0.3:
            can_move = True
        else:
            can_move = False
        rospy.loginfo(f"{min_distance_front}, {max_distance_front}, {min_distance_side}, {max_distance_side}")
        return can_move

    def laser_scan_callback(self, msg):
        can_move = self.check_ahead(msg.ranges, msg.range_min, msg.range_max)

        rospy.loginfo(can_move)
        
        forwardMsg = Twist()
        # Copy state into twist message.
        forwardMsg.linear.x = 0.1
        forwardMsg.linear.y = 0
        forwardMsg.linear.z = 0
        forwardMsg.angular.x = 0
        forwardMsg.angular.y = 0
        forwardMsg.angular.z = 0

        stopMsg = Twist()
        # Copy state into twist message.
        stopMsg.linear.x = -0.3
        stopMsg.linear.y = 0
        stopMsg.linear.z = 0
        stopMsg.angular.x = 0
        stopMsg.angular.y = 0
        stopMsg.angular.z = -0.3

        if can_move:
            self.pub.publish(forwardMsg)
        else:
            self.pub.publish(stopMsg)


if __name__ == '__main__':

    proc = laser_scan_navigator()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("shutting down")