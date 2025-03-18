#!/usr/bin/env python
import rospy

from sensor_msgs.msg import LaserScan

class laser_scan_navigator():

    def __init__(self):
        rospy.init_node('laser_scan_navigator')
        
        rospy.Subscriber('/robot_1/scan_raw', LaserScan, self.laser_scan_callback)

        self.can_move = False

    def check_ahead(self, ranges, min, max):
        min_distance = float("inf")
        
        for i in range(0, 180):
            index = (270 + i) % 360
            if ranges[index] < min_distance and min <= ranges[index] <= max:
                min_distance = ranges[index]

        if float("inf") > min_distance > 0.2:
            self.can_move = True
        else:
            self.can_move = False

    def laser_scan_callback(self, msg):
        self.check_ahead(msg.ranges, msg.range_min, msg.range_max)

        rospy.loginfo(f"Lidar can_move: {self.can_move}")

if __name__ == '__main__':

    proc = laser_scan_navigator()

    def get_stop_message():
        return proc.can_move

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("shutting down")