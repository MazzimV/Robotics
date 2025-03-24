#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import time

class LidarLoop:
    def __init__(self):
        rospy.init_node('lidar_loop')
        
        # Parameters
        self.obstacle_distance = rospy.get_param('~obstacle_distance', 1.0)
        self.forward_speed = rospy.get_param('~forward_speed', 0.3)
        self.reverse_speed = rospy.get_param('~reverse_speed', 0.2)
        self.turn_speed = rospy.get_param('~turn_speed', 1.0)

        self.reverse_time = rospy.get_param('~reverse_time', 1.5)
        self.turn_time = rospy.get_param('~turn_time', 2.715)

        self.state = 'FORWARD'
        self.state_start_time = time.time()
        self.obstacle_detected = False

        # Create publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/robot_1/scan_raw', LaserScan, self.lidar_callback)
        
        rospy.loginfo("Simple LIDAR loop navigation initialized")
    
    def lidar_callback(self, msg):
        """Process LIDAR scan data for obstacle detection"""
        try:
            ranges = []
            for i in range(0, 35):
                index = (342 + i) % 360
                if msg.range_max >= msg.ranges[index] >= msg.range_min:
                    ranges.append(msg.ranges[index])
            
            if ranges:
                # Check if any range is below obstacle distance
                min_range = min(ranges)
                prev_obstacle = self.obstacle_detected
                self.obstacle_detected = min_range < self.obstacle_distance

                if self.obstacle_detected and not prev_obstacle:
                    rospy.loginfo(f"Obstacle detected at {min_range:.2f}m")
            
        except Exception as e:
            rospy.logerr(f"Error processing scan: {e}")
            self.obstacle_detected = True

    def update_state(self):
        current_time = time.time()
        time_in_state = current_time - self.state_start_time

        if self.state == 'FORWARD':
            if self.obstacle_detected:
                self.state = 'TURN_RIGHT'
                self.state_start_time = current_time
                rospy.loginfo("Switching to TURN RIGHT state")

        elif self.state == 'REVERSE':
            if time_in_state > self.reverse_time:
                self.state = 'TURN_RIGHT'
                self.state_start_time = current_time
                rospy.loginfo("Switching to TURN RIGHT state")
        
        elif self.state == 'TURN_RIGHT':
            if time_in_state > self.turn_time:
                self.state = 'FORWARD'
                self.state_start_time = current_time
                rospy.loginfo("Switching to FORWARD state")

    def publish_command(self):
        cmd = Twist()

        if self.state == 'FORWARD':
            cmd.linear.x = self.forward_speed
            cmd.angular.z = 0.0
        elif self.state == 'REVERSE':
            cmd.linear.x = -self.forward_speed
            cmd.angular.z = 0.0
        elif self.state == 'TURN_RIGHT':
            cmd.linear.x = self.forward_speed
            cmd.angular.z = -self.turn_speed
        
        self.cmd_vel_pub.publish(cmd)
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            self.update_state()
            self.publish_command()
            time_in_state = time.time() - self.state_start_time
            rospy.loginfo_throttle(1.0, f"State: {self.state}, Time in state: {time_in_state:.1f}s")

            rate.sleep()

if __name__ == '__main__':
    try:
        controller = LidarLoop()
        controller.run()
    except rospy.ROSInterruptException:
        pass