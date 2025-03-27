#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_msgs.msg import String

class Controller:
    def __init__(self):
        rospy.init_node('controller')

        # State variables
        self.pedestrian_detected = False
        self.pedestrian_block_end_timer = None
        self.pedestrian_lane_duration = rospy.Duration(2.0)

        self.stop_duration = rospy.Duration(3.0)
        self.min_speed = 0.2
        self.max_speed = 0.4
        self.speed_limit = self.max_speed

        self.object_detected = False

        self.current_sign = ""
        self.stop_detected = False
        self.stop_block_end_timer = None
        self.traffic_light_state = "green"
        self.turn_restrictions = {
            "left_allowed": True,
            "right_allowed": True,
            "only_right": False
        }
        
        # Publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

        rospy.Subscriber('/object_detected', Bool, self.object_callback)
        rospy.Subscriber('/pedestrian', Bool, self.pedestrian_callback)
        rospy.Subscriber('/sign_detection', String, self.sign_callback)
        rospy.Subscriber('/check_cmd_vel', Twist, self.cmd_callback)

    def pedestrian_callback(self, msg):
        if msg.data:
            rospy.loginfo("Pedestrian lane detected! Stop for 2 seconds")
            self.pedestrian_detected = True

            self.pedestrian_block_end_timer = rospy.Time.now() + self.pedestrian_lane_duration

            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)

    def object_callback(self, msg):
        if msg.data != self.object_detected:
            self.object_detected = msg.data

            if self.object_detected:
                rospy.loginfo("Object detected! Cannot move forward")
            else:
                rospy.loginfo("Object cleared! Can move forward now")

    def sign_callback(self, msg):
        sign = msg.data

        if sign != self.current_sign:
            self.current_sign = sign
            rospy.loginfo(f"{sign} detected")
            
            if sign == "stop":
                rospy.loginfo("Stop sign detected")
                self.stop_detected = True
                self.stop_block_end_timer = rospy.Time.now() + self.stop_duration
                cmd = Twist()
                self.cmd_vel_pub.publish(cmd)
            elif sign == "red_light":
                rospy.loginfo("Red light detected")
                self.traffic_light_state = "red"
                cmd = Twist()
                self.cmd_vel_pub.publish(cmd)
            elif sign == "yellow_light":
                rospy.loginfo("Yellow light detected")
                self.traffic_light_state = "yellow"
            elif sign == "green_light":
                rospy.loginfo("Green light detected")
            elif sign == "no_left":
                rospy.loginfo("No left turns allowed")
                self.turn_restrictions["left_allowed"] = False
            elif sign == "no_right":
                rospy.loginfo("No right turns allowed")
                self.turn_restrictions["right_allowed"] = False
            elif sign == "right":
                rospy.loginfo("Only right turns allowed")
                self.turn_restrictions["only_right"] = True
            elif sign == "five":
                rospy.loginfo("Reduce speed to 5")
                self.speed_limit = self.min_speed
            elif sign == "no_limit":
                rospy.loginfo("Increase speed to the national speed limit")
                self.speed_limit = self.max_speed

    def reset_turn_restrictions(self):
        self.turn_restrictions = {
            "left_allowed": True,
            "right_allowed": True,
            "only_right": False
        }

    def timer_callback(self, event):
        current_time = rospy.Time.now()

        if self.pedestrian_detected and current_time >= self.pedestrian_block_end_timer:
            self.pedestrian_detected = False
            rospy.loginfo("Done waiting for pedestrians")

        if self.stop_detected and current_time >= self.stop_block_end_timer:
            self.stop_detected = False
            rospy.loginfo("Done waiting for the stop sign")

    def cmd_callback(self, msg):
        cmd = Twist()

        if self.object_detected and msg.linear.x > 0:    
            cmd = msg
            cmd.linear.x = 0.0
            self.cmd_vel_pub.publish(cmd)
            rospy.loginfo("Wait for object to be removed")
            return
        
        if self.stop_detected:
            self.cmd_vel_pub.publish(cmd)
            rospy.loginfo("Stopped at a stop sign")
            return
        
        cmd = self.apply_traffic_rules(msg)
        
        if self.pedestrian_detected:
            self.cmd_vel_pub.publish(cmd)
            rospy.loginfo("Wait for pedestrians")
            return

        self.cmd_vel_pub.publish(msg)

    def apply_traffic_sign_rules(self, msg):
        cmd = msg
        
        # Apply speed limit
        if cmd.linear.x > self.speed_limit:
            cmd.linear.x = self.speed_limit
        
        # Apply traffic light rules
        if self.traffic_light_state == "red" and cmd.linear.x > 0:
            cmd.linear.x = 0.0
        elif self.traffic_light_state == "yellow" and cmd.linear.x > 0:
            cmd.linear.x *= 0.5  # Reduce speed by half for yellow light
        
        # Apply turn restrictions
        if not self.turn_restrictions["left_allowed"] and cmd.angular.z > 0:
            cmd.angular.z = 0.0  # Block left turns
        
        if not self.turn_restrictions["right_allowed"] and cmd.angular.z < 0:
            cmd.angular.z = 0.0  # Block right turns
        
        if self.turn_restrictions["only_right"]:
            if cmd.angular.z >= 0:  # Not turning right
                cmd.angular.z = -0.2  # Force slight right turn
                if cmd.linear.x > 0.3:
                    cmd.linear.x = 0.3  # Reduce speed for right turn
        
        return cmd
        
    def run(self):
        """Main loop"""
        rospy.spin()

if __name__ == '__main__':
    try:
        controller = Controller()
        controller.run()
    except rospy.ROSInterruptException:
        pass