#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
import time

class StateMachine:
    def __init__(self):
        rospy.init_node('state_machine')
        
        # Parameters
        self.forward_speed = 0.2
        self.turn_speed = 0.6

        self.state = 'S'
        self.prev_state = "None"
        self.next_state = '1'
        self.next_next_state = '2'
        self.action = []
        self.route = []
        self.last_cmd = None
    
        self.pedestrian_detected = False
        self.pedestrian_time = 2
        self.pedestrian_state_start_time = None

        self.turn_time = 2

        # Create publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Start subscribers
        rospy.Subscriber('/pedestrian', Bool, self.pedestrian_callback)

        self.state_dict = {
            "S": {
                "None": {
                    "1": "FORWARD,PEDESTRIAN"
                },
                "5": {
                    "1": "SLIGHT_RIGHT,FORWARD,PEDESTRIAN"
                }
            },
            "1": {
                "S": {
                    "2": "FORWARD,PEDESTRIAN,FORWARD,SLIGHT_RIGHT"
                }
            }
        }

        rospy.loginfo("State machine navigation initialized")

    def pedestrian_callback(self, msg):
        if msg.data:
            self.pedestrian_detected = True
            # rospy.loginfo("Pedestrian lane detected")
        else:
            self.pedestrian_detected = False

    def update_state(self):
        self.action.extend(self.find_action(self.state, self.prev_state, self.next_state).split(','))
        rospy.loginfo(f"Going to state {self.next_state}")    

    def publish_command(self):
        current_time = time.time()
        cmd = Twist()

        if self.action[0] == "FORWARD":
            cmd.linear.x = self.forward_speed
            cmd.angular.z = 0.0
            self.action.pop(0)
            
        elif self.action[0] == "PEDESTRIAN":
            if not self.pedestrian_detected:
                cmd = self.last_cmd
            else:
                cmd = Twist()
                self.cmd_vel_pub.publish(cmd)
                self.pedestrian_state_start_time = time.time()
            if self.pedestrian_state_start_time is not None and current_time - self.pedestrian_state_start_time > self.pedestrian_time:
                self.pedestrian_state_start_time = None
                self.action.pop(0)

        elif self.action[0] == "SLIGHT_RIGHT":
            cmd.linear.x = self.forward_speed
            cmd.angular.z = -self.turn_speed
        elif self.state == 'TURN_RIGHT':
            cmd.linear.x = self.forward_speed
            cmd.angular.z = -self.turn_speed
        self.cmd_vel_pub.publish(cmd)
        rospy.loginfo(cmd)
        self.last_cmd = cmd

    def find_action(self, state, prev_state, next_state):
        transition = self.state_dict[state][prev_state][next_state]
        return transition
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(20)
        self.route = ['S', '1', '2', '3', '9', '4', '5', 'S']
        self.state = self.route[0]
        self.prev_state = "None"
        self.next_state = self.route[1]
        while not rospy.is_shutdown():
            self.update_state()
            while len(self.action) != 0:
                self.publish_command()
            rospy.loginfo_throttle(1.0, f"State: {self.state}")
            self.route.pop(0)
            self.prev_state = self.state
            self.state = self.route[0] if len(self.route) >= 1 else "None"
            self.next_state = self.route[1] if len(self.route) >= 2 else "None"
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = StateMachine()
        controller.run()
    except rospy.ROSInterruptException:
        pass