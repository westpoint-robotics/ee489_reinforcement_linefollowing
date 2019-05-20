#!/usr/bin/env python
import rospy
import random
import numpy as np
from std_msgs.msg import UInt16MultiArray
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
# experiences should have state, action, reward, state + 1
state = [0] * 9
action = ""

# a is for append, w for write
#f = open("/home/rrc/path following testing area/training_data.txt", "a")

#controller stuff
buttons = [0] * 11
move_cmd = Twist()

def take_action(a):
        if a == 's':
            move_cmd.linear.x = .3
            move_cmd.angular.z = 0
        elif a == 'r':
            move_cmd.linear.x = .3
            move_cmd.angular.z = -1.60
        elif a == 'l':
            move_cmd.linear.x = .3
            move_cmd.angular.z = 1.60
        else:
            move_cmd.linear.x = 0
            move_cmd.angular.z = 0


def joy_callback(data):
    global buttons
    buttons = data.buttons

def doStuff(data):
    global state, action, buttons, f
    action = ''
    state = list(data.data)

    if buttons[3] == 1: #y, straight
        action = 's'
    elif buttons[2] == 1: #x, left
        action = 'l'
    elif buttons[1] == 1: #b, right
        action = 'r'
    elif buttons[0] == 1: #a, close out
        action = 'c'

    if action == 'c':
        #f.close
        rospy.signal_shutdown("pressed A")

    rospy.loginfo(state)
    # we then send the associated action to the turtlebot
    take_action(action)
    rospy.loginfo("action: " + action)

    # if action != '': #log it into training data
    #     str_state = ''.join(str(e) for e in state)
    #     f.write(str_state + '\n')
    #     f.write(action + '\n')

def callback(data):
    doStuff(data)

def listener():
    global move_cmd
    rospy.init_node('trainer',anonymous=True,disable_signals=True)
    cmd_vel = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)
    rate = rospy.Rate(2)
    rospy.Subscriber("joy", Joy, joy_callback) # sub to controller
    rospy.Subscriber("camera_state", UInt16MultiArray, callback)
    while not rospy.is_shutdown():
        rate.sleep()
        cmd_vel.publish(move_cmd)

if __name__ == '__main__':
    listener()
