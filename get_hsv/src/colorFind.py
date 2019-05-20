#!/usr/bin/env python
#import the necessary packages
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import UInt16MultiArray
import cv2
import numpy as np
from cv_bridge import CvBridge
from shapely.geometry import Polygon
cv_image = np.zeros((640,480,3), np.uint8)
res = np.zeros((640,480,3), np.uint8)
state = [0,0,0,0,0,0,0,0,0]
pub_state = UInt16MultiArray()
pub_state.data = state

#assign strings for ease of codingc
hh='Hue High'
hl='Hue Low'
sh='Saturation High'
sl='Saturation Low'
vh='Value High'
vl='Value Low'

pub = rospy.Publisher('camera_state',UInt16MultiArray,queue_size=1)
#'optional' argument is required for trackbar creation parameters
def nothing(x):
    pass

#set up sector limits
hInt = int(480/3)
wInt = int(640/3)
polyList = []

for y in range (0,3):
    for x in range(0,3):
        newPoly = Polygon([(x*wInt,y*hInt),((x+1)*wInt,y*hInt),((x+1)*wInt,(y+1)*hInt),(x*wInt,(y+1)*hInt)])
        polyList.append(newPoly)

def callback(data):
    global cv_image
    global res
    global state, pub_state
    state = [0,0,0,0,0,0,0,0,0]
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    cv_image=cv2.GaussianBlur(cv_image,(5,5),0)
    hsv=cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    hul=10
    huh=100
    sal=0
    sah=75
    val=160
    vah=220

    #make array for final values
    HSVLOW=np.array([hul,sal,val])
    HSVHIGH=np.array([huh,sah,vah])

    #create a mask for that range
    mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)
    mask = mask#[575:964,0:1284]
    cv_image2 = cv_image#[575:964,0:1284]

    res = cv2.bitwise_and(cv_image2,cv_image2, mask = mask)
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
      sigNum = 0

      for index,contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        (center, (w, h), angle) = rect
        epsilon = 0.1*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if(w*h >= 2000):
            cv2.drawContours(cv_image, contour, -1, (0,255,0), 3)
            boxConvert = map(lambda p: tuple(p), box)
            boxPoly = Polygon(boxConvert)
            for i in range (0,9):
                rospy.loginfo(i)
                if(boxPoly.intersects(polyList[i])):
                    state[i] = 1
    h1, w1, _ = cv_image.shape
    h1 = int(h1)
    w1 = int(w1)
    wP = int(w1/3)
    hP = int(h1/3)
    cv2.line(cv_image,(wP,0),(wP,h1),(50,50,255),3)
    cv2.line(cv_image,(2*wP,0),(2*wP,h1),(50,50,255),3)
    cv2.line(cv_image,(0,hP),(w1,hP),(50,50,255),3)
    cv2.line(cv_image,(0,2*hP),(w1,2*hP),(50,50,255),3)
    cv2.imshow('Camera', cv_image)
    cv2.imshow('res', res)
    cv2.waitKey(1)
    pub_state.data = state


def listener():
    global cv_image
    global res
    global state, pub_state
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.loginfo("please")
    rospy.init_node('my_cam', anonymous=True)
    rate = rospy.Rate(2) # 1hz
    rospy.Subscriber("usb_cam/image_raw",Image, callback)
    while not rospy.is_shutdown():
        pub.publish(pub_state)
        rate.sleep()

if __name__ == '__main__':
    print("kk")
    try:
        print("k")
        listener()
    except KeyboardInterrupt:
        print("Goodbye")
