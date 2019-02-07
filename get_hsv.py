#import the necessary packages
import cv2
import numpy as np
from shapely.geometry import Polygon
cv_image = np.zeros((1280,960,3), np.uint8)
res = np.zeros((1280,960,3), np.uint8)
res2 = np.zeros((1280,960,3), np.uint8)
state = [0,0,0,0,0,0,0,0,0]
#'optional' argument is required for trackbar creation parameters
def nothing(x):
    pass

def callback(data):
    global cv_image
    global res
    global res2
    global frameNumber
    global state
    state = [0,0,0,0,0,0,0,0,0]
    cv_image = data
    cv_image=cv2.GaussianBlur(cv_image,(5,5),0)
    hsv=cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    cv2.imshow('res', hsv)
    #read trackbar positions for each trackbar
    hul=cv2.getTrackbarPos(hl, 'Colorbars')
    huh=cv2.getTrackbarPos(hh, 'Colorbars')
    sal=cv2.getTrackbarPos(sl, 'Colorbars')
    sah=cv2.getTrackbarPos(sh, 'Colorbars')
    val=cv2.getTrackbarPos(vl, 'Colorbars')
    vah=cv2.getTrackbarPos(vh, 'Colorbars')

    hul2=cv2.getTrackbarPos(hl, 'Colorbars2')
    huh2=cv2.getTrackbarPos(hh, 'Colorbars2')
    sal2=cv2.getTrackbarPos(sl, 'Colorbars2')
    sah2=cv2.getTrackbarPos(sh, 'Colorbars2')
    val2=cv2.getTrackbarPos(vl, 'Colorbars2')
    vah2=cv2.getTrackbarPos(vh, 'Colorbars2')

    #make array for final values
    HSVLOW=np.array([hul,sal,val])
    HSVHIGH=np.array([huh,sah,vah])

    HSVLOW2=np.array([hul2,sal2,val2])
    HSVHIGH2=np.array([huh2,sah2,vah2])

    #create a mask for that range
    mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)
    #print(mask)
    mask2 = cv2.inRange(hsv,HSVLOW2, HSVHIGH2)
    #mask = mask[575:964,0:1284]
    #mask2 = mask2[575:964,0:1284]
    cv_image2 = cv_image #used to crop this one too

    res = cv2.bitwise_and(cv_image2,cv_image2, mask = mask)
    res2 = cv2.bitwise_and(cv_image2,cv_image2, mask = mask2)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2, hierarchy2 = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours or contours2:
      frameNumber += 1
      sigNum = 0
      for index, contour in enumerate(contours2):
          sigNum += 1
          x,y,w,h = cv2.boundingRect(contour)
          y += 575
          if(w*h > 10000):
            #print sigNum, x,y,x+w,y+h
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0,0,0), 4)

      for index,contour in enumerate(contours):
        sigNum += 1
        rect = cv2.minAreaRect(contour)
        #print rect[0][0] - rect[1][1]/2 , rect[0][1]+575
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if rect[1][0] * rect[1][1] > 500:
          cv2.drawContours(cv_image,[box],0,(0,0,255),2)
          # calculate intersection
          boxConvert = map(lambda p: tuple(p), box)
          boxPoly = Polygon(boxConvert)
          # check for box intersection
          for i in range (0,9):
              if(boxPoly.intersects(polyList[i])):
                  state[i] = 1
      print(state)

      # not sure this for-loop does anything
      for index,contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        (center, (w, h), angle) = rect

        epsilon = 0.01*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        #if(w*h >= 10000):
           #print(frameNumber)
           #cv2.drawContours(cv_image, contour, -1, (0,255,0), 3)

    h1, w1, _ = cv_image.shape # get height/width to divide up image
    h1 = int(h1)
    w1 = int(w1)
    wP = int(w1/3)
    hP = int(h1/3)
    cv2.line(cv_image,(wP,0),(wP,h1),(0,0,0),1)
    cv2.line(cv_image,(2*wP,0),(2*wP,h1),(0,0,0),1)
    cv2.line(cv_image,(0,hP),(w1,hP),(0,0,0),1)
    cv2.line(cv_image,(0,2*hP),(w1,2*hP),(0,0,0),1)
    cv2.imshow('Camera', cv_image)

# rospy.init_node('pointgrey_mine', anonymous=True)
# rospy.Subscriber("/camera/image_color",Image, callback)
hInt = int(480/3)
wInt = int(640/3)
polyList = []
# create list of polygons for the states
for y in range(0,3):
    for x in range(0,3):
        newPoly = Polygon([(x*wInt,y*hInt),((x+1)*wInt,y*hInt),((x+1)*wInt,(y+1)*hInt),(x*wInt,(y+1)*hInt)])
        polyList.append(newPoly)
#Capture video from the stream
cap = cv2.VideoCapture(1)
#cap.set(cv2.CAP_PROP_EXPOSURE, 10)

cv2.namedWindow('Colorbars') #Create a window named 'Colorbars'
cv2.namedWindow('Colorbars2')

#assign strings for ease of coding
hh='Hue High'
hl='Hue Low'
sh='Saturation High'
sl='Saturation Low'
vh='Value High'
vl='Value Low'
wnd = 'Colorbars'
#Begin Creating trackbars for each
cv2.createTrackbar(hl, 'Colorbars',0,179,nothing)
cv2.createTrackbar(hh, 'Colorbars',0,179,nothing)
cv2.createTrackbar(sl, 'Colorbars',0,255,nothing)
cv2.createTrackbar(sh, 'Colorbars',0,255,nothing)
cv2.createTrackbar(vl, 'Colorbars',0,255,nothing)
cv2.createTrackbar(vh, 'Colorbars',0,255,nothing)

cv2.createTrackbar(hl, 'Colorbars2',0,179,nothing)
cv2.createTrackbar(hh, 'Colorbars2',0,179,nothing)
cv2.createTrackbar(sl, 'Colorbars2',0,255,nothing)
cv2.createTrackbar(sh, 'Colorbars2',0,255,nothing)
cv2.createTrackbar(vl, 'Colorbars2',0,255,nothing)
cv2.createTrackbar(vh, 'Colorbars2',0,255,nothing)

#rate = rospy.Rate(10) # 10hz
frameNumber = 0
#begin our 'infinite' while loop
while(1):
    #it is common to apply a blur to the cv_image
    #cv_image=cv2.GaussianBlur(cv_image,(5,5),0)
    #convert from a BGR stream to an HSV stream

           #print(w, int(round(w)), h, int(round(h)))
           #print(center)
#          box = cv2.boxPoints(rect)
#          box = np.int0(box)
#          ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = box
#          #cv2.drawContours(frame,[box],0,(0,0,0),2)
#
#          rows,cols = cv_image.shape[:2]
#          #(vx, vy) is a normalized vector collinear to the line and (x0, y0) is a point on the #line.
#          [vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
#          lefty = int((-x*vy/vx) + y)
#          righty = int(((cols-x)*vy/vx)+y)
#          #print(lefty, righty)
#          #cv2.line(image, 1st point coords, 2nd point coords, lineColor, thickness)
#          point1 = (cols-1,righty)
#          point2 = (0,lefty)
#          cv2.line(cv_image, point1, point2,(0,0,0),2)
    ret, frame = cap.read()
    callback(frame)
    if not res is None: #avoid Nonetype error
        cv2.imshow('res', res)
        cv2.imshow('res2', res2)

    if cv2.waitKey(1) & 0xFF == ord('q'): #if CTRL-C is pressed then exit the loop
        break

cv2.destroyAllWindows()
