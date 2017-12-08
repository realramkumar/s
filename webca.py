import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
import math
import matplotlib
matplotlib.use('MacOSX') 
import matplotlib.pyplot as plt

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

frame_count = 0

#f=w*W/D (focal length is found by this formula 
f = 600
W = 13
D = 24 #known distance
#assuming the distance between two end points of the field covered by the camera is 190 cm ( 6.233) ( measured manually from my laptop)
#the distance between the person and the camera being 6 feet
#the angle of view of the camera will be equal to 6.233/6 radians which is 59 degree 
#since the wall is 180 degree and the angle of view is 59 degree , the area not covered by the camera will be around 121 degrees.
#assuming the camera is in the middle of the wall, the area not covered by the camera on the left and right side will be equal.
#so, 121/2
teta_0 = 60.5 
#xmax is 10 feet if given room co ordinate is 10 feet by 10 feet
#xmin will be 0
xmax=1000
x1= 4
y1= 0
roomx = 8
roomy = 13
# provided camera is on the center of the wall of dimension 10 feet by 10 feet   

plt.ion()
pre_plot = None

coordinates = list()
frame_det_count = list()

while True:

    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(frame, (5, 5), 0)
    edged = cv2.Canny(frame, 35, 125)

    frame_count = frame_count + 1

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    #(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #c = max(cnts, key = cv2.contourArea)
    # Draw a rectangle around the faces
     
    for index, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2,)
        #f = w * W / D

        d = W * f / w
        d1= d/12
        arc_length = 2*3.14*d1*59/360
        mscr = (2*x+w)/2
        pscr = mscr *100 / xmax
        marc = pscr*arc_length/100
        teta_p = marc *360 / (2*3.14*d1)
        teta_x = teta_p+teta_0
        # the co ordinates of a point on a circle is found by the formula, x2 = x1 + dcos teta and y2 = y1+dsin teta
        # when centred at (cx, cy) co ordinates= (cx + radius * cos(angle), cy + radius * sin(angle))
        x2= x1 + (d1 * math.cos(teta_x))
        y2 = y1 + (d1 * math.sin(teta_x))
        if not index < len(coordinates):
            coordinates.append([x2, y2])
            frame_det_count.append(0)
        else:
            if abs(x2) > 0 and abs(x2) < roomx and abs(y2) > 0 and abs(y2) < roomy:
                frame_det_count[index] = frame_det_count[index] + 1
                coordinates[index][0] = coordinates[index][0] + abs(x2)
                coordinates[index][1] = coordinates[index][1] + abs(y2)

        cv2.putText(frame, 'distance{0} feet'.format(d1) ,(x,y), cv2.FONT_HERSHEY_SIMPLEX,2.0,(0, 255, 0),2)
        cv2.putText(frame, 'x{0},y{1}'.format(x2,y2) ,(x-w,y+h), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0, 255, 0),2)
        #cv2.putText(frame, 'arclength{0} feet'.format(arc_length) ,(x-w,y+h), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0, 255, 0),2)
        
        #cv2.putText(frame, 'distance{0} feet'.format(d1) ,(x,y), cv2.FONT_HERSHEY_SIMPLEX,2.0,(0, 255, 0),2)
        
    # known distance(D) = 24 in ( the distance between my face and the webcam)
    #print faces[].w
    #print w*h
#gender

    if frame_count >= 30:
        frame_count = 0
        print frame_det_count
        xvalues = list()
        yvalues = list()
        for index, (x2, y2) in enumerate(coordinates):
            #for each faces
            if not frame_det_count[index] == 0:
                x2 = x2 / frame_det_count[index]
                y2 = y2 / frame_det_count[index]
                print x2, y2
                xvalues.append(x2)
                yvalues.append(roomy - y2)
                frame_det_count[index] = 0
                del(coordinates[index])
        del(frame_det_count[:])
        del(coordinates[:])
        plt.clf()
        plt.xlim(0, roomx)
        plt.ylim(0, roomy)
        plt.scatter(xvalues, yvalues)
        plt.pause(0.5)
        

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    #cv2.imshow('Video', gray)
    #cv2.imshow('frame',fgmask)
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()