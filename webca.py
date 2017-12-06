import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)

anterior = 0

while True:
    if not video_capture.isOpened():
        

        print('Unable to load camera.')
        sleep(5)
        pass
    
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(frame, (5, 5), 0)
    edged = cv2.Canny(frame, 35, 125)
    


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    #(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #c = max(cnts, key = cv2.contourArea)
    # Draw a rectangle around the faces
    f = 600
    W = 13
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2,)
        d = W * f / w
        #cv2.putText(gray,)
        cv2.putText(frame, 'w{0}'.format(d) ,(x,y), cv2.FONT_HERSHEY_SIMPLEX,2.0,(0, 255, 0),2)
        #cv2.putText(gray, ('width = %d, height = %d' % w, h), (x,y), font,1,(0, 255, 0), 2,cv2.LINE_AA)
    # known distance(D) = 24 in ( the distance between my face and the webcam)
    #print faces[].w
    #print w*h

   

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
