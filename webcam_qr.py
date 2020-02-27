# -*- coding utf-8 -*-
"""
Created on Tue Oct 7 114142 2018

@author Caihao.Cui
"""
from __future__ import print_function

import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import time

# get the webcam
cap = cv2.VideoCapture(0)

cap.set(3,1920)
cap.set(4,1080)

time.sleep(2)

def decode(im):
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)
    # Print results
    for obj in decodedObjects:
        print('Type ', obj.type)
        print('Data ', obj.data,'\n')
    
    return decodedObjects



font = cv2.FONT_HERSHEY_SIMPLEX

#diagnal
#cal_x = [306, 364, 450, 610, 890]

cal_x = np.array([212, 254, 312, 420, 622])
cal_y = np.array([30, 25, 20, 15, 10])

cal_xp = np.array([220, 506, 940, 1374, 1700])
cal_yp = np.array([36, 13, 0, 13, 36])

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    decodedObjects = decode(im)
    
    for decodedObject in decodedObjects:
        points = decodedObject.polygon
        
        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        
        else:
            hull = points
        
        
        cx = (hull[0][0] + hull[2][0]) / 2
        cy = (hull[0][1] + hull[2][1]) / 2
        frame = cv2.circle(frame, (int(cx), int(cy)), 10, (0, 255, 255), thickness=5)
        print(cx, cy)
        
        # Number of points in the convex hull
        n = len(hull)
        # Draw the convext hull
        for j in range(0,n):
            cv2.line(frame, hull[j], hull[ (j+1) % n], (255,0,0), 3)
        
        
        x = decodedObject.rect.left
        y = decodedObject.rect.top
        
        #pdist = np.sqrt(decodedObject.rect.width**2 + decodedObject.rect.height**2)
        pdist = decodedObject.rect.height
        print("height", pdist)
        
        xc = decodedObject.rect.left + decodedObject.rect.width/2
        cfac = np.interp(xc, cal_xp, cal_yp)
        print("correction amount", cfac)
        
        pdist -= cfac
        print("corrected height", pdist)
        print("x", xc)
        
        rdist = np.interp(pdist, cal_x, cal_y)
        print("actual distance", rdist, " cm")
        
        print(x, y)
        
        print('Type ', decodedObject.type)
        print('Data ', decodedObject.data,'\n')
        
        barCode = str(decodedObject.data)
        cv2.putText(frame, barCode, (x, y), font, 1, (0,255,255), 2, cv2.LINE_AA)
        
        # Display the resulting frame
    
    frame = cv2.resize(frame, (640, 360))
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    
    elif key & 0xFF == ord('s'): # wait for 's' key to save
        cv2.imwrite('Capture.png', frame)
        
        # When everything done, release the capture
    

cap.release()
cv2.destroyAllWindows()

