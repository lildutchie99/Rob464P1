#uses measured distances from 2 tags to estimate position using triangulation.

from __future__ import print_function

import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import time
import glob

#get coordinates of third point in triangle based on two points and side lengths
#a and b are points, ac and bc are side lengths
def getThirdPoint(a, b, ac, bc):
    ab = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    cxp = (ab**2 + ac**2 - bc**2)/(2*ab) #coords of C in coordinate system with A at (0, 0) and B at (ab, 0)
    cyp = np.sqrt(ac**2 - cxp**2)
    cosTheta = (b[0] - a[0])/ab # sin and cos of angle for applying rotation matrix
    sinTheta = (b[1] - a[1])/ab
    cx = cosTheta*cxp - sinTheta*cyp + a[0]
    cy = sinTheta*cxp + cosTheta*cyp + a[1]
    return (cx, cy)

with np.load('calib.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

codeSize = 1.625
objp = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32)*codeSize

calib = np.array([0, 0, 0]).T
uscale = 1

tagLocations = {6: (0,0), 2: (10, 0)}

# get the webcam:  
cap = cv2.VideoCapture(0)

cap.set(3,1920)
cap.set(4,1080)
time.sleep(2)

def decode(im) : 
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)
    # Print results
    for obj in decodedObjects:
        print('Type : ', obj.type)
        print('Data : ', obj.data,'\n')     
    return decodedObjects


font = cv2.FONT_HERSHEY_SIMPLEX

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         
    decodedObjects = decode(im)
    worldPos = {}

    for decodedObject in decodedObjects: 
        points = decodedObject.polygon
     
        # If the points do not form a quad, find convex hull
        if len(points) > 4 : 
          hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
          hull = list(map(tuple, np.squeeze(hull)))
        else : 
          hull = points;
         
        # Number of points in the convex hull
        n = len(hull)

        if n == 4:
            cx = (hull[0].x + hull[2].x) / 2
            cy = (hull[0].y + hull[2].y) / 2
            frame = cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 0))
            hull.sort(key=lambda p: np.arctan2(p.y - cy, p.x - cx)) #sort counterclockwise to make points match
            ret, rvecs, tvecs = cv2.solvePnP(objp, np.array(hull, dtype=np.float64), mtx, dist)
            pos = np.array(tvecs.flatten()) - calib
            worldPos[int(decodedObject.data[1:])] = pos
            print(pos)

            # Draw the convext hull
            for j in range(0,n):
                cv2.line(frame, hull[j], hull[ (j+1) % n], (255,0,0), 3)
            cv2.line(frame, hull[1], hull[2], (0,0,255), 3)

        print('Type : ', decodedObject.type)
        print('Data : ', decodedObject.data,'\n')

        #barCode = str(decodedObject.data)
       #cv2.putText(frame, barCode, (x, y), font, 1, (0,255,255), 2, cv2.LINE_AA)

    if len(worldPos) == 2: #two reference points
        aIdx = min(list(worldPos.keys())) #indices of each tag
        bIdx = max(list(worldPos.keys()))
        ac = np.sqrt(worldPos[aIdx][0]**2 + worldPos[aIdx][2]**2)
        bc = np.sqrt(worldPos[bIdx][0]**2 + worldPos[bIdx][2]**2)
        cx, cy = getThirdPoint(tagLocations[aIdx], tagLocations[bIdx], ac, bc)
        print("camera pos: ", cx, ", ", cy)
               
    # Display the resulting frame
    frame = cv2.resize(frame, (640, 360))
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'): # wait for 's' key to save 
        cv2.imwrite('Capture.png', frame)
    elif key & 0xFF == ord('c'):
        calib = np.array(tvecs).flatten()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()