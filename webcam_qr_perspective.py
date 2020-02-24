from __future__ import print_function

import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import time
import glob

with np.load('calib.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
objp = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32)

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
            hull.sort(key=lambda p: np.arctan2(p.y - cy, p.x - cx)) #sort counterclockwise
            print(hull)
            ret, rvecs, tvecs = cv2.solvePnP(objp, np.array(hull, dtype=np.float64), mtx, dist)
            print(tvecs)

            # Draw the convext hull
            for j in range(0,n):
                cv2.line(frame, hull[j], hull[ (j+1) % n], (255,0,0), 3)
            cv2.line(frame, hull[1], hull[2], (0,0,255), 3)

        print('Type : ', decodedObject.type)
        print('Data : ', decodedObject.data,'\n')

        #barCode = str(decodedObject.data)
       #cv2.putText(frame, barCode, (x, y), font, 1, (0,255,255), 2, cv2.LINE_AA)
               
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