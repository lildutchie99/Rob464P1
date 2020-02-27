from __future__ import print_function

import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import time
import glob

class Observation:
    def __init__(self, x, z, vx, vz):
        self.x = x
        self.z = z
        self.vx = vx
        self.vz = vz

with np.load('calib.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

codeSize = 1.625 #width/height of QR codes in inches
objp = np.array([[0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0]], dtype=np.float32)*codeSize #dimensions of each tag (origin is center)

tagLocations = {6: (0, 0, 0), 2: (10, 0, 0)}

curPos = Observation(0, 0, 50, 50) #estimated position and variance of location. Prior for Bayesian updating/Kalman

#coordinates of the 8 corners in Revzen's coordinate system
ref0 = array([
    [66, 66, 66.1, 31.9, 0, 0, 0, 32.5 ],
    [0, 49.1, 86, 86.15, 89.35, 50.75, 0, 0],
    [0]*8 ])
ref0 *= 2.56
ref0 = ref0 - mean(ref0,1)[:,newaxis]
ref0[2,:] = 1
ref = dot([[0,-1,0],[-1,0,0],[0,0,1]],ref0)
ref = ref.T

revzenCornerPts = np.matrix(ref[:2, :-1]).T
ourCornerPts = np.matrix([[0, 0], [5, 0]]).T #TODO: fill with actual values
transformMat = revzenCornerPts*ourCornerPts.I #transform matrix from our coordinate system to the one the waypoints are in

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
    posEstimates = {}

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
            hull.sort(key=lambda p: np.arctan2(p.y - cy, p.x - cx)) #sort counterclockwise to make points match

            tagNumber = int(decodedObject.data[1:]) #get index/id of tag from data (they range from "p0" to "p8")
            ret, rvecs, tvecs = cv2.solvePnP(objp + tagLocations[tagNumber], np.array(hull, dtype=np.float64), mtx, dist)
            
            pos = np.array(tvecs.flatten())
            posEstimates[tagNumber] = Observation(x=pos[0], z=pos[2], vx=1, vz=1)
            print(pos)

            # Draw the convex hull
            for j in range(0,n):
                cv2.line(frame, hull[j], hull[ (j+1) % n], (255,0,0), 3)

        barCode = str(decodedObject.data)
        cv2.putText(frame, barCode, (decodedObject.rect.left, decodedObject.rect.top), font, 1, (0,255,255), 2, cv2.LINE_AA)

    #perform Kalman/Bayesian updating on position
    for key in posEstimates:
        est = posEstimates[key]
        curPos.x = (est.vx*curPos.x + curPos.vx*est.x)/(curPos.vx + est.vx)
        curPos.z = (est.vz*curPos.z + curPos.vz*est.z)/(curPos.vz + est.vz)
        curPos.vx = 1/((1/curPos.vx) + (1/est.vx))
        curPos.vz = 1/((1/curPos.vz) + (1/est.vz))

    print("camera pos: ", curPos.x, ", ", curPos.y)
               
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