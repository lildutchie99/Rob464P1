import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt

frame = cv2.imread("aruco_test.png")
frame = cv2.resize(frame, (120, 90))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
plt.figure()
plt.imshow(frame_markers)
plt.show()