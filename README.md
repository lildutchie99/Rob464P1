ROB 464 P1 - Red Team 2020

Calibration folder - photos taken from our webcam used for calibration (these are camera-specific)
camera_cal.py - script to generate calibration profile from images of a checkerboard. If this is rerun with different photos in the calibration folder, it will write a new "calib.npz" file storing the camera's calibration info.
webcam_qr.py - basic script using linear interpolation to estimate the distance of a QR code from the camera. This is dependent on both the camera and the size of the codes being used
webcam_qr_perspective - more advanced approach that uses calibration info to estimate the pose (position and rotation) of the camera relative to the QR code