import cv2
import cv2.aruco as aruco
import numpy as np
import time
from picamera2 import Picamera2
import platform
import sys 
import os 

width, height = 640, 480
marker_size = 30 #30 cm 
id_to_find = 72 #marker ID you want to track

#Load calibration results
calib_path = '/home/pi/'
cameraMatrix = np.loadtxt(calib_path+'cameraMatrix.txt', delimiter=',')
cameraDistortion = np.loadtxt(calib_path+'cameraDistortion.txt', delimiter=',')

#Aruco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()

#Video output
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30.0
frame_size = (width, height)
out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

#camera setup
cap = Picamera2()
cap.create_preview_configuration(main={"format": "RGB888", "size": (width, height)})
cap.start()

#main loop 
seconds = 1000000 #run indefinitely
start_time = time.time()
counter = 0

while time.time() - start_time < seconds:
    frame = cap.capture_array()
    frame_np = np.array(frame)
    gray = cv2.cvtColor(frame_np,  cv2.COLOR_RGB2GRAY)

    #detect aruco markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and id_to_find in ids:
        #Estimate pose
        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, cameraMatrix, cameraDistortion)
        rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

        #Rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        #Yaw / Pitch / Roll
        yaw = np.degrees(np.arctan2(R[1,0], R[0,0])) % 360 
        pitch = np.degrees(np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))) % 360
        roll = np.degrees(np.arctan2(R[2,1], R[2,2])) % 360
   
        #Marker position
        x, y, z = tvec
        print(f"Yaw: {yaw:.2f} Pitch: {pitch:.2f} Roll: {roll:2f} Positin: x={x:2f}, y={y:2f}, z={z:2f}")
        
        #Optional: Draw marker + axis 
        #aruco.drawDetectedMarkers(frame_np, corners)
        #aruco.drawAxis(frame_np, cameraMatrix, cameraDistortion, rvec, tvec, 0.1)

    else:
       print(f"Marker {id_to_find} NOT FOUND in frame.")
 
    #Show frame
    cv2.imshow('Aruco Tracker', frame_np)
    out.write(frame_np)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    counter += 1

#Release
out.release()
cv2.destroyAllWindows()
