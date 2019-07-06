import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, argrelextrema
import cv2
import glob

from camera import Camera
from perspective import Perspective
from lane_detector import LaneDetector

# Camera and perspective instances
camera = Camera(1280, 720)
perspective = Perspective()

# Calibrate camera
camera = Camera(1280, 720)
perspective = Perspective()
results = camera.calibrate(glob.glob("CALIBRATION/*.jpg"), (9, 6), True)

for result in results:
    file_index = result["filename"][-5]
    cv2.imwrite("OUTPUT/calibration/1-corners%s.jpg" % file_index, result["img"])
    cv2.imwrite("OUTPUT/calibration/2-corners-undistorted-%s.jpg" % file_index, camera.undistort(result["img"]))

# Process images
for filename in glob.glob("raw_images/*.jpg"):
    detector = LaneDetector(camera, perspective)
    detector.run(cv2.imread(filename), "OUTPUT/test%s" % filename[-5])

# Process video
cap = cv2.VideoCapture("road.mp4")
detector = LaneDetector(camera, perspective)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = detector.run(frame)
    cv2.imshow("road.mp4", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.imwrite("test.jpg", frame)
        break

cap.release()
cv2.destroyAllWindows()
