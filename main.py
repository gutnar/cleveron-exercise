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

# HSV color mask
color_mask = (
    ((20, 50, 75), (110, 255, 255)), # yellow lines
    ((0, 0, 220), (255, 255, 255)) # white lines
)

# Process images
for filename in glob.glob("raw_images/*.jpg"):
    detector = LaneDetector(camera, perspective, color_mask)
    detector.run(cv2.imread(filename), "OUTPUT/test%s" % filename[-5])

# Process video
cap = cv2.VideoCapture("road.mp4")
out = cv2.VideoWriter("road_processed.avi", cv2.VideoWriter_fourcc(*'XVID'), 25, (1280, 720))
detector = LaneDetector(camera, perspective, color_mask)
i = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    i += 1
    #if i < 1000:
    #    continue

    key = cv2.waitKey(5)

    if key & 0xFF == ord("s"):
        frame = detector.run(frame, "OUTPUT/frame")
    elif key & 0xFF == ord("q"):
        break
    else:
        frame = detector.run(frame)
    
    out.write(frame)
    cv2.imshow("road.mp4", frame)

cap.release()
out.release()

cv2.destroyAllWindows()
