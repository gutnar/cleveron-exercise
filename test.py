#%%
import numpy as np
import cv2
import glob
import os

class Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def calibrate(self, files, grid, draw_corners=False):
        obj_points = [] # 3d point in real world space
        img_points = [] # 2d points in image plane.
        output = []
        
        grid_points = np.zeros((grid[0]*grid[1], 3), np.float32)
        grid_points[:,:2] = np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1, 2)

        for filename in files:
            # Find the chess board corners
            img = cv2.imread(filename)
            ret, corners = cv2.findChessboardCorners(img, grid, None)

            # If found, add object points, image points (after refining them)
            if not ret:
                continue

            obj_points.append(grid_points)
            img_points.append(corners)

            # Draw and display the corners
            if draw_corners:
                img = cv2.drawChessboardCorners(img, grid, corners, ret)
                output.append({ "filename": filename, "img": img })

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (self.width, self.height), None, None)
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.width, self.height), 1, (self.width, self.height))

        self.mtx = mtx
        self.dist = dist
        self.new_mtx = new_mtx
        self.roi = roi

        return output
    
    def undistort(self, img):
        # undistort
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.new_mtx)

        # crop region of interest
        x, y, w, h = self.roi
        dst = dst[y:y+h, x:x+w]

        return dst


# Calibrate
camera = Camera(1280, 720)
results = camera.calibrate(glob.glob("CALIBRATION/*.jpg"), (9, 6), True)

for result in results:
    file_index = result["filename"][-5]
    cv2.imwrite("OUTPUT/1-corners%s.jpg" % file_index, result["img"])
    cv2.imwrite("OUTPUT/2-corners-undistorted-%s.jpg" % file_index, camera.undistort(result["img"]))


# Analyze raw frames
#for filename in glob.glob("raw_images/*.jpg"):
#    img = cv2.imread(filename)

    # Undistort
#    undistorted = camera.undistort(img)

