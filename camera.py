import cv2
import numpy as np


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

        # Undistort
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (self.width, self.height), None, None)
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.width, self.height), 1, (self.width, self.height))

        self.mtx = mtx
        self.dist = dist
        self.new_mtx = new_mtx
        self.roi = roi

        return output
    
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.new_mtx)
    
    def redistort(self, img):
        return cv2.undistort(img, self.new_mtx, self.dist, None, self.mtx)
