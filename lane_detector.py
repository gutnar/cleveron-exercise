import cv2
import numpy as np
import os
from buffered_poly_fit import BufferedPolyFit


class LaneDetector:
    def __init__(self, camera, perspective, color_mask, fit_sample_size=5000, y_check_steps=15, max_x_std=75, pixels_per_meter=300):
        self.camera = camera
        self.perspective = perspective
        self.left_poly = BufferedPolyFit(2)
        self.right_poly = BufferedPolyFit(2)
        self.fit_sample_size = fit_sample_size
        self.color_mask = color_mask
        self.y_check_steps = y_check_steps
        self.max_x_std = max_x_std
        self.pixels_per_meter = pixels_per_meter

    def apply_color_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(img.shape[:2], np.uint8)

        for color_range in self.color_mask:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, color_range[0], color_range[1]))
        
        return cv2.bitwise_and(gray, mask)

    def get_lane_points(self, img, offset=0):
        y, x = np.nonzero(img)

        # Sample fewer points without replacement to improve processing speed
        i = np.random.choice(len(x), self.fit_sample_size)
        y = y[i]
        x = x[i]        
        
        # Keep points where deviation along x-axis is not too big
        keep = [np.array([], np.uint8), np.array([], np.uint8)]
        discard = [np.array([], np.uint8), np.array([], np.uint8)]
        step = img.shape[0] // self.y_check_steps

        for row in range(0, img.shape[0], step):
            in_row = np.where((y >= row) & (y <= row + step))

            if not len(in_row):
                continue

            if np.std(x[in_row]) < self.max_x_std:
                keep[0] = np.concatenate((keep[0], y[in_row]))
                keep[1] = np.concatenate((keep[1], x[in_row]))
            else:
                discard[0] = np.concatenate((discard[0], y[in_row]))
                discard[1] = np.concatenate((discard[1], x[in_row]))

        keep[1] += offset
        discard[1] += offset

        return keep, discard

    def fit_lanes(self, img):
        #kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
        #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        left, left_discarded = self.get_lane_points(img[:,0:img.shape[1]//3])
        right, right_discarded = self.get_lane_points(img[:,img.shape[1]//3*2:], img.shape[1]//3*2)
        
        left_line = self.left_poly.fit(left[0], left[1])
        right_line = self.right_poly.fit(right[0], right[1])
        
        # Draw lane
        if left_line and right_line:
            for y in range(img.shape[0]):
                l, r = int(left_line(y)), int(right_line(y))
                cv2.line(output, (l, y), (r, y), (0, 255, 0), 1)

        for points in (left, right):
            for i in range(len(points[0])):
                cv2.circle(output, (points[1][i], points[0][i]), 1, (255, 0, 0), 1)

        for points in (left_discarded, right_discarded):
            for i in range(len(points[0])):
                cv2.circle(output, (points[1][i], points[0][i]), 1, (0, 0, 255), 1)

        return output

    def run(self, raw, save_images=False):
        if save_images:
            try:
                os.makedirs(save_images)
            except FileExistsError:
                pass
            
            cv2.imwrite("%s/1-raw.jpg" % save_images, raw)

        # Undistort
        img = self.camera.undistort(raw)

        if save_images:
            cv2.imwrite("%s/2-undistorted.jpg" % save_images, img)

        # Bird's eye view
        img = self.perspective.transform(img)

        if save_images:
            cv2.imwrite("%s/3-birds-eye-view.jpg" % save_images, img)

        # Apply lane color mask
        img = self.apply_color_mask(img)

        if save_images:
            cv2.imwrite("%s/4-color-mask.jpg" % save_images, img)

        # Fit lanes
        img = self.fit_lanes(img)

        if save_images:
            cv2.imwrite("%s/6-fit.jpg" % save_images, img)

        # Transform lanes onto original image
        img = cv2.addWeighted(raw, 1, self.camera.redistort(self.perspective.invert(img)), 0.5, 0)
        
        # Write curvature and position information
        if self.left_poly.last_fit and self.right_poly.last_fit:
            curvature = (self.left_poly.get_curvature(self.camera.height) +\
                self.left_poly.get_curvature(self.camera.height)) / self.pixels_per_meter
            
            if abs(curvature) > 50:
                turn = "otse"
            elif curvature < 0:
                turn = "vasakule"
            else:
                turn = "paremale"
            
            lane_center = self.right_poly.last_fit(self.camera.height) - self.left_poly.last_fit(self.camera.height)
            lane_position = (self.camera.width / 2 - lane_center) / self.pixels_per_meter
            
            cv2.putText(img, "Kurvi raadius: %.2f m (%s)" % (curvature, turn), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(img, "Asukoht keskjoone suhtes: %.2f m" % lane_position, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if save_images:
            cv2.imwrite("%s/7-final.jpg" % save_images, img)

        return img
