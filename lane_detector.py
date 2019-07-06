import cv2
import numpy as np
import os


class BufferedPolyFit:
    def __init__(self, order, size=5, max_skip=3):
        self.order = order
        self.size = size
        self.max_skip = max_skip
        self.x_buffer = []
        self.y_buffer = []
        self.skipped = 0
    
    def get_buffered_fit(self):
        if not len(self.x_buffer):
            return None
        
        return np.poly1d(np.polyfit(
            np.concatenate(self.x_buffer),
            np.concatenate(self.y_buffer),
            self.order,
            #w=sum([[i+1]*len(self.x_buffer[i]) for i in range(len(self.x_buffer))], [])
        ))

    def fit(self, x, y):
        current_fit = self.get_buffered_fit()

        if not len(x):
            return current_fit
        
        self.x_buffer.append(x)
        self.y_buffer.append(y)

        if len(self.x_buffer) > self.size:
            self.x_buffer.pop(0)
            self.y_buffer.pop(0)

        return self.get_buffered_fit()


class LaneDetector:
    def __init__(self, camera, perspective, hood_size=15):
        self.camera = camera
        self.perspective = perspective
        self.left_poly = BufferedPolyFit(2)
        self.right_poly = BufferedPolyFit(2)
        self.hood_size = hood_size

    def apply_color_mask(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_yellow = cv2.inRange(hsv, np.array([20, 50, 75]), np.array([110, 255, 255]))
        mask_white = cv2.inRange(gray, 200, 255)
        mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
        
        return cv2.bitwise_and(gray, mask_yw)

    def get_lane_points(self, img, offset=0):
        y, x = np.nonzero(img)
        i = np.random.choice(len(x), 2000)
        y = y[i]
        x = x[i]        
        
        keep = [np.array([], np.uint8), np.array([], np.uint8)]
        discard = [np.array([], np.uint8), np.array([], np.uint8)]
        step = img.shape[0] // 10

        for row in range(0, img.shape[0], step):
            in_row = np.where((y >= row) & (y <= row + step))

            if not len(in_row):
                continue

            if np.std(x[in_row]) < 100:
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
        
        if left_line and right_line:
            for y in range(img.shape[0] - self.hood_size):
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

        # Apply lane color mask and crop car hood
        img = self.apply_color_mask(img)
        img[img.shape[0] - self.hood_size:,:] = 0

        if save_images:
            cv2.imwrite("%s/4-color-mask.jpg" % save_images, img)

        # Fit lanes
        img = self.fit_lanes(img)

        if save_images:
            cv2.imwrite("%s/6-fit.jpg" % save_images, img)

        # Transform lanes onto original image
        img = cv2.add(raw, self.camera.redistort(self.perspective.invert(img)))
        
        if save_images:
            cv2.imwrite("%s/7-final.jpg" % save_images, img)

        return img
