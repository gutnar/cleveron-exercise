import cv2
import numpy as np


class Perspective:
    def __init__(self):
        src = np.float32([ [570, 450], [715, 450], [1346, 650], [50, 650] ])
        dst = np.float32([ [0, 0], [1280, 0], [1280, 720], [0, 720] ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_i = cv2.getPerspectiveTransform(dst, src)
    
    def transform(self, img):
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]))
    
    def invert(self, img):
        return cv2.warpPerspective(img, self.M_i, (img.shape[1], img.shape[0]))
