import cv2
import numpy as np


class Perspective:
    def __init__(self):
        src = np.float32([ [613, 445], [689, 445], [895, 553], [472, 553] ])
        dst = np.float32([ [225, -70], [1280 - 225, -70], [1280 - 225, 720 - 70], [225, 720 - 70] ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_i = cv2.getPerspectiveTransform(dst, src)
    
    def transform(self, img):
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]))
    
    def invert(self, img):
        return cv2.warpPerspective(img, self.M_i, (img.shape[1], img.shape[0]))
