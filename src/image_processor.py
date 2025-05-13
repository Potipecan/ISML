import cv2
import numpy as np
from PIL import Image
import os

def vect_intersect(a1, a2, b1, b2):
    s = np.vstack([a1, a2, b1, b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return float('inf'), float('inf')
    return x / z, y / z

class ImageProcessor:
    def __init__(self, image: Image, pre_crop = 30):
        self._raw_image = np.array(image)
        self._pre_crop = pre_crop
        self._processed = self._raw_image[pre_crop:-pre_crop, pre_crop:-pre_crop]      
        # self._processed = self._raw_image.copy()
        self._corners = None
    
    def find_corners(self):
        self._corners = np.array(self._get_bounding_rect(), dtype=int)

    
    def _get_bounding_rect(self):
        if len(self._processed.shape) > 2:
            self._processed = cv2.cvtColor(self._processed, cv2.COLOR_BGR2GRAY)             # grayscale
        self._processed = cv2.GaussianBlur(self._processed, (3, 3), 0)      # blur
        self._processed = cv2.Laplacian(self._processed, cv2.CV_8U, ksize=3)            # laplacian
        self._processed = cv2.Canny(self._processed, 10, 300, apertureSize=3)           # canny
        lines = cv2.HoughLinesP(self._processed, 1, np.pi / 180, threshold=10, minLineLength=200, maxLineGap=30)
        
        x_centers = lines[:, 0, 0] + lines[:, 0, 2]
        y_centers = lines[:, 0, 1] + lines[:, 0, 3]
        ax_diffs = np.abs(lines[:, 0, 0 : 2] - lines[:, 0, 2 : 4])
        ax_diffs = np.diff(ax_diffs, axis=1).flatten()
        h_edge_mask = np.where(ax_diffs > 0)[0]
        v_edge_mask = np.where(ax_diffs < 0)[0]
    
        te = lines[h_edge_mask[x_centers[h_edge_mask].argmin()]][0].reshape((2, 2)) # top edge detection by smallest average height
        be = lines[h_edge_mask[x_centers[h_edge_mask].argmax()]][0].reshape((2, 2)) # bottom edge detection by largest average height
        le = lines[v_edge_mask[y_centers[v_edge_mask].argmin()]][0].reshape((2, 2)) # left edge
        re = lines[v_edge_mask[y_centers[v_edge_mask].argmax()]][0].reshape((2, 2)) # right edge
    
        # return intersections of the four edges
        return (
            vect_intersect(*te, *le),
            vect_intersect(*be, *le),
            vect_intersect(*te, *re),
            vect_intersect(*be, *re)
        )
    
    def get_corners(self):
        if self._corners is None:
            return None
        return self._corners + self._pre_crop
        
    def set_corners(self, corners):
        self._corners = corners - self._pre_crop
        

                
        
def _debug_show_processed(self):
        cv2.imshow("Processed", self._processed)
        cv2.waitKey(0)