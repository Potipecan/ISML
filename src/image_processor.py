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
    
        # cv2.line(imdata, te[0], te[1], (255, 255, 0), 2)
        # cv2.line(imdata, be[0], be[1], (255, 255, 0), 2)
        # cv2.line(imdata, le[0], le[1], (255, 255, 0), 2)
        # cv2.line(imdata, re[0], re[1], (255, 255, 0), 2)
        # 
        # plt.imshow(cv2.cvtColor(imdata, cv2.COLOR_BGR2RGB))
        # plt.show()
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
        
    def _align_image(self, corners):
        # Define the 4 corner points of the tetragon in the image
        # Format: np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        # The order should be: top-left, top-right, bottom-right, bottom-left
        pts_src = np.array(corners[[0, 1, 3, 2]], dtype=np.float32) # swap bottom corners
        
        # Compute the width and height of the destination rectangle        
        width_top = np.linalg.norm(pts_src[0] - pts_src[1])
        width_bottom = np.linalg.norm(pts_src[3] - pts_src[2])
        width = max(int(width_top), int(width_bottom))
        
        height_left = np.linalg.norm(pts_src[0] - pts_src[3])
        height_right = np.linalg.norm(pts_src[1] - pts_src[2])
        height = max(int(height_left), int(height_right))
        
        # Destination points for the rectangle (in order: TL, TR, BR, BL)
        pts_dst = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])
        
        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        
        # Apply the perspective warp
        warped = cv2.warpPerspective(self._raw_image, M, (width, height))
        return warped
        
    def process_image(self, settings, schemas, tag_map, output_dir, file_stem):
        corners = np.array(settings['corners'])
        h_width = settings['h_width']
        v_width = settings['v_width']
        rotation = settings.get('rotation', 0)
        tags = schemas[settings['schema_key']]['tags']
        
        aligned = self._align_image(corners)
        
        x_coords = np.linspace(0, aligned.shape[1], 26, dtype=float)
        y_coords = np.linspace(0, aligned.shape[0], 51, dtype=float)
        
        x_tags = tags if len(tags) == 25 else None
        y_tags = tags if len(tags) == 50 else None
        im_index = 0
        
        for x in range(25):
            x_from = int(np.round(x_coords[x] + v_width / 2))
            x_to = int(np.round(x_coords[x + 1] - v_width / 2))
            
            tag = x_tags[x] if x_tags is not None else None
            
            for y in range(50):
                y_from = int(np.round(y_coords[y] + h_width / 2))
                y_to = int(np.round(y_coords[y + 1] - h_width / 2))

                tag = y_tags[y] if y_tags is not None else tag
                sub_img = aligned[y_from:y_to, x_from:x_to]
                
                if rotation != 0:
                    match rotation:
                        case 90:
                            rotate_code = cv2.ROTATE_90_CLOCKWISE
                        case 180:
                            rotate_code = cv2.ROTATE_180
                        case 270:
                            rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
                        case _:
                            raise ValueError('Invalid rotation')
                    sub_img = cv2.rotate(sub_img, rotate_code)
                    
                # further processing here
                    
                cv2.imwrite(os.path.join(output_dir, tag, f"{file_stem}_{im_index}.png"), cv2.cvtcolor(sub_img, cv2.COLOR_BGR2RGB))
                im_index += 1
                
        
def _debug_show_processed(self):
        cv2.imshow("Processed", self._processed)
        cv2.waitKey(0)