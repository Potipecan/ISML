import cv2
import numpy as np

# Load the image
image = cv2.imread("../data/raw/sc070.png")

# Define the 4 corner points of the tetragon in the image
# Format: np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
# The order should be: top-left, top-right, bottom-right, bottom-left
pts_src = np.float32([
    [78, 118],
    [1510, 95],
    [1537, 2216],
    [98, 2244]
])

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
warped = cv2.warpPerspective(image, M, (width, height))

# Save or display the result
cv2.imshow("cropped_rectangle.jpg", warped)
cv2.waitKey(0)
