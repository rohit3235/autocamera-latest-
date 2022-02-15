#from dataclasses import dataclass
#from this import d
import math
import cv2
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
# path = r'C:\Users\arun_\Downloads\CanProjects\AutomatedCamera\auto_python_package\autocameratest2\data\TestImages\rotate4.png'
# img_before = cv2.imread(path)
#cv2.imshow("Before", img_before)
#key = cv2.waitKey(0)

# img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
# img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
# lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0,
#                         100, minLineLength=100, maxLineGap=5)

# angles = []

# for [[x1, y1, x2, y2]] in lines:
#     cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
#     angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#     angles.append(angle)

#cv2.imshow("Detected lines", img_before)
#key = cv2.waitKey(0)

# median_angle = np.median(angles)
#img_rotated = ndimage.rotate(img_before, median_angle)

# print(median_angle)


img1 = cv2.imread('./data/TestImages/perfect.png')
img1 = cv2.resize(img1, (100, 100))
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('./data/TestImages/perfect.png')
img2 = cv2.resize(img2, (100, 100))
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

score = ssim(img1, img2, channel_axis=2)
print(score)
