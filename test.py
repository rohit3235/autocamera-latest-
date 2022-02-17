#from dataclasses import dataclass
#from this import d
# import math
# import cv2
# import numpy as np
# from scipy import ndimage
# from skimage.metrics import structural_similarity as ssim
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


# img1 = cv2.imread('./data/TestImages/perfect.png')
# img1 = cv2.resize(img1, (100, 100))
# # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.imread('./data/TestImages/perfect.png')
# img2 = cv2.resize(img2, (100, 100))
# # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# score = ssim(img1, img2, channel_axis=2)
# print(score)


# import numpy as np
# import cv2
# import math
# from scipy import ndimage

# img_before = cv2.imread('./data/TestImages/rotate45.png')
# # img_before = cv2.resize(img_before, (1024, 576))

# cv2.imshow("Before", img_before)
# cv2.moveWindow('Before', 10, 10)
# key = cv2.waitKey(0)

# img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
# img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
# lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0,
#                         100, minLineLength=100, maxLineGap=5)

# angles = []

# for [[x1, y1, x2, y2]] in lines:
#     cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
#     angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#     angles.append(angle)

# cv2.imshow("Detected lines", img_before)
# cv2.moveWindow('Detected lines', 10, 10)
# key = cv2.waitKey(0)

# median_angle = np.median(angles)
# # img_rotated = ndimage.rotate(img_before, median_angle)

# print(f"Angle is {median_angle:.04f}")
# # cv2.imwrite('rotated.jpg', img_rotated)


# import cv2
# import numpy as np
# import imutils
# from skimage.metrics import structural_similarity as ssim
# img1 = cv2.imread('./data/TestImages/perfect.png', 0)
# img1 = cv2.resize(img1, (600, 600))
# img2 = cv2.imread('./data/TestImages/rotate45.png', 0)
# img2 = cv2.resize(img2, (600, 600))
# ssim_threshold = 95
# ssimscore = ssim(img1, img2)

# angles = []
# for i in range(1, 360):
#     # img2 = imutils.rotate(img2, 1)
#     sampleimg = img2.copy()
#     sampleimg = imutils.rotate(sampleimg, -i)
#     sampleimg = cv2.resize(sampleimg, (600, 400))
#     ssimscoresample = (ssim(sampleimg, img1))
#     ssimscoresample = float('{:.2f}'.format(ssimscoresample))*100
#     ssimscoresample = float(ssimscoresample)
#     print(ssimscoresample)
#     print(f'angle: {i}')
#     angles.append(ssimscoresample)
#     cv2.imshow('img', sampleimg)
#     cv2.waitKey(1)
# angles.sort(reverse=True)
# print(angles)


# import cv2
# import math
# from scipy import ndimage
# import numpy as np

# img_before = cv2.imread('./data/TestImages/perfect.png')
# img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
# img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
# lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0,
#                         100, minLineLength=100, maxLineGap=5)

# angles = []

# for x1, y1, x2, y2 in lines[0]:
#     angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#     #angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
#     angles.append(angle)

# median_angle = np.median(angles)
# rotated_angle = abs(round(median_angle))
# #img_rotated = ndimage.rotate(img_before, median_angle)
# #img_rotated = ndimage.rotate(img_before, rotated_angle)
# #print("Angle is {}".format(median_angle))
# print("Angle is {:.2f}".format(rotated_angle))
# #print(f"Angle is {rotated_angle:.04f}")


import cv2
import numpy as np

CHECKERBOARD = (6, 9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
    cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# read images and for each image:
img = cv2.imread('./data/TestImages/perfect.png')
img_shape = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the chess board corners
ret, corners = cv2.findChessboardCorners(
    gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
    imgpoints.append(corners)
###

# calculate K & D
N_imm =  1
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
