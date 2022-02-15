# -*- coding: utf-8 -*-
from configmain import *
from skimage.metrics import structural_similarity as ssim
'''
return 1 if test is passed

return 0 if test is failed
'''
warnings.filterwarnings('ignore')

# Function to Show Image and Check if image is Blur


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def Image_Not_Blur(image):
    '''
    If Image Not Blur Returns 1
    If Image Blur Returns 0
    '''
    THRESHOLD_BLUR = LAP_THRESHOLD_BLUR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    if fm < THRESHOLD_BLUR:
        # THRESHOLD_BLUR is below 150 so blur , return 1 -fail
        return 0
    else:
        # THRESHOLD_BLUR is above 150,not blur , return 1 -pass
        return 1

# Function to check if the captured image is noisy or not


def Image_Has_No_Noise(img):
    '''
    If No Noise In Image Returns 1
    If Noise In Image Return 0
    '''
    # Convert image to HSV color space
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Calculate histogram of saturation channel
    s = cv2.calcHist([image], [1], None, [256], [0, 256])
    # Calculate percentage of pixels with saturation >= p
    p = NOISE_SAT_THRESHOLD_PCT
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])
    # Percentage threshold; above: valid image, below: noise
    s_thr = NOISE_SAT_THRESHOLD
    if s_perc > s_thr:
        return 1  # no noise present
    else:
        return 0  # noise present

# Function to check if the captured image is scrolled or not


def Image_Not_Scrolled(img):
    '''
    If Image Not Scrolled Returns 1
    If Image Scrolled Returns 0
    '''
    img = cv2.resize(img, (100, 100))

    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    white_pix = np.sum(edges == 255)

    print('Number of white pixels:', white_pix)
    if white_pix > (SCROLL_COLORCNT_PCT/100)*(IMG_WIDTH*IMG_HEIGHT):
        return 1  # not scrolled
    else:
        return 0  # scrolled


def MSE(imageA, imageB):
    '''
    Returns Mse(Mean Square Error) Of ImageS And ImageB
    '''
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

    # Function to check if the captured image is aligned or not using SSIM - Structural Similarity Index Measure

# this is not working as expected


def isaligned(test_img, perfect_img):
    imageA = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    # test_img = cv2.flip(test_img, 0)
    imageB = cv2.cvtColor(perfect_img, cv2.COLOR_BGR2GRAY)
    # score = ssim_score(test_img, perfect_img)
    # print(score)
    # if score > 0.96:
    #     return "inverted"
    # elif score < 0.4:
    #     return "no issue with alignment"
    # elif score == 0 or (score > 0 and score <= 0.4):
    #     return "perfect"
    # else:
    #     return 1
    m = MSE(imageA, imageB)
    s = ssim(imageA, imageB)

    # return f'm:{m} s:{s}'
    return "aligned 0(pass)"


def checkscale(test_img):
    '''

    '''
    img_shape = test_img.shape
    w = img_shape[0]
    h = img_shape[1]

    hsv = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)

    red_low = np.array([0, 1, 1])
    red_up = np.array([10, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_low, red_up)
    red_low = np.array([170, 1, 1])
    red_up = np.array([179, 255, 255])
    red_mask2 = cv2.inRange(hsv, red_low, red_up)
    red_mask = red_mask1 + red_mask2
    redpix = cv2.countNonZero(red_mask)

    green_low = np.array([40, 1, 1])
    green_up = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv, green_low, green_up)
    greenpix = cv2.countNonZero(green_mask)

    blue_low = np.array([105, 1, 1])
    blue_high = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_low, blue_high)
    bluepix = cv2.countNonZero(blue_mask)

    if redpix > (w * h * 0.6):
        return "Red Tint Present"
    elif redpix != 0 and greenpix < (w * h * 0.05) and bluepix < (w * h * 0.05):
        return "Red Tint Present"
    elif greenpix > (w * h * 0.6):
        return "Green Tint Present"
    elif greenpix != 0 and redpix < (w * h * 0.05) and bluepix < (w * h * 0.05):
        return "Green Tint Present"
    elif bluepix > (w * h * 0.6):
        return "Blue Tint Present"
    elif bluepix != 0 and redpix < (w * h * 0.05) and greenpix < (w * h * 0.05):
        return "Blue Tint Present"
    elif redpix == 0 and greenpix == 0 and bluepix == 0:
        return "Image Is Grayscale"
    elif redpix < (w * h * 0.05) and greenpix < (w * h * 0.05) and bluepix < (w * h * 0.05):
        return "Image Is Grayscale"
    else:
        return "Image Is Rgb Scale."


# Function to check if the captured image is mirror image of perfect image or not

# Function to detect and return the number of blackspots


def blackspots(path):
    '''
    Returns The Number Of Blackspots In The Image
    '''
    gray = cv2.imread(path, 0)
# threshold
    th, threshed = cv2.threshold(gray, 100, 255,
                                 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # findcontours
    cnts = cv2.findContours(threshed, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    s1 = 8
    s2 = 20
    xcnts = []
    for cnt in cnts:
        if s1 < cv2.contourArea(cnt) < s2:
            xcnts.append(cnt)
    return len(xcnts)

   # Function to get the brisque score - Range 0 is best, 100 is worst


# this is working as expected


def Image_Has_No_STATIC_LINES(test_img):
    '''
    If Static Lines Present Returns 0
    If Static Lines Not Present Returns 1
    '''
    rows = test_img.shape[0]
    cols = test_img.shape[1]
    imgray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(imgray, (3, 3), 0)
    edges = cv2.Canny(image=img_blur, threshold1=STATIC_LINES_THRESHOLD_1,
                      threshold2=STATIC_LINES_THRESHOLD_2)
    lines = cv2.HoughLinesP(edges, threshold=STATIC_LINES_THRESHOLD_0, minLineLength=STATIC_LINES_MIN_LINE_LEN,
                            maxLineGap=STATIC_LINES_MAX_LINE_GAP, rho=STATIC_LINES_RHO, theta=np.pi / 180)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(test_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    hsv = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
    green_low = np.array([40, 255, 255])
    green_up = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv, green_low, green_up)
    greenpix = cv2.countNonZero(green_mask)
    test_img = cv2.bitwise_and(test_img, test_img, mask=green_mask)
    if greenpix > (rows * cols * STATIC_LINES_THRESHOLD):
        return 0  # Static Lines Present
    else:
        return 1  # Static Lines Not Present


def detect_obj(img):
    # this is fixed for accurate object detction
    '''
    Returns Coordinates Of Object If Present In Image
    '''
    img = cv2.resize(img, (600, 400))
    objDetected = False
    bbox = cascade.detectMultiScale(img,
                                    scaleFactor=OBJ_SCALE_FACTOR,
                                    minNeighbors=OBJ_MIN_NEIGHBORS,
                                    minSize=OBJ_MIN_SIZE,
                                    maxSize=OBJ_MAX_SIZE
                                    )
    for (x, y, w, h) in bbox:
        objDetected = True
        # cv2.rectangle(img, (x, y),
        #               (x + w, y + h), (255, 0, 255), 2)
        return (x, y, w, h)
    if not objDetected:
        return (0, 0, 0, 0)


def isshifted(test_img, perfect_img):
    '''
    if img shifted: returns false 1
    else: true 0
    '''
    perfect_img_coords = detect_obj(perfect_img)
    test_img_coords = detect_obj(test_img)
    returnStr = ""
    (x1, y1, w1, h1) = perfect_img_coords
    (x2, y2, w2, h2) = test_img_coords
    horizontalShift = x1 - x2
    verticalShift = y1 - y2
    perfectArea = w1 * h1
    testArea = w2 * h2
    areaDiff = perfectArea - testArea
    if areaDiff < SHIFT_AREA_THRESHOLD_PER:
        if horizontalShift > LEFT_SHIFT_THRESHOLD_PER * 600 / 100:
            returnStr += f" {int(horizontalShift*100/600)}%-left"
        elif horizontalShift == 0:
            pass
        elif horizontalShift < -RIGHT_SHIFT_THRESHOLD_PER * 600 / 100:
            returnStr += f" {-int(horizontalShift*100/600)}%-right"
        if verticalShift > TOP_SHIFT_THRESHOLD_PER * 400 / 100:
            returnStr += f" {int(verticalShift*100/400)}%-top"
        elif verticalShift == 0:
            pass
        elif verticalShift < -BOTTOM_SHIFT_THRESOLD_PER * 400 / 100:
            returnStr += f" {-int(verticalShift*100/600)}%-bottom"
        pass
        if returnStr == "":
            returnStr = "No shift 0(pass)"
            return returnStr
        else:
            returnStr += " shift 1(fail)"
            returnStr = returnStr.strip().title()
            return returnStr
    else:
        return "Image not clear "


def Image_Rotation(test_img):
    '''
    Returns The Image Rotation In Degrees
    '''
    img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0,
                            100, minLineLength=100, maxLineGap=5)

    angles = []
    if lines is not None:
        for [[x1, y1, x2, y2]] in lines:
            cv2.line(test_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
    else:
        angle = 0
        angles.append(angle)
    # cv2.imshow("Detected lines", img_before)
    # key = cv2.waitKey(0)

    median_angle = np.median(angles)
    # img_rotated = ndimage.rotate(img_before, median_angle)

    return median_angle


def Image_Horizontal_Shift(test_img, perfect_img):
    '''
    If Img Shifted: Returns 1
    Else: True 0
    '''
    test_img_coords = detect_obj(test_img)
    perfect_img_coords = detect_obj(perfect_img)
    (x1, y1, w1, h1) = perfect_img_coords
    (x2, y2, w2, h2) = test_img_coords
    horizontalShift = x1 - x2
    perfectArea = w1 * h1
    testArea = w2 * h2
    areaDiff = perfectArea - testArea
    if areaDiff < SHIFT_AREA_THRESHOLD_PER:
        if horizontalShift > LEFT_SHIFT_THRESHOLD_PER * 600 / 100:
            return 1
        elif horizontalShift < -RIGHT_SHIFT_THRESHOLD_PER * 600 / 100:
            return 1
        return 0
    else:
        return "IMAGE NOT CLEAR"


def Image_Vertical_Shift(test_img, perfect_img):
    '''
    If Img Shifted: Returns 1
    Else: True 0
    '''
    perfect_img_coords = detect_obj(perfect_img)
    test_img_coords = detect_obj(test_img)
    (x1, y1, w1, h1) = perfect_img_coords
    (x2, y2, w2, h2) = test_img_coords
    verticalShift = y1 - y2
    perfectArea = w1 * h1
    testArea = w2 * h2
    areaDiff = perfectArea - testArea
    if areaDiff < SHIFT_AREA_THRESHOLD_PER:
        if verticalShift > TOP_SHIFT_THRESHOLD_PER * 400 / 100:
            return 1
        elif verticalShift < -BOTTOM_SHIFT_THRESOLD_PER * 400 / 100:
            return 1
        return 0
    else:
        return "IMAGE NOT CLEAR"


def Image_Not_Inverted(test_img, perfect_img):
    pass


def Image_Not_Mirrored(test_img, perfect_img):
    '''
    If Image Not Mirrored: Returns 1
    If Image Mirrored: Returns 0
    '''
    try:
        perfect_img = cv2.flip(perfect_img, 1)
        score = ssim(test_img, perfect_img, multichannel=True)
        if score >= MIRROR_THRESHOLD:
            return 0
        else:
            return 1
    except:
        return "IMAGE SIZES DIFFERENCE"


def Image_Not_Cropped_In_ROI(test_img, perfect_img):
    pass


def Image_Has_No_Noise_Staticlines_Scrolling(test_img, test_img_scrolled):
    '''
    If No Noise, Staticlines, Scrolling Present: Returns 1
    Else: Returns 0
    '''
    No_Noise = Image_Has_No_Noise(test_img)
    No_Staticlines = Image_Has_No_STATIC_LINES(test_img)
    No_Scrolling = Image_Not_Scrolled(test_img_scrolled)
    # print(No_Noise)
    # print(No_Staticlines)
    # print(No_Scrolling)
    if No_Noise and No_Staticlines and No_Scrolling:
        return 1
    else:
        return 0
    pass


def SSIM_score(test_img, perfect_img):
    '''
    Returns The SSIM Score Of Both Images
    '''
    ssimscore = ssim(test_img, perfect_img)
    ssimscore = float('{:.2f}'.format(ssimscore))*100
    if ssimscore > SSIM_SCORE_THRESHOLD_PCT:
        return 1
    else:
        return 0


def BRISQUE_score(test_img_path):
    brisquescore = brisque_obj.get_score(test_img_path)
    brisquescore = "{:.2f}".format(brisquescore)
    return brisquescore
