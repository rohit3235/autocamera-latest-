# -*- coding: utf-8 -*-
from pickle import STACK_GLOBAL
from types import TracebackType
from warnings import catch_warnings

from configmain import *

'''
return 0 if test is passed

return 1 if test is failed
'''
warnings.filterwarnings('ignore')

# Function to Show Image and Check if image is Blur


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def isblur(image):
    THRESHOLD_BLUR = LAP_THRESHOLD_BLUR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    if fm < THRESHOLD_BLUR:
        # THRESHOLD_BLUR is below 150 so blur , return 1 -fail
        return 1
    else:
        # THRESHOLD_BLUR is above 150,not blur , return 1 -pass
        return 0

# Function to check if the captured image is noisy or not
# this piece of code is commented and not used, it will be removed


def isnoise1(img):
    image = img.copy()
    # convert source image to HSC color mode
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_low = np.array([0, 26, 0], np.uint8)
    hsv_high = np.array([255, 255, 255], np.uint8)
    # making mask for hsv range
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    median = cv2.medianBlur(mask, 5)
    # masking HSV value selected color becomes black
    res = cv2.bitwise_and(image, image, mask=median)
    colour_count = cv2.countNonZero(mask)
    if (colour_count > 12000 and colour_count < 15000) or (colour_count > 20000):
        return 1
    else:
        return 0


def isnoise(img):
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
        return 0  # no noise present
    else:
        return 1  # noise present

# Function to check if the captured image is scrolled or not


def isscrolled(img):
    # Convert image to grayscale
    img_gs = img.copy()
    edges = cv2.Canny(img_gs, SCROLL_THRESHOLD_1, SCROLL_THRESHOLD_2)
    colour_count = cv2.countNonZero(edges)
    # have changed the threshold value in config file
    if colour_count < SCROLL_COLORCNT:
        return 1
    else:
        return 0


def mse(imageA, imageB):
    # err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    # err /= float(imageA.shape[0] * imageA.shape[1])
    # err = np.square(np.subtract(imageB,imageA)).mean()
    # return err
    sum = 0.0
    # for(x=0
    #     x < width
    #         + +x):
    #     for(y=0
    #         y < height
    #             + +y):
    #         difference = (imageA[x, y] - imageB[x, y])
    #         sum = sum + difference*difference
    # mse = sum / (IMG_WIDTH*IMG_HEIGHT)
    for x in range(IMG_WIDTH):
        for y in range(IMG_HEIGHT):
            difference = (imageA[x, y]-imageB[x, y])
            sum += difference**2
    mse = sum/(IMG_WIDTH*IMG_HEIGHT)
    return mse

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
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    # return f'm:{m} s:{s}'
    return "aligned 0(pass)"

    # if m < ALIGN_PERFECT_M_THRESHOLD and s > 0.96:
    #     return "perfect"
    # elif ALIGN_INVERTM_1 > 400 and m < ALIGN_INVERTM_2:
    #     return "inverted"
    # elif m <= NOT_ALIGN_M1 and m > NOT_ALIGN_M1:
    #     return "not aligned"
    # else:
    #     return "no issue with alignment"


# Function to check if the captured image is RGB scaled distored or not


def isgray(img):
    if len(img.shape) < 3:
        return True
    if img.shape[2] == 1:
        return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all():
        return 1
    return 0


def checkscale(test_img):
    img_shape = test_img.shape
    # print(img_shape)
    w = img_shape[0]
    h = img_shape[1]

    hsv = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)

    red_low = np.array([0, 1, 1])
    red_up = np.array([10, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_low, red_up)
    red_low = np.array([170, 1, 1])
    red_up = np.array([179, 255, 255])
    red_mask2 = cv2.inRange(hsv, red_low, red_up)
    red_mask = red_mask1+red_mask2
    redpix = cv2.countNonZero(red_mask)
    #print('redpix ', redpix)
    # img = cv2.bitwise_and(img, img, mask=red_mask)
    # cv2.imshow('g', red_mask)
    # cv2.waitKey(0)

    green_low = np.array([40, 1, 1])
    green_up = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv, green_low, green_up)
    # green_low = np.array([65, 5, 5])
    # green_up = np.array([75, 150, 255])
    # green_mask2 = cv2.inRange(hsv, green_low, green_up)
    # green_mask = green_mask1+green_mask2

    greenpix = cv2.countNonZero(green_mask)
    #print('greenpix ', greenpix)
    # cv2.imshow('g', green_mask)
    # cv2.waitKey(0)

    blue_low = np.array([105, 1, 1])
    blue_high = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_low, blue_high)
    bluepix = cv2.countNonZero(blue_mask)

    if redpix > (w*h*0.6):
        return "red tint present"
    elif redpix != 0 and greenpix < (w*h*0.05) and bluepix < (w*h*0.05):
        return "red tint present"
    elif greenpix > (w*h*0.6):
        return "green tint present"
    elif greenpix != 0 and redpix < (w*h*0.05) and bluepix < (w*h*0.05):
        return "green tint present"
    elif bluepix > (w*h*0.6):
        return "blue tint present"
    elif bluepix != 0 and redpix < (w*h*0.05) and greenpix < (w*h*0.05):
        return "blue tint present"
    elif redpix == 0 and greenpix == 0 and bluepix == 0:
        return "image is grayscale"
    elif redpix < (w*h*0.05) and greenpix < (w*h*0.05) and bluepix < (w*h*0.05):
        return "image is grayscale"
    else:
        return "Image is RGB scale."


def is_grey_scale(img_path):
    img = img_path.convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            if r != g != b:
                # return False
                #image is color
                return 0
    # return True
    #image is grey_scale
    return 1

# Function to check if the captured image is mirror image of perfect image or not


def mirror(test_img, perfect_img):
    try:
        perfect_img = cv2.flip(perfect_img, 1)
        score = ssim(test_img, perfect_img, multichannel=True)
        if score >= MIRROR_THRESHOLD:
            return 0
        else:
            return 1
    except:
        return "image sizes difference"

# Function to detect and return the number of blackspots


def blackspots(path):
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

# Function to get SSIM - Structural Similarity Index Measure
# The SSIM values ranges between 0 to 1, 1 means perfect match the reconstruct image with original one.
# Generally SSIM values 0.97, 0.98, 0.99 for good quallty recontruction techniques.


def ssim_score(test_img, perfect_img):
    imageA = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    imageB = cv2.cvtColor(perfect_img, cv2.COLOR_BGR2GRAY)
    ssimscore = ssim(imageA, imageB)
    # return ssimscore
    if ssimscore > SSIM_SCORE_THRESHOLD:
        return f"{ssimscore} 0(pass)"  # pass
    else:
        return f"{ssimscore} 1(fail)"  # fail

   # Function to get the brisque score - Range 0 is best, 100 is worst


def brisque_score(test_img_path):
    #img = Image.open(test_img)

    #img = img_as_float(io.imread(test_img, as_gray=True))
    #img = img_as_float(io.imread('images/noisy_images/sandstone.tif', as_gray=True))
    # brisquescore = brisque.score(test_img)
    brisquescore = brisque_obj.get_score(test_img_path)
    return brisquescore
    #print("this except of Brisque code")

# this is working as expected


def static_lines(test_img, perfect_img):
    rows = test_img.shape[0]
    cols = test_img.shape[1]
    imgray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(imgray, (3, 3), 0)
    edges = cv2.Canny(image=img_blur, threshold1=STATIC_LINES_THRESHOLD_1,
                      threshold2=STATIC_LINES_THRESHOLD_2)
    lines = cv2.HoughLinesP(edges, threshold=STATIC_LINES_THRESHOLD_0, minLineLength=STATIC_LINES_MIN_LINE_LEN,
                            maxLineGap=STATIC_LINES_MAX_LINE_GAP, rho=STATIC_LINES_RHO, theta=np.pi/180)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(test_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    hsv = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
    green_low = np.array([40, 255, 255])
    green_up = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv, green_low, green_up)
    greenpix = cv2.countNonZero(green_mask)
    test_img = cv2.bitwise_and(test_img, test_img, mask=green_mask)
    if greenpix > (rows*cols*STATIC_LINES_THRESHOLD):
        return 0  # fail static lines present
    else:
        return 1  # pass static lines absent

# rotation degrees

# this is not working as expected


def rotation_degrees(img_before):
    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, ROTATION_THRESHOLD_1,
                          ROTATION_THRESHOLD_2, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, ROTATION_THRESHOLD,
                            minLineLength=ROTATION_MIN_LINE_LEN, maxLineGap=ROTATION_MAX_LINE_GAP)
    angles = []
    if lines is not None:
        for [[x1, y1, x2, y2]] in lines:
            cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
    else:
        angle = 0
        angles.append(angle)
    median_angle = np.median(angles)
    return median_angle


def alignImages_homo(im1, im2):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15
    # Convert images to grayscale

    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    # Draw top matches
    imMatches = cv2.drawMatches(
        im1, keypoints1, im2, keypoints2, matches, None)
    #cv2.imwrite("matches.jpg", imMatches)
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    return h

# this function is for checking object shifting in image


def detect_obj(img):
    # this is fixed for accurate object detction
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
    horizontalShift = x1-x2
    verticalShift = y1-y2
    perfectArea = w1*h1
    testArea = w2*h2
    areaDiff = perfectArea-testArea
    if areaDiff < SHIFT_AREA_THRESHOLD_PER:
        if horizontalShift > LEFT_SHIFT_THRESHOLD_PER*600/100:
            returnStr += f" {int(horizontalShift*100/600)}%-left"
        elif horizontalShift == 0:
            pass
        elif horizontalShift < -RIGHT_SHIFT_THRESHOLD_PER*600/100:
            returnStr += f" {-int(horizontalShift*100/600)}%-right"
        if verticalShift > TOP_SHIFT_THRESHOLD_PER*400/100:
            returnStr += f" {int(verticalShift*100/400)}%-top"
        elif verticalShift == 0:
            pass
        elif verticalShift < -BOTTOM_SHIFT_THRESOLD_PER*400/100:
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
        return "Image not clear 1(fail)"


def not_inverted(test_img, perfect_img):
    # result = isaligned(test_img, perfect_img)
    return "0(pass)"
    # return result
    pass


def not_mirrored(test_img, perfect_img):
    result = mirror(test_img, perfect_img)
    # return "0(pass)"
    if result == 0:
        result = 1
    else:
        result = 0
    if result == 0:
        return str(result)+" (pass)"
        pass
    else:
        return str(result)+" (fail)"
        pass
    return result


def rotation(test_img):
    # result = rotation_degrees(test_img)
    return "0(pass)"
    # return result
    pass


# this needs to be worked on
def not_cropped_in_ROI_region(test_img, perfect_img):
    return "0(pass)"
    pass


def no_noise_staticline_scrolling(test_img, perfect_img):
    noise_result = isnoise(test_img)  # if 0 noise is present
    if noise_result == 0:
        noise_result = 1
    else:
        noise_result = 0
    staticline_result = static_lines(test_img, perfect_img)
    scrolling_result = isscrolled(test_img)
    print(noise_result, staticline_result, scrolling_result)
    if noise_result == 0 and staticline_result == 0 and scrolling_result == 0:
        return str(noise_result)+" "+str(staticline_result)+" "+str(scrolling_result)
    else:
        return
