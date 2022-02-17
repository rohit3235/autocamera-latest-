from imgtests import *
from configmain import *


def save_results(image_test_results):
    '''
    Saves The Results In The Root Directory In results.csv File
    '''
    filepath = CSV_PATH

    with open(filepath, "a+") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(image_test_results)


def generate_report(camid, test_img_path, perfect_img_path):
    # preprocessing test and perfect images
    test_img = cv2.imread(test_img_path)
    test_img = cv2.resize(test_img, (IMG_WIDTH, IMG_HEIGHT))
    perfect_img = cv2.imread(perfect_img_path)
    perfect_img = cv2.resize(perfect_img, (IMG_WIDTH, IMG_HEIGHT))
    test_img_rotate = cv2.imread(test_img_path)

    # preprocessing image for image shift calculation
    # 0 means image is read in grayscale mode
    test_img_shift = cv2.imread(test_img_path, 0)

    # 0 means image is read in grayscale mode
    perfect_img_shift = cv2.imread(perfect_img_path, 0)

    # reading original image for scrolling calculation
    test_img_scrolled = cv2.imread(test_img_path)

    # preprocessing images for SSIM calculation
    # 0 means image is read in grayscale mode
    test_img_SSIM = cv2.imread(test_img_path, 0)
    test_img_SSIM = cv2.resize(test_img_SSIM, (IMG_WIDTH, IMG_HEIGHT))

    # 0 means image is read in grayscale mode
    perfect_img_SSIM = cv2.imread(perfect_img_path, 0)
    perfect_img_SSIM = cv2.resize(perfect_img_SSIM, (IMG_WIDTH, IMG_HEIGHT))

    # checking SSIM_score of test and perfect images and then accordingly taking further actions
    ssim_score_pct = ssim(test_img_SSIM, perfect_img_SSIM)

    ssim_score_pct = float('{:.2f}'.format(ssim_score_pct))*100

    # checking SSIM condition and if satisfied, pass all the test cases
    if ssim_score_pct >= SSIM_SCORE_THRESHOLD_PCT:
        # return 1 for all tests in the code
        test_img_BRISQUE_score = BRISQUE_score(test_img_path)
        image_test_results = [1,
                              1,
                              1,
                              1,
                              0,
                              0,
                              0,
                              1,
                              1,
                              test_img_BRISQUE_score
                              ]
        image_test_results = [camid] + image_test_results
        save_results(image_test_results)
        return image_test_results

    else:
        # check all the tests
        pass
        image_test_results = [Image_Not_Blur(test_img),
                              Image_Not_Inverted(test_img, perfect_img),
                              Image_Not_Mirrored(test_img, perfect_img),
                              Image_Not_Rotated(test_img_rotate),
                              Image_Horizontal_Shift(
            test_img_shift, perfect_img_shift),
            Image_Vertical_Shift(
            test_img_shift, perfect_img_shift),
            Image_Not_Cropped_In_ROI(test_img, perfect_img),
            Image_Has_No_Noise_Staticlines_Scrolling(
            test_img, test_img_scrolled),
            SSIM_score(test_img_SSIM, perfect_img_SSIM),
            BRISQUE_score(test_img_path)
        ]
        image_test_results = [camid] + image_test_results
        save_results(image_test_results)
        return image_test_results
