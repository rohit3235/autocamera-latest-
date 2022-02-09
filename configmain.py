# -*- coding: utf-8 -*-
# used for relative path calculation
import os
import numpy as np
import math
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image

import warnings
import csv
# this import are for brisque
from skimage import io, img_as_float
#import imquality.brisque as brisque
import traceback
from configparser import ConfigParser
#global_path = "C:/Users/admin/Documents/Dashboard_Camera_Testing"
#global_path = "C:/Users/arun_/Downloads/CanProjects/AutomatedCamera/AutomatedCamera/autocamera"
#global_path = "C:/Users/arun_/Downloads/CanProjects/AutomatedCamera/AutomatedCameraArun"


#global_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
BASE_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
#ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
IMAGE_FOLD_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..'))
#IMAGE_FOLD_PATH = "./autocameratest2/data/TestImages"
# print(global_path)

config = ConfigParser()
config.read('./config.ini')
# image dimensions
IMG_WIDTH = int(config['IMAGEDIMENSIONS']['IMG_WIDTH'])
IMG_HEIGHT = int(config['IMAGEDIMENSIONS']['IMG_HEIGHT'])
# blur threshold for laplcaian blur
LAP_THRESHOLD_BLUR = int(config['BLUR']['BLUR_LAP_THRESHOLD'])
# scrolling thresholds
SCROLL_THRESHOLD_1 = int(config['SCROLLING']['SCROLL_THRESHOLD_1'])
SCROLL_THRESHOLD_2 = int(config['SCROLLING']['SCROLL_THRESHOLD_2'])
SCROLL_COLORCNT = int(config['SCROLLING']['SCROLL_COLORCOUNT'])
# alignment thresholds
ALIGN_PERFECT_M_THRESHOLD = int(
    config['ALIGNMENT']['ALIGN_PERFECT_M_THRESHOLD'])
ALIGN_INVERTM_1 = int(config['ALIGNMENT']['ALIGN_INVERT_M1'])
ALIGN_INVERTM_2 = int(config['ALIGNMENT']['ALIGN_INVERT_M2'])
NOT_ALIGN_M1 = int(config['ALIGNMENT']['NOT_ALIGN_M1'])
NOT_ALIGN_M2 = int(config['ALIGNMENT']['NOT_ALIGN_M2'])
# mirror thresholds
MIRROR_THRESHOLD = float(config['MIRROR']['MIRROR_THRESHOLD'])
# static lines thresholds
STATIC_LINES_THRESHOLD = float(config['STATICLINES']['STATIC_LINES_THRESHOLD'])
STATIC_LINES_THRESHOLD_0 = int(
    config['STATICLINES']['STATIC_LINES_THRESHOLD_0'])
STATIC_LINES_THRESHOLD_1 = int(
    config['STATICLINES']['STATIC_LINES_THRESHOLD_1'])
STATIC_LINES_THRESHOLD_2 = int(
    config['STATICLINES']['STATIC_LINES_THRESHOLD_2'])
STATIC_LINES_MIN_LINE_LEN = int(
    config['STATICLINES']['STATIC_LINES_MIN_LINE_LEN'])
STATIC_LINES_MAX_LINE_GAP = int(
    config['STATICLINES']['STATIC_LINES_MAX_LINE_GAP'])
STATIC_LINES_RHO = int(config['STATICLINES']['STATIC_LINES_RHO'])
# rotation thresholds
ROTATION_THRESHOLD_1 = int(config['ROTATION']['ROTATION_THRESHOLD_1'])
ROTATION_THRESHOLD_2 = int(config['ROTATION']['ROTATION_THRESHOLD_2'])
ROTATION_MIN_LINE_LEN = int(config['ROTATION']['ROTATION_MIN_LINE_LEN'])
ROTATION_MAX_LINE_GAP = int(config['ROTATION']['ROTATION_MAX_LINE_GAP'])
ROTATION_THRESHOLD = int(config['ROTATION']['ROTATION_THRESHOLD'])
# noise thresholds
NOISE_SAT_THRESHOLD_PCT = float(
    config['NOISE']['NOISE_SAT_THRESHOLD_PCT'])
NOISE_SAT_THRESHOLD = float(config['NOISE']['NOISE_SAT_THRESHOLD'])


"""Reading data from config.ini file"""
"""Load configuration from .ini file."""
""" import configparser
# Read local `config.ini` file.
Config = configparser.ConfigParser()
Config.read('config.ini')

# Get values from our .ini file
Config.get('perfectimagepath', 'testimagespath','testresultspath')
Config['perfectimagepath'],['testimagespath'],['testresultspath']


def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1 """
