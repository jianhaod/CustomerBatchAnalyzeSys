#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/3/9
@author: Jianhao
"""

import _init_utils_paths 
from PersonDetect import *
import cv2

detectobj = PersonDetector("../config/persondetect.yaml")
#detectobj.detectimage("../localtest/608.jpg")
#detectobj.detectimage("../localtest/608.jpg")

camera_detect_img = {}
img = cv2.imread("../localtest/608.jpg")
img2 = cv2.imread("../localtest/4240.jpg")
camera_detect_img[0] = img
camera_detect_img[1] = img2
local_path = "/SQ2130/20180310/1520673513.MP4"
reslut = {}
reslut[0] = camera_detect_img
reslut[1] = local_path

print reslut[0]
print reslut[1]
detectobj.fast_rcnn_detect(reslut)
