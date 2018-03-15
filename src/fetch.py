#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/2/12
@author: Jianhao
"""

import cv2
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip

import _init_utils_paths
from Logger import *

logger = Logger(logger='FetchFunc')

def split_video_to_frame(video_path, interval=4, lenth=10):
	"""
	note: need to install moviepy
	:parm video_path: video file local path
	:parm interval:	each seconds to get interval frame
	"""
	#logger = Logger(logger='fetchLog')
	camera_detect_img = {}
	clipobj = VideoFileClip(video_path)

	number = lenth * interval
	count = 0
	total = 0

	for im_index in clipobj.iter_frames():
		if total % interval == 0 and count <= number - 1:
			img = np.array(Image.fromarray(im_index))
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			camera_detect_img[count] = img
			count += 1
		total += 1
	if count == number:
                pass
	elif count < number:
		logger.logger.warn("Waring true frame number:%s is small need to expend" % str(count))
		for index in range(number - count):
			camera_detect_img[count] = camera_detect_img[count - 1]
			count += 1
	elif count > number:
		logger.logger.warn("Waring true frame number:%s is big need to cut" % str(count))
		for index in range(count - number):
			count -= 1
			del camera_detect_img[count]

	clipobj.__del__()
        logger.logger.info("split video to frame %s" % video_path)
	return camera_detect_img
