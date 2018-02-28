#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/2/12
@author: Jianhao
"""

import cv2
from PIL import Image

# add logging lib
import sys
sys.path.append('/root/workspace/CustomerBatchAnalyzeSys/utils')
from Logger import *

logger = Logger(logger='fetchLog')

def split_video_to_frame(video_path, interval=4, lenth=10):
	"""
	note: need to install moviepy
	:parm video_path: video file local path
	:parm interval:	each seconds to get interval frame
	"""
	camera_detect_img = {}
	clipobj = VideoFileClip(video_path)

	number = lenth * interval
	count = 0
	total = 0

	for im_index in clip.iter_frames():
		if count % interval == 0 and total <= number - 1:
			img = np.array(Image.fromarray(im_index))
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			camera_detect_img[count] = img
			count += 1
		total += 1

	if count == number:
		return camera_detect_img
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

	clip.__del__()
	return camera_detect_img
