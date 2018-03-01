#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/3/1
@author: Jianhao
"""

import os, sys
import yaml

import fetch 

sys.path.append('/root/workspace/CustomerBatchAnalyzeSys/utils')
from Logger import *

if __name__ == '__main__':

	logger = Logger(logger='SysMainLog')
	logger.logger.info("Start lanuch CustomerBatch detect system!")

	yamlfile = open('../config/config.yaml')
	sysdict = yaml.load(yamlfile)

	# fetch video stream frame
	input_path = sysdict['video_stream_input_path']
	output_path = sysdict['video_stream_output_path']

	video_detect_dict = {}
	video_detect_dict = fetch.split_video_to_frame(input_path)

	logger.logger.info("End!")
