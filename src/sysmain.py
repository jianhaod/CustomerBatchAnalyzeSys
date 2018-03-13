#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/3/1
@author: Jianhao
"""

import os, sys
import yaml

import fetch
import multimageanaly 

import _init_utils_paths
from Logger import *

if __name__ == '__main__':

	logger = Logger(logger='SysMainLog')
	logger.logger.info("Start lanuch CustomerBatch detect system!")

	yamlfile = open('../config/config.yaml')
	sysdict = yaml.load(yamlfile)

	# fetch video stream frame
	input_path = sysdict['video_stream_input_path']
	output_path = sysdict['video_stream_output_path']
	input_local_path = sysdict['input_local_path']
	paralnum = sysdict['paralnum']

	# get video stream file name list from local
	fetch_list = input_local_path + sysdict['redis']['wait_file_list']
	#for line in fileinput.input(fetch_list):
	multimageanaly.multimageanaly(paralnum, input_path, fetch_list)

	#for line in fileinput.input(fetch_list):
	#	video_detect_dict = {}
	#	video_detect_dict = fetch.split_video_to_frame(input_path + line.strip())

	logger.logger.info("End!")
