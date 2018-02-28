#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/2/13
@author: Jianhao
"""
import logging
import os.path
import time

class Logger(object):

	def __init__(self, logger):
		"""
		"""
		self.logger = logging.getLogger(logger)
		self.logger.setLevel(logging.DEBUG)

		rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
		log_path = os.path.dirname(os.getcwd()) + '/Logs/'
		if os.path.exists(log_path):
			print("yes")
		else:
			os.makedirs(log_path)
		log_name = log_path + rq + '.log'
		fh = logging.FileHandler(log_name)
		self.logger.setLevel(logging.INFO)

		ch = logging.StreamHandler()
		#ch.setlevel(logging.INFO)

		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		fh.setFormatter(formatter)
		ch.setFormatter(formatter)

		self.logger.addHandler(fh)
		self.logger.addHandler(ch)
