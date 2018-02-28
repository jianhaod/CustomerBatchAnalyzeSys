#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/2/28
@author: Jianhao
"""

import time
import sys
sys.path.append('/root/workspace/CustomerBatchAnalyzeSys/utils')
from Logger import *

class TestMyLog(object):
	def __init__(self):
		self.testlogger = Logger(logger='TestMyLog')

	def print_log(self):
		self.testlogger.logger.info("sys print info message")
		self.testlogger.logger.debug("sys print debug message")
		self.testlogger.logger.warn("sys print warn message")
		self.testlogger.logger.error("sys print error message")
		self.testlogger.logger.critical("sys print critical message")

testlog = TestMyLog()
testlog.print_log()
