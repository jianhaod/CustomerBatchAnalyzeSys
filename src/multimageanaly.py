#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/3/6
@author: Jianhao
"""
import fetch
import sys

import multiprocessing
from multiprocessing.dummy import Pool
from multiprocessing import Process, Queue, Lock
import fileinput

sys.path.append('../utils')
from Logger import *

def queuepush(q, path):
	video_detect_dict = {}
	video_detect_dict = fetch.split_video_to_frame(path)
	q.put(video_detect_dict)

def queueget(q, lock):
	while True:
		if not q.empty():
			lock.acquire()
			temp_input=q.get()
			lock.release()
			time.sleep(random.random())
		else:
			break	

# test defind
"""
def queuepush(q):
    for value in ['A', 'B', 'C']:
        print 'Put %s to queue...' % value
        q.put(value)
        time.sleep(random.random())

def queueget(q, lock):
    while True:
        if not q.empty():
            lock.acquire()
            temp_input=q.get()
            lock.release()
            time.sleep(random.random())
        else:
            break
"""

def multimageanaly(paralnum, input_path, videopath):
	"""
	param: paralnum
	param: videopath
	"""
	logger = Logger(logger='multimageanalyLog')
	pool = Pool(paralnum)

	q = Queue()
	lock = Lock()

	logger.logger.info("launch %d process to fetch analye input video!" % paralnum)
	for line in fileinput.input(videopath):
		pool.apply_async(queuepush, (q, input_path + line.strip()))

	logger.logger.info("launch %d process to consumer analye result!" % (paralnum/2))
	for index in range(paralnum/2):
		pool.apply_async(queueget, (q, lock))
	
	pool.close()
	pool.join()
