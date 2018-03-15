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

import _init_utils_paths
from Logger import *

def queuepush(itemkey, q, lock, inner_batch_num):
	for i in range(inner_batch_num):
		video_detect_dict = {}
		item_key, video_path = conn.brpop(itemkey, timeout=60) 
		video_detect_dict = fetch.split_video_to_frame(video_path)
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

def multimageanaly(inner_batch_num, wait_fetch_list):
	"""
	param: inner_batch_num 
	param: videopath
	"""
	logger = Logger(logger='multimageanalyLog')
    paralnum = len(wait_fetch_list)
	pool = Pool(paralnum)

	q = Queue()
	lock = Lock()

	logger.logger.info("launch %d process to fetch analye input video!" % paralnum)
    
	for index in range(len(wait_fetch_list)):
    	pool.apply_async(queuepush, (wait_fetch_list[x],q,lock, inner_batch_num))
  
	logger.logger.info("launch %d process to consumer analye result!" % (paralnum/2))
	for index in range(paralnum/2):
		pool.apply_async(queueget, (q, lock))
	
	pool.close()
	pool.join()
