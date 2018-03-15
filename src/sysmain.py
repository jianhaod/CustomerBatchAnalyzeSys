#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/3/1
@author: Jianhao
"""

import os, sys
import yaml
import redis

import fetch
from multiprocessing.dummy import Pool
from multiprocessing import Process, Queue, Lock

import _init_utils_paths
from Logger import *

pushlog = Logger(logger='QueuePush')
getlog = Logger(logger='QueueGet')

def queuepush(itemkey, q, lock, inner_batch_num, local_video_path):
        try:
                while True:
                #for i in range(inner_batch_num):
                        pushlog.logger.info("start push logger")
                        tmpreslut={}
                        video_detect_dict = {}
                        try:
                                #item_key, video_path = conn.brpop(itemkey, timeout = 60)
                                1/0
                        except KeyboardInterrupt, e:
                                print(e)
                                pushlog.logger.error("Catch KeyboardInterrupt")
                        except Exception, e:
                                pushlog.logger.error("Catch redis connection pop error key:%s" % item_key)
                                break
                        pushlog.logger.info("Get <k,v> from redis <%s, %s>" % (item_key, video_path)) 
                        video_detect_dict = fetch.split_video_to_frame(local_video_path + video_path)
                        tmpreslut[0] = video_detect_dict
                        tmpreslut[1] = video_path
                        q.put(tmpreslut)
        except:
                pushlog.logger.error("Catch redis connection pop error key:%s" % item_key)

def queueget(q, lock):
    while True:
        if not q.empty():
            lock.acquire()
            temp_input = q.get()
            lock.release()
            time.sleep(random.random())
            print temp_input
        else:
            break

if __name__ == '__main__':

	logger = Logger(logger='SysMainLog')
	logger.logger.info("Start lanuch CustomerBatch detect system!")

	yamlfile = open('../config/config.yaml')
	sysdict = yaml.load(yamlfile)

	# fetch video stream frame
	input_path = sysdict['video_stream_input_path']
	output_path = sysdict['video_stream_output_path']
	input_local_path = sysdict['input_local_path']
	inner_batch_num = sysdict['inner_batch_num']
	wait_fetch_list = sysdict['wait_fetch_list']
        local_pre_path = sysdict['local_pre_path']

	# create redis connection
        logger.logger.info("Create redis connection to fetch input video list")
        conn = redis.Redis(host=sysdict['redis']['host'], 
                           password=sysdict['redis']['password'], db=sysdict['redis']['db'])

	paralnum = len(wait_fetch_list)
	pool = Pool(paralnum)
	
	q = Queue()
	lock = Lock()
	
	# connect redis and fetch input video check
	logger.logger.info("connect redis and fetch input video check")
	# only need to one process inner loop fetch 1 video
        for x in range(len(wait_fetch_list)):
	       pool.apply_async(queuepush, (wait_fetch_list[x],q,lock, 1, local_pre_path))
	
	pool.close()
	pool.join()

	logger.logger.info("fetch check success")

        if ((paralnum/2) < 1):
                consumer_num = 1
        else:
                consumer_num = paralnum/2

        processpool = Pool(paralnum)
	consumerpool = Pool(consumer_num)

	logger.logger.info("launch %d process to fetch analye input video!" % paralnum)
	for x in range(len(wait_fetch_list)):
		processpool.apply_async(queuepush, (wait_fetch_list[x],q,lock, inner_batch_num, local_pre_path))

        # launch half of produce process num for consumer
	logger.logger.info("launch %d process to consumer analye result!" % consumer_num)
	for index in range(consumer_num):
		consumerpool.apply_async(queueget, (q, lock))
	
	processpool.close()
	consumerpool.close()

	processpool.join()
	consumerpool.join()
	logger.logger.info("End!")
