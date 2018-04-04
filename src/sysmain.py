#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/3/1
@author: Jianhao
"""

import os, sys
import yaml
import redis
import datetime

import fetch
from multiprocessing.dummy import Pool
from multiprocessing import Process, Queue, Lock

import _init_utils_paths
from Logger import *
from PersonDetect import *

pushlog = Logger(logger='QueuePush')
getlog = Logger(logger='QueueGet')

def queuepush(itemkey, q, lock, local_video_path):
    pushlog.logger.info("start push logger")
    while True:
        tmpreslut={}
        video_detect_dict = {}
        try:
            itemkey, video_path = conn.brpop(itemkey, timeout = 10)
        except Exception, e:
            pushlog.logger.error("Catch redis connection pop error key:%s" % itemkey)
            return None 
        pushlog.logger.info("Get <k,v> from redis <%s, %s>" % (itemkey, video_path)) 
        video_detect_dict = fetch.split_video_to_frame(local_video_path + video_path)
        tmpreslut[0] = video_detect_dict
        tmpreslut[1] = video_path
        q.put(tmpreslut)

def queueget(q, lock):
    detectobj = PersonDetector("../config/persondetect.yaml")
    while True:
        getlog.logger.info("loop in the queue")
        if not q.empty():
            lock.acquire()
            temp_input = q.get()
            lock.release()
            print temp_input[1]
            detectobj.fast_rcnn_detect(temp_input)
        else:
            time.sleep(10)
            getlog.logger.info("queue is empty")
            if finished_check():
                return
            else:
                pass

def finished_check():

    now_hour_time = datetime.datetime.now().hour
    run_hour_time = 0

    if now_hour_time < start_hour_time:
        run_hour_time = (24 - start_hour_time) + now_hour_time
    else:
        run_hour_time = now_hour_time - start_hour_time

    if (now_hour_time > 18 or run_hour_time > 10):
        return True
    else:
        return False


if __name__ == '__main__':

    logger = Logger(logger='SysMainLog')
    logger.logger.info("Start lanuch CustomerBatch detect system!")
    
    yamlfile = open('../config/config.yaml')
    sysdict = yaml.load(yamlfile)

    # fetch video stream frame
    input_path = sysdict['video_stream_input_path']
    output_path = sysdict['video_stream_output_path']
    input_local_path = sysdict['input_local_path']
    #inner_batch_num = sysdict['inner_batch_num']
    wait_fetch_list = sysdict['wait_fetch_list']
    local_pre_path = sysdict['local_pre_path']

    #detectobj = PersonDetector("../config/persondetect.yaml")

    # create redis connection
    logger.logger.info("Create redis connection to fetch input video list")
    conn = redis.Redis(host=sysdict['redis']['host'], 
                        password=sysdict['redis']['password'], db=sysdict['redis']['db'])

    q = Queue()
    lock = Lock()

    start_hour_time = datetime.datetime.now().hour

    finished_check()
    #create processes
    processed = []
    for x in range(len(wait_fetch_list)):
        p1 = Process(target=queuepush,args=(wait_fetch_list[x],q,lock,local_pre_path))
        processed.append(p1)

    p2 = Process(target=queueget,args=(q, lock))
    processed.append(p2)

    for i in range(len(processed)):
        processed[i].start()
        #processed[i].join()

    while True:
        time.sleep(10)
        for j in range(len(processed) - 1):
            if processed[j].is_alive():
                pass
            else:
                processed[j].terminate()
                ptmp = Process(target=queuepush,args=(wait_fetch_list[j],q,lock,local_pre_path))
                time.sleep(1)
                processed[j] = ptmp
                processed[j].start()
        if finished_check():
            break

    for i in range(len(processed)):
        processed[i].join()

    logger.logger.info("End!")
