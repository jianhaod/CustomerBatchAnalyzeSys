#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/3/14
@author: Jianhao
"""

import time
import sys
import _init_utils_paths
from Logger import *
import redis

redis_kv_logger = Logger(logger='RedisLog')

redis_kv_logger.logger.info("Create redis connection pool")
pool = redis.ConnectionPool(host='10.255.131.77', port=6379, password='redisPassw0rd',
                            db='0', decode_responses=True)
redis_kv_logger.logger.info("Create redis obj use connect")
redisobj = redis.Redis(connection_pool=pool)

redis_kv_logger.logger.info("Start to push key value pair into redis")

# testInput video
testDir1 = '/data/jianhaod/testInput/SQ2128/20180310/Channel_03/Append/'
#testDir2 = '/data/ossfs/testInput/SQ2130/20180310/Channel_08/Append/'
#testDir3 = '/data/ossfs/testInput/SQ2132/20180310/Channel_07/Append/'

for root, dirs, files in os.walk(testDir1):
    for file in files:
        vdoName = 'testInput/SQ2128/20180310/Channel_03/Append/%s'%(file)
        redisobj.lpush('jianhao_test_fetch_list_1', vdoName)
        redis_kv_logger.logger.info("push (%s) pair into jianhao_test_fetch_list_1" % vdoName)
