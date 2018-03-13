#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/3/13
@author: Jianhao
"""

import _init_src_paths
from fetch import *

img_dic = {}
img_dic = split_video_to_frame("../input/videostream/test_detect.mp4")
