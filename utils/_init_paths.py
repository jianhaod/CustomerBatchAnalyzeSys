#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/3/12
@author: Jianhao
"""
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add thirdparty caffe to PYTHONPATH (caffe-fast-rcnn)
caffe_path = osp.join(this_dir, '..', 'thirdparty', 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add thirdpart lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'thirdparty', 'lib')
add_path(lib_path)
