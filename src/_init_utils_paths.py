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

# Add utils defined function to PYTHONPATH
utils_path = osp.join(this_dir, '..', 'utils')
add_path(utils_path)
