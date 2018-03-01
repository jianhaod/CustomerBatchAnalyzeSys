#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/3/1
@author: Jianhao
"""

import yaml

yamlfile = open('../config/config.yaml')
sysdict = yaml.load(yamlfile)  

print type(sysdict)
print sysdict
