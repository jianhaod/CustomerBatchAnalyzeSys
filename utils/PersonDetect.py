#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/3/8
@author: Jianhao
"""
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.test import vis_detections
from fast_rcnn.nms_wrapper import nms
import caffe
import yaml
import cv2
import numpy as np
import sys
import os

class PersonDetector(object):

    def __init__(self, configfile):
        # fast_rcnn detect config init
        caffeconfig = yaml.load(open(configfile))
        self.prototxt = caffeconfig['prototxt']
        self.caffemodel = caffeconfig['caffemodel']
        self.conf_thresh = caffeconfig['CONF_THRESH']
        self.nms_thresh = caffeconfig['NMS_THRESH']
        #self.classes = ('__background__', 'customer', 'sale')
        self.classes = tuple(caffeconfig['netclasses'])

        self.pre_output = caffeconfig['pre_output']
        
        # Use RPN for proposals
        cfg.TEST.HAS_RPN = True
        
        caffe.set_mode_gpu()
        caffe.set_device(0)
        
        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)

    def fast_rcnn_detect(self, input):
        '''
        param: input (imagedics list, input_path)
        '''
        image_dics = input[0]
        input_path = input[1]

        path_split = input_path.split('/')

        # output_dir = pre + shopname + data + camera + Append
        output_dir = self.pre_output + '/detOutput/' + path_split[1] + '/' + path_split[2] + '/' + 'Channel_01' + '/Append' 
        time_stamp = path_split[-1].split('.')[0]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        temp_result_file = os.path.join(output_dir, time_stamp +'.txt')
        file_obj = open(temp_result_file, 'a')
        for img_index in range(len(image_dics)):
			im = image_dics[img_index]

			# use caffe fast-rcnn net
			# get candidate object boxes, and each boxs score
			scores, boxes = im_detect(self.net, im)
			
			for cls_ind, cls in enumerate(self.classes[1:]):
				cls_ind += 1  # because we skipped background
				# get all raw, and cls_in:cls_ind 4 colum
				# raw 0 [ [x1,x2,y1,y2; x1,x2,y1,y2;....] ]
				# raw 1 [ [x1,x2,y1,y2; x1,x2,y1,y2;....] ]
				cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
				# get each box score
				cls_scores = scores[:, cls_ind]
				# merge score to boxes array
				dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
				# use NMS to keep overlap candidate box
				keep = nms(dets, self.nms_thresh)
				dets = dets[keep, :]
				# get high Confidence score candidate box
				inds = np.where(dets[:, -1] >= self.conf_thresh)[0]
				if len(inds) == 0:
					continue
				# save all mark box location to txt
				for i in inds:
					line = []
					name = "%07d" % int(img_index + 1)
					file_obj.write(name)
					file_obj.write(',')
					line.extend([-1])
					bbox = dets[i, :4]
					bbox = map(int, bbox)
					# change 4 colum to save name, x1, y1, high, weight
					bbox[2] = bbox[2] - bbox[0]
					bbox[3] = bbox[3] - bbox[1]
					line.extend(bbox)
					score = float(dets[i, -1])
					line.extend([score])
					line.extend([-1, -1])
					file_obj.write(",".join(repr(e) for e in line))
					file_obj.write("," + str(cls_ind))
					file_obj.write('\n')
        file_obj.close()

    def detectimage(self, picture):
        im = cv2.imread(picture)
        scores, boxes = im_detect(self.net, im)

        for cls_ind, cls in enumerate(self.classes[1:]):
		    cls_ind += 1  # because we skipped background
		    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
		    cls_scores = scores[:, cls_ind]
		    dets = np.hstack((cls_boxes,
		                      cls_scores[:, np.newaxis])).astype(np.float32)
		    keep = nms(dets, self.nms_thresh)
		    dets = dets[keep, :]
		    vis_detections(im, cls, dets, thresh=self.conf_thresh)
