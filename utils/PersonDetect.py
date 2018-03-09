#!/usr/bin/env python
#-*- coding:utf8 -*-

"""
Create: 2018/3/8
@author: Jianhao
"""
import os, sys
sys.path.append('../thirdparty/caffe-fast-rcnn/python')
sys.path.append('../thirdparty/lib')

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import caffe
import yaml

class PersonDetector(object):

    def __init__(self, configfile):
	
        caffeconfig = yaml.load(open(configfile))
        self.prototxt = caffeconfig['prototxt']
        self.caffemodel = caffeconfig['caffemodel']
        self.conf_thresh = caffeconfig['CONF_THRESH']
        self.nms_thresh = caffeconfig['NMS_THRESH']
        self.classes = tuple(caffeconfig['netclasses'])
        
        # Use RPN for proposals
        cfg.TEST.HAS_RPN = True
        
        caffe.set_mode_gpu()
        caffe.set_device(0)
        
        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)

    def detect(self, input):
        image_dic = input[0]
        local_path = input[1]
        channel = input[2]
        
        local_path_b = local_path.split('/')
        
        b_num = len(local_path_b)
        
        
        output_file = config['per_det_output_file']+'/'+local_path_b[b_num - 5]+'/'+local_path_b[b_num - 4]+'/Channel_0'+str(channel)+'/Append'
        time_file=(local_path.split('/')[-1]).split('.')[0]
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        
        file_temp=os.path.join(output_file,time_file+'.txt')
        #print (file_temp)
        
        f = open(file_temp, 'a')
        for img_num in range(len(image_dic)):
            im = image_dic[img_num]
            #tmpTime = time.time()
            scores, boxes = im_detect(self.net, im)
            #print("imdetect Time: " + str(time.time() - tmpTime))
            for cls_ind, cls in enumerate(self.classes[1:]):
                cls_ind += 1  # because we skipped background
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, self.nms_thresh)
                dets = dets[keep, :]
                # vis_detections(im, cls, dets, thresh=CONF_THRESH)
                inds = np.where(dets[:, -1] >= self.conf_thresh)[0]
                if len(inds) == 0:
                    continue
                for i in inds:
                    result = []
                    name = "%07d" % int(img_num+1)
                    f.write(name)
                    f.write(',')
                    result.extend([-1])
                    bbox = dets[i, :4]
                    bbox = map(int, bbox)
                    bbox[2] = bbox[2] - bbox[0]
                    bbox[3] = bbox[3] - bbox[1]
                    result.extend(bbox)
                    score = float(dets[i, -1])
                    result.extend([score])
                    result.extend([-1, -1])
                    f.write(",".join(repr(e) for e in result))
                    f.write("," + str(cls_ind))
                    # print result
                    f.write('\n')
        f.close()
