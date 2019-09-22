# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import time
import random
import threading

import numpy as np

from Define import *
from Utils import *
from DataAugmentation import *

from RetinaNet_Utils import *

class Teacher(threading.Thread):
    ready = False
    min_data_size = 0
    max_data_size = 50

    total_indexs = []
    total_data_list = []
    
    batch_data_list = []
    batch_data_length = 0

    debug = False
    name = ''
    retina_utils = None
    
    def __init__(self, npy_path, retina_sizes, min_data_size = 1, max_data_size = 50, name = 'Thread', debug = False):
        self.name = name
        self.debug = debug

        self.retina_utils = RetinaNet_Utils()
        self.retina_utils.generate_anchors(retina_sizes)

        self.min_data_size = min_data_size
        self.max_data_size = max_data_size

        self.total_data_list = np.load(npy_path, allow_pickle = True)
        self.total_indexs = np.arange(len(self.total_data_list)).tolist()

        threading.Thread.__init__(self)
        
    def get_batch_data(self):
        batch_image_data, batch_gt_bboxes, batch_gt_classes = self.batch_data_list[0]
        
        del self.batch_data_list[0]
        self.batch_data_length -= 1

        if self.batch_data_length < self.min_data_size:
            self.ready = False
        
        return batch_image_data, batch_gt_bboxes, batch_gt_classes
    
    def run(self):
        while True:
            while self.batch_data_length >= self.max_data_size:
                continue
            
            batch_image_data = []
            batch_gt_bboxes = []
            batch_gt_classes = []
            batch_indexs = random.sample(self.total_indexs, BATCH_SIZE * 2)

            for data in self.total_data_list[batch_indexs]:
                # if self.debug:
                #     delay = time.time()
                
                image_path, gt_bboxes, gt_classes = data

                image = cv2.imread(image_path)
                image, gt_bboxes, gt_classes = DataAugmentation(image, gt_bboxes, gt_classes)

                image_h, image_w, image_c = image.shape
                image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

                gt_bboxes = gt_bboxes.astype(np.float32)
                gt_classes = np.asarray(gt_classes, dtype = np.int32)
                
                gt_bboxes /= [image_w, image_h, image_w, image_h]
                gt_bboxes *= [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]

                decode_bboxes, decode_classes = self.retina_utils.Encode(gt_bboxes, gt_classes)

                # if self.debug:
                #     delay = time.time() - delay
                #     print('[D] {} - {} = {}ms'.format(self.name, 'xml', int(delay * 1000)))

                batch_image_data.append(image.astype(np.float32))
                batch_gt_bboxes.append(decode_bboxes)
                batch_gt_classes.append(decode_classes)

                if len(batch_image_data) == BATCH_SIZE:
                    break
            
            batch_image_data = np.asarray(batch_image_data, dtype = np.float32) 
            batch_gt_bboxes = np.asarray(batch_gt_bboxes, dtype = np.float32)
            batch_gt_classes = np.asarray(batch_gt_classes, dtype = np.float32)
            
            self.batch_data_list.append([batch_image_data, batch_gt_bboxes, batch_gt_classes])
            self.batch_data_length += 1

            if self.debug:
                print('[D] stack = [{}/{}]'.format(self.batch_data_length, self.max_data_size))

            if self.batch_data_length >= self.min_data_size:
                self.ready = True
            else:
                self.ready = False