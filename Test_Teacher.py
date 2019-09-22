# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import glob
import time
import random

import numpy as np
import tensorflow as tf

from Define import *
from Utils import *
from Teacher import *

from RetinaNet import *
from RetinaNet_Loss import *
from RetinaNet_Utils import *

from mAP_Calculator import *

input_var = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
is_training = tf.placeholder(tf.bool)

retina_dic, retina_sizes = RetinaNet(input_var, is_training)
retina_utils = RetinaNet_Utils()

train_threads = []
for i in range(2):
    train_thread = Teacher('./dataset/train.npy', retina_sizes, debug = True)
    train_thread.start()
    train_threads.append(train_thread)

while True:
    find = False
    while not find:
        for train_thread in train_threads:
            if train_thread.ready:
                find = True
                batch_image_data, batch_gt_bboxes, batch_gt_classes = train_thread.get_batch_data()        
                break
    
    print(batch_image_data.shape)
    print(batch_gt_bboxes.shape)
    print(batch_gt_classes.shape)
    
    image = batch_image_data[0].astype(np.uint8)
    decode_bboxes = batch_gt_bboxes[0]
    decode_classes = batch_gt_classes[0]

    pred_bboxes, pred_classes = retina_utils.Decode(decode_bboxes, decode_classes, [IMAGE_WIDTH, IMAGE_HEIGHT])

    for pred_bbox, pred_class in zip(pred_bboxes, pred_classes):
        xmin, ymin, xmax, ymax = pred_bbox[:4].astype(np.int32)
        conf = pred_bbox[4]

        cv2.putText(image, '{}'.format(pred_class), (xmin, ymin - 10), 1, 1, (0, 255, 0), 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

    cv2.imshow('show', image)
    cv2.waitKey(0)