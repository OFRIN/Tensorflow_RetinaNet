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
from DataAugmentation import *

from RetinaNet import *
from RetinaNet_Loss import *
from RetinaNet_Utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INFO

# 1. dataset
valid_data_list = np.load('./dataset/train_detection.npy', allow_pickle = True)
valid_count = len(valid_data_list)

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

retina_dic, retina_sizes = RetinaNet(input_var, False)

retina_utils = RetinaNet_Utils()
retina_utils.generate_anchors(retina_sizes)

pred_bboxes_op = Decode_Layer(retina_dic['pred_bboxes'], retina_utils.anchors)
pred_classes_op = retina_dic['pred_classes']

# 3. test
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './model/RetinaNet_{}.ckpt'.format(200000))

batch_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.float32)

for i in range(len(valid_data_list) // BATCH_SIZE):
    total_gt_bboxes = []
    batch_data_list = valid_data_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

    for i, data in enumerate(batch_data_list):
        image_name, gt_bboxes, gt_classes = data        
        image_path = TRAIN_DIR + image_name

        gt_bboxes = np.asarray(gt_bboxes, dtype = np.float32)
        gt_classes = np.asarray([CLASS_DIC[c] for c in gt_classes], dtype = np.int32)

        image = cv2.imread(image_path)
        image_h, image_w, c = image.shape

        image, gt_bboxes = random_horizontal_flip(image, gt_bboxes, 1.0)

        gt_bboxes /= [image_w, image_h, image_w, image_h]
        gt_bboxes *= [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]

        tf_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

        batch_image_data[i] = tf_image.copy()
        total_gt_bboxes.append(gt_bboxes)

    total_pred_bboxes, total_pred_classes = sess.run([pred_bboxes_op, pred_classes_op], feed_dict = {input_var : batch_image_data})

    for i in range(BATCH_SIZE):
        image = batch_image_data[i]
        pred_bboxes, pred_classes = retina_utils.Decode(total_pred_bboxes[i], total_pred_classes[i], [IMAGE_WIDTH, IMAGE_HEIGHT], detect_threshold = 0.50)
        
        for bbox, class_index in zip(pred_bboxes, pred_classes):
            xmin, ymin, xmax, ymax = bbox[:4].astype(np.int32)
            conf = bbox[4]
            class_name = CLASS_NAMES[class_index]

            string = "{} : {:.2f}%".format(class_name, conf * 100)
            cv2.putText(image, string, (xmin, ymin - 10), 1, 1, (0, 255, 0))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        for gt_bbox in total_gt_bboxes[i]:
            xmin, ymin, xmax, ymax = gt_bbox.astype(np.int32)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        cv2.imshow('show', image.astype(np.uint8))
        cv2.waitKey(0)
