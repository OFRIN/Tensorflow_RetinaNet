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

from RetinaNet import *
from RetinaNet_Loss import *
from RetinaNet_Utils import *

from mAP_Calculator import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. dataset
train_xml_paths = [ROOT_DIR + line.strip() for line in open('./dataset/train.txt', 'r').readlines()][:100]
valid_xml_paths = train_xml_paths
valid_xml_count = len(valid_xml_paths)

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

retina_utils = RetinaNet_Utils()
retina_dic, retina_sizes = RetinaNet(input_var, False)

retina_utils.generate_anchors(retina_sizes)

pred_bboxes_op = Decode_Layer(retina_dic['pred_bboxes'], retina_utils.anchors)
pred_classes_op = retina_dic['pred_classes']

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './model/RetinaNet_10000.ckpt')

for xml_path in valid_xml_paths:
    image_path, gt_bboxes, gt_classes = xml_read(xml_path)

    image = cv2.imread(image_path)
    h, w, c = image.shape

    tf_image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_CUBIC)
    total_pred_bboxes, total_pred_classes = sess.run([pred_bboxes_op, pred_classes_op], feed_dict = {input_var : [tf_image]})

    pred_bboxes, pred_classes = retina_utils.Decode(total_pred_bboxes[0], total_pred_classes[0], [w, h], detect_threshold = 0.5)

    precision, recall = Precision_Recall(gt_bboxes, gt_classes, pred_bboxes, pred_classes)
    print(precision, recall)

    for bbox in gt_bboxes:
        xmin, ymin, xmax, ymax = bbox.astype(np.int32)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    for bbox, class_index in zip(pred_bboxes, pred_classes):
        xmin, ymin, xmax, ymax = bbox[:4].astype(np.int32)
        conf = bbox[4]
        class_name = CLASS_NAMES[class_index]

        string = "{} : {:.2f}%".format(class_name, conf * 100)
        cv2.putText(image, string, (xmin, ymin - 10), 1, 1, (0, 255, 0))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow('show', image)
    cv2.waitKey(0)
    