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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. dataset
train_xml_paths = glob.glob(ROOT_DIR + 'train2017/xml/*.xml')
valid_xml_paths = glob.glob(ROOT_DIR + 'valid2017/xml/*.xml')
valid_xml_count = len(valid_xml_paths)

open('log.txt', 'w')
log_print('[i] Train : {}'.format(len(train_xml_paths)))
log_print('[i] Valid : {}'.format(len(valid_xml_paths)))

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
is_training = tf.placeholder(tf.bool)

retina_utils = RetinaNet_Utils()
retina_dic, retina_sizes = RetinaNet(input_var, is_training)

retina_utils.generate_anchors(retina_sizes)

pred_bboxes_op = Decode_Layer(retina_dic['pred_bboxes'], retina_utils.anchors)
pred_classes_op = retina_dic['pred_classes']

_, retina_size, _ = pred_bboxes_op.shape.as_list()

gt_bboxes_var = tf.placeholder(tf.float32, [None, retina_size, 4])
gt_classes_var = tf.placeholder(tf.float32, [None, retina_size, CLASSES])

log_print('[i] pred_bboxes_op : {}'.format(pred_bboxes_op))
log_print('[i] pred_classes_op : {}'.format(pred_classes_op))
log_print('[i] gt_bboxes_var : {}'.format(gt_bboxes_var))
log_print('[i] gt_classes_var : {}'.format(gt_classes_var))

loss_op, focal_loss_op, giou_loss_op = RetinaNet_Loss(pred_bboxes_op, pred_classes_op, gt_bboxes_var, gt_classes_var)

vars = tf.trainable_variables()
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * WEIGHT_DECAY
loss_op += l2_reg_loss_op

learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op)
    # train_op = tf.train.MomentumOptimizer(learning_rate_var, momentum = 0.9).minimize(loss_op)

train_summary_dic = {
    'Total_Loss' : loss_op,
    'Focal_Loss' : focal_loss_op,
    'GIoU_Loss' : giou_loss_op,
    'L2_Regularization_Loss' : l2_reg_loss_op,
    'Learning_rate' : learning_rate_var,
}

valid_precision_var = tf.placeholder(tf.float32)
valid_recall_var = tf.placeholder(tf.float32)
valid_mAP_var = tf.placeholder(tf.float32)

valid_summary_dic = {
    'Validation_Preicision' : valid_precision_var,
    'Validation_Recall' : valid_recall_var,
    'Validation_mAP' : valid_mAP_var,
}

train_summary_list = []
for name in train_summary_dic.keys():
    value = train_summary_dic[name]
    train_summary_list.append(tf.summary.scalar(name, value))
train_summary_op = tf.summary.merge(train_summary_list)

valid_summary_list = []
for name in valid_summary_dic.keys():
    value = valid_summary_dic[name]
    valid_summary_list.append(tf.summary.scalar(name, value))
valid_summary_op = tf.summary.merge(valid_summary_list)

# 3. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# '''
pretrained_vars = []
for var in vars:
    if 'resnet_v2_50' in var.name:
        pretrained_vars.append(var)

pretrained_saver = tf.train.Saver(var_list = pretrained_vars)
pretrained_saver.restore(sess, './resnet_v2_model/resnet_v2_50.ckpt')
# '''

saver = tf.train.Saver(max_to_keep = 30)
# saver.restore(sess, './model/RetinaNet_{}.ckpt'.format(30000))

MAX_ITERATION = 90000
DECAY_ITERATIONS = [60000, 80000]

best_valid_mAP = 0.0
learning_rate = INIT_LEARNING_RATE

train_iteration = len(train_xml_paths) // BATCH_SIZE
valid_iteration = len(valid_xml_paths) // BATCH_SIZE

max_iteration = int(MAX_ITERATION * PAPER_BATCH_SIZE / BATCH_SIZE)
decay_iteration = np.asarray(DECAY_ITERATIONS, dtype = np.float32) * PAPER_BATCH_SIZE / BATCH_SIZE
decay_iteration = decay_iteration.astype(np.int32)

log_print('[i] max_iteration : {}'.format(max_iteration))
log_print('[i] decay_iteration : {}'.format(decay_iteration))

loss_list = []
focal_loss_list = []
giou_loss_list = []
l2_reg_loss_list = []
train_time = time.time()

train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
valid_writer = tf.summary.FileWriter('./logs/valid', sess.graph)

train_threads = []
for i in range(NUM_THREADS):
    train_thread = Teacher('./dataset/train.npy', retina_sizes)
    train_thread.start()
    train_threads.append(train_thread)

for iter in range(1, max_iteration + 1):
    if iter in decay_iteration:
        learning_rate /= 10
        log_print('[i] learning rate decay : {} -> {}'.format(learning_rate * 10, learning_rate))

    # Thread
    find = False
    while not find:
        for train_thread in train_threads:
            if train_thread.ready:
                find = True
                batch_image_data, batch_gt_bboxes, batch_gt_classes = train_thread.get_batch_data()        
                break
    
    _feed_dict = {input_var : batch_image_data, gt_bboxes_var : batch_gt_bboxes, gt_classes_var : batch_gt_classes, is_training : True, learning_rate_var : learning_rate}
    log = sess.run([train_op, loss_op, focal_loss_op, giou_loss_op, l2_reg_loss_op, train_summary_op], feed_dict = _feed_dict)
    # print(log[1:-1])
    
    if np.isnan(log[1]):
        print('[!]', log[1:-1])
        input()

    loss_list.append(log[1])
    focal_loss_list.append(log[2])
    giou_loss_list.append(log[3])
    l2_reg_loss_list.append(log[4])
    train_writer.add_summary(log[5], iter)

    if iter % LOG_ITERATION == 0:
        loss = np.mean(loss_list)
        focal_loss = np.mean(focal_loss_list)
        giou_loss = np.mean(giou_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        train_time = int(time.time() - train_time)
        
        log_print('[i] iter : {}, loss : {:.4f}, focal_loss : {:.4f}, giou_loss : {:.4f}, l2_reg_loss : {:.4f}, train_time : {}sec'.format(iter, loss, focal_loss, giou_loss, l2_reg_loss, train_time))

        loss_list = []
        focal_loss_list = []
        giou_loss_list = []
        l2_reg_loss_list = []
        train_time = time.time()

    if iter % VALID_ITERATION == 0:
        mAP_calc = mAP_Calculator(classes = CLASSES)

        valid_time = time.time()

        batch_image_data = []
        batch_image_wh = []
        batch_gt_bboxes = []
        batch_gt_classes = []

        for valid_iter, xml_path in enumerate(valid_xml_paths):
            image_path, gt_bboxes, gt_classes = xml_read(xml_path, CLASS_NAMES)

            ori_image = cv2.imread(image_path)
            image = cv2.resize(ori_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
            
            batch_image_data.append(image.astype(np.float32))
            batch_image_wh.append(ori_image.shape[:-1][::-1])
            batch_gt_bboxes.append(gt_bboxes)
            batch_gt_classes.append(gt_classes)

            # calculate correct/confidence
            if len(batch_image_data) == BATCH_SIZE:
                total_pred_bboxes, total_pred_classes = sess.run([pred_bboxes_op, pred_classes_op], feed_dict = {input_var : batch_image_data, is_training : False})

                for i in range(BATCH_SIZE):
                    gt_bboxes, gt_classes = batch_gt_bboxes[i], batch_gt_classes[i]
                    pred_bboxes, pred_classes = retina_utils.Decode(total_pred_bboxes[i], total_pred_classes[i], batch_image_wh[i], detect_threshold = 0.5)

                    if pred_bboxes.shape[0] == 0:
                        pred_bboxes = np.zeros((0, 5), dtype = np.float32)
                    
                    mAP_calc.update(pred_bboxes, pred_classes, gt_bboxes, gt_classes)
                
                batch_image_data = []
                batch_image_wh = []
                batch_gt_bboxes = []
                batch_gt_classes = []

            sys.stdout.write('\r[i] validation = [{}/{}]'.format(valid_iter, valid_xml_count))
            sys.stdout.flush()

        valid_time = int(time.time() - valid_time)
        print('\n[i] validation time = {}sec'.format(valid_time))

        precision, recall, valid_mAP = mAP_calc.summary()

        valid_log = sess.run(valid_summary_op, feed_dict = {valid_precision_var : precision, valid_recall_var : recall, valid_mAP_var : valid_mAP})
        valid_writer.add_summary(valid_log, iter)

        if best_valid_mAP < valid_mAP and precision >= MIN_PRECISION:
            best_valid_mAP = valid_mAP
            log_print('[i] valid precision : {:.2f}%'.format(precision))
            log_print('[i] valid recall : {:.2f}%'.format(recall))

            saver.save(sess, './model/RetinaNet_{}.ckpt'.format(iter))
            
        log_print('[i] valid mAP : {:.2f}%, best valid mAP : {:.2f}%'.format(valid_mAP, best_valid_mAP))

saver.save(sess, './model/RetinaNet.ckpt')
