# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import tensorflow as tf

from Define import *

'''
pt = {
    p    , if y = 1
    1 − p, otherwise
}
FL(pt) = −(1 − pt)γ * log(pt)
'''
def Focal_Loss(pred_classes, gt_classes, alpha = 0.25, gamma = 2):
    with tf.variable_scope('Focal_Loss'):
        # positive_mask = [BATCH_SIZE, 22890]
        positive_mask = tf.reduce_max(gt_classes[:, :, 1:], axis = -1)
        positive_mask = tf.cast(tf.math.equal(positive_mask, 1.), dtype = tf.float32)

        ignored_mask = tf.cast(tf.math.equal(gt_classes[:, :, 0], -1), dtype = tf.float32) # -1 == 1
        ignored_mask = tf.expand_dims(1 - ignored_mask, axis = -1) # 1 -> 0
        
        # exception 1. log(-1) -> nan !
        gt_classes = ignored_mask * gt_classes
        
        # positive_count = [BATCH_SIZE]
        positive_count = tf.reduce_sum(positive_mask, axis = 1)
        positive_count = tf.clip_by_value(positive_count, 1, positive_count)

        # focal_loss = [BATCH_SIZE, 22890, CLASSES]
        pt = gt_classes * pred_classes + (1 - gt_classes) * (1 - pred_classes) 
        focal_loss = -alpha * tf.pow(1. - pt, gamma) * tf.log(pt + 1e-10)

        # focal_loss = [BATCH_SIZE]
        focal_loss = tf.reduce_sum(ignored_mask * tf.abs(focal_loss), axis = [1, 2])
        focal_loss = tf.reduce_mean(focal_loss / positive_count)

    return focal_loss

'''
x = pred_bboxes - gt_bboxes, if positive
smooth_l1_loss = {
    0.5 * (x ** 2), if |x| < 1
    |x| - 0.5     , otherwise
}
'''
def Smooth_L1_Loss(pred_bboxes, gt_bboxes, gt_classes):
    with tf.variable_scope('smooth_l1_loss'):
        # positive_mask = [BATCH_SIZE, 22890]
        positive_mask = tf.reduce_max(gt_classes[:, :, 1:], axis = -1, keepdims = True)
        positive_mask = tf.cast(tf.math.equal(positive_mask, 1.), dtype = tf.float32)
        
        # positive_count = [BATCH_SIZE]
        positive_count = tf.reduce_sum(positive_mask, axis = 1)
        positive_count = tf.clip_by_value(positive_count, 1, positive_count)

        # pos_pred_bboxes = [BATCH_SIZE, 22890, 4]
        # pos_gt_bboxes = [BATCH_SIZE, 22890, 4]
        pos_pred_bboxes = positive_mask * pred_bboxes
        pos_gt_bboxes = positive_mask * gt_bboxes

        # smooth_l1_loss = [BATCH_SIZE, 22890, 4]
        x = tf.abs(pos_pred_bboxes - pos_gt_bboxes)
        smooth_l1_loss = tf.where(tf.less(x, 1.0), 0.5 * tf.pow(x, 2), x - 0.5)

        # smooth_l1_loss = [BATCH_SIZE]
        smooth_l1_loss = tf.reduce_sum(smooth_l1_loss, axis = [1, 2])
    
    return tf.reduce_mean(smooth_l1_loss / positive_count)

def RetinaNet_Loss(pred_bboxes, pred_classes, gt_bboxes, gt_classes, alpha = 1.0):

    # calculate focal_loss & smooth l1 loss
    focal_loss_op = Focal_Loss(pred_classes, gt_classes)
    smooth_l1_loss_op = Smooth_L1_Loss(pred_bboxes, gt_bboxes, gt_classes)
    
    # summary
    loss_op = focal_loss_op + alpha * smooth_l1_loss_op
    
    return loss_op, focal_loss_op, smooth_l1_loss_op

if __name__ == '__main__':
    ## check loss shape
    pred_bboxes = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, 4])
    pred_classes = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, CLASSES])

    gt_bboxes = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, 4])
    gt_classes = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, CLASSES])

    loss_op, focal_loss_op, smooth_l1_loss_op = RetinaNet_Loss(pred_bboxes, pred_classes, gt_bboxes, gt_classes)
    print(loss_op, focal_loss_op, smooth_l1_loss_op)

    ## check ignored mask
    # import numpy as np
    # gt_classes = np.zeros((1, 5, 5))
    # gt_classes[:, :, 0] = -1
    # gt_classes[:, 0, 0] = 0
    # gt_classes[:, 1, 0] = 0

    # ignored_mask = tf.cast(tf.math.equal(gt_classes[:, :, 0], -1), dtype = tf.float32) # -1 == 1
    # ignored_mask = tf.expand_dims(1 - ignored_mask, axis = -1) # 1 -> 0
    
    # sess = tf.Session()
    # print(sess.run(ignored_mask))