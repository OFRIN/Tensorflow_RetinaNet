# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import numpy as np
import tensorflow as tf

import resnet_v1.resnet_v1 as resnet_v1

from Define import *

kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01, seed = None)
bias_initializer = tf.constant_initializer(value = 0.0)
class_bias_initializer = tf.constant_initializer(value = -np.log((1 - 0.01) / 0.01))

def group_normalization(x, is_training, G = 32, ESP = 1e-5, scope = 'group_norm'):
    with tf.variable_scope(scope):
        # 1. [N, H, W, C] -> [N, C, H, W]
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.shape.as_list()

        # 2. reshape (group normalization)
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        
        # 3. get mean, variance
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        # 4. normalize
        x = (x - mean) / tf.sqrt(var + ESP)

        # 5. create gamma, bete
        gamma = tf.Variable(tf.constant(1.0, shape = [C]), dtype = tf.float32, name = 'gamma')
        beta = tf.Variable(tf.constant(0.0, shape = [C]), dtype = tf.float32, name = 'beta')
        
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])

        # 6. gamma * x + beta
        x = tf.reshape(x, [-1, C, H, W]) * gamma + beta

        # 7. [N, C, H, W] -> [N, H, W, C]
        x = tf.transpose(x, [0, 2, 3, 1])
    return x

def conv_gn_relu(x, filters, kernel_size, strides, padding, is_training, scope, gn = True, activation = True, use_bias = True, upscaling = False):
    with tf.variable_scope(scope):
        if not upscaling:
            x = tf.layers.conv2d(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = kernel_initializer, use_bias = use_bias, name = 'conv2d')
        else:
            x = tf.layers.conv2d_transpose(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = kernel_initializer, use_bias = use_bias, name = 'upconv2d')
        
        if gn:
            x = group_normalization(x, is_training = is_training, scope = 'gn')
        
        if activation:
            x = tf.nn.relu(x, name = 'relu')
    return x

def connection_block(x1, x2, is_training, scope):
    with tf.variable_scope(scope):
        x1 = conv_gn_relu(x1, 256, [3, 3], 1, 'same', is_training, 'conv1', gn = True, activation = False)
        x2 = conv_gn_relu(x2, 256, [1, 1], 1, 'valid', is_training, 'conv2', gn = True, activation = False)
        x = tf.nn.relu(x1 + x2, name = 'relu')
    return x

def build_head_loc(x, is_training, name, depth = 4):
    with tf.variable_scope(name):
        for i in range(depth):
            x = conv_gn_relu(x, 256, (3, 3), 1, 'same', is_training, '{}'.format(i))
        
        x = conv_gn_relu(x, 4 * ANCHORS, (3, 3), 1, 'same', is_training, 'regression', gn = False, activation = False)
    return x

def build_head_cls(x, is_training, name, depth = 4):
    with tf.variable_scope(name):
        for i in range(depth):
            x = conv_gn_relu(x, 256, (3, 3), 1, 'same', is_training, '{}'.format(i))
        
        x = tf.layers.conv2d(inputs = x, filters = CLASSES * ANCHORS, kernel_size = [3, 3], strides = 1, padding = 'same', 
                             kernel_initializer = kernel_initializer, bias_initializer = class_bias_initializer, name = 'classification')
    return x

def Decode_Layer(offset_bboxes, anchors):
    # 1. offset bboxes
    tx = offset_bboxes[..., 0]
    ty = offset_bboxes[..., 1]
    tw = tf.clip_by_value(offset_bboxes[..., 2], -10, 5)
    th = tf.clip_by_value(offset_bboxes[..., 3], -10, 5)
    
    # 2. anchors
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    xa = anchors[:, 0] + wa / 2
    ya = anchors[:, 1] + ha / 2

    # 3. calculate decode bboxes (cxcywh)
    x = tx * wa + xa
    y = ty * ha + ya
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha

    # 5. pred_bboxes (cxcywh -> xyxy)
    xmin = tf.clip_by_value(x - w / 2, 0, IMAGE_WIDTH - 1)
    ymin = tf.clip_by_value(y - h / 2, 0, IMAGE_HEIGHT - 1)
    xmax = tf.clip_by_value(x + w / 2, 0, IMAGE_WIDTH - 1)
    ymax = tf.clip_by_value(y + h / 2, 0, IMAGE_HEIGHT - 1)

    pred_bboxes = tf.stack([xmin, ymin, xmax, ymax])
    pred_bboxes = tf.transpose(pred_bboxes, perm = [1, 2, 0])
    return pred_bboxes

def RetinaNet_ResNet_50(input_var, is_training, reuse = False):
    # convert BGR -> RGB
    x = input_var[..., ::-1] - MEAN
    
    with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, end_points = resnet_v1.resnet_v1_50(x, is_training = is_training, reuse = reuse)
    
    # for key in end_points.keys():
    #     print(key, end_points[key])
    # input()

    pyramid_dic = {}
    feature_maps = [end_points['resnet_v1_50/block{}'.format(i)] for i in [4, 2, 1]]
    
    pyramid_dic['C3'] = feature_maps[2]
    pyramid_dic['C4'] = feature_maps[1]
    pyramid_dic['C5'] = feature_maps[0]

    retina_dic = {}
    retina_sizes = []
    
    with tf.variable_scope('RetinaNet', reuse = reuse):
        x = conv_gn_relu(pyramid_dic['C5'], 256, (1, 1), 1, 'valid', is_training, 'P5_conv')
        pyramid_dic['P5'] = x
        
        x = conv_gn_relu(x, 256, (3, 3), 2, 'same', is_training, 'P6_conv')
        pyramid_dic['P6'] = x
        
        x = conv_gn_relu(x, 256, (3, 3), 2, 'same', is_training, 'P7_conv')
        pyramid_dic['P7'] = x

        x = conv_gn_relu(pyramid_dic['P5'], 256, (3, 3), 2, 'same', is_training, 'P4_conv_1', upscaling = True)
        x = connection_block(x, pyramid_dic['C4'], is_training, 'P4_conv')
        pyramid_dic['P4'] = x

        x = conv_gn_relu(pyramid_dic['P4'], 256, (3, 3), 2, 'same', is_training, 'P3_conv_1', upscaling = True)
        x = connection_block(x, pyramid_dic['C3'], is_training, 'P3_conv')
        pyramid_dic['P3'] = x
        
        '''
        # P3 : Tensor("RetinaNet/P3_conv/relu:0", shape=(8, 64, 64, 256), dtype=float32)
        # P4 : Tensor("RetinaNet/P4_conv/relu:0", shape=(8, 32, 32, 256), dtype=float32)
        # P5 : Tensor("RetinaNet/P5_conv/relu:0", shape=(8, 16, 16, 256), dtype=float32)
        # P6 : Tensor("RetinaNet/P6_conv/relu:0", shape=(8, 8, 8, 256), dtype=float32)
        # P7 : Tensor("RetinaNet/P7_conv/relu:0", shape=(8, 4, 4, 256), dtype=float32)
        '''
        # for i in PYRAMID_LEVELS:
        #    print('# P{} :'.format(i), pyramid_dic['P{}'.format(i)])
        # input()
        
        pred_bboxes = []
        pred_classes = []
        
        for i in PYRAMID_LEVELS:
            feature_map = pyramid_dic['P{}'.format(i)]
            _, h, w, c = feature_map.shape.as_list()
            
            _pred_bboxes = build_head_loc(feature_map, is_training, 'P{}_bboxes'.format(i))
            _pred_classes = build_head_cls(feature_map, is_training, 'P{}_classes'.format(i))

            # reshape bboxes, classes
            _pred_bboxes = tf.reshape(_pred_bboxes, [-1, h * w * ANCHORS, 4])
            _pred_classes = tf.reshape(_pred_classes, [-1, h * w * ANCHORS, CLASSES])
            
            # append sizes, bboxes, classes
            retina_sizes.append([w, h])
            pred_bboxes.append(_pred_bboxes)
            pred_classes.append(_pred_classes)

        # concatenate bboxes, classes (axis = 1)
        pred_bboxes = tf.concat(pred_bboxes, axis = 1, name = 'bboxes')
        pred_classes = tf.concat(pred_classes, axis = 1, name = 'classes')

        # update dictionary 
        retina_dic['pred_bboxes'] = pred_bboxes
        retina_dic['pred_classes'] = tf.nn.sigmoid(pred_classes)

    return retina_dic, retina_sizes

RetinaNet = RetinaNet_ResNet_50

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [8, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    
    retina_dic, retina_sizes = RetinaNet(input_var, False)
    
    print(retina_dic['pred_bboxes'])
    print(retina_dic['pred_classes'])
