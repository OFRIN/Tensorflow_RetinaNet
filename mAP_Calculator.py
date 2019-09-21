# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import math
import numpy as np

from Utils import *

class mAP_Calculator:
    def __init__(self, classes, ap_threshold = 0.5):
        self.ap_threshold = ap_threshold
        self.precision_list = []
        self.recall_list = []

    def get_precision_and_recall(self, pred_bboxes, pred_classes, gt_bboxes, gt_classes):
        if len(gt_bboxes) == 0:
            if len(pred_bboxes) == 0:
                return 1.0, 1.0
            else:
                return 0.0, 0.0

        if len(pred_bboxes) == 0:
            return 0.0, 0.0

        pred_count, gt_count = len(pred_bboxes), len(gt_bboxes)
        
        iou_mask = compute_bboxes_IoU(pred_bboxes, gt_bboxes) >= self.ap_threshold
        class_mask = np.zeros((pred_count, gt_count), dtype = np.bool)

        for i in range(pred_count):
            class_mask[i] = pred_classes[i] == gt_classes

        mask = np.logical_and(iou_mask, class_mask)
        precision = np.mean(np.max(mask, axis = 1))
        recall = np.mean(np.max(mask.T, axis = 1))
        
        return precision, recall
    
    def update(self, pred_bboxes, pred_classes, gt_bboxes, gt_classes):
        precision, recall = self.get_precision_and_recall(pred_bboxes, pred_classes, gt_bboxes, gt_classes)

        self.precision_list.append(precision)
        self.recall_list.append(recall)

    def summary(self):
        precision = np.mean(self.precision_list) * 100
        recall = np.mean(self.recall_list) * 100
        mAP = (precision + recall) / 2

        return precision, recall, mAP

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    mAP_calc = mAP_Calculator(classes = 4)
    
    # case 1
    pred_bboxes = np.array([[0.880688, 0.44609185, 0.95696718, 0.6476958, 0.95],
                     [0.84020283, 0.45787981, 0.99351478, 0.64294884, 0.75],
                     [0.78723741, 0.61799151, 0.9083041, 0.75623035, 0.4],
                     [0.22078986, 0.30151826, 0.36679274, 0.40551913, 0.3],
                     [0.0041579, 0.48359361, 0.06867643, 0.60145104, 1.0],
                     [0.4731401, 0.33888632, 0.75164948, 0.80546954, 1.0],
                     [0.75489414, 0.75228018, 0.87922037, 0.88110524, 0.75],
                     [0.21953127, 0.77934921, 0.34853417, 0.90626764, 0.5],
                     [0.81, 0.11, 0.91, 0.21, 0.5]])
    pred_classes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3], dtype = np.int32)

    gt_bboxes = np.array([[0.86132812, 0.48242188, 0.97460938, 0.6171875],
                    [0.18554688, 0.234375, 0.36132812, 0.41601562],
                    [0., 0.47265625, 0.0703125, 0.62109375],
                    [0.47070312, 0.3125, 0.77929688, 0.78125],
                    [0.8, 0.1, 0.9, 0.2]])
    gt_classes = np.array([0, 0, 1, 2, 2], dtype = np.int32)

    pred_bboxes[:, :4] = pred_bboxes[:, :4] * [100, 100, 100, 100]
    gt_bboxes = gt_bboxes * [100, 100, 100, 100]

    mAP_calc.update(pred_bboxes, pred_classes, gt_bboxes, gt_classes)

    precision, recall, mAP = mAP_calc.summary()
    print(precision, recall, mAP)
    