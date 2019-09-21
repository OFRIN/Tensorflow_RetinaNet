# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import numpy as np

from Define import *
from Utils import *
from DataAugmentation import *

def get_data(xml_path, training, normalize = True, augment = True):
    if training:
        image_path, gt_bboxes, gt_classes = xml_read(xml_path, normalize = False)

        image = cv2.imread(image_path)
        
        if augment:
            image, gt_bboxes, gt_classes = DataAugmentation(image, gt_bboxes, gt_classes)

        image_h, image_w, image_c = image.shape
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

        gt_bboxes = gt_bboxes.astype(np.float32)
        gt_classes = np.asarray(gt_classes, dtype = np.int32)

        if normalize:
            gt_bboxes /= [image_w, image_h, image_w, image_h]
            gt_bboxes *= [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]
    else:
        image_path, gt_bboxes, gt_classes = xml_read(xml_path, normalize = normalize)
        image = cv2.imread(image_path)

    return image, gt_bboxes, gt_classes

class RetinaNet_Utils:
    def __init__(self, anchor_scales = ANCHOR_SCALES, 
                       anchor_sizes = ANCHOR_SIZES,
                       aspect_ratios = ASPECT_RATIOS,
                       image_wh = [IMAGE_WIDTH, IMAGE_HEIGHT],
                       scale_factor = SCALE_FACTOR):
        self.anchor_scales = anchor_scales
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios

        self.image_wh = np.asarray(image_wh, dtype = np.float32)
        self.scale_factor = scale_factor

    def generate_anchors(self, sizes):
        anchors = []
        sizes = np.asarray(sizes, dtype = np.int32)

        for i in range(len(sizes)):
            size = sizes[i]
            strides = self.image_wh / size

            base_anchor_wh = np.asarray([self.anchor_sizes[i], self.anchor_sizes[i]])
            base_anchor_wh_list = [base_anchor_wh * scale for scale in self.anchor_scales]
            # print(i, base_anchor_wh_list)

            anchor_wh_list = []
            for base_anchor_wh in base_anchor_wh_list:
                # for aspect_ratio in self.aspect_ratios:
                for aspect_ratio in self.aspect_ratios:
                    w = base_anchor_wh[0] * np.sqrt(aspect_ratio)
                    h = base_anchor_wh[1] / np.sqrt(aspect_ratio)

                    # print(base_anchor_wh, aspect_ratio, w, h)
                    anchor_wh_list.append([w, h])

            for y in range(size[1]):
                for x in range(size[0]):
                    anchor_cx = (x + 0.5) * strides[0]
                    anchor_cy = (y + 0.5) * strides[1]

                    for anchor_wh in anchor_wh_list:
                        # print([anchor_cx, anchor_cy] + anchor_wh)
                        anchors.append([anchor_cx, anchor_cy] + anchor_wh)
        
        # normalize
        anchors = np.asarray(anchors, dtype = np.float32)

        cx, cy, w, h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        xmin, ymin, xmax, ymax = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

        xmin = np.maximum(np.minimum(xmin, self.image_wh[0] - 1), 0.)
        ymin = np.maximum(np.minimum(ymin, self.image_wh[1] - 1), 0.)
        xmax = np.maximum(np.minimum(xmax, self.image_wh[0] - 1), 0.)
        ymax = np.maximum(np.minimum(ymax, self.image_wh[1] - 1), 0.)

        self.anchors = np.stack([xmin, ymin, xmax, ymax]).T

    def Encode(self, gt_bboxes, gt_classes):
        # default
        decode_bboxes = np.zeros_like(self.anchors)
        decode_classes = np.zeros((self.anchors.shape[0], CLASSES), dtype = np.float32)

        # exception case 1.
        if len(gt_bboxes) == 0:
            return decode_bboxes, decode_classes

        # default ignored (= -1)
        decode_classes[:, 0] = -1

        # calculate anchors & gt_bboxes
        ious = compute_bboxes_IoU(self.anchors, gt_bboxes)

        # get positive & negative indexs
        max_iou_indexs = np.argmax(ious, axis = 1)
        max_ious = ious[np.arange(self.anchors.shape[0]), max_iou_indexs]

        # ignored overlap in [0.4, 0.5) (0.5 > ious >= 0.4)
        positive_indexs = max_ious >= POSITIVE_IOU_THRESHOLD
        negative_indexs = max_ious < NEGATIVE_IOU_THRESHOLD

        # update positive & negative classes
        positive_classes = gt_classes[max_iou_indexs][positive_indexs]
        decode_classes[positive_indexs, positive_classes] = 1.
        decode_classes[positive_indexs, 0] = 0.
        decode_classes[negative_indexs, 0] = 1.
        
        # update positive bboxes
        decode_bboxes = gt_bboxes[max_iou_indexs]

        return decode_bboxes, decode_classes
    
    def Decode(self, decode_bboxes, decode_classes, image_wh, detect_threshold = 0.01, use_nms = True):
        total_class_probs = np.max(decode_classes[:, 1:], axis = -1)
        total_class_indexs = np.argmax(decode_classes[:, 1:], axis = -1)

        cond = total_class_probs >= detect_threshold
        
        pred_bboxes = decode_bboxes[cond]
        class_probs = total_class_probs[cond][..., np.newaxis]
        
        pred_bboxes = np.concatenate([pred_bboxes, class_probs], axis = 1)
        pred_classes = total_class_indexs[cond] + 1

        pred_bboxes[:, :4] = convert_bboxes(pred_bboxes[:, :4], image_wh = image_wh)

        if use_nms:
            pred_bboxes, pred_classes = class_nms(pred_bboxes, pred_classes)

        return pred_bboxes, pred_classes

if __name__ == '__main__':
    from RetinaNet import *

    input_var = tf.placeholder(tf.float32, [8, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    
    retina_utils = RetinaNet_Utils()
    retina_dic, retina_sizes = RetinaNet(input_var, False)
    
    print(retina_dic['pred_bboxes'])
    print(retina_dic['pred_classes'])

    retina_utils.generate_anchors(retina_sizes)

    # 1. Demo Anchors
    # bg = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.uint8)
    
    # for index, anchor in enumerate(retina_utils.anchors):
    #     xmin, ymin, xmax, ymax = anchor.astype(np.int32)
        
    #     cv2.circle(bg, ((xmax + xmin) // 2, (ymax + ymin) // 2), 1, (0, 0, 255), 2)
    #     cv2.rectangle(bg, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    #     if (index + 1) % (len(ASPECT_RATIOS) * len(ANCHOR_SCALES)) == 0:
    #         cv2.imshow('show', bg)
    #         cv2.waitKey(1)

    #         bg = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.uint8)

    # cv2.waitKey(0)

    # 2. Demo GT bboxes (Encode -> Decode)
    xml_paths = glob.glob('D:/DB/VOC2007/train/xml/*.xml')
    
    for xml_path in xml_paths:
        # image_path, gt_bboxes, gt_classes = xml_read(xml_path, normalize = True)
        # print(gt_bboxes, np.min(gt_bboxes), np.max(gt_bboxes), len(gt_bboxes))
        
        # image = cv2.imread(image_path)
        # h, w, c = image.shape
        
        image, gt_bboxes, gt_classes = get_data(xml_path, True, normalize=True, augment = True)
        h, w, c = image.shape
        
        decode_bboxes, decode_classes = retina_utils.Encode(gt_bboxes, gt_classes)
        positive_count = np.sum(decode_classes[:, 1:])
        
        positive_mask = np.max(decode_classes[:, 1:], axis = 1)
        positive_mask = positive_mask[:, np.newaxis]
        print(np.min(positive_mask), np.max(positive_mask))

        print('# ignored :', np.sum(np.sum(decode_classes[:, 0] == -1)))
        
        # (22890, 4) (22890, 21)
        print(np.min(positive_mask * decode_bboxes[:, :2]), np.max(positive_mask * decode_bboxes[:, :2]), np.min(positive_mask * decode_bboxes[:, 2:]), np.max(positive_mask * decode_bboxes[:, 2:]))
        print(decode_bboxes.shape, decode_classes.shape, positive_count)

        pred_bboxes, pred_classes = retina_utils.Decode(decode_bboxes, decode_classes, [w, h])

        positive_mask = np.max(decode_classes[:, 1:], axis = 1)
        for i, mask in enumerate(positive_mask):
            if mask == 1:
                xmin, ymin, xmax, ymax = (retina_utils.anchors[i] / [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT] * [w, h, w, h]).astype(np.int32)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        
        for pred_bbox, pred_class in zip(pred_bboxes, pred_classes):
            xmin, ymin, xmax, ymax = pred_bbox[:4].astype(np.int32)
            conf = pred_bbox[4]

            cv2.putText(image, '{}'.format(pred_class), (xmin, ymin - 10), 1, 1, (0, 255, 0), 1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

        cv2.imshow('show', image)
        cv2.waitKey(0)
