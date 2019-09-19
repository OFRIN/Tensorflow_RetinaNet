# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

# dataset parameters
ROOT_DIR = 'D:/_ImageDataset/'

CLASS_NAMES = ['background'] + ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
CLASS_DIC = {class_name : index for index, class_name in enumerate(CLASS_NAMES)}
CLASSES = len(CLASS_NAMES)

# network parameters
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNEL = 3

PYRAMID_LEVELS = [3, 4, 5, 6, 7]
SCALE_FACTOR = [0.1, 0.1, 0.2, 0.2]

# ASPECT_RATIOS = [[1/2, 1], [1, 1], [1, 1/2]] # [1:2, 1:1, 2:1]
ASPECT_RATIOS = [0.5, 1.0, 2.0]
ANCHOR_SCALES = [2 ** 0, 2 ** (1/3), 2 ** (2/3)] # [1, 1.2599210498948732, 1.5874010519681994]
ANCHOR_SIZES = [2 ** (x + 2) for x in PYRAMID_LEVELS]

ANCHORS = len(ASPECT_RATIOS) * len(ANCHOR_SCALES)

AP_THRESHOLD = 0.5
NMS_THRESHOLD = 0.6

POSITIVE_IOU_THRESHOLD = 0.5
NEGATIVE_IOU_THRESHOLD = 0.4

# loss parameters
WEIGHT_DECAY = 0.0001

# train
NUM_GPU = 2
BATCH_SIZE = 8 * NUM_GPU
INIT_LEARNING_RATE = 1e-2

PAPER_BATCH_SIZE = 16
MAX_ITERATION = 90000
DECAY_ITERATIONS = [60000, 80000]

LOG_ITERATION = 50
VALID_ITERATION = 5000
