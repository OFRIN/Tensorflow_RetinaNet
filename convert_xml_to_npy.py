import sys
import glob
import threading

import numpy as np

from Define import *
from Utils import *

class Reader(threading.Thread):
    self.name = None
    self.xml_paths = []
    self.data_list = []

    def __init__(self, xml_paths, name):
        self.name = name
        self.xml_paths = xml_paths
        threading.Thread.__init__(self)

    def run(self):
        length = len(xml_paths)
        for i, xml_path in enumerate(xml_paths):
            image_path, gt_bboxes, gt_classes = xml_read(xml_path)

            data_list.append([image_path, gt_bboxes, gt_classes])
            
            if i % 10 == 0:
                sys.stdout.write('\r{} - [{}/{}]'.format(self.name, i, length))
                sys.stdout.flush()

num_thread = 8 # same cpu core
xml_paths = glob.glob(ROOT_DIR + 'train2017/xml/*.xml')

size = len(xml_paths) // num_thread
threads = []

for i in range(num_thread):
    thread = Reader(xml_paths[i * size : (i + 1) * size], '{}'.format(i))
    thread.start()
    threads.append(thread)

if len(xml_paths) % num_thread != 0:
    thread = Reader(xml_paths[num_thread * size:], '{}'.format('last'))
    thread.start()
    threads.append(thread)

data_list = []
for thread in threads:
    thread.join()
    data_list += thread.data_list

data_list = np.asarray(data_list)
np.save('./dataset/train.npy', data_list)

# train_xml_paths = glob.glob(ROOT_DIR + 'train2017/xml/*.xml')
# length = len(train_xml_paths)

# data_list = []

# for i, xml_path in enumerate(train_xml_paths):
#     image_path, gt_bboxes, gt_classes = xml_read(xml_path)

#     data = [image_path, gt_bboxes, gt_classes]
#     data_list.append(data)

#     sys.stdout.write('\r[{}/{}]'.format(i, length))
#     sys.stdout.flush()

# data_list = np.asarray(data_list)
# np.save('./dataset/train.npy', data_list)
