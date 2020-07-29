import os
from os.path import join

import numpy as np

from readers.segmentation_reader import SegmentationReader
import readers.city_mapping as city_mapping

import pdb

class DatasetReader(SegmentationReader):
    class_info = city_mapping.get_class_info()
    class_info.extend([[255, 255, 255, 'ood'], [0, 0, 0,  'unknown']])
    name = 'cityscapes'
    ood_id = num_classes = 19
    ignore_id = num_classes + 1

    mapping = np.empty(len(city_mapping.labels))
    mapping.fill(ignore_id)
    for i, city_id in enumerate(city_mapping.get_train_ids()):
        mapping[city_id] = i

    def __init__(self, args, train=False):
        SegmentationReader.__init__(self, args, train)

        data_dir = args.data_path + '/wd_val_01/'
        files = next(os.walk(data_dir))[2]
        self.img_paths = [join(data_dir, f) for f in files if '_100000.png' in f]
        self.label_paths = {f: f[:-4] + '_labelIds.png' for f in self.img_paths
                            if os.path.exists(f[:-4] + '_labelIds.png')}
        self.names = [f[:-4] for f in files if '_100000.png' in f]

        print('\nTotal num images =', len(self.img_paths))

