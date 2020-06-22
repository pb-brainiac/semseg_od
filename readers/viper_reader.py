import os
from os.path import join

import numpy as np

from readers.segmentation_reader import SegmentationReader
import readers.viper_mapping as viper_mapping

import pdb

class DatasetReader(SegmentationReader):
    class_info = viper_mapping.get_class_info()
    class_info.extend([[255, 255, 255, 'ood'], [0, 0, 0,  'unknown']])
    name = 'cityscapes'
    ood_id = num_classes = 23
    ignore_id = num_classes + 1

    mapping = np.empty(len(viper_mapping.labels))
    mapping.fill(ignore_id)
    for i, city_id in enumerate(viper_mapping.get_train_ids()):
        mapping[city_id] = i

    def __init__(self, args, subset='train'):
        SegmentationReader.__init__(self, args)
        data_dir = join(args.data_path, 'viper', subset, 'img')

        print(data_dir)
        cities = next(os.walk(data_dir))[1]

        self.img_paths = []
        self.label_paths = {}
        self.names = []

        for city in cities:
            files_path = join(data_dir, city)
            files = next(os.walk(files_path))[2]
            self.img_paths.extend([join(files_path, f) for f in files])
            self.names.extend([f[:-4] for f in files])

        self.label_paths.update({f: f.replace('img','cls').replace('jpg','png') for f in self.img_paths
                                    if os.path.exists(f.replace('img','cls').replace('jpg','png'))})
        print('\nTotal num images =', len(self.img_paths))
