import os
from os.path import join
from operator import itemgetter
import pickle

import torch
from torch.utils.data import Dataset
import PIL.Image as pimg
import numpy as np

import readers.transform as transform
import readers.mapping as city_mapping

import pdb

class DatasetReader(Dataset):
    class_info = [[128, 64, 128,   'road'],
                  [244, 35, 232,   'sidewalk'],
                  [70, 70, 70,     'building'],
                  [102, 102, 156,  'wall'],
                  [190, 153, 153,  'fence'],
                  [153, 153, 153,  'pole'],
                  [250, 170, 30,   'traffic light'],
                  [220, 220, 0,    'traffic sign'],
                  [107, 142, 35,   'vegetation'],
                  [152, 251, 152,  'terrain'],
                  [70, 130, 180,   'sky'],
                  [220, 20, 60,    'person'],
                  [255, 0, 0,      'rider'],
                  [0, 0, 142,      'car'],
                  [0, 0, 70,       'truck'],
                  [0, 60, 100,     'bus'],
                  [0, 80, 100,     'train'],
                  [0, 0, 230,      'motorcycle'],
                  [119, 11, 32,    'bicycle'],
                  [255, 255, 255, 'ood'],
                  [0, 0, 0,  'unknown']]
    name = 'cityscapes'
    ood_id = num_classes = 19
    ignore_id = num_classes + 1

    mapping = np.empty(len(city_mapping.labels))
    mapping.fill(ignore_id)
    for i, city_id in enumerate(city_mapping.get_train_ids()):
        mapping[city_id] = i

    def __init__(self, args):
        self.args = args
        self.last_block_pooling = args.last_block_pooling
        self.reshape_size = args.reshape_size

        self.mean = [123.68, 116.779, 103.939]
        self.std = [70.59564226, 68.52497082, 71.41913876]

        data_dir = args.data_path + '/wd_val_01/'
        files = next(os.walk(data_dir))[2]
        self.img_paths = [join(data_dir, f) for f in files if '_100000.png' in f]
        self.label_paths = {f: f[:-4] + '_labelIds.png' for f in self.img_paths
                            if os.path.exists(f[:-4] + '_labelIds.png')}
        self.names = [f[:-4] for f in files if '_100000.png' in f]

        print('\nTotal num images =', len(self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = pimg.open(img_path)

        batch = {}

        if img_path in self.label_paths.keys():
            labels = pimg.open(self.label_paths[img_path])

        img_width, img_height = img.size

        if self.reshape_size > 0:
            smaller_side = min(img_width, img_height)
            scale = float(self.reshape_size) / smaller_side
        else:
            scale = 1

        img_size = (int(img_width*scale), int(img_height*scale))

        img_size = transform.pad_size_for_pooling(
            img_size, self.last_block_pooling)
        img = transform.resize_img(img, img_size)

        if labels is not None:
            labels = transform.resize_labels(labels, img_size)
            labels = np.array(labels, dtype=np.int64)
            labels = self.mapping[labels]
        img = np.array(img, dtype=np.float32)

        batch['mean'] = self.mean
        batch['std'] = self.std
        img = transform.normalize(img, self.mean, self.std)

        img = transform.numpy_to_torch_image(img)
        batch['image'] = img

        batch['name'] = self.names[idx]
        if labels is not None:
            labels = torch.LongTensor(labels)
            batch['labels'] = labels
            batch['target_size'] = labels.shape[:2]
        else:
            batch['target_size'] = img.shape[:2]

        return batch

    def denormalize(self, img, mean, std):
        return transform.denormalize(img, mean, std)
