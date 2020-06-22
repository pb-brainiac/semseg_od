import os
from os.path import join
from operator import itemgetter
import pickle

import torch
from torch.utils.data import Dataset
import PIL.Image as pimg
import numpy as np

import readers.transform as transform

import pdb

class DatasetReader(Dataset):
    def __init__(self, args):
        self.args = args
        self.last_block_pooling = args.last_block_pooling
        self.reshape_size = args.reshape_size

        self.mean = [123.68, 116.779, 103.939]
        self.std = [70.59564226, 68.52497082, 71.41913876]

        data_dir = args.data_path + '/lsun'
        files = next(os.walk(data_dir))[2]
        self.img_paths = [join(data_dir, f) for f in files]
        self.names = [f[:-5] for f in files]

        print('\nTotal num images =', len(self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = pimg.open(img_path)

        batch = {}

        img_width, img_height = img.size
        smaller_side = min(img_width, img_height)

        scale = float(self.reshape_size) / smaller_side
        img_size = (int(img_width*scale), int(img_height*scale))

        img_size = transform.pad_size_for_pooling(
            img_size, self.last_block_pooling)
        img = transform.resize_img(img, img_size)

        img = np.array(img, dtype=np.float32)

        batch['mean'] = self.mean
        batch['std'] = self.std
        img = transform.normalize(img, self.mean, self.std)

        img = transform.numpy_to_torch_image(img)
        batch['image'] = img
        batch['name'] = self.names[idx]
        batch['target_size'] = img.shape[:2]

        return batch

    def denormalize(self, img, mean, std):
        return transform.denormalize(img, mean, std)
