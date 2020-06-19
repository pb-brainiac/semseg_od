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
import xml.etree.ElementTree as ET
import pdb
import random

import matplotlib.pyplot as plt

class DatasetReader(Dataset):
    def __init__(self, args):
        self.args = args
        self.last_block_pooling = args.last_block_pooling
        self.reshape_size = args.reshape_size

        self.mean = [123.68, 116.779, 103.939]
        self.std = [70.59564226, 68.52497082, 71.41913876]

        pascal_dir = args.data_path + '/VOCdevkit/VOC2007/'
        data_dir = join(pascal_dir, 'SegmentationObject')
        files = next(os.walk(data_dir))[2]
        animals = {'bird': 3, 'cat': 8, 'cow': 10, 'dog': 12, 'horse': 13, 'sheep': 17}
        self.crops = []
        for f in files:
            annotation = ET.parse(join(pascal_dir, 'Annotations',
                                       f.replace('.png','.xml')))
            anno_iterator = annotation.getiterator()
            objects = [(child.tag, child.text) for child in anno_iterator
                       if child.tag in ['name','xmax','xmin','ymax','ymin']][1:]
            filtered_objects = [(i+1,                      # object index
                                 #animals[objects[i*5][1]],# class
                                 int(objects[i*5+1][1]),   # xmin
                                 int(objects[i*5+2][1]),   # ymin
                                 int(objects[i*5+3][1]),   # xmax
                                 int(objects[i*5+4][1]))   # ymax
                                for i in range(int(len(objects)/5))
                                if objects[i*5][1] in animals.keys()]

            for fo in filtered_objects:
                img_path = join(pascal_dir, 'JPEGImages', f.replace('.png', '.jpg'))
                labels_path = join(pascal_dir, 'SegmentationObject', f)
                self.crops.append((img_path, labels_path, fo))

        wd_dir = './data/wd_val_01/'
        files = next(os.walk(wd_dir))[2]
        self.wd_paths = [join(wd_dir, f) for f in files if '_100000.png' in f]

        print('\nTotal num images =', len(self.wd_paths))

    def __len__(self):
        return len(self.wd_paths)

    def __getitem__(self, idx):
        batch = {}

        img_wd = pimg.open(self.wd_paths[idx])
        w_wd, h_wd = img_wd.size
        img_wd = np.array(img_wd, dtype=np.uint8)

        while True:
            ind = random.randint(0, len(self.crops)-1)
            img_path, label_path, (i, xmin, ymin, xmax, ymax) = self.crops[ind]

            label_pascal = pimg.open(label_path)
            label_pascal = np.array(label_pascal, dtype=np.uint8)
            label_pascal = label_pascal[ymin:ymax, xmin:xmax]
            label_pascal = (label_pascal==i).astype(np.uint8)
            if np.count_nonzero(label_pascal==1) > 0.007 * w_wd * h_wd:
                break

        img_pascal = pimg.open(img_path)
        img_pascal = img_pascal.crop((xmin, ymin, xmax, ymax))
        img_pascal = np.array(img_pascal, dtype=np.uint8)


        h_pascal, w_pascal = ymax - ymin, xmax - xmin
        h_start, w_start = random.randint(0, h_wd-h_pascal), random.randint(0,w_wd-w_pascal)
        img_wd[h_start:h_start+h_pascal, w_start:w_start+w_pascal][label_pascal==1] = \
                img_pascal[label_pascal==1]
        labels = np.zeros((h_wd, w_wd), dtype=np.uint8)
        labels[h_start:h_start+h_pascal, w_start:w_start+w_pascal][label_pascal==1] = 1

        img_wd = pimg.fromarray(img_wd)
        labels = pimg.fromarray(labels)

        if self.reshape_size > 0:
            smaller_side = min(w_wd, h_wd)
            scale = float(self.reshape_size) / smaller_side
        else:
            scale = 1

        img_size = (int(w_wd*scale), int(h_wd*scale))

        img_size = transform.pad_size_for_pooling(
            img_size, self.last_block_pooling)

        img_wd = transform.resize_img(img_wd, img_size)
        labels = transform.resize_labels(labels, img_size)

        img_wd = np.array(img_wd, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)

        batch['mean'] = self.mean
        batch['std'] = self.std
        img_wd = transform.normalize(img_wd, self.mean, self.std)

        img_wd = transform.numpy_to_torch_image(img_wd)
        batch['image'] = img_wd
        batch['labels'] = torch.ByteTensor(labels)
        return batch

    def denormalize(self, img, mean, std):
        return transform.denormalize(img, mean, std)
