import os
from os.path import join

import numpy as np
import json
import xml.etree.ElementTree as ET
import math
from readers.segmentation_reader import SegmentationReader
import readers.city_mapping as city_mapping
import readers.vistas_to_city_mapping as vistas_city_mapping
import PIL.Image as pimg
import random
import readers.transform as transform
import torch
import pdb

class DatasetReader(SegmentationReader):
    class_info = city_mapping.get_class_info()
    class_info.extend([[255, 255, 255, 'ood'], [0, 0, 0,  'unknown']])
    name = 'cityscapes'
    ood_id = num_classes = 19
    ignore_id = num_classes + 1

    cityscapes_mapping = np.empty(len(city_mapping.labels))
    cityscapes_mapping.fill(ignore_id)
    for i, city_id in enumerate(city_mapping.get_train_ids()):
        cityscapes_mapping[city_id] = i

    name_to_id_map = {}
    for i in range(len(class_info)):
        name_to_id_map[class_info[i][-1]] = i
    with open('/mnt/sda1/datasets/mapillary-vistas/config.json') as config_file:
        config = json.load(config_file)
    labels = config['labels']
    vistas_mapping = np.empty(len(labels), dtype=np.uint8)
    vistas_mapping.fill(ignore_id)
    for vistas_id, label in enumerate(labels):
        key = label["name"]
        if key in vistas_city_mapping.vistas_to_cityscapes:
            cityscapes_id = name_to_id_map[vistas_city_mapping.vistas_to_cityscapes[key]]
            vistas_mapping[vistas_id] = cityscapes_id

    labels_to_ood_mapping = np.empty(len(class_info), dtype=np.uint8)
    labels_to_ood_mapping.fill(0)
    labels_to_ood_mapping[-2] = 1
    labels_to_ood_mapping[-1] = 2

    def __init__(self, args, subset='train', train=False):
        SegmentationReader.__init__(self, args, train)
        data_dir = join(args.data_path, 'cityscapes', 'leftImg8bit', subset)

        print(data_dir)
        cities = next(os.walk(data_dir))[1]

        city_img_paths = []
        city_label_paths = {}
        city_names = []

        for city in cities:
            files_path = join(data_dir, city)
            files = next(os.walk(files_path))[2]
            city_img_paths.extend([join(files_path, f) for f in files])
            city_names.extend([f[:-4] for f in files])

        city_label_paths.update({f: (f.replace('_leftImg8bit','_gtFine_labelIds').replace('leftImg8bit','gtFine'), self.cityscapes_mapping)
                                 for f in city_img_paths
                                 if os.path.exists(f.replace('_leftImg8bit','_gtFine_labelIds').replace('leftImg8bit','gtFine'))})

        data_dir =  './data/wd_val_01/'

        wd_img_paths = []
        wd_label_paths = {}
        wd_names = []

        files = next(os.walk(data_dir))[2]
        wd_img_paths = [join(data_dir, f) for f in files if '_100000.png' in f]
        wd_label_paths = {f: (f[:-4] + '_labelIds.png', self.cityscapes_mapping)
                          for f in wd_img_paths
                          if os.path.exists(f[:-4] + '_labelIds.png')}
        wd_names = [f[:-4] for f in files if '_100000.png' in f]

        subset_dir = 'training/' if subset == 'train' else 'validation/'
        data_dir = join(args.data_path, 'vistas',  subset_dir, 'images')

        print(data_dir)
        vistas_img_paths = []
        vistas_label_paths = {}
        vistas_names = []

        files = next(os.walk(data_dir))[2]
        vistas_img_paths.extend([join(data_dir, f) for f in files])
        vistas_label_paths.update({f: (f.replace('images','labels').replace('jpg','png'), self.vistas_mapping) for f in vistas_img_paths})
        vistas_names = [f[:-4] for f in files]

        self.inlier_img_paths = city_img_paths + wd_img_paths + vistas_img_paths
        self.inlier_label_paths = {}
        self.inlier_label_paths.update(city_label_paths)
        self.inlier_label_paths.update(wd_label_paths)
        self.inlier_label_paths.update(vistas_label_paths)
        self.inlier_names = city_names + wd_names + vistas_names

        data_dir = join('/mnt/sdb1/imagenet/ILSVRC2015/Data/CLS-LOC/', subset)
        anno_dir = join('/mnt/sdb1/imagenet/ILSVRC2015/Annotations/CLS-LOC/', subset)
        image_list_path = join(anno_dir, subset+'.txt')
        print(image_list_path)

        self.outlier_img_paths = []
        self.outlier_object_locations = []
        self.outlier_names = []

        image_list_f = open(image_list_path)
        files = [line.split(' ')[0]
                 for line in image_list_f.readlines()]
        files = [f for f in files if os.path.exists(join(anno_dir, f[:-5]+'.xml'))]
        self.outlier_img_paths.extend([join(data_dir, f) for f in files])
        bbs = [ET.parse(join(anno_dir, f[:-5]+'.xml')).getroot()[5][4] for f in files]
        self.outlier_object_locations.extend([(0, (int(bb[0].text), int(bb[1].text), int(bb[2].text), int(bb[3].text)))  for bb in bbs])
        self.outlier_names.extend([f[:-5] for f in files])


        print('\nTotal num images =', len(self.outlier_img_paths))

    def __len__(self):
        return len(self.outlier_img_paths)

    def __getitem__(self, idx):
        img_path = self.outlier_img_paths[idx]
        outlier_img = pimg.open(img_path)

        if outlier_img.mode != 'RGB':
            outlier_img = outlier_img.convert('RGB')

        batch = {}

        full_outlier = 0
        if random.uniform(0,1) < 0.5:
            outlier_img = np.array(outlier_img)

            ind = random.randint(0, len(self.inlier_img_paths)-1)
            inlier_img = pimg.open(self.inlier_img_paths[ind])
            name = self.inlier_names[ind]
            if inlier_img.mode != 'RGB':
                inlier_img = inlier_img.convert('RGB')

            inlier_img = np.array(inlier_img)

            _, bb = self.outlier_object_locations[idx]

            h_crop = bb[3] - bb[1]
            up, down = bb[1], bb[3]

            w_crop = bb[2] - bb[0]
            left, right = bb[0], bb[2]

            crop = outlier_img[up:down, left:right, :]
            crop = pimg.fromarray(crop)

            h_inlier, w_inlier, _ = inlier_img.shape

            scale = random.randint(1,100)/1000.0
            i = math.sqrt((scale * h_inlier * w_inlier) / (h_crop * w_crop))

            if w_crop*i > w_inlier:
                i = w_inlier / w_crop
            if h_crop*i > h_inlier:
                i = h_inlier / h_crop
            crop = crop.resize((int(w_crop*i), int(h_crop*i)))
            crop = np.array(crop)

            h_crop, w_crop, _ = crop.shape
            h_start, w_start = random.randint(0, h_inlier-h_crop), random.randint(0,w_inlier-w_crop)

            inlier_img[h_start:h_start + h_crop, w_start:w_start+w_crop, :] = crop
            path, mapping = self.inlier_label_paths[self.inlier_img_paths[ind]]
            labels = pimg.open(path)
            labels = np.array(labels)
            labels = mapping[labels]
            labels[h_start:h_start + h_crop, w_start:w_start+w_crop] = self.ood_id

            img = pimg.fromarray(inlier_img)

        else:
            full_outlier = 1
            _, bb = self.outlier_object_locations[idx]
            img = outlier_img
            labels = np.ones((img.size[1], img.size[0]), dtype=np.uint8) * self.ignore_id
            labels[bb[1]:bb[3],bb[0]:bb[2]] = self.ood_id

        labels = labels.astype(np.uint8)
        labels = pimg.fromarray(labels)

        img_width, img_height = img.size

        if self.reshape_size > 0:
            smaller_side = min(img_width, img_height)
            scale = float(self.reshape_size) / smaller_side
        else:
            scale = 1

        if self.train:
            if random.uniform(0,1) < 0.7:
                scale_jitter = np.random.uniform(
                    self.min_jitter_scale, self.max_jitter_scale)
                scale *= scale_jitter
            else:
                scale = float(self.crop_size) / smaller_side

        img_size = (int(img_width*scale), int(img_height*scale))

        img_size = transform.pad_size_for_pooling(
            img_size, self.last_block_pooling)
        img = transform.resize_img(img, img_size)

        if labels is not None:
            labels = transform.resize_labels(labels, img_size)

        if self.train:
            if max(img_size) > self.crop_size:
                labels_tmp = labels
                img_tmp = img
                my_count = 0
                while True:
                    my_count += 1
                    img, labels = transform.random_crop(
                          [img_tmp, labels_tmp], self.crop_size)
                    if full_outlier:
                        if np.sum(np.array(labels) == 20) < 0.2 * self.crop_size * self.crop_size or my_count > 20:
                            break
                    else:
                        labels_np = np.array(labels)
                        if len(np.unique(labels_np))>4 or my_count> 20:
                            break

            img, labels = transform.random_flip([img, labels])

        img = np.array(img, dtype=np.float32)
        if labels is not None:
            labels = np.array(labels, dtype=np.int64)

        if self.train:
            img = transform.pad(img, self.crop_size, 0)
            labels = transform.pad(labels, self.crop_size, self.ignore_id)
            labels_ood = self.labels_to_ood_mapping[labels]
            labels[labels==self.ood_id] = self.ignore_id

        batch['mean'] = self.mean
        batch['std'] = self.std
        img = transform.normalize(img, self.mean, self.std)

        img = transform.numpy_to_torch_image(img)
        batch['image'] = img

        batch['name'] = self.outlier_names[idx]
        if labels is not None:
            labels = torch.LongTensor(labels)
            batch['labels'] = labels
            labels_ood = torch.LongTensor(labels_ood)
            batch['labels_ood'] = labels_ood
            batch['target_size'] = labels.shape[:2]
        else:
            batch['target_size'] = img.shape[:2]

        return batch


