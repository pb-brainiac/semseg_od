import os
from os.path import join
from operator import itemgetter
import pickle

import torch
from torch.utils.data import Dataset
import PIL.Image as pimg
import numpy as np

import readers.transform as transform
import readers.viper_mapping as viper_mapping

import pdb

class DatasetReader(Dataset):
    class_info = [
                  [ 70, 130, 180, "sky"            ],
                  [128,  64, 128, "road"           ],
                  [244,  35, 232, "sidewalk"       ],
                  [152, 251, 152, "terrain"        ],
                  [ 87, 182,  35, "tree"           ],
                  [ 35, 142,  35, "vegetation"     ],
                  [ 70,  70,  70, "building"       ],
                  [153, 153, 153, "infrastructure" ],
                  [190, 153, 153, "fence"          ],
                  [150,  20,  20, "billboard"      ],
                  [250, 170,  30, "trafficlight"   ],
                  [220, 220,   0, "trafficsign"    ],
                  [180, 180, 100, "mobilebarrier"  ],
                  [173, 153, 153, "firehydrant"    ],
                  [168, 153, 153, "chair"          ],
                  [ 81,   0,  21, "trash"          ],
                  [ 81,   0,  81, "trashcan"       ],
                  [220,  20,  60, "person"         ],
                  [  0,   0, 230, "motorcycle"     ],
                  [  0,   0, 142, "car"            ],
                  [  0,  80, 100, "van"            ],
                  [  0,  60, 100, "bus"            ],
                  [  0,   0,  70, "truck"          ],
                  [255, 255, 255, "ood"            ],
                  [  0,   0,   0, "unknown"        ]
                 ]
    name = 'cityscapes'
    ood_id = num_classes = 23
    ignore_id = num_classes + 1

    mapping = np.empty(len(viper_mapping.labels))
    mapping.fill(ignore_id)
    for i, city_id in enumerate(viper_mapping.get_train_ids()):
        mapping[city_id] = i

    def __init__(self, args, subset='train'):
        self.args = args
        self.last_block_pooling = args.last_block_pooling
        self.reshape_size = args.reshape_size

        self.mean = [123.68, 116.779, 103.939]
        self.std = [70.59564226, 68.52497082, 71.41913876]

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
