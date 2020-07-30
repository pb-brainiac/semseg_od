import torch
from torch.utils.data import Dataset
import PIL.Image as pimg
import numpy as np
import readers.transform as transform
from tqdm import tqdm

class SegmentationReader(Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.last_block_pooling = args.last_block_pooling
        self.reshape_size = args.reshape_size
        self.mean = [123.68, 116.779, 103.939]
        self.std = [70.59564226, 68.52497082, 71.41913876]
        self.train = train

        if train:
            self.crop_size = args.crop_size
            self.min_jitter_scale = 0.75
            self.max_jitter_scale = 1.5

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

        if self.train:
            jitter_scale = np.random.uniform(
                self.min_jitter_scale, self.max_jitter_scale)
            scale *= jitter_scale

        img_size = (int(img_width*scale), int(img_height*scale))

        img_size = transform.pad_size_for_pooling(
            img_size, self.last_block_pooling)
        img = transform.resize_img(img, img_size)
        if labels is not None:
            labels = transform.resize_labels(labels, img_size)

        if self.train:
            img, labels = transform.random_crop([img, labels],
                                                self.crop_size)
            img, labels = transform.random_flip([img, labels])

        img = np.array(img, dtype=np.float32)
        if labels is not None:
            labels = np.array(labels, dtype=np.int64)
            labels = self.mapping[labels]

        if self.train:
            img = transform.pad(img, self.crop_size, 0)
            labels = transform.pad(labels, self.crop_size, self.ignore_id)

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

    def _oversample(self, oversample_classes):
        oversample_ids = []
        for i in range(len(self.class_info)):
            if self.class_info[i][-1] in oversample_classes:
                oversample_ids.append(i)

        index_oversample = []
        print(oversample_ids)
        for i, path in enumerate(tqdm(self.img_paths)):
            labels = np.array(pimg.open(self.label_paths[path]), dtype=np.uint8)
            labels = self.mapping[labels]
            ids = np.unique(labels)
            for cid in ids:
                if cid in oversample_ids:
                    index_oversample.append(i)
                    break

        for i in index_oversample:
            self.img_paths.append(self.img_paths[i])
            self.names.append(self.names[i])




