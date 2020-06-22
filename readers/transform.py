import torch
import math
import random
import PIL.Image as pimg
from PIL import ImageFilter
import numpy as np


def resize_img(img, size):
  if img.size != size:
    return img.resize(size, pimg.BICUBIC)
  return img


def resize_labels(img, size):
  if img.size != size:
    img = img.resize(size, pimg.NEAREST)
  return img

def random_crop(images, crop_size, snap_margin_prob=0.1):
  if isinstance(crop_size, int):
    crop_h = crop_size
    crop_w = crop_size
  else:
    crop_h, crop_w = crop_size

  height = images[0].size[1]
  width = images[0].size[0]
  sx, crop_w = _sample_location(width, crop_w, snap_margin_prob)
  sy, crop_h = _sample_location(height, crop_h, snap_margin_prob)

  cropped = []
  for img in images:
    cropped.append(img.crop((sx, sy, sx+crop_w, sy+crop_h)))
  return cropped

def _sample_location(dim_size, crop_size, snap_margin_prob):
  if dim_size > crop_size:
    max_start = dim_size - crop_size
    snap_margin = int(snap_margin_prob/2 * max_start)
    start_pos = np.random.randint(-snap_margin, max_start+1+snap_margin)
    start_pos = max(start_pos, 0)
    start_pos = min(start_pos, max_start)
    size = crop_size
  else:
    start_pos = 0
    size = dim_size
  return start_pos, size

def numpy_to_torch_image(img):
  img = torch.FloatTensor(img)
  return img.permute(2,0,1).contiguous()


def normalize(img, mean, std):
  img -= mean
  img /= std
  return img


def denormalize(img, mean, std):
  img = img.permute(1,2,0).contiguous()
  img = img.numpy()
  img *= std
  img += mean
  img[img<0] = 0
  img[img>255] = 255
  img = img.astype(np.uint8)
  return img


def pad_size_for_pooling(size, last_block_pooling):
  new_size = list(size)
  for i in range(len(new_size)):
    mod = new_size[i] % last_block_pooling
    if mod > 0:
      new_size[i] += last_block_pooling - mod
  return tuple(new_size)
