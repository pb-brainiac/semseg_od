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
