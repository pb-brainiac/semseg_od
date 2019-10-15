# -*- coding: utf-8 -*-

from __future__ import print_function

import cv2
import lmdb
import numpy
import os
from os.path import exists, join
from urllib.request import Request, urlopen


def export_images(db_path, out_dir):
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            image_out_dir = out_dir
            if not exists(image_out_dir):
                os.makedirs(image_out_dir)
            image_out_path = join(image_out_dir, key.decode() + '.webp')
            with open(image_out_path, 'wb') as fp:
                fp.write(val)
            count += 1
            if count % 1000 == 0:
                print('Finished', count, 'images')


def list_categories():
    url = 'http://dl.yf.io/lsun/categories.txt'
    with urlopen(Request(url)) as response:
        return response.read().decode().strip().split('\n')

def main():
    categories = list_categories()
    categories.remove('test')
    for category in categories:
        export_images('./{}_val_lmdb'.format(category), './lsun')

if __name__ == '__main__':
    main()
