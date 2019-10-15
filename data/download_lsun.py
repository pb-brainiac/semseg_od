# -*- coding: utf-8 -*-

from __future__ import print_function, division

import subprocess
from urllib.request import Request, urlopen

from os.path import join

def list_categories():
    url = 'http://dl.yf.io/lsun/categories.txt'
    with urlopen(Request(url)) as response:
        return response.read().decode().strip().split('\n')

def download(out_dir, category, set_name):
    url = 'http://dl.yf.io/lsun/scenes/{category}_' \
          '{set_name}_lmdb.zip'.format(**locals())
    out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)


def main():
    categories = list_categories()
    categories.remove('test')
    print('Downloading', len(categories), 'categories')
    for category in categories:
        download('./', category, 'val')

if __name__ == '__main__':
    main()
