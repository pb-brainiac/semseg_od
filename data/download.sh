#!/usr/bin/env bash
python3 download_lsun.py
unzip '*.zip'
python3 export_lsun.py
rm -rf *lmdb*

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
rm -rf *.tar
