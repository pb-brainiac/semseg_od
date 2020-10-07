import os
from os.path import join
import sys
import argparse
import time
import shutil
from multiprocessing import Pool
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import PIL.Image as pimg
import matplotlib.pyplot as plt
import readers.transform as transform
import readers.cityscapes_reader as city_reader
import readers.wilddash_1_reader as wd_reader
import readers.lsun_reader as lsun_reader
import readers.pascal_wd_reader as pascal_wd_reader
import readers.viper_reader as viper_reader
import sklearn.metrics as sm
import evaluation
import pdb
import utils
import random


def colorize_labels(y, class_colors):
    width = y.shape[1]
    height = y.shape[0]
    y_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for cid in range(len(class_colors)):
        cpos = np.repeat((y == cid).reshape((height, width, 1)), 3, axis=2)
        cnum = cpos.sum() // 3
        y_rgb[cpos] = np.array(class_colors[cid][:3] * cnum, dtype=np.uint8)
    return y_rgb

def store_images(img_raw, pred, true, class_info, name):
    img_pred = colorize_labels(pred, class_info)

    error_mask = np.ones(img_raw.shape)
    if true is not None:
        img_true = colorize_labels(true, class_info)
        img_errors = img_pred.copy()
        correct_mask = pred == true
        error_mask = pred != true
        ignore_mask = true == ignore_id
        img_errors[correct_mask] = 0
        img_errors[ignore_mask] = 0
        num_errors = error_mask.sum()
        img1 = np.concatenate((img_raw, img_true), axis=1)
        img2 = np.concatenate((img_errors, img_pred), axis=1)
        img = np.concatenate((img1, img2),axis=0)
        filename = '%s_%07d.jpg' % (name, num_errors)
        save_path = join(save_dir, filename)
    else:
        line = np.zeros((5, pred.shape[1], 3)).astype(np.uint8)
        img = np.concatenate((img_raw, line, img_pred), axis=0)
        save_path = join(save_dir, '%s.jpg' % (name))

    img = pimg.fromarray(img)
    saver_pool.apply_async(img.save, [save_path])

def get_conf_img(img_raw, conf, conf_type):
    conf = (conf * (-1)) + 1
    conf_broad = np.reshape(conf, [conf.shape[0], conf.shape[1], 1])

    conf_save = plt.get_cmap('jet')(conf)
    conf_save = (conf_save * 255).astype(np.uint8)[:, :, :3]
    img = conf_save
    return img

def store_conf(img_raw, conf, name, conf_type='logit'):
    conf = conf[0]
    img_conf = get_conf_img(img_raw, conf, 'conf_' + conf_type)
    img = np.concatenate([img_raw, img_conf], axis=0)
    save_path = join(save_dir, 'confidence',
                     '%s_%s.jpg' % (name, conf_type))
    img = pimg.fromarray(img)
    saver_pool.apply_async(img.save, [save_path])

def store_outputs(batch, pred, pred_w_outlier, conf_probs):
    pred = pred.detach().cpu().numpy().astype(np.int32)
    pred_w_outlier = pred_w_outlier.detach().cpu().numpy().astype(np.int32)
    conf_probs = conf_probs.detach().cpu().numpy()
    img_raw = transform.denormalize(batch['image'][0],
                                    batch['mean'][0].numpy(), batch['std'][0].numpy())
    true = batch['labels'][0].numpy().astype(np.int32)
    name = batch['name'][0]
    store_images(img_raw, pred, true, class_info, 'segmentation/'+name)
    store_images(img_raw, pred_w_outlier, true, class_info, 'seg_with_conf/'+ name)
    store_conf(img_raw, conf_probs, name, 'probs')

def evaluate_segmentation():
    conf_mats = {}
    conf_mats['seg'] = torch.zeros((num_classes, num_classes), dtype=torch.int64).cuda()
    conf_mats['seg_w_outlier'] = torch.zeros((num_classes+1, num_classes+1), dtype=torch.int64).cuda()

    log_interval = max(len(wd_data_loader) // 5, 1)
    for step, batch in enumerate(wd_data_loader):
        try:
            pred, pred_w_outlier, conf_probs = evaluation.segment_image(model, batch, args, conf_mats, ood_id, num_classes)
            print(pred.shape)
            if args.save_outputs:
                store_outputs(batch, pred, pred_w_outlier, conf_probs)

        except Exception as e:
            print('failed on image: {}'.format(batch['name'][0]))
            print('error: {}'.format(e))
            print(traceback.format_exc())

        if step % log_interval == 0:
            print('step {} / {}'.format(step, len(wd_data_loader)))

    print('\nSegmentation:')
    conf_mats['seg'] = conf_mats['seg'].cpu().numpy()
    evaluation.compute_errors(conf_mats['seg'], 'Validation', class_info, nc=num_classes, verbose=True)
    print('\nSegmentation with confidence:')
    conf_mats['seg_w_outlier'] = conf_mats['seg_w_outlier'].cpu().numpy()
    evaluation.compute_errors(conf_mats['seg_w_outlier'], 'Validation', class_info, nc=num_classes)

def evaluate_AP_negative():
    gt_wd = torch.ByteTensor([])
    conf_wd = torch.FloatTensor([])

    log_interval = 20
    print('\ninliers:')
    for step, batch in enumerate(wd_data_loader):
        img = torch.autograd.Variable(batch['image'].cuda(
                non_blocking=True), requires_grad=True)
        with torch.no_grad():
            _, conf_probs = model(img, batch['image'].shape[2:])
            conf_probs = conf_probs.view(-1)

        gt_wd = torch.cat((gt_wd,torch.zeros(conf_probs.shape[0], dtype=torch.uint8)))
        conf_wd = torch.cat((conf_wd, conf_probs.cpu()))

        if step % log_interval == 0:
            print('step {} / {}'.format(step, len(wd_data_loader)))

    AP = []
    for i in range(args.AP_iters):
        pixel_counter = (gt_wd==0).sum()
        gt = gt_wd.clone()
        conf = conf_wd.clone()
        print('\noutliers:')
        for step, batch in enumerate(lsun_data_loader):
            img = torch.autograd.Variable(batch['image'].cuda(
                    non_blocking=True), requires_grad=True)
            with torch.no_grad():
                _, conf_probs = model(img, batch['image'].shape[2:])
                conf_probs = conf_probs.view(-1)

            gt = torch.cat((gt, torch.ones(conf_probs.shape[0], dtype=torch.uint8)))
            conf = torch.cat((conf, conf_probs.cpu()))
            torch.cuda.empty_cache()

            if step % log_interval == 0:
                print('step {} / {}'.format(step, len(lsun_data_loader)))

            pixel_counter -= conf_probs.shape[0]
            if pixel_counter < 0:
                break

        conf = conf * -1 + 1
        average_precision = sm.average_precision_score(gt, conf)
        print(average_precision)
        AP.append(average_precision)

    AP = np.array(AP)
    print('negative images average precision: {} +/- {}'.format(AP.mean(),AP.std()))

def evaluate_AP_patches():
    log_interval = 20
    AP = []

    for i in range(args.AP_iters):
        gt = torch.ByteTensor([])
        conf = torch.FloatTensor([])
        for step, batch in enumerate(pascal_wd_data_loader):
            img = torch.autograd.Variable(batch['image'].cuda(
                    non_blocking=True), requires_grad=True)
            with torch.no_grad():
                _, conf_probs = model(img, batch['image'].shape[2:])
                conf_probs = conf_probs.view(-1)

            gt = torch.cat((gt, batch['labels'].view(-1)))
            conf = torch.cat((conf, conf_probs.cpu()))

            if step % log_interval == 0:
                print('step {} / {}'.format(step, len(pascal_wd_data_loader)))

        conf = conf * -1 + 1
        average_precision = sm.average_precision_score(gt, conf)
        print(average_precision)
        AP.append(average_precision)

    AP = np.array(AP)
    print('negative images average precision: {} +/- {}'.format(AP.mean(),AP.std()))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--params', type=str)
    parser.add_argument('--save-outputs', type=int, default=0)
    parser.add_argument('--reshape-size', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--AP-iters', type=int, default=50)
    parser.add_argument('--save-name', type=str, default='')
    parser.add_argument('--data-path', type=str, default='./data/')
    return parser.parse_args()

def prepare_for_saving():
    global saver_pool, save_dir

    saver_pool = Pool(processes=4)
    save_dir = join('./outputs', args.save_name)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    split_classes = ['segmentation', 'seg_with_conf', 'confidence']

    for class_name in split_classes:
        os.makedirs(join(save_dir, class_name), exist_ok=True)

    log_file = open(join(save_dir, 'log.txt'), 'w')
    sys.stdout = utils.Logger(sys.stdout, log_file)

torch.manual_seed(0)
random.seed(0)

args = get_args()

if args.save_outputs:
    prepare_for_saving()


net_model = utils.import_module('net_model', args.model)

model = net_model.build(args=args)
state_dict = torch.load(args.params,
                        map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict, convert=True) 
model.cuda()
model = model.eval()



wd_dataset = wd_reader.DatasetReader(args)
#wd_dataset = city_reader.DatasetReader(args, subset='val')
wd_data_loader = DataLoader(wd_dataset, batch_size=1,
                         num_workers=8, pin_memory=True, shuffle=False)

class_info = wd_dataset.class_info
ignore_id = wd_dataset.ignore_id
ood_id = wd_dataset.ood_id
num_classes = wd_dataset.num_classes

#lsun_dataset = lsun_reader.DatasetReader(args)
#lsun_data_loader = DataLoader(lsun_dataset, batch_size=1,
#                         num_workers=1, pin_memory=True, shuffle=True)
pascal_wd_dataset = pascal_wd_reader.DatasetReader(args)
pascal_wd_data_loader = DataLoader(pascal_wd_dataset, batch_size=1,
                         num_workers=0, pin_memory=True, shuffle=True)



#evaluate_segmentation()
#evaluate_AP_negative()
evaluate_AP_patches()
