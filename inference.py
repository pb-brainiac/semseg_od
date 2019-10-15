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
import libs.cylib as cylib
import matplotlib.pyplot as plt
import importlib.util
import readers.wilddash_reader as wd_reader
import readers.lsun_reader as lsun_reader
import readers.pascal_wd_reader as pascal_wd_reader
import sklearn.metrics as sm
import pdb

class Logger(object):

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def compute_errors(conf_mat, name, class_infoi, verbose=False):
    num_correct = conf_mat.trace()
    print(num_classes)
    total_size = conf_mat.sum()
    avg_pixel_acc = num_correct / total_size * 100.0
    TPFP = conf_mat.sum(0)
    TPFN = conf_mat.sum(1)
    FN = TPFN - conf_mat.diagonal()
    FP = TPFP - conf_mat.diagonal()
    class_iou = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_precision = np.zeros(num_classes)
    print(name, 'errors:')
    for i in range(num_classes):
        TP = conf_mat[i, i]
        if (TP + FP[i] + FN[i]) > 0:
            class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
        else:
            class_iou[i] = 0
        if TPFN[i] > 0:
            class_recall[i] = (TP / TPFN[i]) * 100.0
        else:
            class_recall[i] = 0
        if TPFP[i] > 0:
            class_precision[i] = (TP / TPFP[i]) * 100.0
        else:
            class_precision[i] = 0

        class_name = class_info[i][3]
        if verbose:
            print('\t%d IoU accuracy = %.2f %%' % (i, class_iou[i]))
            print('\t%d recall = %.2f %%' % (i, class_recall[i]))
            print('\t%d precision = %.2f %%' % (i, class_precision[i]))
    avg_class_iou = class_iou.mean()
    avg_class_recall = class_recall.mean()
    avg_class_precision = class_precision.mean()
    print('IoU mean class accuracy -> TP / (TP+FN+FP) = %.2f %%' %
          avg_class_iou)
    print('mean class recall -> TP / (TP+FN) = %.2f %%' % avg_class_recall)
    print('mean class precision -> TP / (TP+FP) = %.2f %%' %
          avg_class_precision)
    print('pixel accuracy = %.2f %%' % avg_pixel_acc)

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

def segment_image(model, batch, args, conf_mats):
    name = batch['name'][0]
    img = torch.autograd.Variable(batch['image'].cuda(
            non_blocking=True), requires_grad=True)
    with torch.no_grad():
        probs, conf_probs = model(img, batch['image'].shape[2:])

    pred = probs.max(dim=1)[1]

    pred_w_outlier = pred.clone()
    pred_w_outlier[conf_probs < 0.5] = ood_id

    pred = pred.detach().cpu().numpy().astype(np.int32)[0]
    pred_w_outlier = pred_w_outlier.detach().cpu().numpy().astype(np.int32)[0]
    conf_probs = conf_probs.detach().cpu().numpy()

    true = None
    if batch['labels'] is not None:
        labels = batch['labels'][0]
        true = labels.numpy().astype(np.int32)

        cylib.collect_confusion_matrix(
                pred[true<num_classes], true[true<num_classes], conf_mats['seg'])
        cylib.collect_confusion_matrix(
                pred_w_outlier[true<num_classes], true[true<num_classes], conf_mats['seg_w_outlier'])
    if args.save_outputs:
        img_raw = wd_dataset.denormalize(batch['image'][0],
                                      batch['mean'][0].numpy(), batch['std'][0].numpy())
        store_images(img_raw, pred, true, class_info, 'segmentation/'+name)
        store_images(img_raw, pred_w_outlier, true, class_info, 'seg_with_conf/'+ name)
        store_conf(img_raw, conf_probs, name, 'probs')

def evaluate_segmentation():
    conf_mats = {}
    conf_mats['seg'] = np.zeros((num_classes, num_classes), dtype=np.uint64)
    conf_mats['seg_w_outlier'] = np.zeros((num_classes+1, num_classes+1), dtype=np.uint64)

    log_interval = max(len(wd_data_loader) // 5, 1)
    for step, batch in enumerate(wd_data_loader):
        try:
            segment_image(model, batch, args, conf_mats)
        except Exception as e:
            print('failed on image: {}'.format(batch['name'][0]))
            print('error: {}'.format(e))
            print(traceback.format_exc())

        if step % log_interval == 0:
            print('step {} / {}'.format(step, len(wd_data_loader)))

    print('\nSegmentation:')
    compute_errors(conf_mats['seg'], 'Validation', class_info)
    print('\nSegmentation with confidence:')
    compute_errors(conf_mats['seg_w_outlier'], 'Validation', class_info)

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
    sys.stdout = Logger(sys.stdout, log_file)


args = get_args()

if args.save_outputs:
    prepare_for_saving()


net_model = import_module('net_model', args.model)

model = net_model.build(args=args)
state_dict = torch.load(args.params,
                        map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)
model.cuda()
model = model.eval()

class_info = wd_reader.DatasetReader.class_info
ignore_id = wd_reader.DatasetReader.ignore_id
ood_id = wd_reader.DatasetReader.ood_id
num_classes = wd_reader.DatasetReader.num_classes


wd_dataset = wd_reader.DatasetReader(args)
wd_data_loader = DataLoader(wd_dataset, batch_size=1,
                         num_workers=1, pin_memory=True, shuffle=False)
lsun_dataset = lsun_reader.DatasetReader(args)
lsun_data_loader = DataLoader(lsun_dataset, batch_size=1,
                         num_workers=1, pin_memory=True, shuffle=True)
pascal_wd_dataset = pascal_wd_reader.DatasetReader(args)
pascal_wd_data_loader = DataLoader(pascal_wd_dataset, batch_size=1,
                         num_workers=0, pin_memory=True, shuffle=True)



evaluate_segmentation()
evaluate_AP_negative()
evaluate_AP_patches()
