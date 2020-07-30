import os
from os.path import join
import sys
import shutil
import argparse
import utils
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchsummary import summary
import math
import pdb
import evaluation
import readers.transform as transform
import matplotlib.pyplot as plt
import numpy as np
import readers.cityscapes_reader as city_reader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--reader', type=str)
    parser.add_argument('--reshape-size', type=int, default=1)
    parser.add_argument('--crop-size', type=int, default=1)
    parser.add_argument('--save-outputs', type=int, default=0)
    parser.add_argument('--save-name', type=str, default='')
    parser.add_argument('--data-path', type=str, default='./data/')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--pretrained', type=int, default=1)
    return parser.parse_args()

def prepare_for_saving():
    global save_dir

    save_dir = join('./results', args.save_name)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    split_classes = ['params']

    for class_name in split_classes:
        os.makedirs(join(save_dir, class_name), exist_ok=True)

    log_file = open(join(save_dir, 'log.txt'), 'w')
    sys.stdout = utils.Logger(sys.stdout, log_file)

def evaluate_segmentation(model, loader):
    conf_mats = {}
    conf_mats['seg'] = torch.zeros((num_classes, num_classes), dtype=torch.int64).cuda()
    conf_mats['seg_w_outlier'] = torch.zeros((num_classes+1, num_classes+1), dtype=torch.int64).cuda()
    log_interval = max(len(loader) // 5, 1)
    for step, batch in enumerate(loader):
        try:
            evaluation.segment_image(model, batch, args, conf_mats, ood_id, num_classes)

        except Exception as e:
            print('failed on image: {}'.format(batch['name'][0]))
            print('error: {}'.format(e))
            print(traceback.format_exc())
        if step % log_interval == 0:
            print('step {} / {}'.format(step, len(loader)))

    print('\nSegmentation:')
    conf_mats['seg'] = conf_mats['seg'].cpu().numpy()
    evaluation.compute_errors(conf_mats['seg'], 'Validation', class_info, nc=num_classes, verbose=True)
    print('\nSegmentation with confidence:')
    conf_mats['seg_w_outlier'] = conf_mats['seg_w_outlier'].cpu().numpy()
    evaluation.compute_errors(conf_mats['seg_w_outlier'], 'Validation', class_info, nc=num_classes)

seed = 314159
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

args = get_args()

if args.save_outputs:
    prepare_for_saving()

net_model = utils.import_module('net_model', args.model)
reader = utils.import_module('reader', args.reader)
class_info = reader.DatasetReader.class_info
num_classes = reader.DatasetReader.num_classes
ignore_id = reader.DatasetReader.ignore_id
ood_id = reader.DatasetReader.ood_id
model = net_model.build(pretrained=True, args=args, num_classes=num_classes, ignore_id=ignore_id)
model = model.cuda()
summary(model,(3, 512,512))

train_dataset = reader.DatasetReader(args, 'train', train=True)
val_dataset = city_reader.DatasetReader(args, 'val')
#train_eval_dataset = reader.DatasetReader(args, 'train')

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
#train_eval_loader = DataLoader(train_eval_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

fine_tune = []
fine_tune.extend(model.feature_extractor.backbone.features.parameters())
print(len(fine_tune))
random_init = []
random_init.extend(model.feature_extractor.upsample_layers.parameters())
random_init.extend(model.feature_extractor.spp.parameters())
random_init.extend(model.logits.parameters())
#random_init.extend(model.aux_logits.parameters())

optimizer = Adam([{'params':fine_tune, 'lr_factor':4},
                  {'params':random_init, 'lr_factor':1}],
                lr=4e-4, eps=1e-5, weight_decay=1e-4, amsgrad=True)

optimizer.zero_grad()
for epoch in range(0, args.num_epochs):
    lr = 1e-10+(4e-4-1e-10)*(1 + math.cos(epoch/args.num_epochs * math.pi)) / 2

    print('epoch: {}, lr: {}'.format(epoch, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / param_group['lr_factor']

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        print('LR = ', lr)

    model = model.train()
    log_interval = max(len(train_loader) // 10, 1)
    for step, batch in enumerate(train_loader):
        #for img_ind in range(batch['image'].shape[0]):
        #    plt.subplot(1,3,1)
        #    plt.imshow(transform.denormalize(batch['image'][img_ind], (batch['mean'][0][0], batch['mean'][1][0], batch['mean'][2][0]), (batch['std'][0][0], batch['std'][1][0], batch['std'][2][0])))
        #    plt.subplot(1,3,2)
        #    plt.imshow(batch['labels'][img_ind,:,:].numpy())
        #    plt.subplot(1,3,3)
        #    plt.imshow(batch['labels_ood'][img_ind,:,:].numpy())
        #    plt.show()
        #pdb.set_trace()
        optimizer.zero_grad()
        loss = model.get_loss(batch)
        loss.backward()
        optimizer.step()
        if step % log_interval == 0:
            print('step {} / {}, loss: {}'.format(step, len(train_loader), loss.item()))

    torch.save(model.state_dict(), join(
        save_dir, 'params',  'epoch_{}.pt'.format(epoch)))

    with torch.no_grad():
        model = model.eval()
        evaluate_segmentation(model, val_loader)

#with torch.no_grad():
#    model = model.eval()
#    evaluate_segmentation(model, train_eval_loader)


