import torch
import numpy as np
import pdb
import torch
import torch.nn.functional as F


def odin(img, model, T=1, step=0.0005, num_steps=1, target_size=None):
    img_odin = img
    for i in range(num_steps):
        model(img_odin, target_size)
        logits = model.out.to('cuda:1')
        softmax = F.softmax(logits/T, dim=1)
        softmax_max_T = softmax.max(1)[0]
        loss_image = torch.mean(-torch.log(softmax_max_T))
        grad = torch.autograd.grad(loss_image, img_odin)[0]
        grad_abs = grad.abs().clamp(1e-10)
        img_odin = img_odin - step * (grad / grad_abs)
    return img_odin.to('cuda:0')

def get_img_conf_mat(pred, true, size):
    eye_matrix = torch.eye(size[0]).cuda()
    pred_OH = eye_matrix[pred].to(torch.float64)
    true_OH = eye_matrix[true].to(torch.float64)
    return torch.einsum("ni,nj->ij", true_OH, pred_OH).to(torch.int64)

def compute_errors(conf_mat, name, class_info, verbose=False, nc=None):
    num_classes = nc if nc is not None else model.num_classes
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

def segment_image(model, batch, args, conf_mats, ood_id=-1, nc=None):
    num_classes = nc if nc is not None else model.num_classes
    img = torch.autograd.Variable(batch['image'].cuda(
            non_blocking=True), requires_grad=True)

    if args.odin:
        img = odin(img, model, T=args.odin_T, step=args.odin_step, target_size=batch['image'].shape[2:])

    with torch.no_grad():
        probs, conf_probs = model.predictions(img, batch['image'].shape[2:])
        pred = probs.max(dim=1)[1][0]
        pred_w_outlier = pred.clone()
        pred_w_outlier[conf_probs[0] < 0.5] = ood_id

        if batch['labels'] is not None:
            labels = batch['labels'][0].cuda()
            conf_mats['seg'] += get_img_conf_mat(pred[labels<num_classes], labels[labels<num_classes], conf_mats['seg'].shape)
            conf_mats['seg_w_outlier'] += get_img_conf_mat(pred_w_outlier[labels<num_classes], labels[labels<num_classes], conf_mats['seg_w_outlier'].shape)

    return pred, pred_w_outlier, conf_probs
