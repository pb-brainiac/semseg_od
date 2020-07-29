import torch
import torch.nn.functional as F
import pdb


def get_aux_loss(logits, labels, num_classes, average=True):
  factor = labels.shape[1] // logits.shape[2]
  labels_ni = labels.clone()
  labels_ni[labels>=num_classes] = num_classes

  num, nc, height, width = logits.shape

  labels_4d = labels_ni.reshape(num, height, factor, width, factor)
  labels_oh = torch.eye(num_classes+1, dtype=torch.float32).cuda()[labels_4d]

  target_dist = labels_oh.sum((2, 4)) / factor**2

  target_dist = target_dist.reshape(-1, num_classes+1)
  valid_mask = (target_dist[:,-1] < 0.5).unsqueeze(1)

  target_dist = target_dist[:,:-1]
  dist_sum = target_dist.sum(1, keepdims=True)
  dist_sum[dist_sum==0] = 1
  target_dist /= dist_sum
  target_dist = target_dist.masked_select(valid_mask).reshape(-1, num_classes)

  logits_flattened = logits.permute(0,2,3,1).reshape(-1,num_classes).masked_select(valid_mask)
  logits_flattened = logits_flattened.reshape(-1, num_classes)

  log_softmax = F.log_softmax(logits_flattened, dim=1)

  aux_loss = -torch.sum(target_dist*log_softmax, dim=1)

  if average:
    aux_loss = torch.mean(aux_loss)

  return aux_loss

