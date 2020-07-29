import torch
import torch.nn as nn
import torch.nn.functional as F
import models.model_utils as model_utils
import models.layers as layers
import models.losses as losses
import pdb

def build(pretrained=False, **kwargs):
  model = Base(**kwargs)
  if pretrained:
      params = model_utils._load_imagenet_weights()
      model.load_state_dict(params, convert=True)
  return model


class Base(model_utils.DNN):
  def __init__(self, args, num_classes=19, ignore_id=None):
    super(Base, self).__init__()

    self.num_classes = num_classes
    self.ignore_id = num_classes if ignore_id is None else ignore_id

    self.feature_extractor = layers.Ladder(args, num_classes=num_classes)
    num_features = self.feature_extractor.num_features
    self.logits = layers._BNReluConv(num_features, self.num_classes, k=1, bias=True)

    self.aux_logits = nn.Sequential()
    for i in range(len(self.feature_extractor.backbone.skip_sizes)):
        self.aux_logits.add_module('upsample_' + str(i),
                                   layers._BNReluConv(self.feature_extractor.upsample_layers[i].num_maps_in,
                                                      self.num_classes, k=1, bias=True))
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    nn.init.xavier_normal_(self.logits[-1].weight.data)

  def forward(self, x, target_size=None):
    if target_size == None:
      target_size = x.size()[2:4]

    x, aux = self.feature_extractor.forward(x)

    x = self.logits(x)

    self.aux_out = []
    for i in range(len(self.aux_logits)):
        self.aux_out.append(self.aux_logits[i](aux[i]))

    self.out = F.upsample(x, target_size, mode='bilinear', align_corners=False)

    pred = F.softmax(self.out.detach(), dim=1)
    pred_conf = pred.max(dim=1)[0]

    return pred, pred_conf


  def get_loss(self, batch):
    x = batch['image'].cuda(non_blocking=True)
    labels = batch['labels'].cuda(non_blocking=True)

    self.forward(x)

    log_softmax = F.log_softmax(self.out, dim=1)
    main_loss = F.nll_loss(log_softmax, labels, ignore_index=self.ignore_id)

    aux_loss = []
    for aux_out in self.aux_out:
      aux_loss.append(losses.get_aux_loss(aux_out, labels, self.num_classes, average=False))
    aux_loss = torch.mean(torch.cat(aux_loss, dim=0))

    loss = 0.6 * main_loss + 0.4 * aux_loss

    return loss


  def load_state_dict(self, state_dict, convert=False):
    new_state_dict = {}
    for name, param in state_dict.items():
      if convert:
        name = name.replace('features', 'backbone.features')
        if 'aux_logits' in name:
          name = name.replace('upsample_layers.','').replace('aux_logits.','').replace('upsample','aux_logits.upsample')
        if 'logits'  not in name:
          name = 'feature_extractor.' + name

      new_state_dict[name] = param

    super(Base, self).load_state_dict(new_state_dict)

