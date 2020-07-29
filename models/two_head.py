import torch
import torch.nn as nn
import torch.nn.functional as F

import models.model_utils as model_utils
import models.layers as layers

def build(pretrained=False,**kwargs):
  model = TwoHead(**kwargs)
  if pretrained:
      params = model_utils._load_imagenet_weights()
      model.load_state_dict(params)
  return model


class TwoHead(model_utils.DNN):
  def __init__(self, args, num_classes=19):
    super(TwoHead, self).__init__()

    self.num_classes = num_classes

    self.feature_extractor = layers.Ladder(args, num_classes=num_classes)
    num_features = self.feature_extractor.num_features
    self.logits = layers._BNReluConv(num_features, self.num_classes, k=1, bias=True)
    self.logits_conf = layers._BNReluConv(num_features, 2, k=1, bias=True)

    self.aux_logits = nn.Sequential()
    for i in range(len(self.feature_extractor.backbone.skip_sizes)):
        self.aux_logits.add_module('upsample_' + str(i),
                                   layers._BNReluConv(self.feature_extractor.upsample_layers[i].num_maps_in,
                                                      self.num_classes, k=1, bias=True))


  def forward(self, x, target_size=None):
    if target_size == None:
      target_size = x.size()[2:4]

    x, aux = self.feature_extractor.forward(x)

    x_conf = self.logits_conf(x)
    x = self.logits(x)

    self.aux_out = []
    for i in range(len(self.aux_logits)):
        self.aux_out.append(self.aux_logits[i](aux[i]))

    self.out = F.upsample(x, target_size, mode='bilinear', align_corners=False)
    self.out_conf = F.upsample(x_conf, target_size, mode='bilinear', align_corners=False)

    pred = F.softmax(self.out, dim=1)
    pred_conf = F.softmax(self.out_conf, dim=1)[:,0,:,:]

    return pred, pred_conf


  def load_state_dict(self, state_dict):
    new_state_dict = {}
    for name, param in state_dict.items():
      name = name.replace('features', 'backbone.features')
      if 'aux_logits' in name:
        name = name.replace('upsample_layers.','').replace('aux_logits.','').replace('upsample','aux_logits.upsample')
      if 'logits_conf' not in name and 'logits.' not in name:
        name = 'feature_extractor.' + name

      new_state_dict[name] = param

    super(TwoHead, self).load_state_dict(new_state_dict)

