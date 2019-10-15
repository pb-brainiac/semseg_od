import torch
import torch.nn as nn
import torch.nn.functional as F

import models.layers as layers

def build(**kwargs):
  return TwoHead(**kwargs)


class TwoHead(nn.Module):
  def __init__(self, args, num_classes=19):
    super(TwoHead, self).__init__()

    self.num_classes = num_classes

    self.feature_extractor = layers.Ladder(args, num_classes=num_classes)
    num_features = self.feature_extractor.num_features
    self.logits = layers._BNReluConv(num_features, self.num_classes, k=1, bias=True)
    self.logits_conf = layers._BNReluConv(num_features, 2, k=1, bias=True)


  def forward(self, x, target_size=None):
    if target_size == None:
      target_size = x.size()[2:4]

    x = self.feature_extractor.forward(x)

    x_conf = self.logits_conf(x)
    x = self.logits(x)

    x = F.upsample(x, target_size, mode='bilinear', align_corners=False)
    x_conf = F.upsample(x_conf, target_size, mode='bilinear', align_corners=False)

    x = F.softmax(x, dim=1)
    x_conf = F.softmax(x_conf, dim=1)[:,0,:,:]

    return x, x_conf


  def load_state_dict(self, state_dict):
      own_state = self.state_dict()
      state_dict_keys = []
      for name, param in state_dict.items():
          name = name.replace('features', 'backbone.features')
          name = name.replace('aux_logits','logits_aux')
          if 'logits_conf' not in name and 'logits.' not in name:
              name = 'feature_extractor.' + name
          state_dict_keys.append(name)
          if name not in own_state:
              print('Variable "{}" missing in model'.format(name))
              continue
          if isinstance(param, torch.nn.parameter.Parameter):
              # backwards compatibility for serialized parameters
              param = param.data
          try:
              own_state[name].copy_(param)
          except:
              print('While copying the parameter named {}, whose dimensions in the model are'
                    ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                        name, own_state[name].size(), param.size()))
              raise

      missing = set(own_state.keys()) - set(state_dict_keys)
      if len(missing) > 0:
          print('missing keys in state_dict: "{}"'.format(missing))

