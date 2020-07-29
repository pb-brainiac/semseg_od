import re
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch


def _load_imagenet_weights():
  orig_params = model_zoo.load_url('https://download.pytorch.org/models/densenet169-b2777c0a.pth')

  delete_keys = ['features.norm5.weight', 'features.norm5.bias',
                 'features.norm5.running_mean', 'features.norm5.running_var',
                  'classifier.weight', 'classifier.bias']
  for key in delete_keys:
    del orig_params[key]

  params = {}
  pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
  for key in orig_params.keys():
    res = pattern.match(key)
    if res:
      new_key = res.group(1) + res.group(2)
      params[new_key] = orig_params[key]
    else:
      params[key] = orig_params[key]

  return params


class DNN(nn.Module):
  def __init__(self):
    super(DNN, self).__init__()

  def load_state_dict(self, state_dict):
    own_state = self.state_dict()
    state_dict_keys = []
    for name, param in state_dict.items():
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

