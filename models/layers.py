from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint as cp

batchnorm_momentum = 0.01

class _DenseLayer(nn.Sequential):
  @staticmethod
  def _checkpoint_function_factory(bn, relu, conv, bn2, relu2, conv2):
    def func(*inputs):
      inputs = torch.cat(inputs, 1)
      return conv2(relu2(bn2(conv(relu(bn(inputs))))))
    return func

  def __init__(self, num_input_features, growth_rate, bn_size, dilation, checkpointing=True):
    super(_DenseLayer, self).__init__()
    self.add_module('norm1', nn.BatchNorm2d(num_input_features, momentum=batchnorm_momentum))
    self.add_module('relu1', nn.ReLU(inplace=True))
    self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                       kernel_size=1, stride=1, bias=False))
    self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate,
                                            momentum=batchnorm_momentum))
    self.add_module('relu2', nn.ReLU(inplace=True))
    self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                       kernel_size=3, stride=1, padding=1*dilation,
                                       bias=False, dilation=dilation))

    self.checkpointing = checkpointing
    if checkpointing:
      self.func = _DenseLayer._checkpoint_function_factory(self.norm1, self.relu1, self.conv1, self.norm2, self.relu2, self.conv2)

  def forward(self, *x):
    if self.checkpointing and self.training:
      out = cp.checkpoint(self.func, x)
    else:
      x = torch.cat(x, 1)
      out = super(_DenseLayer, self).forward(x)
    return out


class _DenseBlock(nn.Sequential):
  def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
               dilation=1, checkpointing=True):
    super(_DenseBlock, self).__init__()
    for i in range(num_layers):
      layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                          bn_size, dilation, checkpointing=checkpointing)
      self.add_module('denselayer%d' % (i + 1), layer)

  def forward(self, x):
    x = [x]
    for layer in self.children():
      x.append(layer(*x))
    return torch.cat(x, 1)


class _Transition(nn.Sequential):
  @staticmethod
  def _checkpoint_function_factory(bn, relu, conv, pool):
    def func(inputs):
      return pool(conv(relu(bn(inputs))))
    return func

  def __init__(self, num_input_features, num_output_features, checkpointing=True):
    super(_Transition, self).__init__()
    self.add_module('norm', nn.BatchNorm2d(num_input_features, momentum=batchnorm_momentum))
    self.add_module('relu', nn.ReLU(inplace=True))
    self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                      kernel_size=1, stride=1, bias=False))
    self.add_module('pool', nn.AvgPool2d(kernel_size=3, stride=2,
                      padding=1, ceil_mode=False, count_include_pad=False))

    self.checkpointing = checkpointing
    if checkpointing:
      self.func = _Transition._checkpoint_function_factory(self.norm, self.relu, self.conv, self.pool)

  def forward(self, x):
    if self.checkpointing and self.training:
      out = cp.checkpoint(self.func, x)
    else:
      out = super(_Transition, self).forward(x)
    return out


class _BNReluConv(nn.Sequential):
  @staticmethod
  def _checkpoint_function_factory(bn, relu, conv):
    def func(inputs):
      return conv(relu(bn(inputs)))
    return func

  def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True,
               bias=False, dilation=1, checkpointing=True):
    super(_BNReluConv, self).__init__()
    if batch_norm:
      self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=batchnorm_momentum))
    self.add_module('relu', nn.ReLU(inplace=True))
    padding = k // 2
    self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                    kernel_size=k, padding=padding, bias=bias, dilation=dilation))

    self.checkpointing = checkpointing
    if checkpointing:
      self.func = _BNReluConv._checkpoint_function_factory(self.norm, self.relu, self.conv)

  def forward (self, x):
    if self.checkpointing and self.training:
      out = cp.checkpoint(self.func, x)
    else:
      out = super(_BNReluConv, self).forward(x)
    return out


class SpatialPyramidPooling(nn.Module):
  def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=256,
               grids=[6,3,2,1], square_grid=False):
    super(SpatialPyramidPooling, self).__init__()
    self.grids = grids
    self.square_grid = square_grid
    self.spp = nn.Sequential()
    self.spp.add_module('spp_bn', _BNReluConv(num_maps_in, bt_size, k=1))
    num_features = bt_size
    final_size = num_features
    for i in range(num_levels):
      final_size += level_size
      self.spp.add_module('spp'+str(i), _BNReluConv(num_features, level_size, k=1))
    self.spp.add_module('spp_fuse', _BNReluConv(final_size, out_size, k=1))

  def forward(self, x):
    levels = []
    target_size = x.size()[2:4]

    ar = target_size[1] / target_size[0]

    x = self.spp[0].forward(x)
    levels.append(x)
    num = len(self.spp) - 1

    # grid_size = (grids[0], round(ar*grids[0]))
    # x = F.adaptive_avg_pool2d(x, grid_size)

    for i in range(1, num):
      if not self.square_grid:
      # grid_size = [grids[i-1], grids[i-1]]
      # grid_size[smaller] = max(1, round(ar*grids[i-1]))
      # grid_size = (grids[i-1], grids[i-1])
        grid_size = (self.grids[i-1], max(1, round(ar*self.grids[i-1])))
        x_pooled = F.adaptive_avg_pool2d(x, grid_size)
      else:
        x_pooled = F.adaptive_avg_pool2d(x, self.grids[i-1])
      level = self.spp[i].forward(x_pooled)

      # print('x =', x.size())
      # print(grid_size)

      # x = F.avg_pool2d(x, kernel_size=2, stride=2, count_include_pad=False, ceil_mode=True)
      # level = spp[i].forward(x)

      level = F.upsample(level, target_size, mode='bilinear')
      levels.append(level)

    # assert x.size()[2] == 1
    x = torch.cat(levels, 1)
    return self.spp[-1].forward(x)


class _Upsample(nn.Module):
  def __init__(self, num_maps_in, skip_maps_in, num_maps_out, num_classes=0):
    super(_Upsample, self).__init__()
    print('Upsample layer: in =', num_maps_in, ', skip =', skip_maps_in,
          ' out =', num_maps_out)
    self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1)
    self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=3)
    self.logits_aux = _BNReluConv(num_maps_in, num_classes, k=1, bias=True)

  def forward(self, x, skip):
    skip = self.bottleneck(skip)
    skip_size = skip.size()[2:4]
    aux = self.logits_aux(x)
    x = F.interpolate(x, skip_size, mode='bilinear', align_corners=False)
    x = x + skip
    x = self.blend_conv(x)
    return x, aux


class DenseNet(nn.Module):
  def __init__(self, args, growth_rate=32, block_config=(6, 12, 32, 32),
               num_init_features=64, bn_size=4):

    super(DenseNet, self).__init__()
    self.block_config = block_config
    self.growth_rate = growth_rate
    args.last_block_pooling = 2**5

    self.features = nn.Sequential(OrderedDict([
        ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                            padding=3, bias=False)),
        ('norm0', nn.BatchNorm2d(num_init_features, momentum=batchnorm_momentum)),
        ('relu0', nn.ReLU(inplace=True)),
        ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
    ]))
    self.first_block_idx = len(self.features)

    dilations = [1, 1, 1, 1]
    num_features = num_init_features
    self.skip_sizes = []

    for i, num_layers in enumerate(block_config):
      block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                          bn_size=bn_size, growth_rate=growth_rate,
                          dilation=dilations[i])
      self.features.add_module('denseblock%d' % (i + 1), block)
      num_features = num_features + num_layers * growth_rate
      if i != len(block_config) - 1:
        self.skip_sizes.append(num_features)
        trans = _Transition(num_input_features=num_features,
                            num_output_features=num_features // 2)
        self.features.add_module('transition%d' % (i + 1), trans)
        num_features = num_features // 2

    self.num_features = num_features

  def forward(self, x, target_size=None):
    skip_layers = []
    if target_size == None:
      target_size = x.size()[2:4]
    for i in range(self.first_block_idx+1):
      x = self.features[i].forward(x)

    for i in range(self.first_block_idx+1, self.first_block_idx+6, 2):
      if len(self.features[i]) > 3 and self.features[i][3].stride > 1:
        skip_layers.append(x)
      x = self.features[i].forward(x)
      x = self.features[i+1].forward(x)

    return x, skip_layers


class Ladder(nn.Module):
  def __init__(self, args, num_classes=19):
    super(Ladder, self).__init__()
    self.num_classes = num_classes
    self.backbone = DenseNet(args)

    self.upsample_layers = nn.Sequential()
    spp_square_grid = False
    spp_grids = [8,4,2,1]
    num_levels = 4
    args.last_block_pooling = 2**5
    up_sizes = [256, 256, 128]

    num_features = self.backbone.num_features

    self.spp_size = 512
    level_size = self.spp_size // num_levels
    bt_size = self.spp_size
    self.spp = SpatialPyramidPooling(num_features, num_levels, bt_size, level_size,
        self.spp_size, spp_grids, spp_square_grid)
    num_features = self.spp_size

    assert len(up_sizes) == len(self.backbone.skip_sizes)
    for i in range(len(self.backbone.skip_sizes)):
      upsample = _Upsample(num_features, self.backbone.skip_sizes[-1-i], up_sizes[i],
                           num_classes=self.num_classes)
      num_features = up_sizes[i]
      self.upsample_layers.add_module('upsample_'+str(i), upsample)

    self.num_features = num_features

  def forward(self, x, target_size=None):
    x, skip_layers = self.backbone.forward(x)

    x = self.spp(x)

    for i, skip in enumerate(reversed(skip_layers)):
      x, _ = self.upsample_layers[i].forward(x, skip)

    return x
