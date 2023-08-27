""" Defination of CNN & RNN Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
cfg = config.Config()

class ResidualBlock(nn.Module):
  """ Simple residual block that is used in ResNet-18
  """
  def __init__(self, in_channels, num_kernels, stride=1):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv1d(in_channels, num_kernels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.conv2 = nn.Conv1d(num_kernels, num_kernels, kernel_size=3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm1d(num_kernels)
    self.bn2 = nn.BatchNorm1d(num_kernels)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = None
    if stride!=1: 
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, num_kernels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(num_kernels))

  def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out += identity
    out = self.relu(out)
    return out

class CNN(nn.Module):
  """ CNN model for Event detection
  """
  def __init__(self):
    super(CNN, self).__init__()
    # hyper-params
    self.in_channels = cfg.num_chn
    self.win_len = int(cfg.win_len * cfg.samp_rate)
    self.num_kernels = cfg.num_cnn_kernels
    self.stem_kernel_size = cfg.cnn_stem_kernel_size
    self.stem_conv_stride = cfg.cnn_stem_conv_stride
    self.num_res_blocks = cfg.num_cnn_res_blocks
    self.out_classes = cfg.cnn_out_classes
    # def layers
    resblock = ResidualBlock
    self.stem_layers = self._make_stem_layers(self.stem_kernel_size, self.stem_conv_stride)
    self.resnet_layers = self._make_resnet_layers(resblock, self.num_res_blocks)
    self.avgpool = nn.AdaptiveAvgPool1d(1)
    self.fc_layer = nn.Linear(self.num_kernels*2**(len(self.num_res_blocks)-1), self.out_classes)

  def forward(self, x):
    x = self.stem_layers(x)
    x = self.resnet_layers(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.fc_layer(x)

  def _make_stem_layers(self, kernel_size, conv_stride):
    layers = [nn.Conv1d(self.in_channels, self.num_kernels, kernel_size[0], stride=conv_stride[0], bias=False),
              nn.BatchNorm1d(self.num_kernels),
              nn.ReLU(inplace=True)]
    for i in range(1,len(kernel_size)):
        layers += [nn.Conv1d(self.num_kernels, self.num_kernels, kernel_size[i], stride=conv_stride[i], bias=False),
                   nn.BatchNorm1d(self.num_kernels),
                   nn.ReLU(inplace=True)]
    layers += [nn.MaxPool1d(kernel_size=3, stride=2, padding=1)]
    return nn.Sequential(*layers)

  def _make_resnet_layers(self, block, num_res_blocks):
    layers = [] 
    for i,num_res_block in enumerate(num_res_blocks):
      for j in range(num_res_block):
        stride = 2 if j==0 and i!=0 else 1
        in_channels = self.num_kernels*2**(i-1) if i>0 and j==0 else self.num_kernels*2**i
        out_channels = self.num_kernels*2**i
        layers += [block(in_channels, out_channels, stride=stride)]
    return nn.Sequential(*layers)


class RNN(nn.Module):
  """ RNN model for Phase picking
  """
  def __init__(self):
    super(RNN, self).__init__()
    # hyper-params
    self.input_size = int(cfg.step_len * cfg.num_chn * cfg.samp_rate)
    self.hidden_size = cfg.rnn_hidden_size
    self.num_layers = cfg.num_rnn_layers
    # def layers
    self.gru_layer = nn.GRU(input_size=self.input_size, 
        hidden_size=self.hidden_size, 
        num_layers=self.num_layers, 
        batch_first=True, bidirectional=True)
    self.fc_layer = nn.Linear(2*self.hidden_size, 3)

  def forward(self, x):
    x, _ = self.gru_layer(x)
    return self.fc_layer(x)

