""" Defination of Self-Attentioend RNN (SAR) Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
cfg = config.Config()

class SAR(nn.Module):
  def __init__(self):
    super(SAR, self).__init__()
    # hyper-params
    self.num_chn = cfg.num_chn
    self.step_len = int(cfg.rnn_step_len * cfg.samp_rate)
    self.step_stride = int(cfg.rnn_step_stride * cfg.samp_rate)
    self.cnn_num_layers = cfg.cnn_num_layers
    self.cnn_num_kernels = cfg.cnn_num_kernels
    self.cnn_kernel_size = cfg.cnn_kernel_size
    self.rnn_hidden_size = cfg.rnn_hidden_size
    self.rnn_num_layers = cfg.rnn_num_layers
    self.num_att_heads = cfg.num_att_heads
    # def layers
    self.cnn_layers = self._make_cnn_layers()
    self.gru_layer = nn.GRU(input_size=self.step_len * self.num_chn, 
        hidden_size=self.rnn_hidden_size, 
        num_layers=self.rnn_num_layers, 
        bidirectional=True, batch_first=True)
    self.attention = nn.MultiheadAttention(embed_dim=2*self.rnn_hidden_size, 
        num_heads=self.num_att_heads, 
        batch_first=True)
    self.fc_layer = nn.Linear(2*self.rnn_hidden_size, 3)

  def forward(self, x):
    # CNN for local feature extraction
    x = self.cnn_layers(x)  # batch_size * num_chn * win_len
    x = x.unfold(2, self.step_len, self.step_stride).permute(0,2,1,3)
    x = x.reshape(x.size(0), x.size(1), -1)  # batch_size * num_step * (num_chn*step_len)
    # Self-Attentioend RNN for long-term dependency
    x, _ = self.gru_layer(x)
    x, _ = self.attention(query=x, key=x, value=x) # self attention
    return self.fc_layer(x)

  def _make_cnn_layers(self):
    layers = []
    for ii in range(self.cnn_num_layers):
        in_channels = self.num_chn if ii==0 else self.cnn_num_kernels 
        out_channels = self.cnn_num_kernels
        kernel_size = self.cnn_kernel_size
        layers += [nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)]
    layers += [nn.Conv1d(in_channels=out_channels, out_channels=self.num_chn, kernel_size=1)]
    return nn.Sequential(*layers)
