""" Training DetNet (CNN)
"""
import os, time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import torch.multiprocessing as mp
from dataset import Events
from models import DetNet
import config
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

def main():
  # set defaults
  torch.backends.cudnn.benchmark = True
  # i/o paths
  ckpt_dir = args.ckpt_dir
  if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
  # training params
  cfg = config.Config()
  lr = cfg.cnn_lr
  num_epochs = cfg.cnn_num_epochs
  summary_step = cfg.cnn_summary_step
  ckpt_step = cfg.cnn_ckpt_step
  to_init = cfg.to_init_cnn
  # set data loader
  batch_size = cfg.cnn_batch_size
  num_workers = cfg.cnn_num_workers
  train_set = Events(args.zarr_path, 'train')
  valid_set = Events(args.zarr_path, 'valid')
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
  valid_sampler = BatchSampler(RandomSampler(valid_set, replacement=True), batch_size=batch_size, drop_last=False)
  valid_loader = DataLoader(valid_set, batch_sampler=valid_sampler, pin_memory=True)
  num_batch = len(train_loader)

  # import model
  model = DetNet()
  if to_init: model.apply(init_weights)
  device = torch.device("cuda")
  model.to(device)
#  model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
  # loss & optim
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)

  # train loop
  t = time.time()
  for epoch_idx in range(num_epochs):
    for iter_idx, (data, target) in enumerate(train_loader):
        global_step = num_batch * epoch_idx + iter_idx
        # to cuda & reshape data
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        data, target = _reshape_data_target(data, target)
        train_acc_neg, train_acc_pos, train_loss = train_step(model, data, target, criterion, optimizer)
        # save model
        if global_step % ckpt_step == 0: 
            torch.save(model.state_dict(), os.path.join(ckpt_dir,'%s_%s-%s.ckpt'%(global_step, epoch_idx, iter_idx)))
        # valid & print summary
        if global_step % summary_step != 0: continue
        for (data, target) in valid_loader:
            # to cuda & reshape data
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            data, target = _reshape_data_target(data, target)
            valid_acc_neg, valid_acc_pos, valid_loss = valid_step(model, data, target, criterion)
            break
        # visualization
        print('step {} ({}/{}) | train loss {:.2f} | valid loss {:.2f} | {:.2f}s'\
            .format(global_step, iter_idx, epoch_idx, train_loss, valid_loss, time.time()-t))
        print('    noise acc. {:.2f}% {:.2f}% | earthquake acc. {:.2f}% {:.2f}%'\
            .format(100*train_acc_neg, 100*valid_acc_neg, 100*train_acc_pos, 100*valid_acc_pos))
        sum_loss = {'train_loss': train_loss, 'valid_loss':valid_loss}
        sum_acc_neg = {'train_acc_neg':100*train_acc_neg, 'valid_acc_neg':100*valid_acc_neg}
        sum_acc_pos = {'train_acc_pos':100*train_acc_pos, 'valid_acc_pos':100*valid_acc_pos}
        with SummaryWriter(log_dir=ckpt_dir) as writer:
            writer.add_scalars('loss', sum_loss, global_step)
            writer.add_scalars('neg_acc', sum_acc_neg, global_step)
            writer.add_scalars('pos_acc', sum_acc_pos, global_step)



# train one batch
def train_step(model, data, target, criterion, optimizer):
    model.train()
    num_pairs = int(target.size(0)/2)
    # prediction
    pred_logits = model(data)
    pred_class = torch.argmax(pred_logits,1)
    loss = criterion(pred_logits, target)
    acc_neg = pred_class[0:num_pairs].eq(target[0:num_pairs]).sum() / float(num_pairs)
    acc_pos = pred_class[num_pairs:].eq(target[num_pairs:]).sum() / float(num_pairs)
    # update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return acc_neg.item(), acc_pos.item(), loss.item()


# valid one batch
def valid_step(model, data, target, criterion):
    model.eval()
    num_pairs = int(target.size(0)/2)
    # prediction
    pred_logits = model(data)
    pred_class = torch.argmax(pred_logits,1)
    loss = criterion(pred_logits, target)
    acc_neg = pred_class[0:num_pairs].eq(target[0:num_pairs]).sum() / float(num_pairs)
    acc_pos = pred_class[num_pairs:].eq(target[num_pairs:]).sum() / float(num_pairs)
    return acc_neg.item(), acc_pos.item(), loss.item()


# reshape data: [batch_size * 2 (neg,pos)] * num_chn * win_len
def _reshape_data_target(data, target):
    data = data.transpose(0,1)
    target = target.transpose(0,1)
    data = data.reshape(data.size(1)*2, data.size(2), data.size(3))
    target = target.reshape(target.size(1)*2)
    return data, target


# initialize weights of model
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
  mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu_idx', type=str, default="0")
  parser.add_argument('--zarr_path', type=str)
  parser.add_argument('--ckpt_dir', type=str)
  parser.add_argument('--resume', default=False)
  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
  main()

