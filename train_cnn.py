""" Training CNN for Event detection
"""
import os, time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import torch.multiprocessing as mp
from dataset import Events
from models import CNN 
import config
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

def main():
  # set defaults
  torch.backends.cudnn.benchmark = True
  # i/o paths
  ckpt_dir = os.path.join(args.ckpt_dir,'CNN')
  if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
  # training params
  cfg = config.Config()
  lr = cfg.cnn_lr
  num_epochs = cfg.cnn_num_epochs
  summary_step = cfg.cnn_summary_step
  ckpt_step = cfg.cnn_ckpt_step
  num_classes = cfg.cnn_out_classes
  # set data loader
  batch_size = cfg.cnn_batch_size
  train_set = Events(args.zarr_path, 'train')
  valid_set = Events(args.zarr_path, 'valid')
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
  valid_sampler = BatchSampler(RandomSampler(valid_set, replacement=True), batch_size=batch_size, drop_last=False)
  valid_loader = DataLoader(valid_set, batch_sampler=valid_sampler, pin_memory=True)
  num_batch = len(train_loader)
  # import model
  model = CNN() 
  device = torch.device("cuda:%s"%args.gpu_idx)
  model.to(device)
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
        train_acc_list, train_loss = train_step(model, data, target, criterion, optimizer, num_classes)
        # save model
        if global_step % ckpt_step == 0: 
            torch.save(model.state_dict(), os.path.join(ckpt_dir,'%s_%s-%s.ckpt'%(global_step, epoch_idx, iter_idx)))
        # valid & print summary
        if global_step % summary_step != 0: continue
        for (data, target) in valid_loader:
            # to cuda & reshape data
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            data, target = _reshape_data_target(data, target)
            valid_acc_list, valid_loss = valid_step(model, data, target, criterion, num_classes)
            break
        # visualization
        print('step {} ({}/{}) | train loss {:.2f} | valid loss {:.2f} | {:.2f}s'.format(global_step, iter_idx, epoch_idx, train_loss, valid_loss, time.time()-t))
        acc_to_print = ''
        for ii in range(num_classes):
            acc_to_print += ' {} acc. {:.2f}% {:.2f}% |'.format(['earthquake','noise','glitch'][ii], 100*train_acc_list[ii], 100*valid_acc_list[ii])
        print('   %s'%acc_to_print[:-2])
        sum_loss = {'train_loss': train_loss, 'valid_loss':valid_loss}
        sum_acc_pos = {'train_acc_pos':100*train_acc_list[0], 'valid_acc_pos':100*valid_acc_list[0]}
        sum_acc_neg = {'train_acc_neg':100*train_acc_list[1], 'valid_acc_neg':100*valid_acc_list[1]}
        if num_classes==3: 
            sum_acc_glitch = {'train_acc_glitch':100*train_acc_list[2], 'valid_acc_glitch':100*valid_acc_list[2]}
        with SummaryWriter(log_dir=ckpt_dir) as writer:
            writer.add_scalars('loss', sum_loss, global_step)
            writer.add_scalars('neg_acc', sum_acc_neg, global_step)
            writer.add_scalars('pos_acc', sum_acc_pos, global_step)
            if num_classes==3: writer.add_scalars('glitch_acc', sum_acc_glitch, global_step)


# train one batch
def train_step(model, data, target, criterion, optimizer, num_classes):
    model.train()
    num_pairs = int(target.size(0)/num_classes)
    # prediction
    pred_logits = model(data)
    pred_class = torch.argmax(pred_logits,1)
    loss = criterion(pred_logits, target)
    acc_list = [] # P/N or P/N/G
    for ii in range(num_classes):
        acc_list.append(pred_class[ii*num_pairs:(ii+1)*num_pairs].eq(target[ii*num_pairs:(ii+1)*num_pairs]).sum() / float(num_pairs))
    # update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return [acc.item() for acc in acc_list], loss.item()

# valid one batch
def valid_step(model, data, target, criterion, num_classes):
    model.eval()
    num_pairs = int(target.size(0)/num_classes)
    # prediction
    pred_logits = model(data)
    pred_class = torch.argmax(pred_logits,1)
    loss = criterion(pred_logits, target)
    acc_list = [] # P/N or P/N/G
    for ii in range(num_classes):
        acc_list.append(pred_class[ii*num_pairs:(ii+1)*num_pairs].eq(target[ii*num_pairs:(ii+1)*num_pairs]).sum() / float(num_pairs))
    return [acc.item() for acc in acc_list], loss.item()

# reshape data: [batch_size * num_class] * num_chn * win_len
def _reshape_data_target(data, target):
    data = data.transpose(0,1)
    target = target.transpose(0,1)
    data = data.reshape(data.size(0)*data.size(1), data.size(2), data.size(3))
    target = target.reshape(target.size(0)*target.size(1))
    return data, target


if __name__ == '__main__':
  mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu_idx', type=int, default=0)
  parser.add_argument('--num_workers', type=int)
  parser.add_argument('--zarr_path', type=str)
  parser.add_argument('--ckpt_dir', type=str, default='output/eg_ckpt')
  args = parser.parse_args()
  torch.cuda.set_device(int(args.gpu_idx)) 
  main()

