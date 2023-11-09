""" Train SAR with both Positive and Negative samples
"""
import os, time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import torch.multiprocessing as mp
from dataset import Positive_Negative
from models import RSeL
import config
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

def main():
    torch.backends.cudnn.benchmark = True
    # training params
    cfg = config.Config()
    lr = cfg.lr
    neg_ratio = cfg.neg_ratio
    num_epochs = cfg.num_epochs
    summary_step = cfg.summary_step
    ckpt_step = cfg.ckpt_step
    batch_size = cfg.batch_size
    # model config
    num_steps = cfg.rnn_num_steps
    step_stride = cfg.rnn_step_stride
    # set data loader
    train_set = Positive_Negative(args.hdf5_path, 'train')
    valid_set = Positive_Negative(args.hdf5_path, 'valid')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_sampler = BatchSampler(RandomSampler(valid_set, replacement=True), batch_size=batch_size, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_sampler=valid_sampler, pin_memory=True)
    num_batch = len(train_loader)
    # import model
    model = RSeL() 
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
        train_acc_list, train_loss = train_step(model, data, target, neg_ratio, criterion, optimizer)
        # save model
        if global_step % ckpt_step == 0: 
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir,'%s_%s-%s.ckpt'%(global_step, epoch_idx, iter_idx)))
        # valid & print summary
        if global_step % summary_step != 0: continue
        for (data, target) in valid_loader:
            # to cuda & reshape data
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            data, target = _reshape_data_target(data, target)
            valid_acc_list, valid_loss = valid_step(model, data, target, criterion)
            break
        # visualization
        print('step {} ({}/{}) | train loss {:.2f} | valid loss {:.2f} | {:.2f}s'.format(global_step, iter_idx, epoch_idx, train_loss, valid_loss, time.time()-t))
        acc_to_print = ''
        for ii in range(3):
            acc_to_print += ' {} acc. {:.2f}% {:.2f}% |'.format(['seq','pos','neg'][ii], 100*train_acc_list[ii], 100*valid_acc_list[ii])
        print('   %s'%acc_to_print[:-2])
        sum_loss = {'train_loss': train_loss, 'valid_loss':valid_loss}
        sum_acc_seq = {'train_acc_seq':100*train_acc_list[0], 'valid_acc_pos':100*valid_acc_list[0]}
        sum_acc_pos = {'train_acc_pos':100*train_acc_list[1], 'valid_acc_pos':100*valid_acc_list[1]}
        sum_acc_neg = {'train_acc_neg':100*train_acc_list[2], 'valid_acc_neg':100*valid_acc_list[2]}
        with SummaryWriter(log_dir=args.ckpt_dir) as writer:
            writer.add_scalars('loss', sum_loss, global_step)
            writer.add_scalars('seq_acc', sum_acc_seq, global_step)
            writer.add_scalars('pos_acc', sum_acc_pos, global_step)
            writer.add_scalars('neg_acc', sum_acc_neg, global_step)


# train one batch
def train_step(model, data, target, neg_ratio, criterion, optimizer):
    model.train()
    bs = int(target.size(0)/2)
    num_pos, num_neg = bs, int(bs*neg_ratio)
    data = data[0:num_pos+num_neg]
    target = target[0:num_pos+num_neg]
    # model prediction
    pred_logits = model(data) # batch_size * num_step * 3
    pred_class = torch.argmax(pred_logits,2) # batch_size * num_step
    loss = criterion(pred_logits.view(-1,3), target.view(-1))
    # get accuracy
    acc_list = [] 
    for ii in range(2):
        pred_i = pred_class[ii*bs : (ii+1)*bs]
        num_pos_pred = sum((pred_i==1).any(dim=1) * (pred_i==2).any(dim=1))
        tar_pos = target[ii*bs : (ii+1)*bs].view(-1)
        acc_seq = pred_i.view(-1).eq(tar_pos).sum() / float(tar_pos.size(0))
        if ii==0: acc_list += [acc_seq, num_pos_pred/float(num_pos)]
        if ii==1: acc_list += [1 - num_pos_pred/float(num_neg)]
    # update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return [acc.item() for acc in acc_list], loss.item()

# valid one batch
def valid_step(model, data, target, criterion):
    model.eval()
    bs = int(target.size(0)/2)
    # model prediction
    pred_logits = model(data) # batch_size * num_step * 3
    pred_class = torch.argmax(pred_logits,2) # batch_size * num_step
    loss = criterion(pred_logits.view(-1,3), target.view(-1))
    # get accuracy
    acc_list = [] 
    for ii in range(2):
        pred_i = pred_class[ii*bs : (ii+1)*bs]
        num_pos_pred = sum((pred_i==1).any(dim=1) * (pred_i==2).any(dim=1))
        tar_pos = target[ii*bs : (ii+1)*bs].view(-1)
        acc_seq = pred_i.view(-1).eq(tar_pos).sum() / float(tar_pos.size(0))
        if ii==0: acc_list += [acc_seq, num_pos_pred/float(bs)]
        if ii==1: acc_list += [1 - num_pos_pred/float(bs)]
    return [acc.item() for acc in acc_list], loss.item()

# reshape data: [batch_size * 2] * num_step * [num_chn * win_len], 2 for pos & neg
def _reshape_data_target(data, target):
    data = data.transpose(0,1)
    target = target.transpose(0,1)
    data = data.reshape(data.size(0)*data.size(1), *data.shape[2:])
    target = target.reshape(target.size(0)*target.size(1), *target.shape[2:])
    return data, target


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--hdf5_path', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_idx) 
    if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
    main()
