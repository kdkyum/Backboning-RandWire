import argparse
import os
import random
import socket
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import loader.data_loader as data_loader
import model.net as net
from tensorboardX import SummaryWriter
from thop import clever_format, profile
from trainer.trainer import Trainer
from utils.misc import Node_OP, count_node_ops, load_state, save_result
from utils.slurm import init_distributed_mode
from utils.smoothing import LabelSmoothing

datasets = sorted(name for name in data_loader.__dict__
                  if name.islower() and callable(data_loader.__dict__[name]))
model_names = sorted(name for name in net.__dict__
                     if name.islower() and not name.startswith("_")
                     and callable(net.__dict__[name]))

parser = argparse.ArgumentParser(description='Pytorch lottery ticket random wire training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='complete',
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: complete)')
parser.add_argument('--model-dir', metavar='DIR',
                    help='path to model checkpoint director')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='name of dataset (default: cifar10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--cutout', action='store_true',
                    help='cutout data augmentation')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--train-portion',
                    type=float, default=1.0, help='portion of training data')
parser.add_argument('--flops', action='store_true', default=False,
                    help='Calcuate FLOPs of the model.')

param_parser = parser.add_argument_group(title='Hyperparameters')
param_parser.add_argument('--epochs', default=200, type=int, metavar='N',
                          help='number of total epochs to run')
param_parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                          help='manual epoch number (useful on restarts)')
param_parser.add_argument('-b', '--batch-size', default=256, type=int,
                          metavar='N',
                          help='mini-batch size (default: 256), this is the total '
                          'batch size of all GPUs on the current node when '
                          'using Data Parallel or Distributed Data Parallel')
param_parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                          metavar='LR', help='initial learning rate', dest='lr')
param_parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                          help='momentum')
param_parser.add_argument('--wd', '--weight-decay', default=3e-4, type=float,
                          metavar='W', help='weight decay (default: 3e-4)',
                          dest='weight_decay')
param_parser.add_argument('--graph-wd', default=0, type=float,
                          metavar='W', help='graph edge weight decay (default: 0)')
param_parser.add_argument('--drop-path', default=0, type=float,
                          metavar='DR', help='drop path rate')
param_parser.add_argument('--noScheduleDrop', action='store_true', default=False,
                          help='scheduled drop path out')
param_parser.add_argument('--label-smooth', action='store_true', default=False,
                          help='label smoothing')
param_parser.add_argument('--init_norm', action='store_true', default=False, help='normalized initialization')
param_parser.add_argument('--scratch', action='store_true', default=False,
                          help='training from scratch')
param_parser.add_argument('--auxiliary', action='store_true', default=False,
                          help='use auxiliary tower')
param_parser.add_argument('--auxiliary_weight', type=float, default=0.4,
                          help='weight for auxiliary loss')
param_parser.add_argument('--grad-clip', type=float, default=-1,
                          help='gradient clipping')

graph_parser = parser.add_argument_group(title='Graph-parameters')
graph_parser.add_argument('--channels', default=68, type=int, metavar='C', help='nodes')
graph_parser.add_argument('--in-nodes', default=6, type=int, help='number of input node')
graph_parser.add_argument('--out-nodes', default=6, type=int, help='number of output node')
graph_parser.add_argument('--K', default=4, type=int, metavar='K', help='degree')
graph_parser.add_argument('--N', default=32, type=int, metavar='N', help='nodes')
graph_parser.add_argument('--M', default=5, type=int, metavar='M', help='add edges')
graph_parser.add_argument('--P', default=0.2, type=float, help='link probability')
graph_parser.add_argument('--edge-act', default='softplus', type=str, help='edge activation function')
graph_parser.add_argument('--beta', default=10, type=float, help='edge activation function')

slurm_parser = parser.add_argument_group(title='Slurm arguments')
slurm_parser.add_argument("--debug_slurm", action='store_false', default=True,
                          help="Debug multi-GPU / multi-node within a SLURM job")
slurm_parser.add_argument('--distributed', action='store_true', default=False,
                          help='slurm distributed system enable'
                          'multi node data parallel training')
slurm_parser.add_argument('--world-size', default=-1, type=int, metavar='N',
                          help='number of nodes for distributed training')
slurm_parser.add_argument('--local_rank', default=-1, type=int, metavar='N',
                          help='node rank for distributed training')
slurm_parser.add_argument("--master_port", type=int, default=-1, metavar='N',
                          help="Master port (for multi-node SLURM jobs)")
slurm_parser.add_argument('--dist-backend', default='nccl', type=str,
                          help='distributed backend')

args = parser.parse_args()
best_acc1 = 0


def main():
    global args

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if args.distributed:
        init_distributed_mode(args)
    else:
        args.local_rank = 0

    if args.local_rank == 0 and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    model, criterion, optimizer, lr_scheduler = initialize_setting()

    train_dataset, val_dataset, test_dataset, num_classes, in_shape = \
        data_loader.__dict__[args.dataset](
            args.data, augmentation=True, cutout=args.cutout)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_sampler = DistributedSampler(train_dataset) \
        if args.distributed else torch.utils.data.sampler.SubsetRandomSampler(indices[:split])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.workers,
        pin_memory=True, sampler=train_sampler)

    if args.train_portion < 1:
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train])
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, sampler=valid_sampler,
            num_workers=args.workers, pin_memory=True)
    else:
        val_loader = None

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    cudnn.benchmark = True
    writer = SummaryWriter(args.model_dir) if args.local_rank == 0 else None
    best_acc1 = 0.0

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        train_sampler,
        lr_scheduler,
        train_loader,
        val_loader,
        test_loader,
        writer, best_acc1, args)

    acc1, best_acc1 = trainer.fit(init=True)

    if args.local_rank == 0:
        save_result(model, in_shape, acc1, best_acc1, args)
        writer.close()


def initialize_setting():
    global best_acc1
    global args
    if args.dataset == 'cifar10':
        nClass = 10
    elif args.dataset == 'cifar100':
        nClass = 100
    elif args.dataset == 'tiny_imagenet':
        nClass = 200
    kwargs = {
        'drop_path': args.drop_path,
        'auxiliary': args.auxiliary,
        'K': args.K, 'nNode': args.N, 'P': args.P, 'mu': args.mu, 'M': args.M, 'nCommunity': args.nCommunity,
        'init_norm': args.init_norm,
        'in_nodes': args.in_nodes,
        'out_nodes': args.out_nodes,
        'arch': args.arch,
        'dataset': args.dataset,
        'num_classes': nClass,
        'beta': args.beta,
        'channels': args.channels,
        'edge_act': args.edge_act,
        'seed': args.seed
    }

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            kwargs['graphs'] = checkpoint['graph_dict']
            model = net.complete(**kwargs)
            model.cuda(args.local_rank)
            model.load_graph_dict(checkpoint['graph_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            if args.scratch:
                model.init_params()
                args.start_epoch = 0
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model = net.complete(**kwargs)
            model.cuda(args.local_rank)
    else:
        model = net.complete(**kwargs)
        model.cuda(args.local_rank)

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank)

    params = [v for k, v in model.named_parameters() if not 'graph_weights' in k]
    graph_params = [v for k, v in model.named_parameters() if 'graph_weights' in k]
    optimizer = SGD([
        {'lr': args.lr, 'params': params, 'weight_decay': args.weight_decay},
        {'lr': args.lr, 'params': graph_params, 'weight_decay': 0}
    ], momentum=args.momentum)
    lr_scheduler = CosineAnnealingLR(optimizer, args.epochs)

    # define loss function (criterion) and optimizer
    if args.label_smooth:
        criterion = LabelSmoothing().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    return model, criterion, optimizer, lr_scheduler


if __name__ == '__main__':
    main()
