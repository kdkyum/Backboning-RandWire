import argparse
import os
import random
import socket
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import loader.data_loader as data_loader
import model.net as net
from tensorboardX import SummaryWriter
from trainer.pruner import Pruner
from trainer.trainer import Trainer
from utils.misc import load_state
from utils.sgd import SGD
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
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--test', action='store_true', default=False,
                    help='test new network model.')

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
param_parser.add_argument('--label-smooth', action='store_true', default=False,
                          help='label smoothing')
param_parser.add_argument('--noScheduleDrop', action='store_true', default=False,
                          help='use auxiliary tower')
param_parser.add_argument('--init_norm', action='store_true', default=False, help='normalized initialization')
param_parser.add_argument('--auxiliary', action='store_true', default=False,
                          help='use auxiliary tower')
param_parser.add_argument('--auxiliary_weight', type=float, default=0.4,
                          help='weight for auxiliary loss')

prune_parser = parser.add_argument_group(title='Pruning arguments')
prune_parser.add_argument('--prune-init', default='reset', type=str,
                          help='initialization method (default: reset)')
prune_parser.add_argument('--prune-rate', default=0.1, type=float,
                          help='pruning rate (default: 0.1)')
prune_parser.add_argument('--prune-edges', default=-1, type=int,
                          help='number of pruning edges (default: -1)')
prune_parser.add_argument('--prune-criterion', default='naive', type=str,
                          help='pruning criterion (default: naive)')
prune_parser.add_argument('--l1_norm', default=0, type=int,
                          help='scaled with BN l1 norm to graph weight')
prune_parser.add_argument('--reset-path', default='', type=str, metavar='PATH',
                          help='resetting checkpoint')
prune_parser.add_argument('--scratch', action='store_true', default=False,
                          help='scratch')
prune_parser.add_argument('--save', default='', type=str, metavar='',
                          help='save directory')

graph_parser = parser.add_argument_group(title='Graph-parameters')
graph_parser.add_argument('--channels', default=68, type=int, metavar='C', help='nodes')
graph_parser.add_argument('--in-nodes', default=6, type=int, help='number of input node')
graph_parser.add_argument('--out-nodes', default=6, type=int, help='number of output node')
graph_parser.add_argument('--K', default=4, type=int, metavar='K', help='degree')
graph_parser.add_argument('--N', default=32, type=int, metavar='N', help='nodes')
graph_parser.add_argument('--M', default=5, type=int, metavar='M', help='add edges')
graph_parser.add_argument('--P', default=0.2, type=float, help='link probability')
graph_parser.add_argument('--mu', default=0.1, type=float, help='lfr rewire')
graph_parser.add_argument('--nCommunity', default=2, type=int, metavar='N', help='number of communities')
graph_parser.add_argument('--edge-act', default='softplus', type=str, help='edge activation function')
graph_parser.add_argument('--beta', default=10, type=float, help='edge activation function')
graph_parser.add_argument('--drop-path', default=0, type=float,
                          metavar='DR', help='drop path rate')

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
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    if args.distributed:
        init_distributed_mode(args)
    else:
        args.local_rank = 0

    model, model_ref, criterion = initialize_setting()

    _, val_dataset, test_dataset, num_classes, in_shape = \
        data_loader.__dict__[args.dataset](
            args.data, augmentation=True)

    val_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    cudnn.benchmark = True
    pruner = Pruner(model)
    pruner.prune_backbone(args.prune_rate, method=args.prune_criterion, l1_norm=args.l1_norm, prune_edges=args.prune_edges)
    test_acc = 0
    if args.reset_path:
        test_loss, test_acc, correct = test(model, criterion, val_loader, verbose=True)

    model_ref.load_graph_dict(pruner.graphs)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.reset_path:
        save_path = os.path.join(args.save, 'pruned-%d.pth.tar' % args.start_epoch)
        if args.scratch:
            model_ref.init_params()
    else:
        save_path = os.path.join(args.save, 'pruned.pth.tar')
        model_ref.init_params()

    torch.save({
        'epoch': args.start_epoch,
        'state_dict': model_ref.state_dict(),
        'graph_dict': model_ref.graph_dict(),
        'prune_acc1': test_acc,
    }, save_path)

    if args.reset_path:
        tmp = [{'arch': args.arch, 'rewind': args.start_epoch, 'test_loss': test_loss,
                'test_acc': test_acc, 'correct': correct, 'ratio': args.prune_rate}]
        df = pd.DataFrame(tmp)
        df.to_csv(os.path.join(args.save, 'prune.tsv'), sep='\t')


def test(model, criterion, val_loader, verbose=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            output, _ = model(images)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    test_acc = 100. * correct / len(val_loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(val_loader.dataset), test_acc
        ))

    return test_loss, test_acc, correct


def initialize_setting():
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

    # define loss function (criterion) and optimizer
    if args.label_smooth:
        criterion = LabelSmoothing().cuda()
    else:
        criterion = nn.CrossEntropyLoss(reduction='sum').cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        kwargs['graphs'] = checkpoint['graph_dict']
        model = net.complete(**kwargs)
        model.cuda(args.local_rank)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        model = net.complete(**kwargs)
        model.cuda(args.local_rank)
        print("=> no checkpoint found at '{}'".format(args.resume))

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

    if args.reset_path:
        if os.path.isfile(args.reset_path):
            print("=> loading checkpoint '{}'".format(args.reset_path))
        checkpoint = torch.load(args.reset_path)
        kwargs['graphs'] = checkpoint['graph_dict']
        model_ref = net.complete(**kwargs)
        model_ref.cuda(args.local_rank)
        model_ref.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.reset_path, checkpoint['epoch']))
    else:
        model_ref = net.complete(**kwargs)
        model_ref.cuda(args.local_rank)
        print("=> no checkpoint found at '{}'".format(args.reset_path))

    return model, model_ref, criterion


if __name__ == '__main__':
    main()
