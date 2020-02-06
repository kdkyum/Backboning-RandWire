import os
import torch
import shutil
from model.net import Node_OP
from thop import profile


def count_node_ops(m, x, y):
    if len(m.in_nodes) == 1:
        m.total_ops += torch.Tensor([int(0)])
    else:
        nNodes = len(m.in_nodes)
        total_ops = y.nelement()
        m.total_ops += (nNodes - 1) * total_ops + total_ops*nNodes


def load_state(model, optimizer, lr_scheduler, resume, args, verbose=False):
    if os.path.isfile(resume):
        if verbose:
            print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.load_graph_dict(checkpoint['graph_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if verbose:
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        return best_acc1
    else:
        print("=> no checkpoint found at '{}'".format(resume))
        return 0


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    file_path = os.path.join(args.model_dir, filename)
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(args.model_dir, 'model_best.pth.tar'))


def save_result(model, in_shape, acc1, best_acc1, args):
    ret = {}
    if args.flops:
        input = torch.randn(1, *in_shape).cuda()
        flops, params = profile(
            model, (input, ), custom_ops={Node_OP: count_node_ops})
        ret['flops'] = flops
        ret['params'] = params
    ret['config'] = args
    ret['best_acc1'] = best_acc1
    ret['acc1'] = acc1
    ret['graphs'] = model.graph_dict()
    torch.save(ret, os.path.join(args.model_dir, 'result.pt'))
    return ret


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
