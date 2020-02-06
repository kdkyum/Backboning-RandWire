import pickle
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from utils.misc import AverageMeter, ProgressMeter, accuracy, save_checkpoint


class Trainer():
    def __init__(self, model, criterion, optimizer, train_sampler,
                 lr_scheduler, train_loader, val_loader, test_loader, writer, best_acc1, args):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_sampler = train_sampler
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.writer = writer
        self.args = args
        self.moving_loss = 0.0
        self.mu = 1. / len(self.train_loader)
        self.best_acc1 = best_acc1
        self.drop_path = args.drop_path

        # initialize lr scheduler
        if self.args.start_epoch > 0:
            for j in range(self.args.start_epoch):
                self.lr_scheduler.step(j)

    def fit(self, init=False):
        if init:
            self.on_start_train()

        for epoch in range(self.args.start_epoch, self.args.epochs):
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)
            if not self.args.noScheduleDrop:
                self.model.set_dropPath(self.drop_path * epoch / self.args.epochs)
            else:
                self.model.set_dropPath(self.drop_path)

            # train for one epoch
            self.train_epoch(epoch)
            acc1 = self.test(epoch).item()
            if self.val_loader:
                acc1 = self.validate(epoch).item()
            self.lr_scheduler.step(epoch+1)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > self.best_acc1
            self.best_acc1 = max(acc1, self.best_acc1)

            if self.args.local_rank == 0:
                if (epoch < 10 or epoch+1 in [20, 50]) and init:
                    filename = 'checkpoint.pth.tar' if not init \
                        else 'checkpoint-%d.pth.tar' % (epoch+1)
                else:
                    filename = 'checkpoint.pth.tar'
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'graph_dict': self.model.graph_dict(),
                    'best_acc1': self.best_acc1,
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict()
                }, is_best, self.args, filename)
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

        return acc1, self.best_acc1

    def on_start_train(self):
        if self.args.local_rank == 0:
            filename = 'checkpoint-%d.pth.tar' %self.args.start_epoch
            save_checkpoint({
                'epoch': self.args.start_epoch,
                'state_dict': self.model.state_dict(),
                'graph_dict': self.model.graph_dict(),
                'best_acc1': self.best_acc1,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict()
            }, False, self.args, filename)
            self.summary_graph_adj(self.writer, self.args.start_epoch)
            self.summary_graph_histogram(self.writer, self.args.start_epoch)

    def train_epoch(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        self.model.train()
        end = time.time()

        for i, (images, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            images = images.cuda()
            target = target.cuda()

            # compute output
            self.optimizer.zero_grad()
            logits, logits_aux  = self.model(images)
            loss = self.criterion(logits, target)
            if self.args.graph_wd > 0:
                graph_params = [v for k, v in self.model.named_parameters() 
                                if 'graph_weights' in k and v.requires_grad]
                graph_l2 = 0
                for v in graph_params:
                    graph_l2 += (self.model.edge_act(v)**2).sum()
                loss += 0.5 * graph_l2 * self.args.graph_wd
            if self.args.auxiliary:
                loss_aux = self.criterion(logits_aux, target)
                loss += self.args.auxiliary_weight*loss_aux
            loss.backward()
            if self.args.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            self.moving_loss = loss.item() if epoch == self.args.start_epoch and i == 0 else \
                (1. - self.mu) * self.moving_loss + self.mu * loss.item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0 and self.args.local_rank == 0:
                progress.display(i)
                niter = epoch * len(self.train_loader) + i
                self.writer.add_scalar('Train/Sec_per_batch', batch_time.avg, niter)
                self.writer.add_scalar('Train/Avg_Loss', losses.avg, niter)
                self.writer.add_scalar('Train/Avg_Top1', top1.avg, niter)
                self.writer.add_scalar('Train/Avg_Top5', top5.avg, niter)
                self.writer.add_scalar('Train/Moving_Loss', self.moving_loss, niter)

    def validate(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(self.val_loader),
            [batch_time, losses, top1, top5],
            prefix='Val: ')

        # switch to test mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(self.val_loader):
                images = images.cuda()
                target = target.cuda()

                # compute output
                output, _ = self.model(images)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0 and self.args.local_rank == 0:
                    progress.display(i)

            if self.args.local_rank == 0:
                print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                      .format(top1=top1, top5=top5))
                self.writer.add_scalar('Val/Avg_Loss', losses.avg, epoch + 1)
                self.writer.add_scalar('Val/Avg_Top1', top1.avg, epoch + 1)
                self.writer.add_scalar('Val/Avg_Top5', top5.avg, epoch + 1)

        return top1.avg

    def test(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(self.test_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        # switch to test mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(self.test_loader):
                images = images.cuda()
                target = target.cuda()

                # compute output
                output, _ = self.model(images)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0 and self.args.local_rank == 0:
                    progress.display(i)

            if self.args.local_rank == 0:
                print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                      .format(top1=top1, top5=top5))
                self.writer.add_scalar('Test/Avg_Loss', losses.avg, epoch + 1)
                self.writer.add_scalar('Test/Avg_Top1', top1.avg, epoch + 1)
                self.writer.add_scalar('Test/Avg_Top5', top5.avg, epoch + 1)
                self.summary_graph_adj(self.writer, epoch + 1)
                self.summary_graph_histogram(self.writer, epoch + 1)

        return top1.avg

    def summary_graph_histogram(self, writer, epoch):
        with torch.no_grad():
            for k, G in self.model.graph_dict().items():
                labels = nx.get_edge_attributes(G, 'weight')
                weights = np.array([v for k, v in labels.items()])
                writer.add_histogram(k, weights, epoch)

    def summary_graph_adj(self, writer, epoch):
        with torch.no_grad():
            for k, G in self.model.graph_dict().items():
                writer.add_figure('adj/%s' %k, self._draw_adjacency_matrix(G), epoch)

    def summary_graph_figure(self, writer, epoch):
        with torch.no_grad():
            for k, G in self.model.graph_dict().items():
                writer.add_figure('layout/%s' %k, self._draw_graph(G), epoch)
                writer.add_figure('adj/%s' %k, self._draw_adjacency_matrix(G), epoch)

    @staticmethod
    def _draw_graph(G):
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        fig = plt.figure(figsize=(8, 10), dpi=150)
        ax = fig.gca()
        nx.draw_networkx_nodes(
            G, pos, nodelist=[n for n in G.nodes],
            node_size=200, node_color='white', node_shape='o', alpha=1.0, cmap=None, vmin=None, vmax=None, ax=None,
            linewidths=1, edgecolors='black', label=None)
        nx.draw_networkx_labels(G, pos, label={n: str(n) for n in G.nodes}, font_size=9)
        labels = nx.get_edge_attributes(G, 'weight')
        weights = [v for k, v in labels.items()]
        max_weight = max(weights)
        nx.draw_networkx_edges(G, pos, G.edges, width=[v/max_weight for k, v in labels.items()])
        ax.set_axis_off()
        return fig

    @staticmethod
    def _draw_adjacency_matrix(G, vmin=0.0, vmax=1.0, node_order=None, partitions=[], colormap='magma', colors=[]):
        """
            - G is a netorkx graph
            - node_order (optional) is a list of nodes, where each node in G
                appears exactly once
            - partitions is a list of node lists, where each node in G appears
                in exactly one node list
            - colors is a list of strings indicating what color each
                partition should be
            If partitions is specified, the same number of colors needs to be
            specified.
        """
        fig = plt.figure(dpi=100)
        
        adjacency_matrix = nx.to_numpy_matrix(G, nodelist=node_order)
        im = plt.imshow(
            adjacency_matrix,
            cmap=colormap, interpolation="none", vmin=vmin, vmax=vmax)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax = plt.gca()
        current_idx = 0
        for partition, color in zip(partitions, colors):
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                        len(partition), # Width
                                        len(partition), # Height
                                        facecolor="none",
                                        edgecolor=color,
                                        linewidth="2"))
            current_idx += len(partition)
        plt.xlabel("target", fontsize=15)
        plt.ylabel("source", fontsize=15)
        plt.colorbar()
        plt.tight_layout()
        return fig
