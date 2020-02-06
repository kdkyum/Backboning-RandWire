import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx
from collections import OrderedDict

__all__ = ['complete']


def identity(x):
    return x


def _complete_graph(nNode):
    G = nx.complete_graph(nNode)
    rand_seq = np.arange(0, len(G))
    for n, r in zip(G.nodes, rand_seq):
        G.nodes[n]['rank'] = r

    G = nx.to_undirected(G)
    DAG = nx.DiGraph()
    for n in G.nodes:
        DAG.add_node(n, rank=G.nodes[n]['rank'])

    for n in G.nodes:
        for m in G.neighbors(n):
            if G.nodes[m]['rank'] > G.nodes[n]['rank']:
                DAG.add_edge(n, m)
    return DAG

def _complete_graph_constrain(nNode, in_nodes, out_nodes):
    G = nx.complete_graph(nNode)
    rand_seq = np.arange(0, len(G))
    for n, r in zip(G.nodes, rand_seq):
        G.nodes[n]['rank'] = r

    G = nx.to_undirected(G)
    DAG = nx.DiGraph()
    for n in G.nodes:
        DAG.add_node(n, rank=G.nodes[n]['rank'])

    for n in G.nodes:
        for m in G.neighbors(n):
            if n == 0:
                if m <= in_nodes:
                    DAG.add_edge(n, m)
            elif G.nodes[m]['rank'] > G.nodes[n]['rank'] and m!=nNode-1:
                if n < nNode-1-out_nodes:
                    if n <= in_nodes:
                        if m > in_nodes:
                            DAG.add_edge(n, m)
                    else:
                        DAG.add_edge(n, m)
            elif m==nNode-1:
                if n >= nNode-1 - out_nodes:
                    DAG.add_edge(n, m)
    return DAG

def build_graph(**kwargs):
    if kwargs['arch'] == 'complete':
        nNode = kwargs['nNode'] + 2
        return _complete_graph_constrain(nNode, kwargs['in_nodes'], kwargs['out_nodes'])
    else:
        nNode = kwargs['nNode']
        model = kwargs['arch']
        if model == 'er':
            G = nx.random_graphs.erdos_renyi_graph(nNode, kwargs['P'], kwargs['seed'])
        elif model == 'ws':
            G = nx.random_graphs.connected_watts_strogatz_graph(
                nNode, kwargs['K'],
                kwargs['P'],
                tries=200, seed=kwargs['seed'])
        elif model == 'ba':
            G = nx.barabasi_albert_graph(nNode, kwargs['M'])

        rand_seq = np.arange(0, len(G))
        for n, r in zip(G.nodes, rand_seq):
            G.nodes[n]['rank'] = r
        _G = nx.to_undirected(G)
        DAG = nx.DiGraph()
        DAG.add_node(-1, rank=-1)
        for n in _G.nodes:
            DAG.add_node(n, rank=_G.nodes[n]['rank'])
        DAG.add_node(nNode, rank=nNode)

        for n in _G.nodes:
            for m in _G.neighbors(n):
                if _G.nodes[m]['rank'] > _G.nodes[n]['rank']:
                    DAG.add_edge(n, m)

        for n, k_in in DAG.in_degree():
            if k_in == 0 and n >= 0 and n < nNode:
                DAG.add_edge(-1, n)

        for n, k_out in DAG.out_degree():
            if k_out == 0 and n < nNode:
                DAG.add_edge(n, nNode)

        H = nx.relabel_nodes(DAG, lambda x: x + 1)

        return H


class ZeroPad(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ZeroPad, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        pad = (out_channels - in_channels) // 2
        if stride > 1:
            self.pool = nn.AvgPool2d(kernel_size=1, stride=stride)
        self.p1d = (0, 0, 0, 0, pad, pad)

    def forward(self, x):
        if self.stride != 1:
            x = self.pool(x)
        if self.in_channels != self.out_channels:
            x = F.pad(x, self.p1d, "constant", 0)
        return x


class Separable_conv(nn.Sequential):
    def __init__(self, nin, nout, stride):
        super(Separable_conv, self).__init__()
        self.add_module('depthwise',
                        nn.Conv2d(nin, nin, kernel_size=3, stride=stride, padding=1, groups=nin, bias=False))
        self.add_module('pointwise', nn.Conv2d(nin, nout, kernel_size=1, bias=False))


class Pool(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Pool, self).__init__()
        self.add_module('pool', Triplet_unit(in_channels, out_channels, stride=stride, isPool=True))


class Triplet_unit(nn.Sequential):
    def __init__(self, inplanes, outplanes, stride=1, isPool=False):
        super(Triplet_unit, self).__init__()
        # original paper
        if not isPool:
            self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', Separable_conv(inplanes, outplanes, stride))
        self.add_module('norm', nn.BatchNorm2d(outplanes))


class Node_OP(nn.Module):
    def __init__(self, in_nodes, in_channels, out_channels,
                 stride=2, edge_act=identity, droppath=0, dropout=0, output=False):
        super(Node_OP, self).__init__()
        self.in_nodes = in_nodes
        if 0 in self.in_nodes:
            self.input = True
        else:
            self.input = False
        self.output = output
        if self.output:
            self.conv = nn.Identity()
        elif self.input:
            self.conv = Pool(in_channels, out_channels, stride)
        else:
            self.conv = Triplet_unit(out_channels, out_channels)

        self.graph_weights = nn.Parameter(torch.ones(len(in_nodes)))
        self.droppath = droppath
        self.dropout = dropout
        self.edge_act = edge_act

    def forward(self, **features):
        if self.input:
            return self.conv(features['0'])
        inputs = []
        for n in self.in_nodes:
            if n == 0:
                inputs.append(self.pool(features['0']))
            else:
                inputs.append(features['%d' % n])
        weights = [w for n, w in enumerate(self.graph_weights)]
        p = self.droppath
        out = self.dropPath_aggregate(inputs, weights, p)
        out = self.conv(out)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out

    def replace_edge(self, selected_idx):
        self.graph_weights = \
            nn.Parameter(self.graph_weights[selected_idx])

    def dropPath_aggregate(self, inputs, weights, p):
        N = len(self.in_nodes)
        x0 = inputs[0]
        bs = x0.size(0)
        notDrop = p == 0 or not self.training or self.output or N == 1
        if notDrop:
            out = x0 * self.edge_act(weights[0])
            if len(inputs) > 1:
                for x, w in zip(inputs[1:], weights[1:]):
                    out += x*self.edge_act(w)
            return out
        else:
            keep_prob = 1.0 - p
            mask = torch.ones(N, bs, dtype=x0.dtype, device=x0.device).bernoulli_(keep_prob)
            mask_sum = mask.sum(0)
            zeros_idx = [n.item() for n in (mask_sum==0).nonzero().flatten()]
            for j in zeros_idx:
                i = np.random.randint(0, N)
                mask[i, j].fill_(1)
            mask.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)
            scale = N*bs / mask.sum()

            out = x0 * (mask[0] * self.edge_act(weights[0]) * scale)
            if len(inputs) > 1:
                for m, (x, w) in enumerate(zip(inputs[1:], weights[1:])):
                    out += x * (mask[m+1] * self.edge_act(w) * scale)
            return out


class StageBlock(nn.Module):

    # Weight matrix & Adjacency matrix (used as mask)
    def __init__(self, graph, in_channels, out_channels, stride=2, edge_act=identity, droppath=0, dropout=0):
        super(StageBlock, self).__init__()
        self.graph = graph
        self.edge_act = edge_act
        self.num_nodes = len(graph)
        self.in_nodes = {
            '%d' % n: sorted([k for k in graph.predecessors(n)])
            for n in graph.nodes if n > 0}
        self.nodes = nn.ModuleDict(OrderedDict({
            '%d' % n: Node_OP(
                self.in_nodes[str(n)], in_channels, out_channels, stride=stride,
                edge_act=edge_act, droppath=droppath, dropout=dropout,
                output=(n == max(self.graph.nodes)))
            for n in self.graph.nodes if n > 0
        }))

    def _masking_graph(self):
        self.in_nodes = {
            '%d' % n: sorted([k for k in self.graph.predecessors(n)])
            for n in self.graph.nodes if n > 0}
        for name, node_op in self.nodes.items():
            node2idx = {k: n for n, k in enumerate(node_op.in_nodes)}
            predecessors = sorted([node2idx[k] for k in self.graph.predecessors(int(name))])
            node_op.replace_edge(predecessors)
            node_op.in_nodes = self.in_nodes[name]

    def forward(self, x, inplace=True):
        features = OrderedDict({'0': F.relu(x, inplace=inplace)})
        for name, node_op in self.nodes.items():
            features[name] = node_op(**features)

        return features['%d' % (self.num_nodes-1)]


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 16x16"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(7, stride=3, padding=0, count_include_pad=False), # image size = 4 x 4
            nn.Conv2d(C, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class CNN(nn.Module):
    def __init__(self, graphs=None, auxiliary=False, num_classes=10, init_norm=False,
                 conv1x1=True, channels=64, edge_act=identity, drop_path=0., drop_rate=0., **kwargs):
        super(CNN, self).__init__()
        self.dataset = kwargs['dataset']
        self.nNode = kwargs['nNode']
        self.channels = channels
        self.edge_act = edge_act
        self.droppath = drop_path
        self.dropout = drop_rate
        self.auxiliary = auxiliary
        self.init_norm = init_norm
        self.conv1x1 = conv1x1
        self.graphs = graphs

        if self.graphs is None:
            self.graph2 = build_graph(**kwargs)
            self.graph3 = build_graph(**kwargs)
            self.graph4 = build_graph(**kwargs)
        else:
            self.graph2 = self.graphs['conv2.graph']
            self.graph3 = self.graphs['conv3.graph']
            self.graph4 = self.graphs['conv4.graph']

        if self.dataset=='tiny_imagenet':
            self.conv1 = nn.Sequential(OrderedDict([
                ('conv', Separable_conv(3, self.channels // 2, 2)),
                ('bn', nn.BatchNorm2d(self.channels // 2)),
            ]))
            self.conv2 = StageBlock(self.graph2, self.channels // 2, self.channels, 1,
                                    self.edge_act, self.droppath, self.dropout)
            self.conv3 = StageBlock(self.graph3, self.channels, self.channels * 2, 2,
                                    self.edge_act, self.droppath, self.dropout)
            self.conv4 = StageBlock(self.graph4, self.channels * 2, self.channels * 4, 2,
                                    self.edge_act, self.droppath, self.dropout)
            self.conv5 = nn.Sequential(OrderedDict([
                ('relu0', nn.ReLU(inplace=True)),
                ('conv', nn.Conv2d(self.channels * 4, 1280, kernel_size=1, bias=False)),
                ('bn', nn.BatchNorm2d(1280)),
                ('relu1', nn.ReLU(inplace=True))
            ]))
            if self.auxiliary:
                self.auxiliary_head = AuxiliaryHeadCIFAR(self.channels * 2, num_classes)
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(1280, num_classes)
            )
        elif self.dataset in ['cifar10', 'cifar100']:
            self.conv1 = nn.Sequential(OrderedDict([
                ('conv', Separable_conv(3, self.channels // 2, 1)),
                ('bn', nn.BatchNorm2d(self.channels // 2)),
            ]))
            self.conv2 = StageBlock(self.graph2, self.channels // 2, self.channels, 1,
                                    self.edge_act, self.droppath, self.dropout)
            self.conv3 = StageBlock(self.graph3, self.channels, self.channels * 2, 2,
                                    self.edge_act, self.droppath, self.dropout)
            self.conv4 = StageBlock(self.graph4, self.channels * 2, self.channels * 4, 2,
                                    self.edge_act, self.droppath, self.dropout)
            self.conv5 = nn.Sequential(OrderedDict([
                ('relu0', nn.ReLU(inplace=True)),
                ('conv', nn.Conv2d(self.channels * 4, 1280, kernel_size=1, bias=False)),
                ('bn', nn.BatchNorm2d(1280)),
                ('relu1', nn.ReLU(inplace=True))
            ]))
            if self.auxiliary:
                self.auxiliary_head = AuxiliaryHeadCIFAR(self.channels * 2, num_classes)
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(1280, num_classes)
            )

        self.init_params()

    def init_params(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Node_OP):
                if 0 in m.in_nodes:
                    m.graph_weights = nn.Parameter(torch.ones(1), requires_grad=False)
                    m.edge_act = identity
                elif m.output:
                    m.graph_weights = nn.Parameter(torch.ones(len(m.in_nodes)), requires_grad=False)
                    nn.init.constant_(m.graph_weights, 1 / math.sqrt(len(m.in_nodes)))
                    m.edge_act = identity
                elif self.edge_act.__name__ == 'sigmoid':
                    k = 1 / math.sqrt(len(m.in_nodes))
                    nn.init.constant_(m.graph_weights, math.log(k / (1-k)))
                elif self.init_norm:
                    nn.init.constant_(m.graph_weights, 1 / math.sqrt(len(m.in_nodes)))
                elif self.edge_act.__name__ == 'softplus':
                    nn.init.constant_(m.graph_weights, 1)

    def set_dropPath(self, drop_path_prob):
        for name, m in self.named_modules():
            if isinstance(m, Node_OP):
                if 0 in m.in_nodes:
                    m.graph_weights.requires_grad=False
                    m.edge_act = identity
                    m.droppath = 0
                elif m.output:
                    m.graph_weights.requires_grad=False
                    m.edge_act = identity
                    m.droppath = 0
                else:
                    m.droppath = drop_path_prob

    def graph_dict(self):
        _graph_dict = OrderedDict()
        with torch.no_grad():
            for name, m in self.named_modules():
                if isinstance(m, StageBlock):
                    G = m.graph.copy()
                    for u, v in G.edges:
                        node2idx = {k: n for n, k in enumerate(m.in_nodes['%d' % v])}
                        x = m.nodes['%d' % v]
                        G[u][v]['weight'] = x.edge_act(x.graph_weights[node2idx[u]]).item()

                    _graph_dict['%s.graph' % name] = G

            return _graph_dict

    def load_graph_dict(self, graph_dict):
        for name, m in self.named_modules():
            if isinstance(m, StageBlock):
                m.graph = graph_dict['%s.graph' % name]
                m._masking_graph()

    def forward(self, x):
        logits_aux = None
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.auxiliary and self.training:
            logits_aux = self.auxiliary_head(x)
        x = self.conv4(x, inplace=not self.auxiliary)
        x = self.conv5(x)
        x = self.global_pooling(x)
        logits = self.classifier(x.view(x.size(0), -1))

        return logits, logits_aux


def complete(**kwargs):
    if kwargs['edge_act'] == 'relu':
        edge_act = F.relu
    elif kwargs['edge_act'] == 'sigmoid':
        edge_act = F.sigmoid
    elif kwargs['edge_act'] == 'softplus':
        beta = kwargs['beta']

        def softplus(x):
            return F.softplus(x, beta=beta)
        edge_act = softplus
    else:
        edge_act = identity

    kwargs['edge_act'] = edge_act
    kwargs['drop_rate'] = 0

    return CNN(**kwargs)
