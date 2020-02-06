from copy import deepcopy

import networkx as nx
import numpy as np
import torch
import torch.nn as nn


class Pruner:
    def __init__(self, model):
        self.model = model
        self.graphs = deepcopy(self.model.graph_dict())
        self.node_l1 = {}
        for k, v in model.named_parameters():
            if 'norm.weight' in k and 'nodes' in k:
                l1_norm = sum(abs(v)).item()
                key = '.'.join(k.split('.')[0:3])
                self.node_l1[key] = l1_norm

    def prune_backbone(self, ratio, method='naive', l1_norm=0, prune_edges=-1):
        tmp_graphs = {}
        for k, g in self.graphs.items():
            _G2 = g.copy()
            if l1_norm:
                A = nx.to_numpy_matrix(_G2)
                B = self.multiply_l1norm(A, k.split('.')[0], self.node_l1)
                _G2 = nx.from_numpy_matrix(B, create_using=nx.DiGraph)

            if method == 'in_disparity':
                G2 = in_disparity_filter(_G2)
            elif method in ['BC', 'BC_inv']:
                G2 = bc_filter(_G2)
            else:
                G2 = _G2.copy()
            G2 = iterative_cut(G2, _G2, prune_ratio=ratio, method=method, prune_edges=prune_edges)
            tmp_graphs[k] = G2
        self.model.load_graph_dict(tmp_graphs)
        self.graphs = tmp_graphs

    def get_prune_rate(self):
        ret = {}
        for k, G in self.graphs.items():
            # directed graph's density is half of undirected graph's density
            ret[k] = 2*nx.density(G)
        return ret

    @staticmethod
    def multiply_l1norm(A, stage, node_l1):
        B = A.copy()
        for k, v in node_l1.items():
            _stage = k.split('.')[0]
            _node = int(k.split('.')[-1])
            if _stage == stage:
                for i in range(1, A.shape[0]):
                    B[_node, i] = A[_node, i] * v
        return B


def in_alpha(x):
    _, _, w = x
    alpha_in = w['alpha_in']
    return alpha_in


def alpha_key(x):
    _, _, w = x
    alpha_in = w['alpha_in'] if 'alpha_in' in w.keys() else 1.0
    alpha_out = w['alpha_out'] if 'alpha_out' in w.keys() else 1.0
    return min(alpha_in, alpha_out)


def bc_key(x):
    _, _, w = x
    return w['BC']


def weight_key(x):
    _, _, w = x
    return abs(w['weight'])


def random_key(x):
    return np.random.rand(1)[0]


def normalized_weight_key(x):
    _, _, w = x
    p_in = w['p_ij_in'] if 'p_ij_in' in w.keys() else 0.0
    p_out = w['p_ij_out'] if 'p_ij_out' in w.keys() else 0.0
    return max(p_in, p_out)


def iterative_cut(G, _G, weight='weight', prune_ratio=0.4, method='disparity', prune_edges=-1):
    B = G.copy()
    nEdges = len(B.edges)
    stop_count = int((1.0-prune_ratio) * nEdges)
    if prune_edges > 0:
        stop_count = prune_edges
    if method == 'in_disparity':
        important_edges = sorted(G.edges(data=True), key=in_alpha, reverse=True)
    elif method == 'naive':
        important_edges = sorted(G.edges(data=True), key=weight_key, reverse=False)
    elif method == 'BC':
        important_edges = sorted(G.edges(data=True), key=bc_key, reverse=False)
    elif method == 'random':
        important_edges = sorted(G.edges(data=True), key=random_key, reverse=False)

    for n, (u, v, w) in enumerate(important_edges):
        if nEdges == stop_count:
            break

        src_k_out = _G.out_degree(u)
        target_k_in = _G.in_degree(v)

        if src_k_out <= 1 or target_k_in <= 1:
            pass
        else:
            _G.remove_edge(u, v)
            nEdges -= 1
    return _G


def in_disparity_filter(G, weight='weight'):
    ''' Compute significance scores (alpha) for weighted edges in directed acyclic graph
        Args
            G: Weighted directed acyclic NetworkX graph
        Returns
            Weighted directed acyclic graph with a significance score (alpha) assigned to each edge
        References
            [1] M. A. Serrano et al. (2009) Extracting the Multiscale backbone of complex weighted networks. PNAS, 106:16, pp. 6483-6488.
            [2] https://github.com/aekpalakorn/python-backbone-network/blob/master/backbone.py
    '''
    N = nx.DiGraph()
    for u in G:
        k_out = G.out_degree(u)
        k_in = G.in_degree(u)

        if k_in > 1:
            sum_w_in = sum(np.absolute(G[v][u][weight]) for v in G.predecessors(u))
            for v in G.predecessors(u):
                w = G[v][u][weight]
                p_ij_in = float(np.absolute(w))/sum_w_in
                alpha_ij_in = 1 - ((1-p_ij_in)**k_in + p_ij_in - 1) / (p_ij_in-1)
                N.add_edge(v, u, weight=w, alpha_in=alpha_ij_in, p_ij_in=p_ij_in)
        elif k_in == 1:
            v = next(G.predecessors(u))
            w = G[v][u][weight]
            N.add_edge(v, u, weight=w, alpha_in=0., p_ij_in=1.0)
    return N


def bc_filter(G):
    g = G.copy()
    for u, v, d in g.edges(data=True):
        g[u][v]['inverse_weight'] = 1/d['weight']

    for (u, v), d in nx.edge_betweenness_centrality(g, weight='inverse_weight').items():
        g[u][v]['BC'] = d
    return g
