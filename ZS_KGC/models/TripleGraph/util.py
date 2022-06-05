from __future__ import print_function
import os
import numpy as np
import scipy.sparse as sp
from ordered_set import OrderedSet
import torch
import warnings
warnings.filterwarnings("ignore")
import json

def row_normalize(AA):
    A_list = list()
    for i in range(len(AA)):
        d = np.array(AA[i].sum(1)).flatten()
        d_inv = 1.0 / d
        d_inv[np.isinf(d_inv)] = 0.0
        D_inv = sp.diags(d_inv)
        A_list.append(D_inv.dot(AA[i]).tocsr())
    return A_list




def build_graph(args, meta_rel):
    Onto_file_path = os.path.join(args.DATA_DIR, args.DATASET, 'KG_file', 'rdfs_triples.txt')

    ent_set = OrderedSet()
    triples = list()

    for line in open(Onto_file_path):
        sub, rel, obj = line.strip().split('\t')
        # remove namespace prefix
        if args.DATASET == 'NELL':
            sub, obj = sub.replace('NELL:', 'concept:'), obj.replace('NELL:', 'concept:')
        if args.DATASET == 'Wiki':
            sub, obj = sub.split(':')[1], obj.split(':')[1]

        if rel == meta_rel:
            ent_set.add(sub)
            ent_set.add(obj)
            triples.append((sub, rel, obj))
    if args.DATASET == 'Wiki':
        Wiki_relation_file = os.path.join(args.DATA_DIR, args.DATASET, 'relation2ids_1')
        dataset_rels = list(json.load(open(Wiki_relation_file)).keys())
        for rel in dataset_rels:
            if rel not in ent_set:
                ent_set.add(rel)


    nodes = list(ent_set)
    n = len(nodes)

    nodes_dict = {node: i for i, node in enumerate(nodes)}

    edges = [(nodes_dict[h], nodes_dict[t]) for (h, r, t) in triples]

    print("Number of nodes: {}, Number of edges: {}".format(len(nodes), len(edges)))

    # + inverse edges and reflexive edges
    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype='float32')
    adj = normt_spm(adj, method='in')
    adj = spm_to_tensor(adj)

    all_feats = []
    for _ in nodes:
        feat = np.random.uniform(low=-1, high=1, size=args.input_dim)
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        all_feats.append(feat)
    all_feats = np.array(all_feats)


    return nodes_dict, adj, all_feats

def normt_spm(mx, method='in'):
    if method == 'in':
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    if method == 'sym':
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx


def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack(
            (sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)









