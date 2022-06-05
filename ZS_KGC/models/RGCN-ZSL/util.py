from __future__ import print_function
import os
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from ordered_set import OrderedSet

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




def build_graph(args):
    KG_file_path = os.path.join(args.DATA_DIR, args.DATASET, 'KG_file', 'rdfs_triples.txt')
    ent_set, rel_set = OrderedSet(), OrderedSet()
    rel_freq = defaultdict(list)
    triples = list()

    if not args.single:
        for line in open(KG_file_path):
            sub, rel, obj = line.strip().split('\t')
            # remove namespace prefix
            if args.DATASET == 'NELL':
                sub, obj = sub.replace('NELL:', 'concept:'), obj.replace('NELL:', 'concept:')
            if args.DATASET == 'Wiki':
                sub, obj = sub.split(':')[1], obj.split(':')[1]
            ent_set.add(sub)
            ent_set.add(obj)
            rel_set.add(rel)
            rel_freq[rel].append((sub, obj))
            triples.append((sub, rel, obj))
    else:

        for line in open(KG_file_path):
            sub, rel, obj = line.strip().split('\t')
            # remove namespace prefix
            if args.DATASET == 'NELL':
                sub, obj = sub.replace('NELL:', 'concept:'), obj.replace('NELL:', 'concept:')
            if args.DATASET == 'Wiki':
                sub, obj = sub.split(':')[1], obj.split(':')[1]
            ent_set.add(sub)
            ent_set.add(obj)
            rel_set.add('one')
            rel_freq['one'].append((sub, obj))
            triples.append((sub, 'one', obj))

    if args.DATASET == 'Wiki':
        Wiki_relation_file = os.path.join(args.DATA_DIR, args.DATASET, 'relation2ids_1')
        dataset_rels = list(json.load(open(Wiki_relation_file)).keys())
        for rel in dataset_rels:
            if rel not in ent_set:
                ent_set.add(rel)

    nodes = list(ent_set)
    relations = list(rel_set)
    adj_shape = (len(nodes), len(nodes))
    print("Number of nodes: {}, Number of relations in the data: {}".format(len(nodes), len(relations)))
    nodes_dict = {node: i for i, node in enumerate(nodes)}
    assert len(nodes_dict) < np.iinfo(np.int32).max

    adjacencies = []

    for i, rel in enumerate(relations):
        print("Creating adjacency matrix for relation {}: {}, frequency {}".format(
                i, rel, len(rel_freq[rel])))
        edges = np.empty((len(rel_freq[rel]), 2), dtype=np.int32)
        size = 0
        for (s, p, o) in triples:
            if p == rel:
                edges[size] = np.array([nodes_dict[s], nodes_dict[o]])
                size += 1

        print("{} edges added".format(size))
        row, col = np.transpose(edges)
        data = np.ones(len(row), dtype=np.int8)
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.int8)
        adj_transp = sp.csr_matrix(
            (data, (col, row)), shape=adj_shape, dtype=np.int8
        )
        adjacencies.append(adj)
        adjacencies.append(adj_transp)

    # add identity matrix (self-connections)
    adjacencies.append(sp.identity(len(nodes)).tocsr())


    return nodes_dict, adjacencies











