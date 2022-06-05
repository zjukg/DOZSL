from __future__ import print_function
import os
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from ordered_set import OrderedSet

import warnings
warnings.filterwarnings("ignore")


import os
import numpy as np
import json
import scipy.io as scio
from ordered_set import OrderedSet


import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch_scatter import scatter_add



def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param



def build_graph(args):
    KG_file_path = os.path.join(args.DATA_DIR, args.DATASET, 'KG_file', 'rdfs_triples.txt')

    ent_set, rel_set = OrderedSet(), OrderedSet()
    triples = list()

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

        triples.append((sub, rel, obj))

    if args.DATASET == 'Wiki':
        Wiki_relation_file = os.path.join(args.DATA_DIR, args.DATASET, 'relation2ids_1')
        dataset_rels = list(json.load(open(Wiki_relation_file)).keys())
        for rel in dataset_rels:
            if rel not in ent_set:
                ent_set.add(rel)


    ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
    rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
    rel2id.update({rel + '_reverse': idx + len(rel2id) for idx, rel in enumerate(rel_set)})

    num_ent = len(ent2id)
    num_rel = len(rel2id) // 2
    print('num_ent {} num_rel {}'.format(num_ent, num_rel))

    triples = [(ent2id[h], rel2id[r], ent2id[t]) for (h, r, t) in triples]

    edge_index, edge_type = [], []

    for sub, rel, obj in triples:
        edge_index.append((sub, obj))
        edge_type.append(rel)

    # Adding inverse edges
    for sub, rel, obj in triples:
        edge_index.append((obj, sub))
        edge_type.append(rel + num_rel)

    edge_index = torch.LongTensor(edge_index).cuda().t()
    edge_type = torch.LongTensor(edge_type).cuda()

    return edge_index, edge_type, ent2id, num_ent, num_rel











