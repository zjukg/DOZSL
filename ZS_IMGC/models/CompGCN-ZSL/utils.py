from __future__ import print_function
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

class DATA_LOADER(object):
    def __init__(self, args):

        # load KG graph
        self.build_graph(args)


        if args.DATASET == 'AwA2':
            self.read_awa(args)
        else:
            self.read_imagenet(args)


    def read_awa(self, args):
        split = json.load(open(os.path.join(args.DATA_DIR, args.DATASET, 'class.json')))
        seens, unseens = split['seen'], split['unseen']
        seen_wnids, unseen_wnids = list(seens.keys()), list(unseens.keys())
        # print(seen_wnids)
        seen_names = [seens[wnid] for wnid in seen_wnids]
        unseen_names = [unseens[wnid] for wnid in unseen_wnids]

        print("Loading Training Data ...")
        # training data: averaged seen features
        matcontent = scio.loadmat(os.path.join(args.DATA_DIR, args.DATASET, 'att_splits.mat'))
        class_names = matcontent['allclasses_names'].squeeze().tolist()

        feat_matcontent = scio.loadmat(os.path.join(args.DATA_DIR, args.DATASET, 'res101.mat'))
        features = feat_matcontent['features'].T
        labels = feat_matcontent['labels'].astype(int).squeeze() - 1

        feat_set = list()

        for i, name in enumerate(seen_names):
            sample_label = class_names.index(name)
            feat_index = np.where(labels == sample_label)
            cls_feats = features[feat_index]
            cls_feats = np.mean(cls_feats, axis=0)  # (2048)
            feat_set.append(cls_feats)

        self.fc_vectors = np.vstack(tuple(feat_set))


        self.all_nodes = class_names
        self.seen_classes = seen_wnids
        self.seen_names = seen_names
        self.unseen_classes = unseen_wnids
        self.unseen_names = unseen_names


    def read_imagenet(self, args):

        seen_file = os.path.join(args.DATA_DIR, args.DATASET, 'seen.txt')
        unseen_file = os.path.join(args.DATA_DIR, args.DATASET, 'unseen.txt')
        seen_wnids = [line[:-1] for line in open(seen_file)]
        unseen_wnids = [line[:-1] for line in open(unseen_file)]


        print("Loading Training Data ...")
        # training data: averaged seen features
        Seen_Feat = os.path.join(args.DATA_DIR, 'ImageNet', 'Res101_Features/ILSVRC2012_train')
        matcontent = scio.loadmat(os.path.join(args.DATA_DIR, 'ImageNet', 'split.mat'))
        allwnids = matcontent['allwnids'].squeeze().tolist()

        feat_set = list()
        for wnid in seen_wnids:
            feat_index = allwnids.index(wnid) + 1
            feat_path = os.path.join(Seen_Feat, str(feat_index) + '.mat')
            seen_feat = np.array(scio.loadmat(feat_path)['features'])
            feats = np.mean(seen_feat, axis=0)  # (2048)
            feat_set.append(feats)
        self.fc_vectors = np.vstack(tuple(feat_set))


        self.all_nodes = allwnids
        self.seen_classes = seen_wnids
        self.unseen_classes = unseen_wnids



    def build_graph(self, args):
        KG_file_path = os.path.join(args.DATA_DIR, args.DATASET, 'KG_file', 'KG_triples_hie_att.txt')
        ent_set, rel_set = OrderedSet(), OrderedSet()
        triples = list()

        for line in open(KG_file_path):
            sub, rel, obj = line.strip().split('\t')
            # remove namespace prefix
            sub, obj = sub.split(':')[1], obj.split(':')[1]
            ent_set.add(sub)
            ent_set.add(obj)
            rel_set.add(rel)

            triples.append((sub, rel, obj))


        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.num_ent = len(self.ent2id)
        self.num_rel = len(self.rel2id) // 2
        print('num_ent {} num_rel {}'.format(self.num_ent, self.num_rel))


        triples = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for (h, r, t) in triples]


        edge_index, edge_type = [], []

        for sub, rel, obj in triples:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in triples:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.num_rel)

        self.edge_index = torch.LongTensor(edge_index).cuda().t()
        self.edge_type = torch.LongTensor(edge_type).cuda()










