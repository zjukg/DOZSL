from __future__ import print_function
import os
import numpy as np
import scipy.sparse as sp
import json
import scipy.io as scio

from collections import defaultdict
from ordered_set import OrderedSet





def row_normalize(AA):
    A_list = list()
    for i in range(len(AA)):
        d = np.array(AA[i].sum(1)).flatten()
        d_inv = 1.0 / d
        d_inv[np.isinf(d_inv)] = 0.0
        D_inv = sp.diags(d_inv)
        A_list.append(D_inv.dot(AA[i]).tocsr())
    return A_list

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

        train_idx = list()
        feat_set = list()

        for i, name in enumerate(seen_names):
            train_idx.append(self.nodes_dict[seen_wnids[i]])
            sample_label = class_names.index(name)
            feat_index = np.where(labels == sample_label)
            cls_feats = features[feat_index]
            cls_feats = np.mean(cls_feats, axis=0)  # (2048)
            feat_set.append(cls_feats)

        self.fc_vectors = np.vstack(tuple(feat_set))

        print("Loading Test Set ...")
        test_idx = [self.nodes_dict[unseen_wnids[i]] for i, name in enumerate(unseen_names)]

        self.train_idx = train_idx
        self.test_idx = test_idx
        self.all_classes = class_names
        self.seen_classes = seen_names
        self.unseen_classes = unseen_names


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

        train_idx = list()
        feat_set = list()
        for wnid in seen_wnids:
            train_idx.append(self.nodes_dict[wnid])
            feat_index = allwnids.index(wnid) + 1
            feat_path = os.path.join(Seen_Feat, str(feat_index) + '.mat')
            seen_feat = np.array(scio.loadmat(feat_path)['features'])
            feats = np.mean(seen_feat, axis=0)  # (2048)
            feat_set.append(feats)
        self.fc_vectors = np.vstack(tuple(feat_set))

        print("Loading Test Set ...")
        test_idx = [self.nodes_dict[wnid] for wnid in unseen_wnids]

        self.train_idx = train_idx
        self.test_idx = test_idx
        self.all_classes = allwnids
        self.seen_classes = seen_wnids
        self.unseen_classes = unseen_wnids



    def build_graph(self, args):
        KG_file_path = os.path.join(args.DATA_DIR, args.DATASET, 'KG_file', 'KG_triples_hie_att.txt')
        ent_set, rel_set = OrderedSet(), OrderedSet()
        rel_freq = defaultdict(list)
        triples = list()

        if not args.single:
            for line in open(KG_file_path):
                sub, rel, obj = line.strip().split('\t')
                # remove namespace prefix
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
                sub, obj = sub.split(':')[1], obj.split(':')[1]
                ent_set.add(sub)
                ent_set.add(obj)
                rel_set.add('one')
                rel_freq['one'].append((sub, obj))
                triples.append((sub, 'one', obj))

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

        self.nodes_dict = nodes_dict
        self.adjacencies = adjacencies











