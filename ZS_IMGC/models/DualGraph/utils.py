import torch
import os
import numpy as np
import scipy.sparse as sp
import json
import scipy.io as scio

from ordered_set import OrderedSet


def l2_loss(a, b):
    return ((a - b) ** 2).sum() / (len(a) * 2)

class DATA_LOADER:
    def __init__(self, args):
        # load KG graph
        KG_file_path = os.path.join(args.DATA_DIR, args.DATASET, 'KG_file', 'KG_triples_hie_att.txt')

        hie_ent_set, att_ent_set = OrderedSet(), OrderedSet()
        hie_triples, att_triples = [], []

        for line in open(KG_file_path):
            sub, rel, obj = line.strip().split('\t')
            # remove namespace prefix
            sub, rel, obj = sub.split(':')[1], rel.split(':')[1], obj.split(':')[1]
            if rel == 'subClassOf':
                hie_ent_set.add(sub)
                hie_ent_set.add(obj)
                hie_triples.append((sub, rel, obj))
            if rel == 'hasAttribute':
                att_ent_set.add(sub)
                att_ent_set.add(obj)
                att_triples.append((sub, rel, obj))

        self.hie_nodes_dict, self.hie_adj, self.hie_inputs = self.build_graph(args, hie_triples, list(hie_ent_set))
        self.att_nodes_dict, self.att_adj, self.att_inputs = self.build_graph(args, att_triples, list(att_ent_set))


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

        hie_train_idx, att_train_idx = [], []
        feat_set = list()

        for i, name in enumerate(seen_names):
            hie_train_idx.append(self.hie_nodes_dict[seen_wnids[i]])
            att_train_idx.append(self.att_nodes_dict[seen_wnids[i]])

            sample_label = class_names.index(name)
            feat_index = np.where(labels == sample_label)
            cls_feats = features[feat_index]
            cls_feats = np.mean(cls_feats, axis=0)  # (2048)
            feat_set.append(cls_feats)

        self.fc_vectors = np.vstack(tuple(feat_set))

        print("Loading Test Set ...")
        hie_test_idx = [self.hie_nodes_dict[unseen_wnids[i]] for i, name in enumerate(unseen_names)]
        att_test_idx = [self.att_nodes_dict[unseen_wnids[i]] for i, name in enumerate(unseen_names)]


        self.hie_train_idx, self.att_train_idx = hie_train_idx, att_train_idx
        self.hie_test_idx, self.att_test_idx = hie_test_idx, att_test_idx
        self.all_nodes = class_names
        self.seen_classes = seen_wnids
        self.unseen_classes = unseen_wnids
        self.seen_names = seen_names
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

        hie_train_idx, att_train_idx = [], []
        feat_set = list()
        for wnid in seen_wnids:
            hie_train_idx.append(self.hie_nodes_dict[wnid])
            att_train_idx.append(self.att_nodes_dict[wnid])

            feat_index = allwnids.index(wnid) + 1
            feat_path = os.path.join(Seen_Feat, str(feat_index) + '.mat')
            seen_feat = np.array(scio.loadmat(feat_path)['features'])
            feats = np.mean(seen_feat, axis=0)  # (2048)
            feat_set.append(feats)
        self.fc_vectors = np.vstack(tuple(feat_set))

        print("Loading Test Set ...")
        hie_test_idx = [self.hie_nodes_dict[wnid] for wnid in unseen_wnids]
        att_test_idx = [self.att_nodes_dict[wnid] for wnid in unseen_wnids]

        self.hie_train_idx, self.att_train_idx = hie_train_idx, att_train_idx
        self.hie_test_idx, self.att_test_idx = hie_test_idx, att_test_idx
        self.all_nodes = allwnids
        self.seen_classes = seen_wnids
        self.unseen_classes = unseen_wnids

    def build_graph(self, args, triples, nodes):

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

