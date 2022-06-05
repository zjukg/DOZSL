import torch
import os
import numpy as np
import scipy.sparse as sp
import json
import scipy.io as scio



def l2_loss(a, b):
    return ((a - b) ** 2).sum() / (len(a) * 2)

class DATA_LOADER:
    def __init__(self, args):

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

