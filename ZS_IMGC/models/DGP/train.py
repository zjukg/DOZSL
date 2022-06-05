import argparse
import os
import torch
import torch.nn.functional as F
import scipy.io as scio
import numpy as np

from gcn import GCN
from utils import DATA_LOADER

'''
input: imagenet-induced-animal-graph.json, fc-weights.json
get: save prediction model file
function: train with gcn(2 layers) and predict testing features
'''





class Runner:
    def __init__(self, args):
        self.best_acc = 0.
        self.best_H = 0.
        self.best_epoch = 0
        self.best_epoch2 = 0

        self.data = DATA_LOADER(args)

        self.adj, self.y, self.inputs = self.data.adj, self.data.fc_vectors, self.data.inputs


        self.train_idx, self.test_idx = self.data.train_idx, self.data.test_idx
        self.seen_classes, self.unseen_classes, self.all_nodes = self.data.seen_classes, self.data.unseen_classes, self.data.all_nodes

        self.labels = torch.from_numpy(self.y).cuda()
        self.labels = F.normalize(self.labels)  # shape: (28, 2049)


        self.inputs = torch.tensor(self.inputs).float().cuda()
        self.inputs = F.normalize(self.inputs)

        self.adj = self.adj.cuda()




        self.hidden_layers = args.hidden_layers
        # construct gcn model
        self.model = GCN(args.input_dim, self.labels.shape[1], self.hidden_layers).cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.l2)


    def l2_loss(self, a, b):
        return ((a - b) ** 2).sum() / (len(a) * 2)

    def train(self, epoch):



        self.model.train()
        output_vectors = self.model(self.inputs, self.adj)

        # calculate the loss over training seen nodes
        loss = self.l2_loss(output_vectors[self.train_idx], self.labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # calculate loss on training (and validation) seen nodes
        if epoch % args.evaluate_epoch == 0:
            self.model.eval()

            output_vectors = self.model(self.inputs, self.adj)
            train_loss = self.l2_loss(output_vectors[self.train_idx], self.labels).item()

            print('epoch {}, train_loss={:.4f}'.format(epoch, train_loss))

        # save intermediate output_vector of each node of the graph
        if epoch % 50 == 0 and epoch >= args.test_epoch:
            output_vectors = output_vectors[self.train_idx + self.test_idx]
            if args.DATASET == 'AwA2':
                acc = self.test_awa(output_vectors, GZSL=False)
                if args.gzsl:
                    H = self.test_awa(output_vectors, GZSL=True)

            else:
                acc = self.test_imagenet(output_vectors, GZSL=False)
                if args.gzsl:
                    H = self.test_imagenet(output_vectors, GZSL=True)

            if acc > self.best_acc:
                self.best_acc = acc
                self.best_epoch = epoch

            if args.gzsl:
                if H > self.best_H:
                    self.best_H = H
                    self.best_epoch2 = epoch


    def test_awa(self, pred_vectors, GZSL=False):

        matcontent = scio.loadmat(os.path.join(args.DATA_DIR, args.DATASET, 'res101.mat'))
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        n = len(self.seen_classes)

        top = [1, 2, 5, 10, 20]
        # print("********* Testing Unseen Data **********")
        unseen_macro_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
        unseen_total_imgs = 0

        for i, name in enumerate(self.data.unseen_names, 1):
            sample_label = self.all_nodes.index(name)
            feat_index = np.where(label == sample_label)
            features = feature[feat_index]
            feat = torch.from_numpy(features).float().cuda()
            # feat = F.normalize(feat)

            all_label = n + i - 1

            hits = torch.zeros(len(top))
            tot = 0

            fcs = pred_vectors.t()  # [2048, 883]
            table = torch.matmul(feat, fcs)
            # False: filter seen classifiers
            if not GZSL:
                table[:, :n] = -1e18

            # for hit@1 and hit@2
            gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
            rks = (table >= gth_score).sum(dim=1)
            assert (table[:, all_label] == gth_score[:, all_label]).min() == 1
            for j, k in enumerate(top):
                hits[j] += (rks <= k).sum().item()
            tot += len(features)

            unseen_total_imgs += tot
            per_hits = hits / float(tot)
            unseen_macro_hits += per_hits
        print('-----------------------------------------------')
        # ### Macro Acc
        # print("************ Unseen Macro Acc ************")
        unseen_macro_hits = unseen_macro_hits * 1.0 / len(self.unseen_classes)
        zsl_output = [i * 100 for i in unseen_macro_hits]
        print('Unseen Macro Acc: {:.2f}'.format(zsl_output[0]))

        if GZSL:
            # print("********* Testing Seen Data **********")
            seen_micro_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
            seen_macro_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
            seen_total_imgs = 0
            # unseen_wnids = unseen_wnids[0]

            for i, name in enumerate(self.data.seen_names, 1):
                sample_label = self.all_nodes.index(name)
                feat_index = np.where(label == sample_label)

                features = feature[feat_index]

                feat = torch.from_numpy(features).float().cuda()

                all_label = i - 1

                hits = torch.zeros(len(top))
                tot = 0

                fcs = pred_vectors.t()  # [2048, 883]
                table = torch.matmul(feat, fcs)
                # False: filter seen classifiers
                if not GZSL:
                    table[:, :n] = -1e18

                # for hit@1 and hit@2
                gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
                rks = (table >= gth_score).sum(dim=1)
                assert (table[:, all_label] == gth_score[:, all_label]).min() == 1
                for j, k in enumerate(top):
                    hits[j] += (rks <= k).sum().item()
                tot += len(features)

                seen_micro_hits += hits
                seen_total_imgs += tot
                per_hits = hits / float(tot)
                seen_macro_hits += per_hits

            # ### Macro Acc
            seen_macro_hits = seen_macro_hits * 1.0 / len(self.seen_classes)
            output = ['{:.2f}'.format(i * 100) for i in seen_macro_hits]
            print('Seen Macro Acc: ', output[0])

            acc_H = 2 * seen_macro_hits * unseen_macro_hits / (seen_macro_hits + unseen_macro_hits)
            output = [i * 100 for i in acc_H]
            print('H value {:.2f}:'.format(output[0]))

        if GZSL:
            return output[0]
        else:
            return zsl_output[0]

    def test_imagenet(self, pred_vectors, GZSL=False):
        Seen_Test_Feat = os.path.join(args.DATA_DIR, 'ImageNet', 'Res101_Features', 'ILSVRC2012_val')
        Unseen_Test_Feat = os.path.join(args.DATA_DIR, 'ImageNet', 'Res101_Features', 'ILSVRC2011')
        n = len(self.seen_classes)
        top = [1, 2, 5, 10, 20]

        # print("********* Testing Unseen Data **********")

        unseen_macro_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
        unseen_total_imgs = 0
        for i, wnid in enumerate(self.unseen_classes, 1):
            all_label = n + i - 1
            hits = torch.zeros(len(top))
            tot = 0

            # load test features begin
            feat_index = self.data.all_nodes.index(wnid) + 1
            feat_path = os.path.join(Unseen_Test_Feat, str(feat_index) + '.mat')
            features = np.array(scio.loadmat(feat_path)['features'])

            feat = torch.from_numpy(features).float().cuda()

            fcs = pred_vectors.t()  # [2048, 883]
            table = torch.matmul(feat, fcs)
            # False: filter seen classifiers
            if not GZSL:
                table[:, :n] = -1e18

            # for hit@1 and hit@2
            gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
            rks = (table >= gth_score).sum(dim=1)
            assert (table[:, all_label] == gth_score[:, all_label]).min() == 1
            for j, k in enumerate(top):
                hits[j] += (rks <= k).sum().item()
            tot += len(features)

            unseen_total_imgs += tot
            per_hits = hits / float(tot)
            unseen_macro_hits += per_hits

        print('-----------------------------------------------')

        # ### Macro Acc
        # print("************ Unseen Macro Acc ************")
        unseen_macro_hits = unseen_macro_hits * 1.0 / len(self.unseen_classes)
        zsl_output = [i * 100 for i in unseen_macro_hits]
        print("Unseen Macro Acc {:.2f} :".format(zsl_output[0]))
        if GZSL:
            # print("********* Testing Seen Data **********")
            # total_hits, total_imgs = 0, 0
            seen_macro_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
            seen_total_imgs = 0
            for i, wnid in enumerate(self.seen_classes, 1):
                all_label = i - 1

                hits = torch.zeros(len(top))
                tot = 0

                # load test features begin
                feat_index = self.all_nodes.index(wnid) + 1
                feat_path = os.path.join(Seen_Test_Feat, str(feat_index) + '.mat')
                features = np.array(scio.loadmat(feat_path)['features'])

                feat = torch.from_numpy(features).float()

                fcs = pred_vectors.t().cpu()  # [2048, 883]
                table = torch.matmul(feat, fcs)
                # False: filter seen classifiers
                if not GZSL:
                    table[:, :n] = -1e18

                # for hit@1 and hit@2
                gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
                rks = (table >= gth_score).sum(dim=1)
                assert (table[:, all_label] == gth_score[:, all_label]).min() == 1
                for j, k in enumerate(top):
                    hits[j] += (rks <= k).sum().item()
                tot += len(features)

                seen_total_imgs += tot
                per_hits = hits / float(tot)
                seen_macro_hits += per_hits

            # print("************ Seen Macro Acc ************")
            seen_macro_hits = seen_macro_hits * 1.0 / len(self.seen_classes)
            output = ['{:.2f}'.format(i * 100) for i in seen_macro_hits]
            print("Seen Macro Acc:", output[0])

            # print("************* H value ************")
            acc_H = 2 * seen_macro_hits * unseen_macro_hits / (seen_macro_hits + unseen_macro_hits)
            output = [i * 100 for i in acc_H]
            print("H value {:.2f}".format(output[0]))
        # return zsl_output[0]

        if GZSL:
            return output[0]
        else:
            return zsl_output[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--DATA_DIR', default='../../data')
    # parser.add_argument('--DATASET', default='ImageNet/ImNet_O', help='the folder to store subset files')
    parser.add_argument('--DATASET', default='AwA2', help='the folder to store subset files')

    # parameters for GCN
    parser.add_argument('--input_dim', type=int, default=100, help='the dimension of node features')
    parser.add_argument('--hidden_layers', type=str, default='d2048,d', help='hidden layers')

    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2', type=float, default=0.0005)
    parser.add_argument('--device', type=int, default=0, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--seed', default=9999, type=int, help='Seed for randomization')
    parser.add_argument("--gzsl", action="store_true", default=False, help="test generalized zsl setting")

    parser.add_argument('--test_epoch', type=int, default=400)
    parser.add_argument('--evaluate_epoch', type=int, default=10)

    args = parser.parse_args()

    print('random seed:', args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device)
        print('using gpu {}'.format(args.device))

    run = Runner(args)
    for epoch in range(1, args.max_epoch+1):
        if run.train(epoch) is False:
            break
    print("BEST ACC {:.2f} at epoch {:d}".format(run.best_acc, run.best_epoch))
    print("BEST H {:.2f} at epoch {:d}".format(run.best_H, run.best_epoch2))



