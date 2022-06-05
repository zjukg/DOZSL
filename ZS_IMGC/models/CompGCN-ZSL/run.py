import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import warnings
import scipy.io as scio
from utils import *
from models import CompGCNBase

warnings.filterwarnings("ignore")




class Runner:
    def __init__(self, args):
        self.args = args

        self.data = DATA_LOADER(args)



        self.y =  self.data.fc_vectors
        self.seen_classes, self.unseen_classes, self.all_nodes = self.data.seen_classes, self.data.unseen_classes, self.data.all_nodes


        self.ent2id = self.data.ent2id
        self.args.num_ent, self.args.num_rel = self.data.num_ent, self.data.num_rel

        self.train_idx = [self.ent2id[cls] for cls in self.seen_classes]
        self.test_idx = [self.ent2id[cls] for cls in self.unseen_classes]

        self.edge_index, self.edge_type = self.data.edge_index, self.data.edge_type

        self.labels = torch.from_numpy(self.y).cuda()
        self.labels = F.normalize(self.labels)



        # Create Model
        self.model = CompGCNBase(self.edge_index, self.edge_type, params=self.args).cuda()
        # Loss and optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.l2)

        self.best_acc = 0.
        self.best_epoch = 0
        self.best_H = 0.
        self.best_epoch2 = 0



    def train(self, epoch):

        # Start training
        self.model.train()
        emb_train = self.model.forward()
        loss = self.l2_loss(emb_train[self.train_idx], self.labels)
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if epoch % args.evaluate_epoch == 0:
            with torch.no_grad():
                self.model.eval()

                emb_test = self.model.forward()
                loss_test = self.l2_loss(emb_test[self.train_idx], self.labels)
                print('epoch {}, train_loss={:.4f}'.format(epoch, loss_test.item()))
                if epoch % 50 == 0 and epoch >= args.test_epoch:
                    output_vectors = emb_test[self.train_idx+self.test_idx]

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

        return True

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
            print('H value: {:.2f}'.format(output[0]))

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
            feat_index = self.all_nodes.index(wnid) + 1
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
            print('H value: {:.2f}'.format(output[0]))

        if GZSL:
            return output[0]
        else:
            return zsl_output[0]



    def l2_loss(self, a, b):
        return ((a - b) ** 2).sum() / (len(a) * 2)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parameters for data
    parser.add_argument('--DATA_DIR', default='../../data', help='directory for ZSL data')
    parser.add_argument('--DATASET', default='AwA2', help='ImNet_A, ImNet_O, AwA2')
    # parser.add_argument('--DATASET', default='ImageNet/ImNet_O', help='ImNet_A, ImNet_O, AwA2')


    # parameters for CompGCN
    parser.add_argument('--gcn_layer', default=2, type=int, help='Number of GCN Layers to use')
    parser.add_argument("--input_dim", type=int, default=100, help="dimension of input")
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Number of hidden units")
    parser.add_argument("--output_dim", type=int, default=2048, help="")
    parser.add_argument('--opn', default='mult', help='Composition Operation used in CompGCN, [sub, mult, corr or none]')


    # parameters for training
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--l2", type=float, default=5e-4, help="Weight decay (L2 loss on parameters).")
    parser.add_argument("--hid_drop", type=float, default=0.5, help="Dropout of RGCN")
    parser.add_argument('--seed', default=9999, type=int, help='Seed for randomization')
    parser.add_argument("--max_epochs", type=int, default=1000, help="Number of epochs to train.")
    parser.add_argument('--device', type=int, default=0, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')

    parser.add_argument("--gzsl", action="store_true", default=False, help="test generalized zsl setting")
    parser.add_argument('--evaluate_epoch', type=int, default=10)
    parser.add_argument('--test_epoch', type=int, default=300)

    args = parser.parse_args()

    print('random seed:', args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device)
        print('using gpu {}'.format(args.device))

    run = Runner(args)
    for epoch in range(1, args.max_epochs+1):
        if run.train(epoch) is False:
            break

    print("\n BEST ACC {:.2f} at epoch {:d}".format(run.best_acc, run.best_epoch))
    print("BEST H {:.2f} at epoch {:d}".format(run.best_H, run.best_epoch2))