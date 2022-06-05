import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import json
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from gcn import GCN
from util import build_graph
from extractor import Extractor
import warnings
warnings.filterwarnings("ignore")




class Runner:
    def __init__(self, args):
        self.args = args
        # training and testing data
        self.train_tasks = json.load(open(os.path.join(args.DATA_DIR, args.DATASET, 'datasplit', 'train_tasks.json')))
        self.test_tasks = json.load(open(os.path.join(args.DATA_DIR, args.DATASET, 'datasplit', "test_tasks.json")))
        self.val_tasks = json.load(open(os.path.join(args.DATA_DIR, args.DATASET, 'datasplit', "dev_tasks.json")))

        self.seen_rels = sorted(self.train_tasks.keys())
        self.val_rels = sorted(self.val_tasks.keys())
        self.unseen_rels = sorted(self.test_tasks.keys())
        self.all_rels = self.seen_rels + self.val_rels + self.unseen_rels

        # data for feature encoder: build neighbor graph
        # print('LOADING ENTITY ...')
        self.ent2id = json.load(open(os.path.join(args.DATA_DIR, args.DATASET, 'entity2id')))
        num_ents = len(self.ent2id.keys())

        # print('LOADING SYMBOL ID AND SYMBOL EMBEDDING ...')
        self.symbol2id, self.symbol2vec = self.load_embed(args)
        num_symbols = len(self.symbol2id.keys()) - 1
        pad_id = num_symbols

        # print('### BUILDING CONNECTION MATRIX')
        degrees, self.connections, self.e1_degrees = self.build_connection(args, num_ents, pad_id, self.symbol2id,
                                                                           self.ent2id,
                                                                           max_=args.max_neighbor)

        # Pretraining step to obtain reasonable real data embeddings, load already pre-trained
        print('Load Pretrained Feature Encoder!')
        feature_encoder = Extractor(args.embed_dim, num_symbols, embed=self.symbol2vec).cuda()
        feature_encoder.apply(self.weights_init)
        MODEL_PATH = os.path.join(args.DATA_DIR, args.DATASET, 'expri_data', 'models_train',
                                  args.embed_model + '_Extractor')
        feature_encoder.load_state_dict(torch.load(MODEL_PATH))
        self.feature_encoder = feature_encoder
        self.feature_encoder.eval()


        # prepare graph
        self.nodes_dict, self.adj, self.inputs = build_graph(args)

        print("Loading Training Data ... ")
        fc_vectors = self.prepare_train_data()
        self.train_idx = [self.nodes_dict[query] for query in self.seen_rels]
        self.test_idx = [self.nodes_dict[query] for query in self.unseen_rels]

        self.labels = torch.from_numpy(fc_vectors).cuda()
        self.labels = F.normalize(self.labels)


        self.inputs = torch.tensor(self.inputs).float().cuda()
        self.inputs = F.normalize(self.inputs)

        self.adj = self.adj.cuda()





        self.hidden_layers = args.hidden_layers
        # construct gcn model
        self.model = GCN(args.input_dim, self.labels.shape[1], self.hidden_layers).cuda()
        # Loss and optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2
        )


    def prepare_train_data(self):

        feat_list = list()


        for query in self.seen_rels:

            query_triples = self.train_tasks[query]

            query_pairs = [[self.symbol2id[triple[0]], self.symbol2id[triple[2]]] for triple in query_triples]

            query_left = [self.ent2id[triple[0]] for triple in query_triples]
            query_right = [self.ent2id[triple[2]] for triple in query_triples]

            query_meta = self.get_meta(query_left, query_right, self.connections, self.e1_degrees)
            query_pairs = Variable(torch.LongTensor(query_pairs)).cuda()

            entity_pair_vector, _ = self.feature_encoder(query_pairs, query_pairs, query_meta, query_meta)

            entity_pair_vector.detach()
            entity_pair_vector = entity_pair_vector.data.cpu().numpy()

            sample_vector = np.mean(entity_pair_vector, axis=0)

            feat_list.append(sample_vector)

        fc_vectors = np.vstack(tuple(feat_list))

        return fc_vectors

    def get_meta(self, left, right, connections, e1_degrees):
        # if len(left) == 0:
        #     print("left:", left)
        left_connections = Variable(
            torch.LongTensor(np.stack([connections[_, :, :] for _ in left], axis=0))).cuda()
        left_degrees = Variable(torch.FloatTensor([e1_degrees[_] for _ in left])).cuda()

        # print("right:", right)

        right_connections = Variable(
            torch.LongTensor(np.stack([connections[_, :, :] for _ in right], axis=0))).cuda()
        right_degrees = Variable(torch.FloatTensor([e1_degrees[_] for _ in right])).cuda()

        return (left_connections, left_degrees, right_connections, right_degrees)

    def load_embed(self, args):
        trained_embed_path = 'expri_data/Embed_used'
        symbol_id = json.load(open(
            os.path.join(args.DATA_DIR, args.DATASET, trained_embed_path, args.embed_model + '2id')))
        embeddings = np.load(os.path.join(args.DATA_DIR, args.DATASET, trained_embed_path,
                                          args.embed_model + '.npz'))['arr_0']
        symbol2id = symbol_id
        symbol2vec = embeddings

        return symbol2id, symbol2vec

    #  build neighbor connection
    def build_connection(self, args, num_ents, pad_id, symbol2id, ent2id, max_=100):

        connections = (np.ones((num_ents, max_, 2)) * pad_id).astype(int)
        e1_rele2 = defaultdict(list)
        e1_degrees = defaultdict(int)
        # rel_list = list()
        with open(os.path.join(args.DATA_DIR, args.DATASET, 'path_graph')) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split()
                e1_rele2[e1].append((symbol2id[rel], symbol2id[e2]))
                e1_rele2[e2].append((symbol2id[rel], symbol2id[e1]))
        # print("path graph relations:", len(set(rel_list)))
        degrees = {}
        for ent, id_ in ent2id.items():
            neighbors = e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            # degrees.append(len(neighbors))
            degrees[ent] = len(neighbors)
            e1_degrees[id_] = len(neighbors)  # add one for self conn
            for idx, _ in enumerate(neighbors):
                connections[id_, idx, 0] = _[0]
                connections[id_, idx, 1] = _[1]
        return degrees, connections, e1_degrees

    def weights_init(self, m):
        classname = m.__class__.__name__
        if 'Linear' in classname:
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias, 0.0)

    def train(self, epoch):

        # Start training
        self.model.train()
        emb_train = self.model(self.inputs, self.adj)
        loss = self.l2_loss(emb_train[self.train_idx], self.labels)
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if epoch % args.evaluate_epoch == 0:
            with torch.no_grad():
                self.model.eval()

                emb_test = self.model(self.inputs, self.adj)
                loss_test = self.l2_loss(
                    emb_test[self.train_idx], self.labels
                )
                print('epoch {}, train_loss={:.4f}'.format(epoch, loss_test.item()))
                if epoch % 50 == 0 and epoch >= args.test_epoch:
                    output_vectors = emb_test[self.test_idx]

                    self.test(output_vectors)


        return True

    def test(self, pred_vectors):
        test_candidates = json.load(open(os.path.join(args.DATA_DIR, args.DATASET, "test_candidates.json")))

        hits10 = []
        hits5 = []
        hits1 = []
        mrr = []

        for i, query_ in enumerate(self.unseen_rels):

            hits10_ = []
            hits5_ = []
            hits1_ = []
            mrr_ = []

            rel_classifier = pred_vectors[i]
            rel_classifier.detach()
            rel_classifier = rel_classifier.data.cpu().numpy()

            rel_classifier = np.expand_dims(rel_classifier, axis=0)

            for e1_rel, tail_candidates in test_candidates[query_].items():
                if args.DATASET == "NELL":
                    head, rela, _ = e1_rel.split('\t')
                elif args.DATASET == "Wiki":
                    head, rela = e1_rel.split('\t')

                true = tail_candidates[0]
                query_pairs = []
                if head not in self.symbol2id or true not in self.symbol2id:
                    continue
                query_pairs.append([self.symbol2id[head], self.symbol2id[true]])

                query_left = []
                query_right = []
                query_left.append(self.ent2id[head])
                query_right.append(self.ent2id[true])

                for tail in tail_candidates[1:]:
                    if tail not in self.symbol2id:
                        continue
                    query_pairs.append([self.symbol2id[head], self.symbol2id[tail]])

                    query_left.append(self.ent2id[head])
                    query_right.append(self.ent2id[tail])

                query_pairs = Variable(torch.LongTensor(query_pairs)).cuda()

                query_meta = self.get_meta(query_left, query_right, self.connections, self.e1_degrees)
                candidate_vecs, _ = self.feature_encoder(query_pairs, query_pairs, query_meta, query_meta)
                candidate_vecs.detach()
                candidate_vecs = candidate_vecs.data.cpu().numpy()

                # dot product
                # scores = candidate_vecs.dot(relation_vecs.transpose())

                # cosine similarity

                scores = cosine_similarity(candidate_vecs, rel_classifier)
                scores = np.squeeze(scores, axis=1)

                # scores = scores.mean(axis=1)

                assert scores.shape == (len(query_pairs),)

                sort = list(np.argsort(scores))[::-1]  # ascending -> descending
                rank = sort.index(0) + 1
                if rank <= 10:
                    hits10.append(1.0)
                    hits10_.append(1.0)
                else:
                    hits10.append(0.0)
                    hits10_.append(0.0)
                if rank <= 5:
                    hits5.append(1.0)
                    hits5_.append(1.0)
                else:
                    hits5.append(0.0)
                    hits5_.append(0.0)
                if rank <= 1:
                    hits1.append(1.0)
                    hits1_.append(1.0)
                else:
                    hits1.append(0.0)
                    hits1_.append(0.0)
                mrr.append(1.0 / rank)
                mrr_.append(1.0 / rank)

        print('###################################')
        print('HITS10: {:.3f}, HITS5: {:.3f}, HITS1: {:.3f}, MAP: {:.3f}'.format(np.mean(hits10),
                                                                                 np.mean(hits5),
                                                                                 np.mean(hits1),
                                                                                 np.mean(mrr)))
        print('###################################')

    def l2_loss(self, a, b):
        return ((a - b) ** 2).sum() / (len(a) * 2)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parameters for data
    parser.add_argument('--DATA_DIR', default='../../data', help='directory for ZSL data')
    parser.add_argument('--DATASET', default='NELL', help='NELL, Wiki')


    # parameters for GCN
    parser.add_argument('--input_dim', type=int, default=200, help='the dimension of node features')
    parser.add_argument("--hidden_layers", type=str, default='d200,d', help="Number of hidden units.")

    # parameters for training
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--l2", type=float, default=5e-4, help="Weight decay (L2 loss on parameters).")
    parser.add_argument("--drop", type=float, default=0.5, help="Dropout of RGCN")
    parser.add_argument('--seed', default=9999, type=int, help='Seed for randomization')
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train.")
    parser.add_argument('--device', type=int, default=0, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')

    # parameters for feature encoder of ZS-KGC, different for NELL and Wiki
    parser.add_argument("--embed_model", default='TransE', type=str)
    parser.add_argument("--max_neighbor", default=50, type=int, help='neighbor number of each entity')
    parser.add_argument("--embed_dim", default=100, type=int, help='dimension of triple embedding')
    parser.add_argument("--ep_dim", default=200, type=int, help='dimension of entity pair embedding')


    parser.add_argument('--evaluate_epoch', type=int, default=10)
    parser.add_argument('--test_epoch', type=int, default=600)


    args = parser.parse_args()

    print('random seed:', args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device)
        print('using gpu {}'.format(args.device))

    run = Runner(args)
    for epoch in range(1, args.epochs+1):
        if run.train(epoch) is False:
            break

