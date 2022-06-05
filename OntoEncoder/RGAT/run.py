from helper import *
from data_loader import *
from model import *


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True



class Runner(object):

    def load_data(self):


        ent_set, rel_set = OrderedSet(), OrderedSet()
        for line in open(kg_file):
            sub, rel, obj = map(str.lower, line.strip().split('\t'))
            ent_set.add(sub)
            rel_set.add(rel)
            ent_set.add(obj)

        if self.p.dataset == 'Wiki':
            Wiki_relation_file = '../../ZS_KGC/data/Wiki/relation2ids_1'

            dataset_rels = list(json.load(open(Wiki_relation_file)).keys())
            for rel in dataset_rels:
                rel = 'wikidata:'+rel.lower()
                if rel not in ent_set:
                    ent_set.add(rel)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        # save ent2id
        json.dump(self.ent2id, open('{}/{}.txt'.format(self.p.data_path, 'ent2id'), 'w'))
        # save rel2id
        json.dump(self.rel2id, open('{}/{}.txt'.format(self.p.data_path, 'rel2id'), 'w'))


        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        print('num_ent {} num_rel {}'.format(self.p.num_ent, self.p.num_rel))


        self.data = ddict(list)
        sr2o = ddict(set)

        for line in open(kg_file):
            sub, rel, obj = map(str.lower, line.strip().split('\t'))
            sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]

            self.data['train'].append((sub, rel, obj))
            self.data['valid'].append((sub, rel, obj))

            sr2o[(sub, rel)].add(obj)
            sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.data = dict(self.data)
        # self.sr2o: train origin edges and reverse edges
        self.sr2o = {k: list(v) for k, v in sr2o.items()}

        self.triples = ddict(list)

        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)]})
        print('train set num is {}\n'.format(len(self.triples['train'])))


        for sub, rel, obj in self.data['valid']:
            rel_inv = rel + self.p.num_rel
            self.triples['{}_{}'.format('valid', 'tail')].append(
                {'triple': (sub, rel, obj), 'label': self.sr2o[(sub, rel)]})
            self.triples['{}_{}'.format('valid', 'head')].append(
                {'triple': (obj, rel_inv, sub), 'label': self.sr2o[(obj, rel_inv)]})
        print('{}_{} num is {}'.format('valid', 'tail', len(self.triples['{}_{}'.format('valid', 'tail')])))
        print('{}_{} num is {}'.format('valid', 'head', len(self.triples['{}_{}'.format('valid', 'head')])))

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.test_batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.test_batch_size),
        }
        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):

        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)
        # edge_index: 2 * 2E, edge_type: 2E * 1
        edge_index = torch.LongTensor(edge_index).cuda().t()
        edge_type = torch.LongTensor(edge_type).cuda()

        return edge_index, edge_type

    def __init__(self, params):

        self.p = params

        pprint(vars(self.p))



        self.load_data()
        # add model

        self.model = RGAT(self.edge_index, self.edge_type, params=self.p)

        self.model.cuda()

        self.optimizer, self.optimizer_mi = self.add_optimizer(self.model)


    def add_optimizer(self, model):

        return torch.optim.Adam(model.parameters(), lr=self.p.lr, weight_decay=self.p.l2), None

    def read_batch(self, batch, split):

        if split == 'train':
            triple, label = [_.cuda() for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [_.cuda() for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def get_model_params(self):

        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        return state



    def evaluate(self, epoch):
        """
        Function to evaluate the model on validation or test set
        """
        left_results = self.predict(mode='tail_batch')
        right_results = self.predict(mode='head_batch')
        results = get_combined_results(left_results, right_results)
        res_mrr = '\n\tMRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_mrr'],
                                                                              results['right_mrr'],
                                                                              results['mrr'])
        res_mr = '\tMR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_mr'],
                                                                          results['right_mr'],
                                                                          results['mr'])
        res_hit1 = '\tHit-1: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_hits@1'],
                                                                               results['right_hits@1'],
                                                                               results['hits@1'])
        res_hit3 = '\tHit-3: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_hits@3'],
                                                                               results['right_hits@3'],
                                                                               results['hits@3'])
        res_hit10 = '\tHit-10: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(results['left_hits@10'],
                                                                               results['right_hits@10'],
                                                                               results['hits@10'])
        log_res = res_mrr + res_mr + res_hit1 + res_hit3 + res_hit10
        if (epoch + 1) % 10 == 0:
            print('[Evaluating Epoch {}]: {}'.format(epoch, log_res))
        else:
            print('[Evaluating Epoch {}]: {}'.format(epoch, res_mrr))

        return results

    def predict(self, mode='tail_batch'):

        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format('valid', mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, 'valid')
                pred, _ = self.model.forward(sub, rel, 'valid')
                b_range = torch.arange(pred.size()[0])
                target_pred = pred[b_range, obj]
                # filter setting
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]

                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)


        return results

    def run_epoch(self, epoch):
        """
        Function to run one epoch of training
        """
        self.model.train()
        losses = []

        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()

            sub, rel, obj, label = self.read_batch(batch, 'train')

            pred, corr = self.model.forward(sub, rel, 'train')

            loss = self.model.loss(pred, label)


            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        loss = np.mean(losses)

        return loss, 0., 0.

    def fit(self):
        """
        Function to run training and evaluation of model
        """
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        self.best_model_state = {}
        self.best_ent_embeds = None
        self.best_rel_embeds = None

        val_results = {}
        val_results['mrr'] = 0
        kill_cnt = 0
        for epoch in range(self.p.max_epochs):
            train_loss, corr_loss, lld_loss = self.run_epoch(epoch)
            if epoch >= self.p.evaluate_epoch:
                val_results = self.evaluate(epoch)

                if val_results['mrr'] > self.best_val_mrr:
                    self.best_val = val_results
                    self.best_val_mrr = val_results['mrr']
                    self.best_epoch = epoch
                    self.best_model_state = self.get_model_params()
                    self.best_ent_embeds = self.model.all_ent_embeds.detach().cpu().numpy()
                    self.best_rel_embeds = self.model.init_rel.detach().cpu().numpy()
                    kill_cnt = 0
                else:
                    kill_cnt += 1
                    if kill_cnt % 10 == 0 and self.p.gamma > self.p.max_gamma:
                        self.p.gamma -= 5
                        print('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                    if kill_cnt > self.p.early_stop:
                        print("Early Stopping!!")
                        break
                if epoch >= self.p.save_epoch and epoch % 200 == 0:
                    # save best model at this time
                    # name = time.strftime('%H:%M:%S') + '_' + str(epoch) + '_' + str(self.best_epoch)
                    name = '{}_{}'.format(str(epoch), str(self.best_epoch))

                    save_model_name = os.path.join(save_path, name + '_model')
                    save_ent_embed_name = os.path.join(save_path, name + '_ent_embeddings')
                    save_rel_embed_name = os.path.join(save_path, name + '_rel_embeddings')

                    torch.save(self.best_model_state, save_model_name)
                    np.save(save_ent_embed_name, self.best_ent_embeds)
                    np.save(save_rel_embed_name, self.best_rel_embeds)


                if self.p.mi_train:
                    if self.p.mi_method == 'club_s' or self.p.mi_method == 'club_b':
                        print(
                            '[Epoch {}]: Training Loss: {:.5}, corr Loss: {:.5}, lld loss :{:.5}, Best valid MRR: {:.5}\n\n'.format(
                                epoch, train_loss, corr_loss,
                                lld_loss, self.best_val_mrr))
                    else:
                        print(
                            '[Epoch {}]: Training Loss: {:.5}, corr Loss: {:.5}, Best valid MRR: {:.5}\n\n'.format(
                                epoch, train_loss, corr_loss,
                                self.best_val_mrr))
                else:
                    print(
                        '[Epoch {}]: Training Loss: {:.5}, Best valid MRR: {:.5}\n\n'.format(epoch, train_loss,
                                                                                             self.best_val_mrr))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir', default='../data', help='Set name for saving/restoring models')
    parser.add_argument('--dataset', default='NELL', help='Dataset to use, [ImNet-A, ImNet-O, AwA, NELL, Wiki]')
    parser.add_argument('--data_path', default='', help='')
    parser.add_argument('--save_name', default='', help='Set name for saving/restoring models')


    parser.add_argument('--score_func', default='transe', help='Score Function for Link prediction')
    parser.add_argument('--batch_size', default=2048, type=int, help='Batch size')
    parser.add_argument('--test_batch_size', default=2048, type=int, help='Batch size of validation and test data')
    parser.add_argument('--gamma', type=float, default=12.0, help='Margin')

    # parameters for DisenE
    parser.add_argument('--init_dim', default=200, type=int, help='Initial dimension size for entities and relations')
    parser.add_argument('--gcn_dim',  default=200, type=int, help='Number of hidden units in GCN')
    parser.add_argument('--embed_dim', default=200, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('--num_factors', default=1, type=int, help="Number of factors")
    # opn is new hyperparameter
    parser.add_argument('--opn', default='mult', help='Composition Operation to be used in RAGAT')

    # parameters for training
    parser.add_argument('--max_epochs', type=int, default=80000, help='Number of epochs')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('--bias', action='store_true', help='Whether to use bias in the model')
    parser.add_argument('--no_act', dest='no_act', action='store_true', help='whether to use non_linear function')
    parser.add_argument('--no_enc', dest='no_enc', action='store_true', help='whether to use non_linear function')
    parser.add_argument('--max_gamma', type=float, default=5.0, help='Margin')
    parser.add_argument('--fix_gamma', dest='fix_gamma', action='store_true', help='whether to use non_linear function')
    parser.add_argument('--init_gamma', type=float, default=12.0, help='Margin')
    parser.add_argument('--gamma_method', dest='gamma_method', default='norm', help='')
    parser.add_argument('--num_bases', default=-1, type=int, help='Number of basis relation vectors to use')

    parser.add_argument('--early_stop', type=int, default=500, help="number of early_stop")
    parser.add_argument('--evaluate_epoch', default=1500, type=int, help="epoch to start evaluation")
    parser.add_argument('--save_epoch', default=1500, type=int, help="epoch to start save model")

    parser.add_argument('--seed', dest='seed', default=41504, type=int, help='Seed for randomization')
    parser.add_argument('--device', type=int, default=0, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')



    parser.add_argument('--gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
    parser.add_argument('--dropout', default=0.3, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('--hid_drop', default=0.1, type=float, help='Dropout after GCN')

    parser.add_argument('--head_num', default=1, type=int, help="Number of attention heads")
    parser.add_argument('--alpha', default=1e-1, type=float, help='Dropout for Feature')
    parser.add_argument('--mi_train', action='store_true', default=False, help='whether to disentangle')


    parser.add_argument('--mi_method', default='club_s', help='Composition Operation to be used in RAGAT')
    parser.add_argument('--att_mode', default='dot_weight', help='Composition Operation to be used in RAGAT')
    parser.add_argument('--mi_epoch', default=1, type=int, help="Number of MI_Disc training times")
    parser.add_argument('--score_method', default='dot_rel', help='Composition Operation to be used in RAGAT')
    parser.add_argument('--score_order', default='after', help='Composition Operation to be used in RAGAT')
    parser.add_argument('--mi_drop', action='store_true', default=True, help='whether to use non_linear function')

    args = parser.parse_args()



    args.data_path = os.path.join(args.data_dir, args.dataset)
    kg_file = os.path.join(args.data_path, 'triples.txt')

    save_path = os.path.join(args.data_path, args.save_name)
    ensure_path(save_path)

    print('random seed:', args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device)
        print('using gpu {}'.format(args.device))

    model = Runner(args)
    model.fit()
